"""Synthesizer: one privileged LLM call, and a deterministic ladder beneath it.

Contract: ``docs/architecture.md`` 3.4. Input is the question and the ranked
``Evidence`` pool; output is one ``AnswerCandidate``.

Why ``answer_sentence`` is a separate field
-------------------------------------------
It costs nothing extra in the same JSON response and it gives the Verifier a
clean NLI hypothesis, instead of the usual hack of concatenating the question
and the answer -- which produces an interrogative fragment that DeBERTa scores
as neutral against everything. The synthesiser is the only component that knows
what proposition it meant to assert, so it is the right place to state it.

The fallback ladder is not a safety net, it is a baseline
---------------------------------------------------------
Design axiom 3: if the model is unavailable, unparseable or out of budget, the
extractive ladder still produces an EM/F1-comparable span. It also sets
``citations`` to the evidence it actually read, so citation grounding stays a
meaningful signal on the degraded path rather than collapsing to zero and
dragging the verdict down for the wrong reason.

The one honest cost of the ladder is the hypothesis. A deterministic extractor
cannot phrase the proposition it found, so ``answer_sentence`` falls back to a
uniform template. That template deliberately does **not** copy the premise
sentence: copying would make NLI entailment trivially 1.0 and report the
fallback path as *more* confident than a grounded model answer, which would be
a fabricated result.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import Config
from ..eval.metrics import normalize_answer
from ..state import QuestionState
from ..types import AnswerCandidate, AnswerType, Evidence, Origin, Plan
from .base import BaseAgent

__all__ = [
    "PROMPT_ID",
    "Synthesizer",
    "declarative_sentence",
    "infer_answer_type",
    "reconcile_answer",
]

PROMPT_ID = "synth.answer.v1"

_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "answer_sentence": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "sufficient": {"type": "boolean"},
    },
    "required": ["answer", "answer_sentence", "citations", "sufficient"],
}

#: Longest evidence snippet shown to the model. Sentences average ~130 chars in
#: both corpora; the cap only bites on the handful of unsplit paragraphs, and
#: keeps 20 pieces of evidence comfortably inside ``num_ctx: 8192``.
MAX_EVIDENCE_CHARS = 400

#: Rung 4 truncation, per 3.4.
MAX_FALLBACK_TOKENS = 20

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}\s+{_MONTH}\s+\d{{4}}|{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|{_MONTH}\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

#: Comparison polarity. First match wins, so "earliest" is checked before the
#: bare "later" that its sentence may also contain.
_EARLIER_MARKERS = ("first", "earlier", "earliest", "older", "oldest", "before", "sooner")
_LATER_MARKERS = ("later", "latest", "newer", "newest", "younger", "youngest",
                  "more recent", "most recent", "last", "after")
_SAMENESS_MARKERS = ("both", "same", "either", "also")

_YESNO_PREFIXES = (
    "are ", "is ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
    "has ", "have ", "had ", "will ", "would ", "should ",
)


# ---------------------------------------------------------------------------
# Question shape
# ---------------------------------------------------------------------------

def infer_answer_type(
    question: str, plan: Plan | None = None, *, yesno_guard: bool = False
) -> AnswerType:
    """The expected answer type: the plan's own label first, then the question.

    The Planner assigns ``answer_type`` per sub-query and the combiner (or last)
    node carries the type of the *final* answer, so that is the better source
    when it is informative. ``"string"`` is the schema default and therefore
    carries no information -- fall through to the surface form of the question
    instead of trusting it.

    With ``yesno_guard`` the plan may refine the type but never flip it into or
    out of ``"yesno"`` (accuracy plan 1B). A plan node's ``answer_type``
    describes ITS OWN sub-answer, and when the plan has no combiner the last
    node of a bridge chain is often a yes/no verification hop ("Is
    {{q2.answer}} a retired American soccer player?") whose type is not the
    final answer's. ``"yesno"`` is the one label the Synthesizer cannot recover
    from unaided: it makes the model emit the literal string "yes" for a
    question whose answer is an entity. The question's own surface form is
    authoritative on that one distinction. Measured over the 250 questions of
    ``agentic_full_hotpotqa_20260904T221533Z``, the plan flips the yes/no bit on
    9 questions, which score EM 0.333 against 0.444 for the 162 where plan and
    surface already agree.

    ``yesno_guard`` defaults to **off**, which is what the evaluated grid ran,
    and is switched on by ``agents.synthesizer.answer_type_guard``. It is
    gated rather than unconditional -- against the letter of the accuracy plan,
    which calls it a free correctness fix -- because it is not free: this
    function decides the ``answer_type`` rendered into the synthesis prompt, so
    a question whose type changes is a question where the model is asked
    something else and ``agentic_full`` stops being the run the report
    describes. Replaying both 250-question eval traces through the two
    branches, the guard changes ``answer_type`` on **12 of 250** questions of
    ``agentic_full_hotpotqa_20260904T221533Z``.

    Gating costs the one-node configurations nothing, which is the reason it is
    the right trade and not merely the conservative one. In
    ``agentic_no_planner_hotpotqa_20260905T224906Z`` all 250 plans are a single
    node of type ``"string"``, so the refinement branch is never taken and the
    guard is unreachable: the same replay finds **0 of 250** questions changed.
    ``agentic_v2`` is a one-node system and therefore sets the flag for its
    meaning, not for an effect it can have.
    """
    surface = _surface_answer_type(question)
    if plan is not None and plan.subqueries:
        combiner = next((sq for sq in plan.subqueries if sq.is_combiner), None)
        node = combiner or plan.subqueries[-1]
        flips_yesno = (node.answer_type == "yesno") != (surface == "yesno")
        if node.answer_type != "string" and not (yesno_guard and flips_yesno):
            return node.answer_type
    return surface


def _surface_answer_type(question: str) -> AnswerType:
    """The answer type read off the question's surface form alone."""
    q = question.strip().lower()
    if q.startswith(_YESNO_PREFIXES):
        return "yesno"
    if q.startswith(("when ", "what year", "what date", "in what year", "in which year")):
        return "date"
    if q.startswith(("how many", "how much", "how long", "how old", "how tall")):
        return "number"
    if q.startswith(("who ", "whom ", "whose ", "which ", "what ", "where ")):
        return "entity"
    return "string"


def declarative_sentence(question: str, answer: str) -> str:
    """A hypothesis for NLI when the model did not supply one.

    Deliberately a *statement about the answer* rather than a copy of the
    premise: an extractive fallback that echoed its own evidence would score
    entailment 1.0 against itself and report more confidence than a grounded
    answer deserves.
    """
    if not answer:
        return ""
    q = " ".join(question.split()).rstrip("?").strip()
    return f'The answer to the question "{q}" is {answer}.'


# ---------------------------------------------------------------------------
# Answer/sentence reconciliation (accuracy plan 1A)
# ---------------------------------------------------------------------------
#
# The model reliably writes a correct ``answer_sentence`` and then fills
# ``answer`` with the wrong constituent of it. Measured on
# ``agentic_full_hotpotqa_20260904T221533Z``: 24 of the 250 questions carry the
# gold answer *inside the model's own sentence* while ``answer`` holds a
# different span -- 9.6 points of exact match sitting in a slot error rather
# than a reasoning error.
#
# That 9.6 is the ceiling, not the yield. Most of those 24 are surface-form
# mismatches ("Washington State Cougars" for "Washington State", "Los Alamos
# Laboratory" for "Los Alamos") that no deterministic rule can shorten without
# also shortening answers that are already right. What is left after insisting
# on zero breakages is 6 of the 24, worth +0.024 EM on the eval slice and, at
# roughly one firing per fifty questions, not resolvable at all on calib.
#
# This is repaired deterministically, after the JSON is parsed and before the
# candidate is built. No prompt change (the prompt rewrite for "verbatim spans"
# was measured at -1: it lengthens answers), and no extra model call.
#
# The design rule is *decline rather than guess*. A trigger only establishes
# that the emitted answer contradicts **the question** -- never merely that it
# is missing from the sentence, which is a different and much weaker claim (see
# ``reconcile_answer``, where treating absence as contradiction is what made the
# first version of this a measured regression). A separate selector must then
# produce a replacement it can justify, and when no selector fires the original
# answer is returned untouched.
#
# Triggers are kept on the breakage count, not the recovery count, and are
# replayed over every trace set available rather than one: the shape that broke
# the first version does not occur anywhere in the 250-question eval slice.

#: Interrogative phrases, searched anywhere in the question rather than only at
#: its start: "The publication of which magazine ended first, ...?" and "Andrea
#: Camplone is a football coach in which 22-team competition?" both ask for an
#: entity, and both are invisible to a prefix test.
_SHAPE_PHRASES: tuple[tuple[str, AnswerType], ...] = (
    ("how many", "number"), ("how much", "number"), ("how long", "number"),
    ("how old", "number"), ("how tall", "number"), ("what age", "number"),
    ("what year", "date"), ("which year", "date"), ("what date", "date"),
    ("which date", "date"), ("when ", "date"),
    ("who ", "entity"), ("whom ", "entity"), ("whose ", "entity"),
    ("which ", "entity"), ("what ", "entity"), ("where ", "entity"),
)

#: A normalised answer made only of digits and separators -- a bare year, count
#: or measurement. Emitted for a "who/which/what" question it is a type error,
#: never a near-miss.
_BARE_VALUE_RE = re.compile(r"^[\d\s.,]+$")

#: Words that cannot begin or appear inside one arm of a genuine "A or B"
#: disjunction. Without this, "Who is younger Jenny Bae or Lionel Richie?"
#: yields the arm "Who is younger Jenny Bae", which fails to match the sentence
#: and hands the win to the wrong person.
_NOT_AN_ALTERNATIVE = frozenset({
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "should", "both", "either", "neither",
    "and", "that", "this", "than", "the", "a", "an", "of", "in", "on", "at",
    "to", "for", "by", "with", "from", "it", "he", "she", "they",
})

#: Longest arm considered on either side of the "or".
_MAX_ALT_TOKENS = 6

#: The tail of a capitalised token: the dots and ampersands of "A.C.G.T" and
#: "AT&T" belong to the name. The leading capital is tested with ``str.isupper``
#: rather than an ASCII class, so accented names ("Edouard") start a span too.
_CAP_TAIL_RE = re.compile(r"^[\w.&'-]*$")
#: Lowercase words allowed *between* two capitalised tokens ("Castle of
#: Frankenstein"), never at either end of a span. "and" is deliberately absent:
#: it joins two names into one span ("Alan Dean Foster and John Lanchester")
#: rather than spanning one.
_SPAN_JOINERS = frozenset({"of", "the", "de", "del", "van", "von", "da", "di"})
#: A trailing possessive, stripped from a selected span: the sentence says
#: "President Garfield's funeral", the answer is the person.
_POSSESSIVE_RE = re.compile(r"['’]s$")


def _is_capitalised(word: str) -> bool:
    return bool(word) and word[0].isupper() and bool(_CAP_TAIL_RE.match(word[1:]))


def _norm_tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize_answer(text).split())


def _contains(haystack: str, needle: str) -> bool:
    """Is ``needle`` a whole-token subsequence of ``haystack``?

    Token containment rather than a raw substring test, because normalisation
    strips punctuation and a substring test then reports "alcoholic" as present
    inside "non-alcoholic" -- which is exactly the distinction one of the
    disjunctive questions turns on. The tokens are ``normalize_answer``'s, so a
    hit here means the span would also score as exact match.
    """
    hay, need = _norm_tokens(haystack), _norm_tokens(needle)
    if not need or len(need) > len(hay):
        return False
    return any(hay[i:i + len(need)] == need for i in range(len(hay) - len(need) + 1))


def question_shape(question: str) -> AnswerType:
    """The answer type the *question* demands, read off its surface form.

    Deliberately separate from :func:`infer_answer_type`. That function chooses
    the ``answer_type`` shown to the model in the prompt, so changing it changes
    what the model generates; this one only judges an answer already generated,
    which is what makes the reconciliation replayable offline against existing
    traces. Longer phrases break a tie at the same position, so "how many" is
    read before the "how" inside it.

    Which interrogative governs is positional. One at the very start governs the
    whole question; otherwise the **last** one does, because a wh-word in the
    middle of a HotpotQA question is nearly always a relative pronoun and not
    the thing being asked. "The Duke Steps Out stars an actress *who* was ranked
    tenth ... in *what year*?" asks for a year, and reading the relative "who"
    as the interrogative is what made this rule overwrite a correct 1999 with an
    actress's name in the first replay.
    """
    q = " ".join(question.split()).strip().lower()
    if q.startswith(_YESNO_PREFIXES):
        return "yesno"
    hits: list[tuple[int, int, AnswerType]] = []
    for phrase, shape in _SHAPE_PHRASES:
        pos = q.find(phrase)
        if pos >= 0:
            hits.append((pos, len(phrase), shape))
    if not hits:
        return "string"
    # Leftmost then longest; rightmost then longest. Longest either way, so that
    # "what year" is preferred to the "what " nested inside it.
    first = min(hits, key=lambda hit: (hit[0], -hit[1]))
    # <= 3 rather than == 0 so a fronted preposition still counts as the start:
    # "In what year did ...".
    if first[0] <= 3:
        return first[2]
    return max(hits, key=lambda hit: (hit[0], hit[1]))[2]


def _capitalised_spans(sentence: str) -> list[str]:
    """Capitalised spans of ``sentence``, longest first.

    A sentence-initial "The"/"A"/"An" is dropped: it is an artefact of the
    sentence starting, not part of a name ("The Serie B competition has 22
    teams" is about Serie B).
    """
    runs: list[list[str]] = []
    run: list[str] = []
    for idx, raw in enumerate(sentence.split()):
        word = raw.strip('.,;:!?"()[]')
        if not word:
            continue
        if _is_capitalised(word):
            if idx == 0 and word.lower() in ("the", "a", "an"):
                continue
            run.append(_POSSESSIVE_RE.sub("", word))
        elif run and word.lower() in _SPAN_JOINERS:
            run.append(word)          # provisional; trimmed below
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)
    out: list[str] = []
    for candidate in runs:
        while candidate and candidate[-1].lower() in _SPAN_JOINERS:
            candidate.pop()
        if candidate:
            out.append(" ".join(candidate))
    out.sort(key=lambda sp: (-len(sp.split()), -len(sp)))
    return out


def _alternative_arms(question: str) -> list[tuple[str, str]]:
    """Candidate ``(left, right)`` arms of every "A or B" in the question.

    Arms grow outward from the "or" and are constrained to look like each
    other: comparable length, matching capitalisation, no interrogative or
    auxiliary inside either. A disjunction whose arms are not parallel is not a
    disjunction of answers, and admitting one is how this rule would start
    inventing answers.
    """
    q = " ".join(question.split()).strip().rstrip("?").strip()
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"\bor\b", q, re.IGNORECASE):
        left = re.split(r"[,;:]", q[: match.start()])[-1].strip()
        right = re.split(r"[,;:]", q[match.end():])[0].strip()
        lw, rw = left.split(), right.split()
        for i in range(1, min(len(lw), _MAX_ALT_TOKENS) + 1):
            arm_l = " ".join(lw[-i:])
            for j in range(1, min(len(rw), _MAX_ALT_TOKENS) + 1):
                arm_r = " ".join(rw[:j])
                if abs(i - j) > 1:
                    continue
                if arm_l[:1].isupper() != arm_r[:1].isupper():
                    continue
                if not _norm_tokens(arm_l) or not _norm_tokens(arm_r):
                    continue
                if any(
                    token in _NOT_AN_ALTERNATIVE
                    for arm in (arm_l, arm_r)
                    for token in _norm_tokens(arm)
                ):
                    continue
                if _contains(arm_l, arm_r) or _contains(arm_r, arm_l):
                    continue
                pairs.append((arm_l, arm_r))
    pairs.sort(key=lambda pair: -(len(pair[0].split()) + len(pair[1].split())))
    return pairs


def _select_alternative(question: str, sentence: str) -> str | None:
    """The one arm of a disjunctive question that the sentence asserts.

    "Which director, John Schlesinger or Barbara Albert, was also a writer and
    film producer?" against "Barbara Albert was also a writer and film
    producer." Exactly one arm may appear: when the sentence mentions both, or
    neither, the question is not decided and this returns ``None``.
    """
    for arm_l, arm_r in _alternative_arms(question):
        in_l, in_r = _contains(sentence, arm_l), _contains(sentence, arm_r)
        if in_l != in_r:
            return arm_l if in_l else arm_r
    return None


def _select_by_shape(question: str, sentence: str, shape: AnswerType) -> str | None:
    """A span of ``sentence`` of the type the question asks for."""
    if shape == "entity":
        for span in _capitalised_spans(sentence):
            if not _contains(question, span):
                return span          # the name the question did not already give
        return None
    if shape == "date":
        match = _DATE_RE.search(sentence) or _YEAR_RE.search(sentence)
        return match.group(0) if match else None
    if shape == "number":
        match = _NUMBER_RE.search(sentence)
        return match.group(0) if match else None
    return None


def _fuller_date(answer: str, sentence: str) -> str | None:
    """The complete date of ``sentence`` when ``answer`` is only part of it.

    "January 14" against "Wedding Dress was released on January 14, 2010." The
    containment test is what keeps this an *extension* rather than a
    substitution: a date in the sentence that does not contain the emitted one
    is a different date, and swapping it in would be a guess.

    Growing a bare **number** the same way ("650" -> "650 locations") was
    simulated over the same 250 traces and measured +1/-1 -- a null that also
    breaks "played how many minutes: 14" into "14 days". It is deliberately not
    done.
    """
    for match in _DATE_RE.finditer(sentence):
        span = match.group(0)
        if _contains(span, answer) and normalize_answer(span) != normalize_answer(answer):
            return span
    return None


def reconcile_answer(
    question: str, answer: str, sentence: str
) -> tuple[str, str | None]:
    """Repair ``answer`` against the model's own ``answer_sentence``.

    Returns ``(answer, reason)``. ``reason`` is ``None`` when nothing fired, and
    the answer then comes back byte-identical.

    Every trigger requires the emitted answer to **contradict the question**.
    That restriction was bought with a measured regression, and is the whole
    design of this function:

        Which sport has been played at the BayArena in Leverkusen, Germany,
        since 1958?
        answer   "football"                                    <- correct
        sentence "The BayArena has been the home ground of Bayer Leverkusen
                  since 1958."

    An earlier version of this rule also fired when the answer was simply not a
    span of the sentence, on the theory that the model had contradicted itself.
    It has not. The model answers from the evidence pool and writes a sentence
    that need not restate the answer, so **absence is not contradiction**; that
    trigger rewrote the correct "football" into "Bayer Leverkusen" and cost a
    question on calib. It was dropped rather than narrowed. The 250-question
    eval slice contains no example of this shape, which is exactly why replaying
    on one slice and counting only recoveries is not enough to keep a rule.

    ``yesno_slot``
        A bare "yes"/"no" for a question that is not a yes/no question. The
        question's own surface form rules the answer out; nothing about the
        sentence is being second-guessed.
    ``type_mismatch``
        A bare number or year where the question asks who/which/what/where. A
        year cannot answer "which magazine", whatever the sentence says.
    ``underspecified_date``
        The answer is a proper part of a date the sentence spells out in full.
        An extension, never a substitution -- see :func:`_fuller_date`.

    Breakages are the number that governs here, not recoveries: a rule that
    recovers fifteen and breaks ten is worse than no rule at all.

    Replayed offline over the 250-question eval trace and the 50-question calib
    baseline together -- 9 fires, 6 wrong answers made right, **0 right answers
    broken**. Measured live on calib (``results/runs/ans_1a_only``), scoring the
    model's own answer against the substituted one inside the same run so that
    generation drift cannot enter: 3 fires, 1 recovered, **0 broken**.

    The run-level EM of that same calib run fell 2 questions, and none of it is
    this rule: on the 47 questions where nothing fired -- byte-identical code --
    5 answers moved anyway and EM moved -1. At n=50 this configuration's own
    noise is larger than the effect, so the honest claim for 1A is the breakage
    count, not a direction on EM.
    """
    if not answer or not sentence:
        return answer, None
    shape = question_shape(question)
    normalised = normalize_answer(answer)
    if not normalised:
        return answer, None
    is_yesno_answer = normalised in ("yes", "no")

    if is_yesno_answer and shape != "yesno":
        reason = "yesno_slot"
    elif shape == "entity" and _BARE_VALUE_RE.match(normalised):
        reason = "type_mismatch"
    else:
        # The answer's type satisfies the question, so it stands -- whether or
        # not the sentence happens to repeat it. The one repair left is
        # completing a date the model truncated; its selector is its own
        # trigger, so it returns here rather than falling through.
        fuller = _fuller_date(answer, sentence) if shape == "date" else None
        return (fuller, "underspecified_date") if fuller else (answer, None)

    replacement = _select_alternative(question, sentence)
    if replacement is None:
        replacement = _select_by_shape(question, sentence, shape)
    if replacement is None or normalize_answer(replacement) == normalised:
        return answer, None          # nothing defensible to say: leave it alone
    return replacement, reason


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Synthesizer(BaseAgent):
    """Compose the final short answer from the aggregated evidence."""

    name = "synthesizer"

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        client: Any | None = None,
        prompt_dir: Path | None = None,
    ) -> None:
        super().__init__(cfg, client=client, prompt_dir=prompt_dir)
        self.max_evidence = int(self.cfg.get("agents.verifier.max_evidence", 20))
        #: Whether :func:`reconcile_answer` (accuracy plan 1A) runs on a parsed
        #: LLM answer. Ships **false**, which is the behaviour of the eighteen
        #: evaluated runs: ``agentic_full`` must keep meaning what the report
        #: says it means. ``agentic_v2`` sets it true. The precedent is
        #: ``agents.verifier.entail_target``, which ships at its evaluated
        #: value with the correction selectable beside it.
        self.reconcile = bool(self.cfg.get("agents.synthesizer.reconcile_answer", False))
        #: Whether a plan node may flip the answer type into or out of
        #: ``"yesno"`` (accuracy plan 1B). Ships **false**: see
        #: :func:`infer_answer_type` for the 12-of-250 measurement that made
        #: this a flag rather than an unconditional fix.
        self.yesno_guard = bool(
            self.cfg.get("agents.synthesizer.answer_type_guard", False)
        )

    # -- entry point -------------------------------------------------------

    def run(
        self,
        state: QuestionState,
        evidence: Sequence[Evidence] | Mapping[str, Evidence] | None = None,
        *,
        question: str | None = None,
        answer_type: AnswerType | None = None,
    ) -> AnswerCandidate:
        """Produce one ``AnswerCandidate``. Never raises.

        The candidate is returned, not appended: ``state.candidates`` is a
        cycle-level artefact list owned by the orchestrator (transition T6).
        """
        started = time.perf_counter()
        q = question if question is not None else state.question
        pool = self._pool(state, evidence)
        atype = answer_type or infer_answer_type(
            q, state.plan, yesno_guard=self.yesno_guard
        )
        candidate: AnswerCandidate | None = None

        with state.step(
            self.name, n_evidence=len(pool), answer_type=atype
        ) as rec:
            if pool:
                candidate = self._llm_answer(state, q, atype, pool, rec)
            else:
                rec.degrade("no_evidence")
            if candidate is None:
                candidate = self._fallback(state, q, atype, pool, rec)
            # update, not assign: ``_llm_answer`` may already have recorded a
            # reconciliation here and that record is the audit trail for 1A.
            rec.output_summary.update({
                "answer": candidate.answer[:120],
                "origin": candidate.origin,
                "sufficient": candidate.sufficient,
                "n_citations": len(candidate.citations),
            })
        if candidate is None:  # the step swallowed an exception
            candidate = self._empty(state)
        _ = time.perf_counter() - started
        return candidate

    # -- evidence ----------------------------------------------------------

    def _pool(
        self,
        state: QuestionState,
        evidence: Sequence[Evidence] | Mapping[str, Evidence] | None,
    ) -> tuple[Evidence, ...]:
        """The ranked evidence this call may use, capped at ``max_evidence``.

        Sorted ``(-score, evidence_id)``: the tiebreaker is what makes two runs
        of the same configuration cite the same passages.
        """
        if evidence is None:
            items: Sequence[Evidence] = list(state.evidence.values())
        elif isinstance(evidence, Mapping):
            items = list(evidence.values())
        else:
            items = list(evidence)
        ranked = sorted(items, key=lambda e: (-e.score, e.evidence_id))
        return tuple(ranked[: self.max_evidence])

    @staticmethod
    def _render_evidence(pool: Sequence[Evidence]) -> str:
        lines = []
        for ev in pool:
            title = f" ({ev.title})" if ev.title else ""
            text = ev.text.strip()
            if len(text) > MAX_EVIDENCE_CHARS:
                text = text[:MAX_EVIDENCE_CHARS].rstrip() + " ..."
            lines.append(f"[{ev.evidence_id}]{title} {text}")
        return "\n".join(lines)

    # -- LLM path ----------------------------------------------------------

    def _llm_answer(
        self,
        state: QuestionState,
        question: str,
        answer_type: AnswerType,
        pool: Sequence[Evidence],
        rec: Any,
    ) -> AnswerCandidate | None:
        """The single privileged call. Returns ``None`` to fall through.

        Privileged: ``reserve_llm_calls`` exists precisely so that an answer is
        always produced even after a talkative Planner. ``call_json`` spends the
        budget, issues the repair rung and traces the call.
        """
        call = self.call_json(
            state,
            rec,
            prompt_id=PROMPT_ID,
            variables={
                "question": question,
                "answer_type": answer_type,
                "evidence": self._render_evidence(pool),
            },
            schema=_ANSWER_SCHEMA,
            purpose="synthesize",
            privileged=True,
        )
        if call.failed:
            rec.degrade(f"llm_failed:{call.reason}")
            return None

        parsed = call.parsed or {}
        answer = str(parsed.get("answer") or "").strip()
        if not answer:
            rec.degrade("empty_answer")
            return None
        sentence = str(parsed.get("answer_sentence") or "").strip()
        if not sentence:
            sentence = declarative_sentence(question, answer)
        # 1A. The model's own sentence is the better witness when the two
        # disagree. Only the slot is repaired -- the sentence is left alone, so
        # the Verifier still scores the proposition the model actually asserted.
        # Gated on ``agents.synthesizer.reconcile_answer``; when it is false
        # (the shipped default, and what the evaluated grid ran) not even the
        # rule's own pure functions are entered, so this branch cannot perturb
        # the answer, the trace, or the RNG.
        reconciled, reason = (
            reconcile_answer(question, answer, sentence) if self.reconcile else (answer, None)
        )
        if reason is not None:
            # Every firing is written to the trace, so the false-positive rate
            # of this rule is countable after the fact instead of assumed.
            rec.output_summary["reconciled"] = {
                "reason": reason,
                "was": answer[:120],
                "now": reconciled[:120],
            }
            answer = reconciled
        raw_citations = parsed.get("citations")
        citations: list[str] = []
        if isinstance(raw_citations, (list, tuple)):
            for cid in raw_citations:
                cid = str(cid).strip()
                # Kept verbatim, including ids that do not exist: detecting
                # those is the Verifier's job (3.5 step 1), and silently
                # dropping them here would hide a real failure mode.
                if cid and cid not in citations:
                    citations.append(cid)
        if reason is not None:
            origin: Origin = "reconciled"
        else:
            origin = "llm" if call.retries == 0 else "llm_repaired"
        return AnswerCandidate(
            answer=answer,
            answer_sentence=sentence,
            citations=tuple(citations),
            cycle=state.cycle,
            origin=origin,
            sufficient=bool(parsed.get("sufficient", True)),
        )

    # -- deterministic ladder ---------------------------------------------

    def _fallback(
        self,
        state: QuestionState,
        question: str,
        answer_type: AnswerType,
        pool: Sequence[Evidence],
        rec: Any,
    ) -> AnswerCandidate:
        """Rungs 1-5 of 3.4, in order, with 5.4's comparison shortcut first."""
        if not rec.degraded:
            rec.degrade("extractive_fallback")
        if not pool:
            return self._empty(state)  # rung 5

        plan = state.plan
        if plan is not None and plan.strategy == "comparison":
            shortcut = self._comparison_rule(state, question, answer_type, pool)
            if shortcut is not None:  # rung 1 / section 5.4
                return shortcut
        if answer_type == "yesno":
            shortcut = self._comparison_rule(state, question, answer_type, pool)
            if shortcut is not None:
                return shortcut

        if answer_type in ("date", "number"):  # rung 2
            pattern = _DATE_RE if answer_type == "date" else _NUMBER_RE
            for ev in pool:
                match = pattern.search(ev.text) or (
                    _YEAR_RE.search(ev.text) if answer_type == "date" else None
                )
                if match:
                    return self._candidate(state, question, match.group(0), (ev,))

        if answer_type == "entity":  # rung 3
            for ev in pool:
                if ev.title:
                    return self._candidate(state, question, ev.title, (ev,))

        top = pool[0]  # rung 4
        span = " ".join(top.text.split()[:MAX_FALLBACK_TOKENS])
        return self._candidate(state, question, span, (top,))

    def _comparison_rule(
        self,
        state: QuestionState,
        question: str,
        answer_type: AnswerType,
        pool: Sequence[Evidence],
    ) -> AnswerCandidate | None:
        """Section 5.4: answer a comparison by arithmetic, with zero generation.

        Needs two operands with a parseable year each. Operands are the plan's
        root sub-queries when there is a plan, and otherwise the two distinct
        evidence titles ranked highest -- which is what a fallback plan
        (F1/F3) leaves behind. Returns ``None`` rather than guessing whenever
        either operand fails to yield a year.
        """
        operands = self._operand_evidence(state, pool)
        if len(operands) < 2:
            return None
        (title_a, ev_a, year_a), (title_b, ev_b, year_b) = operands[:2]
        q = question.lower()

        if any(marker in q for marker in _EARLIER_MARKERS):
            winner, other = (title_a, title_b) if year_a <= year_b else (title_b, title_a)
            sentence = f"{winner} came before {other}."
            answer = "yes" if answer_type == "yesno" else winner
            if answer_type == "yesno":
                sentence = declarative_sentence(question, answer)
        elif any(marker in q for marker in _LATER_MARKERS):
            winner, other = (title_a, title_b) if year_a >= year_b else (title_b, title_a)
            sentence = f"{winner} came after {other}."
            answer = "yes" if answer_type == "yesno" else winner
            if answer_type == "yesno":
                sentence = declarative_sentence(question, answer)
        elif answer_type == "yesno" and any(m in q for m in _SAMENESS_MARKERS):
            answer = "yes" if year_a == year_b else "no"
            sentence = declarative_sentence(question, answer)
        else:
            return None

        return AnswerCandidate(
            answer=answer,
            answer_sentence=sentence,
            citations=tuple(dict.fromkeys((ev_a.evidence_id, ev_b.evidence_id))),
            cycle=state.cycle,
            origin="fallback_rule",
            sufficient=True,
        )

    @staticmethod
    def _operand_evidence(
        state: QuestionState, pool: Sequence[Evidence]
    ) -> list[tuple[str, Evidence, int]]:
        """``(title, evidence, year)`` for each comparison operand, best first."""
        plan = state.plan
        groups: list[tuple[str, Sequence[Evidence]]] = []
        if plan is not None:
            roots = [sq for sq in plan.subqueries if not sq.depends_on]
            for sq in sorted(roots, key=lambda s: int(s.id[1:]))[:2]:
                groups.append((sq.id, [e for e in pool if sq.id in e.subquery_ids]))
        if len(groups) < 2 or any(not evs for _, evs in groups):
            # No usable plan grouping: fall back to distinct evidence titles.
            groups = []
            for ev in pool:
                key = ev.title or ev.doc_id or ev.evidence_id
                if key not in {k for k, _ in groups}:
                    groups.append((key, [e for e in pool if (e.title or e.doc_id) == key]))
                if len(groups) == 2:
                    break
        out: list[tuple[str, Evidence, int]] = []
        for _key, evs in groups[:2]:
            for ev in evs:
                match = _YEAR_RE.search(ev.text)
                if match:
                    out.append((ev.title or _key, ev, int(match.group(0))))
                    break
        return out

    # -- candidate construction -------------------------------------------

    @staticmethod
    def _candidate(
        state: QuestionState,
        question: str,
        answer: str,
        used: Sequence[Evidence],
    ) -> AnswerCandidate:
        return AnswerCandidate(
            answer=answer.strip(),
            answer_sentence=declarative_sentence(question, answer.strip()),
            citations=tuple(ev.evidence_id for ev in used),
            cycle=state.cycle,
            origin="fallback_rule",
            sufficient=bool(answer.strip()),
        )

    @staticmethod
    def _empty(state: QuestionState) -> AnswerCandidate:
        """Rung 5: no evidence at all. Still a candidate, so FINALIZE works."""
        return AnswerCandidate(
            answer="",
            answer_sentence="",
            citations=(),
            cycle=state.cycle,
            origin="fallback_rule",
            sufficient=False,
        )
