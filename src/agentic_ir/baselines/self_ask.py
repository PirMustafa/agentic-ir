"""``self_ask`` -- the published multi-hop baseline (Press et al., 2023).

*Measuring and Narrowing the Compositionality Gap in Language Models*
(Findings of EMNLP 2023). The method: before answering, the model is asked
whether a **follow-up question** is needed. If it is, the follow-up goes to a
retriever, the retrieved answer is written back into a scratchpad as an
*intermediate answer*, and the model is asked again. When it declares no
further follow-up is needed, it answers. That is the whole loop, and it is the
strongest widely-cited prompting method that decomposes a multi-hop question
without any verification or re-planning.

Why this is the most informative row in the table
-------------------------------------------------
``self_ask`` already has the thing ``naive_rag`` lacks -- **decomposition**, and
therefore a second retrieval that is conditioned on what the first one found.
It does *not* have the two things the agentic system adds on top: it never
checks whether its answer is supported (no Verifier), and it never revises a
decomposition that failed (no re-plan edge). So:

* ``self_ask − naive_rag``  isolates what multi-hop decomposition is worth;
* ``agentic_full − self_ask``  is close to a clean measurement of what the
  **feedback loop** is worth, because decomposition is held constant.

That second quantity is the central claim of the project, which is why this
baseline is implemented in good faith rather than as a formality: a weak
``self_ask`` would inflate the headline result by exactly the amount it was
weakened.

Four adaptations, and why each is the honest one
------------------------------------------------

**1. The corpus replaces the search engine.** Press et al. run Self-Ask + SE
against Google. Here the follow-up goes to the same hybrid+rerank stack
``hybrid_rerank`` and ``naive_rag`` use -- inherited by subclassing
:class:`~agentic_ir.baselines.naive_rag.NaiveRAGBaseline`, so the retrieval,
the passage rendering, the citation resolution and the deterministic fallback
are literally the same code. The only difference between this row and
``naive_rag`` is the loop.

**2. Hop ``q1`` retrieves the original question.** In the paper the model
decides its first follow-up from parametric knowledge alone. Retrieving the
question first is a deliberate *strengthening*, made for two reasons. It grounds
the first decision in the corpus, which is what any competent RAG
implementation of the method would do. And it makes this row's retrieval a
superset of ``hybrid_rerank``'s rather than a lottery on the model's opening
move -- without it, a model that answers "no follow-up needed" would score
Recall@10 = 0 for a *generation* failure, and the retrieval column would stop
measuring retrieval.

**3. The intermediate answer is extracted deterministically, not generated.**
Self-Ask + SE inserts the search engine's own answer snippet; it spends no
model call on it. The analogue here is the leading sentence of the follow-up's
top-ranked passage -- Wikipedia lead sentences are definitional, which is
exactly the shape of an answer-box snippet. Generating the intermediate answer
instead would roughly double this baseline's LLM cost and quietly change what
the ``llm_calls`` column is comparing.

**4. One structured call per turn, not free-form text.** The paper's scaffold
decodes once per turn and stops at ``Intermediate answer:``; a single decode
therefore emits *either* a follow-up *or* the final answer. The JSON schema
here has the same shape -- ``needs_followup`` plus both branches' fields -- so
the call count per question matches the published method turn for turn, while
parsing stays reliable (``llm.max_format_retries`` handles the rest).

Two hard stops, because a confused model must not cost a run
------------------------------------------------------------
``max_followups`` (3) caps the decomposition depth, and ``max_llm_calls`` (5) is
an absolute per-question ceiling that holds even when the loop misbehaves in a
way the follow-up counter does not see -- an empty follow-up, or the same
follow-up asked twice. The last call is reserved for answering, mirroring
``orchestrator.reserve_llm_calls``: whatever the loop does, the question gets an
answer rather than a timeout. Both limits are recorded per question
(``extra.stop_reason``) so Chapter 4 can report how often they bound the method
instead of assuming they never did.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from ..types import ScoredPassage
from .base import BaselineHop
from .naive_rag import (
    ANSWER_SCHEMA,
    NaiveRAGBaseline,
    render_passages,
)

__all__ = [
    "MAX_FOLLOWUPS",
    "STEP_SCHEMA",
    "SelfAskBaseline",
    "intermediate_answer",
    "normalise_query",
]

#: Decomposition depth. Three, because both benchmarks are two-hop by
#: construction: two follow-ups is the shape of the task and the third is
#: slack for a decomposition that names an intermediate entity before using it.
#: A larger cap would only let a confused model wander further from a question
#: whose gold evidence lives in exactly two paragraphs.
MAX_FOLLOWUPS = 3

#: Absolute per-question ceiling on model calls. The loop needs at most
#: ``MAX_FOLLOWUPS`` turns plus one to answer; the fifth is slack for a single
#: degenerate turn (an empty or repeated follow-up) so one glitch does not cost
#: the question its answer.
MAX_LLM_CALLS = 5

#: Calls held back for answering, exactly as ``orchestrator.reserve_llm_calls``
#: holds calls back for the Synthesizer. Without it a model that keeps asking
#: follow-ups spends the whole budget deciding and never answers -- which would
#: appear in the results as an accuracy failure rather than a budget policy.
RESERVE_LLM_CALLS = 1

#: Longest intermediate answer written back into the scratchpad. Long enough
#: for a Wikipedia lead sentence, short enough that three of them plus ten
#: passages stay inside ``llm.options.num_ctx`` (8192).
INTERMEDIATE_ANSWER_CHARS = 300

#: One decode, either branch -- see adaptation 4 in the module docstring. Every
#: field is required because Ollama's schema-constrained decoding fills what it
#: is asked for; optional fields invite a reply that omits the branch it took.
STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "needs_followup": {"type": "boolean"},
        "followup": {"type": "string"},
        "answer": {"type": "string"},
        "answer_sentence": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["needs_followup", "followup", "answer", "answer_sentence", "citations"],
}

_STEP_PROMPT = """\
You answer multi-hop questions by asking yourself follow-up questions.

Look at the question, the scratchpad so far, and the numbered passages. Decide
ONE of these:

A. A follow-up question is still needed. Set needs_followup to true and put the
   single next question in "followup". It must be a self-contained question --
   substitute any entity you have already found instead of writing "he", "it" or
   "that person". Leave "answer" empty.
B. The passages and the scratchpad are enough. Set needs_followup to false and
   answer:
   - "answer" is a SHORT SPAN copied from the passages -- a name, a date, a
     number, or "yes"/"no". Never a sentence, never an explanation.
   - "answer_sentence" is ONE declarative sentence stating the answer as a fact.
   - "citations" lists the passage ids you used, e.g. ["p1","p4"]. Cite only ids
     that appear below.
   Leave "followup" empty.

You have {remaining} follow-up question(s) left. Ask one only if you genuinely
cannot answer yet; never repeat a follow-up already in the scratchpad.

Return ONLY a JSON object of this exact shape:
{{"needs_followup": false, "followup": "", "answer": "Arthur's Magazine", \
"answer_sentence": "Arthur's Magazine was started before First for Women.", \
"citations": ["p1"]}}

Question: {question}

Scratchpad:
{scratchpad}

Passages:
{passages}"""

_FINAL_PROMPT = """\
Answer the question using ONLY the scratchpad and the numbered passages below.
No more follow-up questions are allowed.

Rules:
1. "answer" is a SHORT SPAN copied from the passages -- a name, a date, a
   number, or "yes"/"no". Never a sentence, never an explanation.
2. "answer_sentence" is ONE declarative sentence stating the answer as a fact.
3. "citations" lists the passage ids you used, e.g. ["p1","p4"]. Cite only ids
   that appear below.
4. If the evidence is incomplete, still give your best short span from it.

Return ONLY a JSON object of this exact shape:
{{"answer": "Arthur's Magazine", "answer_sentence": "Arthur's Magazine was \
started before First for Women.", "citations": ["p1"]}}

Question: {question}

Scratchpad:
{scratchpad}

Passages:
{passages}"""

#: Near-duplicate detection: case, articles and punctuation removed. Deliberately
#: crude -- it is a loop breaker, not a semantic matcher.
_SQUASH = re.compile(r"[^a-z0-9 ]+")
_ARTICLES = re.compile(r"\b(a|an|the|of|in|on|at|for|to|is|was|were|are)\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_query(text: str) -> str:
    """A query reduced to its content words, for the repeat-follow-up check.

    The agentic system's near-duplicate plan test (T2b) exists for the same
    reason and at the same crudeness: a model that re-asks its own last
    question will do it verbatim or with a stray article, not with a clever
    paraphrase, and a matcher tuned for paraphrase would start rejecting
    legitimately different second hops.
    """
    squashed = _SQUASH.sub(" ", text.strip().lower())
    return " ".join(_ARTICLES.sub(" ", squashed).split())


def intermediate_answer(
    passages: Sequence[ScoredPassage], *, max_chars: int = INTERMEDIATE_ANSWER_CHARS
) -> str:
    """The snippet written back into the scratchpad for one follow-up.

    The leading non-empty sentence of the top-ranked passage, title-prefixed
    when the sentence does not already begin with it. This is the corpus
    analogue of the search-engine answer box Self-Ask + SE feeds back, and like
    it, it costs zero model calls -- see adaptation 3 in the module docstring.

    Empty when the follow-up retrieved nothing, which the scratchpad renders as
    a stated failure rather than as silence: telling the model that a hop found
    nothing is information it can act on.
    """
    if not passages:
        return ""
    top = passages[0].passage
    sentence = next((s for s in top.sentences if s.strip()), top.text)
    text = " ".join(sentence.split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    title = top.title.strip()
    if title and not text.lower().startswith(title.lower()):
        text = f"{title}: {text}"
    return text


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class SelfAskBaseline(NaiveRAGBaseline):
    """Follow-up loop over the shared retrieval stack, then one answer."""

    name = "self_ask"
    generative = True

    max_followups: int = MAX_FOLLOWUPS
    max_llm_calls: int = MAX_LLM_CALLS
    reserve_llm_calls: int = RESERVE_LLM_CALLS

    def __init__(
        self,
        *args: Any,
        max_followups: int | None = None,
        **kwargs: Any,
    ) -> None:
        """``max_followups`` is exposed for the depth ablation Chapter 4 may want."""
        super().__init__(*args, **kwargs)
        if max_followups is not None:
            self.max_followups = int(max_followups)
        self._followups: list[str] = []
        self._scratchpad: list[tuple[str, str]] = []
        self._issued: set[str] = set()
        self._final: dict[str, Any] | None = None
        self._final_reason: str = ""
        self._stop_reason: str = ""
        self._parse_failures: int = 0

    # -- per-question state -------------------------------------------------
    def _begin(self) -> None:
        """Reset the loop as well as the call counter.

        ``BaselineBase`` calls this before :meth:`retrieve`, which is exactly
        why the hook exists: this baseline spends model calls inside
        ``retrieve`` *and* inside ``answer``, so the budget counter can live in
        neither one.
        """
        super()._begin()
        self._followups = []
        self._scratchpad = []
        self._issued = set()
        self._final = None
        self._final_reason = ""
        self._stop_reason = ""
        self._parse_failures = 0

    def plan_depth(self) -> int:
        """Retrieval rounds this question took: hop ``q1`` plus each follow-up.

        The honest analogue of the agentic ``plan_depth``. It is a *measured*
        depth, not a planned one -- ``self_ask`` decides one hop at a time and
        never commits to a DAG -- which is itself part of what separates this
        row from ``agentic_full``.
        """
        return 1 + len(self._followups)

    # -- the loop -----------------------------------------------------------
    def retrieve(self, question: str, top_k: int) -> list[BaselineHop]:
        """Hop ``q1`` for the question, then one hop per accepted follow-up.

        Each turn is one model call, and each turn sees the RRF fusion of every
        hop so far -- the *same* pool :meth:`answer` will later be handed, so
        the ``p1``..``pk`` labels the model cites mean the same passages at
        both ends. Showing a different pool at decision time and at scoring
        time would turn every citation into a silent off-by-one.
        """
        hops: list[BaselineHop] = [self._hop("q1", question, top_k)]

        while True:
            if len(self._followups) >= self.max_followups:
                self._stop_reason = "max_followups"
                break
            if self.calls_left <= self.reserve_llm_calls:
                self._stop_reason = "budget_reserved_for_answer"
                break

            pool = self.fuse(hops, top_k)
            parsed, reason = self.ask(
                _STEP_PROMPT.format(
                    question=question,
                    scratchpad=self._render_scratchpad(),
                    passages=render_passages(pool) or "(nothing retrieved)",
                    remaining=self.max_followups - len(self._followups),
                ),
                STEP_SCHEMA,
            )
            if parsed is None:
                self._parse_failures += int(reason == "parse_failure")
                self._final_reason = reason
                self._stop_reason = reason
                break

            followup = str(parsed.get("followup") or "").strip()
            if not parsed.get("needs_followup") or not followup:
                # Branch B: this turn is the answer. Stash it; ``answer()``
                # resolves its citations against the pool it was shown, which
                # is the pool the loop stops on.
                self._final = parsed
                self._stop_reason = "model_answered"
                break

            key = normalise_query(followup)
            if not key or key in self._issued:
                # The loop-breaker. A model repeating its own follow-up will
                # repeat it forever, and every extra turn costs ~10s and buys
                # nothing: the hop it would issue has already been issued.
                self._stop_reason = "repeated_followup"
                break

            self._issued.add(key)
            self._followups.append(followup)
            hop = self._hop(f"q{len(hops) + 1}", followup, top_k)
            snippet = intermediate_answer(hop.passages)
            hops.append(
                BaselineHop(
                    hop_id=hop.hop_id,
                    query=hop.query,
                    tool=hop.tool,
                    passages=hop.passages,
                    rerank_applied=hop.rerank_applied,
                    latency_s=hop.latency_s,
                    intermediate_answer=snippet,
                )
            )
            self._scratchpad.append((followup, snippet))

        return hops

    def _hop(self, hop_id: str, query: str, top_k: int) -> BaselineHop:
        """One retrieval round through the inherited hybrid+rerank pipeline."""
        started = time.perf_counter()
        passages, reranked = self.search(query, top_k)
        return BaselineHop(
            hop_id=hop_id,
            query=query,
            tool=self.tool,
            passages=tuple(passages),
            rerank_applied=reranked,
            latency_s=time.perf_counter() - started,
        )

    def _render_scratchpad(self) -> str:
        """The Self-Ask scratchpad, in the paper's own surface form."""
        if not self._scratchpad:
            return "(empty -- no follow-up questions asked yet)"
        lines: list[str] = []
        for followup, snippet in self._scratchpad:
            lines.append(f"Follow up: {followup}")
            lines.append(
                f"Intermediate answer: {snippet}"
                if snippet
                else "Intermediate answer: (the corpus returned nothing for this)"
            )
        return "\n".join(lines)

    # -- answering ----------------------------------------------------------
    def answer(
        self, question: str, hops: Sequence[BaselineHop], passages: Sequence[ScoredPassage]
    ) -> dict[str, Any]:
        """Use the loop's own answer, or spend the reserved call to force one."""
        parsed = self._final
        reason = self._final_reason
        has_answer = parsed is not None and str(parsed.get("answer") or "").strip()

        if not has_answer:
            if not passages:
                reason = "no_passages"
            else:
                forced, forced_reason = self.ask(
                    _FINAL_PROMPT.format(
                        question=question,
                        scratchpad=self._render_scratchpad(),
                        passages=render_passages(passages),
                    ),
                    ANSWER_SCHEMA,
                )
                self._parse_failures += int(forced_reason == "parse_failure")
                if forced is not None:
                    parsed, reason = forced, forced_reason
                else:
                    reason = forced_reason
                    if self._stop_reason in ("model_answered", ""):
                        self._stop_reason = forced_reason

        payload = self._payload(question, passages, parsed, reason)
        payload["extra"].update(
            {
                "followups": list(self._followups),
                "n_followups": len(self._followups),
                "stop_reason": self._stop_reason,
                "parse_failures": self._parse_failures,
                "budget_exhausted": self.calls_left <= 0,
                "terminated_by": f"{self.name}:{self._stop_reason or 'unknown'}",
            }
        )
        return payload
