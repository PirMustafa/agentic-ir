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
from ..state import QuestionState
from ..types import AnswerCandidate, AnswerType, Evidence, Plan
from .base import BaseAgent

__all__ = ["PROMPT_ID", "Synthesizer", "declarative_sentence", "infer_answer_type"]

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

def infer_answer_type(question: str, plan: Plan | None = None) -> AnswerType:
    """The expected answer type: the plan's own label first, then the question.

    The Planner assigns ``answer_type`` per sub-query and the combiner (or last)
    node carries the type of the *final* answer, so that is the better source
    when it is informative. ``"string"`` is the schema default and therefore
    carries no information -- fall through to the surface form of the question
    instead of trusting it.
    """
    if plan is not None and plan.subqueries:
        combiner = next((sq for sq in plan.subqueries if sq.is_combiner), None)
        node = combiner or plan.subqueries[-1]
        if node.answer_type != "string":
            return node.answer_type
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
        atype = answer_type or infer_answer_type(q, state.plan)
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
            rec.output_summary = {
                "answer": candidate.answer[:120],
                "origin": candidate.origin,
                "sufficient": candidate.sufficient,
                "n_citations": len(candidate.citations),
            }
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
        return AnswerCandidate(
            answer=answer,
            answer_sentence=sentence,
            citations=tuple(citations),
            cycle=state.cycle,
            origin="llm" if call.retries == 0 else "llm_repaired",
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
