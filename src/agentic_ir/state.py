"""Working memory for one question, and the budget that bounds it.

Two design rules from ``docs/architecture.md`` live here.

*Budget is checked, never thrown.* :meth:`Budget.try_spend_llm` returns a bool;
agents branch to a deterministic fallback when it returns False. Exception-driven
budget control would make "ran out of calls" indistinguishable from "crashed".

*No agent may raise.* :meth:`QuestionState.step` catches everything, records an
``AgentTrace`` marked degraded, and hands control back to the caller's fallback.
A 250-question run cannot die at question 194.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .types import (
    AgentTrace,
    AnswerCandidate,
    Evidence,
    KGResult,
    LLMCallTrace,
    Plan,
    ReplanDirective,
    RetrievalResult,
    ToolCallTrace,
    VerificationResult,
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    ["a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was", "were", "are", "be", "been", "by", "with", "from", "as", "that", "this", "it", "its", "which", "who", "whom", "whose", "what", "when", "where"]
)


def normalise_text(text: str) -> frozenset[str]:
    """Content-word set used for the near-duplicate plan test and ban list.

    Placeholders are stripped first: ``{{q1.answer}}`` differs between
    revisions purely by binding, not by intent, so leaving it in would make two
    identical plans look different.
    """
    text = re.sub(r"\{\{[^}]*\}\}", " ", text.lower())
    return frozenset(w for w in _WORD_RE.findall(text) if w not in _STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Hard caps on one question's work.

    ``reserve_llm_calls`` is what guarantees an answer always gets produced:
    non-privileged callers (planner, extraction, KG) see the reserve subtracted,
    while the synthesizer and verifier see the true remainder. Without it a
    talkative planner can consume the whole budget and leave nothing to answer
    with, which reads in the results as a system failure rather than a budget
    policy.
    """

    max_llm_calls: int = 20
    max_iterations: int = 6
    max_wall_clock_s: float = 300.0
    max_replans: int = 2
    reserve_llm_calls: int = 2

    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_calls_saved: int = 0
    tool_calls: int = 0
    iterations: int = 0
    replans: int = 0
    t0: float = field(default_factory=perf_counter)

    @classmethod
    def from_config(cls, cfg: Any) -> Budget:
        return cls(
            max_llm_calls=int(cfg.get("orchestrator.max_llm_calls_per_question", 20)),
            max_iterations=int(cfg.get("orchestrator.max_iterations", 6)),
            max_wall_clock_s=float(cfg.get("orchestrator.max_wall_clock_s", 300)),
            max_replans=int(cfg.get("agents.planner.max_replans", 2)),
            reserve_llm_calls=int(cfg.get("orchestrator.reserve_llm_calls", 2)),
        )

    def start(self) -> Budget:
        self.t0 = perf_counter()
        return self

    def elapsed(self) -> float:
        return perf_counter() - self.t0

    def wallclock_exceeded(self) -> bool:
        return self.elapsed() >= self.max_wall_clock_s

    def remaining_llm(self, *, privileged: bool = False) -> int:
        reserve = 0 if privileged else self.reserve_llm_calls
        return max(0, self.max_llm_calls - self.llm_calls - reserve)

    def try_spend_llm(self, *, privileged: bool = False) -> bool:
        """Reserve one LLM call. Returns False instead of raising."""
        if self.wallclock_exceeded():
            return False
        if self.remaining_llm(privileged=privileged) <= 0:
            return False
        self.llm_calls += 1
        return True

    def note_saved(self, n: int = 1) -> None:
        """Record an LLM call avoided by a deterministic rule."""
        self.llm_calls_saved += n

    def note_tool(self, n: int = 1) -> None:
        self.tool_calls += n

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "llm_calls_saved": self.llm_calls_saved,
            "llm_cache_hits": self.llm_cache_hits,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "replans": self.replans,
            "elapsed_s": round(self.elapsed(), 4),
            "budget_exhausted": self.remaining_llm(privileged=True) <= 0
            or self.wallclock_exceeded(),
        }


# ---------------------------------------------------------------------------
# Step recorder
# ---------------------------------------------------------------------------

@dataclass
class StepRecorder:
    """Mutable scratch handed to an agent inside :meth:`QuestionState.step`."""

    agent: str
    state: str
    step: int
    cycle: int
    subquery_id: str | None
    started_at: float
    llm_calls: list[LLMCallTrace] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    fallback_reason: str | None = None

    def note_llm(self, trace: LLMCallTrace) -> None:
        self.llm_calls.append(trace)

    def note_tool(self, trace: ToolCallTrace) -> None:
        self.tool_calls.append(trace)

    def degrade(self, reason: str) -> None:
        """Mark this step as having taken a deterministic fallback path."""
        self.degraded = True
        self.fallback_reason = reason


# ---------------------------------------------------------------------------
# Question state
# ---------------------------------------------------------------------------

@dataclass
class QuestionState:
    """Everything known about one question in flight.

    ``gold`` is carried for the trace only. No agent may read it -- a guard
    test enforces that, because an agent that peeks would make every number in
    Chapter 4 an artefact while the scores merely looked good.
    """

    qid: str
    question: str
    dataset: str
    config_name: str
    gold: Any | None = None

    plans: list[Plan] = field(default_factory=list)
    directives: list[ReplanDirective] = field(default_factory=list)
    results: dict[str, RetrievalResult] = field(default_factory=dict)
    kg_results: dict[str, KGResult] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    bridge_entities: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    candidates: list[AnswerCandidate] = field(default_factory=list)
    verifications: list[VerificationResult] = field(default_factory=list)
    traces: list[AgentTrace] = field(default_factory=list)

    budget: Budget = field(default_factory=Budget)
    state: str = "INIT"
    cycle: int = 0
    terminated_by: str | None = None
    errors: list[str] = field(default_factory=list)
    _step_no: int = 0

    # -- convenience views -------------------------------------------------

    @property
    def plan(self) -> Plan | None:
        return self.plans[-1] if self.plans else None

    @property
    def verification(self) -> VerificationResult | None:
        return self.verifications[-1] if self.verifications else None

    def banned_texts(self) -> tuple[str, ...]:
        """Every sub-query text already tried, across all plan revisions."""
        seen: list[str] = []
        for plan in self.plans:
            for sq in plan.subqueries:
                if sq.text not in seen:
                    seen.append(sq.text)
        return tuple(seen)

    def seen_doc_ids(self) -> tuple[str, ...]:
        ids = {sp.passage.doc_id for r in self.results.values() for sp in r.passages}
        return tuple(sorted(ids))

    def covered_entities(self) -> tuple[str, ...]:
        ents = set(self.bridge_entities.values())
        ents.update(v for v in self.answers.values() if v)
        return tuple(sorted(e for e in ents if e))

    def is_duplicate_plan(self, plan: Plan, threshold: float = 0.85) -> bool:
        """Near-duplicate test (architecture 2.4).

        Two plans match when they have the same node count and a greedy
        max-Jaccard pairing puts every node above ``threshold``. Plans have at
        most five nodes, so greedy pairing is exhaustive enough.
        """
        new = [normalise_text(sq.text) for sq in plan.subqueries]
        for prior in self.plans:
            old = [normalise_text(sq.text) for sq in prior.subqueries]
            if len(old) != len(new):
                continue
            pool = list(old)
            matched = True
            for n in new:
                if not pool:
                    matched = False
                    break
                best_i, best_j = max(
                    ((i, jaccard(n, o)) for i, o in enumerate(pool)), key=lambda t: t[1]
                )
                if best_j < threshold:
                    matched = False
                    break
                pool.pop(best_i)
            if matched:
                return True
        return False

    # -- step recording ----------------------------------------------------

    @contextmanager
    def step(
        self,
        agent: str,
        *,
        subquery_id: str | None = None,
        **input_summary: Any,
    ) -> Iterator[StepRecorder]:
        """Time an agent call, record it, and never let it raise.

        On exception the step is marked degraded, the error is appended to
        ``state.errors``, and control returns to the caller -- which is then
        responsible for producing its deterministic fallback.
        """
        rec = StepRecorder(
            agent=agent,
            state=self.state,
            step=self._step_no,
            cycle=self.cycle,
            subquery_id=subquery_id,
            started_at=self.budget.elapsed(),
            input_summary=dict(input_summary),
        )
        self._step_no += 1
        t0 = perf_counter()
        try:
            yield rec
        except Exception as exc:  # noqa: BLE001 - deliberate: agents never raise
            rec.degrade(f"{type(exc).__name__}: {exc}")
            self.errors.append(f"{agent}: {type(exc).__name__}: {exc}")
        finally:
            self.traces.append(
                AgentTrace(
                    agent=rec.agent,
                    state=rec.state,
                    step=rec.step,
                    cycle=rec.cycle,
                    subquery_id=rec.subquery_id,
                    started_at=round(rec.started_at, 4),
                    latency_s=round(perf_counter() - t0, 4),
                    llm_calls=tuple(rec.llm_calls),
                    tool_calls=tuple(rec.tool_calls),
                    input_summary=rec.input_summary,
                    output_summary=rec.output_summary,
                    degraded=rec.degraded,
                    fallback_reason=rec.fallback_reason,
                )
            )

    # -- aggregate metrics -------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        """The ``metrics`` block of this question's trace record."""
        plan = self.plan
        best = self.best_candidate()
        ver = self.verification
        llm_latency = sum(c.latency_s for t in self.traces for c in t.llm_calls)
        parse_failures = sum(
            1 for t in self.traces for c in t.llm_calls if not c.parse_ok
        )
        return {
            "llm_calls": self.budget.llm_calls,
            "llm_calls_saved": self.budget.llm_calls_saved,
            "llm_cache_hits": self.budget.llm_cache_hits,
            "tool_calls": self.budget.tool_calls,
            "parse_failures": parse_failures,
            "latency_s": round(self.budget.elapsed(), 4),
            "llm_latency_s": round(llm_latency, 4),
            "plan_depth": plan.depth if plan else 0,
            "n_subqueries": len(plan.subqueries) if plan else 0,
            "cycles": len(self.plans),
            "replans": self.budget.replans,
            "replanned": self.budget.replans > 0,
            "best_cycle": best.cycle if best else None,
            "citation_grounding": ver.citation_grounding if ver else None,
            "nli_support": ver.nli_support if ver else None,
            "hallucinated_citations": len(ver.hallucinated_citations) if ver else 0,
            "answered": bool(best and best.answer),
            "degraded_steps": sum(1 for t in self.traces if t.degraded),
            "budget_exhausted": self.budget.to_dict()["budget_exhausted"],
        }

    def best_candidate(self) -> AnswerCandidate | None:
        """argmax(confidence) across ALL cycles, ties to the earlier cycle.

        Never simply the last cycle: a re-plan can make an answer worse, and
        letting it overwrite a better first answer would make the feedback loop
        look harmful in the results for the wrong reason.
        """
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: (c.confidence, -c.cycle))


__all__ = ["Budget", "QuestionState", "StepRecorder", "normalise_text", "jaccard"]
