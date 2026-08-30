"""The orchestrator: an explicit state machine (architecture sections 2 and 4).

INIT, PLAN, EXECUTE, AGGREGATE, SYNTHESIZE, VERIFY, REPLAN_GATE, FINALIZE, DONE,
and the fourteen transitions between them. Explicit rather than implicit
control flow for two reasons: the mechanism has to be visible to be assessable,
and a machine you can draw is a machine you can test transition by transition.

Three rules here are load-bearing and easy to get subtly wrong:

* **T2b discards a degenerate re-plan.** Told "that was insufficient, try
  again", an 8B model re-emits the same decomposition. Detecting that and
  stopping is what keeps the feedback loop from burning both re-plans to
  produce the same answer twice.
* **FINALIZE takes argmax confidence over ALL cycles**, never the last one. A
  re-plan can make an answer worse; letting it overwrite silently would make
  the feedback loop look harmful in the results for the wrong reason.
* **Placeholder rung 4 deletes rather than aborts.** A node with a residual
  query still contributes evidence, and the Verifier -- not the resolver -- is
  the component entitled to judge whether that evidence is enough.

The KG navigator, synthesizer and verifier live in a sibling milestone. They
are imported lazily and their absence is a degraded path, not a crash, so this
module runs before they land.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .agents.base import collapse_whitespace, strip_placeholder
from .agents.planner import Planner, fallback_plan
from .agents.retriever import RetrievalAgent, SpanExtractor
from .config import Config, load_config
from .state import Budget, QuestionState, jaccard, normalise_text
from .tools.registry import ToolRegistry
from .trace import TraceWriter
from .types import (
    AnswerCandidate,
    Evidence,
    Plan,
    ReplanDirective,
    ReplanReason,
    RetrievalResult,
    ScoredPassage,
    SubQuery,
    VerificationResult,
)

__all__ = ["Orchestrator", "STATES", "TRANSITIONS"]

STATES = (
    "INIT", "PLAN", "EXECUTE", "AGGREGATE", "SYNTHESIZE",
    "VERIFY", "REPLAN_GATE", "FINALIZE", "DONE",
)

#: The transition table of section 2.2, as data. Every id the machine can emit
#: appears here, which is what lets a test assert that a path used T2b and not
#: T2 without reading the log by eye.
TRANSITIONS: dict[str, tuple[str, str, str]] = {
    "T1": ("INIT", "PLAN", "always"),
    "T2": ("PLAN", "EXECUTE", "plan validated"),
    "T2b": ("PLAN", "FINALIZE", "re-plan is a near-duplicate"),
    "T3": ("EXECUTE", "EXECUTE", "next ready node"),
    "T4": ("EXECUTE", "AGGREGATE", "all nodes terminal or budget spent"),
    "T5": ("AGGREGATE", "SYNTHESIZE", "always"),
    "T6": ("SYNTHESIZE", "VERIFY", "verifier enabled"),
    "T7": ("SYNTHESIZE", "FINALIZE", "verifier disabled"),
    "T8": ("VERIFY", "FINALIZE", "verdict accept"),
    "T9": ("VERIFY", "REPLAN_GATE", "verdict revise"),
    "T10": ("VERIFY", "FINALIZE", "verdict abstain"),
    "T11": ("REPLAN_GATE", "PLAN", "G1..G5 all pass"),
    "T12": ("REPLAN_GATE", "FINALIZE", "a guard failed"),
    "T13": ("FINALIZE", "DONE", "always"),
    "T14": ("*", "FINALIZE", "unhandled exception"),
}

#: An outer stop so a mis-wired handler cannot spin forever. Six iterations of
#: PLAN..VERIFY with five nodes each cannot approach this.
_MAX_MACHINE_STEPS = 400

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{3,4}\b|\b(1[0-9]{3}|20[0-9]{2})\b",
    re.I,
)
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

#: Lazily-imported collaborators: module, then the class names it might use.
_OPTIONAL_AGENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "kg": (".agents.kg_navigator", ("KGNavigator", "KgNavigator", "KGAgent", "Navigator")),
    "synthesizer": (".agents.synthesizer", ("Synthesizer", "SynthesizerAgent", "AnswerSynthesizer")),
    "verifier": (".agents.verifier", ("Verifier", "VerifierAgent", "Validator")),
}


class Orchestrator:
    """Runs one question through the machine and returns its :class:`QuestionState`."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        cfg: Config | None = None,
        config_name: str = "agentic_full",
        dataset: str = "hotpotqa",
        planner: Any = None,
        retriever: Any = None,
        extractor: Any = None,
        kg: Any = None,
        synthesizer: Any = None,
        verifier: Any = None,
        writer: TraceWriter | None = None,
        client: Any = None,
        bm25: Any = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.config_name = config_name
        self.dataset = dataset
        self.registry = registry if registry is not None else ToolRegistry()
        self.writer = writer
        self.client = client
        self.planner = planner if planner is not None else Planner(self.cfg, client=client)
        self.retriever = (
            retriever
            if retriever is not None
            else RetrievalAgent(self.registry, self.cfg, bm25=bm25, client=client)
        )
        self.extractor = extractor if extractor is not None else SpanExtractor(self.cfg, client=client)
        self._injected: dict[str, Any] = {"kg": kg, "synthesizer": synthesizer, "verifier": verifier}
        self._resolved: dict[str, Any] = {}
        self.kg_enabled = bool(self.cfg.get("agents.kg.enabled", True))
        self.verifier_enabled = bool(self.cfg.get("agents.verifier.enabled", True))
        self.max_evidence = int(self.cfg.get("agents.verifier.max_evidence", 20))
        self.evidence_passages = int(self.cfg.get("agents.verifier.evidence_passages", 3))
        self.rrf_k = int(self.cfg.get("retrieval.hybrid.rrf_k", 60))
        #: Transition ids of the most recent :meth:`run`, for the trace and tests.
        self.transitions: list[str] = []
        self._directive: ReplanDirective | None = None

    def __repr__(self) -> str:
        return f"Orchestrator(config={self.config_name!r}, dataset={self.dataset!r})"

    # -- entry point -------------------------------------------------------
    def run(
        self,
        qid: str,
        question: str,
        *,
        gold: Any = None,
        state: QuestionState | None = None,
    ) -> QuestionState:
        """Drive one question from INIT to DONE. Never raises."""
        state = state or QuestionState(
            qid=qid,
            question=collapse_whitespace(question),
            dataset=self.dataset,
            config_name=self.config_name,
            gold=gold,
        )
        self.transitions = []
        self._directive = None
        handlers: dict[str, Callable[[QuestionState], tuple[str, str]]] = {
            "INIT": self._init,
            "PLAN": self._plan,
            "EXECUTE": self._execute,
            "AGGREGATE": self._aggregate,
            "SYNTHESIZE": self._synthesize,
            "VERIFY": self._verify,
            "REPLAN_GATE": self._replan_gate,
            "FINALIZE": self._finalize,
        }
        steps = 0
        while state.state != "DONE" and steps < _MAX_MACHINE_STEPS:
            steps += 1
            current = state.state
            try:
                nxt, transition = handlers[current](state)
            except Exception as exc:  # noqa: BLE001 - T14: nothing escapes the machine
                state.errors.append(f"orchestrator[{current}]: {type(exc).__name__}: {exc}")
                if current == "FINALIZE":  # a broken FINALIZE must still terminate
                    self._note(state, "T14", current, "DONE")
                    state.terminated_by = state.terminated_by or "error"
                    state.state = "DONE"
                    break
                state.terminated_by = "error"
                nxt, transition = "FINALIZE", "T14"
            self._note(state, transition, current, nxt)
            state.state = nxt
        if state.state != "DONE":  # only reachable via the outer stop
            state.terminated_by = state.terminated_by or "machine_stalled"
            state.state = "DONE"
        return state

    def _note(self, state: QuestionState, transition: str, src: str, dst: str) -> None:
        self.transitions.append(f"{transition}:{src}->{dst}")

    # -- INIT --------------------------------------------------------------
    def _init(self, state: QuestionState) -> tuple[str, str]:
        state.budget = Budget.from_config(self.cfg).start()
        return "PLAN", "T1"

    # -- PLAN --------------------------------------------------------------
    def _plan(self, state: QuestionState) -> tuple[str, str]:
        directive = self._directive
        self._directive = None
        # Set the cycle before the call so the Planner's own step is attributed
        # to the revision it is producing, not to the one that failed.
        state.cycle = directive.revision if directive is not None else 0
        plan = self.planner.run(state, directive=directive)
        if plan is None:  # defensive: the Planner's own floor should prevent this
            plan = fallback_plan(state.question, revision=len(state.plans), reason="planner_none")
        if plan.revision > 0 and state.is_duplicate_plan(plan):
            state.terminated_by = "degenerate_replan"
            state.errors.append(f"discarded near-duplicate re-plan (revision {plan.revision})")
            return "FINALIZE", "T2b"
        state.plans.append(plan)
        state.cycle = plan.revision
        state.budget.iterations += 1
        return "EXECUTE", "T2"

    # -- EXECUTE -----------------------------------------------------------
    def _execute(self, state: QuestionState) -> tuple[str, str]:
        """Level-synchronous Kahn order, ascending numeric id, strictly sequential.

        No threads: a single 8 GB card is already the bottleneck, and sequential
        execution makes the trace a total order, which is what makes the
        qualitative error analysis readable.
        """
        plan = state.plan
        if plan is None:
            return "AGGREGATE", "T4"
        executed: set[str] = set()
        for level in plan.topo_order():
            for subquery in level:
                if state.budget.wallclock_exceeded():
                    state.terminated_by = state.terminated_by or "budget_wallclock"
                    break
                # Literally the T4 guard: the non-privileged budget gates the
                # whole EXECUTE phase, not only the nodes that would spend it.
                if state.budget.remaining_llm(privileged=False) <= 0:
                    state.terminated_by = state.terminated_by or "budget_llm"
                    break
                self._run_node(state, plan, subquery)
                executed.add(subquery.id)
                self._note(state, "T3", "EXECUTE", "EXECUTE")
            else:
                continue
            break
        skipped = [sq.id for sq in plan.subqueries if sq.id not in executed]
        if skipped:
            state.errors.append(f"skipped nodes: {', '.join(skipped)}")
        return "AGGREGATE", "T4"

    def _run_node(self, state: QuestionState, plan: Plan, subquery: SubQuery) -> None:
        """RESOLVE -> ROUTE -> RETRIEVE -> KG -> EXTRACT for one node."""
        if subquery.is_combiner:
            with state.step("orchestrator", subquery_id=subquery.id, stage="combiner") as rec:
                rec.output_summary = {"skipped": "combiner", "text": subquery.text}
            return

        text, degraded, reason = self.resolve_placeholders(state, subquery)
        result = self.retriever.run(
            state, subquery, query_text=text, degraded=degraded, degraded_reason=reason
        )
        state.results[subquery.id] = result

        if self.kg_enabled:
            kg_result = self._invoke_optional(
                "kg", state, subquery=subquery, result=result, query_text=text
            )
            if kg_result is not None:
                state.kg_results[subquery.id] = kg_result

        if _has_dependents(plan, subquery.id):
            self._extract_answer(state, subquery, result)

    # -- placeholder resolution (section 4.3) ------------------------------
    def resolve_placeholders(
        self, state: QuestionState, subquery: SubQuery
    ) -> tuple[str, bool, str]:
        """Fill ``{{qN.field}}`` by the four-rung ladder. Never aborts the node.

        Rung 4 deletes the token and collapses the whitespace, leaving a
        residual natural-language query ("When was born?"). Ugly, retrievable,
        and strictly better than dropping a node that would still have produced
        evidence.
        """
        text = subquery.text
        degraded = False
        reasons: list[str] = []
        for qid, field in subquery.placeholders():
            value, rung = self._placeholder_value(state, qid, field)
            if not value:
                text = strip_placeholder(text, qid, field)
                degraded = True
                reasons.append(f"unresolved_placeholder:{qid}.{field}")
                continue
            if rung > 1:
                degraded = True
                reasons.append(f"placeholder_rung{rung}:{qid}.{field}")
            text = collapse_whitespace(text.replace(f"{{{{{qid}.{field}}}}}", value))
        return text, degraded, "; ".join(reasons)

    def _placeholder_value(
        self, state: QuestionState, qid: str, field: str
    ) -> tuple[str, int]:
        """``(value, rung)``; rung 1 is the primary source for that field."""
        answer = (state.answers.get(qid) or "").strip()
        entity = (state.bridge_entities.get(qid) or "").strip()
        title = _top_title(state.results.get(qid))
        if field == "answer":
            order = ((answer, 1), (entity, 2), (title, 3))
        elif field == "entity":
            order = ((entity, 1), (answer, 2), (title, 3))
        else:  # title
            order = ((title, 1), (answer, 2), (entity, 3))
        for value, rung in order:
            if value:
                return value, rung
        return "", 4

    # -- answer extraction (section 4.4) -----------------------------------
    def _extract_answer(
        self, state: QuestionState, subquery: SubQuery, result: RetrievalResult
    ) -> None:
        """Rungs 1-5. Runs only for nodes something else depends on."""
        with state.step("extractor", subquery_id=subquery.id) as rec:
            kg_result = state.kg_results.get(subquery.id)
            bridge = getattr(kg_result, "bridge_entity", None) if kg_result else None
            answer, rung, is_bridge = "", 0, False

            if bridge:
                answer, rung, is_bridge = str(bridge), 1, True
            elif subquery.answer_type == "entity" and result.passages:
                answer, rung, is_bridge = result.passages[0].passage.title, 2, True
            elif subquery.answer_type in ("date", "number"):
                answer = _regex_span(result.passages, subquery.answer_type)
                rung = 3 if answer else 0

            if not answer:
                answer = self.extractor.run(
                    state,
                    rec,
                    question=result.query_text,
                    passages=result.passages,
                    answer_type=subquery.answer_type,
                )
                rung = 4 if answer else rung

            if not answer and result.passages:
                answer, rung, is_bridge = result.passages[0].passage.title, 5, True
                rec.degrade("extraction_fell_through")

            if answer:
                state.answers[subquery.id] = answer
                if is_bridge:
                    state.bridge_entities[subquery.id] = answer
            rec.output_summary = {"rung": rung, "answer": answer, "bridge": is_bridge}

    # -- AGGREGATE ---------------------------------------------------------
    def _aggregate(self, state: QuestionState) -> tuple[str, str]:
        with state.step("orchestrator", stage="aggregate") as rec:
            state.evidence = self.build_evidence(state)
            rec.output_summary = {
                "n_evidence": len(state.evidence),
                "n_results": len(state.results),
            }
        return "SYNTHESIZE", "T5"

    def build_evidence(self, state: QuestionState) -> dict[str, Evidence]:
        """Sentence-granular evidence, deduplicated, ranked, numbered e1..eN.

        Sentence granularity matches the supporting-fact ground truth, keeps NLI
        premises short enough for DeBERTa to score well, and keeps the
        synthesiser prompt inside ``num_ctx``. Ranking is rank-reciprocal with a
        small lexical-overlap bonus: purely deterministic, with an explicit
        tiebreaker, so two runs agree exactly.
        """
        pooled: dict[tuple[str, int], dict[str, Any]] = {}
        for sq_id in sorted(state.results, key=_numeric_id):
            result = state.results[sq_id]
            query_terms = normalise_text(result.query_text)
            for rank, scored in enumerate(result.passages[: self.evidence_passages]):
                passage = scored.passage
                for sent_id, sentence in enumerate(passage.sentences):
                    cleaned = collapse_whitespace(sentence)
                    if len(cleaned) < 3:
                        continue
                    score = 1.0 / (self.rrf_k + rank + 1)
                    score += 0.05 * jaccard(normalise_text(cleaned), query_terms)
                    entry = pooled.setdefault(
                        (passage.doc_id, sent_id),
                        {
                            "text": cleaned, "score": 0.0, "subqueries": [],
                            "title": passage.title, "provenance": scored.provenance,
                        },
                    )
                    entry["score"] = max(entry["score"], score)
                    if sq_id not in entry["subqueries"]:
                        entry["subqueries"].append(sq_id)

        ordered = sorted(
            pooled.items(), key=lambda item: (-item[1]["score"], item[0][0], item[0][1])
        )
        evidence: dict[str, Evidence] = {}
        for index, ((doc_id, sent_id), entry) in enumerate(ordered[: self.max_evidence], start=1):
            eid = f"e{index}"
            evidence[eid] = Evidence(
                evidence_id=eid,
                kind="passage",
                text=entry["text"],
                score=round(float(entry["score"]), 6),
                subquery_ids=tuple(entry["subqueries"]),
                provenance=entry["provenance"],
                doc_id=doc_id,
                title=entry["title"],
                sent_id=sent_id,
            )
        for extra in self._kg_evidence(state, limit=max(0, self.max_evidence - len(evidence))):
            eid = f"e{len(evidence) + 1}"
            evidence[eid] = replace(extra, evidence_id=eid)
        return evidence

    def _kg_evidence(self, state: QuestionState, *, limit: int) -> list[Evidence]:
        """Citable KG edge sentences, if the navigator produced any."""
        if limit <= 0:
            return []
        out: list[Evidence] = []
        seen: set[str] = set()
        for sq_id in sorted(state.kg_results, key=_numeric_id):
            for item in getattr(state.kg_results[sq_id], "evidence", ()) or ():
                text = collapse_whitespace(getattr(item, "text", ""))
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(replace(item, text=text, subquery_ids=(sq_id,)))
                if len(out) >= limit:
                    return out
        return out

    # -- SYNTHESIZE --------------------------------------------------------
    def _synthesize(self, state: QuestionState) -> tuple[str, str]:
        cycle = state.plan.revision if state.plan else 0
        candidate = self._invoke_optional(
            "synthesizer",
            state,
            evidence=state.evidence,
            question=state.question,
            cycle=cycle,
        )
        if candidate is None:
            candidate = self._extractive_candidate(state, cycle)
            state.candidates.append(candidate)
            state.terminated_by = state.terminated_by or (
                "synthesizer_unavailable" if self._optional("synthesizer") is None else "synthesizer_failed"
            )
            return "FINALIZE", "T7"
        state.candidates.append(candidate)
        if self.verifier_enabled and self._optional("verifier") is not None:
            return "VERIFY", "T6"
        state.terminated_by = state.terminated_by or (
            "verifier_disabled" if not self.verifier_enabled else "verifier_unavailable"
        )
        return "FINALIZE", "T7"

    def _extractive_candidate(self, state: QuestionState, cycle: int) -> AnswerCandidate:
        """A deterministic answer when no Synthesizer is available.

        Deliberately minimal -- the real extractive ladder is the Synthesizer's
        (section 3.4). This exists so the pipeline still yields an EM/F1
        comparable answer while that module is being written elsewhere.
        """
        if not state.evidence:
            return AnswerCandidate(
                answer="", answer_sentence="", citations=(), cycle=cycle,
                origin="fallback_rule", sufficient=False,
            )
        eid = min(state.evidence, key=_numeric_id)
        top = state.evidence[eid]
        answer = (top.title or " ".join(top.text.split()[:20])).strip()
        return AnswerCandidate(
            answer=answer,
            answer_sentence=top.text,
            citations=(eid,),
            cycle=cycle,
            origin="fallback_rule",
            sufficient=False,
        )

    # -- VERIFY ------------------------------------------------------------
    def _verify(self, state: QuestionState) -> tuple[str, str]:
        candidate = state.candidates[-1]
        verification = self._invoke_optional(
            "verifier", state, candidate=candidate, evidence=state.evidence
        )
        if verification is None:
            state.terminated_by = state.terminated_by or "verifier_error"
            return "FINALIZE", "T8"
        state.verifications.append(verification)
        state.candidates[-1] = replace(
            candidate, confidence=float(getattr(verification, "confidence", 0.0))
        )
        verdict = getattr(verification, "verdict", "accept")
        if verdict == "accept":
            state.terminated_by = "verified"
            return "FINALIZE", "T8"
        if verdict == "abstain":
            state.terminated_by = "abstained"
            return "FINALIZE", "T10"
        self._directive = self.build_directive(state, verification)
        state.directives.append(self._directive)
        return "REPLAN_GATE", "T9"

    def build_directive(
        self, state: QuestionState, verification: VerificationResult
    ) -> ReplanDirective:
        """The feedback edge: the only object that flows backwards.

        ``failed_subquery_ids`` is unioned with the sub-queries that came back
        empty. Those are a fact of the run rather than an opinion of the
        Verifier, and omitting them would let G5 reject a re-plan that had a
        perfectly concrete thing to fix.
        """
        revision = state.budget.replans + 1
        empty = tuple(
            sq_id for sq_id in sorted(state.results, key=_numeric_id)
            if not state.results[sq_id].passages
        )
        failed = tuple(
            dict.fromkeys([*getattr(verification, "failed_subquery_ids", ()), *empty])
        )
        return ReplanDirective(
            directive_id=f"{state.qid}:r{revision}",
            revision=revision,
            reason=_reason_of(verification),
            confidence=float(getattr(verification, "confidence", 0.0)),
            missing_information=tuple(getattr(verification, "missing_information", ()) or ()),
            failed_subquery_ids=failed,
            suggested_subqueries=tuple(getattr(verification, "suggested_subqueries", ()) or ()),
            covered_entities=state.covered_entities(),
            seen_doc_ids=state.seen_doc_ids(),
            banned_subquery_texts=state.banned_texts(),
            previous_plan_summary=_plan_summary(state.plan),
        )

    # -- REPLAN_GATE -------------------------------------------------------
    def _replan_gate(self, state: QuestionState) -> tuple[str, str]:
        """G1..G5 in order; the first failure names ``terminated_by``."""
        budget = state.budget
        directive = self._directive
        checks: tuple[tuple[bool, str], ...] = (
            (budget.replans < budget.max_replans, "max_replans"),
            (budget.iterations < budget.max_iterations, "budget_iterations"),
            (budget.remaining_llm(privileged=True) >= 3, "budget_llm"),
            (budget.elapsed() < 0.6 * budget.max_wall_clock_s, "budget_wallclock"),
            (directive is not None and directive.is_informative(), "uninformative_feedback"),
        )
        for passed, name in checks:
            if not passed:
                state.terminated_by = name
                self._directive = None
                return "FINALIZE", "T12"
        budget.replans += 1
        return "PLAN", "T11"

    # -- FINALIZE ----------------------------------------------------------
    def _finalize(self, state: QuestionState) -> tuple[str, str]:
        best = state.best_candidate()
        if state.terminated_by is None:
            state.terminated_by = "completed" if best else "no_answer"
        with state.step("orchestrator", stage="finalize") as rec:
            rec.output_summary = {
                "terminated_by": state.terminated_by,
                "best_cycle": best.cycle if best else None,
                "confidence": best.confidence if best else 0.0,
                "n_candidates": len(state.candidates),
            }
        if self.writer is not None:
            self.writer.write_question(state, transitions=self.transitions)
        return "DONE", "T13"

    # -- optional collaborators -------------------------------------------
    def _optional(self, role: str) -> Any:
        """Resolve an injected or lazily-imported collaborator, or None.

        Absence is cached: a missing module must not cost an import attempt per
        sub-query, and a module that raises on import is not going to start
        working later in the same run.
        """
        if self._injected.get(role) is not None:
            return self._injected[role]
        if role in self._resolved:
            return self._resolved[role]
        module_name, class_names = _OPTIONAL_AGENTS[role]
        agent: Any = None
        try:
            module = importlib.import_module(module_name, __package__)
            factory = next(
                (getattr(module, n) for n in class_names if hasattr(module, n)),
                None,
            )
            if factory is not None:
                agent = _instantiate(factory, self.cfg, self.client, self.registry)
        except Exception as exc:  # noqa: BLE001 - a sibling milestone may not exist yet
            agent = None
            self._resolved.setdefault(f"{role}_error", f"{type(exc).__name__}: {exc}")
        self._resolved[role] = agent
        return agent

    def _invoke_optional(self, role: str, state: QuestionState, **kwargs: Any) -> Any:
        """Call ``agent.run`` with whatever subset of ``kwargs`` it accepts.

        The KG, synthesis and verification agents are written against the same
        specification but not by this milestone, so their exact keyword names
        are not knowable here. Filtering by signature makes the seam tolerant
        without making it silent: a failure is recorded and degrades.
        """
        agent = self._optional(role)
        if agent is None:
            return None
        run = getattr(agent, "run", None)
        if not callable(run):
            return None
        try:
            accepted = _acceptable_kwargs(run, kwargs)
            return run(state, **accepted)
        except Exception as exc:  # noqa: BLE001 - axiom 2, at the orchestrator layer
            state.errors.append(f"{role}: {type(exc).__name__}: {exc}")
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instantiate(factory: Any, cfg: Config, client: Any, registry: ToolRegistry) -> Any:
    """Construct a collaborator with the richest signature it will accept."""
    attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
        ((cfg,), {"client": client, "registry": registry}),
        ((cfg,), {"client": client}),
        ((), {"cfg": cfg, "client": client}),
        ((cfg,), {}),
        ((), {"cfg": cfg}),
        ((), {}),
    )
    last: Exception | None = None
    for args, kwargs in attempts:
        try:
            return factory(*args, **kwargs)
        except TypeError as exc:
            last = exc
    if last is not None:
        raise last
    return factory()


def _acceptable_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keywords the callee does not declare (unless it takes ``**kwargs``)."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def _numeric_id(identifier: str) -> tuple[int, str]:
    digits = identifier[1:]
    return (int(digits), identifier) if digits.isdigit() else (10**6, identifier)


def _has_dependents(plan: Plan, sq_id: str) -> bool:
    """Section 4.4: extracting an answer nobody consumes is pure waste."""
    return any(sq_id in other.depends_on for other in plan.subqueries)


def _top_title(result: RetrievalResult | None) -> str:
    if result is None or not result.passages:
        return ""
    return result.passages[0].passage.title.strip()


def _regex_span(passages: Sequence[ScoredPassage], answer_type: str) -> str:
    """Rung 3: first date/number match over the top-3 passages' sentences."""
    pattern = _DATE_RE if answer_type == "date" else _NUMBER_RE
    for scored in passages[:3]:
        for sentence in scored.passage.sentences:
            match = pattern.search(sentence)
            if match:
                return match.group(0).strip()
    if answer_type == "date":
        for scored in passages[:3]:
            match = _YEAR_RE.search(scored.passage.text)
            if match:
                return match.group(0)
    return ""


def _plan_summary(plan: Plan | None) -> str:
    """Compact ``q1: ... -> q2: ...`` rendering for the re-plan prompt."""
    if plan is None:
        return ""
    return " -> ".join(f"{sq.id}: {sq.text}" for sq in plan.subqueries)


def _reason_of(verification: VerificationResult) -> ReplanReason:
    reason = getattr(verification, "reason", None)
    valid = (
        "low_confidence", "missing_evidence", "contradiction",
        "no_citations", "empty_retrieval", "synthesizer_insufficient",
    )
    return reason if reason in valid else "low_confidence"  # type: ignore[return-value]
