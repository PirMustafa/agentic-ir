"""Orchestrator state-machine tests.

The machine is the contribution, so the assertions are about *paths*, not just
outcomes: which transition ids fired, which guard rejected a re-plan, whether a
degenerate re-plan was discarded before it cost anything, and whether FINALIZE
really took the best cycle rather than the last one.

Everything runs offline: a stub retrieval registry, fake collaborator agents,
and no LLM client at all on the happy paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agentic_ir.agents.planner import _finish
from agentic_ir.agents.retriever import RetrievalAgent
from agentic_ir.config import load_config
from agentic_ir.orchestrator import TRANSITIONS, Orchestrator
from agentic_ir.state import Budget, QuestionState
from agentic_ir.tools.registry import ToolRegistry, ToolSpec
from agentic_ir.trace import TraceWriter, build_trace_record
from agentic_ir.types import (
    AnswerCandidate,
    Evidence,
    Passage,
    Plan,
    ReplanDirective,
    ScoredPassage,
    SubQuery,
    VerificationResult,
)
from tests.test_planner import COMPARISON_Q, StubBM25, StubClient

CFG = load_config()


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------

def plan_of(*nodes: SubQuery, question: str = COMPARISON_Q, revision: int = 0) -> Plan:
    return _finish(question, list(nodes), revision=revision, origin="llm")


class FakePlanner:
    """Emits a queued list of plans; the last one repeats if asked again."""

    name = "planner"

    def __init__(self, *plans: Plan) -> None:
        self.plans = list(plans)
        self.directives: list[ReplanDirective | None] = []

    def run(self, state: QuestionState, *, directive: ReplanDirective | None = None) -> Plan:
        self.directives.append(directive)
        plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        with state.step("planner", revision=plan.revision):
            pass
        return plan


class RaisingPlanner:
    name = "planner"

    def run(self, state: QuestionState, *, directive: Any = None) -> Plan:
        raise RuntimeError("planner exploded")


class FakeSynthesizer:
    name = "synthesizer"

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers) or ["an answer"]
        self.calls = 0

    def run(
        self,
        state: QuestionState,
        *,
        evidence: dict[str, Evidence],
        question: str,
        cycle: int,
    ) -> AnswerCandidate:
        self.calls += 1
        answer = self.answers[min(cycle, len(self.answers) - 1)]
        return AnswerCandidate(
            answer=answer,
            answer_sentence=f"{answer} is the answer.",
            citations=tuple(sorted(evidence)[:1]),
            cycle=cycle,
            origin="llm",
        )


class FakeVerifier:
    name = "verifier"

    def __init__(self, *outcomes: tuple[str, float]) -> None:
        self.outcomes = list(outcomes) or [("accept", 0.9)]
        self.calls = 0

    def run(
        self,
        state: QuestionState,
        *,
        candidate: AnswerCandidate,
        evidence: dict[str, Evidence],
    ) -> VerificationResult:
        verdict, confidence = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        return VerificationResult(
            verdict=verdict,  # type: ignore[arg-type]
            candidate=candidate,
            confidence=confidence,
            missing_information=("the founding year of First for Women",),
            reason="low_confidence",
        )


def registry_of(*passages: ScoredPassage, record: list[str] | None = None) -> ToolRegistry:
    def search(query: str, top_k: int | None = None) -> list[ScoredPassage]:
        if record is not None:
            record.append(query)
        return list(passages)

    return ToolRegistry(
        [
            ToolSpec(name=name, fn=search, arg_schema={}, description=name)
            for name in ("bm25_search", "dense_search", "hybrid_search")
        ]
    )


def orchestrator(
    *,
    planner: Any,
    registry: ToolRegistry | None = None,
    synthesizer: Any = None,
    verifier: Any = None,
    cfg: Any = None,
    writer: TraceWriter | None = None,
) -> Orchestrator:
    cfg = cfg or CFG
    registry = registry if registry is not None else registry_of(_hit("Arthur's Magazine"))
    client = StubClient()
    return Orchestrator(
        registry,
        cfg=cfg,
        planner=planner,
        retriever=RetrievalAgent(registry, cfg, bm25=StubBM25(), client=client),
        synthesizer=synthesizer,
        verifier=verifier,
        writer=writer,
        client=client,
        config_name="test_config",
    )


def _hit(title: str, rank: int = 0) -> ScoredPassage:
    return ScoredPassage(
        passage=Passage(
            doc_id=f"hotpotqa:{title.replace(' ', '_')}",
            title=title,
            text=f"{title} was an American periodical started in 1844.",
            sentences=(
                f"{title} was an American periodical started in 1844.",
                f"{title} ceased publication in 1846.",
            ),
            source="hotpotqa",
        ),
        score=9.0 - rank,
        rank=rank,
        provenance="hybrid",
    )


def ids(transitions: Sequence[str]) -> list[str]:
    return [t.split(":", 1)[0] for t in transitions]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_full_path_reaches_done_and_records_every_transition():
    planner = FakePlanner(
        plan_of(
            SubQuery(id="q1", text="When was Arthur's Magazine started?"),
            SubQuery(id="q2", text="When was First for Women started?"),
        )
    )
    orch = orchestrator(
        planner=planner, synthesizer=FakeSynthesizer("Arthur's Magazine"),
        verifier=FakeVerifier(("accept", 0.82)),
    )
    state = orch.run("qid-1", COMPARISON_Q)
    assert state.state == "DONE"
    assert ids(orch.transitions) == ["T1", "T2", "T3", "T3", "T4", "T5", "T6", "T8", "T13"]
    assert state.terminated_by == "verified"
    assert state.best_candidate().answer == "Arthur's Magazine"
    assert state.best_candidate().confidence == pytest.approx(0.82)
    assert set(state.results) == {"q1", "q2"}
    assert state.evidence and min(state.evidence) == "e1"


def test_every_declared_transition_id_is_known_to_the_machine():
    assert set(TRANSITIONS) >= {f"T{i}" for i in range(1, 15)} | {"T2b"}
    for src, dst, _ in TRANSITIONS.values():
        assert src == "*" or src in {s for s in ("INIT", "PLAN", "EXECUTE", "AGGREGATE",
                                                 "SYNTHESIZE", "VERIFY", "REPLAN_GATE",
                                                 "FINALIZE")}
        assert dst in {"PLAN", "EXECUTE", "AGGREGATE", "SYNTHESIZE", "VERIFY",
                       "REPLAN_GATE", "FINALIZE", "DONE"}


def test_pipeline_finalizes_gracefully_without_a_synthesizer():
    """The sibling milestone may not have landed; the run must still answer."""
    planner = FakePlanner(plan_of(SubQuery(id="q1", text="When was Arthur's Magazine started?")))
    orch = orchestrator(planner=planner)
    orch._resolved["synthesizer"] = None  # simulate the module not existing yet
    orch._resolved["verifier"] = None
    state = orch.run("qid-2", COMPARISON_Q)
    assert state.state == "DONE"
    assert "T7" in ids(orch.transitions)
    assert state.terminated_by == "synthesizer_unavailable"
    assert state.candidates and state.candidates[0].origin == "fallback_rule"
    assert state.candidates[0].answer  # EM/F1 stays computable


def test_verifier_disabled_takes_t7():
    cfg = CFG.with_overrides({"agents.verifier.enabled": False})
    planner = FakePlanner(plan_of(SubQuery(id="q1", text="When was Arthur's Magazine started?")))
    orch = orchestrator(planner=planner, synthesizer=FakeSynthesizer("x"), cfg=cfg)
    state = orch.run("qid-3", COMPARISON_Q)
    assert state.terminated_by == "verifier_disabled"
    assert "T7" in ids(orch.transitions)


def test_abstain_takes_t10():
    planner = FakePlanner(plan_of(SubQuery(id="q1", text="When was Arthur's Magazine started?")))
    orch = orchestrator(
        planner=planner, synthesizer=FakeSynthesizer("x"),
        verifier=FakeVerifier(("abstain", 0.2)),
    )
    state = orch.run("qid-4", COMPARISON_Q)
    assert state.terminated_by == "abstained"
    assert "T10" in ids(orch.transitions)
    assert state.best_candidate() is not None  # abstention is a label, not a refusal


def test_unhandled_exception_takes_t14_and_still_finishes():
    orch = orchestrator(planner=RaisingPlanner())
    state = orch.run("qid-5", COMPARISON_Q)
    assert state.state == "DONE"
    assert "T14" in ids(orch.transitions)
    assert state.terminated_by == "error"
    assert any("planner exploded" in e for e in state.errors)


# ---------------------------------------------------------------------------
# The feedback loop
# ---------------------------------------------------------------------------

def _two_cycle_planner() -> FakePlanner:
    return FakePlanner(
        plan_of(SubQuery(id="q1", text="When was Arthur's Magazine started?")),
        plan_of(
            SubQuery(id="q1", text="What year did First for Women begin publication?"),
            revision=1,
        ),
    )


def test_revise_replans_and_finalize_keeps_the_better_cycle():
    planner = _two_cycle_planner()
    orch = orchestrator(
        planner=planner,
        synthesizer=FakeSynthesizer("first answer", "second answer"),
        verifier=FakeVerifier(("revise", 0.30), ("accept", 0.71)),
    )
    state = orch.run("qid-6", COMPARISON_Q)
    assert ids(orch.transitions).count("T9") == 1
    assert "T11" in ids(orch.transitions)
    assert state.budget.replans == 1
    assert len(state.plans) == 2
    assert state.best_candidate().answer == "second answer"
    assert state.metrics()["replanned"] is True


def test_finalize_prefers_the_earlier_cycle_when_the_replan_is_worse():
    planner = _two_cycle_planner()
    orch = orchestrator(
        planner=planner,
        synthesizer=FakeSynthesizer("first answer", "second answer"),
        verifier=FakeVerifier(("revise", 0.54), ("abstain", 0.10)),
    )
    state = orch.run("qid-7", COMPARISON_Q)
    best = state.best_candidate()
    assert best.answer == "first answer"
    assert best.cycle == 0  # a bad re-plan must not overwrite a better answer


def test_directive_carries_the_ban_list_and_the_previous_plan():
    planner = _two_cycle_planner()
    orch = orchestrator(
        planner=planner,
        synthesizer=FakeSynthesizer("a", "b"),
        verifier=FakeVerifier(("revise", 0.3), ("accept", 0.8)),
    )
    state = orch.run("qid-8", COMPARISON_Q)
    directive = state.directives[0]
    assert directive.directive_id == "qid-8:r1"
    assert "When was Arthur's Magazine started?" in directive.banned_subquery_texts
    assert directive.previous_plan_summary.startswith("q1: ")
    assert directive.missing_information
    assert planner.directives[1] is directive


def test_near_duplicate_replan_is_discarded_by_t2b():
    same = SubQuery(id="q1", text="When was Arthur's Magazine started?")
    planner = FakePlanner(
        plan_of(same),
        plan_of(SubQuery(id="q1", text="When was Arthur's Magazine started"), revision=1),
    )
    orch = orchestrator(
        planner=planner,
        synthesizer=FakeSynthesizer("a", "b"),
        verifier=FakeVerifier(("revise", 0.3), ("accept", 0.9)),
    )
    state = orch.run("qid-9", COMPARISON_Q)
    assert "T2b" in ids(orch.transitions)
    assert state.terminated_by == "degenerate_replan"
    assert len(state.plans) == 1  # the duplicate was never appended


# ---------------------------------------------------------------------------
# Re-plan guards, evaluated in order (G1..G5)
# ---------------------------------------------------------------------------

def _gate_state(**budget: Any) -> QuestionState:
    defaults = {
        "max_llm_calls": 20, "max_iterations": 6, "max_wall_clock_s": 300.0,
        "max_replans": 2, "reserve_llm_calls": 2,
    }
    defaults.update(budget)
    return QuestionState(
        qid="g", question=COMPARISON_Q, dataset="hotpotqa", config_name="test",
        budget=Budget(**defaults).start(),
    )


def _gate(state: QuestionState, directive: ReplanDirective | None) -> tuple[str, str]:
    orch = orchestrator(planner=FakePlanner(plan_of(SubQuery(id="q1", text="anything at all"))))
    orch._directive = directive
    return orch._replan_gate(state)


def _informative(**kwargs: Any) -> ReplanDirective:
    payload: dict[str, Any] = {
        "directive_id": "g:r1", "revision": 1, "reason": "low_confidence", "confidence": 0.3,
        "missing_information": ("something",),
    }
    payload.update(kwargs)
    return ReplanDirective(**payload)


def test_g1_blocks_when_replans_are_spent():
    state = _gate_state()
    state.budget.replans = 2
    assert _gate(state, _informative()) == ("FINALIZE", "T12")
    assert state.terminated_by == "max_replans"


def test_g2_blocks_when_iterations_are_spent():
    state = _gate_state()
    state.budget.iterations = 6
    assert _gate(state, _informative())[1] == "T12"
    assert state.terminated_by == "budget_iterations"


def test_g3_blocks_when_fewer_than_three_privileged_calls_remain():
    state = _gate_state()
    state.budget.llm_calls = 18  # 2 left, and a cycle needs plan+synth+verify
    assert _gate(state, _informative())[1] == "T12"
    assert state.terminated_by == "budget_llm"


def test_g4_blocks_past_sixty_percent_of_the_wall_clock():
    from time import perf_counter

    state = _gate_state()
    state.budget.t0 = perf_counter() - 200.0  # 200s of a 300s budget: past the 0.6 fraction
    assert _gate(state, _informative())[1] == "T12"
    assert state.terminated_by == "budget_wallclock"


def test_g5_blocks_an_uninformative_directive():
    state = _gate_state()
    empty = ReplanDirective(
        directive_id="g:r1", revision=1, reason="low_confidence", confidence=0.3
    )
    assert _gate(state, empty)[1] == "T12"
    assert state.terminated_by == "uninformative_feedback"


def test_guards_are_evaluated_in_order_and_the_first_failure_names_the_reason():
    from time import perf_counter

    state = _gate_state()
    state.budget.t0 = perf_counter() - 200.0  # G4 would fail
    state.budget.replans = 2  # so would G1, and G1 is checked first
    assert _gate(state, None)[1] == "T12"
    assert state.terminated_by == "max_replans"


def test_all_guards_passing_returns_to_plan():
    state = _gate_state()
    assert _gate(state, _informative()) == ("PLAN", "T11")
    assert state.budget.replans == 1


# ---------------------------------------------------------------------------
# Placeholder resolution ladder (section 4.3)
# ---------------------------------------------------------------------------

def _resolver() -> Orchestrator:
    return orchestrator(planner=FakePlanner(plan_of(SubQuery(id="q1", text="anything at all"))))


def _resolvable_state() -> QuestionState:
    return QuestionState(qid="p", question="q", dataset="hotpotqa", config_name="test")


def test_rung1_primary_source():
    state = _resolvable_state()
    state.answers["q1"] = "James Cameron"
    node = SubQuery(id="q2", text="When was {{q1.answer}} born?", depends_on=("q1",))
    text, degraded, reason = _resolver().resolve_placeholders(state, node)
    assert text == "When was James Cameron born?"
    assert not degraded and reason == ""


def test_rung2_secondary_source_degrades():
    state = _resolvable_state()
    state.bridge_entities["q1"] = "James Cameron"
    node = SubQuery(id="q2", text="When was {{q1.answer}} born?", depends_on=("q1",))
    text, degraded, reason = _resolver().resolve_placeholders(state, node)
    assert text == "When was James Cameron born?"
    assert degraded and "placeholder_rung2" in reason


def test_rung3_dependency_title_degrades():
    state = _resolvable_state()
    agent = RetrievalAgent(registry_of(_hit("James Cameron")), CFG, bm25=StubBM25(),
                           client=StubClient())
    agent.run(state, SubQuery(id="q1", text="Who directed Titanic?"))
    node = SubQuery(id="q2", text="When was {{q1.answer}} born?", depends_on=("q1",))
    text, degraded, reason = _resolver().resolve_placeholders(state, node)
    assert text == "When was James Cameron born?"
    assert degraded and "placeholder_rung3" in reason


def test_rung4_deletes_the_placeholder_and_collapses_whitespace():
    state = _resolvable_state()
    node = SubQuery(id="q2", text="When was {{q1.answer}} born?", depends_on=("q1",))
    text, degraded, reason = _resolver().resolve_placeholders(state, node)
    assert text == "When was born?"  # ugly but retrievable; the node is never aborted
    assert degraded and "unresolved_placeholder:q1.answer" in reason


def test_title_field_prefers_the_retrieved_title():
    state = _resolvable_state()
    state.answers["q1"] = "an answer"
    agent = RetrievalAgent(registry_of(_hit("Arthur's Magazine")), CFG, bm25=StubBM25(),
                           client=StubClient())
    agent.run(state, SubQuery(id="q1", text="Arthur's Magazine"))
    node = SubQuery(id="q2", text="{{q1.title}} publication history", depends_on=("q1",))
    text, degraded, _ = _resolver().resolve_placeholders(state, node)
    assert text == "Arthur's Magazine publication history"
    assert not degraded


# ---------------------------------------------------------------------------
# DAG execution, combiners and extraction
# ---------------------------------------------------------------------------

def test_combiner_node_skips_retrieval_entirely():
    issued: list[str] = []
    nodes = [
        SubQuery(id="q1", text="When was Arthur's Magazine started?"),
        SubQuery(id="q2", text="When was First for Women started?"),
        SubQuery(id="q3", text=COMPARISON_Q, depends_on=("q1", "q2")),
    ]
    plan = _finish(COMPARISON_Q, nodes, revision=0, origin="llm")
    assert plan.subqueries[2].is_combiner
    orch = orchestrator(
        planner=FakePlanner(plan),
        registry=registry_of(_hit("Arthur's Magazine"), record=issued),
        synthesizer=FakeSynthesizer("Arthur's Magazine"),
        verifier=FakeVerifier(("accept", 0.9)),
    )
    state = orch.run("qid-10", COMPARISON_Q)
    assert set(state.results) == {"q1", "q2"}  # nothing retrieved for the combiner
    assert COMPARISON_Q not in issued


def test_dependent_node_is_executed_after_its_dependency_and_gets_the_answer():
    plan = _finish(
        "Who directed Titanic and when was that person born?",
        [
            SubQuery(id="q1", text="Who directed Titanic?", answer_type="entity"),
            SubQuery(id="q2", text="When was {{q1.answer}} born?", depends_on=("q1",)),
        ],
        revision=0,
        origin="llm",
    )
    issued: list[str] = []
    orch = orchestrator(
        planner=FakePlanner(plan),
        registry=registry_of(_hit("James Cameron"), record=issued),
        synthesizer=FakeSynthesizer("1954"),
        verifier=FakeVerifier(("accept", 0.9)),
    )
    state = orch.run("qid-11", "Who directed Titanic and when was that person born?")
    # Rung 2 of the extraction ladder: the top title, with no LLM call.
    assert state.answers["q1"] == "James Cameron"
    assert state.bridge_entities["q1"] == "James Cameron"
    assert "When was James Cameron born?" in issued
    assert state.budget.llm_calls == 0


def test_leaf_nodes_do_not_run_extraction():
    """Section 4.4: extracting an answer nobody consumes is pure waste."""
    plan = _finish(
        COMPARISON_Q,
        [SubQuery(id="q1", text="When was Arthur's Magazine started?", answer_type="entity")],
        revision=0,
        origin="llm",
    )
    orch = orchestrator(planner=FakePlanner(plan), synthesizer=FakeSynthesizer("x"),
                        verifier=FakeVerifier(("accept", 0.9)))
    state = orch.run("qid-12", COMPARISON_Q)
    assert state.answers == {}
    assert not any(t.agent == "extractor" for t in state.traces)


def test_evidence_is_sentence_granular_deduplicated_and_numbered():
    plan = _finish(
        COMPARISON_Q,
        [
            SubQuery(id="q1", text="When was Arthur's Magazine started?"),
            SubQuery(id="q2", text="When was Arthur's Magazine founded?"),
        ],
        revision=0,
        origin="llm",
    )
    orch = orchestrator(
        planner=FakePlanner(plan),
        registry=registry_of(_hit("Arthur's Magazine")),
        synthesizer=FakeSynthesizer("x"),
        verifier=FakeVerifier(("accept", 0.9)),
    )
    state = orch.run("qid-13", COMPARISON_Q)
    # One passage with two sentences, surfaced by both sub-queries: two pieces
    # of evidence, each crediting both sub-queries.
    assert sorted(state.evidence) == ["e1", "e2"]
    assert all(e.kind == "passage" for e in state.evidence.values())
    assert set(state.evidence["e1"].subquery_ids) == {"q1", "q2"}
    assert state.evidence["e1"].sent_id in (0, 1)


def test_evidence_is_capped_at_max_evidence():
    cfg = CFG.with_overrides({"agents.verifier.max_evidence": 3})
    hits = [_hit(f"Title {i}", rank=i) for i in range(5)]
    plan = _finish(COMPARISON_Q, [SubQuery(id="q1", text="anything at all")], revision=0,
                   origin="llm")
    orch = orchestrator(
        planner=FakePlanner(plan), registry=registry_of(*hits), cfg=cfg,
        synthesizer=FakeSynthesizer("x"), verifier=FakeVerifier(("accept", 0.9)),
    )
    state = orch.run("qid-14", COMPARISON_Q)
    assert len(state.evidence) == 3


# ---------------------------------------------------------------------------
# Trace artefacts (section 6)
# ---------------------------------------------------------------------------

def test_trace_writer_emits_jsonl_metrics_and_meta(tmp_path):
    writer = TraceWriter.create(
        config_name="test_config", dataset="hotpotqa", cfg=CFG, root=tmp_path,
        run_id="test_config_hotpotqa_20260101T0000Z",
    )
    writer.write_meta(CFG, cache_cold=True)
    planner = FakePlanner(
        plan_of(
            SubQuery(id="q1", text="When was Arthur's Magazine started?"),
            SubQuery(id="q2", text="When was First for Women started?"),
        )
    )
    orch = orchestrator(
        planner=planner, synthesizer=FakeSynthesizer("Arthur's Magazine"),
        verifier=FakeVerifier(("accept", 0.82)), writer=writer,
    )
    orch.run("qid-trace", COMPARISON_Q, gold={"answer": "Arthur's Magazine"})

    records = writer.read_records()
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "1.0"
    assert record["qid"] == "qid-trace"
    assert record["final_answer"] == "Arthur's Magazine"
    assert record["verdict"] == "accept"
    assert record["terminated_by"] == "verified"
    assert record["gold"] == {"answer": "Arthur's Magazine"}
    assert record["metrics"]["n_subqueries"] == 2
    assert record["metrics"]["llm_calls_saved"] == 2  # one per routed sub-query
    assert set(record["retrieved"]) == {"q1", "q2"}
    assert record["retrieved"]["q1"]["rule_id"]
    assert record["transitions"][0].startswith("T1:")
    assert record["evidence"][0]["evidence_id"] == "e1"

    assert writer.existing_qids() == {"qid-trace"}
    assert writer.meta_path.exists()
    header = writer.metrics_path.read_text(encoding="utf-8").splitlines()[0]
    assert "qid" in header and "terminated_by" in header


def test_trace_record_is_json_safe_without_a_writer():
    state = QuestionState(qid="x", question="q", dataset="hotpotqa", config_name="c")
    state.terminated_by = "no_answer"
    record = build_trace_record(state, run_id="r", model="qwen3:8b")
    import orjson

    assert orjson.loads(orjson.dumps(record))["run_id"] == "r"


def test_resume_skips_questions_already_traced(tmp_path):
    writer = TraceWriter(tmp_path / "run", run_id="run")
    writer.append({"qid": "a", "metrics": {}})
    writer.append({"qid": "b", "metrics": {}})
    assert TraceWriter(tmp_path / "run", run_id="run").existing_qids() == {"a", "b"}


def test_torn_final_line_does_not_break_resume(tmp_path):
    writer = TraceWriter(tmp_path / "run", run_id="run")
    writer.append({"qid": "a", "metrics": {}})
    with writer.traces_path.open("a", encoding="utf-8") as fh:
        fh.write('{"qid": "b", "metr')
    assert writer.existing_qids() == {"a"}


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------

def test_heuristic_routing_makes_the_whole_run_cost_zero_llm_calls():
    """With the shipped config the only LLM spend is planning and answering,
    both of which are stubbed here -- so routing really is free."""
    planner = FakePlanner(
        plan_of(
            SubQuery(id="q1", text="When was Arthur's Magazine started?"),
            SubQuery(id="q2", text="When was First for Women started?"),
        )
    )
    orch = orchestrator(
        planner=planner, synthesizer=FakeSynthesizer("x"), verifier=FakeVerifier(("accept", 0.9))
    )
    state = orch.run("qid-15", COMPARISON_Q)
    assert state.budget.llm_calls == 0
    assert state.budget.llm_calls_saved == 2
    assert state.budget.tool_calls >= 2
