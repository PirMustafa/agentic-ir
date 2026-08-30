"""Planner and Retrieval-Agent unit tests.

Everything here runs offline against a stub LLM client, so the suite is a
second-scale feedback loop rather than a minutes-scale one. What is asserted is
the deterministic machinery the report leans on: the repair table, the derived
strategy (the model's own label is measurably unreliable), the three fallback
rungs, and all seven rows of the routing table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agentic_ir.agents.planner import (
    Planner,
    assign_hops,
    comparison_template,
    derive_strategy,
    fallback_plan,
    identity_plan,
    iterative_bridge,
    mark_combiners,
    validate_subqueries,
)
from agentic_ir.agents.retriever import (
    RetrievalAgent,
    RoutingFeatures,
    extract_features,
    route,
)
from agentic_ir.config import load_config
from agentic_ir.state import Budget, QuestionState
from agentic_ir.tools.registry import ToolRegistry, ToolSpec
from agentic_ir.types import Passage, ScoredPassage, SubQuery

COMPARISON_Q = "Which magazine was started first, Arthur's Magazine or First for Women?"
BRIDGE_Q = "Who directed the film Titanic and won an Academy Award for it?"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubResponse:
    def __init__(self, parsed: dict[str, Any] | None, text: str = "") -> None:
        self.parsed = parsed
        self.text = text or str(parsed)
        self.thinking = None
        self.thinking_chars = 0
        self.model = "stub"
        self.retries = 0
        self.latency_s = 0.01
        self.agent = "stub"


class StubClient:
    """Returns queued replies; an Exception in the queue is raised instead.

    Records every call so a test can assert that the heuristic path really did
    issue zero of them.
    """

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def model_for(self, agent: str) -> str:
        return "stub-model"

    def chat(self, messages: Sequence[Any], *, agent: str, **kwargs: Any) -> StubResponse:
        self.calls.append({"agent": agent, "messages": list(messages), **kwargs})
        reply = self.replies.pop(0) if self.replies else None
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            return StubResponse(reply)
        return StubResponse(None, str(reply or ""))


def make_state(question: str = COMPARISON_Q, **budget: Any) -> QuestionState:
    return QuestionState(
        qid="q-test",
        question=question,
        dataset="hotpotqa",
        config_name="test",
        budget=Budget(**budget).start() if budget else Budget().start(),
    )


def passage(title: str, text: str, sentences: Sequence[str] | None = None) -> Passage:
    return Passage(
        doc_id=f"hotpotqa:{title.replace(' ', '_')}",
        title=title,
        text=text,
        sentences=tuple(sentences or [text]),
        source="hotpotqa",
    )


def scored(title: str, text: str, rank: int = 0, score: float = 1.0) -> ScoredPassage:
    return ScoredPassage(
        passage=passage(title, text), score=score, rank=rank, provenance="hybrid"
    )


# ---------------------------------------------------------------------------
# Validation and repair (architecture 3.1)
# ---------------------------------------------------------------------------

def test_bad_ids_are_renumbered_and_dependencies_follow():
    nodes, repairs = validate_subqueries(
        [
            {"id": "step-one", "text": "Who directed Titanic?", "depends_on": []},
            {"id": "step-two", "text": "Where was that person born?", "depends_on": ["step-one"]},
        ]
    )
    assert [n.id for n in nodes] == ["q1", "q2"]
    assert nodes[1].depends_on == ("q1",)
    assert any(r.startswith("renumbered:step-one->q1") for r in repairs)


def test_duplicate_ids_are_renumbered():
    nodes, repairs = validate_subqueries(
        [
            {"id": "q1", "text": "First question here", "depends_on": []},
            {"id": "q1", "text": "Second question here", "depends_on": []},
        ]
    )
    assert [n.id for n in nodes] == ["q1", "q2"]
    assert any(r.startswith("renumbered:") for r in repairs)


def test_dangling_dependency_is_dropped():
    nodes, repairs = validate_subqueries(
        [
            {"id": "q1", "text": "Who founded Acme?", "depends_on": []},
            {"id": "q2", "text": "Where is Acme based?", "depends_on": ["q9"]},
        ]
    )
    assert nodes[1].depends_on == ()
    assert "dropped_dep:q2->q9" in repairs


def test_cycle_is_broken_on_the_numerically_higher_target():
    nodes, repairs = validate_subqueries(
        [
            {"id": "q1", "text": "Who directed Titanic?", "depends_on": ["q2"]},
            {"id": "q2", "text": "Who starred in Titanic?", "depends_on": ["q1"]},
        ]
    )
    assert "cycle_broken:q1->q2" in repairs
    assert nodes[0].depends_on == ()
    assert nodes[1].depends_on == ("q1",)
    assert [n.hop for n in nodes] == [1, 2]


def test_self_dependency_is_dropped():
    nodes, repairs = validate_subqueries(
        [{"id": "q1", "text": "Who directed Titanic?", "depends_on": ["q1"]}]
    )
    assert nodes[0].depends_on == ()
    assert "dropped_dep:q1->q1" in repairs


def test_plan_is_truncated_to_max_subqueries_and_orphan_edges_dropped():
    raw = [
        {"id": f"q{i}", "text": f"Question number {i} about something", "depends_on": []}
        for i in range(1, 8)
    ]
    raw[0]["depends_on"] = ["q7"]
    nodes, repairs = validate_subqueries(raw, max_subqueries=5)
    assert len(nodes) == 5
    assert "truncated_subqueries:2" in repairs
    assert nodes[0].depends_on == ()


def test_placeholder_adds_the_missing_dependency():
    nodes, repairs = validate_subqueries(
        [
            {"id": "q1", "text": "Who directed Titanic?", "depends_on": []},
            {"id": "q2", "text": "When was {{q1.answer}} born?", "depends_on": []},
        ]
    )
    assert nodes[1].depends_on == ("q1",)
    assert "added_dep:q2->q1" in repairs
    assert nodes[1].is_template()


def test_placeholder_for_an_unknown_node_is_stripped_and_whitespace_collapsed():
    nodes, repairs = validate_subqueries(
        [{"id": "q1", "text": "When was {{q4.answer}} born?", "depends_on": []}]
    )
    assert nodes[0].text == "When was born?"
    assert "stripped_placeholder:q1:q4.answer" in repairs


def test_rewrites_are_truncated_and_empty_nodes_dropped():
    nodes, repairs = validate_subqueries(
        [
            {"id": "q1", "text": "  ", "depends_on": []},
            {
                "id": "q2",
                "text": "Who founded Acme?",
                "depends_on": [],
                "rewrites": ["a", "founder of Acme", "Acme founder", "who created Acme"],
            },
        ],
        max_rewrites=2,
    )
    assert len(nodes) == 1
    assert any(r.startswith("dropped_empty:") for r in repairs)
    # "a" is below the 3-character floor and never counted as a rewrite.
    assert nodes[0].rewrites == ("founder of Acme", "Acme founder")
    assert "truncated_rewrites:q1" in repairs or "truncated_rewrites:q2" in repairs


def test_bad_enum_values_fall_back_and_are_recorded():
    nodes, repairs = validate_subqueries(
        [
            {
                "id": "q1",
                "text": "Who founded Acme?",
                "depends_on": [],
                "intent": "wondering",
                "answer_type": "paragraph",
                "tool_hint": "google",
            }
        ]
    )
    assert nodes[0].intent == "lookup"
    assert nodes[0].answer_type == "string"
    assert nodes[0].tool_hint is None
    assert "bad_intent:q1:wondering" in repairs
    assert "bad_answer_type:q1:paragraph" in repairs
    assert "bad_tool_hint:q1:google" in repairs


def test_all_nodes_unusable_returns_empty_so_the_ladder_takes_over():
    nodes, repairs = validate_subqueries([{"id": "q1", "text": ""}, {"nope": 1}])
    assert nodes == []
    assert repairs


# ---------------------------------------------------------------------------
# Derived strategy (the model labels this wrongly 7 times in 8)
# ---------------------------------------------------------------------------

def _nodes(*specs: tuple[str, tuple[str, ...], str, str]) -> list[SubQuery]:
    return assign_hops(
        [
            SubQuery(id=i, text=text, depends_on=deps, intent=intent)  # type: ignore[arg-type]
            for i, deps, text, intent in specs
        ]
    )


def test_strategy_single_hop():
    assert derive_strategy(_nodes(("q1", (), "Who founded Acme?", "lookup"))) == "single_hop"


def test_strategy_bridge_for_a_chain_with_a_placeholder():
    nodes = _nodes(
        ("q1", (), "Who directed Titanic?", "lookup"),
        ("q2", ("q1",), "When was {{q1.answer}} born?", "attribute"),
    )
    # The spec's own worked example is exactly this shape and calls it a bridge,
    # so a value flowing along the edge outranks the dependent node's intent.
    assert derive_strategy(nodes) == "bridge"


def test_strategy_attribute_when_no_value_flows():
    nodes = _nodes(
        ("q1", (), "Who founded Acme?", "lookup"),
        ("q2", ("q1",), "What is the revenue of Acme?", "attribute"),
    )
    assert derive_strategy(nodes) == "attribute"


def test_strategy_comparison_for_two_roots_and_a_combiner():
    nodes = _nodes(
        ("q1", (), "When was Arthur's Magazine started?", "lookup"),
        ("q2", (), "When was First for Women started?", "lookup"),
        ("q3", ("q1", "q2"), "Which magazine was started first?", "comparison"),
    )
    assert derive_strategy(nodes) == "comparison"


def test_strategy_comparison_for_two_independent_roots():
    nodes = _nodes(
        ("q1", (), "When was Arthur's Magazine started?", "lookup"),
        ("q2", (), "When was First for Women started?", "lookup"),
    )
    assert derive_strategy(nodes) == "comparison"


def test_strategy_bridge_comparison_for_two_roots_and_a_deep_chain():
    nodes = _nodes(
        ("q1", (), "Who directed film A?", "lookup"),
        ("q2", (), "Who directed film B?", "lookup"),
        ("q3", ("q1",), "When was {{q1.answer}} born?", "attribute"),
        ("q4", ("q2", "q3"), "Which director is older?", "comparison"),
    )
    assert derive_strategy(nodes) == "bridge_comparison"


def test_llm_strategy_label_is_kept_but_not_trusted():
    client = StubClient(
        {
            "strategy": "single_hop",  # measured: wrong on 7 of 8 questions
            "subqueries": [
                {"id": "q1", "text": "When was Arthur's Magazine started?", "depends_on": []},
                {"id": "q2", "text": "When was First for Women started?", "depends_on": []},
            ],
        }
    )
    state = make_state()
    plan = Planner(load_config(), client=client).run(state)
    assert plan.strategy == "comparison"
    assert plan.strategy_llm == "single_hop"
    assert plan.origin == "llm"


# ---------------------------------------------------------------------------
# Combiner detection
# ---------------------------------------------------------------------------

def test_combiner_node_is_flagged():
    nodes = mark_combiners(
        _nodes(
            ("q1", (), "When was Arthur's Magazine started?", "lookup"),
            ("q2", (), "When was First for Women started?", "lookup"),
            ("q3", ("q1", "q2"), COMPARISON_Q, "comparison"),
        ),
        COMPARISON_Q,
    )
    assert [n.is_combiner for n in nodes] == [False, False, True]


def test_dependent_node_that_does_not_restate_the_question_is_not_a_combiner():
    nodes = mark_combiners(
        _nodes(
            ("q1", (), "Who directed Titanic?", "lookup"),
            ("q2", ("q1",), "What is the capital of Peru?", "lookup"),
        ),
        BRIDGE_Q,
    )
    assert [n.is_combiner for n in nodes] == [False, False]


# ---------------------------------------------------------------------------
# The fallback ladder
# ---------------------------------------------------------------------------

def test_f1_comparison_template_emits_one_node_per_operand():
    nodes = comparison_template(COMPARISON_Q)
    assert nodes is not None
    assert [n.id for n in nodes] == ["q1", "q2"]
    assert nodes[0].text.startswith("Arthur's Magazine")
    assert nodes[1].text.startswith("First for Women")
    assert all(n.depends_on == () for n in nodes)


def test_f1_does_not_fire_on_a_plain_lookup():
    assert comparison_template("Who founded the Acme Corporation?") is None


def test_f2_iterative_bridge_uses_a_title_placeholder():
    nodes = iterative_bridge(BRIDGE_Q)
    assert nodes is not None
    assert nodes[1].depends_on == ("q1",)
    assert "{{q1.title}}" in nodes[1].text


def test_f3_identity_is_the_floor_and_routes_to_hybrid():
    nodes = identity_plan("What?")
    assert len(nodes) == 1
    assert nodes[0].tool_hint == "hybrid_search"


def test_fallback_ladder_walks_f1_then_f2_then_f3():
    assert fallback_plan(COMPARISON_Q).strategy == "comparison"
    assert fallback_plan(BRIDGE_Q).strategy == "bridge"
    floor = fallback_plan("what happened")
    assert floor.strategy == "single_hop"
    assert floor.subqueries[0].tool_hint == "hybrid_search"
    assert floor.origin == "fallback_rule"


def test_parse_failure_falls_back_without_raising():
    from agentic_ir.llm import LLMFormatError

    client = StubClient(
        LLMFormatError("bad", raw="{not json", agent="planner", model="stub", attempts=3)
    )
    state = make_state(BRIDGE_Q)
    plan = Planner(load_config(), client=client).run(state)
    assert plan.origin == "fallback_rule"
    assert plan.subqueries
    assert any("fallback:" in r for r in plan.repairs)
    assert state.traces[-1].degraded


def test_budget_exhaustion_takes_the_ladder_and_spends_nothing():
    client = StubClient({"strategy": "bridge", "subqueries": []})
    state = make_state(BRIDGE_Q, max_llm_calls=2, reserve_llm_calls=2)
    plan = Planner(load_config(), client=client).run(state)
    assert client.calls == []
    assert plan.origin == "fallback_rule"
    assert state.budget.llm_calls == 0


def test_validation_empty_falls_back_and_records_why():
    client = StubClient({"strategy": "bridge", "subqueries": [{"id": "q1", "text": ""}]})
    state = make_state(BRIDGE_Q)
    plan = Planner(load_config(), client=client).run(state)
    assert plan.origin == "fallback_rule"
    assert state.traces[-1].fallback_reason == "validation_empty"


def test_repaired_plan_is_marked_llm_repaired():
    client = StubClient(
        {
            "strategy": "bridge",
            "subqueries": [
                {"id": "one", "text": "Who directed Titanic?", "depends_on": []},
                {"id": "two", "text": "When was {{one.answer}} born?", "depends_on": []},
            ],
        }
    )
    plan = Planner(load_config(), client=StubClient(*client.replies)).run(make_state(BRIDGE_Q))
    assert plan.origin == "llm_repaired"
    assert plan.depth == 2
    assert plan.subqueries[1].depends_on == ("q1",)


def test_template_shortcut_costs_no_llm_call_when_enabled():
    cfg = load_config().with_overrides({"agents.planner.template_shortcut": True})
    client = StubClient()
    state = make_state(COMPARISON_Q)
    plan = Planner(cfg, client=client).run(state)
    assert plan.origin == "template_shortcut"
    assert client.calls == []
    assert state.budget.llm_calls_saved == 1


# ---------------------------------------------------------------------------
# Routing table (architecture 5.1) -- all seven rows
# ---------------------------------------------------------------------------

class StubBM25:
    """Just the two lexicon statistics the routing features need."""

    def __init__(self, oov: float = 0.0, min_df: float = 100.0) -> None:
        self._oov = oov
        self._min_df = min_df

    def oov_rate(self, text: str) -> float:
        return self._oov

    def min_df(self, text: str) -> float:
        return self._min_df


def features(**kwargs: Any) -> RoutingFeatures:
    base = {
        "n_tokens": 10,
        "has_quoted": False,
        "entity_runs": 1,
        "entity_ratio": 0.2,
        "oov_rate": 0.0,
        "min_df": 100.0,
        "is_dependent": False,
    }
    base.update(kwargs)
    return RoutingFeatures(**base)  # type: ignore[arg-type]


def test_r1_planner_hint_wins():
    tool, rule, selector, _ = route(features(oov_rate=0.9), tool_hint="bm25_search")
    assert (tool, rule, selector) == ("bm25_search", "R1_planner_hint", "planner_hint")


def test_r2_high_oov_routes_dense():
    tool, rule, _, _ = route(features(oov_rate=0.35))
    assert (tool, rule) == ("dense_search", "R2_oov")


def test_r3_quoted_or_entity_dense_routes_bm25():
    assert route(features(has_quoted=True))[1] == "R3_entity_dense"
    tool, rule, _, _ = route(features(entity_ratio=0.6, n_tokens=6))
    assert (tool, rule) == ("bm25_search", "R3_entity_dense")


def test_r4_rare_term_routes_bm25():
    tool, rule, _, _ = route(features(min_df=2, entity_runs=1, entity_ratio=0.1, n_tokens=10))
    assert (tool, rule) == ("bm25_search", "R4_rare_term")


def test_r5_dependent_routes_hybrid():
    tool, rule, _, _ = route(features(is_dependent=True, n_tokens=9, entity_ratio=0.1))
    assert (tool, rule) == ("hybrid_search", "R5_dependent")


def test_r6_long_or_entity_sparse_routes_hybrid():
    assert route(features(n_tokens=14, entity_ratio=0.1))[1] == "R6_long_or_entity_sparse"
    assert route(features(entity_runs=0, entity_ratio=0.0, n_tokens=9))[1] == "R6_long_or_entity_sparse"


def test_r7_default_is_hybrid():
    tool, rule, _, _ = route(features(n_tokens=10, entity_runs=1, entity_ratio=0.2))
    assert (tool, rule) == ("hybrid_search", "R7_default")


def test_rule_order_r2_beats_r3():
    """R2 is checked before R3: an out-of-vocabulary query BM25 cannot serve
    must not be handed to BM25 merely because it is short and capitalised."""
    tool, rule, _, _ = route(features(oov_rate=0.9, entity_ratio=0.9, n_tokens=4))
    assert (tool, rule) == ("dense_search", "R2_oov")


def test_features_are_computed_on_the_resolved_text():
    resolved = extract_features(
        "James Cameron date of birth", is_dependent=True, bm25=StubBM25(oov=0.1, min_df=7)
    )
    assert resolved.entity_runs >= 1
    assert resolved.is_dependent
    assert resolved.oov_rate == pytest.approx(0.1)
    assert resolved.as_dict()["min_df"] == 7.0


def test_infinite_min_df_is_json_safe():
    all_oov = extract_features("zzz qqq", bm25=None)
    assert all_oov.as_dict()["min_df"] == 1e9


# ---------------------------------------------------------------------------
# Retrieval execution
# ---------------------------------------------------------------------------

def _registry(hits: Sequence[ScoredPassage], record: list[str] | None = None) -> ToolRegistry:
    def search(query: str, top_k: int | None = None) -> list[ScoredPassage]:
        if record is not None:
            record.append(query)
        return list(hits)

    return ToolRegistry(
        [
            ToolSpec(name=name, fn=search, arg_schema={}, description=name)
            for name in ("bm25_search", "dense_search", "hybrid_search")
        ]
    )


def test_heuristic_routing_issues_zero_llm_calls_and_counts_the_saving():
    client = StubClient()
    hits = [scored("Titanic", "Titanic is a 1997 film.", rank=0)]
    agent = RetrievalAgent(_registry(hits), load_config(), bm25=StubBM25(), client=client)
    state = make_state(BRIDGE_Q)
    result = agent.run(state, SubQuery(id="q1", text="Who directed Titanic?"))
    assert client.calls == []
    assert state.budget.llm_calls == 0
    assert state.budget.llm_calls_saved == 1
    assert result.selection.selector == "heuristic"
    assert result.passages and result.passages[0].rank == 0
    assert result.selection.features["n_tokens"] > 0


def test_rewrites_are_issued_as_extra_queries_for_free():
    issued: list[str] = []
    hits = [scored("Titanic", "Titanic is a 1997 film.")]
    agent = RetrievalAgent(
        _registry(hits, issued), load_config(), bm25=StubBM25(), client=StubClient()
    )
    state = make_state(BRIDGE_Q)
    subquery = SubQuery(
        id="q1", text="Who directed Titanic?", rewrites=("director of Titanic", "Titanic director")
    )
    result = agent.run(state, subquery)
    assert len(issued) == 3
    assert result.queries_issued == (
        "Who directed Titanic?", "director of Titanic", "Titanic director",
    )
    assert state.budget.llm_calls == 0


def test_a_failing_tool_falls_back_to_hybrid():
    def boom(query: str, top_k: int | None = None) -> list[ScoredPassage]:
        raise RuntimeError("index unavailable")

    hits = [scored("Titanic", "Titanic is a 1997 film.")]
    registry = ToolRegistry(
        [
            ToolSpec(name="bm25_search", fn=boom, arg_schema={}, description="x"),
            ToolSpec(
                name="hybrid_search",
                fn=lambda q, k=None: list(hits),
                arg_schema={},
                description="y",
            ),
        ]
    )
    agent = RetrievalAgent(registry, load_config(), bm25=StubBM25(), client=StubClient())
    state = make_state()
    result = agent.run(
        state, SubQuery(id="q1", text="Titanic film", tool_hint="bm25_search")
    )
    assert result.selection.tool == "hybrid_search"
    assert result.selection.selector == "fallback"
    assert result.passages


def test_no_search_tool_returns_a_degraded_empty_result():
    agent = RetrievalAgent(ToolRegistry(), load_config(), client=StubClient())
    state = make_state()
    result = agent.run(state, SubQuery(id="q1", text="Anything at all"))
    assert result.passages == ()
    assert result.degraded
    assert result.error == "no_search_tool"


def test_llm_routing_path_is_used_when_the_shortcut_is_off():
    cfg = load_config().with_overrides({"agents.retriever.heuristic_shortcut": False})
    client = StubClient({"tool": "dense_search", "rerank": True, "reason": "paraphrase"})
    hits = [scored("Titanic", "Titanic is a 1997 film.")]
    agent = RetrievalAgent(_registry(hits), cfg, bm25=StubBM25(), client=client)
    state = make_state()
    result = agent.run(state, SubQuery(id="q1", text="a long natural language question here"))
    assert len(client.calls) == 1
    assert result.selection.selector == "llm"
    assert result.selection.tool == "dense_search"
