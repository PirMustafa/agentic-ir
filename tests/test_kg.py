"""Knowledge-graph unit tests.

Everything runs offline over a four-passage hand-built corpus, so the suite is
a second-scale feedback loop. What is asserted is the machinery the report
leans on: that identity is stable under the normalisation in architecture 3.3,
that every edge carries citable provenance, that the bidirectional search finds
a bridge entity that is genuinely there, and that the navigator issues zero LLM
calls in the shipped configuration.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_ir.agents.kg_navigator import KGNavigator, intermediate_entity
from agentic_ir.config import load_config
from agentic_ir.kg.build import (
    build_alias_table,
    build_graph,
    mention_df_cap,
    prune_hub_mentions,
    relation_label,
)
from agentic_ir.kg.entity_link import (
    AliasTable,
    alias_variants,
    fold,
    normalise_entity,
    tokenize,
)
from agentic_ir.kg.graph import KnowledgeGraph
from agentic_ir.kg.traverse import (
    bfs_neighborhood,
    bidirectional_bfs,
    bridge_paths,
    path_score,
)
from agentic_ir.state import Budget, QuestionState
from agentic_ir.types import (
    Passage,
    RetrievalResult,
    ScoredPassage,
    SubQuery,
    ToolSelection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINI_CORPUS = [
    (
        "Titanic (1997 film)",
        [
            "Titanic is a 1997 American epic romance film directed by James Cameron.",
            "It stars Leonardo DiCaprio and Kate Winslet.",
        ],
    ),
    (
        "James Cameron",
        [
            "James Cameron is a Canadian filmmaker.",
            "He directed Avatar in 2009.",
        ],
    ),
    (
        "Avatar (2009 film)",
        [
            "Avatar is a 2009 epic science fiction film.",
            "It grossed over 2 billion dollars worldwide.",
        ],
    ),
    (
        "DiCaprio, Leonardo",
        ["Leonardo DiCaprio is an American actor born in 1974."],
    ),
]


def _passages() -> list[Passage]:
    return [
        Passage(
            doc_id=f"hotpotqa:doc{i}",
            title=title,
            text=" ".join(sentences),
            sentences=tuple(sentences),
            source="hotpotqa",
        )
        for i, (title, sentences) in enumerate(MINI_CORPUS)
    ]


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    kg, _stats = build_graph(_passages(), dataset="hotpotqa")
    return kg


@pytest.fixture
def state() -> QuestionState:
    return QuestionState(
        qid="q-test",
        question="Who directed the film that starred Leonardo DiCaprio?",
        dataset="hotpotqa",
        config_name="agentic_full",
        budget=Budget().start(),
    )


class StubResponse:
    def __init__(self, parsed: dict[str, Any] | None) -> None:
        self.parsed = parsed
        self.text = str(parsed)
        self.thinking = None
        self.thinking_chars = 0
        self.model = "stub"
        self.retries = 0
        self.latency_s = 0.01
        self.agent = "stub"


class StubClient:
    """Records calls, so a test can assert the LLM-free path made none."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def model_for(self, agent: str) -> str:
        return "stub-model"

    def chat(self, messages: Any, *, agent: str, **kwargs: Any) -> StubResponse:
        self.calls.append({"agent": agent, **kwargs})
        reply = self.replies.pop(0) if self.replies else None
        if isinstance(reply, Exception):
            raise reply
        return StubResponse(reply if isinstance(reply, dict) else None)


# ---------------------------------------------------------------------------
# Normalisation and aliases
# ---------------------------------------------------------------------------

def test_normalise_strips_diacritics_parentheticals_and_apostrophes():
    assert normalise_entity("Arthur's Magazine") == "arthurs magazine"
    assert normalise_entity("Jimmy Butler (basketball)") == "jimmy butler"
    assert normalise_entity("Volker Schlondorff") == normalise_entity("Volker Schlöndorff")
    assert normalise_entity("Xawery Żuławski") == "xawery zulawski"
    assert normalise_entity("  Foo   Bar  ") == "foo bar"


def test_normalise_never_collapses_punctuation_titles_onto_one_id():
    """`!!!` is a real HotpotQA page; so is `?`. They must stay distinct."""
    assert normalise_entity("!!!") != normalise_entity("?")
    assert normalise_entity("!!!") == "!!!"


def test_parenthetical_variants_merge_onto_one_node():
    assert normalise_entity("Titanic (1997 film)") == normalise_entity("Titanic (ship)")


def test_alias_variants_include_comma_inversion():
    variants = alias_variants("DiCaprio, Leonardo")
    assert "Leonardo DiCaprio" in variants
    variants = alias_variants("Titanic (1997 film)")
    assert "Titanic (1997 film)" in variants and "Titanic" in variants
    # Multi-comma titles are noun phrases, not surname-first names.
    assert "Anhalt-Zerbst John II, Prince of" not in alias_variants(
        "John II, Prince of Anhalt-Zerbst"
    )


def test_alias_table_prefers_the_longest_match():
    table = AliasTable.build(
        [("Chicago", "chicago"), ("University of Chicago", "university of chicago")]
    )
    matches = table.find_all("She studied at the University of Chicago last year.")
    assert [m.alias for m in matches] == ["university of chicago"]


def test_alias_table_rejects_stopwords_and_short_and_numeric_aliases():
    table = AliasTable()
    assert table.add("The", "the") is False
    assert table.add("Yo", "yo") is False
    assert table.add("1996", "1996") is False
    assert table.add("Avatar", "avatar") is True


def test_matching_is_token_level_not_substring():
    table = AliasTable.build([("Ohio", "ohio")])
    assert table.find_all("An Ohioan wrote it.") == ()
    assert len(table.find_all("She lives in Ohio.")) == 1


def test_fold_is_idempotent_between_builder_and_linker():
    """The builder and the navigator must agree, or nothing ever links."""
    assert fold("Arthur's Magazine") == fold("ARTHURS  magazine")


# ---------------------------------------------------------------------------
# Relation labelling
# ---------------------------------------------------------------------------

def test_relation_label_is_verb_anchored_and_truncated_to_six_tokens():
    sentence = "Titanic is a 1997 American epic romance film directed by James Cameron."
    tokens = tokenize(sentence)
    label = relation_label(tokens, object_span=(10, 12), subject_spans=[(0, 1)])
    assert label.split()[0] == "is"
    assert len(label.split()) == 6


def test_relation_label_falls_back_to_mentions_without_a_verb():
    tokens = tokenize("Paris, France, Europe")
    assert relation_label(tokens, object_span=(2, 3), subject_spans=[(0, 1)]) == "mentions"


def test_relation_label_uses_left_context_when_the_subject_is_a_pronoun():
    tokens = tokenize("He directed Avatar in 2009.")
    assert relation_label(tokens, object_span=(2, 3)) == "directed"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_build_creates_one_node_per_normalised_title(graph):
    assert set(graph.g.nodes) == {
        "titanic", "james cameron", "avatar", "leonardo dicaprio"
    }


def test_every_edge_is_independently_citable(graph):
    """The whole justification for the KG being scoreable rather than decorative."""
    for u, v, data in graph.g.edges(data=True):
        assert data["doc_id"], (u, v)
        assert isinstance(data["sent_id"], int)
        assert data["sentence"], (u, v)
        # The sentence must actually be the one that licensed the edge.
        assert data["sentence"] in [s for _, sents in MINI_CORPUS for s in sents]


def test_edges_follow_mention_direction(graph):
    assert graph.g.has_edge("titanic", "james cameron")
    assert graph.g.has_edge("james cameron", "avatar")
    assert not graph.g.has_edge("avatar", "james cameron")


def test_comma_inverted_alias_links_the_page(graph):
    """`DiCaprio, Leonardo` must be findable from "Leonardo DiCaprio" in prose."""
    assert graph.g.has_edge("titanic", "leonardo dicaprio")


def test_no_self_edges(graph):
    assert not any(u == v for u, v in graph.g.edges())


def test_hub_weight_formula(graph):
    from math import log

    for subject in graph.g.nodes:
        out_degree = graph.g.out_degree(subject)
        if not out_degree:
            continue
        expected = 1.0 / (1.0 + log(1.0 + out_degree))
        for _, _, data in graph.g.out_edges(subject, data=True):
            assert data["weight"] == pytest.approx(expected)


def test_hub_pruning_removes_incoming_edges_but_keeps_the_node():
    kg, _ = build_graph(_passages(), dataset="hotpotqa")
    stats = prune_hub_mentions(kg, cap=0 if False else 1)
    assert stats["hub_nodes_pruned"] >= 1
    # Every node survives; only in-edges of over-mentioned entities go.
    assert len(kg) == 4
    assert all(kg.g.in_degree(n) <= 1 for n in kg.g.nodes)


def test_mention_df_cap_is_a_ratio_with_a_floor():
    assert mention_df_cap(66581, 0.002) == 133
    assert mention_df_cap(100, 0.002) == 50      # floor protects tiny corpora
    assert mention_df_cap(66581, 0.0) == 0       # 0 disables pruning


def test_alias_table_counts_rejections(graph):
    _table, stats = build_alias_table(graph)
    assert stats["aliases_accepted"] > 0
    assert stats["ambiguous_alias_keys"] == 0


def test_build_is_deterministic():
    a, _ = build_graph(_passages(), dataset="hotpotqa")
    b, _ = build_graph(_passages(), dataset="hotpotqa")
    assert sorted(a.g.edges(data=True)) == sorted(b.g.edges(data=True))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path, graph):
    graph.save(dataset="hotpotqa", directory=tmp_path, meta={"stats": {"mention_df_cap": 7}})
    reloaded = KnowledgeGraph.load("hotpotqa", directory=tmp_path)
    assert set(reloaded.g.nodes) == set(graph.g.nodes)
    assert reloaded.num_edges == graph.num_edges
    assert reloaded.mention_df_cap == 7
    for u, v, data in graph.g.edges(data=True):
        assert reloaded.g.edges[u, v]["sentence"] == data["sentence"]
        assert reloaded.g.edges[u, v]["weight"] == pytest.approx(data["weight"])


def test_load_without_a_built_graph_raises_actionably(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_kg.py"):
        KnowledgeGraph.load("hotpotqa", directory=tmp_path)


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def test_bfs_reaches_two_hops_and_records_the_edges(graph):
    hood = bfs_neighborhood(graph, "titanic", max_hops=2, max_neighbors=25)
    assert "james cameron" in hood.nodes_by_hop[0]
    assert "avatar" in hood.nodes_by_hop[1]
    assert all(t.doc_id for t in hood.triples)


def test_bfs_respects_max_neighbors(graph):
    hood = bfs_neighborhood(graph, "titanic", max_hops=1, max_neighbors=1)
    assert len(hood.nodes_by_hop[0]) == 1


def test_bfs_on_an_unknown_seed_is_empty_not_an_error(graph):
    assert bfs_neighborhood(graph, "nobody", max_hops=2).nodes == ()


def test_bidirectional_bfs_finds_the_known_bridge_entity(graph):
    """The highest-value operation in the agent (architecture 3.3).

    Titanic and Avatar share no edge; the only thing joining them is the page
    of the director who made both, and the search must return exactly that.
    """
    path = bidirectional_bfs(graph, "titanic", "avatar", max_hops=2, max_neighbors=25)
    assert path is not None
    assert path.nodes == ("titanic", "james cameron", "avatar")
    assert path.hops == 2
    assert path.bridge_entity == "james cameron"
    assert intermediate_entity(path, ("titanic", "avatar")) == "james cameron"
    # Every edge on the path is real and citable.
    assert [e.doc_id for e in path.edges] == ["hotpotqa:doc0", "hotpotqa:doc1"]


def test_bidirectional_bfs_returns_the_direct_edge_when_adjacent(graph):
    path = bidirectional_bfs(graph, "titanic", "james cameron", max_hops=2)
    assert path is not None and path.hops == 1
    # Adjacent seeds have no node strictly between them, so there is no bridge
    # to substitute into a query -- and saying so is better than naming a seed.
    assert intermediate_entity(path, ("titanic", "james cameron")) is None


def test_bidirectional_bfs_returns_none_when_unreachable(graph):
    assert bidirectional_bfs(graph, "avatar", "leonardo dicaprio", max_hops=0) is None


def test_path_score_is_length_normalised(graph):
    short = bidirectional_bfs(graph, "titanic", "james cameron", max_hops=2)
    long = bidirectional_bfs(graph, "titanic", "avatar", max_hops=2)
    assert 0.0 < long.score <= 1.0
    assert 0.0 < short.score <= 1.0
    assert path_score(()) == 0.0


def test_bridge_paths_ranking_is_total_and_deterministic(graph):
    a = bridge_paths(graph, ("titanic", "avatar", "leonardo dicaprio"), max_hops=2)
    b = bridge_paths(graph, ("titanic", "avatar", "leonardo dicaprio"), max_hops=2)
    assert a == b
    assert [p.nodes for p in a] == sorted(
        [p.nodes for p in a], key=lambda n: (-dict((p.nodes, p.score) for p in a)[n], len(n), n)
    )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

def test_navigator_makes_zero_llm_calls_in_the_shipped_config(graph, state):
    """`entity_linker: alias_match` is budgeted at exactly zero calls (2.6)."""
    client = StubClient()
    nav = KGNavigator(graph, client=client)
    sq = SubQuery(id="q1", text="Who directed Titanic?", entities=("Titanic",))
    result = nav.run(state, sq)
    assert client.calls == []
    assert state.budget.llm_calls == 0
    assert state.budget.llm_calls_saved >= 1
    assert result.linked_by == "alias_match"
    assert [e.entity_id for e in result.seeds] == ["titanic"]


def test_navigator_returns_citable_evidence(graph, state):
    nav = KGNavigator(graph, client=StubClient())
    result = nav.run(state, SubQuery(id="q1", text="Tell me about Titanic."))
    assert result.evidence
    for ev in result.evidence:
        assert ev.kind == "kg_triple"
        assert ev.provenance == "kg"
        assert ev.doc_id and ev.text and ev.triple is not None


def test_navigator_finds_and_publishes_the_bridge_entity(graph, state):
    nav = KGNavigator(graph, client=StubClient())
    sq = SubQuery(
        id="q1",
        text="What connects Titanic and Avatar?",
        entities=("Titanic", "Avatar"),
    )
    result = nav.run(state, sq)
    assert result.bridge_entity == "James Cameron"      # the surface name, not the id
    assert state.bridge_entities["q1"] == "James Cameron"
    assert result.paths and result.paths[0].hops == 2


def test_navigator_does_not_force_a_path_for_a_comparison(graph, state):
    nav = KGNavigator(graph, client=StubClient())
    sq = SubQuery(
        id="q1",
        text="Were Avatar and Leonardo DiCaprio both from the 1970s?",
        entities=("Avatar", "Leonardo DiCaprio"),
        intent="comparison",
        answer_type="yesno",
    )
    result = nav.run(state, sq)
    assert result.paths == ()
    assert result.bridge_entity is None
    # Instead: each seed's own attribute-bearing sentences.
    assert any("2009" in ev.text for ev in result.evidence)
    assert any("1974" in ev.text for ev in result.evidence)


def test_navigator_falls_back_to_retrieved_titles(graph, state):
    """Rung 2 of the 3.3 ladder: no alias match, but retrieval found something."""
    passage = Passage(
        doc_id="hotpotqa:doc2",
        title="Avatar (2009 film)",
        text="Avatar is a 2009 epic science fiction film.",
        sentences=("Avatar is a 2009 epic science fiction film.",),
        source="hotpotqa",
    )
    state.results["q1"] = RetrievalResult(
        subquery_id="q1",
        query_text="zzz",
        selection=ToolSelection(tool="hybrid_search", selector="heuristic"),
        passages=(ScoredPassage(passage=passage, score=1.0, rank=0, provenance="hybrid"),),
    )
    nav = KGNavigator(graph, client=StubClient())
    result = nav.run(state, SubQuery(id="q1", text="qqqq zzzz wwww"))
    assert result.linked_by == "retrieval_titles"
    assert [e.entity_id for e in result.seeds] == ["avatar"]


def test_navigator_returns_an_empty_result_rather_than_raising(state):
    """An empty KGResult is legal and non-fatal (3.3)."""
    nav = KGNavigator(KnowledgeGraph(), client=StubClient())
    result = nav.run(state, SubQuery(id="q1", text="anything at all"))
    assert result.degraded is True
    assert result.linked_by == "none"
    assert result.evidence == ()
    assert state.kg_results["q1"] is result


def test_navigator_survives_a_broken_graph(state, monkeypatch):
    graph = KnowledgeGraph()
    graph.add_entity("x", name="X", aliases=("X",))

    def boom(*_a, **_k):
        raise RuntimeError("graph corrupted")

    monkeypatch.setattr(KnowledgeGraph, "link", boom)
    result = KGNavigator(graph, client=StubClient()).run(
        state, SubQuery(id="q1", text="X")
    )
    assert result.degraded is True
    assert state.errors  # the exception was recorded, not swallowed silently


def test_navigator_honours_configured_hop_and_neighbor_caps(graph, state):
    cfg = load_config().with_overrides(
        {"agents.kg.max_hops": 1, "agents.kg.max_neighbors": 1}
    )
    nav = KGNavigator(graph, cfg=cfg, client=StubClient())
    result = nav.run(state, SubQuery(id="q1", text="Tell me about Titanic."))
    assert len(result.neighbors) <= 1
    assert len(result.evidence) <= 1
