"""Baseline tests: the five comparison systems, offline.

Everything here runs against an in-memory corpus, fake indexes and a stub LLM
client, so the suite stays a second-scale feedback loop and needs neither
Ollama nor the 130 MB FAISS index on disk. :class:`RetrievalStack` is a plain
dataclass precisely so this is possible -- any object with ``corpus``,
``hybrid`` and ``reranker`` works.

What is asserted is the machinery Chapter 4's table rests on, and in particular
the three things that would invalidate the comparison while leaving every
number looking plausible:

* **the generative baselines retrieve identically to ``hybrid_rerank``** -- if
  they did not, ``naive_rag - hybrid_rerank`` would measure retrieval drift
  rather than one generation step;
* **no baseline reads the answer key** -- ``run_all`` is handed records whose
  ``answer`` and ``supporting_facts`` raise on access;
* **budgets are hard** -- a model that keeps asking follow-ups cannot spend
  more calls than ``self_ask`` declares, and a dead model still produces an
  answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from agentic_ir.baselines.base import (
    BASELINE_NAMES,
    BaselineBase,
    BaselineHop,
    BaselineResult,
    RetrievalStack,
    build_baseline,
    gold_doc_ids,
    index_dir,
    iter_baselines,
    predict_supporting_facts,
)
from agentic_ir.baselines.bm25_only import BM25OnlyBaseline
from agentic_ir.baselines.dense_only import DenseOnlyBaseline
from agentic_ir.baselines.hybrid_rerank import HybridRerankBaseline
from agentic_ir.baselines.naive_rag import (
    NaiveRAGBaseline,
    extractive_answer,
    passage_label,
    render_passages,
    resolve_citations,
)
from agentic_ir.baselines.self_ask import (
    MAX_FOLLOWUPS,
    SelfAskBaseline,
    intermediate_answer,
    normalise_query,
)
from agentic_ir.config import load_config
from agentic_ir.llm import LLMError, LLMFormatError
from agentic_ir.types import GoldAnswer, Passage, ScoredPassage

QUESTION = "Which Italian-American composer wrote the opera Maria Golovin?"
FOLLOWUP = "Who composed the opera Maria Golovin?"


# ---------------------------------------------------------------------------
# A tiny corpus
# ---------------------------------------------------------------------------

def _passage(title: str, *sentences: str) -> Passage:
    return Passage(
        doc_id=f"hotpotqa:{title.replace(' ', '_')}",
        title=title,
        text=" ".join(sentences),
        sentences=tuple(sentences),
        source="hotpotqa",
    )


CORPUS_PASSAGES: tuple[Passage, ...] = (
    _passage(
        "Maria Golovin",
        "Maria Golovin is an English language opera in three acts.",
        "It was written by Gian Carlo Menotti in 1958.",
    ),
    _passage(
        "Gian Carlo Menotti",
        "Gian Carlo Menotti was an Italian-American composer and librettist.",
        "He was born on 7 July 1911 in Cadegliano.",
    ),
    _passage(
        "The Old Maid and the Thief",
        "The Old Maid and the Thief is a comic opera in fourteen scenes.",
    ),
    _passage(
        "Arthur's Magazine",
        "Arthur's Magazine was an American literary periodical published in the 1840s.",
    ),
    _passage(
        "First for Women",
        "First for Women is a woman's magazine published by Bauer Media Group.",
        "It was started in 1989.",
    ),
    _passage("Vogue (magazine)", "Vogue is an American monthly fashion magazine."),
)


class FakeCorpus:
    """The three-method ``CorpusLike`` slice the indexes actually use."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        self._by_id = {p.doc_id: p for p in passages}
        self._passages = tuple(passages)

    def __len__(self) -> int:
        return len(self._passages)

    def __iter__(self):
        return iter(self._passages)

    def get(self, doc_id: str) -> Passage:
        return self._by_id[doc_id]


def _overlap(query: str, passage: Passage) -> float:
    """Content-word overlap. Crude on purpose: the tests assert plumbing."""
    terms = {t for t in query.lower().replace("?", "").split() if len(t) > 3}
    haystack = f"{passage.title} {passage.text}".lower()
    return float(sum(1 for t in terms if t in haystack))


@dataclass
class FakeChannel:
    """A single retrieval channel over :data:`CORPUS_PASSAGES`.

    Ranking is ``(-score, doc_id)`` -- the same explicit tiebreaker the real
    indexes use, so a determinism test here means what it means in production.
    """

    corpus: FakeCorpus
    provenance: str
    boost_titles: tuple[str, ...] = ()
    queries: list[str] | None = None

    def __post_init__(self) -> None:
        if self.queries is None:
            self.queries = []

    def search(self, query: str, top_k: int | None = None, **_: Any) -> list[ScoredPassage]:
        self.queries.append(query)
        pairs = []
        for passage in self.corpus:
            score = _overlap(query, passage)
            if passage.title in self.boost_titles:
                score += 3.0  # the channels must not rank identically
            if score > 0:
                pairs.append((passage, score))
        pairs.sort(key=lambda item: (-item[1], item[0].doc_id))
        limit = len(pairs) if top_k is None else int(top_k)
        return [
            ScoredPassage(
                passage=p,
                score=s,
                rank=i,
                provenance=self.provenance,  # type: ignore[arg-type]
                component_scores={self.provenance: s},
            )
            for i, (p, s) in enumerate(pairs[:limit])
        ]


class FakeHybrid:
    """``bm25`` + ``dense`` + a fused ``search``, shaped like ``HybridIndex``."""

    def __init__(self, corpus: FakeCorpus) -> None:
        self.corpus = corpus
        self.bm25 = FakeChannel(corpus, "bm25", boost_titles=("Maria Golovin",))
        self.dense = FakeChannel(corpus, "dense", boost_titles=("Gian Carlo Menotti",))
        self.queries: list[tuple[str, int | None, int | None]] = []

    def search(
        self, query: str, top_k: int | None = None, *, candidate_k: int | None = None
    ) -> list[ScoredPassage]:
        self.queries.append((query, top_k, candidate_k))
        from agentic_ir.indexing.hybrid import fuse_results

        return fuse_results(
            [self.bm25.search(query, candidate_k), self.dense.search(query, candidate_k)],
            rrf_k=60,
            top_k=top_k,
            provenance="hybrid",
        )

    def warmup(self) -> FakeHybrid:
        return self


@dataclass
class FakeOutcome:
    passages: tuple[ScoredPassage, ...]
    ran: bool


class FakeReranker:
    """Promotes any passage whose title is named in the query.

    Observable on purpose: a test can tell whether the cross-encoder pass
    actually ran by looking at the ranking, not only at a boolean.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []

    def rerank(
        self, query: str, passages: Sequence[ScoredPassage], *, force: bool = True
    ) -> FakeOutcome:
        self.calls.append((query, len(passages), force))
        if not force:
            return FakeOutcome(tuple(passages), False)
        lowered = query.lower()
        scored = sorted(
            passages,
            key=lambda sp: (
                -(2.0 if sp.passage.title.lower() in lowered else _overlap(query, sp.passage)),
                sp.passage.doc_id,
            ),
        )
        return FakeOutcome(
            tuple(
                ScoredPassage(
                    passage=sp.passage,
                    score=float(len(scored) - i),
                    rank=i,
                    provenance="rerank",
                    component_scores=dict(sp.component_scores),
                )
                for i, sp in enumerate(scored)
            ),
            True,
        )

    def warmup(self) -> FakeReranker:
        return self


@pytest.fixture
def stack() -> RetrievalStack:
    corpus = FakeCorpus(CORPUS_PASSAGES)
    return RetrievalStack(
        corpus=corpus,  # type: ignore[arg-type]
        hybrid=FakeHybrid(corpus),  # type: ignore[arg-type]
        reranker=FakeReranker(),  # type: ignore[arg-type]
        top_k=4,
        candidate_k=6,
        dataset="hotpotqa",
    )


@pytest.fixture
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------

class StubResponse:
    def __init__(self, parsed: dict[str, Any] | None) -> None:
        self.parsed = parsed
        self.text = str(parsed)
        self.thinking = None
        self.thinking_chars = 0
        self.model = "stub"
        self.agent = "stub"
        self.retries = 0
        self.latency_s = 0.01


class StubClient:
    """Returns queued replies; an ``Exception`` in the queue is raised instead.

    Records every call so a test can assert the exact number spent -- which is
    the whole point of the ``llm_calls`` column.
    """

    def __init__(self, *replies: Any, repeat_last: bool = False) -> None:
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    def model_for(self, agent: str) -> str:
        return "stub-model"

    def chat(self, messages: Sequence[Any], *, agent: str, **kwargs: Any) -> StubResponse:
        self.calls.append({"agent": agent, "messages": list(messages), **kwargs})
        if self.replies:
            reply = self.replies[-1] if (self.repeat_last and len(self.replies) == 1) \
                else self.replies.pop(0)
        else:
            reply = None
        if isinstance(reply, Exception):
            raise reply
        return StubResponse(reply if isinstance(reply, dict) else None)


def answer_reply(answer: str, citations: Sequence[str] = ("p1",)) -> dict[str, Any]:
    return {
        "answer": answer,
        "answer_sentence": f"The composer is {answer}.",
        "citations": list(citations),
    }


def step_reply(
    *,
    followup: str = "",
    answer: str = "",
    citations: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "needs_followup": bool(followup),
        "followup": followup,
        "answer": answer,
        "answer_sentence": f"The composer is {answer}." if answer else "",
        "citations": list(citations),
    }


class TrapGold:
    """A ``GoldAnswer``-shaped record whose answer key detonates on access.

    ``run_all`` takes the evaluation slice, and the slice carries the answers.
    A baseline that peeked would invalidate the very comparison it exists to
    provide, and the failure would be invisible in the numbers -- they would
    simply be too good.
    """

    def __init__(self, qid: str, question: str) -> None:
        self.qid = qid
        self.question = question

    @property
    def answer(self) -> str:  # pragma: no cover - the assertion is the point
        raise AssertionError("a baseline read the gold answer")

    @property
    def supporting_facts(self) -> tuple:  # pragma: no cover - same
        raise AssertionError("a baseline read the gold supporting facts")


# ---------------------------------------------------------------------------
# The supporting-fact rule
# ---------------------------------------------------------------------------

def ranked(*titles: str) -> list[ScoredPassage]:
    by_title = {p.title: p for p in CORPUS_PASSAGES}
    return [
        ScoredPassage(
            passage=by_title[t], score=float(len(titles) - i), rank=i, provenance="hybrid"
        )
        for i, t in enumerate(titles)
    ]


def test_supporting_facts_are_every_sentence_of_the_top_two():
    facts = predict_supporting_facts(ranked("Maria Golovin", "Gian Carlo Menotti", "Vogue (magazine)"))
    assert facts == (
        ("Maria Golovin", 0),
        ("Maria Golovin", 1),
        ("Gian Carlo Menotti", 0),
        ("Gian Carlo Menotti", 1),
    )


def test_supporting_facts_never_double_count_a_repeated_title():
    """Two ranks of one title are one paragraph; counting it twice would
    inflate the predicted set and depress sp_precision for a non-reason."""
    pool = ranked("Maria Golovin") + ranked("Maria Golovin")
    assert predict_supporting_facts(pool) == (("Maria Golovin", 0), ("Maria Golovin", 1))


def test_supporting_facts_truncation_is_available_for_the_sensitivity_check():
    facts = predict_supporting_facts(ranked("Maria Golovin", "Gian Carlo Menotti"), max_sentences=1)
    assert facts == (("Maria Golovin", 0), ("Gian Carlo Menotti", 0))


def test_gold_doc_ids_are_sorted_and_deduplicated():
    gold = GoldAnswer(
        qid="q1",
        question=QUESTION,
        answer="Gian Carlo Menotti",
        dataset="hotpotqa",
        supporting_facts=(("Maria Golovin", 0), ("Maria Golovin", 1), ("Gian Carlo Menotti", 0)),
    )
    assert gold_doc_ids(gold) == ("hotpotqa:Gian%20Carlo%20Menotti", "hotpotqa:Maria%20Golovin")


def test_index_dir_is_where_the_build_script_writes(cfg):
    assert index_dir("twowiki", cfg=cfg).parts[-2:] == ("twowiki", "hybrid")


# ---------------------------------------------------------------------------
# Retrieval-only baselines
# ---------------------------------------------------------------------------

def test_bm25_only_makes_one_hop_and_claims_no_answer(stack):
    result = BM25OnlyBaseline(stack).run(QUESTION, qid="q1")
    assert len(result.hops) == 1
    assert result.hops[0].tool == "bm25_search"
    assert result.hops[0].query == QUESTION  # verbatim: no rewriting
    assert result.answer is None  # not "": it never attempted an answer
    assert result.llm_calls == 0
    assert result.answered is False
    assert result.doc_ids == result.hops[0].doc_ids()


def test_dense_only_uses_the_dense_channel_only(stack):
    result = DenseOnlyBaseline(stack).run(QUESTION, qid="q1")
    assert result.hops[0].tool == "dense_search"
    assert result.answer is None
    assert stack.hybrid.queries == []  # never touched the fused channel


def test_the_two_single_channel_baselines_disagree(stack):
    """If they ranked identically the hybrid rung would have nothing to fuse,
    and the whole 'RRF earns its complexity' argument would be untestable."""
    sparse = BM25OnlyBaseline(stack).run(QUESTION).doc_ids
    dense = DenseOnlyBaseline(stack).run(QUESTION).doc_ids
    assert sparse != dense


def test_hybrid_rerank_forces_the_cross_encoder(stack):
    result = HybridRerankBaseline(stack).run(QUESTION, qid="q1")
    assert result.hops[0].rerank_applied is True
    assert stack.reranker.calls[0][2] is True  # force=True, the gate is off
    # the pool is the full rerank depth, not top_k: reranking top_k can only
    # reorder what fusion already found, never promote what it ranked 40th
    assert stack.hybrid.queries[0] == (QUESTION, stack.candidate_k, stack.candidate_k)
    assert result.tool_calls == 2  # one search + one rerank


def test_hybrid_rerank_gated_variant_lets_the_gate_decide(stack):
    """The efficiency ablation. Default off -- see the module docstring."""
    result = HybridRerankBaseline(stack, gated=True).run(QUESTION)
    assert stack.reranker.calls[0][2] is False
    assert result.hops[0].rerank_applied is False


def test_a_missing_reranker_degrades_rather_than_raises(stack):
    stack.reranker = None
    result = HybridRerankBaseline(stack).run(QUESTION)
    assert result.hops[0].rerank_applied is False
    assert result.doc_ids  # still a full ranking


# ---------------------------------------------------------------------------
# The base contract
# ---------------------------------------------------------------------------

def test_run_never_raises_when_retrieval_explodes(stack):
    class Exploding(BaselineBase):
        name = "bm25_only"

        def retrieve(self, question: str, top_k: int) -> list[BaselineHop]:
            raise RuntimeError("index went away")

    result = Exploding(stack).run(QUESTION, qid="q9")
    assert result.degraded is True
    assert result.doc_ids == ()
    assert result.errors and "index went away" in result.errors[0]
    assert result.answer is None  # a retrieval-only baseline still claims nothing


def test_run_all_never_touches_the_answer_key(stack):
    slice_ = [TrapGold("q1", QUESTION), TrapGold("q2", "Who published First for Women?")]
    results = HybridRerankBaseline(stack).run_all(slice_)  # type: ignore[arg-type]
    assert [r.qid for r in results] == ["q1", "q2"]


def test_results_are_deterministic_across_runs(stack):
    baseline = HybridRerankBaseline(stack)
    first, second = baseline.run(QUESTION), baseline.run(QUESTION)
    assert first.doc_ids == second.doc_ids
    assert first.supporting_facts == second.supporting_facts


def test_to_record_emits_the_full_trace_key_set(stack, cfg):
    result = BM25OnlyBaseline(stack).run(QUESTION, qid="q1")
    record = result.to_record(run_id="r1", dataset="hotpotqa", model="stub")
    for key in (
        "schema_version", "run_id", "config_name", "qid", "metrics", "retrieved",
        "plans", "directives", "steps", "kg", "evidence", "candidates",
        "verifications", "supporting_facts", "errors",
    ):
        assert key in record
    assert record["plans"] == [] and record["verifications"] == []
    assert record["metrics"]["replans"] == 0
    assert record["metrics"]["n_subqueries"] == 1
    assert set(record["retrieved"]) == {"q1"}


# ---------------------------------------------------------------------------
# Rendering, citations, fallback
# ---------------------------------------------------------------------------

def test_passages_render_with_the_ids_the_model_is_told_to_cite(stack):
    block = render_passages(ranked("Maria Golovin", "Gian Carlo Menotti"))
    assert block.splitlines()[0].startswith("[p1] (Maria Golovin)")
    assert block.splitlines()[1].startswith("[p2] (Gian Carlo Menotti)")
    assert passage_label(0) == "p1"


def test_long_passages_are_truncated_not_dropped():
    long_passage = ranked("Maria Golovin")
    block = render_passages(long_passage, max_chars=20)
    assert block.endswith("...") and len(block) < 80


def test_citations_resolve_to_doc_ids_and_keep_unresolvable_ids_verbatim():
    pool = ranked("Maria Golovin", "Gian Carlo Menotti")
    citations, doc_ids, bad = resolve_citations(["p2", "p9", "p2"], pool)
    assert citations == ("p2", "p9")  # deduplicated, order preserved, p9 KEPT
    assert doc_ids == ("hotpotqa:Gian_Carlo_Menotti",)
    assert bad == 1


def test_citations_from_a_non_list_are_empty_rather_than_an_error():
    assert resolve_citations("p1", ranked("Maria Golovin")) == ((), (), 0)


def test_fallback_returns_a_title_for_an_entity_question():
    out = extractive_answer(QUESTION, ranked("Gian Carlo Menotti", "Maria Golovin"))
    assert out["answer"] == "Gian Carlo Menotti"
    assert out["rung"] == "title"
    assert out["citations"] == ("p1",)
    assert out["cited_doc_ids"] == ("hotpotqa:Gian_Carlo_Menotti",)


def test_fallback_returns_a_year_for_a_date_question():
    out = extractive_answer(
        "When was Gian Carlo Menotti born?", ranked("Gian Carlo Menotti")
    )
    assert out["answer"] == "7 July 1911"
    assert out["rung"] == "pattern:date"


def test_fallback_on_an_empty_pool_is_an_empty_attempt_not_a_crash():
    out = extractive_answer(QUESTION, [])
    assert out["answer"] == "" and out["rung"] == "empty"


def test_fallback_sentence_does_not_copy_its_own_premise():
    """Copying would score NLI entailment 1.0 against itself and report the
    degraded path as more confident than a grounded answer."""
    out = extractive_answer(QUESTION, ranked("Gian Carlo Menotti"))
    assert out["answer_sentence"].startswith("The answer to the question")


# ---------------------------------------------------------------------------
# naive_rag
# ---------------------------------------------------------------------------

def test_naive_rag_spends_exactly_one_call_and_uses_the_reply(stack):
    client = StubClient(answer_reply("Gian Carlo Menotti", ["p1"]))
    result = NaiveRAGBaseline(stack, client=client).run(QUESTION, qid="q1")
    assert len(client.calls) == 1
    assert result.llm_calls == 1
    assert result.answer == "Gian Carlo Menotti"
    assert result.answer_sentence == "The composer is Gian Carlo Menotti."
    assert result.citations == ("p1",)
    assert result.degraded is False
    assert result.extra["answer_origin"] == "llm"
    assert result.extra["citation_resolution"] == 1.0
    assert client.calls[0]["agent"] == "naive_rag"  # ledgered apart from the agents


def test_naive_rag_retrieval_is_identical_to_hybrid_rerank(stack):
    """The one comparison this rung exists to make. Any difference here is a
    bug, never a finding: it would confound 'one LLM call' with 'better
    retrieval'."""
    strong = HybridRerankBaseline(stack).run(QUESTION, qid="q1")
    naive = NaiveRAGBaseline(
        stack, client=StubClient(answer_reply("Gian Carlo Menotti"))
    ).run(QUESTION, qid="q1")
    assert naive.doc_ids == strong.doc_ids
    assert naive.supporting_facts == strong.supporting_facts
    assert [h.query for h in naive.hops] == [h.query for h in strong.hops]


def test_naive_rag_falls_back_when_the_server_is_down(stack):
    client = StubClient(LLMError("connection refused"))
    result = NaiveRAGBaseline(stack, client=client).run(QUESTION, qid="q1")
    assert result.degraded is True
    assert result.answer  # axiom 3: an EM/F1-comparable span even so
    assert result.extra["answer_origin"].startswith("fallback:")
    assert any("llm_error" in e for e in result.errors)


def test_naive_rag_falls_back_on_unparseable_output(stack):
    client = StubClient(
        LLMFormatError("bad json", raw="{", agent="naive_rag", model="stub", attempts=3)
    )
    result = NaiveRAGBaseline(stack, client=client).run(QUESTION)
    assert result.degraded is True
    assert result.answer
    assert result.extra["answer_origin"].startswith("fallback:")
    # classified apart from a transport error: the schema failed, not the server
    assert result.extra["parse_failures"] == 1
    assert result.metrics()["parse_failures"] == 1


def test_naive_rag_treats_an_empty_answer_as_a_failed_generation(stack):
    client = StubClient(answer_reply("", []))
    result = NaiveRAGBaseline(stack, client=client).run(QUESTION)
    assert result.degraded is True
    assert result.answer
    assert any("empty answer" in e for e in result.errors)


def test_naive_rag_with_no_budget_makes_no_call_at_all(stack):
    client = StubClient(answer_reply("Gian Carlo Menotti"))
    result = NaiveRAGBaseline(stack, client=client, max_llm_calls=0).run(QUESTION)
    assert client.calls == []
    assert result.llm_calls == 0
    assert result.extra["budget_exhausted"] is True
    assert result.answer  # still answered, deterministically


def test_naive_rag_counts_a_hallucinated_citation_without_deleting_it(stack):
    client = StubClient(answer_reply("Gian Carlo Menotti", ["p1", "p42"]))
    result = NaiveRAGBaseline(stack, client=client).run(QUESTION)
    assert result.citations == ("p1", "p42")
    assert result.extra["hallucinated_citations"] == 1
    assert result.extra["citation_resolution"] == pytest.approx(0.5)
    assert result.metrics()["citation_grounding"] is None  # a different quantity


def test_naive_rag_budget_does_not_leak_between_questions(stack):
    client = StubClient(answer_reply("A"), answer_reply("B"))
    baseline = NaiveRAGBaseline(stack, client=client)
    results = baseline.run_all([TrapGold("q1", QUESTION), TrapGold("q2", QUESTION)])  # type: ignore[arg-type]
    assert [r.llm_calls for r in results] == [1, 1]
    assert [r.answer for r in results] == ["A", "B"]


def test_naive_rag_sets_plan_depth_one(stack):
    result = NaiveRAGBaseline(stack, client=StubClient(answer_reply("X"))).run(QUESTION)
    assert result.metrics()["plan_depth"] == 1
    assert result.metrics()["n_subqueries"] == 1


# ---------------------------------------------------------------------------
# self_ask
# ---------------------------------------------------------------------------

def test_normalise_query_ignores_articles_and_punctuation():
    assert normalise_query("Who composed the opera Maria Golovin?") == normalise_query(
        "who composed opera maria golovin"
    )


def test_intermediate_answer_is_the_leading_sentence_title_prefixed():
    snippet = intermediate_answer(ranked("Maria Golovin"))
    assert snippet == "Maria Golovin is an English language opera in three acts."
    other = intermediate_answer(ranked("Vogue (magazine)"))
    assert other.startswith("Vogue (magazine): ")  # title added when absent


def test_intermediate_answer_of_an_empty_hop_is_empty():
    assert intermediate_answer([]) == ""


def test_self_ask_asks_a_followup_then_answers(stack):
    client = StubClient(
        step_reply(followup=FOLLOWUP),
        step_reply(answer="Gian Carlo Menotti", citations=["p1"]),
    )
    result = SelfAskBaseline(stack, client=client).run(QUESTION, qid="q1")

    assert [h.hop_id for h in result.hops] == ["q1", "q2"]
    assert result.hops[0].query == QUESTION
    assert result.hops[1].query == FOLLOWUP
    assert result.hops[1].intermediate_answer  # fed back into the scratchpad
    assert result.hops[0].intermediate_answer == ""  # q1 is not a follow-up
    assert result.llm_calls == 2
    assert result.answer == "Gian Carlo Menotti"
    assert result.degraded is False
    assert result.extra["stop_reason"] == "model_answered"
    assert result.extra["followups"] == [FOLLOWUP]
    assert result.metrics()["plan_depth"] == 2
    assert result.metrics()["n_subqueries"] == 2


def test_self_ask_scratchpad_reaches_the_second_call(stack):
    client = StubClient(
        step_reply(followup=FOLLOWUP),
        step_reply(answer="Gian Carlo Menotti", citations=["p1"]),
    )
    SelfAskBaseline(stack, client=client).run(QUESTION)
    second_prompt = client.calls[1]["messages"][-1]["content"]
    assert f"Follow up: {FOLLOWUP}" in second_prompt
    assert "Intermediate answer: Maria Golovin is an English" in second_prompt


def test_self_ask_that_needs_no_followup_is_naive_rag(stack):
    """The degenerate case has to be exactly one hop and one call, or the
    ``self_ask - naive_rag`` difference stops being about decomposition."""
    client = StubClient(step_reply(answer="Gian Carlo Menotti", citations=["p1"]))
    result = SelfAskBaseline(stack, client=client).run(QUESTION)
    assert len(result.hops) == 1
    assert result.llm_calls == 1
    assert result.metrics()["plan_depth"] == 1


def test_self_ask_still_retrieves_when_the_model_never_answers(stack):
    """Hop q1 is unconditional, so a generation failure cannot zero the
    retrieval columns -- see adaptation 2 in the module docstring."""
    client = StubClient(LLMError("connection refused"), repeat_last=True)
    result = SelfAskBaseline(stack, client=client).run(QUESTION)
    assert [h.hop_id for h in result.hops] == ["q1"]
    assert result.doc_ids  # a full ranking despite a dead model
    assert result.degraded is True
    assert result.answer  # deterministic fallback still answers


def test_self_ask_caps_followups_and_still_answers(stack):
    """A model that always wants one more follow-up must not run away."""
    client = StubClient(
        step_reply(followup="Who composed Maria Golovin?"),
        step_reply(followup="When was Menotti born?"),
        step_reply(followup="Where was Menotti born?"),
        step_reply(followup="What else did Menotti write?"),  # never reached
        answer_reply("Gian Carlo Menotti", ["p1"]),
        repeat_last=False,
    )
    result = SelfAskBaseline(stack, client=client).run(QUESTION)
    assert len(result.extra["followups"]) == MAX_FOLLOWUPS
    assert len(result.hops) == MAX_FOLLOWUPS + 1
    assert result.extra["stop_reason"] == "max_followups"
    assert result.llm_calls == MAX_FOLLOWUPS + 1  # 3 turns + the forced answer
    assert result.answer == "Gian Carlo Menotti"


def test_self_ask_never_exceeds_its_hard_call_budget(stack):
    """The follow-up cap and the call cap are separate hard stops: a model
    confused in a way the follow-up counter cannot see still stops."""
    client = StubClient(step_reply(followup=FOLLOWUP), repeat_last=True)
    baseline = SelfAskBaseline(stack, client=client, max_llm_calls=2, max_followups=10)
    result = baseline.run(QUESTION)
    assert len(client.calls) <= 2
    assert result.llm_calls <= 2
    assert result.answer  # the reserved call, or the fallback, produced one


def test_self_ask_breaks_a_repeated_followup_loop(stack):
    client = StubClient(
        step_reply(followup=FOLLOWUP),
        step_reply(followup="Who composed the opera Maria Golovin"),  # same question
        answer_reply("Gian Carlo Menotti", ["p1"]),
    )
    result = SelfAskBaseline(stack, client=client).run(QUESTION)
    assert result.extra["stop_reason"] == "repeated_followup"
    assert len(result.extra["followups"]) == 1  # the repeat was not issued
    assert len(result.hops) == 2
    assert result.answer == "Gian Carlo Menotti"


def test_self_ask_ignores_an_empty_followup_and_answers(stack):
    client = StubClient(
        {"needs_followup": True, "followup": "  ", "answer": "", "answer_sentence": "",
         "citations": []},
        answer_reply("Gian Carlo Menotti", ["p1"]),
    )
    result = SelfAskBaseline(stack, client=client).run(QUESTION)
    assert len(result.hops) == 1
    assert result.answer == "Gian Carlo Menotti"


def test_self_ask_fuses_its_hops_into_one_ranking(stack):
    """The retrieval metrics score one list per question, and for a multi-hop
    baseline that list has to be the RRF fusion of its hops -- not the last
    hop, which would throw away everything the first one found."""
    client = StubClient(
        step_reply(followup="Who was Gian Carlo Menotti?"),
        step_reply(answer="Gian Carlo Menotti", citations=["p1"]),
    )
    result = SelfAskBaseline(stack, client=client).run(QUESTION)
    hop_ids = {d for hop in result.hops for d in hop.doc_ids()}
    assert set(result.doc_ids) <= hop_ids
    assert len(result.doc_ids) <= 4  # truncated to top_k
    assert result.doc_ids != result.hops[-1].doc_ids()[: len(result.doc_ids)]


def test_self_ask_citations_are_resolved_against_the_pool_the_model_saw(stack):
    """The loop shows the fused pool it stops on; ``answer()`` scores against
    the same pool. A different pool at either end silently shifts every p-id."""
    client = StubClient(
        step_reply(followup=FOLLOWUP),
        step_reply(answer="Gian Carlo Menotti", citations=["p1", "p2"]),
    )
    result = SelfAskBaseline(stack, client=client).run(QUESTION)
    assert result.extra["hallucinated_citations"] == 0
    assert result.extra["citation_resolution"] == 1.0


def test_self_ask_state_does_not_leak_between_questions(stack):
    client = StubClient(
        step_reply(followup=FOLLOWUP),
        step_reply(answer="A", citations=["p1"]),
        step_reply(answer="B", citations=["p1"]),
    )
    baseline = SelfAskBaseline(stack, client=client)
    results = baseline.run_all([TrapGold("q1", QUESTION), TrapGold("q2", QUESTION)])  # type: ignore[arg-type]
    assert [len(r.hops) for r in results] == [2, 1]
    assert [r.llm_calls for r in results] == [2, 1]
    assert [r.answer for r in results] == ["A", "B"]
    assert results[1].extra["followups"] == []


def test_self_ask_terminated_by_names_the_stop_reason(stack):
    client = StubClient(step_reply(answer="X", citations=["p1"]))
    result = SelfAskBaseline(stack, client=client).run(QUESTION, qid="q1")
    record = result.to_record(run_id="r1", dataset="hotpotqa")
    assert record["terminated_by"] == "self_ask:model_answered"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_every_configured_baseline_is_constructible(stack, cfg):
    client = StubClient(answer_reply("X"), repeat_last=True)
    for name in BASELINE_NAMES:
        baseline = build_baseline(name, stack, cfg=cfg, client=client)
        assert baseline.name == name


def test_iter_baselines_yields_the_ladder_weakest_first(stack, cfg):
    client = StubClient(answer_reply("X"), repeat_last=True)
    names = [b.name for b in iter_baselines(stack, cfg=cfg, client=client)]
    assert names == list(BASELINE_NAMES)
    assert names == ["bm25_only", "dense_only", "hybrid_rerank", "naive_rag", "self_ask"]


def test_baseline_names_match_the_configured_ladder(cfg):
    configured = list(cfg.get("evaluation.configurations", []))
    assert configured[: len(BASELINE_NAMES)] == list(BASELINE_NAMES)


def test_unknown_baseline_name_is_a_keyerror(stack):
    with pytest.raises(KeyError):
        build_baseline("gpt5_only", stack)


def test_only_the_generative_baselines_declare_themselves_generative(stack, cfg):
    client = StubClient(answer_reply("X"), repeat_last=True)
    flags = {
        name: build_baseline(name, stack, cfg=cfg, client=client).generative
        for name in BASELINE_NAMES
    }
    assert flags == {
        "bm25_only": False,
        "dense_only": False,
        "hybrid_rerank": False,
        "naive_rag": True,
        "self_ask": True,
    }


def test_the_result_is_a_baseline_result(stack):
    assert isinstance(BM25OnlyBaseline(stack).run(QUESTION), BaselineResult)
