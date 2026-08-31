"""The baseline contract: one interface, five non-agentic systems.

Chapter 4 compares the agentic orchestrator against five systems that are not
agentic. For that comparison to mean anything, every one of them -- including
the orchestrator -- has to emit the *same* per-question object, scored by the
*same* code. That object is :class:`BaselineResult`, and this module defines it,
the :class:`Baseline` protocol that produces it, the shared retrieval stack the
baselines run on, and the one rule that turns a passage ranking into supporting
facts.

Three decisions here are load-bearing.

**1. Supporting facts come from a stated rule, not from a model.** ``sp_em`` and
``sp_f1`` are set metrics over ``(title, sent_id)`` pairs, and a retrieval-only
baseline produces no such pairs on its own -- it produces a ranking. The rule is
:func:`predict_supporting_facts`: *every sentence of the top-``SP_TOP_N`` (2)
ranked passages*. It is applied identically by all five baselines, so a
difference in ``sp_f1`` between them is a difference in where the gold paragraphs
landed in the ranking and nothing else. See that function's docstring for what
the rule costs; the number is large enough that Chapter 4 must quote the rule
beside the metric.

**2. One retrieval stack, loaded once.** A ``Corpus`` is ~66k passages and the
FAISS index is 100 MB; loading them per baseline would triple the memory and the
start-up cost of an evaluation sweep. :class:`RetrievalStack` is the shared
handle, and every baseline takes one.

**3. The result serialises into the orchestrator's trace record.**
:meth:`BaselineResult.to_record` emits exactly the keys
``trace.build_trace_record`` emits, with the agentic blocks (``plans``,
``directives``, ``verifications``, ``kg``) empty. ``eval/run_eval.py`` can
therefore write a baseline and an agentic run through the same
:class:`~agentic_ir.trace.TraceWriter`, and ``eval/error_analysis.py`` can read
both with one parser.

Specification: ``docs/architecture.md`` sections 6 and 7; ``README.md``
"Evaluation design".
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config import Config, Paths, load_config
from ..indexing.corpus import Corpus, make_doc_id
from ..indexing.hybrid import HybridIndex
from ..indexing.rerank import CrossEncoderReranker
from ..types import GoldAnswer, ScoredPassage, Source

__all__ = [
    "BASELINE_NAMES",
    "SP_TOP_N",
    "Baseline",
    "BaselineBase",
    "BaselineHop",
    "BaselineResult",
    "RetrievalStack",
    "SupportingFact",
    "build_baseline",
    "gold_doc_ids",
    "index_dir",
    "predict_supporting_facts",
]

#: A supporting fact is a ``(paragraph title, sentence index)`` pair. Same shape
#: as ``GoldAnswer.supporting_facts`` and as ``eval.metrics.SupportingFact``;
#: redeclared here so the system under test does not import its own scorer.
SupportingFact = tuple[str, int]

#: How many top-ranked passages contribute supporting facts. Two, because both
#: benchmarks' gold evidence is drawn from exactly two paragraphs per question.
SP_TOP_N = 2

#: The five non-agentic configurations named in ``config.yaml``
#: (``evaluation.configurations``) and in the README's evaluation ladder,
#: weakest first.
BASELINE_NAMES: tuple[str, ...] = (
    "bm25_only", "dense_only", "hybrid_rerank", "naive_rag", "self_ask",
)

#: Trace schema version. Must match ``trace.SCHEMA_VERSION`` or a mixed run
#: directory would hold two record shapes under one version label.
SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Supporting-fact prediction
# ---------------------------------------------------------------------------

def predict_supporting_facts(
    passages: Sequence[ScoredPassage],
    *,
    top_n: int = SP_TOP_N,
    max_sentences: int | None = None,
) -> tuple[SupportingFact, ...]:
    """Every sentence of the top-``top_n`` ranked passages, in ranking order.

    **This rule materially determines ``sp_em`` and ``sp_f1``, so Chapter 4 must
    state it verbatim beside those columns.** A retrieval system ranks
    paragraphs; the supporting-fact metrics score sentences. Something has to
    bridge the two, and the bridge is a choice, not a measurement.

    Why this particular rule:

    * **Two passages, because the gold is two paragraphs.** Every HotpotQA and
      2Wiki question draws its supporting facts from exactly two titles. Taking
      the top two ranked passages makes the *recall* ceiling of the rule equal to
      the retrieval system's Recall@2, which is the honest thing to compare --
      a deeper cut would buy recall the retriever did not earn.
    * **Every sentence, because nothing here can choose between them.** A
      retrieval-only baseline has no sentence-level signal at all. Picking (say)
      only sentence 0 would encode a dataset artefact -- gold facts skew heavily
      to the leading sentence of a Wikipedia paragraph -- and would flatter the
      baselines for a reason that has nothing to do with retrieval.

    The cost is precision, and it is not small: HotpotQA paragraphs average
    ~4-5 sentences, so this predicts ~9 facts where the gold has ~2.4.
    ``sp_precision`` is therefore capped around 0.25 and ``sp_em`` is very
    nearly always 0 -- EM demands set equality, and an over-generating rule can
    never achieve it. Both numbers are *real* results for a system that does not
    select sentences; they are not a bug to be tuned away, and reporting them
    beside the agentic system's sentence-level ``Evidence`` is precisely the
    comparison the supporting-fact metrics exist to make.

    ``max_sentences`` truncates each passage's contribution (``None`` = all). It
    exists for the sensitivity check Chapter 4 may want -- "how much of the
    precision gap is the rule?" -- and is not used by any shipped baseline.
    """
    facts: list[SupportingFact] = []
    seen: set[SupportingFact] = set()
    for scored in passages[: max(0, int(top_n))]:
        sentences = scored.passage.sentences
        limit = len(sentences) if max_sentences is None else min(len(sentences), max_sentences)
        for sent_id in range(limit):
            fact = (scored.passage.title, sent_id)
            if fact not in seen:  # two ranks of one title must not double-count
                seen.add(fact)
                facts.append(fact)
    return tuple(facts)


def facts_from_passages(
    passages: Iterable[ScoredPassage],
    doc_ids: Iterable[str],
) -> tuple[SupportingFact, ...]:
    """Every sentence of the named passages, ranking order preserved.

    The citation-driven variant: a generative baseline that cites passage ids
    can name its own supporting evidence instead of inheriting the top-2 rule.
    Off by default (see :class:`BaselineBase`) because it would make ``sp_f1``
    incomparable between the retrieval-only and the generative baselines.
    """
    wanted = set(doc_ids)
    facts: list[SupportingFact] = []
    seen: set[SupportingFact] = set()
    for scored in passages:
        if scored.passage.doc_id not in wanted:
            continue
        for sent_id in range(len(scored.passage.sentences)):
            fact = (scored.passage.title, sent_id)
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
    return tuple(facts)


def gold_doc_ids(gold: GoldAnswer) -> tuple[str, ...]:
    """The relevant doc ids for one question, derived from its gold titles.

    ``doc_id`` is a pure function of ``(source, title)``
    (:func:`~agentic_ir.indexing.corpus.make_doc_id`), so the qrels for the
    retrieval metrics need no corpus lookup. Sorted, deduplicated: two gold
    facts in one paragraph are one relevant document.
    """
    return tuple(sorted({make_doc_id(gold.dataset, title) for title, _ in gold.supporting_facts}))


# ---------------------------------------------------------------------------
# Retrieval stack
# ---------------------------------------------------------------------------

def index_dir(
    dataset: Source,
    *,
    cfg: Config | None = None,
    root: Path | None = None,
) -> Path:
    """``{paths.indexes}/{dataset}/hybrid`` -- where ``build_indexes.py`` writes.

    Both channels live under one directory because :class:`HybridIndex` refuses
    to load a sparse and a dense index that disagree on their doc-id ordering,
    and keeping them together is what makes that check meaningful.
    """
    base = Path(root) if root is not None else Paths.from_config(cfg or load_config()).indexes
    return base / dataset / "hybrid"


@dataclass
class RetrievalStack:
    """Corpus, hybrid index and cross-encoder, loaded once and shared.

    Every baseline reads its channels from here rather than loading its own, so
    an evaluation sweep pays the ~100 MB FAISS read and the corpus parse exactly
    once. It is also the seam the tests use: any object with the same three
    attributes works, which is what lets the whole suite run offline against
    stubs.
    """

    corpus: Corpus
    hybrid: HybridIndex
    reranker: CrossEncoderReranker | None = None
    top_k: int = 10
    candidate_k: int = 50
    dataset: Source | None = None

    @classmethod
    def load(
        cls,
        dataset: Source,
        *,
        cfg: Config | None = None,
        root: Path | None = None,
        with_reranker: bool = True,
        warmup: bool = False,
    ) -> RetrievalStack:
        """Load ``{dataset}``'s corpus and hybrid index from disk.

        The dense channel loads in *query* mode, which pins its encoder to
        ``retrieval.dense.query_device`` (cpu). That is not a preference: the
        GPU holds ``qwen3:8b``, and an encoder co-resident on this 8 GiB card
        OOMs partway through a 250-question run
        (``docs/environment-validation.md`` section 6).

        ``warmup`` loads the encoders now rather than inside the first question,
        so a reported per-question latency is not contaminated by a one-off
        ~2 s model load.
        """
        cfg = cfg or load_config()
        corpus = Corpus.load(dataset, cfg=cfg)
        path = index_dir(dataset, cfg=cfg, root=root)
        if not path.exists():
            raise FileNotFoundError(
                f"no index at {path} -- run: python scripts/build_indexes.py --dataset {dataset}"
            )
        hybrid = HybridIndex.load(path, corpus=corpus)
        reranker = CrossEncoderReranker.from_config(cfg) if with_reranker else None
        stack = cls(
            corpus=corpus,
            hybrid=hybrid,
            reranker=reranker,
            top_k=int(cfg.get("retrieval.top_k", 10)),
            candidate_k=int(cfg.get("retrieval.rerank.top_n", 50)),
            dataset=dataset,
        )
        if warmup:
            stack.warmup()
        return stack

    # -- channels ----------------------------------------------------------
    @property
    def bm25(self) -> Any:
        return self.hybrid.bm25

    @property
    def dense(self) -> Any:
        return self.hybrid.dense

    def warmup(self) -> RetrievalStack:
        """Load the dense query encoder and the cross-encoder now."""
        self.hybrid.warmup()
        if self.reranker is not None:
            self.reranker.warmup()
        return self

    def __repr__(self) -> str:
        return (
            f"RetrievalStack(dataset={self.dataset!r}, passages={len(self.corpus)}, "
            f"rerank={self.reranker is not None})"
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BaselineHop:
    """One retrieval round: the query issued and what came back.

    Retrieval-only baselines have exactly one hop. ``self_ask`` has one per
    follow-up plus hop 0 for the question itself. Naming them ``q1``, ``q2``,
    ... matches the sub-query ids in the agentic trace's ``retrieved`` block, so
    ``error_analysis.py`` walks both with one loop.
    """

    hop_id: str
    query: str
    tool: str
    passages: tuple[ScoredPassage, ...]
    rerank_applied: bool = False
    latency_s: float = 0.0
    #: The snippet ``self_ask`` fed back into its scratchpad; "" for the rest.
    intermediate_answer: str = ""

    def doc_ids(self) -> tuple[str, ...]:
        return tuple(sp.passage.doc_id for sp in self.passages)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One question's output from one baseline. The unit Chapter 4 scores.

    ``answer is None`` distinguishes a *retrieval-only* baseline, which does not
    attempt an answer, from a generative one that tried and produced nothing
    (``answer == ""``). Scoring an unattempted answer as EM=0 would be arithmetic
    on a quantity the system never claimed; the eval harness should skip the
    answer columns entirely for a ``None``.
    """

    qid: str
    question: str
    name: str
    doc_ids: tuple[str, ...] = ()
    supporting_facts: tuple[SupportingFact, ...] = ()
    answer: str | None = None
    answer_sentence: str = ""
    citations: tuple[str, ...] = ()
    hops: tuple[BaselineHop, ...] = ()
    llm_calls: int = 0
    tool_calls: int = 0
    latency_s: float = 0.0
    llm_latency_s: float = 0.0
    retrieval_latency_s: float = 0.0
    degraded: bool = False
    errors: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return bool(self.answer)

    @property
    def passages(self) -> tuple[ScoredPassage, ...]:
        """The final ranked pool, i.e. the last hop's. Empty when nothing ran."""
        return self.hops[-1].passages if self.hops else ()

    # -- serialisation -----------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        """The ``metrics`` block, keyed exactly as ``QuestionState.metrics()``.

        Agentic-only fields are present and zero rather than absent: a baseline
        genuinely made zero re-plans, and ``metrics.csv`` needs one stable
        column set across every configuration in the run directory.
        """
        return {
            "llm_calls": self.llm_calls,
            "llm_calls_saved": 0,
            "llm_cache_hits": 0,
            "tool_calls": self.tool_calls,
            "parse_failures": int(self.extra.get("parse_failures", 0)),
            "latency_s": round(self.latency_s, 4),
            "llm_latency_s": round(self.llm_latency_s, 4),
            "retrieval_latency_s": round(self.retrieval_latency_s, 4),
            "kg_latency_s": 0.0,
            "nli_latency_s": 0.0,
            "rerank_skipped": int(self.extra.get("rerank_skipped", 0)),
            "plan_depth": int(self.extra.get("plan_depth", 1)),
            "n_subqueries": len(self.hops),
            "cycles": 1,
            "replans": 0,
            "replanned": False,
            "best_cycle": 0,
            # The agentic ``citation_grounding`` is claim-level and NLI-backed.
            # A baseline has neither claims nor NLI, so it leaves that column
            # None and reports the weaker quantity it *can* measure -- the
            # share of emitted citations that resolve to a shown passage --
            # under its own name. Two different numbers, two columns.
            "citation_grounding": self.extra.get("citation_grounding"),
            "citation_resolution": self.extra.get("citation_resolution"),
            "nli_support": None,
            "hallucinated_citations": int(self.extra.get("hallucinated_citations", 0)),
            "answered": self.answered,
            "degraded_steps": int(self.degraded),
            "budget_exhausted": bool(self.extra.get("budget_exhausted", False)),
            "n_evidence": len(self.supporting_facts),
            "n_results": len(self.hops),
        }

    def retrieved_block(self, *, top_k: int = 10) -> dict[str, Any]:
        """The ``retrieved`` block of the trace record, one entry per hop.

        Passage *text* is omitted for the same reason the agentic writer omits
        it: error analysis needs ids, titles and scores, and full paragraphs
        would multiply the trace size by roughly forty for no analytical gain.
        """
        block: dict[str, Any] = {}
        for hop in self.hops:
            block[hop.hop_id] = {
                "query_text": hop.query,
                "queries_issued": [hop.query],
                "tool": hop.tool,
                "selector": "fallback",  # a baseline routes by construction
                "rule_id": None,
                "rerank_applied": hop.rerank_applied,
                "reason": f"{self.name} fixed pipeline",
                "features": {},
                "n_candidates": len(hop.passages),
                "latency_s": round(hop.latency_s, 4),
                "degraded": False,
                "error": None,
                "intermediate_answer": hop.intermediate_answer,
                "passages": [
                    {
                        "doc_id": sp.passage.doc_id,
                        "title": sp.passage.title,
                        "rank": sp.rank,
                        "score": round(float(sp.score), 6),
                        "provenance": sp.provenance,
                        "component_scores": {
                            k: round(float(v), 6) for k, v in sp.component_scores.items()
                        },
                    }
                    for sp in hop.passages[:top_k]
                ],
            }
        return block

    def to_record(
        self,
        *,
        run_id: str,
        dataset: str,
        gold: GoldAnswer | None = None,
        seed: int = 42,
        model: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """A trace record with the same keys ``build_trace_record`` produces.

        The agentic-only blocks are empty lists, not missing keys. That is what
        lets ``TraceWriter.append`` write a baseline and an agentic run into one
        directory and ``metrics_row`` flatten both without a schema branch.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "config_name": self.name,
            "dataset": dataset,
            "qid": self.qid,
            "seed": seed,
            "model": model,
            "question": self.question,
            "gold": None if gold is None else _gold_json(gold),
            "final_answer": self.answer or "",
            "answer_sentence": self.answer_sentence,
            "citations": list(self.citations),
            "confidence": 0.0,
            "verdict": None,
            "terminated_by": self.extra.get("terminated_by", self.name),
            "best_cycle": 0,
            "metrics": self.metrics(),
            "transitions": [],
            "plans": [],
            "directives": [],
            "steps": [],
            "retrieved": self.retrieved_block(top_k=top_k),
            "kg": {},
            "evidence": [],
            "candidates": [],
            "verifications": [],
            "supporting_facts": [[t, i] for t, i in self.supporting_facts],
            "errors": list(self.errors),
        }


def _gold_json(gold: GoldAnswer) -> dict[str, Any]:
    """Trace-shaped view of the answer key. Written, never read by a baseline."""
    return {
        "qid": gold.qid,
        "answer": gold.answer,
        "supporting_facts": [[t, i] for t, i in gold.supporting_facts],
        "level": gold.level,
        "type": gold.qtype,
    }


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

@runtime_checkable
class Baseline(Protocol):
    """A comparison system: a name, one question in, one result out.

    The agentic orchestrator satisfies the same protocol through a thin adapter
    in ``eval/run_eval.py``, which is the whole point -- the harness should not
    know which side of the comparison it is running.
    """

    name: str

    def run(self, question: str, top_k: int | None = None, *, qid: str = "") -> BaselineResult: ...

    def run_all(
        self, eval_set: Sequence[GoldAnswer], *, top_k: int | None = None
    ) -> list[BaselineResult]: ...


class BaselineBase:
    """Shared machinery: timing, the supporting-fact rule, and ``run_all``.

    Subclasses implement :meth:`retrieve` (and, for the generative ones,
    :meth:`answer`). Nothing here raises: a baseline that dies on question 194
    of 250 costs hours, so a failed question returns a degraded, empty result
    with the error recorded, exactly as the agentic system's ``step()`` does.
    """

    #: Overridden by each subclass; must match a name in ``BASELINE_NAMES``.
    name: str = "baseline"

    #: Retrieval-only baselines never call the model. Flipped by the two that do.
    generative: bool = False

    def __init__(
        self,
        stack: RetrievalStack,
        *,
        cfg: Config | None = None,
        top_k: int | None = None,
        sp_top_n: int = SP_TOP_N,
        sp_from_citations: bool = False,
    ) -> None:
        self.cfg = cfg or load_config()
        self.stack = stack
        self.top_k = int(top_k if top_k is not None else stack.top_k)
        self.sp_top_n = int(sp_top_n)
        #: Derive supporting facts from the model's own citations instead of the
        #: top-2 rule. Off by default: it would make ``sp_f1`` incomparable
        #: between the generative and the retrieval-only baselines, which is the
        #: one comparison the supporting-fact columns are there to support.
        self.sp_from_citations = bool(sp_from_citations)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, top_k={self.top_k})"

    # -- subclass hooks ----------------------------------------------------
    def _begin(self) -> None:
        """Reset per-question state before retrieval starts.

        The seam a generative baseline needs: ``self_ask`` spends LLM calls
        inside :meth:`retrieve` as well as inside :meth:`answer`, so its budget
        counter cannot live in either one.
        """

    def retrieve(self, question: str, top_k: int) -> list[BaselineHop]:
        """Run this baseline's retrieval and return its hops, in issue order."""
        raise NotImplementedError

    def answer(
        self, question: str, hops: Sequence[BaselineHop], passages: Sequence[ScoredPassage]
    ) -> dict[str, Any]:
        """Produce ``{answer, answer_sentence, citations, llm_calls, ...}``.

        Retrieval-only baselines do not override this; ``answer=None`` is what
        marks a result as having made no answer claim at all.
        """
        return {}

    # -- the contract ------------------------------------------------------
    def run(self, question: str, top_k: int | None = None, *, qid: str = "") -> BaselineResult:
        """Answer one question. Never raises."""
        k = int(top_k if top_k is not None else self.top_k)
        started = time.perf_counter()
        self._begin()
        try:
            hops = self.retrieve(question, k)
        except Exception as exc:  # noqa: BLE001 - a sweep must not die mid-run
            return BaselineResult(
                qid=qid,
                question=question,
                name=self.name,
                answer="" if self.generative else None,
                latency_s=time.perf_counter() - started,
                degraded=True,
                errors=(f"{self.name}.retrieve: {type(exc).__name__}: {exc}",),
            )

        ranked = self.fuse(hops, k)
        retrieval_latency = sum(hop.latency_s for hop in hops)
        errors: list[str] = []
        payload: dict[str, Any] = {}
        if self.generative:
            try:
                payload = self.answer(question, hops, ranked) or {}
            except Exception as exc:  # noqa: BLE001 - same reason
                errors.append(f"{self.name}.answer: {type(exc).__name__}: {exc}")
                payload = {"answer": "", "degraded": True}
            errors.extend(payload.get("errors", ()))

        facts = self.supporting_facts(ranked, payload)
        return BaselineResult(
            qid=qid,
            question=question,
            name=self.name,
            doc_ids=tuple(sp.passage.doc_id for sp in ranked),
            supporting_facts=facts,
            answer=payload.get("answer") if self.generative else None,
            answer_sentence=str(payload.get("answer_sentence", "")),
            citations=tuple(payload.get("citations", ())),
            hops=tuple(hops),
            llm_calls=int(payload.get("llm_calls", 0)),
            tool_calls=sum(1 + int(hop.rerank_applied) for hop in hops),
            latency_s=time.perf_counter() - started,
            llm_latency_s=float(payload.get("llm_latency_s", 0.0)),
            retrieval_latency_s=retrieval_latency,
            degraded=bool(payload.get("degraded", False)),
            errors=tuple(errors),
            extra=dict(payload.get("extra", {})),
        )

    def run_all(
        self,
        eval_set: Sequence[GoldAnswer],
        *,
        top_k: int | None = None,
        progress: bool = False,
    ) -> list[BaselineResult]:
        """Run the whole slice, in the order given. Gold is passed for the qid only.

        ``GoldAnswer`` is the evaluation slice's record type, so it is what the
        harness has to hand, and it carries the answer key. Only ``qid`` and
        ``question`` are read. ``tests/test_baselines.py`` proves it by handing
        ``run_all`` a record whose ``answer`` and ``supporting_facts`` raise on
        access: a baseline that peeked would invalidate the very comparison it
        exists to provide.
        """
        items: Iterable[GoldAnswer] = eval_set
        if progress:
            from tqdm import tqdm

            items = tqdm(eval_set, desc=self.name, unit="q")
        return [self.run(g.question, top_k, qid=g.qid) for g in items]

    def __call__(self, question: str, top_k: int | None = None, *, qid: str = "") -> BaselineResult:
        return self.run(question, top_k, qid=qid)

    # -- shared steps ------------------------------------------------------
    def fuse(self, hops: Sequence[BaselineHop], top_k: int) -> list[ScoredPassage]:
        """The single ranking the retrieval metrics score.

        One hop: its own ranking, unchanged. Several hops (``self_ask``): fused
        by RRF over the per-hop rankings, which is rank-based and therefore
        immune to the fact that a BM25 score, a cosine and a cross-encoder logit
        do not live on a common scale. Ties break on ``doc_id`` ascending, so
        the fused order is a deterministic function of the hops.
        """
        if not hops:
            return []
        if len(hops) == 1:
            return list(hops[0].passages[:top_k])

        from ..indexing.hybrid import fuse_results

        return fuse_results(
            [hop.passages for hop in hops],
            rrf_k=int(self.cfg.get("retrieval.hybrid.rrf_k", 60)),
            top_k=top_k,
            provenance="hybrid",
        )

    def supporting_facts(
        self, ranked: Sequence[ScoredPassage], payload: dict[str, Any]
    ) -> tuple[SupportingFact, ...]:
        """Apply the documented rule (see :func:`predict_supporting_facts`)."""
        if self.sp_from_citations:
            cited = payload.get("cited_doc_ids") or ()
            if cited:
                return facts_from_passages(ranked, cited)
        return predict_supporting_facts(ranked, top_n=self.sp_top_n)

    # -- retrieval helpers used by more than one subclass -------------------
    def _rerank(
        self, query: str, passages: Sequence[ScoredPassage], *, force: bool = True
    ) -> tuple[list[ScoredPassage], bool]:
        """Cross-encoder pass over ``passages``; returns ``(ranked, ran)``.

        ``force=True`` by default. The margin gate
        (``agents.retriever.rerank_margin_gate``) is an *efficiency* feature of
        the agentic system, and letting it skip work inside the baseline the
        agentic system must beat would hand the comparison a head start. The
        strong baseline always reranks; the gate's cost in nDCG is then a
        measurable property of the agentic system rather than a hidden subsidy.
        """
        reranker = self.stack.reranker
        if reranker is None or not passages:
            return list(passages), False
        outcome = reranker.rerank(query, list(passages), force=force)
        return list(outcome.passages), bool(outcome.ran)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_baseline(
    name: str,
    stack: RetrievalStack,
    *,
    cfg: Config | None = None,
    client: Any = None,
    **kwargs: Any,
) -> BaselineBase:
    """Construct a baseline by its ``config.yaml`` name.

    Imports are deferred to the call so that ``base`` stays importable from the
    modules that import it, and so that a harness listing the available names
    does not drag in five modules it will not use.
    """
    if name not in BASELINE_NAMES:
        raise KeyError(f"unknown baseline {name!r}; have {BASELINE_NAMES}")
    module = importlib.import_module(f".{name}", __package__)
    factory = getattr(module, _CLASS_NAMES[name])
    if name in ("naive_rag", "self_ask"):
        return factory(stack, cfg=cfg, client=client, **kwargs)
    return factory(stack, cfg=cfg, **kwargs)


_CLASS_NAMES: dict[str, str] = {
    "bm25_only": "BM25OnlyBaseline",
    "dense_only": "DenseOnlyBaseline",
    "hybrid_rerank": "HybridRerankBaseline",
    "naive_rag": "NaiveRAGBaseline",
    "self_ask": "SelfAskBaseline",
}


def iter_baselines(
    stack: RetrievalStack, *, cfg: Config | None = None, client: Any = None
) -> Iterator[BaselineBase]:
    """Every baseline, weakest first -- the ladder the README describes."""
    for name in BASELINE_NAMES:
        yield build_baseline(name, stack, cfg=cfg, client=client)
