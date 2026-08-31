"""``hybrid_rerank`` -- the strong baseline the agentic system has to beat.

RRF fusion of the sparse and dense channels over a 50-document candidate pool,
then a cross-encoder pass over all 50, then truncation to
``retrieval.top_k``. Zero LLM calls. This is the standard modern
retrieve-and-rerank stack, and it is deliberately the *best* non-agentic
retrieval this project can build -- not a strawman erected to be knocked over.

Three choices keep it strong, and each is a place where it would have been easy
to quietly weaken it:

**1. The margin gate is off.** ``agents.retriever.rerank_margin_gate`` skips the
cross-encoder when the fused top-1 is already 15% clear of the runner-up. That
is an efficiency feature *of the agentic system*, and letting it skip work
inside the baseline would hand the comparison a head start. Here ``force=True``:
every question gets all 50 forward passes. The gate's cost in nDCG then shows up
as a property of the agentic system, where it belongs, instead of hiding inside
its opponent.

**2. Fusion is by rank, never by score.** A BM25 score is an unbounded sum of
idf-weighted contributions; a bge cosine lives in [-1, 1]. Min-max normalising
them onto a shared scale makes the weighting depend on how many documents
happened to come back -- a silent per-query rescaling. RRF ignores magnitude
entirely.

**3. The candidate pool is the full ``retrieval.rerank.top_n`` (50), not
``top_k``.** Reranking the top 10 can only reorder what fusion already found;
reranking 50 can *promote* a document that fusion ranked 40th, which is where
most of the cross-encoder's gain actually comes from. Both channels are exact,
so going deeper costs almost nothing.

Design axiom 3 makes this the agentic system's floor as well: if the Planner LLM
fails completely, the plan degenerates to one sub-query equal to the question
routed to ``hybrid_search``+rerank, which is exactly this pipeline. The agentic
system therefore cannot score *below* this baseline for reasons of
infrastructure failure -- only for reasons of judgement. That is worth a
sentence in Chapter 4.
"""

from __future__ import annotations

import time

from ..types import ScoredPassage
from .base import BaselineBase, BaselineHop

__all__ = ["HybridRerankBaseline"]


class HybridRerankBaseline(BaselineBase):
    """RRF over BM25 + dense, then an ungated cross-encoder pass."""

    name = "hybrid_rerank"
    generative = False

    hop_id = "q1"
    tool = "hybrid_search"

    def __init__(self, *args, gated: bool = False, **kwargs) -> None:
        """``gated=True`` re-enables the margin gate, for the efficiency ablation.

        Default ``False``: see the module docstring. The flag exists so that
        "what does the gate cost?" can be measured on this exact pipeline rather
        than inferred from two systems that differ in more than one way.
        """
        super().__init__(*args, **kwargs)
        self.gated = bool(gated)

    def retrieve(self, question: str, top_k: int) -> list[BaselineHop]:
        """Fuse, rerank, truncate. One hop, one query, no rewriting."""
        started = time.perf_counter()
        passages, reranked = self.search(question, top_k)
        return [
            BaselineHop(
                hop_id=self.hop_id,
                query=question,
                tool=self.tool,
                passages=tuple(passages),
                rerank_applied=reranked,
                latency_s=time.perf_counter() - started,
            )
        ]

    def search(self, query: str, top_k: int) -> tuple[list[ScoredPassage], bool]:
        """The reusable pipeline: ``(top-k passages, rerank ran)``.

        Split out because ``naive_rag`` and ``self_ask`` retrieve with exactly
        this stack. Sharing the method rather than the numbers is what makes
        "naive_rag adds one LLM call to hybrid_rerank" literally true, so any
        difference between their retrieval columns is a bug, not a finding.
        """
        pool = max(int(self.stack.candidate_k), top_k)
        candidates = self.stack.hybrid.search(query, pool, candidate_k=pool)
        ranked, ran = self._rerank(query, candidates, force=not self.gated)
        return ranked[:top_k], ran
