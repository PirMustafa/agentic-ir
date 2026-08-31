"""``bm25_only`` -- the weakest rung: sparse retrieval, nothing else.

One BM25 query over the whole corpus, top-``retrieval.top_k`` passages, zero
LLM calls, zero reranking. It exists to establish the floor: every number the
agentic system reports has to be read against what a 1994 algorithm achieves on
the same corpus with the same qrels, because on entity-heavy multi-hop
benchmarks that floor is a good deal higher than a dense-retrieval narrative
suggests.

It also answers nothing. ``BaselineResult.answer`` is ``None``, not ``""`` --
this system makes no answer claim, and scoring an unattempted answer as EM=0
would be arithmetic on a quantity that was never produced. Only the retrieval
and supporting-fact columns are defined for it.

Parameters (``k1=0.9``, ``b=0.4``, Snowball English stemming) are persisted in
the index itself, so a query-time config edit cannot silently re-score an index
built under other values.
"""

from __future__ import annotations

import time

from .base import BaselineBase, BaselineHop

__all__ = ["BM25OnlyBaseline"]


class BM25OnlyBaseline(BaselineBase):
    """Single-shot BM25. No fusion, no rerank, no model."""

    name = "bm25_only"
    generative = False

    #: The single hop's id. Matches the agentic trace's sub-query naming, so
    #: ``error_analysis.py`` walks a baseline's ``retrieved`` block and an
    #: agentic one with the same loop.
    hop_id = "q1"

    #: The tool label written into the trace. A real ``ToolName``, because the
    #: baseline really does issue exactly the tool the registry exposes.
    tool = "bm25_search"

    def retrieve(self, question: str, top_k: int) -> list[BaselineHop]:
        """The question, verbatim, as one BM25 query.

        Verbatim is the point: no rewriting, no decomposition, no expansion.
        Any of those would be a form of query planning, and query planning is
        what the agentic system is being asked to justify.
        """
        started = time.perf_counter()
        passages = tuple(self.stack.bm25.search(question, top_k))
        return [
            BaselineHop(
                hop_id=self.hop_id,
                query=question,
                tool=self.tool,
                passages=passages,
                rerank_applied=False,
                latency_s=time.perf_counter() - started,
            )
        ]
