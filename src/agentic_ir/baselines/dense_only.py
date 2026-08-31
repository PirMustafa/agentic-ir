"""``dense_only`` -- single-shot dense retrieval, nothing else.

One ``bge-small-en-v1.5`` query vector, exact inner-product search over the
FAISS flat index, top-``retrieval.top_k`` passages. Zero LLM calls, zero
reranking, no fusion with the sparse channel.

Why it is a *separate* rung rather than a footnote to ``bm25_only``: the two
channels fail differently, and the report needs that contrast to justify the
hybrid one. BM25 cannot match a paraphrase; bge cannot reliably match a rare
proper noun or a bare year. Reporting each alone is what makes "RRF is worth its
complexity" a measured claim rather than an assertion, and it is what makes the
Retrieval Agent's routing table (architecture 5.1) legible -- rules R2 and R3
route on exactly this asymmetry.

The query encoder runs on ``retrieval.dense.query_device`` (cpu). That is not a
tuning preference: ``qwen3:8b`` occupies ~6.5 GiB of this machine's 8 GiB card,
and an encoder co-resident on it turns into an OOM partway through an
evaluation. :class:`~agentic_ir.indexing.dense_index.DenseIndex` binds the
device to the encoder's *role*, so a baseline cannot get this wrong even by
trying.
"""

from __future__ import annotations

import time

from .base import BaselineBase, BaselineHop

__all__ = ["DenseOnlyBaseline"]


class DenseOnlyBaseline(BaselineBase):
    """Single-shot dense retrieval. No fusion, no rerank, no model."""

    name = "dense_only"
    generative = False

    hop_id = "q1"
    tool = "dense_search"

    def retrieve(self, question: str, top_k: int) -> list[BaselineHop]:
        """The question, verbatim, as one dense query.

        ``retrieval.dense.query_prefix`` is applied inside the index, never
        here. bge wants its instruction prefix on queries and nothing on
        documents; applying it backwards costs several nDCG points and raises no
        error, so there is exactly one expression in the codebase that adds it.
        """
        started = time.perf_counter()
        passages = tuple(self.stack.dense.search(question, top_k))
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
