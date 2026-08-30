"""Cross-encoder reranking, behind the margin gate.

``cross-encoder/ms-marco-MiniLM-L6-v2`` scores each ``(query, passage)`` pair
jointly instead of comparing two independently produced vectors, which is why
it reorders a fused candidate list better than either channel that produced it.
It is also ~50 transformer forward passes per sub-query on CPU, and this
project's whole efficiency argument is that work which cannot change the answer
should not be done.

Hence the gate (``docs/architecture.md`` section 5.2). The cross-encoder runs
only when **all** of these hold:

* ``retrieval.rerank.enabled``
* the candidate pool has at least :data:`MIN_POOL` (10) documents
* ``(s1 - s2) / s1 < agents.retriever.rerank_margin_gate`` (0.15)

The last condition is the interesting one: when the fused top-1 is already 15%
clear of the runner-up, reranking is very unlikely to change which passage wins,
so the 50 forward passes buy nothing. :func:`margin_gate` is a pure function of
the incoming scores -- it loads no model and can be unit-tested on synthetic
score lists -- and every :class:`RerankOutcome` reports whether the pass
actually ran, so ``rerank_skipped`` in the trace is a count, not an estimate.

``torch`` and ``sentence_transformers`` are imported lazily inside functions.

Specification: ``docs/architecture.md`` sections 3.2 and 5.2.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Config, load_config
from ..types import ScoredPassage
from .bm25_index import RetrievalIndexError, document_text
from .dense_index import DEFAULT_SEED, VramContentionWarning

__all__ = [
    "MIN_POOL",
    "CrossEncoderReranker",
    "GateDecision",
    "RerankOutcome",
    "RerankSettings",
    "margin_gate",
]

#: Smallest candidate pool worth a cross-encoder pass (architecture section 5.2).
MIN_POOL = 10


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GateDecision:
    """Why the reranker did or did not run, in a form the trace can record."""

    run: bool
    reason: str
    margin: float | None
    pool_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "reason": self.reason,
            "margin": None if self.margin is None else round(self.margin, 6),
            "pool_size": self.pool_size,
        }


def margin_gate(
    passages: Sequence[ScoredPassage],
    *,
    margin_threshold: float,
    enabled: bool = True,
    min_pool: int = MIN_POOL,
) -> GateDecision:
    """Decide whether a candidate pool is worth a cross-encoder pass.

    Pure: takes scores, returns a decision, touches no model. ``passages`` must
    already be in descending score order -- it is a retrieval result, so it is.

    The margin is only defined for a positive top score. RRF scores always are;
    a raw BM25 or cosine list need not be, and when it is not the gate opens
    rather than guessing, because "the top document scores zero" is precisely
    when the ordering is least trustworthy.
    """
    pool_size = len(passages)
    if not enabled:
        return GateDecision(False, "disabled", None, pool_size)
    if pool_size < 2:
        return GateDecision(False, "pool_below_minimum", None, pool_size)
    if pool_size < min_pool:
        return GateDecision(False, "pool_below_minimum", None, pool_size)

    top, runner_up = float(passages[0].score), float(passages[1].score)
    if top <= 0.0:
        return GateDecision(True, "", None, pool_size)
    margin = (top - runner_up) / top
    if margin >= margin_threshold:
        return GateDecision(False, "margin_decisive", margin, pool_size)
    return GateDecision(True, "", margin, pool_size)


# --------------------------------------------------------------------------
# Outcome
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankOutcome:
    """The reranked (or untouched) list, plus what the gate decided."""

    passages: tuple[ScoredPassage, ...]
    ran: bool
    reason: str
    margin: float | None
    pool_size: int
    n_scored: int
    latency_s: float

    @property
    def skipped(self) -> bool:
        """Feeds the ``rerank_skipped`` counter in the tool-call budget table."""
        return not self.ran

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "reason": self.reason,
            "margin": None if self.margin is None else round(self.margin, 6),
            "pool_size": self.pool_size,
            "n_scored": self.n_scored,
            "latency_s": round(self.latency_s, 4),
        }


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankSettings:
    """``retrieval.rerank`` plus ``agents.retriever.rerank_margin_gate``.

    The threshold lives under ``agents.retriever`` in the shipped config while
    the model lives under ``retrieval.rerank``; both are read here so no call
    site has to know that.
    """

    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "cpu"
    top_n: int = 50
    margin_gate: float = 0.15
    min_pool: int = MIN_POOL
    batch_size: int = 32
    index_title: bool = True
    seed: int = DEFAULT_SEED

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> RerankSettings:
        cfg = cfg or load_config()
        return cls(
            enabled=bool(cfg.get("retrieval.rerank.enabled", True)),
            model=str(cfg.get("retrieval.rerank.model", "cross-encoder/ms-marco-MiniLM-L6-v2")),
            device=str(cfg.get("retrieval.rerank.device", "cpu")),
            top_n=int(cfg.get("retrieval.rerank.top_n", 50)),
            margin_gate=float(cfg.get("agents.retriever.rerank_margin_gate", 0.15)),
            min_pool=int(cfg.get("agents.retriever.rerank_min_pool", MIN_POOL)),
            batch_size=int(cfg.get("retrieval.rerank.batch_size", 32)),
            index_title=bool(cfg.get("retrieval.sparse.index_title", True)),
            seed=int(cfg.get("project.seed", DEFAULT_SEED)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "device": self.device,
            "top_n": self.top_n,
            "margin_gate": self.margin_gate,
            "min_pool": self.min_pool,
            "batch_size": self.batch_size,
        }


# --------------------------------------------------------------------------
# Reranker
# --------------------------------------------------------------------------
class CrossEncoderReranker:
    """Gated cross-encoder reranking of a retrieved candidate pool."""

    def __init__(self, settings: RerankSettings | None = None) -> None:
        self.settings = settings or RerankSettings()
        if self.settings.device.strip().lower().startswith("cuda"):
            warnings.warn(
                f"cross-encoder placed on {self.settings.device!r}: "
                "retrieval.rerank.device should be 'cpu' so the GPU stays reserved for "
                "Ollama (docs/environment-validation.md section 6)",
                VramContentionWarning,
                stacklevel=2,
            )
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> CrossEncoderReranker:
        return cls(RerankSettings.from_config(cfg))

    # -- model lifecycle ---------------------------------------------------
    @property
    def model(self) -> Any:
        """The ``CrossEncoder``, loaded on first use on ``retrieval.rerank.device``."""
        if self._model is None:
            _seed_everything(self.settings.seed)
            sentence_transformers = _import_sentence_transformers()
            self._model = sentence_transformers.CrossEncoder(
                self.settings.model, device=self.settings.device
            )
        return self._model

    def warmup(self) -> CrossEncoderReranker:
        """Load the model now, so the first gated question is not slow."""
        self.score_pairs("warmup", ["warmup passage"])
        return self

    def release(self) -> None:
        """Drop the model (frees ~90 MB of RAM)."""
        self._model = None

    # -- scoring -----------------------------------------------------------
    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Raw cross-encoder logits for ``(query, text)`` pairs, order preserved."""
        if not texts:
            return []
        torch = _import_torch()
        with torch.inference_mode():
            scores = self.model.predict(
                [(query, text) for text in texts],
                batch_size=self.settings.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(score) for score in scores]

    # -- the public operation ----------------------------------------------
    def rerank(
        self,
        query: str,
        passages: Sequence[ScoredPassage],
        *,
        top_n: int | None = None,
        force: bool = False,
    ) -> RerankOutcome:
        """Rerank the head of ``passages``, or skip and say why.

        Only the first ``top_n`` (50) candidates are scored: the tail is kept in
        its fused order behind them, with its original scores and provenance,
        because it was never examined. Callers truncate to ``retrieval.top_k``
        (10) afterwards, so the tail almost never survives -- it is retained so
        that ``len(out.passages) == len(passages)`` always holds and no caller
        silently loses candidates.

        ``force=True`` bypasses the gate; it exists for the ablation that
        measures what the gate costs in nDCG.
        """
        started = time.perf_counter()
        limit = self.settings.top_n if top_n is None else int(top_n)
        decision = margin_gate(
            passages,
            margin_threshold=self.settings.margin_gate,
            enabled=self.settings.enabled,
            min_pool=self.settings.min_pool,
        )
        if not decision.run and not force:
            return RerankOutcome(
                passages=tuple(passages),
                ran=False,
                reason=decision.reason,
                margin=decision.margin,
                pool_size=decision.pool_size,
                n_scored=0,
                latency_s=time.perf_counter() - started,
            )

        head = list(passages[:limit])
        tail = list(passages[limit:])
        scores = self.score_pairs(
            query,
            [document_text(p.passage, index_title=self.settings.index_title) for p in head],
        )
        ordered = sorted(
            zip(head, scores, strict=True),
            key=lambda item: (-item[1], item[0].passage.doc_id),
        )

        reranked: list[ScoredPassage] = []
        for scored, ce_score in ordered:
            components = dict(scored.component_scores)
            components["ce"] = float(ce_score)
            reranked.append(
                ScoredPassage(
                    passage=scored.passage,
                    score=float(ce_score),
                    rank=len(reranked),
                    provenance="rerank",
                    component_scores=components,
                )
            )
        for scored in tail:
            reranked.append(
                ScoredPassage(
                    passage=scored.passage,
                    score=scored.score,
                    rank=len(reranked),
                    provenance=scored.provenance,
                    component_scores=dict(scored.component_scores),
                )
            )
        return RerankOutcome(
            passages=tuple(reranked),
            ran=True,
            reason="forced" if (force and not decision.run) else "",
            margin=decision.margin,
            pool_size=decision.pool_size,
            n_scored=len(head),
            latency_s=time.perf_counter() - started,
        )

    def rerank_many(
        self,
        queries: Sequence[str],
        passage_lists: Sequence[Sequence[ScoredPassage]],
        *,
        top_n: int | None = None,
        force: bool = False,
    ) -> list[RerankOutcome]:
        """Rerank several pools. The gate is evaluated per query, as usual."""
        if len(queries) != len(passage_lists):
            raise ValueError(
                f"queries and passage_lists must be the same length, "
                f"got {len(queries)} and {len(passage_lists)}"
            )
        return [
            self.rerank(query, passages, top_n=top_n, force=force)
            for query, passages in zip(queries, passage_lists, strict=True)
        ]

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model={self.settings.model!r}, "
            f"device={self.settings.device!r}, top_n={self.settings.top_n}, "
            f"margin_gate={self.settings.margin_gate})"
        )


# --------------------------------------------------------------------------
# Seeds and lazy imports
# --------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    import numpy

    numpy.random.seed(seed)
    torch = _import_torch()
    torch.manual_seed(seed)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RetrievalIndexError(
            "torch is required for reranking: see docs/environment-validation.md"
        ) from exc
    return torch


def _import_sentence_transformers() -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RetrievalIndexError(
            "sentence-transformers is required for reranking: pip install sentence-transformers"
        ) from exc
    return sentence_transformers
