"""Hybrid retrieval: Reciprocal Rank Fusion over the sparse and dense channels.

Fusion is by **rank**, never by raw score. A BM25 score is an unbounded sum of
idf-weighted term contributions; a bge cosine lives in [-1, 1]. Normalising
them onto a common scale (min-max over the returned window, say) makes the
weighting depend on how many documents happened to come back, which is a silent
per-query rescaling. RRF ignores magnitude entirely::

    rrf(d) = sum over channels of 1 / (rrf_k + rank(d))     rank is 1-based

``rrf_k`` (60, from ``retrieval.hybrid.rrf_k``) damps the head so a single
channel's top hit cannot dominate a document both channels agree on.

The raw component scores are not thrown away -- they are carried into
``ScoredPassage.component_scores`` as ``{"bm25": ..., "dense": ..., "rrf": ...}``
plus the 1-based ranks the fusion actually used, so an error analysis can ask
which channel found a passage without re-running retrieval.

:func:`fuse_rankings` is the same operation as a pure function over doc-id
lists. The Retrieval Agent's multi-query path (a sub-query plus its planner
rewrites) reuses it to fuse one channel's results across query variants, which
is why it takes no index and no corpus.

Specification: ``docs/architecture.md`` sections 3.2 and 5.2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..types import Provenance, ScoredPassage
from .bm25_index import (
    INDEX_FORMAT,
    BM25Index,
    BM25Settings,
    CorpusLike,
    CorpusRequiredError,
    IndexFormatError,
    IndexNotBuiltError,
    RetrievalIndexError,
    materialise,
    read_json,
    write_json,
)
from .dense_index import DenseIndex, DenseSettings

__all__ = [
    "RRF_K_DEFAULT",
    "HybridIndex",
    "HybridSettings",
    "fuse_components",
    "fuse_rankings",
    "fuse_results",
    "rrf_scores",
]

RRF_K_DEFAULT = 60


# --------------------------------------------------------------------------
# Pure fusion
# --------------------------------------------------------------------------
def rrf_scores(
    rankings: Sequence[Sequence[str]], *, rrf_k: int = RRF_K_DEFAULT
) -> dict[str, float]:
    """RRF score per doc id. Rank is 1-based; absence from a ranking scores 0."""
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be non-negative, got {rrf_k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for position, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # a channel must not vote twice for one document
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + position)
    return scores


def fuse_rankings(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = RRF_K_DEFAULT,
    top_k: int | None = None,
) -> list[str]:
    """Fuse ranked doc-id lists with RRF. Pure: no index, no corpus, no I/O.

    Ties break on ``doc_id`` ascending, so the output is a deterministic
    function of the inputs. This is the function the multi-query path calls to
    fuse a sub-query with its rewrites.
    """
    scores = rrf_scores(rankings, rrf_k=rrf_k)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    doc_ids = [doc_id for doc_id, _ in ordered]
    return doc_ids if top_k is None else doc_ids[:top_k]


def fuse_components(
    components: Mapping[str, Sequence[tuple[str, float]]],
    *,
    rrf_k: int = RRF_K_DEFAULT,
    top_k: int | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuse named ``(doc_id, score)`` channels, keeping each channel's evidence.

    Returns ``(doc_id, rrf_score, component_scores)`` best first, where
    ``component_scores`` holds every channel that ranked the document (its raw
    score and its 1-based rank) plus the fused ``"rrf"`` value. A missing key
    means that channel did not return the document at all, which is itself
    worth reading in a trace.
    """
    names = sorted(components)
    fused = rrf_scores([[doc_id for doc_id, _ in components[name]] for name in names], rrf_k=rrf_k)

    per_doc: dict[str, dict[str, float]] = {}
    for name in names:
        seen: set[str] = set()
        for position, (doc_id, score) in enumerate(components[name], start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            entry = per_doc.setdefault(doc_id, {})
            entry[name] = float(score)
            entry[f"{name}_rank"] = float(position)

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    out: list[tuple[str, float, dict[str, float]]] = []
    for doc_id, score in ordered:
        entry = dict(per_doc.get(doc_id, {}))
        entry["rrf"] = float(score)
        out.append((doc_id, float(score), entry))
    return out


def fuse_results(
    results: Sequence[Sequence[ScoredPassage]],
    *,
    rrf_k: int = RRF_K_DEFAULT,
    top_k: int | None = None,
    provenance: Provenance | None = None,
) -> list[ScoredPassage]:
    """Fuse several ranked :class:`ScoredPassage` lists into one.

    The multi-query path: issue the sub-query and each rewrite, then fuse the
    result lists by rank. Component scores are unioned across the lists (the
    best value per key wins) and ``"rrf"`` is added. ``provenance`` defaults to
    whatever the inputs agree on, and to ``"hybrid"`` when they disagree --
    fusing bm25 and dense result lists really does produce hybrid evidence.
    """
    rankings = [[scored.passage.doc_id for scored in result] for result in results]
    fused = rrf_scores(rankings, rrf_k=rrf_k)

    best: dict[str, ScoredPassage] = {}
    merged: dict[str, dict[str, float]] = {}
    provenances: set[str] = set()
    for result in results:
        for scored in result:
            doc_id = scored.passage.doc_id
            provenances.add(scored.provenance)
            best.setdefault(doc_id, scored)
            entry = merged.setdefault(doc_id, {})
            for key, value in scored.component_scores.items():
                if key == "rrf":
                    continue  # a previous fusion's score is not a channel score
                entry[key] = max(entry.get(key, float(value)), float(value))

    if provenance is not None:
        resolved: Provenance = provenance
    elif len(provenances) == 1:
        resolved = provenances.pop()  # type: ignore[assignment]
    else:
        resolved = "hybrid"

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    out: list[ScoredPassage] = []
    for rank, (doc_id, score) in enumerate(ordered):
        components = dict(merged.get(doc_id, {}))
        components["rrf"] = float(score)
        out.append(
            ScoredPassage(
                passage=best[doc_id].passage,
                score=float(score),
                rank=rank,
                provenance=resolved,
                component_scores=components,
            )
        )
    return out


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HybridSettings:
    """The ``retrieval.hybrid`` block, materialised."""

    method: str = "rrf"
    rrf_k: int = RRF_K_DEFAULT
    top_k: int = 10

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> HybridSettings:
        cfg = cfg or load_config()
        return cls(
            method=str(cfg.get("retrieval.hybrid.method", "rrf")),
            rrf_k=int(cfg.get("retrieval.hybrid.rrf_k", RRF_K_DEFAULT)),
            top_k=int(cfg.get("retrieval.top_k", 10)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "rrf_k": self.rrf_k, "top_k": self.top_k}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridSettings:
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------
class HybridIndex:
    """BM25 and dense channels, fused by RRF.

    Owns the two sub-indexes so that a caller has one object to build, save,
    load and query -- and so the two channels can never be built over different
    corpus snapshots, which would make their ranks incomparable.
    """

    provenance: Provenance = "hybrid"

    def __init__(
        self,
        settings: HybridSettings,
        *,
        bm25: BM25Index | None = None,
        dense: DenseIndex | None = None,
        corpus: CorpusLike | None = None,
    ) -> None:
        if settings.method != "rrf":
            raise RetrievalIndexError(
                f"retrieval.hybrid.method={settings.method!r} is not implemented; "
                "this project ships reciprocal rank fusion only"
            )
        self.settings = settings
        self.bm25 = bm25
        self.dense = dense
        self._corpus = corpus
        if corpus is not None:
            self.attach_corpus(corpus)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        corpus: CorpusLike,
        *,
        cfg: Config | None = None,
        settings: HybridSettings | None = None,
        sparse_settings: BM25Settings | None = None,
        dense_settings: DenseSettings | None = None,
        show_progress: bool = False,
        allow_cpu_fallback: bool = False,
    ) -> HybridIndex:
        """Build both channels over one corpus."""
        cfg = cfg or load_config()
        return cls(
            settings or HybridSettings.from_config(cfg),
            bm25=BM25Index.build(
                corpus, cfg=cfg, settings=sparse_settings, show_progress=show_progress
            ),
            dense=DenseIndex.build(
                corpus,
                cfg=cfg,
                settings=dense_settings,
                show_progress=show_progress,
                allow_cpu_fallback=allow_cpu_fallback,
            ),
            corpus=corpus,
        )

    def attach_corpus(self, corpus: CorpusLike) -> HybridIndex:
        """Attach a corpus to this index and to both channels."""
        self._corpus = corpus
        if self.bm25 is not None:
            self.bm25.attach_corpus(corpus)
        if self.dense is not None:
            self.dense.attach_corpus(corpus)
        return self

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write both channels under ``path`` plus the fusion settings."""
        self._require_built()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.bm25.save(root / "bm25")
        self.dense.save(root / "dense")
        write_json(
            root / "meta.json",
            {
                "format": INDEX_FORMAT,
                "kind": "hybrid",
                "n_docs": self.bm25.n_docs,
                "settings": self.settings.to_dict(),
            },
        )
        return root

    @classmethod
    def load(cls, path: str | Path, *, corpus: CorpusLike | None = None) -> HybridIndex:
        """Load both channels. The dense channel loads in query mode (CPU)."""
        root = Path(path)
        meta = read_json(root / "meta.json")
        if int(meta.get("format", -1)) != INDEX_FORMAT:
            raise IndexFormatError(
                f"{root}: index format {meta.get('format')!r}, expected {INDEX_FORMAT}"
            )
        bm25 = BM25Index.load(root / "bm25", corpus=corpus)
        dense = DenseIndex.load(root / "dense", corpus=corpus)
        if bm25.doc_ids != dense.doc_ids:
            raise IndexFormatError(
                f"{root}: the sparse and dense channels index different documents; "
                "rebuild both from one corpus snapshot"
            )
        return cls(
            HybridSettings.from_dict(dict(meta.get("settings", {}))),
            bm25=bm25,
            dense=dense,
            corpus=corpus,
        )

    # -- search ------------------------------------------------------------
    def search_components(
        self,
        query: str,
        top_k: int | None = None,
        *,
        candidate_k: int | None = None,
    ) -> list[tuple[str, float, dict[str, float]]]:
        """``(doc_id, rrf, component_scores)`` best first. Needs no corpus."""
        return self.search_many_components([query], top_k, candidate_k=candidate_k)[0]

    def search_many_components(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        *,
        candidate_k: int | None = None,
    ) -> list[list[tuple[str, float, dict[str, float]]]]:
        """Batched :meth:`search_components`.

        ``candidate_k`` is how deep each channel is asked to go before fusion
        (default: ``top_k``). Going deeper costs almost nothing here -- both
        channels are exact -- and lets a document that one channel ranks 40th
        still be promoted by the other's agreement.
        """
        self._require_built()
        k = self.settings.top_k if top_k is None else int(top_k)
        pool = k if candidate_k is None else max(int(candidate_k), k)
        sparse_lists = self.bm25.search_many_ids(queries, pool)
        dense_lists = self.dense.search_many_ids(queries, pool)
        return [
            fuse_components(
                {"bm25": sparse_lists[i], "dense": dense_lists[i]},
                rrf_k=self.settings.rrf_k,
                top_k=k,
            )
            for i in range(len(queries))
        ]

    def search_ids(
        self, query: str, top_k: int | None = None, *, candidate_k: int | None = None
    ) -> list[tuple[str, float]]:
        """``(doc_id, rrf_score)`` best first. Needs no corpus."""
        return [
            (doc_id, score)
            for doc_id, score, _ in self.search_components(query, top_k, candidate_k=candidate_k)
        ]

    def search_many_ids(
        self, queries: Sequence[str], top_k: int | None = None, *, candidate_k: int | None = None
    ) -> list[list[tuple[str, float]]]:
        """Batched :meth:`search_ids`."""
        return [
            [(doc_id, score) for doc_id, score, _ in row]
            for row in self.search_many_components(queries, top_k, candidate_k=candidate_k)
        ]

    def search(
        self, query: str, top_k: int | None = None, *, candidate_k: int | None = None
    ) -> list[ScoredPassage]:
        """Top-``top_k`` fused passages, ``provenance="hybrid"``."""
        return self.search_many(
            [query],
            top_k,
            candidate_k=candidate_k,
        )[0]

    def search_many(
        self, queries: Sequence[str], top_k: int | None = None, *, candidate_k: int | None = None
    ) -> list[list[ScoredPassage]]:
        """Batched :meth:`search`."""
        corpus = self._require_corpus()
        out: list[list[ScoredPassage]] = []
        for row in self.search_many_components(queries, top_k, candidate_k=candidate_k):
            out.append(
                materialise(
                    corpus,
                    [(doc_id, score) for doc_id, score, _ in row],
                    provenance=self.provenance,
                    component_scores=[components for _, _, components in row],
                )
            )
        return out

    def warmup(self) -> HybridIndex:
        """Load the dense query encoder now rather than on the first question."""
        if self.dense is not None:
            self.dense.warmup()
        return self

    # -- introspection -----------------------------------------------------
    @property
    def doc_ids(self) -> tuple[str, ...]:
        self._require_built()
        return self.bm25.doc_ids

    @property
    def n_docs(self) -> int:
        self._require_built()
        return self.bm25.n_docs

    def __len__(self) -> int:
        return self.n_docs

    def __repr__(self) -> str:
        n_docs = self.bm25.n_docs if self.bm25 is not None else 0
        return f"HybridIndex(n_docs={n_docs}, method='rrf', rrf_k={self.settings.rrf_k})"

    # -- internals ---------------------------------------------------------
    def _require_built(self) -> None:
        if self.bm25 is None or self.dense is None:
            raise IndexNotBuiltError(
                "HybridIndex needs both channels; call build() or load() first"
            )

    def _require_corpus(self) -> CorpusLike:
        self._require_built()
        if self._corpus is None:
            raise CorpusRequiredError(
                "HybridIndex has no corpus attached: pass corpus= to load(), call "
                "attach_corpus(), or use search_ids() which returns doc ids only"
            )
        return self._corpus
