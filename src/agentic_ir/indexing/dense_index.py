"""Dense retrieval: ``BAAI/bge-small-en-v1.5`` embeddings in a FAISS ``IndexFlatIP``.

Vectors are L2-normalised, so inner product *is* cosine similarity and exact
flat search needs no training, no nlist and no tuning. At the corpus sizes this
project uses (tens of thousands of passages, 384 dimensions) a flat scan is
sub-millisecond, and an approximate index would trade recall for nothing.

Two mistakes are cheap to make here and expensive to detect, so both are made
structurally impossible rather than merely documented.

**1. Device.** ``retrieval.dense.build_device`` (``cuda``) and
``retrieval.dense.query_device`` (``cpu``) are separate settings for a hard
reason: ``qwen3:8b`` occupies roughly 6.5 GiB of this machine's 8 GiB card at
query time, and an encoder co-resident on the GPU turns into an OOM in the
middle of a multi-hour evaluation. The device is therefore not a parameter any
call site can pass. It is implied by the encoder's *role*: a ``build`` encoder
always runs on ``build_device``, a ``query`` encoder always on
``query_device``, and :meth:`DenseIndex.build` releases the build encoder --
and empties the CUDA cache -- before it returns.

**2. Prefix direction.** bge wants ``retrieval.dense.query_prefix`` on queries
and *nothing* on documents; applying it backwards costs several nDCG points and
raises no error. There is consequently no general ``encode()`` in this module.
A ``build`` encoder exposes only :meth:`_RoleEncoder.encode_documents` (never
prefixes), a ``query`` encoder only :meth:`_RoleEncoder.encode_queries` (always
prefixes), and calling the other one raises :class:`EncoderRoleError`. The
prefix appears in exactly one expression in the file.

``torch``, ``faiss`` and ``sentence_transformers`` are imported lazily inside
functions, so importing this module is free.

Specification: ``docs/architecture.md`` section 3.2; VRAM budget in
``docs/environment-validation.md`` section 6.
"""

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..config import Config, load_config
from ..types import Provenance, ScoredPassage
from .bm25_index import (
    INDEX_FORMAT,
    CorpusLike,
    CorpusRequiredError,
    IndexFormatError,
    IndexNotBuiltError,
    RetrievalIndexError,
    corpus_columns,
    materialise,
    rank_pairs,
    read_json,
    write_json,
)

__all__ = [
    "DenseIndex",
    "DenseSettings",
    "DeviceUnavailableError",
    "EncoderRole",
    "EncoderRoleError",
    "VramContentionWarning",
]

EncoderRole = Literal["build", "query"]

#: Seed applied before any encoder is constructed. Matches ``project.seed``.
DEFAULT_SEED = 42


class EncoderRoleError(RetrievalIndexError):
    """A document was encoded with a query encoder, or the reverse."""


class DeviceUnavailableError(RetrievalIndexError):
    """A CUDA device was requested but ``torch.cuda.is_available()`` is False."""


class VramContentionWarning(UserWarning):
    """A query-time encoder was placed on the GPU that Ollama needs.

    Not an error -- an ablation may legitimately want it -- but it is the
    documented cause of mid-run OOM on this hardware, so it is never silent.
    """


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DenseSettings:
    """The ``retrieval.dense`` block, materialised."""

    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    build_device: str = "cuda"
    query_device: str = "cpu"
    normalize: bool = True
    batch_size: int = 128
    query_prefix: str = ""
    index_type: str = "flat_ip"
    index_title: bool = True
    top_k: int = 10
    seed: int = DEFAULT_SEED

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> DenseSettings:
        cfg = cfg or load_config()
        return cls(
            model=str(cfg.get("retrieval.dense.model", "BAAI/bge-small-en-v1.5")),
            dim=int(cfg.get("retrieval.dense.dim", 384)),
            build_device=str(cfg.get("retrieval.dense.build_device", "cuda")),
            query_device=str(cfg.get("retrieval.dense.query_device", "cpu")),
            normalize=bool(cfg.get("retrieval.dense.normalize", True)),
            batch_size=int(cfg.get("retrieval.dense.batch_size", 128)),
            query_prefix=str(cfg.get("retrieval.dense.query_prefix", "")),
            index_type=str(cfg.get("retrieval.dense.index_type", "flat_ip")),
            index_title=bool(cfg.get("retrieval.sparse.index_title", True)),
            top_k=int(cfg.get("retrieval.top_k", 10)),
            seed=int(cfg.get("project.seed", DEFAULT_SEED)),
        )

    def device_for(self, role: EncoderRole) -> str:
        """The configured device for ``role``. The only place either is read."""
        if role == "build":
            return self.build_device
        if role == "query":
            return self.query_device
        raise EncoderRoleError(f"unknown encoder role: {role!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dim": self.dim,
            "build_device": self.build_device,
            "query_device": self.query_device,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "query_prefix": self.query_prefix,
            "index_type": self.index_type,
            "index_title": self.index_title,
            "top_k": self.top_k,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenseSettings:
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


# --------------------------------------------------------------------------
# Role-bound encoder
# --------------------------------------------------------------------------
class _RoleEncoder:
    """A ``SentenceTransformer`` bound to one role, one device, one direction.

    Private by design: nothing outside this module should be able to obtain an
    encoder without also choosing the role that fixes its device and its prefix
    behaviour.
    """

    def __init__(
        self,
        role: EncoderRole,
        settings: DenseSettings,
        *,
        allow_cpu_fallback: bool = False,
    ) -> None:
        if role not in ("build", "query"):
            raise EncoderRoleError(f"unknown encoder role: {role!r}")
        self.role: EncoderRole = role
        self.settings = settings
        self.device = _resolve_device(
            settings.device_for(role), role=role, allow_cpu_fallback=allow_cpu_fallback
        )
        if role == "query" and self.device.startswith("cuda"):
            warnings.warn(
                "dense query encoder placed on "
                f"{self.device}: retrieval.dense.query_device should be 'cpu' so the "
                "GPU stays reserved for Ollama (docs/environment-validation.md section 6)",
                VramContentionWarning,
                stacklevel=3,
            )
        self._model: Any = None

    # -- model lifecycle ---------------------------------------------------
    @property
    def model(self) -> Any:
        if self._model is None:
            _seed_everything(self.settings.seed)
            sentence_transformers = _import_sentence_transformers()
            self._model = sentence_transformers.SentenceTransformer(
                self.settings.model, device=self.device
            )
            self._model.eval()
            reported = int(self._model.get_sentence_embedding_dimension())
            if reported != self.settings.dim:
                raise RetrievalIndexError(
                    f"{self.settings.model} produces {reported}-d vectors but "
                    f"retrieval.dense.dim is {self.settings.dim}"
                )
        return self._model

    def release(self) -> None:
        """Drop the model and hand its VRAM back.

        Dropping the reference is not enough. ``SentenceTransformerModelCardData``
        holds a reference back to the model, so the parameters survive both
        ``del`` and ``gc.collect()`` and ``empty_cache()`` frees nothing --
        measured: 135 MiB still resident after build. Moving the module to CPU
        first releases the parameter storage whoever else is holding the object.
        """
        model, self._model = self._model, None
        if model is None:
            return
        if not self.device.startswith("cuda"):
            return
        torch = _import_torch()
        with suppress(RuntimeError, AttributeError):
            model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # -- the only two ways to produce a vector -----------------------------
    def encode_documents(self, texts: Sequence[str], *, show_progress: bool = False) -> Any:
        """Embed passages. Never prefixed -- bge prefixes queries only."""
        if self.role != "build":
            raise EncoderRoleError(
                "encode_documents() requires a 'build' encoder (it runs on "
                f"retrieval.dense.build_device); this one has role {self.role!r}. "
                "Documents are embedded once, offline, by DenseIndex.build()."
            )
        return self._encode(list(texts), show_progress=show_progress)

    def encode_queries(self, texts: Sequence[str], *, show_progress: bool = False) -> Any:
        """Embed queries. Always prefixed with ``retrieval.dense.query_prefix``."""
        if self.role != "query":
            raise EncoderRoleError(
                "encode_queries() requires a 'query' encoder (it runs on "
                f"retrieval.dense.query_device); this one has role {self.role!r}."
            )
        prefix = self.settings.query_prefix
        return self._encode([f"{prefix}{text}" for text in texts], show_progress=show_progress)

    def _encode(self, texts: list[str], *, show_progress: bool) -> Any:
        numpy = _import_numpy()
        if not texts:
            return numpy.zeros((0, self.settings.dim), dtype=numpy.float32)
        torch = _import_torch()
        with torch.inference_mode():
            vectors = self.model.encode(
                texts,
                batch_size=self.settings.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.settings.normalize,
                show_progress_bar=show_progress,
            )
        return numpy.ascontiguousarray(vectors, dtype=numpy.float32)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------
class DenseIndex:
    """FAISS flat inner-product search over normalised bge embeddings."""

    provenance: Provenance = "dense"

    def __init__(
        self,
        settings: DenseSettings,
        *,
        faiss_index: Any = None,
        doc_ids: Sequence[str] = (),
        corpus: CorpusLike | None = None,
        build_seconds: float = 0.0,
        built_device: str = "",
    ) -> None:
        self.settings = settings
        self._index = faiss_index
        self._doc_ids: tuple[str, ...] = tuple(doc_ids)
        self._corpus = corpus
        self.build_seconds = build_seconds
        self.built_device = built_device
        self._query_encoder: _RoleEncoder | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        corpus: CorpusLike,
        *,
        cfg: Config | None = None,
        settings: DenseSettings | None = None,
        show_progress: bool = False,
        allow_cpu_fallback: bool = False,
    ) -> DenseIndex:
        """Embed every passage on ``build_device`` and add them to FAISS.

        The build encoder is released before returning, so no VRAM is still
        held when the caller goes on to start Ollama. Pass
        ``allow_cpu_fallback=True`` only to make a CUDA-less machine build
        slowly instead of failing loudly.
        """
        settings = settings or DenseSettings.from_config(cfg)
        if settings.index_type != "flat_ip":
            raise RetrievalIndexError(
                f"retrieval.dense.index_type={settings.index_type!r} is not implemented; "
                "this project ships exact flat inner-product search only"
            )
        faiss = _import_faiss()

        doc_ids, texts = corpus_columns(corpus, index_title=settings.index_title)
        encoder = _RoleEncoder("build", settings, allow_cpu_fallback=allow_cpu_fallback)
        started = time.perf_counter()
        try:
            vectors = encoder.encode_documents(texts, show_progress=show_progress)
        finally:
            encoder.release()
        elapsed = time.perf_counter() - started

        index = faiss.IndexFlatIP(settings.dim)
        index.add(vectors)
        return cls(
            settings,
            faiss_index=index,
            doc_ids=doc_ids,
            corpus=corpus,
            build_seconds=elapsed,
            built_device=encoder.device,
        )

    def attach_corpus(self, corpus: CorpusLike) -> DenseIndex:
        """Attach a corpus so :meth:`search` can materialise passages."""
        self._corpus = corpus
        return self

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the FAISS index and its doc-id ordering to ``path``."""
        self._require_built()
        faiss = _import_faiss()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(root / "dense.faiss"))
        write_json(root / "doc_ids.json", list(self._doc_ids))
        write_json(
            root / "meta.json",
            {
                "format": INDEX_FORMAT,
                "kind": "dense",
                "n_docs": len(self._doc_ids),
                "build_seconds": round(self.build_seconds, 3),
                "built_device": self.built_device,
                "settings": self.settings.to_dict(),
            },
        )
        return root

    @classmethod
    def load(cls, path: str | Path, *, corpus: CorpusLike | None = None) -> DenseIndex:
        """Load a saved index for **querying**.

        No encoder is constructed here. The first :meth:`search` builds one with
        role ``query``, which pins it to ``retrieval.dense.query_device`` -- so
        an index built on the GPU cannot drag the GPU into the online loop.
        """
        faiss = _import_faiss()
        root = Path(path)
        meta = read_json(root / "meta.json")
        if int(meta.get("format", -1)) != INDEX_FORMAT:
            raise IndexFormatError(
                f"{root}: index format {meta.get('format')!r}, expected {INDEX_FORMAT}"
            )
        settings = DenseSettings.from_dict(dict(meta.get("settings", {})))
        doc_ids = list(read_json(root / "doc_ids.json"))
        index_path = root / "dense.faiss"
        if not index_path.exists():
            raise IndexFormatError(f"missing index file: {index_path}")
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(doc_ids):
            raise IndexFormatError(
                f"{root}: FAISS holds {index.ntotal} vectors but doc_ids.json has {len(doc_ids)}"
            )
        return cls(
            settings,
            faiss_index=index,
            doc_ids=doc_ids,
            corpus=corpus,
            build_seconds=float(meta.get("build_seconds", 0.0)),
            built_device=str(meta.get("built_device", "")),
        )

    # -- search ------------------------------------------------------------
    def search_ids(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """``(doc_id, cosine)`` best first. Needs no corpus."""
        return self.search_many_ids([query], top_k)[0]

    def search_many_ids(
        self, queries: Sequence[str], top_k: int | None = None
    ) -> list[list[tuple[str, float]]]:
        """Batched :meth:`search_ids`; one encoder pass for the whole batch.

        Note for anyone comparing traces: a transformer pads each batch to its
        longest member, so batching *n* queries together and encoding them one
        at a time give scores that differ in the last bits (measured: 1.2e-7
        on this model). Rankings were identical in every case tested, but a
        genuinely tied pair could order differently between the two paths. Each
        path is individually deterministic; pick one per experiment.
        """
        self._require_built()
        k = self._effective_k(top_k)
        results: list[list[tuple[str, float]]] = [[] for _ in queries]
        if k == 0 or not queries:
            return results

        vectors = self.query_encoder().encode_queries(list(queries))
        scores, indices = self._index.search(vectors, k)
        for row in range(len(queries)):
            pairs = [
                (self._doc_ids[int(indices[row][j])], float(scores[row][j]))
                for j in range(k)
                if int(indices[row][j]) >= 0
            ]
            results[row] = rank_pairs(pairs)
        return results

    def search(self, query: str, top_k: int | None = None) -> list[ScoredPassage]:
        """Top-``top_k`` passages, ``provenance="dense"``."""
        return materialise(
            self._require_corpus(),
            self.search_ids(query, top_k),
            provenance=self.provenance,
            component_key="dense",
        )

    def search_many(
        self, queries: Sequence[str], top_k: int | None = None
    ) -> list[list[ScoredPassage]]:
        """Batched :meth:`search`."""
        corpus = self._require_corpus()
        return [
            materialise(corpus, pairs, provenance=self.provenance, component_key="dense")
            for pairs in self.search_many_ids(queries, top_k)
        ]

    # -- encoder -----------------------------------------------------------
    def query_encoder(self) -> _RoleEncoder:
        """The query-role encoder, created on first use on ``query_device``."""
        if self._query_encoder is None:
            self._query_encoder = _RoleEncoder("query", self.settings)
        return self._query_encoder

    def warmup(self) -> DenseIndex:
        """Load the query encoder now, so the first real query is not slow."""
        self.query_encoder().encode_queries(["warmup"])
        return self

    def release_encoder(self) -> None:
        """Drop the query encoder (frees ~130 MB of RAM)."""
        if self._query_encoder is not None:
            self._query_encoder.release()
            self._query_encoder = None

    @property
    def query_device(self) -> str:
        """Device the query encoder runs on, resolved."""
        return self.query_encoder().device

    # -- introspection -----------------------------------------------------
    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self._doc_ids

    @property
    def n_docs(self) -> int:
        return len(self._doc_ids)

    def __len__(self) -> int:
        return len(self._doc_ids)

    def __repr__(self) -> str:
        return (
            f"DenseIndex(n_docs={self.n_docs}, model={self.settings.model!r}, "
            f"dim={self.settings.dim}, built_on={self.built_device or 'unknown'!r})"
        )

    # -- internals ---------------------------------------------------------
    def _effective_k(self, top_k: int | None) -> int:
        k = self.settings.top_k if top_k is None else int(top_k)
        return max(0, min(k, len(self._doc_ids)))

    def _require_built(self) -> None:
        if self._index is None:
            raise IndexNotBuiltError("DenseIndex has no index; call build() or load() first")

    def _require_corpus(self) -> CorpusLike:
        self._require_built()
        if self._corpus is None:
            raise CorpusRequiredError(
                "DenseIndex has no corpus attached: pass corpus= to load(), call "
                "attach_corpus(), or use search_ids() which returns doc ids only"
            )
        return self._corpus


# --------------------------------------------------------------------------
# Devices, seeds and lazy imports
# --------------------------------------------------------------------------
def _resolve_device(requested: str, *, role: str, allow_cpu_fallback: bool) -> str:
    """Validate a configured device string against the machine."""
    device = (requested or "cpu").strip().lower()
    if not device.startswith("cuda"):
        return device
    torch = _import_torch()
    if torch.cuda.is_available():
        return device
    if allow_cpu_fallback:
        warnings.warn(
            f"{role} device {device!r} is unavailable (torch.cuda.is_available() is False); "
            "falling back to CPU",
            VramContentionWarning,
            stacklevel=3,
        )
        return "cpu"
    raise DeviceUnavailableError(
        f"retrieval.dense.{role}_device is {device!r} but torch.cuda.is_available() is False. "
        "Install a CUDA build of torch (cu128 for this RTX 5060 / sm_120), or pass "
        "allow_cpu_fallback=True to build on CPU."
    )


def _seed_everything(seed: int) -> None:
    """Fix the seeds any encoder construction could touch."""
    import random

    random.seed(seed)
    numpy = _import_numpy()
    numpy.random.seed(seed)
    torch = _import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _import_numpy() -> Any:
    import numpy

    return numpy


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RetrievalIndexError(
            "torch is required for dense retrieval: see docs/environment-validation.md"
        ) from exc
    return torch


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RetrievalIndexError(
            "faiss is required for dense retrieval: pip install faiss-cpu"
        ) from exc
    return faiss


def _import_sentence_transformers() -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RetrievalIndexError(
            "sentence-transformers is required for dense retrieval: "
            "pip install sentence-transformers"
        ) from exc
    return sentence_transformers
