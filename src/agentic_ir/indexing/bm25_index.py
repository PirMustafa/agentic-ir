"""Sparse retrieval: BM25 over the passage corpus.

Backed by ``bm25s`` (a NumPy sparse scorer -- no Java, no server) with Snowball
stemming from ``PyStemmer``. ``k1`` and ``b`` come from ``retrieval.sparse``,
and the values actually used are persisted with the index, so a query-time
config edit can never silently re-score an index built under other parameters.

Two things beyond plain search live here, because the Retrieval Agent's routing
table (``docs/architecture.md`` section 5.1) needs them at query time and
nothing else can supply them:

* :meth:`BM25Index.lexicon` -- the indexed term set, for rule **R2**'s
  ``oov_rate``;
* :meth:`BM25Index.doc_freq` -- per-term document frequency, for rule **R4**'s
  ``min_df``.

Both are stated over the *analysed* (stopword-filtered, stemmed) vocabulary,
which is why :meth:`BM25Index.analyze` is public. A caller that splits a query
on whitespace and probes the lexicon with raw surface forms measures an
``oov_rate`` near 1.0 on every query and routes everything to ``dense_search``.
Use the analyser, or the :meth:`BM25Index.oov_rate` / :meth:`BM25Index.min_df`
wrappers that apply it for you.

This module is also the bottom of the ``indexing`` dependency order
(bm25_index -> dense_index -> hybrid -> rerank), so the pieces shared by all
four -- the :class:`CorpusLike` protocol, :func:`document_text`,
:func:`materialise` and the error types -- are defined here rather than in a
fifth module.

``bm25s``, ``PyStemmer`` and ``numpy`` are imported lazily inside functions, so
importing this module is free and never fails on a machine where the retrieval
extras are not installed.

Specification: ``docs/architecture.md`` sections 3.2 and 5.1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..config import Config, load_config
from ..types import Passage, Provenance, ScoredPassage

__all__ = [
    "INDEX_FORMAT",
    "BM25Index",
    "BM25Settings",
    "CorpusLike",
    "CorpusRequiredError",
    "IndexFormatError",
    "IndexNotBuiltError",
    "RetrievalIndexError",
    "corpus_columns",
    "document_text",
    "materialise",
    "rank_pairs",
]

#: On-disk layout version. Bump when ``save``/``load`` stop being compatible.
INDEX_FORMAT = 1


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class RetrievalIndexError(RuntimeError):
    """Base class for every failure raised by the indexing package."""


class IndexNotBuiltError(RetrievalIndexError):
    """``search`` was called before ``build`` or ``load``."""


class CorpusRequiredError(RetrievalIndexError):
    """A ``Passage``-returning call was made on an index with no corpus.

    Indexes persist their doc-id ordering, so ``load`` needs no corpus to
    answer :meth:`BM25Index.search_ids`. Materialising :class:`ScoredPassage`
    objects does need one -- attach it at load time or via ``attach_corpus``.
    """


class IndexFormatError(RetrievalIndexError):
    """The on-disk index is missing, truncated, or written by another version."""


# --------------------------------------------------------------------------
# Corpus protocol
# --------------------------------------------------------------------------
class CorpusLike(Protocol):
    """The slice of ``indexing.corpus.Corpus`` the indexes actually use.

    Deliberately narrower than the real class (which also offers ``by_title``,
    ``titles`` and ``sentence``): a structural type should constrain only what
    is called, so a test double stays a three-method object.
    """

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Passage]: ...

    def get(self, doc_id: str) -> Passage: ...


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def document_text(passage: Passage, *, index_title: bool = True) -> str:
    """The text a scorer sees for ``passage``.

    One definition for all three scorers (BM25, bge, cross-encoder): a passage
    that is title-prefixed in one channel and not in another produces rankings
    that are not comparable, and RRF would fuse the mismatch silently. HotpotQA
    and 2Wiki titles are entity names and carry real signal, so they are
    prepended by default.
    """
    title = passage.title.strip()
    if index_title and title:
        return f"{title}. {passage.text}"
    return passage.text


def rank_pairs(pairs: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sort ``(doc_id, score)`` by ``(-score, doc_id)``.

    Every ranking in this package goes through here. A bare sort on score
    leaves float ties in whatever order the backend happened to emit, which is
    exactly the run-to-run drift that makes a reported nDCG irreproducible.
    """
    return sorted(pairs, key=lambda item: (-float(item[1]), item[0]))


def materialise(
    corpus: CorpusLike,
    pairs: Sequence[tuple[str, float]],
    *,
    provenance: Provenance,
    component_key: str | None = None,
    component_scores: Sequence[dict[str, float]] | None = None,
) -> list[ScoredPassage]:
    """Turn ``(doc_id, score)`` pairs into ranked :class:`ScoredPassage` objects.

    ``pairs`` must already be in final order; ``rank`` is assigned 0-based from
    that order. Doc ids the corpus does not know are skipped rather than
    raising -- an index built against an older corpus snapshot should degrade,
    not abort a 250-question run.
    """
    out: list[ScoredPassage] = []
    for position, (doc_id, score) in enumerate(pairs):
        try:
            passage = corpus.get(doc_id)
        except KeyError:
            continue
        if passage is None:  # a corpus that returns None instead of raising
            continue
        if component_scores is not None:
            components = dict(component_scores[position])
        elif component_key is not None:
            components = {component_key: float(score)}
        else:
            components = {}
        out.append(
            ScoredPassage(
                passage=passage,
                score=float(score),
                rank=len(out),
                provenance=provenance,
                component_scores=components,
            )
        )
    return out


def corpus_columns(corpus: CorpusLike, *, index_title: bool = True) -> tuple[list[str], list[str]]:
    """``(doc_ids, texts)`` in corpus iteration order, duplicates rejected."""
    doc_ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for passage in corpus:
        if passage.doc_id in seen:
            raise ValueError(f"duplicate doc_id in corpus: {passage.doc_id!r}")
        seen.add(passage.doc_id)
        doc_ids.append(passage.doc_id)
        texts.append(document_text(passage, index_title=index_title))
    if not doc_ids:
        raise ValueError("cannot build an index over an empty corpus")
    return doc_ids, texts


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file. Windows defaults to cp1252; always be explicit."""
    if not path.exists():
        raise IndexFormatError(f"missing index file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON, non-ASCII kept verbatim (titles carry diacritics)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BM25Settings:
    """The ``retrieval.sparse`` block, materialised.

    ``stopwords``, ``method_variant`` and ``index_title`` have no key in
    ``config/config.yaml``; the defaults here are what the report should cite
    until keys are added.

    ``stopwords`` defaults to ``"english_plus"``, **not** to the ``"english"``
    that ``bm25s`` itself defaults to. Measured on a 50-passage fixture: the
    ``"english"`` list is 33 words and keeps question words and modals, so
    ``"Who won the 1500 metres in 2016?"`` analyses to ``("who", "won", ...)``
    and ``who`` -- absent from any passage -- counts as an out-of-vocabulary
    content term. Every wh-question then carries a spurious 0.12-0.25 of
    ``oov_rate`` against R2's 0.34 threshold, and a stray modal (``can``,
    df=1) becomes the corpus's rarest term and takes over R4's ``min_df``.
    ``"english_plus"`` is the 179-word NLTK list; it removed both effects and
    left top-1 accuracy unchanged (11/12 either way).
    """

    k1: float = 0.9
    b: float = 0.4
    stemmer: str | None = "english"
    stopwords: str = "english_plus"
    method: str = "lucene"
    index_title: bool = True
    top_k: int = 10

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> BM25Settings:
        cfg = cfg or load_config()
        stemmer = cfg.get("retrieval.sparse.stemmer", "english")
        return cls(
            k1=float(cfg.get("retrieval.sparse.k1", 0.9)),
            b=float(cfg.get("retrieval.sparse.b", 0.4)),
            stemmer=None if stemmer in (None, "", "none") else str(stemmer),
            stopwords=str(cfg.get("retrieval.sparse.stopwords", "english_plus")),
            method=str(cfg.get("retrieval.sparse.method_variant", "lucene")),
            index_title=bool(cfg.get("retrieval.sparse.index_title", True)),
            top_k=int(cfg.get("retrieval.top_k", 10)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "stemmer": self.stemmer,
            "stopwords": self.stopwords,
            "method": self.method,
            "index_title": self.index_title,
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Settings:
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------
class BM25Index:
    """BM25 over passages, plus the lexicon statistics tool routing needs."""

    provenance: Provenance = "bm25"

    def __init__(
        self,
        settings: BM25Settings,
        *,
        retriever: Any = None,
        doc_ids: Sequence[str] = (),
        doc_freqs: dict[str, int] | None = None,
        corpus: CorpusLike | None = None,
    ) -> None:
        self.settings = settings
        self._retriever = retriever
        self._doc_ids: tuple[str, ...] = tuple(doc_ids)
        self._doc_freqs: dict[str, int] = doc_freqs if doc_freqs is not None else {}
        self._corpus = corpus
        self._lexicon: frozenset[str] | None = None
        self._stemmer_fn: Any = None

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        corpus: CorpusLike,
        *,
        cfg: Config | None = None,
        settings: BM25Settings | None = None,
        show_progress: bool = False,
    ) -> BM25Index:
        """Tokenise, stem and index every passage in ``corpus``."""
        settings = settings or BM25Settings.from_config(cfg)
        bm25s = _import_bm25s()
        np = _import_numpy()

        doc_ids, texts = corpus_columns(corpus, index_title=settings.index_title)
        tokenized = bm25s.tokenize(
            texts,
            stopwords=settings.stopwords,
            stemmer=_stemmer_fn(settings.stemmer),
            return_ids=True,
            show_progress=show_progress,
        )
        # Snapshot before index(): bm25s appends an empty-token entry to the
        # very dict it was handed, which would otherwise land in the lexicon.
        vocab = dict(tokenized.vocab)
        frequencies = np.zeros(len(vocab), dtype=np.int64)
        for row in tokenized.ids:
            if row:
                frequencies[np.unique(np.asarray(row, dtype=np.int64))] += 1
        doc_freqs = {
            term: int(frequencies[tid]) for term, tid in vocab.items() if 0 <= tid < len(vocab)
        }

        retriever = bm25s.BM25(k1=settings.k1, b=settings.b, method=settings.method)
        retriever.index(tokenized, show_progress=show_progress)
        return cls(
            settings,
            retriever=retriever,
            doc_ids=doc_ids,
            doc_freqs=doc_freqs,
            corpus=corpus,
        )

    def attach_corpus(self, corpus: CorpusLike) -> BM25Index:
        """Attach a corpus so :meth:`search` can materialise passages."""
        self._corpus = corpus
        return self

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the index, its doc-id ordering and its lexicon to ``path``."""
        self._require_built()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(root / "bm25s"), show_progress=False)
        write_json(root / "doc_ids.json", list(self._doc_ids))
        with (root / "lexicon.tsv").open("w", encoding="utf-8", newline="\n") as fh:
            for term in sorted(self._doc_freqs):
                fh.write(f"{term}\t{self._doc_freqs[term]}\n")
        write_json(
            root / "meta.json",
            {
                "format": INDEX_FORMAT,
                "kind": "bm25",
                "n_docs": len(self._doc_ids),
                "n_terms": len(self._doc_freqs),
                "settings": self.settings.to_dict(),
            },
        )
        return root

    @classmethod
    def load(cls, path: str | Path, *, corpus: CorpusLike | None = None) -> BM25Index:
        """Load a saved index. No corpus and no re-tokenisation required."""
        bm25s = _import_bm25s()
        root = Path(path)
        meta = read_json(root / "meta.json")
        if int(meta.get("format", -1)) != INDEX_FORMAT:
            raise IndexFormatError(
                f"{root}: index format {meta.get('format')!r}, expected {INDEX_FORMAT}"
            )
        settings = BM25Settings.from_dict(dict(meta.get("settings", {})))
        doc_ids = list(read_json(root / "doc_ids.json"))
        if int(meta.get("n_docs", len(doc_ids))) != len(doc_ids):
            raise IndexFormatError(f"{root}: doc_ids.json disagrees with meta.json")

        lexicon_path = root / "lexicon.tsv"
        if not lexicon_path.exists():
            raise IndexFormatError(f"missing index file: {lexicon_path}")
        doc_freqs: dict[str, int] = {}
        with lexicon_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                term, _, count = line.rstrip("\n").partition("\t")
                if term:
                    doc_freqs[term] = int(count)

        retriever = bm25s.BM25.load(str(root / "bm25s"), load_corpus=False, show_progress=False)
        return cls(
            settings,
            retriever=retriever,
            doc_ids=doc_ids,
            doc_freqs=doc_freqs,
            corpus=corpus,
        )

    # -- search ------------------------------------------------------------
    def search_ids(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """``(doc_id, bm25_score)`` best first. Needs no corpus.

        Zero-scoring documents are dropped: under the Lucene BM25 variant every
        document that contains a query term scores strictly positive, so a zero
        means "matched nothing" and keeping it would inflate the candidate pool
        the rerank gate measures.
        """
        return self.search_many_ids([query], top_k)[0]

    def search_many_ids(
        self, queries: Sequence[str], top_k: int | None = None
    ) -> list[list[tuple[str, float]]]:
        """Batched :meth:`search_ids`; one result list per query, order preserved."""
        self._require_built()
        k = self._effective_k(top_k)
        results: list[list[tuple[str, float]]] = [[] for _ in queries]
        if k == 0 or not queries:
            return results

        analysed = [self.analyze(q) for q in queries]
        known = [[t for t in tokens if t in self._doc_freqs] for tokens in analysed]
        live = [i for i, tokens in enumerate(known) if tokens]
        if not live:
            # No query term is in the lexicon: BM25 has no signal at all, and
            # bm25s would answer with an arbitrary all-zero top-k. Empty is honest.
            return results

        retrieved = self._retriever.retrieve(
            [known[i] for i in live], k=k, show_progress=False, n_threads=0
        )
        doc_indices, scores = retrieved[0], retrieved[1]
        for row, query_index in enumerate(live):
            pairs = [
                (self._doc_ids[int(doc_indices[row][j])], float(scores[row][j]))
                for j in range(len(doc_indices[row]))
                if float(scores[row][j]) > 0.0
            ]
            results[query_index] = rank_pairs(pairs)
        return results

    def search(self, query: str, top_k: int | None = None) -> list[ScoredPassage]:
        """Top-``top_k`` passages, ``provenance="bm25"``."""
        return materialise(
            self._require_corpus(),
            self.search_ids(query, top_k),
            provenance=self.provenance,
            component_key="bm25",
        )

    def search_many(
        self, queries: Sequence[str], top_k: int | None = None
    ) -> list[list[ScoredPassage]]:
        """Batched :meth:`search`."""
        corpus = self._require_corpus()
        return [
            materialise(corpus, pairs, provenance=self.provenance, component_key="bm25")
            for pairs in self.search_many_ids(queries, top_k)
        ]

    # -- routing features (architecture 5.1, rules R2 and R4) --------------
    def analyze(self, text: str) -> tuple[str, ...]:
        """Apply the *index-time* analyser: stopword filter, then Snowball stem.

        Public because the routing features are meaningless without it: raw
        surface forms miss the stemmed lexicon almost every time.
        """
        bm25s = _import_bm25s()
        tokens = bm25s.tokenize(
            [text],
            stopwords=self.settings.stopwords,
            stemmer=self._stemmer(),
            return_ids=False,
            show_progress=False,
        )
        return tuple(tokens[0]) if tokens else ()

    def lexicon(self) -> frozenset[str]:
        """Every analysed term in the index. Feeds ``oov_rate`` (rule R2)."""
        if self._lexicon is None:
            self._lexicon = frozenset(self._doc_freqs)
        return self._lexicon

    def doc_freq(self, term: str) -> int:
        """Documents containing ``term``; 0 when unknown. Feeds ``min_df`` (R4).

        ``term`` may be a surface form or an already-analysed stem. A surface
        form that analyses to several stems (rare -- a hyphenated token, say)
        reports the rarest component, which is what R4 is asking about.
        """
        direct = self._doc_freqs.get(term)
        if direct is not None:
            return direct
        stems = self.analyze(term)
        if not stems:
            return 0
        return min(self._doc_freqs.get(stem, 0) for stem in stems)

    def oov_rate(self, text: str) -> float:
        """Share of analysed terms absent from the lexicon; 0.0 when none remain."""
        terms = self.analyze(text)
        if not terms:
            return 0.0
        lexicon = self.lexicon()
        return sum(1 for term in terms if term not in lexicon) / len(terms)

    def min_df(self, text: str) -> float:
        """Smallest document frequency among analysed terms; ``inf`` if all OOV."""
        terms = self.analyze(text)
        known = [self._doc_freqs[term] for term in terms if term in self._doc_freqs]
        return float(min(known)) if known else math.inf

    # -- introspection -----------------------------------------------------
    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self._doc_ids

    @property
    def n_docs(self) -> int:
        return len(self._doc_ids)

    @property
    def n_terms(self) -> int:
        return len(self._doc_freqs)

    def __len__(self) -> int:
        return len(self._doc_ids)

    def __repr__(self) -> str:
        return (
            f"BM25Index(n_docs={self.n_docs}, n_terms={self.n_terms}, "
            f"k1={self.settings.k1}, b={self.settings.b})"
        )

    # -- internals ---------------------------------------------------------
    def _stemmer(self) -> Any:
        if self._stemmer_fn is None:
            self._stemmer_fn = _stemmer_fn(self.settings.stemmer)
        return self._stemmer_fn

    def _effective_k(self, top_k: int | None) -> int:
        k = self.settings.top_k if top_k is None else int(top_k)
        return max(0, min(k, len(self._doc_ids)))

    def _require_built(self) -> None:
        if self._retriever is None:
            raise IndexNotBuiltError("BM25Index has no index; call build() or load() first")

    def _require_corpus(self) -> CorpusLike:
        self._require_built()
        if self._corpus is None:
            raise CorpusRequiredError(
                "BM25Index has no corpus attached: pass corpus= to load(), call "
                "attach_corpus(), or use search_ids() which returns doc ids only"
            )
        return self._corpus


# --------------------------------------------------------------------------
# Lazy imports
# --------------------------------------------------------------------------
def _import_bm25s() -> Any:
    try:
        import bm25s
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RetrievalIndexError(
            "bm25s is required for sparse retrieval: pip install bm25s"
        ) from exc
    return bm25s


def _import_numpy() -> Any:
    import numpy

    return numpy


def _stemmer_fn(name: str | None) -> Any:
    """``Stemmer.stemWords`` for ``name``, or ``None`` for no stemming.

    A ``PyStemmer`` object is not thread-safe, so one is created per index
    instance rather than shared through a module-level cache.
    """
    if not name:
        return None
    try:
        import Stemmer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RetrievalIndexError(
            "PyStemmer is required for retrieval.sparse.stemmer: pip install PyStemmer"
        ) from exc
    return Stemmer.Stemmer(name).stemWords
