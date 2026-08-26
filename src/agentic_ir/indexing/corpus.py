"""The passage corpus, the qrels, and the evaluation slices -- read side.

``scripts/build_corpus.py`` writes the corpus once and this module is the only
thing that reads it back:

* ``{dataset}_corpus.jsonl``  -- one :class:`~agentic_ir.types.Passage` per line
* ``{dataset}_qrels.jsonl``   -- one :class:`~agentic_ir.types.GoldAnswer` per line,
  for every question in the split
* ``{dataset}_eval_250.jsonl`` / ``{dataset}_calib_50.jsonl`` -- the sampled
  slices written by ``scripts/sample_eval_set.py``, same ``GoldAnswer`` schema

Two invariants make the supporting-fact metrics computable. Both are
established at build time, not here:

1. ``Passage.sentences`` is the dataset's **own** sentence split, persisted
   verbatim. Gold supporting facts are ``(title, sent_id)`` pairs indexing into
   that list, so re-splitting anywhere downstream would renumber the ground
   truth while every metric still looked plausible. Nothing in this module
   splits text.
2. ``doc_id`` is ``f"{source}:{quote(title, safe='')}"`` -- stable, URL-safe,
   and derivable from a gold title alone, which is what lets ``eval/metrics.py``
   turn ``(title, sent_id)`` qrels into document ids without a corpus lookup.

The read API is deliberately small -- :class:`Corpus`, :func:`load_qrels`,
:func:`load_eval_set`. The rest is the on-disk schema, exposed so that the
builder and the reader cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import orjson

from ..config import Config, Paths, load_config
from ..types import GoldAnswer, Passage, Source

__all__ = [
    "DATASETS",
    "DEFAULT_CALIB_SIZE",
    "Corpus",
    "EvalSplit",
    "corpus_path",
    "eval_set_path",
    "gold_from_json",
    "gold_to_json",
    "load_eval_set",
    "load_qrels",
    "make_doc_id",
    "passage_from_json",
    "passage_to_json",
    "qrels_path",
    "read_jsonl",
    "write_jsonl",
]

#: The configured datasets, in ``config.yaml`` order.
DATASETS: tuple[Source, ...] = ("hotpotqa", "twowiki")

#: Size of the calibration slice. Not a config key: it is fixed by the
#: verifier-threshold sweep (docs/architecture.md 10.4), not tuned.
DEFAULT_CALIB_SIZE = 50

EvalSplit = Literal["eval", "calib"]


# ---------------------------------------------------------------------------
# Identity and paths
# ---------------------------------------------------------------------------

def make_doc_id(source: str, title: str) -> str:
    """The stable, URL-safe document id of a titled paragraph.

    ``safe=''`` is load-bearing: the default leaves ``/`` unescaped, so a title
    like ``AC/DC`` would yield an id that breaks anything path- or URL-shaped.
    Percent-encoding everything keeps the id one opaque token.
    """
    return f"{source}:{quote(title, safe='')}"


def _processed_dir(cfg: Config | None = None) -> Path:
    return Paths.from_config(cfg or load_config()).processed


def corpus_path(dataset: Source, cfg: Config | None = None) -> Path:
    return _processed_dir(cfg) / f"{dataset}_corpus.jsonl"


def qrels_path(dataset: Source, cfg: Config | None = None) -> Path:
    return _processed_dir(cfg) / f"{dataset}_qrels.jsonl"


def eval_set_path(
    dataset: Source,
    *,
    split: EvalSplit = "eval",
    size: int | None = None,
    cfg: Config | None = None,
) -> Path:
    """Path of a sampled slice, e.g. ``hotpotqa_eval_250.jsonl``.

    The size is in the filename so that re-sampling at a different size cannot
    quietly overwrite the slice whose SHA-256 is already cited in ``meta.json``.
    """
    cfg = cfg or load_config()
    if size is None:
        size = (
            int(cfg.get(f"datasets.{dataset}.eval_sample"))
            if split == "eval"
            else DEFAULT_CALIB_SIZE
        )
    return _processed_dir(cfg) / f"{dataset}_{split}_{size}.jsonl"


# ---------------------------------------------------------------------------
# JSONL I/O
#
# orjson everywhere, binary handles everywhere. orjson emits UTF-8 bytes
# directly, which sidesteps the cp1252 default Windows applies to text handles
# -- and both datasets are full of names like "Xawery Zulawski".
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write ``records`` as JSONL. Returns the number of lines written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("wb") as fh:
        for record in records:
            fh.write(orjson.dumps(record))
            fh.write(b"\n")
            written += 1
    return written


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file. Blank lines are skipped; malformed lines raise."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/download_data.py then "
            "scripts/build_corpus.py first"
        )
    with path.open("rb") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield orjson.loads(stripped)
            except orjson.JSONDecodeError as exc:  # pragma: no cover - corrupt file
                raise ValueError(f"{path}:{lineno}: malformed JSON ({exc})") from exc


# ---------------------------------------------------------------------------
# Record <-> dataclass
# ---------------------------------------------------------------------------

def passage_to_json(passage: Passage) -> dict[str, Any]:
    return {
        "doc_id": passage.doc_id,
        "title": passage.title,
        "text": passage.text,
        "sentences": list(passage.sentences),
        "source": passage.source,
        "meta": dict(passage.meta),
    }


def passage_from_json(record: Mapping[str, Any]) -> Passage:
    return Passage(
        doc_id=record["doc_id"],
        title=record["title"],
        text=record["text"],
        sentences=tuple(record["sentences"]),
        source=record["source"],
        meta=dict(record.get("meta") or {}),
    )


def gold_to_json(gold: GoldAnswer) -> dict[str, Any]:
    return {
        "qid": gold.qid,
        "question": gold.question,
        "answer": gold.answer,
        "dataset": gold.dataset,
        "supporting_facts": [[title, sent_id] for title, sent_id in gold.supporting_facts],
        "evidence_triples": [list(triple) for triple in gold.evidence_triples],
        "level": gold.level,
        "qtype": gold.qtype,
        "context_titles": list(gold.context_titles),
    }


def gold_from_json(record: Mapping[str, Any]) -> GoldAnswer:
    return GoldAnswer(
        qid=record["qid"],
        question=record["question"],
        answer=record["answer"],
        dataset=record["dataset"],
        supporting_facts=tuple(
            (title, int(sent_id)) for title, sent_id in record.get("supporting_facts") or ()
        ),
        evidence_triples=tuple(
            (triple[0], triple[1], triple[2]) for triple in record.get("evidence_triples") or ()
        ),
        level=record.get("level"),
        qtype=record.get("qtype"),
        context_titles=tuple(record.get("context_titles") or ()),
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class Corpus:
    """An in-memory, read-only view over one dataset's passages.

    Small enough to hold whole: HotpotQA's distractor validation corpus is
    ~66k paragraphs and 2Wiki's ~120k. Loading once and sharing the instance is
    what keeps the BM25 index, the dense index and the qrels agreeing on
    sentence ids.
    """

    __slots__ = ("_by_id", "_by_title", "_passages", "dataset", "path")

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        dataset: Source | None = None,
        path: Path | None = None,
    ) -> None:
        self._passages: tuple[Passage, ...] = tuple(passages)
        self.dataset = dataset
        self.path = path
        self._by_id: dict[str, Passage] = {}
        self._by_title: dict[str, Passage] = {}
        for passage in self._passages:
            # First wins, matching the builder's dedupe rule, so that a corpus
            # assembled from more than one file still resolves deterministically.
            self._by_id.setdefault(passage.doc_id, passage)
            self._by_title.setdefault(passage.title, passage)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        dataset: Source,
        *,
        cfg: Config | None = None,
        path: Path | None = None,
        progress: bool = False,
    ) -> Corpus:
        """Load ``{dataset}_corpus.jsonl`` from ``paths.processed``."""
        target = path or corpus_path(dataset, cfg)
        records: Iterable[dict[str, Any]] = read_jsonl(target)
        if progress:
            from tqdm import tqdm

            records = tqdm(records, desc=f"load {target.name}", unit="doc")
        return cls([passage_from_json(r) for r in records], dataset=dataset, path=target)

    # -- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._passages)

    def __iter__(self) -> Iterator[Passage]:
        """Passages in file order, which the builder sorts by ``doc_id``."""
        return iter(self._passages)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._by_id

    def __repr__(self) -> str:
        return f"Corpus(dataset={self.dataset!r}, passages={len(self._passages)})"

    # -- lookup -------------------------------------------------------------

    def get(self, doc_id: str) -> Passage:
        """The passage with ``doc_id``. Raises ``KeyError`` if absent.

        A miss here means an index and a corpus are out of step -- a programmer
        error -- so it raises rather than returning ``None``. The data-driven
        lookups (:meth:`by_title`, :meth:`sentence`) return ``None`` instead,
        because a gap in the ground truth is data, not a bug.
        """
        try:
            return self._by_id[doc_id]
        except KeyError:
            raise KeyError(f"no passage {doc_id!r} in {self.path or 'corpus'}") from None

    def by_title(self, title: str) -> Passage | None:
        """The passage with exactly this title, or ``None``.

        Exact match only. Gold supporting facts cite the same title strings the
        context ships, so a fuzzy fallback here would turn a real corpus gap
        into a plausible-looking wrong sentence.
        """
        return self._by_title.get(title)

    def titles(self) -> tuple[str, ...]:
        """Every title in the corpus, sorted."""
        return tuple(sorted(self._by_title))

    def sentence(self, doc_id: str, sent_id: int) -> str | None:
        """One sentence by ``(doc_id, sent_id)``, or ``None`` if unresolvable.

        Negative ids are rejected rather than wrapping: ``sent_id`` is a
        position in the ground truth, and Python's wrap-around would answer a
        broken qrel with a real-looking sentence.
        """
        passage = self._by_id.get(doc_id)
        if passage is None or sent_id < 0 or sent_id >= len(passage.sentences):
            return None
        return passage.sentences[sent_id]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_qrels(dataset: Source, *, cfg: Config | None = None) -> dict[str, GoldAnswer]:
    """Every question in the split, keyed by qid.

    NO agent may read these. They are the answer key: gold answer, gold
    supporting facts, and -- 2Wiki only -- gold Wikidata evidence triples.
    """
    return {r["qid"]: gold_from_json(r) for r in read_jsonl(qrels_path(dataset, cfg))}


def load_eval_set(
    dataset: Source,
    *,
    split: EvalSplit = "eval",
    size: int | None = None,
    cfg: Config | None = None,
) -> tuple[GoldAnswer, ...]:
    """The sampled evaluation slice, or -- ``split="calib"`` -- the disjoint
    calibration slice that tunes ``agents.verifier.confidence_threshold``.

    Returned in qid order, as written.
    """
    path = eval_set_path(dataset, split=split, size=size, cfg=cfg)
    return tuple(gold_from_json(r) for r in read_jsonl(path))
