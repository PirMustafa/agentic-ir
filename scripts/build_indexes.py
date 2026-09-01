"""Build the BM25 and dense retrieval indexes over the processed corpus.

    python scripts/build_indexes.py --dataset all

Writes, per dataset, into ``index_dir(dataset)`` -- the one path
``baselines.base.RetrievalStack.load`` and ``eval/run_eval.py`` look in:

* ``data/indexes/{dataset}/hybrid/bm25/``   bm25s index, doc-id order, lexicon
* ``data/indexes/{dataset}/hybrid/dense/``  FAISS flat-IP index, doc-id order
* ``data/indexes/{dataset}/hybrid/meta.json``   the RRF settings
* ``data/indexes/{dataset}_index_stats.json``   the numbers Chapter 3 quotes

Three things this script exists to get right
--------------------------------------------

**1. Both channels are built from one corpus snapshot, in one process.**
:class:`~agentic_ir.indexing.hybrid.HybridIndex` refuses to load a sparse and a
dense index whose ``doc_ids.json`` disagree, because RRF fuses *ranks* and two
channels indexing different documents would fuse silently and wrongly. Building
them side by side from a single :class:`~agentic_ir.indexing.corpus.Corpus` is
what makes that check pass for the right reason rather than by luck.

**2. The build runs on ``retrieval.dense.build_device`` (cuda); querying does
not.** Encoding 55-67k passages on CPU is roughly an hour per dataset, so the
build wants the GPU. The *query* encoder is pinned to
``retrieval.dense.query_device`` (cpu) by :meth:`DenseIndex.load`, and
:meth:`DenseIndex.build` releases its build encoder before returning -- so no
VRAM is still held when Ollama is started afterwards. Nothing here can leak the
GPU into the online loop; that binding lives in ``dense_index.py`` and is not
overridable from this script except by ``--cpu``, which exists only so a
machine without CUDA fails slowly instead of loudly.

**3. It is idempotent, and ``--force`` really replaces.** Without ``--force`` an
existing index is *loaded and smoke-tested* rather than skipped on the strength
of the directory existing -- a half-written index from an interrupted build
would otherwise look done. With ``--force`` the directory is removed before
rebuilding: ``bm25s`` and FAISS write several files each, and overwriting in
place can leave a stale file from a build at a different corpus size, which
would then fail the doc-count check with a confusing message.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # sibling scripts, however this file is invoked
    sys.path.insert(0, str(_HERE))

from download_data import force_utf8_stdout, rule  # noqa: E402

from agentic_ir.baselines.base import index_dir  # noqa: E402
from agentic_ir.config import Config, Paths, load_config  # noqa: E402
from agentic_ir.indexing.bm25_index import BM25Index, BM25Settings  # noqa: E402
from agentic_ir.indexing.corpus import DATASETS, Corpus  # noqa: E402
from agentic_ir.indexing.dense_index import DenseIndex, DenseSettings  # noqa: E402
from agentic_ir.indexing.hybrid import HybridIndex, HybridSettings  # noqa: E402
from agentic_ir.types import Source  # noqa: E402

#: The query used to smoke-test a freshly built or already-present index. Both
#: corpora are Wikipedia paragraphs, so this hits something in either one; what
#: is being checked is that all three channels answer at all, not what they say.
SMOKE_QUERY = "Which magazine was started first, Arthur's Magazine or First for Women?"


# ---------------------------------------------------------------------------
# Disk accounting
# ---------------------------------------------------------------------------

def dir_bytes(path: Path) -> int:
    """Total size of every file under ``path``. Zero when it does not exist."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human(n_bytes: float) -> str:
    """Bytes as MB/GB, for a progress line a human reads once."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024.0 or unit == "GB":
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:,.1f} GB"  # pragma: no cover - unreachable, loop returns


def file_sizes(root: Path) -> dict[str, int]:
    """Every file under ``root``, keyed by its path relative to ``root``.

    Recorded in the stats file so that "the FAISS index is 98 MB" in the report
    is traceable to a specific artefact rather than to a directory total that
    also counts the doc-id map.
    """
    return {
        str(p.relative_to(root)).replace("\\", "/"): p.stat().st_size
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_dataset(
    dataset: Source,
    cfg: Config,
    *,
    progress: bool = True,
    allow_cpu_fallback: bool = False,
) -> dict[str, Any]:
    """Build both channels over ``dataset``'s corpus and save them.

    The channels are built here rather than through
    :meth:`HybridIndex.build` for one reason: that method reports a single
    elapsed time, and the report needs the sparse and dense costs separately --
    they differ by two orders of magnitude, and the argument for keeping BM25
    in the stack rests partly on that.
    """
    target = index_dir(dataset, cfg=cfg)
    corpus_started = time.perf_counter()
    corpus = Corpus.load(dataset, cfg=cfg, progress=progress)
    corpus_s = time.perf_counter() - corpus_started
    print(f"  corpus  {len(corpus):,} passages loaded in {corpus_s:,.1f}s")

    sparse_settings = BM25Settings.from_config(cfg)
    dense_settings = DenseSettings.from_config(cfg)
    hybrid_settings = HybridSettings.from_config(cfg)

    print(
        f"  bm25    building (k1={sparse_settings.k1}, b={sparse_settings.b}, "
        f"stemmer={sparse_settings.stemmer}, stopwords={sparse_settings.stopwords})"
    )
    sparse_started = time.perf_counter()
    bm25 = BM25Index.build(
        corpus, cfg=cfg, settings=sparse_settings, show_progress=progress
    )
    sparse_s = time.perf_counter() - sparse_started
    print(f"  bm25    {bm25.n_docs:,} docs, {bm25.n_terms:,} terms in {sparse_s:,.1f}s")

    print(
        f"  dense   building ({dense_settings.model}, dim={dense_settings.dim}, "
        f"device={dense_settings.build_device}, batch={dense_settings.batch_size})"
    )
    dense_started = time.perf_counter()
    dense = DenseIndex.build(
        corpus,
        cfg=cfg,
        settings=dense_settings,
        show_progress=progress,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    dense_s = time.perf_counter() - dense_started
    print(
        f"  dense   {dense.n_docs:,} vectors on {dense.built_device} in {dense_s:,.1f}s "
        f"({dense.build_seconds:,.1f}s encoding)"
    )

    index = HybridIndex(hybrid_settings, bm25=bm25, dense=dense, corpus=corpus)
    save_started = time.perf_counter()
    index.save(target)
    save_s = time.perf_counter() - save_started
    print(f"  saved   {target} in {save_s:,.1f}s  ({human(dir_bytes(target))})")

    return {
        "dataset": dataset,
        "path": str(target),
        "n_docs": bm25.n_docs,
        "n_terms": bm25.n_terms,
        "seconds": {
            "corpus_load": round(corpus_s, 3),
            "bm25_build": round(sparse_s, 3),
            "dense_build": round(dense_s, 3),
            "dense_encode": round(dense.build_seconds, 3),
            "save": round(save_s, 3),
            "total": round(corpus_s + sparse_s + dense_s + save_s, 3),
        },
        "dense_device": dense.built_device,
        "settings": {
            "sparse": sparse_settings.to_dict(),
            "dense": dense_settings.to_dict(),
            "hybrid": hybrid_settings.to_dict(),
        },
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(dataset: Source, cfg: Config, *, query: str = SMOKE_QUERY) -> dict[str, Any]:
    """Reload from disk and query all three channels.

    Re-reading through the production loader is the point: an index that only
    works from the in-memory objects it was built from is an index that will
    fail on the first evaluation run instead of here. The dense channel comes
    back in *query* mode, so this also proves the CPU query encoder loads.
    """
    target = index_dir(dataset, cfg=cfg)
    corpus = Corpus.load(dataset, cfg=cfg)
    index = HybridIndex.load(target, corpus=corpus)

    if len(index) != len(corpus):
        raise SystemExit(
            f"{target}: index holds {len(index):,} docs but the corpus has {len(corpus):,}; "
            f"rebuild with --force --dataset {dataset}"
        )

    started = time.perf_counter()
    sparse_hits = index.bm25.search(query, 10)
    sparse_s = time.perf_counter() - started
    started = time.perf_counter()
    dense_hits = index.dense.search(query, 10)
    dense_s = time.perf_counter() - started
    started = time.perf_counter()
    fused_hits = index.search(query, 10, candidate_k=50)
    fused_s = time.perf_counter() - started

    if not (sparse_hits and dense_hits and fused_hits):
        raise SystemExit(
            f"{target}: a channel returned nothing for the smoke query; the index is unusable"
        )

    print(f"  smoke   {query!r}")
    print(
        f"    bm25  {len(sparse_hits):>2} hits in {sparse_s * 1000:6.1f}ms  "
        f"top={sparse_hits[0].passage.title!r}"
    )
    print(
        f"    dense {len(dense_hits):>2} hits in {dense_s * 1000:6.1f}ms  "
        f"top={dense_hits[0].passage.title!r}"
    )
    print(
        f"    rrf   {len(fused_hits):>2} hits in {fused_s * 1000:6.1f}ms  "
        f"top={fused_hits[0].passage.title!r}"
    )

    return {
        "query": query,
        "n_docs": len(index),
        "n_terms": index.bm25.n_terms,
        "bm25": {
            "hits": len(sparse_hits),
            "latency_s": round(sparse_s, 4),
            "top": sparse_hits[0].passage.title,
        },
        "dense": {
            "hits": len(dense_hits),
            "latency_s": round(dense_s, 4),
            "top": dense_hits[0].passage.title,
            "query_device": index.dense.query_device,
        },
        "hybrid": {
            "hits": len(fused_hits),
            "latency_s": round(fused_s, 4),
            "top": fused_hits[0].passage.title,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(stats: dict[str, Any]) -> None:
    target = Path(stats["path"])
    sizes = stats["files"]
    print(f"\n  passages indexed  : {stats['n_docs']:,}")
    print(f"  analysed terms    : {stats['n_terms']:,}")
    if "seconds" in stats:
        s = stats["seconds"]
        print(f"  corpus load       : {s['corpus_load']:8.1f}s")
        print(f"  bm25 build        : {s['bm25_build']:8.1f}s")
        print(f"  dense build       : {s['dense_build']:8.1f}s "
              f"({s['dense_encode']:.1f}s encoding on {stats['dense_device']})")
        print(f"  save              : {s['save']:8.1f}s")
        print(f"  TOTAL             : {s['total']:8.1f}s")
    print(f"  on disk           : {human(sizes['total_bytes'])} at {target}")
    print(f"    bm25/           : {human(sizes['bm25_bytes'])}")
    print(f"    dense/          : {human(sizes['dense_bytes'])}")
    for name, size in sorted(sizes["files"].items()):
        print(f"      {name:<28} {human(size):>12}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the BM25 + dense retrieval indexes from data/processed/.",
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument(
        "--force", action="store_true",
        help="delete and rebuild an index that already exists",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="build the dense index on CPU when retrieval.dense.build_device is "
             "unavailable (roughly 20x slower; the default is to fail loudly instead)",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="suppress the per-batch progress bars (for a log file)",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdout()
    args = parse_args(argv)
    cfg = load_config(args.config)
    indexes_root = Paths.from_config(cfg).indexes
    progress = not args.no_progress

    targets: list[Source] = list(DATASETS) if args.dataset == "all" else [args.dataset]
    exit_code = 0
    for dataset in targets:
        target = index_dir(dataset, cfg=cfg)
        rule(f"{dataset}: hybrid index -> {target}")

        if target.exists() and args.force:
            print(f"  --force: removing {target}")
            shutil.rmtree(target)

        stats: dict[str, Any]
        if target.exists():
            # Present is not the same as usable. verify() is the real check,
            # and it raises if the directory is a half-written build.
            print("  exists  (pass --force to rebuild); verifying instead")
            stats = {
                "dataset": dataset,
                "path": str(target),
                "rebuilt": False,
            }
        else:
            try:
                stats = build_dataset(
                    dataset, cfg, progress=progress, allow_cpu_fallback=args.cpu
                )
            except FileNotFoundError as exc:
                print(f"\nFAILED: {exc}\n        Run: python scripts/build_corpus.py "
                      f"--dataset {dataset}", file=sys.stderr)
                exit_code = 1
                continue
            except Exception as exc:  # noqa: BLE001 - one dataset must not kill the other
                print(f"\nFAILED: {dataset}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if "out of memory" in str(exc).lower():
                    # Measured on this machine: qwen3:8b holds 6.2 GiB of an
                    # 8 GiB card, which leaves the bge encoder ~1.1 GiB and it
                    # needs ~2. The build is an offline step that runs *before*
                    # the model is needed, so unloading is the fix -- not a
                    # smaller batch, which only moves the failure later.
                    print(
                        "        The GPU is still holding an Ollama model. This build is an\n"
                        "        offline step: free the card first, then re-run.\n"
                        "          ollama ps                # what is resident\n"
                        "          ollama stop <model>      # unload it\n"
                        f"          python scripts/build_indexes.py --dataset {dataset}\n"
                        "        Or pass --cpu to build on CPU instead (roughly 20x slower).",
                        file=sys.stderr,
                    )
                exit_code = 1
                continue
            stats["rebuilt"] = True

        stats["verification"] = verify(dataset, cfg)
        stats.setdefault("n_docs", stats["verification"]["n_docs"])
        stats.setdefault("n_terms", stats["verification"]["n_terms"])
        stats["files"] = {
            "total_bytes": dir_bytes(target),
            "bm25_bytes": dir_bytes(target / "bm25"),
            "dense_bytes": dir_bytes(target / "dense"),
            "files": file_sizes(target),
        }
        report(stats)

        stats_path = indexes_root / f"{dataset}_index_stats.json"
        if stats["rebuilt"] or not stats_path.exists():
            with stats_path.open("w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2, ensure_ascii=False, sort_keys=True)
            print(f"  stats   {stats_path}")

    if exit_code == 0:
        print("\nNext: python scripts/build_kg.py --dataset all")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
