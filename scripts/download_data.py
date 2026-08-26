"""Fetch the two evaluation datasets to ``data/raw/``.

    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset twowiki --force

Both datasets come from the Hugging Face Hub as **parquet** repos. That is not
incidental: ``datasets>=4.0`` removed script-based loaders entirely ("Dataset
scripts are no longer supported") and there is no ``trust_remote_code`` escape
hatch any more, so the classic ``xanhho/2WikiMultihopQA`` script repo cannot be
loaded at all. ``config/config.yaml`` pins parquet mirrors for both.

Only the configured split's parquet shards are fetched, not the whole repo.
``load_dataset(hf_id, config, split="validation")`` resolves the *config's*
data files and downloads all of them -- 388 MB of train and test for 2Wiki to
reach a 29 MB validation split -- and on this machine that transfer stalls
part-way through the 165 MB train shard often enough to matter. Resolving the
shard list from the Hub and loading those files directly turns a ~10 minute
gamble into a ~5 second download. ``--full-download`` restores the plain
``load_dataset`` path if a repo's file layout ever defeats the resolver.

Each split is materialised with ``save_to_disk`` into ``data/raw/{dataset}/``
so that every later stage reads a frozen local snapshot rather than whatever
the Hub serves today, and a sidecar ``{dataset}.meta.json`` records the row
count, the column list, the shard list and the dataset fingerprint for
``meta.json`` provenance. Re-running is a no-op unless ``--force`` is given.

This script is also the first place in the pipeline that prints dataset text,
which on this machine is where cp1252 bites: ``print(row["answer"])`` on
``'Malgorzata Braunek'`` raises ``UnicodeEncodeError`` under the Windows
default encoding. Standard output is reconfigured to UTF-8 on entry and every
file handle in this project is opened with an explicit encoding.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # `pip install -e .` is optional for the scripts
    sys.path.insert(0, str(_SRC))

from agentic_ir.config import Config, Paths, load_config  # noqa: E402
from agentic_ir.indexing.corpus import DATASETS  # noqa: E402

#: Row counts verified against the mirrors on 2026-08-26. A mismatch does not
#: stop the build -- it means the mirror was revised, which is worth knowing
#: before it turns up as an unexplained change in Chapter 4.
EXPECTED_ROWS: dict[str, int] = {"hotpotqa": 7405, "twowiki": 12576}


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def force_utf8_stdout() -> None:
    """Make ``print`` survive dataset text on a cp1252 console.

    ``line_buffering`` as well as the encoding: piped into a log file, a
    block-buffered stdout reorders itself against unbuffered stderr, and the
    failure message ends up above the section header it belongs to.
    """
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, ValueError):  # pragma: no cover - exotic consoles
            stream.reconfigure(  # type: ignore[union-attr]
                encoding="utf-8", errors="replace", line_buffering=True,
            )


def rule(label: str = "") -> None:
    print(f"\n{'=' * 78}")
    if label:
        print(label)
        print("=" * 78)


def fail(message: str, hint: str = "") -> None:
    """A readable failure. A traceback here tells the user nothing useful."""
    print(f"\nFAILED: {message}", file=sys.stderr)
    if hint:
        print(f"        {hint}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Paths and shared loading (imported by build_corpus.py)
# ---------------------------------------------------------------------------

def raw_path(dataset: str, cfg: Config | None = None) -> Path:
    """Where ``dataset``'s frozen split lives on disk."""
    return Paths.from_config(cfg or load_config()).raw / dataset


def raw_meta_path(dataset: str, cfg: Config | None = None) -> Path:
    return Paths.from_config(cfg or load_config()).raw / f"{dataset}.meta.json"


def load_raw_split(dataset: str, cfg: Config | None = None):
    """Load the frozen split saved by this script.

    Raises ``FileNotFoundError`` with an actionable message rather than the
    ``datasets`` internal error, because "run download_data.py first" is the
    only thing the caller can do about it.
    """
    target = raw_path(dataset, cfg)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found -- run: python scripts/download_data.py --dataset {dataset}"
        )
    from datasets import load_from_disk

    return load_from_disk(str(target))


def dataset_spec(cfg: Config, dataset: str) -> dict[str, Any]:
    """The ``datasets.<name>`` block, thawed into a plain dict."""
    try:
        spec = cfg.get(f"datasets.{dataset}")
    except KeyError:
        raise SystemExit(f"unknown dataset {dataset!r}: not in {cfg.path}") from None
    return dict(spec)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def describe(dataset: str, ds: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Print a human sanity-check block and return the meta record."""
    columns = list(ds.column_names)
    row = ds[0]
    context = row.get("context") or {}
    titles = list(context.get("title") or ())
    sentences = list(context.get("sentences") or ())
    facts = row.get("supporting_facts") or {}

    print(f"  rows          : {len(ds):,}")
    expected = EXPECTED_ROWS.get(dataset)
    if expected is not None and len(ds) != expected:
        print(f"  WARNING       : expected {expected:,} rows -- the mirror has been revised")
    print(f"  columns       : {columns}")
    print("  first row     :")
    print(f"    id          : {row.get('id')}")
    print(f"    question    : {row.get('question')}")
    print(f"    answer      : {row.get('answer')}")
    print(f"    type/level  : {row.get('type')} / {row.get('level')}")
    print(f"    context     : {len(titles)} paragraphs, "
          f"{sum(len(s) for s in sentences)} sentences")
    print(f"    first title : {titles[0] if titles else '-'}")
    print("    sup. facts  : "
          f"{list(zip(facts.get('title', ()), facts.get('sent_id', ()), strict=False))}")
    if "evidences" in columns:
        print(f"    evidences   : {len(row.get('evidences') or ())} gold triples "
              f"e.g. {(row.get('evidences') or [['-']])[0]}")

    return {
        "dataset": dataset,
        "hf_id": spec.get("hf_id"),
        "hf_config": spec.get("hf_config"),
        "split": spec.get("split"),
        "rows": len(ds),
        "columns": columns,
        # The datasets-library fingerprint is recomputed on every load, so it
        # identifies this process's view, not the data. Real provenance is the
        # shard list plus the corpus SHA-256 that build_corpus.py records.
        "fingerprint": getattr(ds, "_fingerprint", None),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def resolve_shards(hf_id: str, hf_config: str | None, split: str) -> list[str]:
    """The repo-relative parquet files that make up ``split``.

    HotpotQA keeps them under a config directory (``distractor/validation-*``)
    and 2Wiki under ``data/`` with no config at all, so the rule is: basename
    starts with the split name, and -- when a config is configured and at least
    one path carries it -- the path is under that config.
    """
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(hf_id, repo_type="dataset") if f.endswith(".parquet")]
    matches = [
        f for f in files
        if Path(f).name.startswith(f"{split}-") or Path(f).name == f"{split}.parquet"
    ]
    if hf_config:
        scoped = [f for f in matches if hf_config in Path(f).parts[:-1]]
        if scoped:
            matches = scoped
        else:
            # A single-config repo may not use a directory per config.
            configs = {part for f in files for part in Path(f).parts[:-1]}
            if hf_config in configs:  # the config exists but holds no such split
                return []
    return sorted(matches)


def load_split(hf_id: str, hf_config: str | None, split: str, *, full: bool) -> tuple[Any, list[str]]:
    """Load one split. Returns ``(dataset, shards)``; ``shards`` is [] if
    the whole-config loader was used."""
    from datasets import load_dataset

    shards: list[str] = []
    if not full:
        try:
            shards = resolve_shards(hf_id, hf_config, split)
        except Exception as exc:  # noqa: BLE001 - listing is best-effort
            print(f"  note          : could not list {hf_id} ({type(exc).__name__}); "
                  "falling back to the full-config download")
    if not shards:
        print("  downloading   : every parquet shard of the config, not just the requested "
              "split")
        ds = (
            load_dataset(hf_id, hf_config, split=split)
            if hf_config
            else load_dataset(hf_id, split=split)
        )
        return ds, []

    from huggingface_hub import hf_hub_download

    print(f"  downloading   : {len(shards)} shard(s) -> {shards}")
    local = [hf_hub_download(hf_id, shard, repo_type="dataset") for shard in shards]
    ds = load_dataset("parquet", data_files={split: local}, split=split)
    return ds, shards


def fetch(dataset: str, cfg: Config, *, force: bool, full: bool = False) -> bool:
    """Download (or reuse) one dataset. Returns True on success."""
    spec = dataset_spec(cfg, dataset)
    hf_id, hf_config, split = spec.get("hf_id"), spec.get("hf_config"), spec.get("split")
    target = raw_path(dataset, cfg)

    rule(f"{dataset}  <-  {hf_id}" + (f" [{hf_config}]" if hf_config else "") + f"  split={split}")
    if not spec.get("enabled", True):
        print("  note          : datasets."
              f"{dataset}.enabled is false -- fetching anyway because it was requested")

    if target.exists() and not force:
        try:
            ds = load_raw_split(dataset, cfg)
        except Exception as exc:  # noqa: BLE001 - a corrupt snapshot must be recoverable
            fail(f"{target} exists but could not be loaded ({type(exc).__name__}: {exc})",
                 "re-run with --force to re-fetch it")
            return False
        print(f"  cached        : {target}")
        describe(dataset, ds, spec)
        return True

    try:
        import datasets  # noqa: F401
    except ImportError:
        fail("the `datasets` package is not installed",
             "pip install -r requirements.txt")
        return False

    try:
        ds, shards = load_split(hf_id, hf_config, split, full=full)
    except Exception as exc:  # noqa: BLE001 - the whole point is a readable message
        name = type(exc).__name__
        hint = "check the network, then `huggingface-cli whoami` if the repo is gated"
        text = str(exc)
        if "Dataset scripts are no longer supported" in text:
            hint = (f"{hf_id} is a script loader; datasets>={_datasets_version()} cannot load it. "
                    "Point datasets.*.hf_id at a parquet mirror.")
        elif "Couldn't find" in text or "RepositoryNotFound" in text or "404" in text:
            hint = f"{hf_id!r} (config={hf_config!r}, split={split!r}) does not resolve on the Hub"
        fail(f"could not load {hf_id} ({name}: {text.splitlines()[0][:200]})", hint)
        return False

    tmp = target.with_name(target.name + ".tmp")
    try:
        if tmp.exists():
            shutil.rmtree(tmp)
        ds.save_to_disk(str(tmp))
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    except OSError as exc:
        fail(f"could not write {target} ({exc})",
             "close anything holding the old snapshot open, then re-run")
        return False

    print(f"  saved         : {target}")
    meta = describe(dataset, ds, spec)
    meta["shards"] = shards
    meta_path = raw_meta_path(dataset, cfg)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"  meta          : {meta_path}")
    return True


def _datasets_version() -> str:
    try:
        import datasets

        return datasets.__version__
    except Exception:  # noqa: BLE001 # pragma: no cover
        return "4.0"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HotpotQA and 2WikiMultihopQA into data/raw/.",
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even when a local snapshot already exists")
    parser.add_argument("--full-download", action="store_true",
                        help="fetch every shard of the config via load_dataset, not just "
                             "the configured split's shards")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdout()
    args = parse_args(argv)
    cfg = load_config(args.config)
    Paths.from_config(cfg)  # create data/raw, data/processed, ... up front

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    ok = [fetch(name, cfg, force=args.force, full=args.full_download) for name in targets]

    rule("summary")
    for name, success in zip(targets, ok, strict=True):
        print(f"  {name:<10} {'ok' if success else 'FAILED'}")
    if not all(ok):
        print("\nNothing downstream can run until every dataset above is ok.")
        return 1
    print("\nNext: python scripts/build_corpus.py --dataset all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
