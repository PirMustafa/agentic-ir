"""Draw the stratified evaluation sample and a disjoint calibration slice.

    python scripts/sample_eval_set.py --dataset all

Writes, per dataset:

* ``data/processed/{dataset}_eval_250.jsonl``  -- the 250 questions every
  configuration in Chapter 4 is scored on
* ``data/processed/{dataset}_calib_50.jsonl``  -- 50 questions, **disjoint**,
  used only to calibrate ``agents.verifier.confidence_threshold``

Both files carry the same ``GoldAnswer`` schema as the qrels they are drawn
from, so ``indexing.corpus.load_eval_set`` reads either one.

Why the slices are materialised, not recomputed
-----------------------------------------------
A full agentic run over HotpotQA's 7,405-question validation split is ~10h on
an 8 GB laptop GPU, so the report is computed on a sample -- which makes the
identity of that sample part of the result. It is drawn once, with
``project.seed``, written to disk, and its SHA-256 printed here for
``meta.json``. Nothing re-samples at run time.

Why calibration is disjoint
---------------------------
``confidence_threshold`` controls the re-plan rate, which is itself a headline
agent metric. Tuning it on the questions it is then measured on would be
leakage of exactly the kind the report is meant to rule out, so the calibration
slice is drawn from the *complement* of the eval sample and the disjointness is
asserted before either file is written.

Allocation is proportional with largest remainders: each stratum gets
``floor(n_stratum / N * k)`` questions, and the leftover places go to the
strata with the largest fractional parts, ties broken by stratum name so the
result does not depend on dictionary order.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from download_data import force_utf8_stdout, rule  # noqa: E402

from agentic_ir.config import Config, load_config  # noqa: E402
from agentic_ir.indexing.corpus import (  # noqa: E402
    DATASETS,
    DEFAULT_CALIB_SIZE,
    eval_set_path,
    gold_to_json,
    load_qrels,
    write_jsonl,
)
from agentic_ir.types import GoldAnswer  # noqa: E402

#: ``datasets.*.strata`` names a dataset column; ``GoldAnswer`` calls HotpotQA's
#: ``type`` column ``qtype`` (``type`` is a builtin). One place to reconcile it.
STRATUM_FIELDS: dict[str, str] = {"level": "level", "type": "qtype", "qtype": "qtype"}


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def stratum_of(gold: GoldAnswer, keys: Sequence[str]) -> str:
    """The stratum label of one question, e.g. ``"hard"`` or ``"comparison"``."""
    parts = []
    for key in keys:
        value = getattr(gold, STRATUM_FIELDS.get(key, key), None)
        parts.append(str(value) if value is not None else "unknown")
    return "|".join(parts)


def group_by_stratum(
    qrels: Mapping[str, GoldAnswer], keys: Sequence[str], qids: Sequence[str]
) -> dict[str, list[str]]:
    """``stratum -> sorted qids``. Sorted so sampling cannot depend on dict order."""
    groups: dict[str, list[str]] = {}
    for qid in qids:
        groups.setdefault(stratum_of(qrels[qid], keys), []).append(qid)
    return {stratum: sorted(members) for stratum, members in sorted(groups.items())}


def allocate(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Proportional allocation of ``total`` places over strata, largest remainder.

    Capped by availability, and any place freed by a cap is redistributed, so
    the allocation sums to ``min(total, population)`` whenever that is possible.
    """
    population = sum(sizes.values())
    if population == 0:
        return {stratum: 0 for stratum in sizes}
    total = min(total, population)
    exact = {stratum: total * n / population for stratum, n in sizes.items()}
    alloc = {stratum: min(int(q), sizes[stratum]) for stratum, q in exact.items()}
    order = sorted(sizes, key=lambda s: (-(exact[s] - int(exact[s])), s))
    while (remaining := total - sum(alloc.values())) > 0:
        progressed = False
        for stratum in order:
            if remaining == 0:
                break
            if alloc[stratum] < sizes[stratum]:
                alloc[stratum] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # every stratum is at capacity
            break
    return alloc


def draw(
    groups: Mapping[str, Sequence[str]], alloc: Mapping[str, int], rng: random.Random
) -> list[str]:
    """Sample within each stratum, strata visited in sorted order."""
    picked: list[str] = []
    for stratum in sorted(groups):
        pool = list(groups[stratum])
        picked.extend(rng.sample(pool, alloc.get(stratum, 0)))
    return sorted(picked)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_table(
    groups: Mapping[str, Sequence[str]],
    eval_qids: Sequence[str],
    calib_qids: Sequence[str],
    qrels: Mapping[str, GoldAnswer],
    keys: Sequence[str],
) -> None:
    eval_counts: dict[str, int] = {}
    calib_counts: dict[str, int] = {}
    for qid in eval_qids:
        label = stratum_of(qrels[qid], keys)
        eval_counts[label] = eval_counts.get(label, 0) + 1
    for qid in calib_qids:
        label = stratum_of(qrels[qid], keys)
        calib_counts[label] = calib_counts.get(label, 0) + 1

    population = sum(len(v) for v in groups.values())
    print(f"\n  {'stratum':<22}{'population':>12}{'share':>9}"
          f"{'eval':>7}{'share':>9}{'calib':>7}")
    for stratum in sorted(groups):
        n = len(groups[stratum])
        n_eval = eval_counts.get(stratum, 0)
        print(f"  {stratum:<22}{n:>12,}{n / population:>9.1%}"
              f"{n_eval:>7}{n_eval / max(len(eval_qids), 1):>9.1%}"
              f"{calib_counts.get(stratum, 0):>7}")
    print(f"  {'TOTAL':<22}{population:>12,}{1:>9.1%}"
          f"{len(eval_qids):>7}{1:>9.1%}{len(calib_qids):>7}")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample(dataset: str, cfg: Config, *, size: int | None, calib_size: int) -> bool:
    keys = list(cfg.get(f"datasets.{dataset}.strata", ()))  # frozen tuple -> list
    eval_size = size if size is not None else int(cfg.get(f"datasets.{dataset}.eval_sample"))
    seed = int(cfg.get("project.seed"))

    qrels = load_qrels(dataset, cfg=cfg)  # type: ignore[arg-type]
    all_qids = sorted(qrels)
    rule(f"{dataset}: sampling {eval_size} eval + {calib_size} calib "
         f"from {len(all_qids):,} questions (strata={keys}, seed={seed})")

    groups = group_by_stratum(qrels, keys, all_qids)
    if set(groups) == {"unknown"}:
        print(f"  WARNING: strata {keys} are empty for this dataset -- sampling is unstratified")

    # One RNG, drawn from in a fixed order: eval first, then calibration from
    # what is left. Same seed and same qrels therefore reproduce both files.
    rng = random.Random(seed)
    eval_alloc = allocate({s: len(m) for s, m in groups.items()}, eval_size)
    eval_qids = draw(groups, eval_alloc, rng)

    chosen = set(eval_qids)
    remaining = {s: [q for q in members if q not in chosen] for s, members in groups.items()}
    calib_alloc = allocate({s: len(m) for s, m in remaining.items()}, calib_size)
    calib_qids = draw(remaining, calib_alloc, rng)

    overlap = chosen & set(calib_qids)
    if overlap:  # defensive: the complement construction makes this unreachable
        raise SystemExit(f"calibration slice overlaps the eval sample on {len(overlap)} qids")
    if len(eval_qids) != eval_size or len(calib_qids) != calib_size:
        print(f"  WARNING: asked for {eval_size}+{calib_size}, got "
              f"{len(eval_qids)}+{len(calib_qids)} -- population too small")

    print_table(groups, eval_qids, calib_qids, qrels, keys)

    ok = True
    for split, qids, expected in (
        ("eval", eval_qids, eval_size),
        ("calib", calib_qids, calib_size),
    ):
        path = eval_set_path(dataset, split=split, size=expected, cfg=cfg)  # type: ignore[arg-type]
        n = write_jsonl(path, (gold_to_json(qrels[qid]) for qid in qids))
        print(f"\n  {split:<6} {n:>4} questions -> {path}")
        print(f"  {'':<6} sha256 = {sha256_of(path)}")
        ok = ok and n == expected
    print(f"\n  disjoint: yes ({len(overlap)} shared qids)")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the stratified eval sample and a disjoint calibration slice.",
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--size", type=int, default=None,
                        help="override datasets.<name>.eval_sample")
    parser.add_argument("--calib-size", type=int, default=DEFAULT_CALIB_SIZE)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdout()
    args = parse_args(argv)
    cfg = load_config(args.config)

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    exit_code = 0
    for dataset in targets:
        try:
            if not sample(dataset, cfg, size=args.size, calib_size=args.calib_size):
                exit_code = 1
        except FileNotFoundError as exc:
            print(f"\nFAILED: {exc}", file=sys.stderr)
            exit_code = 1
    if exit_code == 0:
        print("\nRecord the SHA-256 values above in the run's meta.json; never re-sample.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
