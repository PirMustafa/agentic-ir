"""How much does the same configuration move when you simply run it again?

Every interval in this report is a paired bootstrap over *questions*. That
measures one source of uncertainty -- which 250 questions were sampled -- and
is silent about another: what the same system does on the same questions on a
second execution. Temperature is 0.0 and the seed is 42, so the intended
answer was "nothing".

It is not nothing. Re-running ``agentic_no_verifier`` on HotpotQA, a
configuration with no verifier and therefore no feedback loop at all, moved
exact match by 0.012 and F1 by 0.014, with 15 of 250 answers changing. 13 of
those 15 had a *different evidence pool*, so the divergence begins in
retrieval and not in generation.

The cause is recorded in the runs themselves: ``meta.json`` stores
``seeding.pythonhashseed``, which is ``null`` for the evaluated grid and
``"0"`` for the re-run. Python randomises string hashing per process unless
that variable is set, set iteration order follows, and the evidence pool is
built from sets. ``cli eval`` sets it; ``run_eval`` invoked directly inherits
whatever the environment had, which for the evaluated grid was nothing.

This module exists so the report can state that in numbers it did not type by
hand. It compares two runs of the *same* configuration and reports the drift.
Nothing here is a defect being hidden: an effect smaller than this drift is
not established by one execution, and the report says so where its own effects
are that small.

    python -m agentic_ir.eval.replication RUN_A RUN_B [RUN_C RUN_D ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = ["ReplicationPair", "compare", "render_table"]


@dataclass
class ReplicationPair:
    """Two executions of one configuration, and how far apart they landed."""

    config_name: str
    dataset: str
    run_a: str
    run_b: str
    n: int
    em_a: float
    em_b: float
    f1_a: float
    f1_b: float
    flipped: int
    flipped_with_different_evidence: int
    hashseed_a: str | None
    hashseed_b: str | None

    @property
    def em_delta(self) -> float:
        return self.em_b - self.em_a

    @property
    def f1_delta(self) -> float:
        return self.f1_b - self.f1_a

    @property
    def flip_rate(self) -> float:
        return self.flipped / self.n if self.n else 0.0


def _scores(run_dir: Path) -> dict[str, tuple[float, float]]:
    path = run_dir / "scores.csv"
    if not path.exists():
        return {}
    out: dict[str, tuple[float, float]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("em") not in ("", None):
                out[row["qid"]] = (float(row["em"]), float(row["f1"]))
    return out


def _evidence_keys(run_dir: Path) -> dict[str, frozenset[str]]:
    """A comparable fingerprint of each question's evidence pool.

    Identifier plus a text prefix, because the identifiers are positional --
    ``e1..en`` are assigned in pool order, so comparing ids alone would call
    two different pools identical whenever they happen to be the same size.
    """
    out: dict[str, frozenset[str]] = {}
    path = run_dir / "traces.jsonl"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            out[record["qid"]] = frozenset(
                f"{e.get('title')}#{e.get('sent_id')}"
                for e in record.get("evidence") or ()
            )
    return out


def _hashseed(run_dir: Path) -> str | None:
    path = run_dir / "meta.json"
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    seeding = meta.get("seeding") or {}
    return seeding.get("pythonhashseed")


def _describe(run_dir: Path) -> tuple[str, str]:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = str(meta.get("config_name", ""))
            dataset = str(meta.get("dataset", ""))
            if name:
                return name, dataset or "?"
        except (OSError, json.JSONDecodeError):
            pass
    return run_dir.name, "?"


def compare(run_a: Path, run_b: Path) -> ReplicationPair | None:
    """Drift between two runs, over the questions both actually scored."""
    scores_a, scores_b = _scores(run_a), _scores(run_b)
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        return None

    evidence_a, evidence_b = _evidence_keys(run_a), _evidence_keys(run_b)
    flipped = [q for q in shared if scores_a[q][0] != scores_b[q][0]]
    differing_pool = sum(
        1 for q in flipped
        if evidence_a.get(q) is not None
        and evidence_b.get(q) is not None
        and evidence_a[q] != evidence_b[q]
    )

    config_name, dataset = _describe(run_a)
    mean = lambda rows, i: sum(rows[q][i] for q in shared) / len(shared)  # noqa: E731
    return ReplicationPair(
        config_name=config_name,
        dataset=dataset,
        run_a=run_a.name,
        run_b=run_b.name,
        n=len(shared),
        em_a=mean(scores_a, 0), em_b=mean(scores_b, 0),
        f1_a=mean(scores_a, 1), f1_b=mean(scores_b, 1),
        flipped=len(flipped),
        flipped_with_different_evidence=differing_pool,
        hashseed_a=_hashseed(run_a),
        hashseed_b=_hashseed(run_b),
    )


def render_table(pairs: Sequence[ReplicationPair]) -> str:
    if not pairs:
        return "% no configuration has been run twice\n"

    header = [
        "% Chapter 5 -- run-to-run drift on an identical configuration",
        "% GENERATED by src/agentic_ir/eval/replication.py -- do not edit by hand.",
        "% Regenerate with: python -m agentic_ir.eval.replication RUN_A RUN_B",
        "% Source runs:",
    ]
    for pair in pairs:
        header.append(f"%   {pair.config_name}/{pair.dataset}: "
                      f"{pair.run_a} vs {pair.run_b} ({pair.n} shared questions)")

    caption = (
        r"Run-to-run drift: the same configuration, the same questions, the "
        r"same seed and a decoding temperature of 0.0, executed twice. "
        r"\textit{Flipped} counts questions whose exact match changed between "
        r"the two executions, and the column beside it how many of those also "
        r"drew a different evidence pool -- locating the divergence in "
        r"retrieval rather than in generation. The cause is the "
        r"\texttt{PYTHONHASHSEED} column: Python randomises string hashing per "
        r"process unless it is set, the evidence pool is assembled from sets, "
        r"and set iteration order follows. \texttt{cli eval} sets it; "
        r"\texttt{run\_eval} invoked directly inherits whatever the "
        r"environment had, which for the evaluated grid was nothing. "
        r"\textbf{This is the report's own error bar on execution}, and it is "
        r"separate from every bootstrap interval elsewhere, all of which "
        r"resample questions and hold the execution fixed. An effect smaller "
        r"than this drift is not established by a single run."
    )

    lines = [
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        f"  \\caption{{{caption}}}",
        r"  \label{tab:replication-drift}",
        r"  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"  \begin{tabular}{llrrrrrrl}",
        r"    \toprule",
        r"    Configuration & Dataset & $n$ & EM run 1 & EM run 2 & $\Delta$EM "
        r"& $\Delta$F1 & Flipped & \texttt{PYTHONHASHSEED} \\",
        r"    \midrule",
    ]
    for pair in pairs:
        seeds = (f"{pair.hashseed_a or 'unset'} vs {pair.hashseed_b or 'unset'}")
        lines.append(
            f"    \\texttt{{{pair.config_name.replace('_', chr(92) + '_')}}} & "
            f"\\texttt{{{pair.dataset}}} & {pair.n} & "
            f"{pair.em_a:.3f} & {pair.em_b:.3f} & "
            f"{pair.em_delta:+.3f} & {pair.f1_delta:+.3f} & "
            f"{pair.flipped} ({pair.flipped_with_different_evidence} via evidence) & "
            f"\\texttt{{{seeds}}} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]
    return "\n".join(header + lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories, in pairs")
    parser.add_argument("--tex", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    if len(args.runs) % 2:
        parser.error("runs must come in pairs")

    root = Path(__file__).resolve().parents[3]
    pairs: list[ReplicationPair] = []
    for i in range(0, len(args.runs), 2):
        pair = compare(Path(args.runs[i]), Path(args.runs[i + 1]))
        if pair is None:
            print(f"no shared scored questions: {args.runs[i]} vs {args.runs[i+1]}")
            continue
        pairs.append(pair)
        print(f"{pair.config_name}/{pair.dataset}  n={pair.n}")
        print(f"  EM {pair.em_a:.3f} -> {pair.em_b:.3f}  ({pair.em_delta:+.3f})")
        print(f"  F1 {pair.f1_a:.3f} -> {pair.f1_b:.3f}  ({pair.f1_delta:+.3f})")
        print(f"  flipped {pair.flipped}/{pair.n} ({pair.flip_rate:.1%}), "
              f"{pair.flipped_with_different_evidence} with a different evidence pool")
        print(f"  PYTHONHASHSEED {pair.hashseed_a or 'unset'} vs "
              f"{pair.hashseed_b or 'unset'}")

    tex = Path(args.tex) if args.tex else root / "results" / "tables" / "replication.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text(render_table(pairs), encoding="utf-8")
    print(f"wrote {tex}")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([vars(p) for p in pairs], indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
