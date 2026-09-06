"""Does the Verifier's confidence tell correct answers from incorrect ones?

This exists because of a result the ablation grid produced and could not
explain. Removing the backward edge costs 0.037 F1 on HotpotQA at $p = 0.008$
and 0.001 F1 on 2WikiMultihopQA at $p = 1.000$, while the loop fires on 98 of
that dataset's 250 questions. A mechanism that runs on more than a third of a
dataset and changes nothing is either doing the wrong thing or choosing the
wrong questions, and the trigger is a single number: the confidence blend,
against a threshold of 0.55.

So this module asks the narrow question directly. Given the confidences one
run actually produced, how well do they rank correct answers above incorrect
ones (AUC), how far is the confidence from the observed accuracy (ECE), and do
the questions the Verifier sends back differ in accuracy from the ones it
accepts?

**This is a diagnostic, not a calibration.** It reads the *evaluation* slice,
because that is where the ablation result lives and the question is about that
result. Nothing here selects a threshold, a weight, or any other parameter --
doing so on evaluation questions would be leakage, which is what
``eval/calibrate.py`` and its disjoint 50-question slice exist to avoid. The
generated caption says this in the document too, so a reader cannot mistake
one for the other.

Both estimates carry a bootstrap interval, because the honest finding here
turned out to be a null and a null without an interval is not a finding.

    python -m agentic_ir.eval.confidence
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = [
    "ConfidenceDiagnostic",
    "auc",
    "expected_calibration_error",
    "threshold_gap",
    "diagnose",
    "render_table",
]

#: Pairs are ``(confidence, correct)`` with ``correct`` in ``{0.0, 1.0}``.
Pair = tuple[float, float]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def auc(pairs: Sequence[Pair]) -> float | None:
    """Probability a correct answer outranks an incorrect one, ties at a half.

    Computed directly rather than through a library so the tie handling is
    visible: the confidence blend is a weighted sum of four bounded components
    and produces exact ties often enough that dropping them, or counting them
    as wins, moves the third decimal.

    ``None`` when one class is empty, which is not a score of zero.
    """
    positive = [c for c, y in pairs if y == 1.0]
    negative = [c for c, y in pairs if y != 1.0]
    if not positive or not negative:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in positive for b in negative)
    return wins / (len(positive) * len(negative))


def expected_calibration_error(pairs: Sequence[Pair], bins: int = 10) -> float | None:
    """Mean gap between stated confidence and observed accuracy, bin-weighted.

    Equal-width bins over ``[0, 1]``, the convention the calibration chapter
    already uses, so the two numbers are comparable. Empty bins contribute
    nothing rather than contributing a zero, which would flatter a model whose
    confidences all sit in one place.
    """
    if not pairs:
        return None
    total = len(pairs)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [
            (c, y) for c, y in pairs
            if lower <= c < upper or (index == bins - 1 and c == 1.0)
        ]
        if not bucket:
            continue
        accuracy = sum(y for _, y in bucket) / len(bucket)
        confidence = sum(c for c, _ in bucket) / len(bucket)
        error += len(bucket) / total * abs(accuracy - confidence)
    return error


def threshold_gap(pairs: Sequence[Pair], threshold: float) -> float | None:
    """Accuracy below the threshold minus accuracy above it.

    This is the quantity the re-plan loop depends on. A useful trigger sends
    back questions that were *more* likely to be wrong, so this should be
    clearly negative. Zero means the Verifier is choosing which questions to
    retry without regard to whether they needed retrying.
    """
    below = [y for c, y in pairs if c < threshold]
    above = [y for c, y in pairs if c >= threshold]
    if not below or not above:
        return None
    return sum(below) / len(below) - sum(above) / len(above)


def _bootstrap(
    pairs: Sequence[Pair], statistic, resamples: int, seed: int
) -> tuple[float, float] | None:
    """Percentile interval over questions, resampled with replacement."""
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceDiagnostic:
    """What one run's confidences do, with intervals."""

    dataset: str
    run_id: str
    n: int
    base_rate: float
    mean_confidence: float
    auc: float | None
    auc_ci: tuple[float, float] | None
    ece: float | None
    gap: float | None
    gap_ci: tuple[float, float] | None
    threshold: float
    n_below: int
    n_above: int
    notes: list[str] = field(default_factory=list)

    @property
    def discriminates(self) -> bool:
        """Whether the AUC interval excludes chance.

        The whole point of the diagnostic. False means the confidence signal
        cannot be shown to rank correct answers above incorrect ones at all.
        """
        return bool(self.auc_ci and self.auc_ci[0] > 0.5)


def load_pairs(run_dir: Path) -> list[Pair]:
    """``(confidence, em)`` per question, from the trace and the score file.

    EM rather than F1 because the re-plan decision is binary and a partial
    credit score would blur the question being asked. Questions the scorer
    never saw are dropped rather than counted as wrong.
    """
    confidences: dict[str, float] = {}
    traces = run_dir / "traces.jsonl"
    with open(traces, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get("confidence")
            if value is not None:
                confidences[record["qid"]] = float(value)

    pairs: list[Pair] = []
    scores = run_dir / "scores.csv"
    if not scores.exists():
        return pairs
    with open(scores, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qid = row.get("qid")
            if qid in confidences and row.get("em") not in ("", None):
                pairs.append((confidences[qid], float(row["em"])))
    return pairs


def diagnose(
    run_dir: Path,
    dataset: str,
    *,
    threshold: float = 0.55,
    resamples: int = 2000,
    seed: int = 42,
) -> ConfidenceDiagnostic | None:
    pairs = load_pairs(run_dir)
    if not pairs:
        return None

    notes: list[str] = []
    value = auc(pairs)
    interval = _bootstrap(pairs, auc, resamples, seed) if value is not None else None
    gap = threshold_gap(pairs, threshold)
    gap_interval = (
        _bootstrap(pairs, lambda s: threshold_gap(s, threshold), resamples, seed)
        if gap is not None else None
    )
    if interval and interval[0] <= 0.5 <= interval[1]:
        notes.append("the AUC interval contains 0.5, so the signal cannot be "
                     "shown to rank correct answers above incorrect ones")
    if gap_interval and gap_interval[0] <= 0.0 <= gap_interval[1]:
        notes.append("the re-plan trigger selects questions whose accuracy is "
                     "indistinguishable from the accuracy of the ones it accepts")

    return ConfidenceDiagnostic(
        dataset=dataset,
        run_id=run_dir.name,
        n=len(pairs),
        base_rate=sum(y for _, y in pairs) / len(pairs),
        mean_confidence=sum(c for c, _ in pairs) / len(pairs),
        auc=value,
        auc_ci=interval,
        ece=expected_calibration_error(pairs),
        gap=gap,
        gap_ci=gap_interval,
        threshold=threshold,
        n_below=sum(1 for c, _ in pairs if c < threshold),
        n_above=sum(1 for c, _ in pairs if c >= threshold),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt(value: float | None, places: int = 3) -> str:
    return "--" if value is None else f"{value:.{places}f}"


def _ci(interval: tuple[float, float] | None, places: int = 3) -> str:
    if interval is None:
        return "--"
    return f"[{interval[0]:.{places}f}, {interval[1]:.{places}f}]"


def render_table(rows: Sequence[ConfidenceDiagnostic]) -> str:
    """A LaTeX fragment whose caption states what it is and is not."""
    if not rows:
        return "% no run carried both confidences and scores\n"

    header = [
        "% Chapter 5 -- does the confidence signal discriminate?",
        "% GENERATED by src/agentic_ir/eval/confidence.py -- do not edit by hand.",
        "% Regenerate with: python -m agentic_ir.eval.confidence",
        "% Source runs:",
    ]
    header += [f"%   {r.dataset}: {r.run_id} ({r.n} questions)" for r in rows]

    caption = (
        r"Whether the Verifier's confidence separates correct answers from "
        r"incorrect ones, on the runs the ablation study reports. "
        r"\textit{AUC} is the probability that a correct answer is scored above "
        r"an incorrect one, so 0.5 is chance; \textit{ECE} is the mean gap "
        r"between stated confidence and observed accuracy over ten equal-width "
        r"bins; \textit{trigger gap} is the accuracy of the questions the "
        r"Verifier sends back minus the accuracy of the ones it accepts. A "
        r"useful trigger sends back the answers that were wrong, so the gap "
        r"should be clearly \emph{negative}; a gap whose interval spans zero "
        r"means the loop is choosing which questions to retry without regard "
        r"to whether they needed retrying. Intervals are 95\% "
        r"percentile bootstraps over questions. "
        r"\textbf{This is a diagnostic, not a calibration.} It is computed on "
        r"the evaluation slice, because that is where the ablation result it "
        r"explains was measured, and it selects no threshold, weight or other "
        r"parameter -- tuning anything on these questions would be leakage, "
        r"which is what the disjoint calibration slice of "
        r"Table~\ref{tab:calibration} exists to prevent."
    )

    lines = [
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        f"  \\caption{{{caption}}}",
        r"  \label{tab:confidence-diagnostic}",
        r"  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Dataset & $n$ & Base rate & Mean conf. & AUC [95\% CI] & ECE "
        r"& Trigger gap [95\% CI] \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            f"    \\texttt{{{row.dataset}}} & {row.n} & {_fmt(row.base_rate)} & "
            f"{_fmt(row.mean_confidence)} & {_fmt(row.auc)} {_ci(row.auc_ci)} & "
            f"{_fmt(row.ece)} & {row.gap:+.3f} {_ci(row.gap_ci)} \\\\"
            if row.gap is not None else
            f"    \\texttt{{{row.dataset}}} & {row.n} & {_fmt(row.base_rate)} & "
            f"{_fmt(row.mean_confidence)} & {_fmt(row.auc)} {_ci(row.auc_ci)} & "
            f"{_fmt(row.ece)} & -- \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]
    return "\n".join(header + lines)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _newest_full_run(runs: Path, dataset: str) -> Path | None:
    """The longest, then newest, ``agentic_full`` run for a dataset.

    The same rule the tables use, so the diagnostic describes the run the
    ablation study reports rather than some other one.
    """
    candidates = []
    for directory in sorted(runs.glob(f"agentic_full_{dataset}_*")):
        traces = directory / "traces.jsonl"
        if not traces.exists():
            continue
        with open(traces, "rb") as handle:
            n = sum(1 for line in handle if line.strip())
        candidates.append((n, directory.name, directory))
    return max(candidates)[2] if candidates else None


def main(argv: Sequence[str] | None = None) -> int:
    from ..config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=None, help="run directory root")
    parser.add_argument("--tex", default=None, help="LaTeX fragment path")
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    root = Path(__file__).resolve().parents[3]
    runs = Path(args.runs) if args.runs else root / "results" / "runs"
    tex_path = Path(args.tex) if args.tex else root / "results" / "tables" / "confidence_diagnostic.tex"
    json_path = (
        Path(args.json_path) if args.json_path
        else root / "results" / "calibration" / "confidence_diagnostic.json"
    )
    threshold = float(cfg.get("agents.verifier.confidence_threshold", 0.55))
    seed = args.seed if args.seed is not None else int(cfg.get("project.seed", 42))

    rows = []
    for dataset in ("hotpotqa", "twowiki"):
        directory = _newest_full_run(runs, dataset)
        if directory is None:
            continue
        row = diagnose(
            directory, dataset, threshold=threshold,
            resamples=args.bootstrap, seed=seed,
        )
        if row is not None:
            rows.append(row)

    for row in rows:
        print(f"{row.dataset:9} n={row.n}  base rate {row.base_rate:.3f}  "
              f"mean confidence {row.mean_confidence:.3f}")
        print(f"          AUC {_fmt(row.auc)} {_ci(row.auc_ci)}  "
              f"{'DISCRIMINATES' if row.discriminates else 'AT CHANCE'}")
        print(f"          ECE {_fmt(row.ece)}   trigger gap "
              f"{row.gap:+.3f} {_ci(row.gap_ci)}" if row.gap is not None
              else f"          ECE {_fmt(row.ece)}")
        for note in row.notes:
            print(f"          - {note}")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(render_table(rows), encoding="utf-8")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            [
                {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in vars(row).items()
                }
                for row in rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {tex_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
