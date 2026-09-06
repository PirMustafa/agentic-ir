r"""Post-hoc calibration of ``agents.verifier.confidence_threshold``.

``config/config.yaml`` ships ``confidence_threshold: 0.55``. That number was
chosen a priori and it gates the re-plan decision (``docs/architecture.md``
3.5 step 6), which makes ``replan_rate`` -- a headline agent metric -- a
function of a guess. Architecture risk 4 names the mitigation: calibrate on the
50-question slice at ``data/processed/{dataset}_calib_50.jsonl``, which is
disjoint from the frozen 250-question evaluation slice, and freeze the value
before the evaluation run. Tuning on the evaluation set would be leakage and
would invalidate every number in Chapter 4.

WHAT THIS SWEEP CANNOT ESTABLISH
--------------------------------
This is a **post-hoc** sweep over confidences that were produced under one
threshold (0.55), and it matters that the report does not claim more.

The threshold is not a read-out filter. It decides whether a re-plan fires; a
re-plan changes the plan, which changes retrieval, which changes the evidence,
the answer, and the confidence attached to it. A question that scores 0.52
under 0.55 was re-planned, and the answer scored here is the answer that
re-planning produced. Had the threshold been 0.45 that question would have
stopped a cycle earlier with a *different* answer -- one this run never
computed. A genuine sweep therefore needs one full evaluation run per candidate
threshold, at roughly 1.7 h each (architecture risk 2), which the project's
compute budget does not support.

So the question answered here is deliberately narrower, and it is a real
question with a real answer:

    Given the confidences this system actually produced, where does the
    accept/reject boundary best separate correct answers from incorrect ones?

That is a statement about the *discriminative power of the confidence signal*,
not about the downstream behaviour of the loop. It is sufficient to detect the
failure mode that matters most -- a threshold sitting somewhere the confidence
signal carries no information -- and insufficient to predict what the re-plan
rate would become. The ``replan_rate`` column below is likewise a counterfactual
read of the first-cycle confidences, not a measurement.

CALIBRATION QUALITY COMES FIRST
-------------------------------
A threshold read off a badly calibrated signal is not worth much, so the module
reports expected calibration error over 10 equal-width bins, the reliability
curve those bins describe, and the Brier score, alongside the sweep. If ECE is
large, the honest report sentence is "the confidence is poorly calibrated and
the threshold is a ranking cut-point, not a probability cut-point"; if ECE is
small, the threshold can be read as an operating point on a probability. The
module says which case it is in rather than leaving the reader to guess.

Calibration is not the same property as *discrimination*, and the two are
reported separately because they fail separately. A signal can be badly
calibrated -- systematically over-confident, large ECE, Brier worse than the
base-rate constant -- and still rank correct answers above incorrect ones
perfectly well, in which case a threshold on it works and only its
interpretation as a probability is wrong. AUC is the statistic that tells those
apart, so it is AUC, not the Brier skill, that licenses the strongest negative
finding this module can report: that the confidence carries no information
about correctness and no threshold read off it means anything.

FIFTY QUESTIONS IS SMALL
------------------------
Every recommendation carries a bootstrap interval over the *selected
threshold*, obtained by re-running the selection on resampled calibration sets.
When that interval is wide -- and on 50 questions it usually is -- the
recommendation is reported as under-powered and the configured value is kept.
Emitting a bare number off 50 questions would look like evidence and would not
be any.

Artefacts, both deterministic given the same run:

* ``results/calibration/{dataset}_threshold.json`` -- the full swept table, the
  calibration quality block, the recommendation and its reasoning.
* ``results/tables/calibration.tex`` -- ``tab:calibration``, ``\input``-able,
  built with the primitives in :mod:`~agentic_ir.eval.tables` so the fragment
  cannot drift from the rest of the report's tables.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..config import PROJECT_ROOT, Config, load_config
from ..indexing.corpus import DATASETS, load_eval_set
from ..trace import iter_records
from ..types import GoldAnswer

# ``_percentile`` is private, and imported deliberately: the interpolated
# percentile rule belongs to bootstrap.py, and re-deriving it here would let the
# two modules disagree about an interval bound at the seams.
from .bootstrap import (  # noqa: PLC2701
    BootstrapResult,
    _percentile,
    bootstrap_mean,
    paired_bootstrap,
)
from .metrics import score_question
from .run_eval import runs_root
from .tables import MISSING, fmt, latex_escape, table

__all__ = [
    "BINS",
    "LIMITATIONS",
    "MAX_RECOMMENDATION_CI_WIDTH",
    "MIN_RECOMMENDATION_N",
    "SWEEP",
    "CalibrationQuality",
    "CalibrationReport",
    "QuestionPoint",
    "Recommendation",
    "ReliabilityBin",
    "ThresholdRow",
    "auc_score",
    "brier_score",
    "build_parser",
    "calibrate",
    "calibration_quality",
    "discover_calibration_run",
    "latex_fragment",
    "load_points",
    "main",
    "recommend",
    "run",
    "sweep",
    "write_artefacts",
]

#: The candidate grid from architecture risk 4: 0.40--0.75 inclusive, step 0.01.
#: Built from integer basis points so no two grid values can collide or drift
#: after a float addition, and so a threshold is safe to round into a dict key.
SWEEP: tuple[float, ...] = tuple(bp / 100.0 for bp in range(40, 76))

#: Equal-width reliability bins. Ten is the convention ECE is usually reported
#: with; on 50 questions that already leaves single-digit counts per occupied
#: bin, and more bins would report noise at higher resolution.
BINS = 10

#: Below this many labelled questions the sweep is reported as under-powered and
#: no threshold change is recommended. 50 is under it, deliberately: the
#: calibration slice is large enough to *audit* 0.55 and too small to *replace*
#: it on a point estimate alone.
MIN_RECOMMENDATION_N = 100

#: How wide the bootstrap interval on the selected threshold may be before the
#: selection is treated as undetermined. 0.10 is over a quarter of the swept
#: range; an interval wider than that names no operating point at all.
MAX_RECOMMENDATION_CI_WIDTH = 0.10

Label = Literal["em", "f1"]

#: What "the answer is correct" means for the accept/reject confusion matrix.
LABEL_DEFINITIONS: dict[str, str] = {
    "em": "exact match against the gold answer (HotpotQA normalisation), EM == 1",
    "f1": "token F1 against the gold answer >= 0.5",
}


# ---------------------------------------------------------------------------
# Per-question observations
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QuestionPoint:
    """One calibration question, reduced to what the sweep needs.

    ``confidence`` is the confidence of the answer the system actually returned;
    ``first_confidence`` is the first cycle's, which is the one the threshold
    consulted when it decided whether to re-plan. Keeping both is what lets the
    implied re-plan rate be computed from the decision that was really gated,
    rather than from a post-selection number no threshold ever saw.
    """

    qid: str
    confidence: float
    first_confidence: float
    em: float
    f1: float
    verdict: str
    replanned: bool
    forced_revise: bool
    answered: bool

    def correct(self, label: Label = "em") -> bool:
        return self.em >= 1.0 if label == "em" else self.f1 >= 0.5


def _as_float(value: Any, default: float = 0.0) -> float:
    """Float or ``default`` -- NaN included, since a NaN confidence would poison
    every comparison it takes part in and silently empty the accept set."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _first_cycle(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """The cycle-0 ``VerificationResult`` of a trace record, or an empty mapping."""
    verifications = record.get("verifications") or ()
    first = verifications[0] if verifications else None
    return first if isinstance(first, Mapping) else {}


def load_points(
    records: Iterable[Mapping[str, Any]],
    golds: Mapping[str, GoldAnswer],
) -> list[QuestionPoint]:
    """Score every trace record against the calibration slice's gold answers.

    Golds come from the slice on disk rather than from the ``gold`` block inside
    the trace, for the reason ``error_analysis.analyse`` gives: the trace's copy
    was written by the run under analysis and the slice was not. A record whose
    qid is absent from the slice is dropped -- it belongs to some other split,
    and scoring it here would mix the two sets this module exists to keep apart.

    Results are returned in qid order so that the bootstrap, which indexes into
    this list, does not inherit the order questions happened to finish in.
    """
    points: list[QuestionPoint] = []
    for record in records:
        qid = str(record.get("qid") or "")
        gold = golds.get(qid)
        if gold is None:
            continue
        answer = str(record.get("final_answer") or "")
        scores = score_question(answer, gold.answer)
        first = _first_cycle(record)
        raw_candidate = first.get("candidate")
        candidate: Mapping[str, Any] = (
            raw_candidate if isinstance(raw_candidate, Mapping) else {}
        )
        raw_metrics = record.get("metrics")
        metrics: Mapping[str, Any] = (
            raw_metrics if isinstance(raw_metrics, Mapping) else {}
        )
        confidence = _clamp(_as_float(record.get("confidence")))
        points.append(
            QuestionPoint(
                qid=qid,
                confidence=confidence,
                first_confidence=_clamp(
                    _as_float(first.get("confidence"), confidence)
                ),
                em=scores.em,
                f1=scores.f1,
                verdict=str(record.get("verdict") or ""),
                replanned=bool(metrics.get("replanned")),
                # 3.5 step 6 re-plans on an insufficient or empty candidate
                # whatever the confidence is. Those questions re-plan at every
                # threshold, so folding them into the threshold's own column
                # would credit the threshold with re-plans it did not cause.
                forced_revise=bool(candidate) and (
                    not bool(candidate.get("sufficient", True))
                    or not str(candidate.get("answer") or "")
                ),
                answered=bool(answer),
            )
        )
    points.sort(key=lambda p: p.qid)
    return points


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ThresholdRow:
    """One candidate threshold, scored as a classifier of answer correctness."""

    threshold: float
    n: int
    n_accept: int
    accept_rate: float
    accept_em: float | None
    accept_f1: float | None
    reject_em: float | None
    reject_f1: float | None
    replan_rate: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    specificity: float
    youden_j: float
    decision_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 2),
            "n": self.n,
            "n_accept": self.n_accept,
            "accept_rate": self.accept_rate,
            "accept_em": self.accept_em,
            "accept_f1": self.accept_f1,
            "reject_em": self.reject_em,
            "reject_f1": self.reject_f1,
            "replan_rate": self.replan_rate,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "youden_j": self.youden_j,
            "decision_accuracy": self.decision_accuracy,
        }


def _mean(values: Sequence[float]) -> float | None:
    """Mean, or ``None`` for an empty set -- which is not the same as zero.

    At a threshold that accepts nothing, "the EM of accepted answers" is
    undefined. Reporting 0.0 there would draw a curve falling to the floor and
    invite the reader to take a measurement off the empty set.
    """
    return sum(values) / len(values) if values else None


def _row(
    points: Sequence[QuestionPoint], threshold: float, label: Label
) -> ThresholdRow:
    n = len(points)
    accepted = [p for p in points if p.confidence >= threshold]
    rejected = [p for p in points if p.confidence < threshold]

    tp = sum(1 for p in accepted if p.correct(label))
    fp = len(accepted) - tp
    fn = sum(1 for p in rejected if p.correct(label))
    tn = len(rejected) - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    # Counterfactual, and only that: it reads the first-cycle confidences this
    # run produced under the configured threshold and asks which of them would
    # have fallen below a different bar. It is not what the re-plan rate would be
    # under that bar, because a different bar produces different confidences.
    replans = sum(
        1 for p in points if p.forced_revise or p.first_confidence < threshold
    )

    return ThresholdRow(
        threshold=threshold,
        n=n,
        n_accept=len(accepted),
        accept_rate=len(accepted) / n if n else 0.0,
        accept_em=_mean([p.em for p in accepted]),
        accept_f1=_mean([p.f1 for p in accepted]),
        reject_em=_mean([p.em for p in rejected]),
        reject_f1=_mean([p.f1 for p in rejected]),
        replan_rate=replans / n if n else 0.0,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        youden_j=recall + specificity - 1.0,
        decision_accuracy=(tp + tn) / n if n else 0.0,
    )


def sweep(
    points: Sequence[QuestionPoint],
    *,
    label: Label = "em",
    grid: Sequence[float] = SWEEP,
) -> list[ThresholdRow]:
    """Score every candidate threshold on the same calibration questions."""
    return [_row(points, t, label) for t in grid]


# ---------------------------------------------------------------------------
# Calibration quality
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of the reliability curve. ``gap`` is what ECE averages."""

    index: int
    lower: float
    upper: float
    n: int
    mean_confidence: float | None
    accuracy: float | None
    gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "lower": self.lower, "upper": self.upper,
            "n": self.n, "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy, "gap": self.gap,
        }


@dataclass(frozen=True, slots=True)
class CalibrationQuality:
    """Two different questions, kept apart because they have different answers.

    **Discrimination** -- does a higher confidence mean a likelier-correct
    answer? That is ``auc``, the probability that a randomly chosen correct
    answer outscores a randomly chosen incorrect one (ties counted as a half).
    It is the property a *threshold* depends on, and it is invariant to any
    monotone rescaling of the confidences.

    **Calibration** -- is the number a probability? That is ``ece``, ``mce`` and
    ``brier``, with ``brier_skill`` comparing the Brier score against the
    constant predictor that always emits the observed base rate.

    Conflating them is the easy mistake and the expensive one. A signal that
    ranks well but is systematically over-confident scores a large ECE and a
    *negative* Brier skill while still separating correct from incorrect
    answers perfectly well; calling that "no usable information" would throw
    away a working threshold. Only ``auc <= 0.5`` licenses that claim, so the
    two are reported, and interpreted, separately.
    """

    n: int
    n_correct: int
    n_incorrect: int
    base_rate: float
    mean_confidence: float
    auc: float | None
    ece: float
    mce: float
    brier: float
    brier_baseline: float
    brier_skill: float | None
    bins: tuple[ReliabilityBin, ...]

    @property
    def well_calibrated(self) -> bool:
        """ECE under 0.10 -- the usual informal bar for "reads as a probability"."""
        return self.ece < 0.10

    @property
    def discriminative(self) -> bool:
        """Does the confidence rank correct answers above incorrect ones at all?

        ``None`` -- every answer correct, or every answer wrong -- is not
        evidence of discrimination, so it is not reported as any.
        """
        return self.auc is not None and self.auc > 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "base_rate": self.base_rate,
            "mean_confidence": self.mean_confidence,
            "auc": self.auc,
            "discriminative": self.discriminative,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "brier_baseline": self.brier_baseline,
            "brier_skill": self.brier_skill,
            "well_calibrated": self.well_calibrated,
            "n_bins": len(self.bins),
            "reliability": [b.to_dict() for b in self.bins],
        }


def brier_score(points: Sequence[QuestionPoint], *, label: Label = "em") -> float:
    """Mean squared error of the confidence against the binary outcome."""
    if not points:
        return 0.0
    return sum(
        (p.confidence - (1.0 if p.correct(label) else 0.0)) ** 2 for p in points
    ) / len(points)


def auc_score(
    points: Sequence[QuestionPoint], *, label: Label = "em"
) -> float | None:
    """Area under the ROC curve, as the Mann--Whitney statistic.

    ``P(conf of a correct answer > conf of an incorrect one)``, with ties
    contributing a half. Computed pairwise rather than by sorting: on 50
    questions the quadratic cost is nothing, and the pairwise form is the
    definition, so it needs no separate tie-handling argument.

    ``None`` when one of the two classes is empty. An AUC is undefined there,
    and 0.5 -- the number a "return chance on degenerate input" convention
    would produce -- reads as a measured coin flip rather than as an absence.
    """
    correct = [p.confidence for p in points if p.correct(label)]
    wrong = [p.confidence for p in points if not p.correct(label)]
    if not correct or not wrong:
        return None
    wins = 0.0
    for c in correct:
        for w in wrong:
            wins += 1.0 if c > w else 0.5 if c == w else 0.0
    return wins / (len(correct) * len(wrong))


def calibration_quality(
    points: Sequence[QuestionPoint],
    *,
    label: Label = "em",
    bins: int = BINS,
) -> CalibrationQuality:
    """Reliability curve, ECE, MCE and Brier over equal-width confidence bins.

    Equal-width rather than equal-mass bins: the report wants to say *where* on
    the confidence scale the signal misleads, and equal-mass bins move their own
    boundaries with the data, so the same bin index would mean something
    different in every run and across the two datasets.
    """
    n = len(points)
    if n == 0:
        empty = tuple(
            ReliabilityBin(i, i / bins, (i + 1) / bins, 0, None, None, None)
            for i in range(bins)
        )
        return CalibrationQuality(
            n=0, n_correct=0, n_incorrect=0, base_rate=0.0, mean_confidence=0.0,
            auc=None, ece=0.0, mce=0.0, brier=0.0, brier_baseline=0.0,
            brier_skill=None, bins=empty,
        )

    buckets: list[list[QuestionPoint]] = [[] for _ in range(bins)]
    for p in points:
        # A confidence of exactly 1.0 belongs in the top bin rather than an
        # eleventh one; the min() is the whole of that special case.
        buckets[min(bins - 1, int(p.confidence * bins))].append(p)

    rows: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for i, bucket in enumerate(buckets):
        lower, upper = i / bins, (i + 1) / bins
        if not bucket:
            rows.append(ReliabilityBin(i, lower, upper, 0, None, None, None))
            continue
        mean_conf = sum(p.confidence for p in bucket) / len(bucket)
        accuracy = sum(1.0 for p in bucket if p.correct(label)) / len(bucket)
        gap = abs(accuracy - mean_conf)
        ece += (len(bucket) / n) * gap
        mce = max(mce, gap)
        rows.append(
            ReliabilityBin(i, lower, upper, len(bucket), mean_conf, accuracy, gap)
        )

    n_correct = sum(1 for p in points if p.correct(label))
    base_rate = n_correct / n
    brier = brier_score(points, label=label)
    baseline = base_rate * (1.0 - base_rate)
    return CalibrationQuality(
        n=n,
        n_correct=n_correct,
        n_incorrect=n - n_correct,
        base_rate=base_rate,
        mean_confidence=sum(p.confidence for p in points) / n,
        auc=auc_score(points, label=label),
        ece=ece,
        mce=mce,
        brier=brier,
        brier_baseline=baseline,
        # A base rate of exactly 0 or 1 makes the constant predictor perfect, so
        # the skill ratio divides by zero. ``None`` says the comparison was not
        # available; 0.0 would say it came out even, which is a measurement.
        brier_skill=(1.0 - brier / baseline) if baseline > 0 else None,
        bins=tuple(rows),
    )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Recommendation:
    """A threshold, an interval, and whether the data supports changing anything."""

    threshold: float
    criterion: str
    youden_j: float
    f1_optimal: float
    f1_optimal_score: float
    ci_low: float
    ci_high: float
    ci_width: float
    samples: int
    seed: int
    n: int
    sufficient: bool
    keep_configured: bool
    configured: float
    configured_inside_ci: bool
    accept_accuracy: BootstrapResult | None
    configured_accept_accuracy: BootstrapResult | None
    decision_delta_p: float | None
    decision_delta_ci: tuple[float, float] | None
    reasoning: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 2),
            "criterion": self.criterion,
            "youden_j": self.youden_j,
            "f1_optimal_threshold": round(self.f1_optimal, 2),
            "f1_optimal_score": self.f1_optimal_score,
            "ci_low": round(self.ci_low, 2),
            "ci_high": round(self.ci_high, 2),
            "ci_width": round(self.ci_width, 2),
            "bootstrap_samples": self.samples,
            "seed": self.seed,
            "n": self.n,
            "sufficient": self.sufficient,
            "keep_configured": self.keep_configured,
            "configured_threshold": self.configured,
            "configured_inside_ci": self.configured_inside_ci,
            "accept_accuracy": (
                self.accept_accuracy.to_dict() if self.accept_accuracy else None
            ),
            "configured_accept_accuracy": (
                self.configured_accept_accuracy.to_dict()
                if self.configured_accept_accuracy else None
            ),
            "decision_accuracy_delta_p": self.decision_delta_p,
            "decision_accuracy_delta_ci": (
                list(self.decision_delta_ci) if self.decision_delta_ci else None
            ),
            "reasoning": list(self.reasoning),
        }


def _argmax_plateau(
    values: Sequence[float], grid: Sequence[float]
) -> tuple[float, float]:
    """``(threshold, best)`` -- the MIDDLE of the tied maximum, not its edge.

    A criterion computed from a finite sample is piecewise constant between two
    observed confidences, so its maximum is a plateau rather than a point. Both
    ends of that plateau sit against a data point that could move under
    resampling; the middle is the choice furthest from either. Taking the first
    index ``max()`` reports instead would bias every recommendation downwards by
    half a plateau, for no reason but argument order.
    """
    if not values:
        return (grid[0] if grid else 0.0), 0.0
    best = max(values)
    # strict: a values/grid length mismatch means the caller scored a different
    # grid from the one it is selecting over, which would silently truncate.
    tied = [g for g, v in zip(grid, values, strict=True) if v >= best - 1e-12]
    return tied[len(tied) // 2], best


def _select(
    points: Sequence[QuestionPoint], label: Label, grid: Sequence[float]
) -> float:
    """The Youden-optimal threshold on one (possibly resampled) set of points."""
    return _argmax_plateau([_row(points, t, label).youden_j for t in grid], grid)[0]


def recommend(
    points: Sequence[QuestionPoint],
    rows: Sequence[ThresholdRow],
    *,
    label: Label = "em",
    configured: float = 0.55,
    grid: Sequence[float] = SWEEP,
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    min_n: int = MIN_RECOMMENDATION_N,
) -> Recommendation:
    """Select a threshold, bound it, and decide whether to act on it.

    Youden's J is the selection criterion rather than the accept-F1: J weights a
    false accept and a false reject equally and does not move with the base
    rate, whereas F1 ignores true negatives entirely and drifts towards
    accepting everything as accuracy rises. The F1-optimal point is reported
    beside it so that a disagreement between the two criteria is visible rather
    than hidden inside the choice of criterion.

    The interval comes from re-running the *entire selection* on bootstrap
    resamples of the calibration questions, not from resampling one fixed
    threshold's score. It therefore answers the question that actually matters:
    how much of this threshold is the data, and how much is these 50 questions.
    """
    n = len(points)
    if n:
        j_threshold, best_j = _argmax_plateau([r.youden_j for r in rows], grid)
        f1_threshold, best_f1 = _argmax_plateau([r.f1 for r in rows], grid)
    else:
        # Nothing was selected, so nothing may be reported as selected. Falling
        # back to the bottom of the grid would print "0.40" in a table cell and
        # make an absence of data look like a decision.
        j_threshold = f1_threshold = configured
        best_j = best_f1 = 0.0

    selections: list[float] = []
    if n:
        rng = random.Random(seed)
        for _ in range(samples):
            resample = [points[rng.randrange(n)] for _ in range(n)]
            selections.append(_select(resample, label, grid))
        selections.sort()

    alpha = (1.0 - confidence) / 2.0
    ci_low = _percentile(selections, alpha) if selections else 0.0
    ci_high = _percentile(selections, 1.0 - alpha) if selections else 0.0
    width = ci_high - ci_low

    def accept_accuracy(threshold: float) -> BootstrapResult | None:
        subset = [p for p in points if p.confidence >= threshold]
        if not subset:
            return None
        return bootstrap_mean(
            [1.0 if p.correct(label) else 0.0 for p in subset],
            samples=samples, seed=seed, confidence=confidence,
        )

    # Is the configured threshold's accept/reject decision distinguishable from
    # the selected one's? Paired over the same questions, because they are the
    # same questions. Note the direction of the bias: the selected threshold was
    # chosen to maximise separation ON THIS DATA, so the comparison is already
    # tilted in its favour -- which is what makes an interval that still spans
    # zero a strong argument for leaving the configured value alone.
    delta_p: float | None = None
    delta_ci: tuple[float, float] | None = None
    if points:
        def decisions(threshold: float) -> dict[str, float]:
            return {
                p.qid: 1.0 if ((p.confidence >= threshold) == p.correct(label)) else 0.0
                for p in points
            }

        comparison = paired_bootstrap(
            decisions(configured), decisions(j_threshold),
            name_a=f"threshold={configured:.2f}",
            name_b=f"threshold={j_threshold:.2f}",
            samples=samples, seed=seed, confidence=confidence,
        )
        delta_p = comparison.p_value
        delta_ci = (comparison.ci_low, comparison.ci_high)

    inside = ci_low - 1e-9 <= configured <= ci_high + 1e-9
    sufficient = n >= min_n and width <= MAX_RECOMMENDATION_CI_WIDTH
    keep = (not sufficient) or inside or (
        delta_ci is not None and delta_ci[0] <= 0.0 <= delta_ci[1]
    )

    reasoning: list[str] = [
        f"Selection criterion: Youden's J, maximised at {j_threshold:.2f} "
        f"(J = {best_j:.3f}); the F1-optimal point is {f1_threshold:.2f} "
        f"(F1 = {best_f1:.3f}).",
        f"Bootstrap over the selection itself ({samples} resamples, seed {seed}): "
        f"95% interval [{ci_low:.2f}, {ci_high:.2f}], width {width:.2f}.",
    ]
    if n == 0:
        reasoning.append(
            "No labelled calibration questions were found, so nothing was selected."
        )
    elif n < min_n:
        reasoning.append(
            f"UNDER-POWERED: {n} labelled questions, below the {min_n} this module "
            "requires before it will propose replacing a configured value. A point "
            "estimate from this many questions looks like evidence and is not. "
            "Note what this means for the verdict below: at this sample size the "
            "module CANNOT return 'change' whatever the data say, so 'keep' here "
            "is a refusal to act on too little evidence, not a positive "
            "endorsement of the configured value. The independent evidence for "
            "the configured value is whether it falls inside the interval."
        )
    if width > MAX_RECOMMENDATION_CI_WIDTH:
        reasoning.append(
            f"The interval on the selected threshold spans {width:.2f} of the "
            f"{grid[0]:.2f}-{grid[-1]:.2f} swept range, which does not identify an "
            "operating point."
        )
    if inside:
        reasoning.append(
            f"The configured threshold {configured:.2f} lies inside that interval: "
            "the calibration data does not distinguish it from the optimum."
        )
    else:
        reasoning.append(
            f"The configured threshold {configured:.2f} lies OUTSIDE the interval, "
            "which is evidence -- on this slice -- that it is mis-set."
        )
    if delta_ci is not None and delta_p is not None:
        reasoning.append(
            "Paired bootstrap of accept/reject decision accuracy, "
            f"{j_threshold:.2f} minus {configured:.2f}: "
            f"[{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}], p = {delta_p:.3f}. The "
            "selected threshold was fitted on these same questions, so this "
            "difference is optimistically biased in its favour."
        )
    reasoning.append(
        f"RECOMMENDATION: {'keep' if keep else 'change'} confidence_threshold = "
        f"{(configured if keep else j_threshold):.2f}."
    )
    reasoning.append(
        "This is a post-hoc sweep over confidences produced under "
        f"threshold={configured:.2f}. It measures how well the confidence signal "
        "separates correct from incorrect answers; it does NOT measure what the "
        "re-plan rate, or the answers themselves, would become under a different "
        "threshold -- that needs one full run per threshold."
    )

    return Recommendation(
        threshold=j_threshold,
        criterion="youden_j",
        youden_j=best_j,
        f1_optimal=f1_threshold,
        f1_optimal_score=best_f1,
        ci_low=ci_low, ci_high=ci_high, ci_width=width,
        samples=samples, seed=seed, n=n,
        sufficient=sufficient,
        keep_configured=keep,
        configured=configured,
        configured_inside_ci=inside,
        accept_accuracy=accept_accuracy(j_threshold),
        configured_accept_accuracy=accept_accuracy(configured),
        decision_delta_p=delta_p,
        decision_delta_ci=delta_ci,
        reasoning=tuple(reasoning),
    )


# ---------------------------------------------------------------------------
# The full report
# ---------------------------------------------------------------------------

#: Reproduced verbatim into the JSON summary and into the table caption. The
#: sweep is only honest if the limitation travels with the number.
LIMITATIONS: tuple[str, ...] = (
    "Post-hoc: every confidence in this table was produced by a run executed at "
    "the configured threshold. Changing the threshold changes whether a re-plan "
    "fires, which changes the plan, the evidence and the answer -- so the "
    "accepted answers at threshold t are not the answers a run at threshold t "
    "would have produced.",
    "The 'replan_rate' column is a counterfactual read of the first-cycle "
    "confidences, not a measurement of what the re-plan rate would be.",
    "A true sweep requires one full evaluation run per candidate threshold "
    "(~1.7 h each, architecture risk 2), which the compute budget does not "
    "support.",
    "What the sweep does establish: how well the confidence signal, as produced, "
    "separates correct answers from incorrect ones, and whether the configured "
    "boundary sits where that separation is best.",
    "The calibration slice is disjoint from the frozen 250-question evaluation "
    "slice by construction, so nothing here is tuned on the test set.",
    "Fifty questions is a small sample: the recommendation is reported with a "
    "bootstrap interval over the selected threshold and is withheld when that "
    "interval fails to identify an operating point.",
    "Calibration (ECE, Brier) and discrimination (AUC) are different properties "
    "and are reported separately. A large ECE says the confidence is not a "
    "probability; only an AUC at or below 0.5 would say it carries no "
    "information about correctness. Neither statement implies the other.",
    "The selected threshold is fitted on the same questions it is then scored "
    "on, so its apparent advantage over the configured value is optimistically "
    "biased; an interval that still spans zero under that bias is a strong "
    "argument for leaving the configured value alone.",
    "'n_questions' is what was scored, 'n_slice' what the slice holds. When "
    "they differ the run was still in flight and every number here is computed "
    "on a subset of the calibration slice.",
)


@dataclass(slots=True)
class CalibrationReport:
    """Everything the JSON artefact and the LaTeX fragment are rendered from."""

    dataset: str
    run_id: str
    run_dir: Path | None
    split: str
    config_name: str
    label: Label
    configured: float
    weights: dict[str, float]
    uncertainty_band: float
    #: Questions in the calibration slice on disk, or 0 when it could not be
    #: read. Held beside ``n_questions`` so a run still in flight is visible as
    #: a coverage fraction rather than as a smaller, healthier-looking sample.
    n_slice: int = 0
    points: list[QuestionPoint] = field(default_factory=list)
    rows: list[ThresholdRow] = field(default_factory=list)
    quality: CalibrationQuality | None = None
    recommendation: Recommendation | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_questions(self) -> int:
        return len(self.points)

    @property
    def complete(self) -> bool:
        """Has every question in the slice been scored? Unknown counts as no."""
        return bool(self.n_slice) and self.n_questions >= self.n_slice

    @property
    def coverage(self) -> float | None:
        """Scored fraction of the slice, or ``None`` when the slice size is unknown."""
        return self.n_questions / self.n_slice if self.n_slice else None

    def row_at(self, threshold: float) -> ThresholdRow | None:
        """The swept row for one threshold, matched on the 0.01 grid."""
        return next((r for r in self.rows if abs(r.threshold - threshold) < 5e-3), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "generated_by": "src/agentic_ir/eval/calibrate.py",
            "dataset": self.dataset,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "split": self.split,
            "config_name": self.config_name,
            "n_questions": self.n_questions,
            "n_slice": self.n_slice,
            "coverage": self.coverage,
            "complete": self.complete,
            "correctness_label": {
                "kind": self.label,
                "definition": LABEL_DEFINITIONS[self.label],
            },
            "configured_threshold": self.configured,
            "verifier_weights": self.weights,
            "uncertainty_band": self.uncertainty_band,
            "grid": {
                "low": SWEEP[0], "high": SWEEP[-1], "step": 0.01, "n": len(SWEEP),
            },
            "limitations": list(LIMITATIONS),
            "notes": list(self.notes),
            "calibration_quality": self.quality.to_dict() if self.quality else None,
            "recommendation": (
                self.recommendation.to_dict() if self.recommendation else None
            ),
            "sweep": [r.to_dict() for r in self.rows],
        }


def calibrate(
    points: Sequence[QuestionPoint],
    *,
    dataset: str,
    run_id: str = "",
    run_dir: Path | None = None,
    split: str = "calib",
    config_name: str = "",
    cfg: Config | None = None,
    label: Label = "em",
    n_slice: int = 0,
    samples: int | None = None,
    seed: int | None = None,
) -> CalibrationReport:
    """Sweep, measure calibration quality, and recommend -- all from one seed.

    The notes this attaches are the sentences the report should be able to lift
    verbatim: whether the signal is calibrated, whether it carries any
    information at all, and whether the confidences are too concentrated for the
    binned estimate to mean much.
    """
    cfg = cfg or load_config()
    samples = (
        int(cfg.get("evaluation.bootstrap_samples", 1000)) if samples is None
        else samples
    )
    seed = int(cfg.get("project.seed", 42)) if seed is None else seed
    configured = float(cfg.get("agents.verifier.confidence_threshold", 0.55))
    weights = {
        str(k): float(v)
        for k, v in (cfg.get("agents.verifier.weights", {}) or {}).items()
    }

    # No questions means no swept table, not 36 rows of zeros: a precision of
    # 0.000 and a Youden J of -1.000 computed over an empty set are arithmetic,
    # not measurements, and they render identically to measured failures.
    rows = sweep(points, label=label) if points else []
    quality = calibration_quality(points, label=label)
    recommendation = recommend(
        points, rows, label=label, configured=configured, samples=samples, seed=seed,
    )

    notes: list[str] = []
    if not points:
        notes.append(
            "No calibration questions were scored: no recommendation is possible."
        )

    # The run this reads may still be in flight. A reader who sees only
    # ``n_questions`` has no way to tell 50-of-50 from 17-of-50, and the second
    # is a materially weaker result reported in the same shape as the first.
    if n_slice and len(points) < n_slice:
        notes.insert(
            0,
            f"PARTIAL RUN: {len(points)} of the {n_slice} questions in the "
            f"calibration slice have been scored ({len(points) / n_slice:.0%}). "
            "Every number below is computed on that subset, and the sweep should "
            "be regenerated when the run completes.",
        )

    # Calibration and discrimination are reported as the two separate findings
    # they are; see CalibrationQuality's docstring for why conflating them
    # would be the expensive mistake here.
    if quality.n and not quality.well_calibrated:
        notes.append(
            f"ECE = {quality.ece:.3f} (>= 0.10): the confidence blend is NOT well "
            "calibrated as a probability. The threshold is usable as a ranking "
            "cut-point only, and the report must not describe it as a probability."
        )
    elif quality.n:
        notes.append(
            f"ECE = {quality.ece:.3f} (< 0.10): the confidence blend reads as an "
            "approximately calibrated probability on this slice."
        )
    if quality.n and quality.brier_skill is not None and quality.brier_skill <= 0.0:
        notes.append(
            f"Brier skill = {quality.brier_skill:+.3f} against the base-rate "
            "constant predictor: as *probabilities* the confidences are worse "
            "than a system that always predicts the base rate. That is a "
            "statement about their scale, not about their ordering -- read it "
            "with the AUC below, not instead of it."
        )
    if quality.n and quality.brier_skill is None:
        notes.append(
            f"Every scored answer was {'correct' if quality.base_rate else 'incorrect'}: "
            "with one class empty the base-rate baseline is degenerate and neither "
            "Brier skill nor AUC is defined."
        )
    if quality.auc is None and quality.n:
        notes.append(
            "AUC is undefined on this slice (one of the two outcome classes is "
            "empty), so the confidence signal's ability to separate correct from "
            "incorrect answers could not be measured at all."
        )
    elif quality.auc is not None and quality.auc <= 0.5:
        notes.append(
            f"AUC = {quality.auc:.3f} (<= 0.5): the confidence signal does not "
            "rank correct answers above incorrect ones, so it carries no usable "
            "information about correctness on this slice and NO threshold read "
            "off it is meaningful."
        )
    elif quality.auc is not None and quality.auc < 0.7:
        notes.append(
            f"AUC = {quality.auc:.3f}: the confidence signal separates correct "
            "from incorrect answers better than chance but weakly. A threshold "
            "on it is a real but blunt instrument, and the sweep's optimum is "
            "correspondingly ill-determined."
        )
    elif quality.auc is not None:
        notes.append(
            f"AUC = {quality.auc:.3f}: the confidence signal ranks correct "
            "answers above incorrect ones, so an accept/reject threshold on it "
            "is doing real work even where the probabilities are miscalibrated."
        )
    occupied = sum(1 for b in quality.bins if b.n)
    if quality.n and occupied <= 3:
        notes.append(
            f"Only {occupied} of {len(quality.bins)} confidence bins are occupied: "
            "the confidences are concentrated, so ECE and the sweep are estimated "
            "from very few distinct operating points."
        )

    return CalibrationReport(
        dataset=dataset, run_id=run_id, run_dir=run_dir, split=split,
        config_name=config_name, label=label, configured=configured,
        weights=weights,
        uncertainty_band=float(cfg.get("agents.verifier.uncertainty_band", 0.15)),
        n_slice=n_slice,
        points=list(points), rows=rows, quality=quality,
        recommendation=recommendation, notes=notes,
    )


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_calibration_run(
    dataset: str,
    *,
    cfg: Config | None = None,
    root: Path | None = None,
    run_id: str | None = None,
    config_name: str = "agentic_full",
) -> Path | None:
    """The newest run directory holding the calibration split for ``dataset``.

    ``meta.json``'s ``split`` field is the filter, not the directory name: a run
    id says nothing about which slice it consumed, and calibrating on a run that
    silently used the evaluation 250 is exactly the leakage this module exists
    to prevent. A directory whose ``meta.json`` is missing or unreadable cannot
    vouch for its own split, so it is skipped rather than assumed innocent.
    """
    import orjson

    cfg = cfg or load_config()
    base = Path(root) if root is not None else runs_root(cfg)
    if not base.is_dir():
        return None
    if run_id:
        candidate = base / run_id
        return candidate if candidate.is_dir() else None

    prefix = f"{config_name}_{dataset}_"
    for path in sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.name, reverse=True,
    ):
        meta_path = path / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = orjson.loads(meta_path.read_bytes())
        except Exception:  # noqa: BLE001 - an unreadable meta cannot vouch for a split
            continue
        if str(meta.get("split") or "") != "calib":
            continue
        # A run that died before its first question would otherwise shadow a
        # complete earlier one and calibrate on nothing.
        if any(True for _ in iter_records(path / "traces.jsonl")):
            return path
    return None


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

#: Which thresholds get a printed row. The full 36-row sweep lives in the JSON;
#: a printed table nobody can scan is not a communication.
_TEX_LADDER: tuple[float, ...] = tuple(bp / 100.0 for bp in range(40, 80, 5))


def _tex_thresholds(report: CalibrationReport) -> list[float]:
    """The printed thresholds: the 0.05 ladder, plus the configured and selected."""
    wanted = {round(t, 2) for t in _TEX_LADDER}
    wanted.add(round(report.configured, 2))
    if report.recommendation is not None:
        wanted.add(round(report.recommendation.threshold, 2))
        wanted.add(round(report.recommendation.f1_optimal, 2))
    return sorted(t for t in wanted if SWEEP[0] - 1e-9 <= t <= SWEEP[-1] + 1e-9)


def _preamble(report: CalibrationReport) -> str:
    """Provenance header, carrying no wall-clock timestamp.

    Same rule as ``tables.py``: regenerating from an unchanged run must produce a
    byte-identical file, so a diff under ``results/tables/`` always means a
    number actually moved.
    """
    lines = [
        "% Chapter 4 -- verifier threshold calibration",
        "% GENERATED by src/agentic_ir/eval/calibrate.py -- do not edit by hand.",
        "% Regenerate with: python -m agentic_ir.eval.calibrate "
        f"--dataset {report.dataset}",
        "% Requires \\usepackage{booktabs} and \\usepackage{graphicx};",
        "% report/main.tex already loads both.",
        f"% Source run: {report.dataset}/{report.config_name or '?'}: "
        f"{report.run_id or 'none'} "
        f"({report.n_questions} questions, split={report.split})",
        "% POST-HOC SWEEP. See the caption, and "
        f"results/calibration/{report.dataset}_threshold.json,",
        "% for what it can and cannot establish.",
    ]
    return "\n".join(lines) + "\n\n"


def latex_fragment(report: CalibrationReport) -> str:
    r"""Two ``booktabs`` tables: the sweep (``tab:calibration``) and reliability.

    The caption carries the post-hoc limitation rather than a footnote, because a
    table travels: it is read on its own page, projected in a viva, and pasted
    into slides, and the qualification has to travel with it.
    """
    rec = report.recommendation
    quality = report.quality

    marks: dict[float, str] = {}
    if rec is not None:
        marks[round(rec.threshold, 2)] = r"$^{\ast}$"
        marks.setdefault(round(rec.f1_optimal, 2), r"$^{\ddagger}$")
    key = round(report.configured, 2)
    marks[key] = marks.get(key, "") + r"$^{\dagger}$"

    body: list[Sequence[str]] = []
    for t in _tex_thresholds(report):
        row = report.row_at(t)
        if row is None:
            continue
        body.append([
            f"{t:.2f}" + marks.get(round(t, 2), ""),
            str(row.n_accept),
            fmt(row.accept_rate, 2),
            fmt(row.accept_em),
            fmt(row.accept_f1),
            fmt(row.reject_em),
            fmt(row.replan_rate, 2),
            fmt(row.precision),
            fmt(row.recall),
            fmt(row.youden_j),
        ])

    if rec is None or report.n_questions == 0:
        summary = "No calibration run was found, so no threshold is recommended."
    elif rec.keep_configured:
        summary = (
            f"Recommendation: KEEP {report.configured:.2f}. The $J$-optimal point "
            f"on this slice is {rec.threshold:.2f}, 95\\% interval "
            f"[{rec.ci_low:.2f}, {rec.ci_high:.2f}] over {rec.samples} bootstrap "
            f"resamples of the {rec.n} calibration questions"
            + (
                "; that interval is too wide, and the sample too small, to justify "
                "replacing a configured value."
                if not rec.sufficient else
                "; the configured value is not distinguishable from it."
            )
        )
    else:
        summary = (
            f"Recommendation: CHANGE to {rec.threshold:.2f}, 95\\% interval "
            f"[{rec.ci_low:.2f}, {rec.ci_high:.2f}] over {rec.samples} bootstrap "
            f"resamples of the {rec.n} calibration questions."
        )

    # A partial run is reported in the caption, not only in the JSON. The table
    # is the artefact that travels, and "17 questions" reads as a small study
    # while "17 of 50" reads as an unfinished one -- which is what it is.
    scope = (
        f"{report.n_questions} of {report.n_slice} questions scored so far"
        if report.n_slice and not report.complete
        else f"{report.n_questions} questions"
    )
    caption = (
        "Post-hoc threshold sweep for the Verifier's confidence blend on the "
        f"{latex_escape(report.dataset)} calibration slice "
        f"({scope}, disjoint from the evaluation 250). "
        r"$\dagger$~configured; $\ast$~Youden-optimal; $\ddagger$~F1-optimal. "
        "\\textbf{This sweep is post-hoc:} every confidence was produced by a "
        f"single run at threshold {report.configured:.2f}, and changing the "
        "threshold would change whether a re-plan fires and therefore the answers "
        "themselves. It shows where the accept/reject boundary best separates "
        "correct from incorrect answers \\emph{given the confidences this system "
        "produced}; it does not predict the re-plan rate or the accuracy of a run "
        "executed at a different threshold, which would require one run per "
        "threshold. The re-plan column is a counterfactual read of the first-cycle "
        "confidences. " + summary
    )

    sweep_table = table(
        colspec="lrrrrrrrrr",
        header=[
            "Threshold", "$n_{\\mathrm{acc}}$", "Acc.\\ rate",
            "EM (acc.)", "F1 (acc.)", "EM (rej.)", "Re-plan rate",
            "Prec.", "Rec.", "Youden $J$",
        ],
        groups=[(None, body or [[MISSING] * 10])],
        caption=caption,
        label="tab:calibration",
    )

    if quality is None or quality.n == 0:
        rel_rows: list[Sequence[str]] = [[MISSING] * 4]
        quality_caption = (
            "Reliability of the Verifier's confidence blend: no calibration run "
            "was found, so no bin could be filled."
        )
    else:
        rel_rows = [
            [
                f"{{[}}{b.lower:.1f}, {b.upper:.1f})",  # braced: a bare [ after \ is an optional arg to TeX
                str(b.n),
                fmt(b.mean_confidence),
                fmt(b.accuracy),
            ]
            for b in quality.bins
        ]
        skill = (
            f" (skill {quality.brier_skill:+.3f})"
            if quality.brier_skill is not None else
            " (skill undefined: one outcome class is empty)"
        )
        # Calibration and discrimination, in that order and named apart. A
        # reader who takes the large ECE as proof the threshold is worthless
        # has drawn the wrong conclusion, and the caption is where that gets
        # forestalled rather than in a paragraph the table travels without.
        discrimination = (
            "Discrimination is a separate question from calibration: "
            f"AUC $=$ {quality.auc:.3f}, i.e.\\ the confidence ranks a correct "
            "answer above an incorrect one that often. "
            + (
                "The ordering is informative even though the scale is not, so a "
                "threshold on it is meaningful as a cut-point."
                if quality.auc > 0.5 else
                "At or below chance the ordering carries no information about "
                "correctness, and no threshold read off it is meaningful."
            )
            if quality.auc is not None else
            "AUC is undefined here because one of the two outcome classes is "
            "empty, so the signal's discrimination could not be measured."
        )
        quality_caption = (
            "Reliability of the Verifier's confidence blend on the "
            f"{latex_escape(report.dataset)} calibration slice: "
            f"ECE $=$ {quality.ece:.3f} over {len(quality.bins)} equal-width bins, "
            f"MCE $=$ {quality.mce:.3f}, Brier $=$ {quality.brier:.3f} against a "
            f"base-rate baseline of {quality.brier_baseline:.3f}{skill}. "
            + (
                "The signal reads as an approximately calibrated probability."
                if quality.well_calibrated else
                "The signal is \\emph{not} well calibrated as a probability, so the "
                "threshold must be read as a ranking cut-point rather than as a "
                "probability of correctness."
            )
            + " " + discrimination
            + " An empty bin renders as a zero count and a \\texttt{"
            + MISSING
            + "} accuracy, never as an accuracy of zero."
        )

    reliability_table = table(
        colspec="lrrr",
        header=["Confidence bin", "$n$", "Mean conf.", "Accuracy"],
        groups=[(None, rel_rows)],
        caption=quality_caption,
        label="tab:calibration-reliability",
    )

    return _preamble(report) + sweep_table + "\n\n" + reliability_table + "\n"


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def write_artefacts(
    report: CalibrationReport,
    *,
    cfg: Config | None = None,
    json_path: Path | None = None,
    tex_path: Path | None = None,
) -> dict[str, Path]:
    """Write the JSON summary and the LaTeX fragment. Returns ``{kind: path}``.

    Both are written even when no run was found: a chapter's ``\\input`` has to
    resolve, or the report stops building for a reason unrelated to the results.
    """
    cfg = cfg or load_config()
    results = cfg.resolve_path("paths.results")
    target_json = json_path or (
        results / "calibration" / f"{report.dataset}_threshold.json"
    )
    target_tex = tex_path or (results / "tables" / "calibration.tex")

    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_tex.parent.mkdir(parents=True, exist_ok=True)

    with target_json.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, sort_keys=False)
        fh.write("\n")
    with target_tex.open("w", encoding="utf-8") as fh:
        fh.write(latex_fragment(report))

    return {"json": target_json, "tex": target_tex}


def run(
    dataset: str,
    *,
    cfg: Config | None = None,
    run_id: str | None = None,
    runs_dir: Path | None = None,
    config_name: str = "agentic_full",
    label: Label = "em",
    samples: int | None = None,
    seed: int | None = None,
    json_path: Path | None = None,
    tex_path: Path | None = None,
) -> tuple[CalibrationReport, dict[str, Path]]:
    """Locate the calibration run, build the report, write both artefacts."""
    cfg = cfg or load_config()
    run_dir = discover_calibration_run(
        dataset, cfg=cfg, root=runs_dir, run_id=run_id, config_name=config_name
    )

    records = list(iter_records(run_dir / "traces.jsonl")) if run_dir else []
    try:
        golds = {g.qid: g for g in load_eval_set(dataset, split="calib", cfg=cfg)}
    except (FileNotFoundError, KeyError, ValueError):
        golds = {}

    points = load_points(records, golds)

    meta: dict[str, Any] = {}
    if run_dir is not None and (run_dir / "meta.json").exists():
        import orjson

        try:
            meta = orjson.loads((run_dir / "meta.json").read_bytes())
        except Exception:  # noqa: BLE001 - a broken meta must not lose the sweep
            meta = {}

    report = calibrate(
        points,
        dataset=dataset,
        run_id=str(meta.get("run_id") or (run_dir.name if run_dir else "")),
        run_dir=run_dir,
        split=str(meta.get("split") or ("calib" if run_dir else "none")),
        config_name=str(meta.get("config_name") or config_name),
        cfg=cfg, label=label, n_slice=len(golds), samples=samples, seed=seed,
    )

    if run_dir is None:
        report.notes.insert(
            0,
            f"No {config_name} run on the {dataset} calibration split was found "
            "under the runs root.",
        )
    elif not golds:
        report.notes.insert(
            0,
            f"The calibration slice data/processed/{dataset}_calib_50.jsonl could "
            "not be loaded, so no question could be scored.",
        )
    elif len(points) < len(records):
        report.notes.append(
            f"{len(records) - len(points)} of {len(records)} trace records had no "
            "matching qid in the calibration slice and were dropped."
        )
    if run_dir is not None and str(meta.get("split") or "") not in ("calib", ""):
        report.notes.insert(
            0,
            f"WARNING: run {report.run_id} reports split="
            f"{meta.get('split')!r}, not 'calib'. Calibrating on the evaluation "
            "slice is leakage; this result must not be used.",
        )

    return report, write_artefacts(
        report, cfg=cfg, json_path=json_path, tex_path=tex_path
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentic_ir.eval.calibrate",
        description=(
            "Post-hoc sweep of agents.verifier.confidence_threshold over the "
            "disjoint 50-question calibration slice. Reports what the sweep can "
            "and cannot establish alongside the number."
        ),
    )
    parser.add_argument("--dataset", choices=DATASETS, default="hotpotqa",
                        help="dataset whose calibration slice to sweep")
    parser.add_argument("--run-id", default=None,
                        help="a specific run directory name (default: newest calib run)")
    parser.add_argument("--config-name", default="agentic_full",
                        help="configuration whose run to calibrate on")
    parser.add_argument("--runs", type=Path, default=None, help="run directory root")
    parser.add_argument("--label", choices=("em", "f1"), default="em",
                        help="what counts as a correct answer (default: em)")
    parser.add_argument("--bootstrap", type=int, default=None,
                        help="resamples (default: evaluation.bootstrap_samples)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed (default: project.seed)")
    parser.add_argument("--json", dest="json_path", type=Path, default=None,
                        help="override the JSON summary path")
    parser.add_argument("--tex", dest="tex_path", type=Path, default=None,
                        help="override the LaTeX fragment path")
    return parser


def _relative(path: Path) -> Path | str:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, written = run(
        args.dataset,
        cfg=load_config(),
        run_id=args.run_id,
        runs_dir=args.runs,
        config_name=args.config_name,
        label=args.label,
        samples=args.bootstrap,
        seed=args.seed,
        json_path=args.json_path,
        tex_path=args.tex_path,
    )

    print(f"calibration run : {report.run_id or '(none found)'}")
    scope = (
        f"{report.n_questions} of {report.n_slice}" if report.n_slice
        else str(report.n_questions)
    )
    print(
        f"questions       : {scope} (split={report.split}"
        + ("" if report.complete or not report.n_slice else ", PARTIAL RUN")
        + ")"
    )
    quality = report.quality
    if quality is not None and quality.n:
        skill = (
            f"{quality.brier_skill:+.3f}" if quality.brier_skill is not None
            else "n/a"
        )
        print(
            f"calibration     : ECE {quality.ece:.3f}  MCE {quality.mce:.3f}  "
            f"Brier {quality.brier:.3f}  skill {skill}  "
            f"base rate {quality.base_rate:.3f}"
        )
        print(
            "discrimination  : AUC "
            + (f"{quality.auc:.3f}" if quality.auc is not None else "n/a")
            + f"  ({quality.n_correct} correct / {quality.n_incorrect} incorrect)"
        )
    rec = report.recommendation
    if rec is not None:
        print(
            f"threshold       : {rec.threshold:.2f} "
            f"[{rec.ci_low:.2f}, {rec.ci_high:.2f}]  "
            f"(configured {rec.configured:.2f}: "
            f"{'KEEP' if rec.keep_configured else 'CHANGE'})"
        )
        for line in rec.reasoning:
            print(f"  - {line}")
    for note in report.notes:
        print(f"  ! {note}")
    for kind in ("json", "tex"):
        if kind in written:
            print(f"wrote {_relative(written[kind])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
