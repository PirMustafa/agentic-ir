"""Paired bootstrap resampling for confidence intervals and significance.

With a 250-question evaluation sample, a two-point difference between systems
is often noise. This is what lets Chapter 4 say a gap is real instead of
implying it, and what keeps an honest negative result honest.

The resampling is PAIRED: both systems are scored on the same resampled set of
questions on every iteration. Bootstrapping them independently would inflate
the variance of the difference and hide real effects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Point estimate and interval for a single system's metric."""

    mean: float
    ci_low: float
    ci_high: float
    n: int
    samples: int
    confidence: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def format(self, digits: int = 3) -> str:
        return (
            f"{self.mean:.{digits}f} "
            f"[{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}]"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "ci_width": self.ci_width, "n": self.n, "samples": self.samples,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """A paired comparison between two systems on one metric."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    delta: float               # b - a, so positive means B is better
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    samples: int
    confidence: float

    @property
    def significant(self) -> bool:
        """True when the interval for the difference excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def format(self, digits: int = 3) -> str:
        star = "*" if self.significant else "ns"
        return (
            f"{self.name_b} - {self.name_a} = {self.delta:+.{digits}f} "
            f"[{self.ci_low:+.{digits}f}, {self.ci_high:+.{digits}f}] "
            f"p={self.p_value:.4f} ({star})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name_a": self.name_a, "name_b": self.name_b,
            "mean_a": self.mean_a, "mean_b": self.mean_b, "delta": self.delta,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p_value": self.p_value, "significant": self.significant,
            "n": self.n, "samples": self.samples, "confidence": self.confidence,
        }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_mean(
    values: Sequence[float],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap confidence interval for the mean of per-question scores."""
    n = len(values)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0, samples, confidence)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        mean=sum(values) / n,
        ci_low=_percentile(means, alpha),
        ci_high=_percentile(means, 1.0 - alpha),
        n=n,
        samples=samples,
        confidence=confidence,
    )


def paired_bootstrap(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    name_a: str = "A",
    name_b: str = "B",
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ComparisonResult:
    """Paired bootstrap comparison of two systems, keyed by question id.

    Only questions present in BOTH systems are compared -- a question one
    system skipped is not evidence about the other. Question ids are sorted
    before resampling so the result is reproducible regardless of dict order.

    The p-value is two-sided, computed as the fraction of resamples whose
    difference falls on the opposite side of zero from the observed
    difference, doubled and clamped to 1.0.
    """
    qids = sorted(set(scores_a) & set(scores_b))
    n = len(qids)
    if n == 0:
        return ComparisonResult(name_a, name_b, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0,
                                samples, confidence)

    a = [scores_a[q] for q in qids]
    b = [scores_b[q] for q in qids]
    observed = (sum(b) - sum(a)) / n

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            total += b[i] - a[i]
        deltas.append(total / n)
    deltas.sort()

    alpha = (1.0 - confidence) / 2.0
    ci_low = _percentile(deltas, alpha)
    ci_high = _percentile(deltas, 1.0 - alpha)

    if observed >= 0:
        tail = sum(1 for d in deltas if d <= 0.0)
    else:
        tail = sum(1 for d in deltas if d >= 0.0)
    p_value = min(1.0, 2.0 * tail / samples)

    return ComparisonResult(
        name_a=name_a, name_b=name_b,
        mean_a=sum(a) / n, mean_b=sum(b) / n, delta=observed,
        ci_low=ci_low, ci_high=ci_high, p_value=p_value,
        n=n, samples=samples, confidence=confidence,
    )


__all__ = ["BootstrapResult", "ComparisonResult", "bootstrap_mean", "paired_bootstrap"]
