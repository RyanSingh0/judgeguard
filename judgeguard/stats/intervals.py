"""Interval estimation. Built before the experiments, on purpose.

The rule for the whole project: **if you cannot put an interval on it, you do
not report it.** Building this module first is what makes that rule enforceable
rather than aspirational -- there is no code path in the repo that emits a bare
point estimate into a result file.

BCa (bias-corrected and accelerated) rather than percentile bootstrap, because
accuracy near 0 or 1 is skewed and the percentile interval is visibly wrong
there. BCa corrects for both median bias and skew.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import bootstrap, norm


@dataclass(frozen=True, slots=True)
class Estimate:
    """A number that is allowed to appear in the README."""

    value: float
    lo: float
    hi: float
    n: int
    method: str = "bca"
    confidence: float = 0.95

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    def pct(self, dp: int = 1) -> str:
        return f"{100 * self.value:.{dp}f}% [{100 * self.lo:.{dp}f}, {100 * self.hi:.{dp}f}]"

    @property
    def width(self) -> float:
        return self.hi - self.lo


def bca_ci(
    values: Sequence[float],
    statistic: Callable[..., float] = np.mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Estimate:
    """BCa bootstrap CI. Use for every headline number."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    point = float(statistic(arr)) if n else float("nan")
    if n < 3 or np.allclose(arr, arr[0]):
        # Degenerate sample: fall back to a Wilson interval for the 0/1 case so
        # a constant column still reports an honest interval rather than a point.
        if n and set(np.unique(arr)).issubset({0.0, 1.0}):
            lo, hi = wilson_ci(int(arr.sum()), n, confidence=confidence)
            return Estimate(point, lo, hi, n, method="wilson", confidence=confidence)
        return Estimate(point, point, point, n, method="degenerate", confidence=confidence)
    rng = np.random.default_rng(seed)
    try:
        res = bootstrap(
            (arr,),
            statistic,
            confidence_level=confidence,
            method="BCa",
            n_resamples=n_resamples,
            random_state=rng,
            vectorized=False,
        )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
        if not np.isfinite([lo, hi]).all():
            raise ValueError
        return Estimate(point, lo, hi, n, method="bca", confidence=confidence)
    except Exception:
        idx = rng.integers(0, n, size=(n_resamples, n))
        draws = np.array([statistic(arr[i]) for i in idx])
        a = (1 - confidence) / 2
        lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)])
        return Estimate(point, float(lo), float(hi), n, method="percentile", confidence=confidence)


def wilson_ci(successes: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Correct where the normal approximation is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


@dataclass(frozen=True, slots=True)
class PairedDiff:
    diff: float
    lo: float
    hi: float
    p_value: float
    n: int
    significant: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

    def __str__(self) -> str:
        verdict = "real" if self.significant else "inside the interval (noise)"
        return (
            f"Δ={self.diff:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}] p={self.p_value:.4f} — {verdict}"
        )


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Callable[..., float] = np.mean,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedDiff:
    """Is A better than B?

    Resamples ITEM INDICES jointly so the pairing survives the bootstrap. Two
    judges scored on the same items are not independent samples, and treating
    them as such inflates the variance and hides real differences.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size != y.size:
        raise ValueError("paired test needs aligned per-item scores")
    n = x.size
    if n == 0:
        return PairedDiff(float("nan"), float("nan"), float("nan"), float("nan"), 0, False)
    observed = float(statistic(x) - statistic(y))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = np.array([statistic(x[i]) - statistic(y[i]) for i in idx])
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    centred = diffs - diffs.mean()
    p = float((np.abs(centred) >= abs(observed)).mean())
    return PairedDiff(observed, float(lo), float(hi), p, n, bool(lo > 0 or hi < 0))


def mean_ci(values: Sequence[float], **kw: object) -> Estimate:
    return bca_ci(values, np.mean, **kw)  # type: ignore[arg-type]


__all__ = ["Estimate", "PairedDiff", "bca_ci", "mean_ci", "paired_bootstrap_diff", "wilson_ci"]
