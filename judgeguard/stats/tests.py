"""Hypothesis tests, agreement coefficients and multiplicity control.

The multiplicity correction is here because the bias battery runs dozens of
paired comparisons. At 40 tests and alpha = 0.05 you expect two false positives
by construction, so an uncorrected "p < 0.05" in this repo would be a bug rather
than a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True, slots=True)
class McNemarResult:
    b: int  # A correct, B wrong
    c: int  # B correct, A wrong
    statistic: float
    p_value: float
    exact: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def mcnemar_test(correct_a: Sequence[int], correct_b: Sequence[int]) -> McNemarResult:
    """Paired binary decisions on the same items. Exact when discordants are few."""
    b = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if y and not x)
    exact = (b + c) < 25
    if b + c == 0:
        return McNemarResult(b, c, 0.0, 1.0, exact)
    res = mcnemar([[0, b], [c, 0]], exact=exact, correction=not exact)
    return McNemarResult(b, c, float(res.statistic), float(res.pvalue), exact)


def judge_agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa between two judges' discrete verdicts."""
    if len(set(a)) < 2 and len(set(b)) < 2:
        return 1.0 if list(a) == list(b) else 0.0
    return float(cohen_kappa_score(list(a), list(b)))


def krippendorff_alpha_nominal(matrix: Sequence[Sequence[Any]]) -> float:
    """Krippendorff's alpha for nominal data, tolerant of missing values (None).

    ``matrix`` is raters x items. Preferred over pairwise kappa when comparing
    more than two raters -- which is exactly the self-consistency setting, where
    the raters are five reruns of one judge.
    """
    raters = [list(r) for r in matrix]
    n_items = len(raters[0]) if raters else 0
    values: list[Any] = []
    per_item: list[list[Any]] = []
    for j in range(n_items):
        col = [r[j] for r in raters if r[j] is not None]
        if len(col) > 1:
            per_item.append(col)
            values.extend(col)
    if not per_item:
        return float("nan")
    cats = sorted({str(v) for v in values})
    counts = {c: sum(1 for v in values if str(v) == c) for c in cats}
    n_total = len(values)
    do_num = 0.0
    do_den = 0.0
    for col in per_item:
        m = len(col)
        disagree = sum(
            1 for i in range(m) for j in range(m) if i != j and str(col[i]) != str(col[j])
        )
        do_num += disagree / (m - 1)
        do_den += m
    d_o = do_num / do_den if do_den else 0.0
    d_e = (
        1.0 - sum((counts[c] * (counts[c] - 1)) for c in cats) / (n_total * (n_total - 1))
        if n_total > 1
        else 0.0
    )
    if d_e == 0:
        return 1.0
    return float(1 - d_o / d_e)


def severity_monotonicity(severities: Sequence[float], scores: Sequence[float]) -> dict[str, float]:
    """A judge worth trusting scores lower as the damage gets worse.

    Where this correlation collapses is where the judge has stopped
    discriminating -- and that point is more informative than its mean accuracy.
    """
    if len(set(severities)) < 2 or len(set(scores)) < 2:
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n": len(scores)}
    rho, p = spearmanr(list(severities), list(scores))
    return {"spearman_rho": float(rho), "p_value": float(p), "n": len(scores)}


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> dict[str, Any]:
    """Step-down family-wise error control. Uniformly more powerful than Bonferroni."""
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    if m == 0:
        return {"reject": [], "adjusted": [], "alpha": alpha, "n_tests": 0}
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return {
        "reject": (adjusted <= alpha).tolist(),
        "adjusted": adjusted.tolist(),
        "alpha": alpha,
        "n_tests": int(m),
        "n_significant": int((adjusted <= alpha).sum()),
    }


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size. Reported alongside p so 'significant' can be
    distinguished from 'large'."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


__all__ = [
    "McNemarResult",
    "cliffs_delta",
    "holm_bonferroni",
    "judge_agreement",
    "krippendorff_alpha_nominal",
    "mcnemar_test",
    "severity_monotonicity",
]
