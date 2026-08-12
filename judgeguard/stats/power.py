"""Sample-size justification.

Sample sizes in this project are chosen for statistical power, not for budget.
That distinction matters: "500 items because that is what the free tier allowed"
is an excuse, while "500 items because it resolves a 7-point accuracy difference
at 80% power" is a design decision. This module is what lets the README make the
second claim honestly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class PowerResult:
    n_required: int
    n_available: int
    effect: float
    power: float
    alpha: float
    adequate: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def n_for_proportion_diff(p1: float, p2: float, *, power: float = 0.80, alpha: float = 0.05) -> int:
    """Items needed to detect a difference between two independent accuracies."""
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (
        z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(math.ceil(num / (p1 - p2) ** 2))


def n_for_mcnemar(
    p_discordant: float, odds_ratio: float = 2.0, *, power: float = 0.80, alpha: float = 0.05
) -> int:
    """Items needed for a PAIRED comparison of two judges on the same items.

    Substantially smaller than the independent-samples number, which is the
    whole reason the battery is designed as a paired experiment.
    """
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    psi = odds_ratio
    num = (z_a * (psi + 1) + z_b * math.sqrt((psi + 1) ** 2 - (psi - 1) ** 2 * p_discordant)) ** 2
    den = (psi - 1) ** 2 * p_discordant
    return int(math.ceil(num / den))


def achieved_power_proportion(n: int, p1: float, p2: float, *, alpha: float = 0.05) -> float:
    """Power actually achieved at the sample size we ran."""
    if p1 == p2:
        return alpha
    z_a = norm.ppf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    z = (abs(p1 - p2) - z_a * se0) / se1
    return float(norm.cdf(z))


def assess(
    n_available: int, p1: float, p2: float, *, power: float = 0.80, alpha: float = 0.05
) -> PowerResult:
    need = n_for_proportion_diff(p1, p2, power=power, alpha=alpha)
    return PowerResult(
        n_required=need,
        n_available=n_available,
        effect=abs(p1 - p2),
        power=achieved_power_proportion(n_available, p1, p2, alpha=alpha),
        alpha=alpha,
        adequate=n_available >= need,
    )


__all__ = [
    "PowerResult",
    "achieved_power_proportion",
    "assess",
    "n_for_mcnemar",
    "n_for_proportion_diff",
]
