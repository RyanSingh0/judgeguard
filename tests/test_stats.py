"""If the statistics layer is wrong, every number in the README is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff, wilson_ci
from judgeguard.stats.power import assess, n_for_proportion_diff
from judgeguard.stats.tests import (
    cliffs_delta,
    holm_bonferroni,
    judge_agreement,
    krippendorff_alpha_nominal,
    mcnemar_test,
    severity_monotonicity,
)


def test_bca_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    x = rng.binomial(1, 0.72, size=400)
    est = bca_ci(x, seed=0)
    assert est.lo <= est.value <= est.hi
    assert est.lo >= 0.0 and est.hi <= 1.0
    assert est.n == 400


def test_bca_interval_covers_the_truth_most_of_the_time():
    """A confidence interval that does not cover is decoration, not statistics."""
    p, covered, trials = 0.7, 0, 120
    for s in range(trials):
        x = np.random.default_rng(s).binomial(1, p, size=250)
        est = bca_ci(x, seed=s, n_resamples=1200)
        covered += int(est.lo <= p <= est.hi)
    assert covered / trials >= 0.88


def test_degenerate_sample_still_returns_an_interval():
    est = bca_ci([1] * 40, seed=0)
    assert est.method in ("wilson", "degenerate")
    assert est.lo <= est.value <= est.hi


def test_wilson_is_asymmetric_at_the_boundary():
    lo, hi = wilson_ci(50, 50)
    assert lo > 0.9 and hi <= 1.0


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(1)
    base = rng.binomial(1, 0.6, 500)
    better = np.where(rng.random(500) < 0.15, 1, base)
    d = paired_bootstrap_diff(better, base, seed=0)
    assert d.diff > 0 and d.significant and d.p_value < 0.05


def test_paired_bootstrap_does_not_invent_a_difference():
    rng = np.random.default_rng(2)
    a = rng.binomial(1, 0.6, 500)
    d = paired_bootstrap_diff(a, a.copy(), seed=0)
    assert not d.significant
    assert abs(d.diff) < 1e-9


def test_paired_bootstrap_requires_alignment():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0, 1], [1, 0])


def test_mcnemar_uses_exact_when_discordants_are_few():
    r = mcnemar_test([1, 1, 0, 1, 0], [1, 0, 0, 1, 1])
    assert r.exact and r.b == 1 and r.c == 1


def test_mcnemar_is_one_when_there_is_no_disagreement():
    r = mcnemar_test([1, 0, 1], [1, 0, 1])
    assert r.p_value == 1.0


def test_kappa_and_alpha_agree_on_perfect_agreement():
    a = ["good", "bad", "good", "bad"]
    assert judge_agreement(a, a) == pytest.approx(1.0)
    assert krippendorff_alpha_nominal([a, a]) == pytest.approx(1.0, abs=1e-9)


def test_alpha_drops_when_raters_disagree():
    a = ["good", "bad", "good", "bad"]
    b = ["bad", "good", "bad", "good"]
    assert krippendorff_alpha_nominal([a, b]) < 0.1


def test_monotonicity_is_negative_when_scores_fall_with_severity():
    out = severity_monotonicity([0.2, 0.5, 0.9] * 12, [9, 6, 3] * 12)
    assert out["spearman_rho"] < -0.9


def test_holm_is_more_conservative_than_raw_but_less_than_bonferroni():
    ps = [0.001, 0.02, 0.04, 0.3]
    h = holm_bonferroni(ps)
    assert h["adjusted"][0] == pytest.approx(0.004)
    assert all(a >= p for a, p in zip(h["adjusted"], ps, strict=True))
    assert h["n_significant"] >= 1


def test_cliffs_delta_signs_correctly():
    assert cliffs_delta([3, 4, 5], [1, 2, 3]) > 0.5
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_power_analysis_is_sane():
    assert n_for_proportion_diff(0.80, 0.73) > n_for_proportion_diff(0.80, 0.60)
    r = assess(5000, 0.80, 0.73)
    assert r.adequate and 0.0 <= r.power <= 1.0
