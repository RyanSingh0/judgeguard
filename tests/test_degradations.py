"""The degradations are the gold labels. If they are wrong, everything is."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from judgeguard.degrade.text import (
    DEGRADATIONS,
    ERROR_DEGRADATIONS,
    PROBE_DEGRADATIONS,
    build_variants,
    numeric_swap,
    verbosity,
)


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_every_degradation_changes_the_text(items, name):
    for it in items[:6]:
        v = DEGRADATIONS[name](it, 0.5, seed=0, neighbours=items)
        assert v.text != it.reference, f"{name} was a no-op"


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_every_degradation_records_an_audit_trail(items, name):
    v = DEGRADATIONS[name](items[0], 0.5, seed=0, neighbours=items)
    assert len(v.edit) > 20
    assert v.degradation == name
    assert 0.0 <= v.severity <= 1.0


def test_verbosity_is_the_only_probe():
    assert PROBE_DEGRADATIONS == ["verbosity"]
    assert "verbosity" not in ERROR_DEGRADATIONS


def test_verbosity_preserves_every_fact(items):
    """The probe must add length and nothing else, or it measures nothing."""
    for it in items[:6]:
        v = verbosity(it, 0.9, seed=0)
        assert v.introduces_error is False
        assert it.reference in v.text  # reference is present verbatim
        assert v.len_ratio > 1.4
        for f in it.facts:
            assert f.render() in v.text


def test_numeric_swap_actually_changes_a_number(items):
    for it in items[:8]:
        v = numeric_swap(it, 0.9, seed=0)
        assert v.facts_corrupted, "no numeric fact was altered"
        originals = {f.value for f in it.facts if f.key in v.facts_corrupted}
        assert not all(o in v.text for o in originals)


def test_numeric_swap_severity_is_monotone(items):
    """Severity has to mean something, or the x-axis of the main figure is a lie."""
    it = items[0]
    counts = [len(numeric_swap(it, sev, seed=3).facts_corrupted) for sev in (0.2, 0.5, 0.9)]
    assert counts[0] <= counts[-1]

    # And a single perturbation must grow in magnitude with severity. The edit
    # string carries "key: old -> new", which is exactly what makes this auditable.
    def first_delta(sev: float) -> float:
        edit = numeric_swap(it, sev, seed=3).edit
        body = edit.split("[", 1)[1].split("]", 1)[0].split(";")[0]
        old, new = body.split("->")
        return abs(float(new.strip()) - float(old.split(":")[1].strip()))

    assert first_delta(0.9) > first_delta(0.2)


def test_omission_removes_information(items):
    from judgeguard.degrade.text import omission

    for it in items[:6]:
        v = omission(it, 0.9, seed=0)
        assert v.facts_removed
        assert len(v.text) < len(it.reference)


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(sev=st.floats(min_value=0.05, max_value=1.0))
def test_severity_never_crashes_and_stays_in_range(items, sev):
    for name, fn in DEGRADATIONS.items():
        v = fn(items[1], sev, seed=1, neighbours=items)
        assert v.text
        assert v.len_ratio > 0
        assert v.degradation == name


def test_build_variants_is_the_full_cross_product(items):
    vs = build_variants(items[:5], seed=0)
    assert len(vs) == 5 * len(DEGRADATIONS) * 3
    assert len({v.uid for v in vs}) == len(vs)


def test_degradations_are_deterministic(items):
    a = build_variants(items[:4], seed=11)
    b = build_variants(items[:4], seed=11)
    assert [v.text for v in a] == [v.text for v in b]
