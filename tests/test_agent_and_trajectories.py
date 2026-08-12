"""The toy agent must be deterministic, and its degradations provably wrong."""

from __future__ import annotations

import pytest

from judgeguard.agent.rollout import build_trajectories
from judgeguard.agent.tools import TOOLS, ToolError, calculator, call, date_diff, unit_convert
from judgeguard.degrade.trajectory import (
    TRAJECTORY_DEGRADATIONS,
    build_trajectory_variants,
    length_padding,
    phantom_tool,
    silent_failure,
    wrong_argument,
)


def test_calculator_evaluates_arithmetic():
    assert calculator("2 + 3 * 4") == "14"
    assert calculator("38 * 38.5") == "1463"


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "1 if True else 2",
        "[x for x in range(3)]",
        "2 ** 999",
    ],
)
def test_calculator_refuses_anything_that_is_not_arithmetic(expr):
    """A judge-reliability project that shells strings into eval would be a bad look."""
    with pytest.raises((ToolError, ValueError)):
        calculator(expr)


def test_date_diff_and_unit_convert():
    assert date_diff("2026-01-01", "2026-03-01") == "59"
    assert unit_convert(1, "kg", "lb").startswith("2.2")


def test_undefined_tool_raises():
    with pytest.raises(ToolError):
        call("web_search", {"query": "x"})


def test_trajectories_are_deterministic_and_correct():
    a = build_trajectories(12, seed=5)
    b = build_trajectories(12, seed=5)
    assert [t.render() for t in a] == [t.render() for t in b]
    for t in a:
        assert 2 <= len(t.steps) <= 5
        for s in t.steps:
            assert s.tool in TOOLS
            assert call(s.tool, s.args) == s.result  # the trajectory is genuinely correct


def test_phantom_tool_inserts_a_tool_outside_the_action_space():
    t = build_trajectories(3, seed=1)[0]
    v = phantom_tool(t, 0.9, seed=0)
    assert any(s.tool not in TOOLS for s in v.trajectory.steps)
    assert v.introduces_error


def test_wrong_argument_keeps_tools_valid_but_breaks_a_value():
    t = build_trajectories(3, seed=1)[0]
    v = wrong_argument(t, 0.9, seed=0)
    assert all(s.tool in TOOLS for s in v.trajectory.steps)
    assert any(s.args != orig.args for s, orig in zip(v.trajectory.steps, t.steps, strict=True))


def test_wrong_argument_is_genuinely_wrong():
    """Re-executing the degraded call must not reproduce the recorded result."""
    broken = 0
    for t in build_trajectories(12, seed=2):
        v = wrong_argument(t, 0.9, seed=0)
        for s in v.trajectory.steps:
            try:
                if call(s.tool, s.args) != s.result:
                    broken += 1
            except ToolError:
                broken += 1
    assert broken > 0


def test_length_padding_is_a_probe_not_an_error():
    t = build_trajectories(3, seed=1)[0]
    v = length_padding(t, 0.9, seed=0)
    assert v.introduces_error is False
    assert v.trajectory.final_answer == t.final_answer
    assert len(v.trajectory.steps) > len(t.steps)


def test_silent_failure_keeps_the_right_answer():
    t = build_trajectories(3, seed=1)[0]
    v = silent_failure(t, 0.9, seed=0)
    assert v.trajectory.final_answer == t.final_answer
    assert v.trajectory.render() != t.render()


def test_full_cross_product():
    trajs = build_trajectories(5, seed=0)
    vs = build_trajectory_variants(trajs, seed=0)
    assert len(vs) == 5 * len(TRAJECTORY_DEGRADATIONS) * 3
    assert len({v.uid for v in vs}) == len(vs)
