"""Prompts and the swap protocol."""

from __future__ import annotations

from judgeguard.judges.prompts import CONFIGS, pairwise_prompt, score_prompt
from judgeguard.judges.run import PairTask, ScoreTask, run_pairwise, run_scores


def test_no_prompt_leaks_the_reference_answer(item):
    """The judge is reference-free by design; leaking gold would make it trivial."""
    for config in ("vague", "rubric", "cot"):
        p = score_prompt(config, item, "a candidate answer")
        assert item.reference not in p
        assert item.question in p


def test_pairwise_prompt_contains_both_options(item):
    p = pairwise_prompt("pairwise", item, "AAA", "BBB")
    assert "AAA" in p and "BBB" in p and "VERDICT" in p


def test_trajectory_prompt_declares_the_action_space(item):
    p = score_prompt("rubric", item, "TASK: x", kind="trajectory")
    for tool in ("calculator", "lookup", "date_diff", "unit_convert"):
        assert tool in p


def test_all_configs_are_registered():
    assert set(CONFIGS) == {"vague", "rubric", "cot", "pairwise"}


def test_run_scores_returns_parsed_judgments(items):
    tasks = [ScoreTask(uid=f"{i.id}:ref", item=i, answer=i.reference) for i in items[:6]]
    out = run_scores("llama70b", "rubric", tasks)
    assert len(out) == 6
    assert all(j.parsed_ok and 1 <= j.score <= 10 for j in out)
    assert all(j.simulated for j in out)


def test_swap_protocol_runs_both_orderings(items):
    from judgeguard.degrade.text import numeric_swap

    tasks = [
        PairTask(
            uid=i.id,
            item=i,
            reference=i.reference,
            degraded=numeric_swap(i, 0.5, seed=0).text,
            degradation="numeric_swap",
            severity=0.5,
        )
        for i in items[:10]
    ]
    out = run_pairwise("gptoss20b", "pairwise", tasks)
    assert len(out) == 10
    for o in out:
        assert o.forward in {"REF", "DEG", "tie", None}
        assert o.reverse in {"REF", "DEG", "tie", None}
        assert o.consistent == (o.forward is not None and o.forward == o.reverse)
        # swap-and-average can never be more generous than the lucky ordering
        assert o.correct_swapped <= o.correct_ref_first or not o.consistent


def test_inconsistent_comparisons_are_scored_as_misses(items):
    from judgeguard.degrade.text import numeric_swap

    tasks = [
        PairTask(
            uid=i.id,
            item=i,
            reference=i.reference,
            degraded=numeric_swap(i, 0.2, seed=0).text,
            degradation="numeric_swap",
            severity=0.2,
        )
        for i in items[:12]
    ]
    out = run_pairwise("gptoss20b", "pairwise", tasks)
    for o in out:
        if not o.consistent:
            assert o.correct_swapped == 0
            assert o.verdict == "inconsistent"
