"""Provider layer: caching, parsing and the simulator's contract."""

from __future__ import annotations

import pytest

from judgeguard.judges.parse import parse_score, parse_verdict
from judgeguard.providers.cache import cache_key, cache_stats
from judgeguard.providers.registry import complete, is_simulated, resolve
from judgeguard.providers.simulated import SimulatedProvider


def test_registry_resolves_the_panel():
    from judgeguard.config import load_registry

    for alias in load_registry().panel:
        spec = resolve(alias)
        assert spec.id and spec.family and spec.provider


def test_cache_key_is_sensitive_to_every_input_that_changes_the_answer():
    base = cache_key("p", "m", "prompt", temperature=0.0)
    assert base != cache_key("p", "m", "prompt", temperature=0.7)
    assert base != cache_key("p", "m2", "prompt", temperature=0.0)
    assert base != cache_key("p", "m", "prompt!", temperature=0.0)
    assert base == cache_key("p", "m", "prompt", temperature=0.0)


def test_second_identical_call_is_a_cache_hit():
    before = cache_stats()["hits"]
    kw = dict(
        meta={"task": "score", "config": "rubric", "uid": "cache-test", "degradation": "none"}
    )
    complete("llama70b", "cache probe prompt", **kw)
    complete("llama70b", "cache probe prompt", **kw)
    assert cache_stats()["hits"] > before


def test_replicates_are_not_collapsed_onto_one_cache_entry():
    """Five reruns at temperature 0.7 must be five samples, not one repeated."""
    texts = {
        complete(
            "gptoss20b",
            "replicate probe",
            temperature=0.7,
            meta={"task": "score", "config": "rubric", "uid": "rep", "replicate": r},
        ).text
        for r in range(5)
    }
    assert len(texts) > 1


def test_simulated_mode_is_the_default_without_keys():
    assert is_simulated("llama70b")


def test_simulator_output_parses_in_every_config():
    sim = SimulatedProvider()
    for config in ("vague", "rubric", "cot"):
        c = sim.complete(
            "p",
            model="m",
            meta={
                "judge": "llama70b",
                "config": config,
                "task": "score",
                "uid": "x",
                "degradation": "omission",
                "severity": 0.5,
                "len_ratio": 1.0,
            },
        )
        assert parse_score(c.text) is not None, config
    for config in ("pairwise", "cot"):
        c = sim.complete(
            "p",
            model="m",
            meta={
                "judge": "llama70b",
                "config": config,
                "task": "pairwise",
                "uid": "x",
                "degradation": "omission",
                "severity": 0.5,
                "len_ratio": 1.0,
                "first_is_reference": True,
            },
        )
        assert parse_verdict(c.text) in {"A", "B", "tie"}


def test_simulator_is_deterministic():
    sim = SimulatedProvider()
    meta = {
        "judge": "qwen27b",
        "config": "rubric",
        "task": "score",
        "uid": "d",
        "degradation": "numeric_swap",
        "severity": 0.5,
        "len_ratio": 1.0,
    }
    a = sim.complete("p", model="m", meta=meta).text
    b = sim.complete("p", model="m", meta=meta).text
    assert a == b


def test_simulator_penalises_severity_monotonically():
    """The simulator is a prior, but it must at least be internally coherent."""
    sim = SimulatedProvider()
    scores = [
        sim.latent(
            judge="gemini-flash",
            config="rubric",
            degradation="topic_drift",
            severity=s,
            noise_key=("fixed",),
        )
        for s in (0.0, 0.5, 1.0)
    ]
    assert scores[0] > scores[1] > scores[2]


def test_simulator_marks_its_own_output():
    c = complete(
        "llama70b", "provenance probe", meta={"task": "score", "config": "rubric", "uid": "prov"}
    )
    assert c.simulated is True
    assert "sim" in c.model_served


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SCORE: 7.5", 7.5),
        ("**SCORE:** 3", 3.0),
        ("REASONING: blah\nSCORE: 9.1", 9.1),
        ("8", 8.0),
        ("I rate this 6/10", 6.0),
        ("no number here", None),
        ("SCORE: 42", None),  # out of range, must not be accepted
    ],
)
def test_score_parsing(text, expected):
    assert parse_score(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("VERDICT: A", "A"),
        ("verdict: b", "B"),
        ("VERDICT: tie", "tie"),
        ("Answer A is better overall", "A"),
        ("nothing", None),
    ],
)
def test_verdict_parsing(text, expected):
    assert parse_verdict(text) == expected
