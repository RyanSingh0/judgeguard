"""The service contract, including the guardrail's latency promise."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

from judgeguard.serve.app import app  # noqa: E402
from judgeguard.store import RESULTS_DIR  # noqa: E402

client = TestClient(app)
HAS_STUDENT = (RESULTS_DIR / "student_model.joblib").exists()
needs_student = pytest.mark.skipif(not HAS_STUDENT, reason="run experiments/09_distill.py first")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_renders():
    r = client.get("/")
    assert r.status_code == 200 and "JudgeGuard" in r.text


@needs_student
def test_guard_returns_a_decision_and_a_latency_header():
    r = client.post(
        "/guard", json={"question": "q", "answer": "A committed, specific answer with 42 units."}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"allow", "block"}
    assert 0.0 <= body["probability_good"] <= 1.0
    assert body["latency_ms"] > 0
    assert r.headers["X-JudgeGuard-Path"] == "inline-guardrail"
    assert float(r.headers["X-JudgeGuard-Latency-Ms"]) < body["budget_ms"]


@needs_student
def test_guard_respects_an_explicit_threshold():
    payload = {"question": "q", "answer": "some answer"}
    allow_all = client.post("/guard", json={**payload, "threshold": 0.0}).json()
    block_all = client.post("/guard", json={**payload, "threshold": 1.0}).json()
    assert allow_all["decision"] == "allow"
    assert block_all["decision"] == "block"


def test_guard_rejects_an_empty_answer():
    assert client.post("/guard", json={"answer": ""}).status_code == 422


def test_evaluate_attaches_judge_reliability():
    r = client.post(
        "/evaluate",
        json={
            "question": "q",
            "context": "c",
            "answer": "an answer",
            "judge": "llama70b",
            "config": "cot",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["judge"] == "llama70b"
    assert body["path"] == "async-evaluator"
    assert "caveat" in body["judge_reliability"]


def test_evaluate_rejects_an_unknown_judge():
    r = client.post("/evaluate", json={"answer": "x", "judge": "not-a-model"})
    assert r.status_code == 400


def test_report_and_summary_are_served_from_committed_results():
    assert client.get("/report").status_code == 200
    s = client.get("/summary").json()
    assert "findings" in s


def test_metrics_are_prometheus_shaped():
    client.get("/health")
    body = client.get("/metrics").text
    assert "judgeguard_request_latency_ms" in body or body.strip() == ""


def test_examples_endpoint_gives_a_one_click_demo():
    r = client.get("/examples").json()
    assert len(r["candidates"]) >= 4
    assert any("reference" in c["label"] for c in r["candidates"])
    assert all({"label", "text", "expect", "note"} <= set(c) for c in r["candidates"])


@needs_student
def test_guard_allows_the_reference_and_blocks_obvious_defects():
    """The demo is the first thing anyone clicks; it must not embarrass itself."""
    ex = client.get("/examples").json()
    by_label = {c["label"]: c for c in ex["candidates"]}
    payload = {"question": ex["question"], "context": ex["context"]}

    ref = by_label["reference (known good)"]
    assert (
        client.post("/guard", json={**payload, "answer": ref["text"]}).json()["decision"] == "allow"
    )

    for label in ("hedging (vague, uncheckable)", "omission (a required fact removed)"):
        got = client.post("/guard", json={**payload, "answer": by_label[label]["text"]}).json()
        assert got["decision"] == "block", label


@needs_student
def test_numeric_swap_is_a_documented_blind_spot_not_an_accident():
    """The student scores a numeric substitution identically to the reference.

    Asserting it keeps the limitation honest: if a future change makes the
    student catch these, this test fails and the claim in the docs must change.
    """
    ex = client.get("/examples").json()
    by_label = {c["label"]: c for c in ex["candidates"]}
    payload = {"question": ex["question"], "context": ex["context"]}
    ref = client.post(
        "/guard", json={**payload, "answer": by_label["reference (known good)"]["text"]}
    ).json()
    swapped = client.post(
        "/guard",
        json={**payload, "answer": by_label["numeric swap — the known blind spot"]["text"]},
    ).json()
    assert abs(ref["probability_good"] - swapped["probability_good"]) < 0.05
