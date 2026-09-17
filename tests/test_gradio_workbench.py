"""Deployment callbacks and the committed data bundle must agree with the library."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from judgeguard.agent.audit import trace_examples
from judgeguard.serve.app import app
from scripts.build_demo_bundle import build_bundle


def test_demo_bundle_is_current():
    assert json.loads(Path("demo-data.json").read_text(encoding="utf-8")) == build_bundle()


def test_trace_api_exposes_real_replay():
    client = TestClient(app)
    r = client.post("/audit", json=trace_examples()["Wrong argument, correct final answer"])
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert r.json()["answer_matches_expected"] is True
    assert client.post("/audit", json={"steps": []}).status_code == 422


def test_gradio_callbacks():
    import pytest

    pytest.importorskip("gradio")
    from gradio_app import inspect_pair, run_trace, sample_trace

    _, rows, audit = run_trace(sample_trace("Wrong argument, correct final answer"))
    assert audit["status"] == "failed" and len(rows) == 3
    inspected = inspect_pair("57115c7450c2381900b54aa1")
    assert inspected[2:4] == ("3600", "3960")
    assert "Failure" in inspected[4]
