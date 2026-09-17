"""Regression checks for provenance, serving, held-out splits and real data."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from judgeguard.config import ProviderMode
from judgeguard.data.pilot import load_pairs
from judgeguard.distill.features import render_example
from judgeguard.distill.train import Student
from judgeguard.judges.parse import parse_score, parse_verdict
from judgeguard.providers.base import ProviderError
from judgeguard.providers.registry import complete
from judgeguard.store import load
from scripts.run_benchmark import summarize


@pytest.mark.parametrize("text", ["<think>SCORE: 9", "<think>SCORE: 9</think>"])
def test_thinking_without_final_answer_is_parse_failure(text):
    assert parse_score(text) is None
    assert parse_verdict(text.replace("SCORE: 9", "VERDICT: A")) is None


def test_live_missing_key_never_falls_back_to_simulation(monkeypatch):
    settings = SimpleNamespace(effective_mode=lambda: ProviderMode.LIVE, key_for=lambda p: "")
    monkeypatch.setattr("judgeguard.providers.registry.get_settings", lambda: settings)
    with pytest.raises(ProviderError, match="No API key"):
        complete("llama70b", "Must not fabricate an answer")


@pytest.mark.parametrize("name", ["../configs/models", "../outside.json", "/tmp/x", "a/b"])
def test_reports_cannot_escape_results_directory(name):
    with pytest.raises(ValueError):
        load(name)


def test_guard_passes_context_into_inference(monkeypatch):
    service = importlib.import_module("judgeguard.serve.app")
    received = []

    def predict(texts):
        received.extend(texts)
        return np.array([0.8])

    monkeypatch.setattr(
        service, "get_student", lambda: SimpleNamespace(threshold=0.5, predict_proba=predict)
    )
    response = TestClient(service.app).post(
        "/guard", json={"question": "q", "answer": "a", "context": "source evidence"}
    )
    assert response.status_code == 200
    assert received == [render_example("q", "a", "source evidence")]


def test_live_judgment_does_not_inherit_simulated_reliability():
    service = importlib.import_module("judgeguard.serve.app")
    assert service._reliability_for("llama70b", simulated=False)["available"] is False


def test_ui_models_are_current_registry_aliases():
    from judgeguard.config import load_registry

    service = importlib.import_module("judgeguard.serve.app")
    response = TestClient(service.app).get("/models")
    assert response.json()["panel"] == load_registry().panel


def test_published_pilot_matches_raw_evidence():
    import hashlib

    service = importlib.import_module("judgeguard.serve.app")
    published = TestClient(service.app).get("/benchmark").json()
    raw = Path("results/pilot-evidence/judgments.jsonl").read_bytes()
    records = [json.loads(s) for s in raw.splitlines()]
    assert hashlib.sha256(raw).hexdigest() == published["provenance"]["raw_sha256"]
    pairs = load_pairs(Path("data/pilot.jsonl"))
    assert summarize(pairs, records) == published["summary"]
    assert all(not r["completion"]["simulated"] for r in records)


def test_source_items_are_disjoint_across_student_splits():
    groups = [str(i) for i in range(30) for _ in range(2)]
    labels = [j for _ in range(30) for j in (0, 1)]
    texts = [
        f"source {g} " + ("specific answer 42" if y else "perhaps maybe unclear")
        for g, y in zip(groups, labels, strict=True)
    ]
    _, extra = Student().fit(texts, labels, truth=labels, item_groups=groups)
    split_groups = [
        {groups[i] for i in extra[f"{s}_indices"]} for s in ("train", "calibration", "test")
    ]
    assert not split_groups[0] & split_groups[1]
    assert not split_groups[0] & split_groups[2]
    assert not split_groups[1] & split_groups[2]
    assert set.union(*split_groups) == set(groups)


def test_real_pilot_has_auditable_source_spans():
    pairs = load_pairs(Path("data/pilot.jsonl"))
    assert len(pairs) == 100
    assert len({p.context for p in pairs}) == 100
    for p in pairs:
        assert p.reference in p.context and p.candidate not in p.context
        assert p.split == "test"
        for a in p.metadata["answer_spans"]:
            assert p.context[a["answer_start"] : a["answer_start"] + len(a["text"])] == a["text"]


def test_benchmark_counts_ties_and_parse_failures_as_failures():
    pairs = load_pairs(Path("data/pilot.jsonl"))[:3]
    records = []
    for p, scores in zip(pairs, [(9, 2), (9, 9), (None, 2)], strict=True):
        for role, score in zip(("reference", "candidate"), scores, strict=True):
            records.append(
                {
                    "pair_id": p.id,
                    "role": role,
                    "score": score,
                    "completion": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "model_served": "test",
                    },
                }
            )
    summary = summarize(pairs, records)
    assert summary["discrimination_accuracy"]["value"] == pytest.approx(1 / 3)
    assert summary["pairs_with_parse_failure"] == 1
    assert summary["complete"]
    assert not summarize(pairs, records[:-1])["complete"]


def test_benchmark_resumes_without_repeating_calls(tmp_path, monkeypatch):
    import sys

    from judgeguard.providers.base import Completion
    from scripts.run_benchmark import main

    pair = load_pairs(Path("data/pilot.jsonl"))[0]
    data = tmp_path / "data.jsonl"
    data.write_text(pair.model_dump_json() + "\n", encoding="utf-8")
    calls = []

    def fake_complete(*args, **kwargs):
        calls.append(args)
        return Completion(
            text="SCORE: 9", model_requested="test", model_served="test", provider="test"
        )

    monkeypatch.setattr("judgeguard.providers.registry.is_simulated", lambda alias: False)
    monkeypatch.setattr("judgeguard.providers.registry.complete", fake_complete)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark",
            "--data",
            str(data),
            "--out",
            str(tmp_path / "run"),
            "--model",
            "llama70b",
            "--max-new-calls",
            "1",
        ],
    )
    assert main() == 2
    assert main() == 0
    assert main() == 0
    assert len(calls) == 2
    manifest = json.loads((tmp_path / "run/run.json").read_text())
    assert manifest["resolved_model"]
