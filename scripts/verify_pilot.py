"""Check frozen real evidence integrity; does not call or re-evaluate a live model."""

import hashlib
import json
from pathlib import Path

from judgeguard.data.pilot import load_pairs
from judgeguard.judges.parse import parse_score
from scripts.run_benchmark import summarize

ROOT = Path(__file__).resolve().parent.parent


def verify() -> None:
    report = json.loads((ROOT / "results/pilot_live.json").read_text(encoding="utf-8"))
    dataset = ROOT / "data/pilot.jsonl"
    raw = (ROOT / "results/pilot-evidence/judgments.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["provenance"]["raw_sha256"]
    assert (
        hashlib.sha256(dataset.read_bytes()).hexdigest() == report["provenance"]["dataset_sha256"]
    )
    pairs = load_pairs(dataset)
    records = [json.loads(s) for s in raw.splitlines()]
    expected = {(p.id, role) for p in pairs for role in ("reference", "candidate")}
    assert len(records) == len(expected)
    assert {(r["pair_id"], r["role"]) for r in records} == expected
    for r in records:
        assert not r["completion"]["simulated"] and not r["completion"]["cached"]
        assert r["completion"]["finish_reason"] == "stop"
        assert parse_score(r["completion"]["text"]) == r["score"]
    assert summarize(pairs, records) == report["summary"]
    assert report["summary"]["complete"]
    print("Frozen evidence verified: 100 pairs, 200 real responses; aggregate and checksums match.")


if __name__ == "__main__":
    verify()
