"""Validate a completed local Qwen pilot and publish its aggregate and evidence."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from judgeguard.data.pilot import load_pairs
from judgeguard.judges.parse import parse_score
from scripts.run_benchmark import summarize


def publish(run: Path, data: Path, output: Path) -> dict:
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    pairs = load_pairs(data)
    if manifest["dataset_sha256"] != hashlib.sha256(data.read_bytes()).hexdigest():
        raise ValueError("Dataset checksum mismatch")
    records = [
        json.loads(s) for s in (run / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected = {(p.id, role) for p in pairs for role in ("reference", "candidate")}
    observed = [(r["pair_id"], r["role"]) for r in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete or duplicated evidence")
    for r in records:
        if r["completion"]["simulated"] or r["completion"]["cached"]:
            raise ValueError("Expected fresh real responses")
        if parse_score(r["completion"]["text"]) != r["score"]:
            raise ValueError("Saved score disagrees with raw response")
    summary = summarize(pairs, records)
    payload = {
        "provenance": {
            "mode": "live",
            "generated_at": records[-1]["created_at"],
            "dataset_sha256": manifest["dataset_sha256"],
            "raw_sha256": hashlib.sha256((run / "judgments.jsonl").read_bytes()).hexdigest(),
            "code_base": "05be5b381609eddb237489a8c8a1f88784709b34 plus September 2026 repair patch",
        },
        "model": {
            "name": "Qwen3-4B Q4_K_M",
            "repository": "https://huggingface.co/Qwen/Qwen3-4B-GGUF",
            "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
            "file_sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
            "runtime": "llama.cpp b11012 Vulkan; RTX 4060 Laptop 8GB; reasoning off; context 8192; one slot",
            "license": "Apache-2.0",
            "paid_api_cost_usd": 0,
            "cost_caveat": "Uses existing hardware and electricity; not zero physical compute cost.",
        },
        "run": manifest,
        "dataset": json.loads(data.with_suffix(".manifest.json").read_text(encoding="utf-8")),
        "summary": summary,
        "evidence": "results/pilot-evidence/judgments.jsonl",
        "interpretation": "One local quantized model, 100 constructed numeric errors. No claim about hosted models, production RAG, training performance, or agent safety. Public benchmark contamination cannot be ruled out. Human review pending.",
    }
    if (
        manifest["resolved_model"] != "qwen3-4b-q4km-bc640142"
        or manifest["revision"] != payload["model"]["revision"]
    ):
        raise ValueError("This publication profile is for the pinned local Qwen run only")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    evidence = output.parent / "pilot-evidence"
    evidence.mkdir(exist_ok=True)
    for name in ("judgments.jsonl", "run.json", "summary.json"):
        shutil.copyfile(run / name, evidence / name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/pilot.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/pilot_live.json"))
    args = parser.parse_args()
    print(json.dumps(publish(args.run, args.data, args.out), indent=2))
