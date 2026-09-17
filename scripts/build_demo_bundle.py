"""Package the public pilot for the lightweight Gradio deployment."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_bundle() -> dict:
    return {
        "pilot": json.loads((ROOT / "results/pilot_live.json").read_text(encoding="utf-8")),
        "pairs": [
            json.loads(s)
            for s in (ROOT / "data/pilot.jsonl").read_text(encoding="utf-8").splitlines()
        ],
        "judgments": [
            json.loads(s)
            for s in (ROOT / "results/pilot-evidence/judgments.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
    }


if __name__ == "__main__":
    (ROOT / "demo-data.json").write_text(
        json.dumps(build_bundle(), ensure_ascii=False), encoding="utf-8"
    )
