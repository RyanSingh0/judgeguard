"""Prepare a numeric QA pilot from a local official SQuAD v2 dev JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from judgeguard.data.pilot import prepare_squad

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/pilot.jsonl"))
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()
    print(json.dumps(prepare_squad(args.source, args.out, args.pairs), indent=2))
