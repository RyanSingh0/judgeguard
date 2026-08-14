#!/usr/bin/env python
"""Check every judge can answer before starting a multi-day run.

Wrote this after nearly losing four days to it. llama-3.3-70b spends about 380
tokens reasoning before it writes SCORE:, and the cot budget was 320. Every call
truncated mid-sentence and parsed as nothing. Nothing raised either: all HTTP
200s, failure-rate guard happy, and experiment 01 would have burned 2,400 calls
to report that a judge can't follow an output format.

Twelve calls here rule that out.

    uv run python scripts/preflight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from judgeguard.data.load import load_items
from judgeguard.judges.parse import parse_score
from judgeguard.judges.prompts import POINTWISE_CONFIGS, SYSTEM, score_prompt
from judgeguard.judges.run import _truncated, panel
from judgeguard.providers.registry import complete, is_simulated, mode_banner, resolve

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

# With less than this much of the budget left over, the next slightly longer
# answer truncates. Passing with 5% headroom isn't really passing.
MIN_HEADROOM = 0.15


def main() -> int:
    print(f"\n=== preflight ===\n    {mode_banner()}\n")
    item = load_items(1, seed=20260731)[0]
    failures = 0

    def record(judge: str, config: str, ok: bool, note: str) -> None:
        # Print as we go rather than collecting and dumping at the end. Pacing
        # against 8,000 TPM makes twelve calls take a few minutes, and a silent
        # terminal that long looks like a hang. I killed a healthy run that way
        # once already.
        nonlocal failures
        failures += not ok
        mark = f"{GREEN}OK{RESET}" if ok else f"{RED}XX{RESET}"
        print(f"  [{mark}] {judge:<14} {config:<8} {note}", flush=True)

    for judge in panel():
        if is_simulated(judge):
            record(judge, "-", False, "SIMULATED: no key, or provider_mode=simulated")
            continue
        spec = resolve(judge)
        for config in POINTWISE_CONFIGS:
            prompt = score_prompt(config, item, item.reference, kind="text")
            budget = spec.token_budget(512 if config == "cot" else 160)
            try:
                c = complete(judge, prompt, max_tokens=budget, system=SYSTEM, use_cache=False)
            except Exception as exc:
                record(judge, config, False, f"{type(exc).__name__}: {exc}"[:110])
                continue

            score = parse_score(c.text)
            used = c.completion_tokens
            headroom = 1 - (used / budget) if budget else 0.0

            if score is None:
                why = "TRUNCATED at the budget" if _truncated(c.finish_reason) else "unparseable"
                record(judge, config, False, f"{why} ({used}/{budget} tok)")
            elif headroom < MIN_HEADROOM:
                record(
                    judge,
                    config,
                    False,
                    f"{YELLOW}too tight{RESET}: {used}/{budget} tok, {headroom:.0%} headroom",
                )
            else:
                record(
                    judge,
                    config,
                    True,
                    f"score={score} served={c.model_served} {used}/{budget} tok",
                )

    if failures:
        print(
            f"\n{RED}{failures} preflight check(s) failed.{RESET}\n"
            "Do NOT start the battery. A judge that cannot emit a parseable score\n"
            "will burn its whole daily allowance producing rows that get dropped.\n"
        )
        return 1
    print(f"\n{GREEN}All judges answer and parse with headroom.{RESET} Safe to start.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
