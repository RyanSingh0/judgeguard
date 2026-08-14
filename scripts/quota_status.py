#!/usr/bin/env python
"""How much allowance is left today, and how many days the battery still needs.

Free-tier limits are the constraint that actually governs this project, so they
get their own readout instead of turning up as a stack trace 33 calls in.

    uv run python scripts/quota_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from judgeguard.config import load_registry
from judgeguard.providers.http_base import LIMITER
from judgeguard.providers.registry import is_simulated, resolve

# Counted, not estimated: I ran the battery in simulated mode and counted cache
# misses at the provider boundary. Per judge, per phase. Phase 1 is 4,467 for
# the three Groq judges; the gemini seat needs another 335 or so because it also
# acts as the distillation teacher.
PHASE_CALLS = {"1": 4467, "2": 3523}


def main() -> None:
    reg = load_registry()
    report = LIMITER.report()

    print("\n    daily allowance")
    print("    " + "-" * 68)
    if not report:
        print("    no calls yet; limits get learned from the first response")
    for key, s in sorted(report.items()):
        rpd = s["rpd"]
        left = s["remaining_today"]
        bar = ""
        if rpd:
            used = min(s["day_requests"] / rpd, 1.0)
            bar = "  [" + "#" * int(used * 20) + "." * (20 - int(used * 20)) + "]"
        print(
            f"    {key:<42} {s['day_requests']:>6} used"
            + (f" / {rpd:<6}" if rpd else " / unknown")
            + (f"{bar}  {left} left" if left is not None else "")
        )

    print("\n    remaining work, per judge")
    print("    " + "-" * 68)
    for alias in reg.panel:
        spec = resolve(alias)
        if is_simulated(alias):
            print(f"    {alias:<14} SIMULATED (no key, or provider_mode=simulated)")
            continue
        key = f"{spec.provider}:{spec.id}"
        s = report.get(key, {})
        rpd = s.get("rpd")
        left = s.get("remaining_today")
        note = f"{left} calls left today" if left is not None else "limit not yet observed"
        days = ""
        if rpd:
            days = f"  ~{PHASE_CALLS['1'] / rpd:.1f} days for phase 1, {PHASE_CALLS['2'] / rpd:.1f} for phase 2"
        print(f"    {alias:<14} {note}{days}")

    print(
        "\n    Phase 1 needs ~{:,} calls per judge, phase 2 ~{:,}. Counted by running\n"
        "    the battery against the simulator with the cache instrumented, not\n"
        "    guessed.\n".format(PHASE_CALLS["1"], PHASE_CALLS["2"])
    )


if __name__ == "__main__":
    main()
