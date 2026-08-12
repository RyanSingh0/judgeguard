"""Latency instrumentation.

A guardrail without a latency number is a claim, not an engineering artefact.
Every request is timed, percentiles are exposed on ``/metrics`` and on the demo
page, and a p99 breach of the configured budget is surfaced rather than
smoothed over by a mean.

Means are the enemy here: a p50 of 3 ms and a p99 of 400 ms is a broken service
that looks healthy on a dashboard showing averages.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any

from judgeguard.config import get_settings


class LatencyRecorder:
    """Fixed-window recorder. Bounded memory, no external dependency."""

    def __init__(self, window: int = 5000) -> None:
        self._window = window
        self._samples: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, endpoint: str, ms: float, *, error: bool = False) -> None:
        with self._lock:
            d = self._samples.setdefault(endpoint, deque(maxlen=self._window))
            d.append(ms)
            self._counts[endpoint] = self._counts.get(endpoint, 0) + 1
            if error:
                self._errors[endpoint] = self._errors.get(endpoint, 0) + 1

    @contextmanager
    def time(self, endpoint: str) -> Any:
        t0 = time.perf_counter()
        err = False
        try:
            yield
        except Exception:
            err = True
            raise
        finally:
            self.record(endpoint, (time.perf_counter() - t0) * 1000, error=err)

    @staticmethod
    def _q(sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return float("nan")
        k = min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    def snapshot(self, endpoint: str | None = None) -> dict[str, Any]:
        budget = get_settings().guard_p99_budget_ms
        with self._lock:
            keys = [endpoint] if endpoint else list(self._samples)
            out: dict[str, Any] = {}
            for k in keys:
                vals = sorted(self._samples.get(k, []))
                if not vals:
                    continue
                p99 = self._q(vals, 0.99)
                out[k] = {
                    "count": self._counts.get(k, 0),
                    "errors": self._errors.get(k, 0),
                    "window": len(vals),
                    "mean_ms": sum(vals) / len(vals),
                    "p50_ms": self._q(vals, 0.50),
                    "p95_ms": self._q(vals, 0.95),
                    "p99_ms": p99,
                    "max_ms": vals[-1],
                    "budget_ms": budget,
                    "within_budget": bool(p99 <= budget),
                }
        return out

    def prometheus(self) -> str:
        lines = [
            "# HELP judgeguard_request_latency_ms Request latency percentiles by endpoint.",
            "# TYPE judgeguard_request_latency_ms summary",
        ]
        for ep, s in self.snapshot().items():
            safe = ep.strip("/").replace("/", "_") or "root"
            for q, key in (("0.5", "p50_ms"), ("0.95", "p95_ms"), ("0.99", "p99_ms")):
                lines.append(
                    f'judgeguard_request_latency_ms{{endpoint="{safe}",quantile="{q}"}} {s[key]:.4f}'
                )
            lines.append(f'judgeguard_requests_total{{endpoint="{safe}"}} {s["count"]}')
            lines.append(f'judgeguard_request_errors_total{{endpoint="{safe}"}} {s["errors"]}')
            lines.append(f'judgeguard_budget_ok{{endpoint="{safe}"}} {int(s["within_budget"])}')
        return "\n".join(lines) + "\n"


RECORDER = LatencyRecorder()

__all__ = ["RECORDER", "LatencyRecorder"]
