"""Structured logging, optional OpenTelemetry tracing, optional MLflow tracking.

All three degrade to no-ops when the optional dependency is absent, so the core
pipeline installs and runs from a minimal dependency set. That is deliberate:
a portfolio repo that only runs after a 2 GB install does not get run.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    json_logs = os.getenv("JUDGEGUARD_JSON_LOGS", "0") == "1" if json_logs is None else json_logs
    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "judgeguard") -> Any:
    configure_logging()
    return structlog.get_logger(name)


# --------------------------------------------------------------------- OTel
_tracer = None


def get_tracer() -> Any:
    """Return an OTel tracer, or None if opentelemetry is not installed."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "judgeguard"}))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("judgeguard")
    except Exception:  # pragma: no cover - optional dep
        _tracer = False
    return _tracer or None


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    tracer = get_tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name) as sp:  # pragma: no cover - optional dep
        for k, v in attrs.items():
            sp.set_attribute(k, v)
        yield


# ------------------------------------------------------------------ MLflow
class _NullRun:
    def log_params(self, *_a: Any, **_k: Any) -> None: ...
    def log_metrics(self, *_a: Any, **_k: Any) -> None: ...
    def log_artifact(self, *_a: Any, **_k: Any) -> None: ...


@contextlib.contextmanager
def tracking_run(
    experiment: str, run_name: str, params: dict[str, Any] | None = None
) -> Iterator[Any]:
    """MLflow run if MLflow is installed and MLFLOW_TRACKING_URI is set; else no-op."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    try:
        import mlflow  # type: ignore
    except Exception:
        yield _NullRun()
        return
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

        class _Run:
            def log_params(self, p: dict[str, Any]) -> None:
                mlflow.log_params({k: str(v)[:500] for k, v in p.items()})

            def log_metrics(self, m: dict[str, float]) -> None:
                mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})

            def log_artifact(self, path: str) -> None:
                mlflow.log_artifact(path)

        yield _Run()


__all__ = ["configure_logging", "get_logger", "get_tracer", "span", "tracking_run"]
