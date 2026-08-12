# syntax=docker/dockerfile:1.7
# --------------------------------------------------------------------------- #
# JudgeGuard — multi-stage build.
#
# The image ships the committed results and the trained student, so /report,
# /summary, the figures and the guardrail all work with no API key, no volume
# and no network. That is the difference between a demo that survives a recruiter
# clicking it six months from now and one that does not.
# --------------------------------------------------------------------------- #

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app
# Dependency layer first: it changes far less often than the source.
COPY pyproject.toml README.md ./
COPY judgeguard/__init__.py judgeguard/__init__.py
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

COPY judgeguard/ judgeguard/
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps .


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUDGEGUARD_PROVIDER_MODE=auto \
    JUDGEGUARD_CACHE_DIR=/tmp/judgeguard-cache \
    JUDGEGUARD_GUARD_P99_BUDGET_MS=150 \
    PORT=8080

RUN useradd --create-home --uid 10001 judge
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY judgeguard/ judgeguard/
COPY configs/ configs/
COPY results/ results/

USER judge
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=4s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8080\")}/health',timeout=3).status==200 else 1)"

CMD ["sh", "-c", "uvicorn judgeguard.serve.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
