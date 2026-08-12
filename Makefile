# JudgeGuard — one-command workflows. Requires `uv` (https://docs.astral.sh/uv/).
.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install everything (dev extras included)
	uv sync --extra dev
	@echo "-> cp .env.example .env  (optional: add provider keys)"

setup-all: ## Install with heavy extras too (embeddings, tracking, otel)
	uv sync --extra all

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-format
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run the test suite
	uv run pytest

typecheck: ## mypy
	uv run mypy judgeguard

data: ## Build the degraded dataset (writes results/degraded_set.json)
	uv run python experiments/00_build_dataset.py

battery: ## Run the full bias battery (experiments 01-08)
	uv run python scripts/run_all.py --stage battery

distill: ## Train + calibrate the student, benchmark latency
	uv run python experiments/09_distill.py
	uv run python experiments/10_latency_bench.py

figures: ## Regenerate every figure from results/*.json
	uv run python experiments/make_figures.py

all: data battery distill figures ## Full reproduction, end to end
	@echo "Done. See results/ and docs/findings.md"

serve: ## Run the API + demo UI on :8080
	uv run uvicorn judgeguard.serve.app:app --host 0.0.0.0 --port 8080 --reload

gate: ## The CI eval gate, run locally
	uv run python experiments/regression_suite.py --fail-under 0.70

verify: ## Cross-check every number quoted in the docs against results/
	uv run python scripts/verify_claims.py

site: ## Build the static Space (browser guardrail) into site/
	uv run python scripts/export_web_model.py
	uv run python scripts/build_site.py
	node site/parity.test.js

site-serve: ## Preview the static site on :8000
	@echo 'http://localhost:8000' && cd site && python3 -m http.server 8000

check: lint test gate verify site ## Everything CI runs
	@echo "all checks green"

docker: ## Build the container
	docker build -t judgeguard:local .

docker-run: ## Run the container on :8080
	docker run --rm -p 8080:8080 judgeguard:local

clean: ## Remove caches (keeps results/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ .cache

.PHONY: help setup setup-all lint fmt test typecheck data battery distill figures all serve gate verify site site-serve check docker docker-run clean
