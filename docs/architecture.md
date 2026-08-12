# Architecture

```
                          ┌──────────────────────── configs/ ────────────────────────┐
                          │  models.yaml      pinned model IDs, families, list prices │
                          │  simulator.yaml   the priors used when no API key exists  │
                          └──────────────────────────────────────────────────────────┘
                                                    │
   ┌────────────────┐    ┌─────────────────┐    ┌───▼────────────────┐   ┌──────────────────┐
   │ data/corpus.py │    │ degrade/text.py │    │ providers/registry │   │  stats/          │
   │                │    │                 │    │                    │   │                  │
   │ 500 items      │───▶│ 6 degradations  │───▶│ complete(alias,…)  │──▶│ BCa bootstrap    │
   │ fact-composed  │    │ x 3 severities  │    │   ├ gemini         │   │ paired bootstrap │
   │ + SOURCE RECORD│    │ = 9,000 variants│    │   ├ groq           │   │ McNemar          │
   └────────────────┘    └─────────────────┘    │   ├ cerebras       │   │ kappa / alpha    │
   ┌────────────────┐    ┌─────────────────┐    │   ├ openrouter     │   │ Spearman         │
   │ agent/rollout  │    │degrade/trajectory│   │   ├ ollama         │   │ Holm-Bonferroni  │
   │ 4 pure tools   │───▶│ 4 degradations  │───▶│   └ simulated ◀────┼───│ power analysis   │
   │ 60 trajectories│    │ = 720 variants  │    │      + disk cache  │   └──────────────────┘
   └────────────────┘    └─────────────────┘    └────────────────────┘             │
                                                    │                              │
                                          ┌─────────▼──────────┐                   │
                                          │ judges/            │                   │
                                          │  prompts  4 configs│                   │
                                          │  run      swap     │───────────────────┘
                                          │           protocol │
                                          └─────────┬──────────┘
                                                    │
                         ┌──────────────────────────▼───────────────────────────┐
                         │ experiments/ 01…10  →  results/*.json  (committed)   │
                         │ make_figures.py     →  results/figures/*.png         │
                         │ regression_suite.py →  CI eval gate, exit 1 on drift │
                         │ verify_claims.py    →  prose vs data, exit 1 on drift│
                         └──────────────────────────┬───────────────────────────┘
                                                    │
                        ┌───────────────────────────▼────────────────────────────┐
                        │ distill/                                                │
                        │   labels   teacher decisions → training labels          │
                        │   features n-grams + structural + source coverage       │
                        │   train    LR + isotonic, then FREE recalibration on    │
                        │            held-out degradation ground truth            │
                        └───────────────────────────┬────────────────────────────┘
                                                    │  student_model.joblib
                        ┌───────────────────────────▼────────────────────────────┐
                        │ serve/app.py — FastAPI                                  │
                        │                                                         │
                        │  POST /guard     INLINE. student only, no network call, │
                        │                  p99 budget enforced + reported         │
                        │  POST /evaluate  ASYNC. full LLM judge + that judge's    │
                        │                  measured reliability attached          │
                        │  GET  /report    the battery, from committed files       │
                        │  GET  /summary   the three findings, computed not typed  │
                        │  GET  /metrics   Prometheus percentiles                  │
                        │  GET  /          single-file Tailwind demo UI            │
                        └─────────────────────────────────────────────────────────┘
                                    Docker (multi-stage) → Fly.io / Railway / HF Spaces
```

## The load-bearing design decisions

**One entry point for every model call.** Nothing downstream imports a concrete provider.
Everything calls `complete("llama70b", prompt, meta=…)` and `providers/registry.py` decides whether
that becomes HTTPS to Groq or a draw from the simulator. Swapping the entire panel from simulated to
live is one environment variable, which is what keeps the offline mode a development convenience
rather than a separate code path that rots.

**`meta` is non-semantic.** Real HTTP providers ignore it entirely; only the simulator reads it.
That is how ground truth reaches the simulator without contaminating the prompt a real judge sees. A
test asserts the reference never appears in any prompt.

**Cache key includes the replicate index.** Without it, five reruns at temperature 0.7 collapse onto
one entry and the harness reports perfect self-agreement. That bug lived here for exactly one commit
and is now covered by a test.

**Token budgets are model-aware.** Reasoning models get 10× output tokens and the parser strips
`<think>` blocks. A judge that looks broken because of our own token cap is a measurement error, not
a model property.

**The statistics layer was written before the experiments.** There is no code path that emits a bare
point estimate into a result file, because the function that would let you do it does not exist.

**Results are committed.** `/report`, `/summary`, the README and the figures all render from files in
git. The demo works with no API key, no quota and no network — six months from now, when every free
tier has changed, it still works.

**`--quick` writes to `results/_smoke/`.** A 40-item smoke run silently replacing the 500-item
committed artefacts is a mistake that looks like nothing at all until someone checks the numbers.

**The guardrail contains no network call.** Not "usually avoids" — cannot. That is the only way a
hard millisecond budget is a promise rather than an aspiration.

## Module map

| path | responsibility |
|---|---|
| `judgeguard/config.py` | pydantic-settings; model registry; `ProviderMode`; token budgets |
| `judgeguard/telemetry.py` | structlog, optional OpenTelemetry, optional MLflow (all no-op cleanly) |
| `judgeguard/store.py` | result persistence, provenance stamping, NaN-safe JSON, DuckDB view |
| `judgeguard/providers/` | protocol, disk cache, 3 HTTP clients, the simulator, the registry |
| `judgeguard/data/` | schema, procedural corpus, loaders, per-family response generation |
| `judgeguard/degrade/` | 6 text degradations, 4 trajectory degradations |
| `judgeguard/judges/` | 4 prompt configs, robust parsing, batch runner, swap protocol |
| `judgeguard/agent/` | 4 deterministic tools (AST-safe calculator), task templates, rollout |
| `judgeguard/stats/` | BCa, paired bootstrap, McNemar, kappa/alpha, Spearman, Holm, power |
| `judgeguard/distill/` | featurisation, teacher labelling, calibrated student |
| `judgeguard/serve/` | FastAPI app, latency recorder, schemas, single-file UI |
| `judgeguard/cli.py` | `judgeguard status / degrade / judge / guard / findings / serve` |
| `scripts/emit_numbers.py` | prints every headline figure from `results/` — source of truth for prose |
| `scripts/verify_claims.py` | re-derives each quoted number and fails if the prose drifted |

## Request paths

```
POST /guard                                    POST /evaluate
   │                                              │
   ├─ featurize   ~0.4 ms                         ├─ build prompt (rubric/cot)
   ├─ logistic    ~0.1 ms                         ├─ provider call    299–703 ms
   ├─ isotonic    ~0.0 ms                         ├─ strip reasoning, parse score
   ├─ threshold                                   ├─ attach measured reliability
   └─ allow/block   p99 7.2 ms  ✓ 150 ms          └─ score + reasoning + caveats
```
