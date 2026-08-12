# RUNBOOK — build it, run it, ship it

Everything below has been executed end to end.

---

## 0. Prerequisites

| tool | why | install |
|---|---|---|
| **uv** | packaging + venv; the standard Python toolchain in 2026 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (macOS/Linux)<br>`powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` (Windows) |
| **git** | the eval gate stamps the commit SHA into every artefact | — |
| Docker | optional, for the container and deploy | docker.com |
| make | optional, convenience only | preinstalled on macOS/Linux |

Python itself is **not** a prerequisite — `uv` installs the pinned 3.12 for you.

---

## 1. Setup (2 minutes)

```bash
git clone https://github.com/RyanSingh0/judgeguard.git
cd judgeguard

uv sync --extra dev
cp .env.example .env         # optional; works fine with zero keys
```

Verify:

```bash
uv run judgeguard status
```

```
JudgeGuard v0.1.0   mode=simulated
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ alias        ┃ provider   ┃ family  ┃ tier        ┃ served by ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ gemini-flash │ gemini     │ gemini  │ frontier    │ simulator │
│ llama70b     │ groq       │ llama   │ open-weight │ simulator │
│ qwen27b      │ groq       │ qwen    │ open-weight │ simulator │
│ gptoss20b    │ groq       │ gpt-oss │ open-weight │ simulator │
└──────────────┴────────────┴─────────┴─────────────┴───────────┘
```

> **On `mode=simulated`.** With no API keys the harness runs against a deterministic model of
> judge behaviour defined in `configs/simulator.yaml`. Every code path — prompting, parsing,
> retries, caching, statistics, distillation, serving — is exercised identically. Artefacts are
> stamped `provenance.mode="simulated"` and figures are watermarked. Read `docs/limitations.md`
> §1 before quoting any number produced this way.

Windows users: everything works in PowerShell. Use `uv run` rather than `make`, or run under WSL.

---

## 2. Smoke test (30 seconds)

```bash
uv run python scripts/run_all.py --quick
```

Runs all ten experiments at tiny sample sizes, regenerates figures and runs the eval gate in
smoke mode. If it ends with `gate passed`, the whole system works.

Quick runs write to `results/_smoke/` — **they cannot overwrite the committed results.** That
guard exists because a 40-item run silently replacing the 500-item artefacts is a mistake that
looks like nothing at all until someone checks the numbers.

---

## 3. Full reproduction (~4 minutes on 2 cores)

```bash
make all
```

Stage by stage, if you prefer to watch:

```bash
uv run python experiments/00_build_dataset.py        --items 500
uv run python experiments/01_discrimination.py       --items 500     # the main figure
uv run python experiments/02_position_bias.py        --items 250
uv run python experiments/03_verbosity_bias.py       --items 500
uv run python experiments/04_self_enhancement.py     --items 300
uv run python experiments/05_rubric_ablation.py      --items 400
uv run python experiments/06_self_consistency.py     --items 200 --replicates 5
uv run python experiments/07_cost_accuracy.py
uv run python experiments/08_trajectory_blindness.py --trajectories 60
uv run python experiments/09_distill.py              --items 500
uv run python experiments/10_latency_bench.py        --requests 1200
uv run python experiments/make_figures.py
uv run python experiments/regression_suite.py --fail-under 0.70
uv run python scripts/verify_claims.py
```

Outputs:

```
results/degraded_set.json          every gold pair + its audit trail
results/01…10_*.json               one file per experiment, every number with a CI
results/student_model.joblib       the trained guardrail
results/figures/*.png              11 figures, watermarked if simulated
results/regression_gate.json       the CI gate's verdict
```

`scripts/emit_numbers.py` prints every headline figure from `results/`. It is the single source of
truth for the prose — write docs from its output, never the other way round.

---

## 4. Run it against real models

### 4.1 Get keys (all free, no card)

| provider | role in the panel | where |
|---|---|---|
| **Google AI Studio** | frontier-class judge | aistudio.google.com/apikey |
| **Groq** | fast open-weight judges | console.groq.com/keys |
| **Cerebras** | second open-weight family | cloud.cerebras.ai |
| **OpenRouter** | model variety behind one key | openrouter.ai/keys |

You need **at least two model families** — self-enhancement bias is unmeasurable with one.

### 4.2 Configure

```bash
# .env
JUDGEGUARD_PROVIDER_MODE=live
GEMINI_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
JUDGEGUARD_MAX_CONCURRENCY=4       # raise carefully; free tiers throttle
```

```bash
uv run judgeguard status           # every panel row should say 'live API'
```

### 4.3 Three rules, and the first one cost a whole afternoon

1. **Pin model IDs in `configs/models.yaml`, and trust `model_served` over what you asked for.**
   Every single one of the originally pinned IDs — `gemini-2.0-flash`, `qwen-3-32b`,
   `mistral-small-3.2-24b-instruct:free` — was retired or de-freed by its provider between writing
   and running. The panel in the repo is the one **verified live on 2026-07-31**. Re-verify before
   any serious run:

   ```bash
   uv run python -c "
   import httpx, os; from dotenv import load_dotenv; load_dotenv()
   r = httpx.get('https://api.groq.com/openai/v1/models',
                 headers={'Authorization': f\"Bearer {os.environ['GROQ_API_KEY']}\"})
   print(sorted(m['id'] for m in r.json()['data']))"
   ```

2. **Never trust a published rate limit — or a published free tier.** Cerebras returned
   `402 Payment required` on *every* model for a brand-new account. It is configured but left out
   of the panel for that reason. Check your own account before sizing a run.

3. **Reasoning models need 10× the token budget.** `qwen3.6-27b` spent 1,175 output tokens on a
   single judgement, ~1,100 of them in the thinking block. At 128 tokens it returns an empty
   string and looks like a broken provider. Models flagged `reasoning: true` in
   `configs/models.yaml` get their budget scaled automatically, and the parser strips
   `<think>` blocks so a number inside the scratchpad is never mistaken for the verdict.

### 4.4 Scale sensibly

```bash
uv run python experiments/01_discrimination.py --items 20    # confirm it completes
uv run python experiments/01_discrimination.py --items 150   # ~18k calls, still 80% power at 7 pp
uv run python scripts/run_all.py --items 500                 # the full battery
```

Measured per-call latency on the verified panel: `llama70b` ~0.15 s, `gptoss20b` ~0.33 s,
`gemini-flash` ~1.0 s, `qwen27b` ~4.7 s. **Budget for the reasoning models** — a full 500-item
battery across four judges is an overnight job, and `--items 150` is the sensible first live run.

The disk cache means **a crashed run resumes for free** and a rerun costs zero API calls. Do not
delete `.cache/` casually.

### 4.5 Optional: the $25

Once the pipeline is proven, add one paid frontier arm to experiments 01 and 08 so you can write
*"the free judge reached X% of frontier accuracy at 1/N the cost."* Add it to
`configs/models.yaml` and to `panel:`. If you never spend it, the project still stands.

---

## 5. Serve it

```bash
make serve      # http://localhost:8080
```

Five example buttons load a reference answer, two defects the guardrail catches, the numeric
substitution it is documented to miss, and a padded-but-correct answer.

| endpoint | path | notes |
|---|---|---|
| demo UI | `GET /` | single-file Tailwind, no build step |
| inline guardrail | `POST /guard` | student only, `X-JudgeGuard-Latency-Ms` header |
| async judge | `POST /evaluate` | full LLM call + that judge's measured reliability |
| battery | `GET /report` | served from committed files |
| findings | `GET /summary` | computed from `results/`, not typed |
| percentiles | `GET /metrics` | Prometheus |
| OpenAPI | `GET /docs` | interactive |

```bash
curl -s -X POST localhost:8080/guard -H 'content-type: application/json' \
  -d '{"question":"Summarise the trial.","answer":"The trial enrolled 3013 participants."}' | jq
```

CLI equivalents:

```bash
uv run judgeguard findings
uv run judgeguard degrade --show numeric_swap --severity 0.5
uv run judgeguard guard "It is generally understood that this was a modest amount."
uv run judgeguard judge "The trial enrolled 3013 participants." --model llama70b
```

---

## 6. Container

```bash
make docker && make docker-run
# or
docker compose up --build
```

The image ships `results/` and the trained student, so `/report`, `/summary`, the figures and the
guardrail all work with **no key, no volume, no network**. Multi-stage build on `python:3.12-slim`,
non-root user, healthcheck included.

---

## 7. Deploy

### Fly.io — scale-to-zero, so an idle demo is free

```bash
fly auth login
fly launch --no-deploy        # fly.toml is already in the repo
fly secrets set GEMINI_API_KEY=... GROQ_API_KEY=...
fly deploy && fly open
```

### Hugging Face Spaces — no card required, and the recommended option

Spaces need a README with YAML frontmatter, so the Space card and the push procedure live in
**[`deploy/hf-space/DEPLOY.md`](../deploy/hf-space/DEPLOY.md)**. Create a **Docker** Space on CPU
basic, push, optionally add keys as secrets. Works with no keys at all.

**Put the URL in the README's `## Try it` section and in LinkedIn Featured. A live link is
verifiable in ten seconds; a repo is not.**

---

## 8. CI

- **`.github/workflows/ci.yml`** — ruff lint + format, pytest with coverage on 3.11 and 3.12,
  mypy, Docker build, and a container smoke test that curls `/health` and `/guard`.
- **`.github/workflows/eval.yml`** — the **eval gate**. Rebuilds the dataset and the student from
  scratch, then fails the build if guardrail accuracy drops below 0.70, if it ever blocks a correct
  answer, if calibration error exceeds 0.15, if p99 latency exceeds the budget, if a documented
  blind spot silently changes in *either* direction, or if any structural invariant of the gold
  labels breaks. It also runs `verify_claims.py`, so prose that drifts from the data fails CI.

```bash
make gate      # the gate
make verify    # the claim checker
make check     # lint + test + gate + verify
```

---

## 9. Optional extras

```bash
uv sync --extra all          # embeddings + MLflow + OpenTelemetry

uv run python experiments/09_distill.py --featurizer minilm
export MLFLOW_TRACKING_URI=file:./mlruns && uv run mlflow ui
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

---

## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Readme file does not exist` on install | `README.md` missing | required by `pyproject.toml`; restore it |
| `sqlite3.OperationalError: disk I/O error` | cache dir on a network mount | automatic fallback to memory; or set `JUDGEGUARD_CACHE_DIR=/tmp/jg-cache` |
| `Student model not found` from `/guard` | distillation never ran | `uv run python experiments/09_distill.py` |
| `FileNotFoundError: results/01_*.json` | experiments never ran | `make all` |
| **404 "model is no longer available"** | provider retired the pinned ID | list live models (§4.3) and update `configs/models.yaml` |
| **402 "Payment required"** | that provider has no free tier on your account | drop it from `panel:`; the harness needs only two families |
| judge returns empty strings | reasoning model, budget too small | set `reasoning: true` on it in `configs/models.yaml` |
| judge scores the *correct* answer 2/10 | source context lacks the values | the corpus embeds them; a custom JSONL corpus must too |
| 429s during a live run | free-tier throttling | lower `JUDGEGUARD_MAX_CONCURRENCY`; backoff is automatic |
| gate fails on a "documented blind spot" | the student changed | intentional — update the docs, then move the bound |
| figures missing the watermark | you are running live | correct behaviour |

---

## 11. Where each claim comes from

| claim | script | file |
|---|---|---|
| discrimination accuracy, severity curves | `01_discrimination.py` | `results/01_discrimination.json` |
| position-bias disagreement, fixed-order inflation | `02_position_bias.py` | `results/02_position_bias.json` |
| verbosity bias in points | `03_verbosity_bias.py` | `results/03_verbosity_bias.json` |
| self-enhancement, leave-one-out | `04_self_enhancement.py` | `results/04_self_enhancement.json` |
| rubric / CoT ablation, Holm-corrected | `05_rubric_ablation.py` | `results/05_rubric_ablation.json` |
| self-consistency, Krippendorff α | `06_self_consistency.py` | `results/06_self_consistency.json` |
| cost per correct judgment, Pareto frontier | `07_cost_accuracy.py` | `results/07_cost_accuracy.json` |
| argument / hallucination / path blindness | `08_trajectory_blindness.py` | `results/08_trajectory_blindness.json` |
| student vs teacher, calibration, cascade | `09_distill.py` | `results/09_distill.json` |
| guardrail p99, concurrency sweep | `10_latency_bench.py` | `results/10_latency_bench.json` |

If a number appears anywhere in the prose and is not in one of these files, `make verify` fails.
