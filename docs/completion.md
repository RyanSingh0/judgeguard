# Working project, real evidence, and deployment

> **Release update:** the public workbench now uses Gradio on the account's free
> ZeroGPU option. See [release instructions](release.md). The Docker instructions
> below remain an alternative for accounts that can use Docker. The experimental
> student-promotion gate remains strict and failing; workbench release checks now
> separately verify tool replay and the frozen real-model evidence.

## What is ready

The repaired project runs its CLI, API, browser student and a resumable real-model
benchmark. The original results remain historical simulations. A separate published
pilot contains 200 actual responses from Qwen3-4B Q4_K_M, run locally on an NVIDIA
RTX 4060 Laptop GPU with 8 GB VRAM. It ranked the correct answer above the corrupted
answer in 96/100 pairs; article-cluster bootstrap 95% interval 92.45%–99.03%, across
28 articles. Four ties failed the scoring rule; no response failed parsing. Token
usage was 87,238 prompt and 9,459 generated tokens. Paid API charges: $0.

See `results/pilot_live.json`, `results/pilot-evidence/`, and `data/pilot.jsonl`.
The source passage and prompt template allow each judgment to be reconstructed.
Model revision, model-file checksum, data checksum, raw-response checksum, runtime
and request settings are recorded. Inference used temperature 0 and reasoning off.

This is a narrow numeric-support pilot with constructed errors, not a general RAG
score. Public benchmark contamination is possible. Labels have automatic span and
mutation checks; an independent human review is still needed. Do not train on it.

The student guardrail remains an experimental simulation-trained classifier.
Source-separated retraining fails the existing quality gate. A successful software
test is not evidence that the classifier is ready to block production answers.
The corrected quality gate intentionally remains strict; do not lower its thresholds
to turn the badge green. The shipped historical student is preserved, not silently
replaced with the failed retrained artifact.

## Run the application

From the extracted repository folder in PowerShell:

```powershell
uv sync --locked --extra dev --python 3.12
$env:PYTHONUTF8='1'
uv run --no-sync judgeguard status
uv run --no-sync judgeguard serve --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080. The first card shows the real pilot. Historical simulation
charts and the prototype guardrail are explicitly labelled. `/models` lists current
aliases; `/benchmark` returns real-pilot provenance; `/docs` describes the API.
The default interactive full judge remains simulated unless live mode is selected.
`/evaluate` is a blocking request/response API, not a background queue; its legacy
response path string `async-evaluator` is retained for compatibility.

## Free local model route

The laptop already has suitable hardware. Run this in one terminal:

```powershell
./scripts/start_local_model.ps1
```

This downloads a pinned portable llama.cpp runtime and 2.5 GB model into `.models/`,
verifies both SHA-256 digests, detects the NVIDIA Vulkan device, and listens only
on localhost. No provider API key is needed. It uses existing hardware, electricity,
disk space and memory. The original audit download lives outside the source package;
you can copy the verified model/runtime archive into `.models/` to avoid re-downloading.

In another terminal, from the project folder:

```powershell
$env:JUDGEGUARD_PROVIDER_MODE='live'
$env:JUDGEGUARD_MODELS_CONFIG='configs/models.local.yaml'
$env:PYTHONUTF8='1'
uv run --no-sync python scripts/run_benchmark.py --data data/pilot.jsonl --out results/live/my-qwen-run --model local-qwen --revision bc640142c66e1fdd12af0bd68f40445458f3869b --max-new-calls 200 --max-tokens 256
```

Run the same command to resume. Exit 0 means complete; 2 means the requested call
budget ended before completion; 1 means an error stopped collection. Successful
responses are saved immediately. Failed parses count against performance. Provider
transport retries can make more HTTP requests than the logical judgment budget.
Change output folders when changing data, model, prompt or generation settings.
For other models, create another registry config and record its exact revision and
quantization. Do not pretend a small local model is a hosted 70B model.

The optional `transformers` backend also works in a CUDA notebook after installing
`torch`, `transformers` and `accelerate`; those heavy packages are not needed by the
CPU Space. Pin a model revision and use a separate run directory. GPU notebooks have
availability limits, so checkpoint and download results before the session expires.

## Why an aggregator is not unlimited free compute

[OpenRouter](https://openrouter.ai/pricing) provides access to multiple providers,
but its free allowance is shared and currently advertised as 50 requests/day.
This 100-pair pilot needs 200 judgments: at least four allowance-days at that limit,
before retries or other usage. More API keys do not remove an account-wide quota.
It is an inference aggregator; access to a model is not access to its training weights.

[Kaggle GPU notebooks](https://www.kaggle.com/docs/efficient-gpu-usage) are a useful
fallback, subject to quotas and availability. [Colab](https://research.google.com/colaboratory/faq.html)
does not guarantee a fixed free GPU allowance. Use your existing GPU for the predictable
path. Later, compare two or three openly licensed local models; keep each model's
results separate and report failures as well as averages.

## Deploy to the existing Hugging Face Docker Space

Use the supplied `judgeguard-hf-docker.zip`, which has the Docker README at its root.
Extract it and upload its contents to the **Files** tab of your existing Docker Space.
Upload the extracted files, not the zip itself. Keep the directories intact. The root
must contain `Dockerfile`, `README.md`, `pyproject.toml`, `uv.lock`, `judgeguard/`,
`configs/`, and `results/`. The full source archive is also deployable: replace its
root README with `deploy/README.hf.md` first.

Space Settings: CPU Basic, `JUDGEGUARD_PROVIDER_MODE=simulated`, and no API secrets
for this initial public demo. The README declares `sdk: docker` and `app_port: 8080`.
The container serves port 8080 as UID 1000 and uses `/tmp` for cache. The image uses
the committed dependency lock. Never upload `.env`, `.venv`, `.models`, or raw private
run folders. The prepared package excludes them.

After the build, check `/health`, `/benchmark`, `/docs`, and one `/guard` request.
The real report is precomputed, so the Space needs neither the 2.5 GB model nor a GPU.
Your Space cannot reach the laptop's `127.0.0.1`; do not select the local-model config
there. If you later enable a paid or quota-limited live API, add authentication,
per-user limits and a queue before exposing it publicly. A secret alone does not
stop visitors from spending the server's quota.

Official [Docker Spaces instructions](https://huggingface.co/docs/hub/spaces-sdks-docker)
cover ports, UID and secrets. Runtime files are ephemeral unless persistent storage
is configured. [Space plan requirements](https://huggingface.co/docs/hub/spaces-overview)
and hardware charges are separate: current documentation lists a paid-plan requirement
for Docker/Gradio creation even though CPU Basic compute is listed as free. Use your
existing Docker Space and check its account settings before creating another.

The Docker daemon was unavailable during this audit, so the changed image has not
been built locally. Its application path was tested outside Docker; the Space build
and post-deployment smoke checks remain required. Nothing has been pushed or deployed.

## Data needed for a stronger release

1. **RAG evaluation:** 300–500 permission-cleared question/context/reference records
   from the intended domain. Include genuinely supported, unsupported, partially
   supported, unanswerable, and contradictory cases. Preserve document IDs and source
   versions. Split by document before generating errors; reserve a sealed test set.
2. **Training:** a separate 2,000–5,000 diverse examples is a reasonable starting
   experiment, not a guarantee. Collect local teacher scores, human corrections and
   naturally occurring failures. Compare supervised truth-trained and distilled
   students explicitly. Never reuse the 100-pair pilot as training/calibration data.
3. **Label review:** have a reviewer check at least 50–100 varied records and all
   ambiguous failures, with a second reviewer on a subset. Record disagreements.
   Generated corruptions supply scale; human review supplies evidence of label validity.
4. **Agents:** provide declared tool schemas, task, ordered calls, arguments, observed
   results, expected final state and execution outcome. Use replayable deterministic
   checks for tool validity and results; LLM judgment supplements those checks.
5. **Release criteria:** choose false-block and missed-error tolerances before testing,
   calibrate on a separate split, and report per-error rates with source-cluster CIs.
   The current classifier cannot be promoted merely because it is fast.

The most defensible product now is an evaluation workbench: import labelled pairs,
run a bounded local or hosted judge, inspect failures, compare versions, and export
evidence. A reliable universal inline guardrail is a separate, larger validation task.
