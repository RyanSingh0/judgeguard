# Public workbench release

The public product is **deterministic agent trace auditing plus real judge evidence
inspection**. It does not deploy the failed simulation-trained student as a reliable
blocker. This release supersedes the Docker-first deployment suggestion in the initial
audit. The account's creation screen was checked: Gradio + ZeroGPU Free is available;
new Docker and CPU Basic Spaces are disabled.

## Free Hugging Face deployment

The Space is `RugFace/judgeguard-app`, using Gradio 6.27.0 and Python 3.12.
The deployment has four files: `README.md`, `app.py`, `requirements.txt` and
`demo-data.json`. `app.py` is the repository's `gradio_app.py`. Requirements pin
the project to an immutable GitHub commit. The data bundle contains only the public
SQuAD-derived pilot and recorded model responses. No provider key, local model,
training cache or private user data belongs in the upload.

The workbench performs CPU-only replay and evidence inspection; it never asks
ZeroGPU to allocate a GPU. ZeroGPU is the account's available hosting option, not
a claim of live inference. For model experiments, run the local benchmark separately.
Free hosting availability and account limits can change.

To redeploy, generate `demo-data.json` with `scripts/build_demo_bundle.py`, copy the
app and bundle, use `deploy/README.gradio.md` as the Space README, and update the
immutable commit in requirements after testing. Match the SDK version to the pinned
Gradio version. Upload through the Space Files tab or commit through the Hub CLI.

Export deployment dependencies with `uv export --locked --extra demo --no-dev
--no-hashes --no-emit-project --output-file requirements.txt`, then append
`judgeguard @ git+https://github.com/RyanSingh0/judgeguard.git@<tested-full-commit>`.
The demo extra includes Gradio's `oauth,mcp` constraints because Spaces injects
those extras during its build. They constrain Pydantic to a compatible version;
exporting the plain Gradio environment caused the initial hosted build to fail.
These optional integrations are not enabled as product features by the app.

## What the release gates mean

The normal workflows test code, tool replay and the immutable measured evidence.
`verify_pilot.py` verifies hashes, response identity, completion, parse results and
recomputed statistics. It cannot detect a hosted provider's future model drift because
it does not call that provider. This is deliberately named an evidence/replay check.

The separate manual `student-promotion` workflow keeps the original strict accuracy,
block precision and false-block requirements. Its latest failed result is published
as `results/student_promotion.json`. Moving experimental promotion out of the workbench
release path is a product boundary change, not a relaxation of the model's criteria.
The student must pass before it can be promoted to production blocking.

## Interview demonstration

Lead with agent traces for AI/ML engineering or agent-evaluation roles. Show a correct
answer reached through a wrong argument, explain independent replay and the missing
causal guarantees, then contrast it with a judge that accepts a wrong numeric answer.
For retrieval-focused roles, lead with source passages, leakage prevention and the
100-pair pilot instead. Neither focus guarantees recruiter interest; the strong signal
is explaining the implementation, evidence, failure cases and limits clearly.

[Anthropic's engineering account of agent evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
describes combining execution checks, model-based graders and human review. This
project implements a deliberately small part of that pattern. Its trace examples are
constructed, not production traces, and the real Qwen pilot evaluates text answers.

## Next research milestone

Collect genuinely generated agent traces in a versioned sandbox, including all tool
observations, expected final state and failure provenance. Add dependency-aware replay
and task-specific outcome checks. Compare the deterministic checks with two or three
local model judges on held-out tasks. Use independent human review for ambiguous
labels, retain negative results, and report uncertainty by task/source cluster.
