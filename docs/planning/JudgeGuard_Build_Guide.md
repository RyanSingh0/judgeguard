# JudgeGuard — Complete Build Guide

**Can you trust the judge?** Measuring LLM evaluator reliability, then shipping the calibrated judge as a live guardrail.

Aryan Meena | Target: ML Engineer / AI Engineer roles | Budget: $0 (optional $25 at the end)
Realistic timeline: **3 weeks part-time**, ~15-20 hrs/week. Build this one first.

---

## 0. What you are building, in one paragraph

Everyone building with LLMs uses an LLM to grade outputs. Almost nobody measures whether that grader is any good. JudgeGuard manufactures ground truth without human annotators, runs a battery of controlled bias probes across judge configurations and model families, reports every number with a confidence interval and a significance test, then distills the winning judge into a cheap classifier served as an inline guardrail with a real latency budget.

**The one-line version for a recruiter:** "I measured how often LLM judges are wrong, including on agent trajectories, and shipped the calibrated judge as a sub-200ms guardrail."

---

## 1. Setup: the zero-budget provider stack

You need **multiple judges from different model families**. This is not optional: self-enhancement bias is unmeasurable with one model, and the free tiers hand you the diversity for free.

| Provider | What you use it for | Get key at |
|---|---|---|
| **Google AI Studio** | Your frontier-class judge. Gemini Flash, generous daily quota, 1M context, no card. | aistudio.google.com |
| **Groq** | Fast open-weight judge (Llama 3.3 70B). Very high tokens/sec, so bulk runs finish fast. | console.groq.com |
| **Cerebras** | Second open-weight family, high throughput. | cloud.cerebras.ai |
| **OpenRouter** | Model variety behind one key; use `:free` suffixed models. Good failover. | openrouter.ai |
| **Ollama (local)** | Student model experiments, embeddings, offline dev. No rate limit, your CPU/GPU is the cap. | ollama.com |

**Rate limits multiply if you route across providers.** Build the provider layer first (Phase 1) so switching costs you nothing.

### Three operational rules, learned from other people's pain

1. **Pin model IDs in config, log the served model on every call.** Providers retire models without notice; one provider's model list dropped to two entries overnight in May 2026. Your experiment must record what actually answered.
2. **Never trust a published rate limit.** Check your own account's active limits before you size a run. Quotas changed multiple times in the last year.
3. **Assume free-tier prompts may be used for training.** Your data is public benchmark text so this doesn't matter here, but never put anything private through a free tier.

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install httpx tenacity pydantic pydantic-settings diskcache \
            numpy pandas scipy statsmodels scikit-learn \
            sentence-transformers matplotlib seaborn \
            fastapi uvicorn python-dotenv pytest
```

Keys go in `.env`, `.env` goes in `.gitignore`, and `.env.example` gets committed with empty values.

---

## 2. Repo scaffold

```
judgeguard/
├── README.md                      # RESULTS FIRST. Setup goes at the bottom.
├── .env.example
├── pyproject.toml
├── configs/
│   ├── models.yaml                # pinned model ids per provider
│   └── experiments/               # one yaml per experiment run
├── judgeguard/
│   ├── providers/
│   │   ├── base.py                # LLMProvider protocol
│   │   ├── gemini.py  groq.py  cerebras.py  openrouter.py  ollama.py
│   │   └── cache.py               # disk cache keyed on (model, prompt, params)
│   ├── data/
│   │   ├── load.py                # reference QA/summarization items
│   │   └── schema.py              # Item, Response, Pair, Trajectory
│   ├── degrade/
│   │   ├── text.py                # 6 text degradations, severity-parameterised
│   │   └── trajectory.py          # 4 agent-trajectory degradations
│   ├── judges/
│   │   ├── configs.py             # vague / rubric / cot / pairwise
│   │   └── run.py                 # batch execution, retries, position swapping
│   ├── agent/
│   │   ├── tools.py               # 4 deterministic tools
│   │   └── rollout.py             # produce reference trajectories
│   ├── stats/
│   │   ├── intervals.py           # BCa bootstrap, paired bootstrap
│   │   └── tests.py               # McNemar, kappa, Spearman
│   ├── distill/
│   │   ├── labels.py              # generate training labels from best judge
│   │   └── train.py               # embeddings + classifier
│   └── serve/
│       ├── app.py                 # FastAPI: /evaluate /guard /report
│       └── latency.py             # p50/p95/p99 instrumentation
├── experiments/                   # one script per figure, all reproducible
├── results/                       # committed JSON + PNG so README renders offline
├── docs/
│   ├── methodology.md
│   ├── findings.md                # the paper-shaped writeup
│   └── limitations.md
├── tests/
├── Dockerfile
└── .github/workflows/eval.yml     # CI gate
```

---

## 3. Phase 1 — Provider layer and cache (Days 1-2)

This is the single highest-leverage two days. Everything downstream depends on being able to swap models and never pay twice for the same call.

```python
# judgeguard/providers/base.py
from typing import Protocol
from pydantic import BaseModel

class Completion(BaseModel):
    text: str
    model_served: str          # what ACTUALLY answered, not what you asked for
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    provider: str

class LLMProvider(Protocol):
    name: str
    def complete(self, prompt: str, *, model: str,
                 temperature: float = 0.0, max_tokens: int = 512) -> Completion: ...
```

```python
# judgeguard/providers/cache.py
import hashlib, json
from diskcache import Cache

_cache = Cache(".cache/llm")

def cache_key(provider: str, model: str, prompt: str, **params) -> str:
    blob = json.dumps({"p": provider, "m": model, "prompt": prompt, **params},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def cached_complete(provider, prompt, *, model, **params):
    key = cache_key(provider.name, model, prompt, **params)
    if key in _cache:
        return _cache[key]
    result = provider.complete(prompt, model=model, **params)
    _cache[key] = result
    return result
```

Wrap every provider call in `tenacity` retry with exponential backoff and jitter, because free tiers throttle. Handle 429 by backing off, not by failing the run.

**Definition of done for Phase 1:** you can run the same prompt against five providers with one function call, results are cached to disk, and a rerun of the whole script costs zero API calls.

---

## 4. Phase 2 — Data and the degradation engine (Days 3-4)

### 4.1 Choosing data

You need items with a **reference answer**, not human preference labels. Good free options: a public QA set with gold answers, or a summarization set with reference summaries. Start with **300-500 items**. You can scale later; you cannot un-waste quota.

Store everything as:

```python
class Item(BaseModel):
    id: str
    context: str        # source doc or question context
    question: str
    reference: str      # the known-good answer
```

### 4.2 The methodological core: gold labels by degradation

You have no human raters and no budget for them. The solution is to **manufacture ground truth by breaking a known-good answer in a controlled way.** If a judge cannot rank the reference above a deliberately broken variant, that is a measured failure with no annotation required.

```python
# judgeguard/degrade/text.py
"""
Each degradation takes (item, severity: float in [0,1]) -> str
severity=0.1 is a subtle break, severity=1.0 is an obvious one.
Severity is what lets you plot the discrimination curve.
"""

DEGRADATIONS = {
    "omission":     drop_required_facts,      # remove n facts, n scales with severity
    "fabrication":  inject_plausible_falsehood,
    "numeric_swap": perturb_a_number,         # 42 -> 47; subtle but definitively wrong
    "verbosity":    pad_without_information,  # 1.5x..3x length, zero new content
    "topic_drift":  answer_adjacent_question,
    "hedging":      replace_claims_with_vagueness,
}
```

Two design rules that make or break this:

- **`verbosity` must be applied to the reference, not to a degraded answer.** It is the one degradation that adds no error. If the judge scores the padded reference *higher* than the clean reference, you have measured verbosity bias directly, in points.
- **Every degradation must be verifiably worse.** If you cannot state in one sentence why a human would prefer the reference, drop that degradation. Log the exact edit made so you can audit any surprising result.

Generate degradations with your cheapest fast model (Groq), and spot-check 30 by hand. Commit the generated set to `results/` so the experiment is reproducible without regenerating.

**Definition of done:** `results/degraded_set.json` containing ~500 items × 6 degradation types × 3 severity levels, each with the reference, the degraded variant, and the edit description.

---

## 5. Phase 3 — Judge configurations (Days 5-6)

Four configurations, escalating in structure:

| Config | Prompt shape | Hypothesis being tested |
|---|---|---|
| `vague` | "Is this a good answer? Score 1-10." | Baseline. Expect worst discrimination. |
| `rubric` | Explicit criteria + score bands (0.8-1.0 excellent, 0.5-0.7 adequate...) | Structure improves consistency. |
| `cot` | Rubric + "reason step by step before scoring" | Chain-of-thought judging is documented to improve alignment with human judgment. |
| `pairwise` | Show two answers, pick the better one | Different failure surface; enables position-bias measurement. |

### The position-swap protocol (non-negotiable for pairwise)

```python
def pairwise_judge(judge, a: str, b: str) -> dict:
    fwd = judge.compare(first=a, second=b)   # -> "first" | "second" | "tie"
    rev = judge.compare(first=b, second=a)
    # translate rev back into A/B space
    rev_norm = {"first": "B", "second": "A", "tie": "tie"}[rev]
    fwd_norm = {"first": "A", "second": "B", "tie": "tie"}[fwd]
    return {
        "forward": fwd_norm,
        "reverse": rev_norm,
        "consistent": fwd_norm == rev_norm,   # <-- position bias signal
        "verdict": fwd_norm if fwd_norm == rev_norm else "inconsistent",
    }
```

**Disagreement rate between forward and reverse orderings IS your position-bias metric.** Report it per judge model. Then report accuracy with and without swap-and-average to quantify how much the standard mitigation buys you.

Run every configuration at `temperature=0` for the main results, and a separate run at `temperature=0.7` × 5 repeats to measure **self-consistency** (how often the same judge disagrees with itself).

---

## 6. Phase 4 — The statistics layer (Days 7-8)

**This is your differentiator. Build it before you run the big experiments so no bare number ever enters your results.**

```python
# judgeguard/stats/intervals.py
import numpy as np
from scipy.stats import bootstrap

def bca_ci(values, statistic=np.mean, confidence=0.95, n_resamples=10_000, seed=0):
    """BCa bootstrap CI. Use for every headline accuracy number."""
    rng = np.random.default_rng(seed)
    res = bootstrap((np.asarray(values),), statistic,
                    confidence_level=confidence, method="BCa",
                    n_resamples=n_resamples, random_state=rng)
    return float(statistic(values)), float(res.confidence_interval.low), \
           float(res.confidence_interval.high)

def paired_bootstrap_diff(a, b, statistic=np.mean, n_resamples=10_000, seed=0):
    """
    Is judge A better than judge B? Resample ITEM INDICES jointly so the
    pairing is preserved. Returns (observed_diff, ci_low, ci_high, p_two_sided).
    """
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b), "paired test needs aligned per-item scores"
    rng = np.random.default_rng(seed)
    n = len(a)
    observed = statistic(a) - statistic(b)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = statistic(a[idx], axis=1) - statistic(b[idx], axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    centered = diffs - diffs.mean()
    p = float((np.abs(centered) >= abs(observed)).mean())
    return float(observed), float(lo), float(hi), p
```

```python
# judgeguard/stats/tests.py
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

def mcnemar_test(correct_a, correct_b):
    """Paired binary decisions. Use exact=True when discordant counts are small."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    table = [[0, b], [c, 0]]
    return mcnemar(table, exact=(b + c) < 25)

def judge_agreement(scores_a, scores_b):
    """Cohen's kappa between two judges' discrete verdicts."""
    return cohen_kappa_score(scores_a, scores_b)

def severity_monotonicity(severities, scores):
    """A good judge's score should decrease monotonically with degradation severity."""
    rho, p = spearmanr(severities, scores)
    return {"spearman_rho": rho, "p_value": p}
```

**Rule for the whole project:** every number in the README, in `findings.md`, and on your resume carries a CI. If you cannot put an interval on it, you do not report it.

---

## 7. Phase 5 — The bias battery (Days 9-11)

Run these as separate scripts in `experiments/`, each writing JSON to `results/`.

| Experiment | Output | The headline you want |
|---|---|---|
| `01_discrimination.py` | accuracy vs severity, per degradation type, per judge | **The main figure.** Where does each judge go blind? |
| `02_position_bias.py` | fwd/rev disagreement rate, accuracy with vs without swap | "Order swapping cut position-bias disagreement from X% to Y%." |
| `03_verbosity_bias.py` | Δscore when padding the reference | "Padding a correct answer to 2x length raised its score by X points." |
| `04_self_enhancement.py` | judge X on X-output vs Y-output | Needs ≥2 families. Free tiers give you 4. |
| `05_rubric_ablation.py` | vague vs rubric vs CoT, paired tests | "CoT rubric beat vague prompting by X pts, p = ..." |
| `06_self_consistency.py` | kappa across 5 reruns at temp 0.7 | "The same judge disagreed with itself on X% of items." |
| `07_cost_accuracy.py` | accuracy per dollar and per second, per model | "The free 70B judge reached X% of frontier accuracy at 1/Nth the cost." |

Experiment 07 is your most quotable result and it is the argument for the entire distillation phase.

---

## 8. Phase 6 — Agent trajectories (Days 12-14)

**This is the part that separates you from every other eval project**, and it maps directly onto job descriptions mentioning agentic workflows.

### 8.1 The toy agent: scope discipline

Four **deterministic** tools. Deterministic matters, because it means the correct trajectory is unambiguous and you need no human to label it.

```python
TOOLS = {
    "calculator":  lambda expr: eval_safe(expr),
    "lookup":      lambda key: STATIC_TABLE[key],       # fixed local dict
    "date_diff":   lambda d1, d2: (parse(d2) - parse(d1)).days,
    "unit_convert":lambda v, frm, to: CONVERSIONS[(frm, to)] * v,
}
```

Write ~60 tasks that require 2-4 tool calls each. Record the correct trajectory for each:

```python
class Trajectory(BaseModel):
    task: str
    steps: list[dict]      # [{tool, args, result, reasoning}, ...]
    final_answer: str
```

### 8.2 The four trajectory degradations

```python
# judgeguard/degrade/trajectory.py
def length_padding(traj):      # insert redundant but harmless steps
def phantom_tool(traj):        # insert a call to a tool that does not exist
def wrong_argument(traj):      # correct tool, wrong argument value
def silent_failure(traj):      # right final answer, wrong path
```

Then ask each judge to compare reference vs degraded and measure:

- **Trajectory-length bias** — does it prefer the longer path when the extra steps add nothing?
- **Hallucination blindness** — does it notice a tool that was never defined?
- **Argument blindness** — does it notice a wrong argument value passed to a correct tool?
- **Path blindness** — does it notice a correct answer reached the wrong way?

Argument blindness and hallucination blindness are the interesting ones, because they are exactly the failures that matter in production and exactly what output-only evaluation cannot catch. If your judges are bad at these, say so loudly. **A negative result, measured rigorously, is a real contribution.**

---

## 9. Phase 7 — Distillation (Days 15-16)

The current industry pattern is a cheap distilled evaluator scoring high volumes continuously, with the expensive judge reserved for deep verification. You are implementing that pattern, not a two-year-old tutorial.

```python
# judgeguard/distill/train.py
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

encoder = SentenceTransformer("all-MiniLM-L6-v2")   # runs locally, free

X = encoder.encode([f"{q} [SEP] {a}" for q, a in pairs])
y = best_judge_labels                                # from Phase 5's winner

clf = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="isotonic", cv=5)
clf.fit(X, y)
```

Report the trade honestly: accuracy retained, cost reduction, latency reduction. Use `paired_bootstrap_diff` between the teacher judge and the student to say whether the accuracy loss is statistically real. **If the gap is inside the CI, say so.** "The student was statistically indistinguishable from the teacher at 1/40th the cost" is a much stronger sentence than a bare accuracy number.

Calibration matters: the guardrail needs a probability you can threshold, not a raw score. Plot a reliability curve.

---

## 10. Phase 8 — Serving and deployment (Days 17-19)

### 10.1 The distinction that makes this an engineering project

An **async evaluator** scores after the fact for dashboards and regression tracking; latency is irrelevant. An **inline guardrail** blocks bad output before the user sees it and has a hard millisecond budget. These are not interchangeable, and building an async evaluator where you needed a guardrail means harmful outputs reach users.

Build both paths and measure both.

```python
# judgeguard/serve/app.py
from fastapi import FastAPI
app = FastAPI(title="JudgeGuard")

@app.post("/evaluate")   # async path: full LLM judge, returns score + reasoning + CI
@app.post("/guard")      # inline path: distilled classifier, allow/block + latency header
@app.get("/report")      # the bias battery results as JSON
@app.get("/health")
```

Set an explicit **p99 budget of 150 ms** for `/guard`. Instrument p50/p95/p99 under load (`locust` or a simple asyncio driver). Report which configurations meet the budget and what accuracy you trade to get there. This one design discussion is worth more in an interview than the entire modeling section.

### 10.2 Deploy

```dockerfile
FROM python:3.12-slim AS builder
# ... install deps into a venv
FROM python:3.12-slim
COPY --from=builder /opt/venv /opt/venv
COPY judgeguard/ results/ ./
CMD ["uvicorn", "judgeguard.serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Host on **Fly.io or Railway** (free allowance, scale-to-zero) or **Hugging Face Spaces** (best for a clickable UI). Commit `results/*.json` so `/report` and the README render even if every API key expires.

### 10.3 Front end

A small Streamlit or static page: paste a response, see the judge score, see the guardrail verdict, see the measured latency. Plus a results page with the bias battery charts. **Recruiters click; they do not clone.**

---

## 11. Phase 9 — CI gate, README, writeup (Days 20-21)

### CI gate (rare in portfolios, strong signal)

```yaml
# .github/workflows/eval.yml
name: eval-gate
on: [push, pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .[dev]
      - run: pytest tests/
      - run: python experiments/regression_suite.py --fail-under 0.85
```

Eval-driven development with a failing build on regression is a current practice and almost no portfolio has it.

### README structure — results first

```markdown
# JudgeGuard
**Finding:** [judge] failed to detect malformed tool arguments in X% of agent
trajectories (95% CI [a, b]) — a failure class invisible to output-only evaluation.

[HEADLINE CHART: accuracy vs degradation severity]

## Three findings
1. ...  2. ...  3. ...

## Try it        <- live link, right here
## How it works  <- architecture diagram
## Reproduce     <- setup instructions LAST
```

### `docs/findings.md`

Write this as you go, not at the end. This is the artifact you send when someone says "tell me about a project," and it is what makes the work read as research rather than a tutorial follow-along. You have done exactly this twice already (WikiFlow, the CS 767 report). Same muscle.

### `docs/limitations.md`

Be explicit that degradation-based gold labels test *discrimination against known-bad*, not *alignment with human preference on genuinely ambiguous pairs*. Naming what your method cannot show is the single clearest senior signal in the whole repo.

---

## 12. Where the $25 goes (only at the very end)

Build everything on free tiers. When the pipeline is proven and you are not burning calls on debugging, spend once:

- One paid frontier arm on the discrimination + trajectory experiments, so you can write: *"the free 70B judge reached X% of frontier judge accuracy at 1/Nth the cost."*
- Budget: ~$4 per full battery run on a cheap paid model; $25 covers the final runs plus headroom.
- Cheapest paid step when you outgrow free limits is usually DeepSeek's API.

If you never spend it, the project still stands.

---

## 13. Deliverables checklist

- [ ] Repo with results committed so it renders without an API key
- [ ] Live demo URL (guardrail + judge, with latency shown)
- [ ] Headline chart: discrimination accuracy vs severity, per judge, with CIs
- [ ] Position, verbosity, self-enhancement bias measured with intervals
- [ ] Four agent-trajectory blindness measurements
- [ ] Distilled student with the accuracy/cost/latency triangle stated
- [ ] p99 latency number for the guardrail path
- [ ] CI gate that fails on eval regression
- [ ] `findings.md` + `limitations.md`
- [ ] LinkedIn Featured: live link, repo, findings doc

---

## 14. Resume bullets (fill in real numbers)

> Built an LLM judge-reliability harness using degradation-generated gold labels, quantifying position, verbosity and self-enhancement bias across N judge configurations and 4 model families with BCa bootstrap CIs and paired significance tests; order-swapping reduced position-bias disagreement from X% to Y%.

> Extended evaluation to agent trajectories, measuring failure classes invisible to output-only scoring: judges missed malformed tool arguments in X% of cases and undefined tool calls in Y%.

> Distilled the calibrated judge into a classifier statistically indistinguishable from the teacher at 1/Nth the cost, served as an inline guardrail at p99 X ms behind FastAPI on Docker, with an eval suite gating CI.

---

## 15. Failure modes to avoid

**Letting it become "I evaluated some LLMs."** The subject is judge reliability. If a reader thinks you built a benchmark, you built the common thing. If they think you built a way to know when your evaluation is lying to you, you built the rare thing. Every README sentence should reinforce that.

**The research half eating the engineering half.** Given your instincts this is the real risk: a beautiful bias study and no shipped service. The deployed guardrail with a latency number is what makes this an *engineering* portfolio piece. Timebox the analysis and force yourself into Phase 8 by day 17.

**Scope creep on the agent.** Four deterministic tools. Not a framework, not a general agent. It exists only to generate trajectories you can break.

**Reporting bare numbers.** You built the stats layer in Phase 4 specifically so this cannot happen. Use it.

**The paper option:** if the trajectory results are interesting, this is workshop-publishable. The novel angle is not the bias taxonomy (that exists) but the systematic transfer of degradation-based gold labelling from text to agent trajectories with a full statistical treatment. You have taken a course project to a Springer acceptance once already.
