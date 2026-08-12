# Methodology

> Every number in this repository was produced by a script in `experiments/`, written to a
> committed JSON file in `results/`, and carries a 95% BCa bootstrap interval. `make verify`
> re-derives each quoted figure from those files and fails if the prose has drifted.

---

## 1. The problem this method solves

Evaluating an LLM judge appears to require human preference labels, and those are expensive, slow
and noisy. That constraint is why most people using an LLM-as-a-judge in production have never
measured whether their judge is any good.

**Gold labelling by degradation** removes the constraint. Start from a known-good reference answer
and apply a controlled perturbation that definitionally makes it worse:

```
reference response  (known good)
        │
        ├── drop a required fact              → known worse   (omission)
        ├── inject a plausible falsehood      → known worse   (fabrication)
        ├── perturb a numeric value           → known worse   (numeric_swap)
        ├── answer a neighbouring question    → known worse   (topic_drift)
        ├── replace claims with vagueness     → known worse   (hedging)
        └── pad to 3x length, no new content  → NOT worse     (verbosity — the probe)
```

Now there is a gold-ordered pair with no annotator involved. If a judge cannot rank the reference
above a deliberately broken copy, that is a measured, reportable failure.

Severity is a knob in `[0, 1]`, which turns "judges are unreliable" into a curve with an x-axis.
Where that curve collapses is where the judge has gone blind, and that point is far more useful
operationally than a mean accuracy.

## 2. Why the corpus is generated rather than downloaded

The method is only defensible if you can state exactly what a perturbation destroyed. With a
free-text gold answer you can delete a sentence, but you cannot enumerate which required facts went
with it, nor regenerate a fluent answer identical to the reference except for one corrupted number.

So each reference answer is **composed from an explicit fact list**:

```python
Fact(
    key="effect",
    value="11.7",
    kind=NUMERIC,
    unit="percent",
    clause="The primary endpoint improved by {value} relative to placebo.",
)
```

The reference is the join of the per-fact clauses. Dropping a fact and recomposing yields a
grammatical, fluent answer that is *provably* missing exactly one piece of required information.
Every `Variant` therefore carries:

| field | what it records |
|---|---|
| `edit` | one sentence a human can audit: `"effect: 11.7 -> 15.2"` |
| `facts_removed` / `facts_corrupted` | the semantic footprint of the change |
| `introduces_error` | `False` only for the verbosity probe |
| `len_ratio` | length relative to the reference, so length can be controlled for |

500 items across 10 domains, 6 required facts each, seeded and reproducible offline.

### 2.1 The source document must contain the values

This is load-bearing, and it was found on the very first live API call.

Originally the context named which quantities were reported but not their values. Asked to score
the *correct* reference, a real judge returned:

> `SCORE: 2 — The candidate answer fabricates specific numerical values for all trial metrics that
> are entirely absent from the provided source context.`

The judge was right. Under that design the reference and a `numeric_swap` variant are equally
unsupported, discrimination collapses to chance for entirely the wrong reason, and the experiment
measures the corpus rather than the judge. Each context now embeds a `SOURCE RECORD` listing every
fact and its value, which makes numeric substitution **checkable from the prompt** — and turns "the
judge missed it" into a finding rather than an artefact.

## 3. Two rules that make or break the degradation set

1. **Every degradation must be verifiably worse, in one sentence.** If you cannot state why a human
   would prefer the reference, it does not belong in the battery.
2. **`verbosity` is the exception, and it is applied to the reference itself.** It adds length and
   nothing else, so the correct score change is exactly zero. `introduces_error=False` marks it so
   no accuracy metric ever counts it as a discrimination item.

The eval gate enforces both structurally: it fails the build if any error variant is textually
identical to its reference, if any variant lacks an audit trail, or if the probe set contains
anything but `verbosity`. Two real no-op bugs were caught this way — a `topic_drift` clause
substitution that happened to match, and a `numeric_swap` on a small value that rounded back to the
original string.

## 4. Judge configurations

The judge is **reference-free**: it sees the source context, the question and a candidate, never
the gold answer. A test asserts the reference never leaks into a prompt.

| config | shape | hypothesis under test |
|---|---|---|
| `vague` | "Is this a good answer? Score 1–10." | baseline; expect the worst discrimination |
| `rubric` | 5 explicit criteria + score bands | structure improves consistency |
| `cot` | rubric + reason-before-scoring | chain-of-thought judging improves alignment |
| `pairwise` | show two answers, pick one | different failure surface; enables position bias |

Main results use `cot` at `temperature=0`. Self-consistency uses `rubric` at `temperature=0.7` with
5 replicates.

**Reasoning models get a 10× output budget** (`reasoning: true` in `configs/models.yaml`). Measured
live: `qwen3.6-27b` spent 1,175 tokens on one judgement, ~1,100 of them thinking. At a small budget
it returns an empty string and looks like a broken provider. The parser also strips `<think>`
blocks, so a number in the scratchpad is never mistaken for the verdict.

## 5. The position-swap protocol

Every comparison runs in **both** orderings, and verdicts are normalised out of displayed-position
space (A/B) into semantic space (REF/DEG). The disagreement rate between the two orderings is not a
proxy for position bias; it **is** position bias.

| view | meaning |
|---|---|
| `correct_ref_first` | reference always shown as option A — what a naive harness reports |
| `correct_deg_first` | the same comparison flipped |
| `correct_single_random` | one ordering by seeded coin flip — the unbiased single-call estimate |
| `correct_swapped` | correct only if both orderings agree *and* agree on the reference |

Swap-and-average scores an ordering disagreement as a **miss** rather than resolving it in the
judge's favour. That is the conservative choice and it is stated, not hidden.

## 6. Agent trajectories: the transfer

The contribution is applying the same construction one level up — to the agent's **process**.

Four **deterministic** tools (`calculator`, `lookup`, `date_diff`, `unit_convert`). Determinism is
the whole point: because every tool is a pure function over a fixed table, the correct trajectory
is unambiguous and needs no human to label it. Every reference trajectory is executed for real, and
a test re-executes every recorded step to confirm the transcript is genuinely correct.
`calculator` evaluates through a whitelisted AST walk, never `eval`.

60 tasks, 3–4 tool calls each, four degradations:

| degradation | what it breaks | why it matters |
|---|---|---|
| `length_padding` | nothing — redundant but harmless steps | **probe** for trajectory-length bias |
| `phantom_tool` | calls a tool absent from the declared action space | the action space is printed in the prompt |
| `wrong_argument` | correct tool, wrong argument value | the transcript still reads as valid |
| `silent_failure` | correct final answer via a corrupted path | output-only evaluation scores this perfect |

## 7. The statistics layer

Built **before** the experiments, so "every number carries an interval" is enforceable rather than
aspirational.

- **BCa bootstrap** (not percentile) for every accuracy — accuracy near 0 or 1 is skewed and the
  percentile interval is visibly wrong there. Degenerate samples fall back to a Wilson interval.
- **Paired bootstrap** for judge-vs-judge and config-vs-config, resampling item indices *jointly*.
  Two judges scored on the same items are not independent samples.
- **McNemar's test** for paired binary decisions, exact when discordants are below 25.
- **Cohen's kappa** and **Krippendorff's alpha** for agreement; alpha in the 5-rater
  self-consistency setting.
- **Spearman monotonicity** of score against severity.
- **Holm–Bonferroni** across the 12 config comparisons.
- **Cliff's delta** alongside p, so "significant" can be distinguished from "large".
- **Power analysis** — 7,500 gold pairs resolve a 7 percentage-point difference at >99% power,
  α = 0.05, against a requirement of 575 for independent samples and 735 for paired McNemar. Sample
  sizes are chosen for power, not for budget.

## 8. Distillation, and the calibration finding

The student is trained on the **teacher's labels**, not on the degradation ground truth. That is
what distillation means, and it is the honest version of the experiment: the student is supposed to
inherit the teacher's mistakes.

Two sampling decisions are load-bearing, and both cost accuracy on paper while making the model
worth deploying:

1. **One-to-one balance.** The battery enumerates 15 error variants per reference; training at that
   ratio produces a model that scores 94.8% by learning the base rate. A correctly calibrated such
   model assigns p(good) ≈ 0.06 to *any* unpadded text, the reference included.
2. **Verbosity probes held out of training.** A padded reference carries label 1 and is trivially
   identifiable by its filler, which would make "contains filler" the dominant positive feature.

Three splits — train (teacher labels), calibration (**degradation ground truth**), test (untouched
by either fit). Distillation inherits the teacher's *calibration error* along with its decisions, so
a student fit purely on teacher labels emits probabilities meaning "the teacher would accept this".
Because degradation labels cost nothing, an isotonic map fitted on a held-out slice corrects the
mismatch at zero annotation cost: ECE against ground truth **0.193 → 0.015**.

Featurisation is word + character n-gram hashing over the response, twelve interpretable structural
signals (hedge density, filler count, numeric density, type-token ratio), and six coverage features
against the source. A sentence-transformers backend is available behind
`pip install judgeguard[embeddings]`, but its encode step alone spends more of the 150 ms budget
than the entire hashing pipeline — shipping the cheaper one *because the budget said so* is the
point.

The operating threshold is chosen from a stated risk tolerance, not from 0.5: among thresholds
whose false-allow rate stays within tolerance, take the one maximising balanced accuracy.

## 9. Evaluator versus guardrail

| | async evaluator | inline guardrail |
|---|---|---|
| when | after the fact | before the user sees the output |
| latency | irrelevant | hard millisecond budget |
| here | `POST /evaluate` — full LLM judge | `POST /guard` — distilled student only |

No network call happens inside `/guard`. That is the only way a hard budget can be honoured. The
benchmark reports a concurrency sweep rather than a single p99, because one p99 is a machine
fingerprint while "the budget holds up to concurrency N on M cores" is a deployment decision.

`/evaluate` attaches the judge's **measured reliability** to its score — discrimination accuracy,
weakest degradation class, order-inconsistency rate — so a caller can see how much to trust the
number. That is the thesis of the project rendered as an API field.

## 10. Reproducibility and provenance

- Every artefact carries `provenance`: mode, seed, git SHA, Python version, platform, timestamp.
- Every provider call is cached on `(provider, model, prompt, temperature, max_tokens, replicate)`.
  The replicate index is in the key on purpose: without it, five reruns at temperature 0.7 collapse
  onto one entry and the harness reports perfect self-agreement. That bug existed for one commit.
- `Completion.model_served` records what actually answered. Providers alias and retire model IDs
  without notice — every ID originally pinned in this repo was retired or de-freed before the first
  live run.
- `--quick` runs write to `results/_smoke/`, so a smoke test can never overwrite committed results.
- With no API keys the harness runs against a deterministic simulator (`configs/simulator.yaml`)
  whose parameters are written down as explicit, falsifiable priors. Artefacts are stamped
  `provenance.mode = "simulated"` and every figure is watermarked. See `limitations.md` §1.
