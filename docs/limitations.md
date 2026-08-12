# Limitations

> Naming what a method cannot show is not a disclaimer section. It is the part of the work that
> decides whether the rest of it can be believed.

---

## 1. The committed numbers are simulated, and that is the biggest caveat here

The run that produced `results/` had no provider API keys. Every judge call was served by the
deterministic model in `configs/simulator.yaml`.

**What this does not compromise.** The harness, the degradation engine, the swap protocol, the
statistics layer, the distillation pipeline, the guardrail service and the CI gate are all real and
all exercised identically whether the bytes came from Google or from `simulated.py`. The latency
benchmark is a genuine measurement of a genuine model on real hardware. The calibration finding in
§9 of `findings.md` is a property of thresholding a scalar score and would survive any teacher.

**What it does compromise.** Every claim of the form *"model X missed Y% of Z"* is a property of
the simulator, not of model X. The parameters are priors — informed ones, written where they can be
argued with, but priors.

**Why it was built this way.** A repository that only runs for someone holding four API keys does
not get run. A CI eval-gate that needs a live quota is not a gate, and cannot execute on a fork's
pull request. The simulator makes the whole pipeline executable, testable and gated by anyone who
clones it, in seconds, for free.

**How to remove this limitation:** set one key, `JUDGEGUARD_PROVIDER_MODE=live`, rerun the
identical commands. A live run that **contradicts** the priors would be the most valuable result
this project could produce, and the priors are version-controlled specifically so that
contradiction is legible rather than embarrassing.

## 2. Degradation-based gold labels test discrimination against known-bad, not alignment with human preference

This is the central methodological limitation.

What the method establishes: *given a reference answer and a copy broken in a specific, auditable
way, does the judge prefer the reference?*

What it does **not** establish:

- **Alignment with human preference on genuinely ambiguous pairs.** Real evaluation is mostly two
  plausible answers with different emphases, tone or structure — the hard, interesting case. Every
  pair here has a right answer by construction. A judge could score 100% on this battery and still
  rank two good answers exactly backwards.
- **Calibration to human severity judgements.** Severity is a *construction* parameter, not a
  human-rated one. The x-axis of the headline figure should be read as "magnitude of the injected
  defect", not "human-perceived badness".
- **That the degradations are representative of real failure modes.** They were selected because
  they are controllable and auditable. Real LLM failures include tone mismatch, unhelpful refusal,
  subtle reasoning errors and instruction non-compliance, none of which appear here.
- **Cross-degradation comparability.** Saying numeric swap is "harder" than topic drift compares two
  constructions, not two natural error rates.

**The honest framing:** this measures a *necessary* condition for judge reliability, not a
sufficient one. A judge that fails here is definitely unreliable. A judge that passes has cleared a
floor.

**What would fix it:** ~200 gold pairs rated by 3 human annotators, reporting inter-annotator
agreement and the correlation between human preference and the degradation label. That validates the
construct rather than assuming it. It is the first thing I would spend money on.

## 3. The corpus is generated, which trades realism for auditability

Procedural generation is what makes the perturbations exact. The cost is real:

- **Register.** Templated declarative summaries. Uniform sentence structure, no discourse markers,
  no idiom. Real model output is messier.
- **Distribution.** 10 domains, 6 facts each, ~41 words per reference. Nothing long-form,
  conversational, multi-turn, or code.
- **Fact independence.** Facts do not entail one another, so omitting one never makes another
  incoherent. Real answers have dependency structure and errors propagate through it.
- **Verifiability is total.** The source record states every value, so every perturbation is
  checkable from the prompt. That is deliberate — it removes an excuse from the judge — but it is
  friendlier than reality, where sources are partial and contradictory.
- **Simulator interaction.** In simulated mode the same seed governs the corpus and the judge model.
  In live mode this concern disappears.

`judgeguard/data/load.py` accepts a JSONL corpus for live runs. Degradation auditing is coarser
there, and a custom corpus **must** embed the source values or the experiment measures the corpus
rather than the judge (methodology §2.1).

## 4. The agent is a toy, deliberately

Four deterministic, stateless, pure-function tools over a 20-entry table; 60 tasks of 3–4 steps.
Determinism is load-bearing: it makes the correct trajectory unambiguous without a human.

That excludes non-deterministic tools, side effects, retries and error recovery, long-horizon
planning, multi-agent handoff, tool selection under ambiguity, and any task with more than one
correct trajectory. Real agent evaluation is mostly those.

Read the argument-blindness result as: *judges miss malformed tool arguments even in the easiest
possible setting* — deterministic tools, short traces, action space printed in the prompt. That is a
lower bound, which is the useful direction for a negative result to point, but it is not a
measurement of production agent evaluation.

Only three trajectory degradations count toward detection accuracy, so per-class intervals rest on
180 comparisons per judge and are correspondingly wide.

## 5. Statistical caveats

- **Multiplicity is corrected within one family, not globally.** Holm–Bonferroni covers the 12
  config comparisons. Comparisons *across* experiments are not jointly corrected; individual
  intervals are marginal, not simultaneous.
- **The bootstrap resamples pairs, not items.** Variants from the same item share a reference and
  are correlated. The stricter unit is the item, giving slightly wider intervals; resampling pairs
  is mildly anti-conservative for the by-degradation breakdowns.
- **Ties count as failures** (1.7–2.7% of pointwise comparisons). A judge that ties honestly is
  penalised identically to one that is confidently wrong.
- **Order inconsistency counts as a failure** under swap-and-average. Defensible for a production
  gate, but it makes these numbers incomparable with any published figure that resolves ties in the
  judge's favour.
- **The teacher's "oracle threshold" is tuned on the labels it is scored against.** It is an upper
  bound, labelled as one in the results JSON. The student never sees the test split.
- **Self-consistency at temperature 0.7 is one point on a curve**, chosen as a common production
  default, not swept.

## 6. Cost and latency caveats

- **Costs are list-price estimates, not paid invoices**, and are flagged as such in
  `configs/models.yaml`. Every call ran on a free tier. Prices move; the cost ranking in §7 of
  `findings.md` will age faster than any other result here. Verify them before publishing that
  table.
- **Judge latencies in simulated mode are modelled, not measured.** Only the guardrail path is a
  real measurement, and it is labelled as such on the figure.
- **The latency benchmark is single-process on 2 cores** with no network, TLS, load balancer or cold
  starts. Absolute numbers are hardware-dependent; the ratio between the guardrail path and the
  judge path is not.
- **Token accounting in simulated mode is approximate** (~4 characters per token).

## 7. Distillation caveats

- **The student is a bag-of-n-grams model over the answer plus coverage features against the
  source.** It cannot check whether a number is *right*. Its 0.0% on `numeric_swap` and on
  same-domain `topic_drift` is a property of the model class, not a tuning failure. Its 77.0%
  overall would not transfer to a different corpus without retraining, and it is not a
  general-purpose evaluator — which is why the deployable configuration is a cascade rather than a
  replacement.
- **The class balance of the distillation set is a design choice, not a measurement.** The battery
  enumerates 15 error variants per reference; training at that ratio teaches the base rate rather
  than the judgement. The student is trained one-to-one. On production traffic with a different base
  rate the operating point must be re-chosen, and `block_precision` and `false_block_rate` — not
  accuracy — are the numbers to re-check.
- **Free recalibration is only free here.** It works because degradation ground truth is
  manufacturable. Where ground truth requires annotation, the recalibration slice costs exactly what
  the annotations cost.
- **Student and teacher were evaluated on the same corpus the student was trained on** (disjoint
  splits, same generator). No out-of-distribution generalisation is claimed.

### 7b. The student's features can memorise the degradation templates

`fabrication` draws from six fixed filler-claim sentences and `verbosity` from six fixed filler
sentences. A character/word n-gram model sees the same templates in training and test, so its 100%
accuracy on `fabrication` is partly memorisation of a closed vocabulary rather than detection of
unsupported claims.

The honest reading of the per-degradation table: `hedging` and `omission` reflect real signal
(hedge-word density and source coverage generalise), `fabrication` is inflated, and `numeric_swap`
and `topic_drift` at 0.0% are real and unimprovable by this model class. A template-disjoint
evaluation — train on three of six templates, test on the others — would separate the two, and is
the first fix I would make.

### 7c. The student blocks padded-but-correct answers

Held-out verbosity probes are allowed 0% of the time. Padded text contains no error, so under the
labelling used for the *bias* measurement it is a false block. Under the rubric used by the judges —
which names padding as a defect — blocking it is arguably correct. The two definitions disagree, and
the project uses both: the probe defines ground truth for measuring judge bias, and does not define
ground truth for the guardrail. That is stated rather than resolved, because resolving it requires
deciding whether padding is a defect in *your* product.

For symmetry: the judges scored padding *up* by as much as 1.45 points, and the student blocks it
outright. Neither is calibrated to "length is neutral".

## 8. Threats to validity, briefly

- **Construct validity** — does "prefers the reference over a broken copy" measure judge quality?
  Partially (§2).
- **Internal validity** — in simulated mode the data-generating process and the judge model share an
  author. Live mode removes this.
- **External validity** — one corpus, one language, one register, four models, one temperature, one
  toy action space.
- **Researcher degrees of freedom** — simulator parameters, severity levels, accept threshold and
  rubric wording were all chosen by me. They live in version-controlled config files rather than in
  code, so they can be changed and the effect observed, which is the mitigation available without
  preregistration.

## 9. What this project does *not* claim

- Not that any named commercial model has the properties reported here.
- Not that judge accuracy on this battery predicts downstream product quality.
- Not that the distilled student is a general-purpose evaluator.
- Not that 150 ms is the correct guardrail budget for anyone else's system.
- Not that the degradation taxonomy is complete.

What it does claim: **judge reliability is measurable without annotators, the measurement is cheap
enough to run in CI, and the failure modes it exposes on agent trajectories are invisible to
output-only evaluation by construction.**
