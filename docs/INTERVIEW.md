# Interview prep — know this project like you built it

*(You did. This is so you can talk about it under pressure without reaching.)*

Read §1 and §7 the night before. Read §4 if the role mentions evals, agents or guardrails.

---

## 1. The 30-second answer

> "Everyone building with LLMs uses another LLM to grade the output. Almost nobody measures
> whether that grader is any good — because checking it looks like it needs human annotators.
>
> I found a way around that: instead of collecting labels, manufacture them. Take an answer you
> know is good, break it in a controlled, auditable way, and now the correct ranking is known by
> construction. If the judge ranks them backwards, that's a measurement.
>
> The headline finding is that judges miss the errors that look most normal. They catch an
> off-topic answer 95–100% of the time, but a single changed number only 68–86% — even when the
> correct value is printed in the source document I gave them. And on agent trajectories it gets
> worse: a wrong argument passed to the right tool goes undetected 37 to 64% of the time.
>
> Then I distilled the judge into a 4-millisecond classifier and shipped it as an inline guardrail."

**Stop there.** Let them pick the thread. Do not narrate all ten experiments.

---

## 2. The 2-minute version (when they say "walk me through it")

Four beats, in this order:

**1. The problem.** LLM-as-a-judge is the default eval method now. It's known to be biased —
position, verbosity, self-preference — but almost nobody measures it *on their own task*, because
that seems to require human preference labels.

**2. The method.** Gold labelling by degradation. Start from a known-good reference and apply a
perturbation that definitionally makes it worse: drop a required fact, change a number, pad it to
3× length with filler. The correct ranking is known with zero annotation. Severity is a parameter,
so instead of "judges are unreliable" you get a curve with an x-axis — accuracy against how badly
you broke the answer.

**3. The contribution.** The bias taxonomy already exists. What's new is (a) making the
perturbation *auditable at the fact level* — every reference is composed from an explicit fact
list, so I can say exactly what each edit destroyed — and (b) transferring the whole construction
from text output to **agent trajectories**, breaking the agent's *process* instead of its answer.
That exposes failure classes output-only evaluation cannot see by construction.

**4. The engineering.** Every number carries a BCa bootstrap interval; judge comparisons are
paired; multiplicity is corrected. Then I distilled the winning judge into a classifier that runs
inline at p99 7 ms against a stated 150 ms budget, with a CI gate that fails the build if the
evaluator regresses.

---

## 3. The three findings, with the numbers

Memorise the shape, not every decimal. If you blank on a figure, say "roughly" — never invent one.

**Finding 1 — judges are blindest to the errors that look most fluent.**
Numeric substitution was the worst-detected class for all four judges (67.8%–85.9%), while
off-topic answers were caught 95–100% of the time. At a subtle ~4% perturbation the best judge sat
at 66.2% [62.0, 70.4] and the weakest at 54.8% [50.4, 59.0].
*The line that lands:* "Detection tracks how much of the surface the error disturbs, not how much
of the meaning it destroys."

**Finding 2 — order changes the verdict, and fixed-order harnesses over-report.**
The weakest judge flipped its verdict on 32.3% of comparisons purely from swapping which option
was shown first. A harness that always shows the reference first over-reports accuracy by up to
12.1 percentage points [11.1, 13.2].
*The line that lands:* "Swap-and-average doesn't make the judge more accurate. It makes the number
you were already reporting less of an illusion." (It *lowers* every reported accuracy, by 4–13 pp,
because it scores an ordering disagreement as a miss.)

**Finding 3 — ranking well and deciding well are different properties.**
The best judge ranks at 95.4%. As a yes/no gate at its own rubric's threshold it classifies at
80.0%. Moving the cut point recovers 94.0% — 14 points from an unchanged signal.
*The line that lands:* "A judge evaluated only on ranking accuracy can be shipped as an
accept/reject gate and quietly lose 14 points, and nothing on a leaderboard would show it."

**Agent trajectories (the differentiator).** Malformed tool arguments: 37.2%–64.4% missed, vs
10.0%–43.9% for a call to a tool that doesn't exist. The action space is printed in the prompt, so
phantom tools are a lookup the judge can do — a wrong argument needs an execution model it doesn't
have.

**Free labels buy free calibration.** Distillation copies the teacher's *calibration* error along
with its judgement: the student's probability means "the teacher would accept this", not "this is
acceptable" — ECE 0.193 against ground truth. Because the gold labels are manufactured, they're
free, so recalibrating on 200 of them was free too: ECE → 0.015.

---

## 4. Deep dives — be ready for the follow-up

### "How do you know the degraded version is actually worse?"

The best question they can ask. Answer:

> "Because I don't write free text — I compose it. Each reference is built from an explicit fact
> list, so dropping a fact and recomposing gives a fluent answer *provably* missing exactly one
> required item. Every variant carries a one-sentence audit trail like `effect: 11.7 -> 15.2`.
>
> And there's one degradation that is deliberately *not* worse: I pad the reference with
> content-free filler. No fact added, removed or altered, so the correct score change is exactly
> zero. Whatever the judge does instead is verbosity bias measured in its own points — up to +1.45
> out of 10, with 85–94% of padded answers beating the identical original."

### "Why not just use human labels / an existing benchmark?"

> "Cost and control. But the honest answer is that this method measures something *narrower* than
> human preference — it tests discrimination against known-bad, not alignment on genuinely
> ambiguous pairs. A judge could score 100% here and still rank two good answers backwards. It's a
> necessary condition, not a sufficient one, and that's the first thing in my limitations doc."

### "Why BCa bootstrap instead of a normal CI?"

> "Accuracy near 0 or 1 is skewed, and the percentile interval is visibly wrong there. BCa corrects
> for both median bias and acceleration. And the judge comparisons are *paired* — every judge sees
> every item — so I resample item indices jointly. Treating them as independent samples inflates
> the variance and hides real differences."

### "Why paired tests, and why Holm?"

> "Paired because the same items go to every judge, so McNemar and the paired bootstrap apply and
> are much more powerful. Holm because the config ablation is 12 comparisons — at α = 0.05 you
> expect a false positive by construction, so an uncorrected p < 0.05 would be a bug, not a result."

### "How did you pick the sample size?"

> "Power analysis, not budget. 7,500 gold pairs resolve a 7-percentage-point difference at over
> 99% power, against a requirement of 575 for independent samples and 735 for paired McNemar. I
> wanted to be able to say the size was a design decision rather than what the free tier allowed."

### "Walk me through the distillation."

> "Teacher is whichever judge/config won on measured accuracy — selected from the results, not
> picked by reputation. The student is a logistic model over hashed word and character n-grams of
> the response, plus twelve interpretable structural features and six coverage features against the
> source document.
>
> Crucially it's trained on the *teacher's* labels, not on ground truth. That's what distillation
> means, and it's the honest version — the student is supposed to inherit the teacher's mistakes.
> It does, and then some: it scores 0% on numeric substitution where the teacher gets 57.9%."

### "Why is the student so much worse than the teacher?" *(17 points — own it)*

> "It is, and there's no interval to hide in. But look at *where* it loses. Hedging: 100 vs 100.
> Correctly allowing good answers: 100 vs 99. It loses entirely on two classes — numeric
> substitution and same-domain topic drift — and both are structural. A bag-of-n-grams model cannot
> verify that a stated number matches a record.
>
> So the answer isn't a bigger student, it's a cascade: the small model absorbs what it can
> actually see and escalates the rest. Escalating 75% of traffic matched judging everything with
> the LLM — 94.5% vs 94.0%, overlapping intervals — at 75% of the cost. A 25% saving is modest,
> and it's the number the data supports."

### "Is that guardrail actually usable?"

> "The metric that decides it isn't accuracy — it's block precision. It blocks 53% of bad output at
> **100% block precision and a 0% false-block rate**. It never blocked a correct answer. A guardrail
> that blocks good output gets removed in a week no matter how accurate it is."

### "Why does the guardrail have no LLM call in it?"

> "Because a hard millisecond budget has to be a property of the code path, not a hope. An async
> evaluator scores after the fact for dashboards — latency is irrelevant. An inline guardrail
> blocks output before the user sees it and has a real budget. Those aren't interchangeable, and
> shipping an async evaluator where you needed a guardrail means harmful output reaches users while
> the evaluation is still in flight.
>
> I also report a concurrency sweep instead of a single p99, because one p99 is a machine
> fingerprint. 'The budget holds to concurrency 4 on 2 cores, and past that the queue owns the
> tail' is a sizing decision — and it says the fix is horizontal replicas, not a smaller model."

### "What's the CI gate?"

> "Eval-driven development. A change that makes the evaluator worse breaks the build exactly like a
> failing unit test. It checks guardrail accuracy, block precision, false-block rate, calibration
> error, p99 latency, and the structural invariants of the gold labels — no error variant may be
> identical to its reference, every variant must carry an audit trail.
>
> It also gates the *documented blind spots in both directions*: if the student ever starts
> detecting numeric substitution, the build fails, because my findings doc would then describe a
> model that no longer exists. And it runs against the simulator, so it needs no secrets and works
> on forked pull requests."

---

## 5. The bug stories — use these, they're your best material

Interviewers remember debugging stories far longer than metrics. Each one shows a different muscle.

**1. The cache key that faked perfect reliability.**
Self-consistency came back α = 1.000 — every judge perfectly agreeing with itself across five
reruns at temperature 0.7. That's impossible. The cache key didn't include the replicate index, so
all five reruns were one cached call replayed. *Shows:* you don't accept a result that's too good.

**2. The classifier that learned the base rate.**
First student scored 94.8%. I was pleased for about an hour, then worked out it was blocking 83% of
everything. The battery enumerates 15 error variants per reference, and at that class balance a
*correctly calibrated* model assigns "probably bad" to any answer, including the correct one. It
had learned the prior, not the judgement. Rebuilt one-to-one: 77.0%, and actually deployable.
*Shows:* you interrogate a good number as hard as a bad one.

**3. The corpus flaw a model found before I did.** *(the best one)*
On the very first live API call, judges scored my *correct* reference answers 2 out of 10. One
explained why: *"the candidate answer fabricates specific numerical values for all trial metrics
that are entirely absent from the provided source context."* It was right. My source documents
named the metrics but never stated their values, so nothing was verifiable — the reference and a
corrupted copy were equally unsupported, and the whole experiment would have measured my corpus
instead of the judges. Fixed the corpus; all four judges immediately went to 10/10 on references
and caught the numeric swap 3/3.
*Shows:* you read model output instead of just parsing a score out of it.

**4. Every pinned model ID was dead.**
Between writing the config and running it, all three open-weight IDs had been retired or de-freed,
and one provider returned `402 Payment required` on every model. That's why the code logs
`model_served` — what actually answered — separately from what was requested.
*Shows:* you've felt the operational reality of third-party model dependencies.

---

## 6. Own these before they find them

Volunteering a limitation is a seniority signal. Volunteering the *right* one is a strong one.

| weakness | how to own it |
|---|---|
| **Committed results are simulated** | "No API keys on the run that produced them. The harness, the statistics and the latency numbers are real; the model-specific numbers are explicit priors in a version-controlled config, every figure is watermarked, and one env var replaces them with measurements. Honestly, a live run that contradicted my priors would be the most interesting outcome available." |
| **Procedural corpus, not real data** | "Realism traded for auditability, deliberately. No public corpus lets me say exactly which required fact a perturbation destroyed. There's a JSONL loader for real corpora, and the trade is written down." |
| **The agent is a toy** | "Four deterministic stateless tools, on purpose — determinism is what makes the correct trajectory unambiguous without a human. So read the argument-blindness result as a *lower bound*: judges miss this even in the easiest possible setting, with the action space printed in the prompt." |
| **Student's 100% on fabrication** | "Partly memorisation — fabrication draws from six fixed templates and an n-gram model sees them in both splits. A template-disjoint split would separate signal from recall. It's in the limitations doc." |
| **The cascade only saves 25%** | "It's modest and it's what the data supports. I'd rather report a real 25% than a fabricated 10×." |

---

## 7. Questions *you* ask them

Signals you think about evaluation as a system, not a script.

- "How do you currently know when your evals are wrong? Is there anything gating a regression?"
- "Do you run online guardrails, offline evals, or both — and are they the same model?"
- "If you use an LLM judge, has anyone measured its position or verbosity bias on your task?"
- "For agent work — are you evaluating final outputs, or trajectories?"

---

## 8. Vocabulary to be fluent in

If any of these feel shaky, that's your revision list.

**Statistics:** BCa vs percentile bootstrap · paired vs independent tests · McNemar · Cohen's κ vs
Krippendorff's α · Spearman ρ · family-wise error and Holm · statistical power · Cliff's delta ·
expected calibration error · Brier score · isotonic vs Platt calibration · balanced accuracy vs
precision/recall.

**LLM eval:** LLM-as-a-judge · position/verbosity/self-enhancement bias · pointwise vs pairwise ·
rubric and CoT judging · self-consistency · reference-free vs reference-based · knowledge
distillation · guardrails vs evaluators · eval-driven development.

**Engineering:** p50/p95/p99 and why means lie · latency budget · concurrency sweep vs single p99 ·
scale-to-zero · multi-stage Docker builds · provenance and reproducibility · disk caching keyed on
call parameters.

---

## 9. Tailor the emphasis

| role | lead with |
|---|---|
| **ML / AI Engineer** | the guardrail: distillation, calibration, p99 budget, cascade, CI gate |
| **Applied Scientist / Research** | the method: degradation gold labels, the trajectory transfer, the statistics |
| **Data Scientist** | the statistics layer and the "this difference is noise" discipline |
| **Platform / Infra** | evaluator vs guardrail, the concurrency sweep, the eval gate, Docker/CI |

---

## 10. Two things not to do

**Don't oversell the simulated numbers.** Say "simulated" before they ask. Anyone senior who spots
you glossing over it will discount everything else you said.

**Don't lead with the tech stack.** "I built an eval harness with FastAPI and scikit-learn" is a
sentence people stop listening to. "I measured how often LLM judges miss malformed tool arguments"
is one they follow up on. The stack comes out in the follow-ups anyway.

---

## Cheat sheet

| | |
|---|---|
| Gold pairs | 7,500 text + 540 trajectory comparisons per judge |
| Panel | 4 judges, 4 model families, 3 providers |
| Numeric-swap detection | 67.8% – 85.9% (worst class for every judge) |
| Order-swap verdict flips | 11.6% – 32.3% |
| Fixed-order inflation | up to +12.1 pp [11.1, 13.2] |
| Verbosity bias | up to +1.45 points / 10 for pure padding |
| Self-enhancement | +0.12 to +0.79 pts, largest for the *best* judge |
| Self-consistency | Krippendorff α 0.182–0.545; none reach the 0.667 floor |
| **Argument blindness** | **37.2% – 64.4% missed** |
| Judge as a gate | 80.0% at its own rubric cut → 94.0% at the best cut |
| Student | 77.0% vs teacher 94.0%; blocks 53% of bad output at 100% block precision, 0% false-block |
| Calibration | ECE 0.193 → 0.015, free |
| Cascade | escalate 75% → 94.5% [90.5, 97.0] vs judge-only 94.0% [90.0, 96.5] |
| Guardrail latency | p50 3.8 ms, p99 7.2 ms vs a 150 ms budget |
