# LinkedIn — ready to post

Three versions. **Version A is the one to post.** It leads with the finding, not the project.

Attach `results/figures/fig_00_headline.png` natively. Put links in the **first comment** — LinkedIn
suppresses reach on posts with outbound links in the body.

---

## ✅ Version A — lead with the finding (recommended)

> I asked four LLM judges to review an AI agent's work.
>
> I had secretly changed one number in one tool call. The tool was right. The step count was
> right. The reasoning read perfectly. One argument was wrong.
>
> They missed it between 37% and 64% of the time.
>
> That's the part that stuck with me. Not that judges make mistakes — everyone knows that. It's
> *which* mistakes. Ask a judge to spot an answer that's obviously off-topic and it catches it
> 95–100% of the time. Ask it to spot a number quietly changed from 11.7 to 15.2 and it drops to
> 68%.
>
> And here's the uncomfortable bit: the correct value was printed in the source document I handed
> the judge. The information was right there.
>
> Judges catch the errors that *look* wrong. They miss the errors that *are* wrong.
>
> On agent trajectories this gets worse, because the failure that matters most in production — the
> right tool called with a wrong value — leaves no trace on the surface. And if you only evaluate
> the final answer, you can't see it at all. I built a case where the agent reaches the correct
> answer through a broken path: output-only evaluation scores that perfect, 100% of the time, by
> construction.
>
> So I built JudgeGuard to measure it properly.
>
> The hard part was the labels. Measuring a judge seems to need human annotators, and I don't have
> any. The way around it: don't collect labels — manufacture them. Start with an answer you know is
> good, then break it in a controlled, auditable way. Drop a required fact. Change one number. Pad
> it to 3× length with sentences that say nothing. Now the right ranking is known by construction,
> and if the judge gets it backwards, that's a measurement, not an opinion.
>
> Then I turned severity into a dial and watched where each judge goes blind.
>
> A few things I didn't expect:
>
> → Padding a **correct** answer to 3× length with pure filler raised its score by up to 1.45
> points out of 10. 85–94% of padded answers beat the identical unpadded original. A rubric line
> that explicitly said "padding is a defect" didn't stop it.
>
> → Swapping which answer is shown first flipped the verdict on up to 32% of comparisons. If your
> eval harness always puts the reference first, your accuracy number is inflated by up to 12
> points.
>
> → The best judge in the panel lost **14 points** purely to where its accept threshold sat. It
> ranks answers beautifully — 95.4%. As a yes/no gate at its own rubric's cutoff: 80.0%. Same
> signal, different cut. Ranking well and deciding well are not the same skill, and almost nobody
> measures the second one.
>
> The last piece is the one I'd defend in an interview, partly because it didn't go how I wanted.
>
> I distilled the winning judge into a classifier small enough to run *inline* — 4ms instead of
> 700ms. My first version scored 94.8% and I was pleased with myself for about an hour, until I
> worked out it was blocking 83% of everything. At the class balance my test set happened to have,
> a *correctly calibrated* model assigns "probably bad" to any answer, including the correct one.
> It was learning the base rate, not the judgement.
>
> Fixed properly, it scores 77.0% against the teacher's 94.0%. That's a real 17-point gap and
> there's no confidence interval to hide in.
>
> But look at *where* it loses. Hedging: 100% vs 100%. Correctly allowing good answers: 100%. Then
> numeric substitution: 0%, versus 57.9% for the judge. Not a tuning failure — a bag-of-words model
> can't verify that a number matches a record.
>
> So the answer isn't a bigger student. It's a cascade: the small model absorbs what it can
> actually see and escalates the rest. Escalating 75% of traffic matched judging everything with
> the LLM (94.5% vs 94.0%, overlapping intervals) at 75% of the cost. A 25% saving is modest, and
> it's the number the data supports.
>
> Two things I'd put on a slide:
>
> The guardrail blocks 53% of bad output at **100% block precision and a 0% false-block rate** — it
> never blocked a correct answer. For an inline gate that's the number that decides whether anyone
> lets it near production. Accuracy isn't.
>
> And distillation copies the teacher's *calibration* error along with its judgement — the
> student's probability means "the teacher would accept this", not "this is acceptable". Since my
> gold labels cost nothing to manufacture, recalibrating on them cost nothing either: ECE 0.193 →
> 0.015.
>
> Free labels → free calibration. That's the part I think is genuinely new.
>
> (My favourite detail: the judges scored padding **up** by 1.45 points. The distilled student
> blocks padded answers outright. One has a pro-verbosity bias, the other an anti-verbosity bias,
> and neither is neutral about length.)
>
> Every number carries a 95% bootstrap confidence interval. Every judge comparison is paired.
> Multiplicity is corrected. There's a CI gate that fails the build if the evaluator gets worse —
> and it's already caught real bugs in my own gold labels.
>
> The best bug was found by a model, not by me. Early on, judges were scoring my *correct*
> reference answers 2 out of 10. One of them explained why: "the candidate answer fabricates
> specific numerical values that are entirely absent from the provided source context." It was
> right. My source documents named the metrics but never stated their values, so nothing was
> verifiable and the whole experiment would have measured my corpus instead of the judges. Fixed
> the corpus; the judges immediately started scoring references 10/10.
>
> One caveat I'll state loudly, because I'd want someone else to: the committed results come from a
> deterministic simulator, since the run had no API keys. The harness is real, the latency
> measurements are real, the method is real. The model-specific numbers are priors waiting to be
> replaced by one environment variable — and I'd find it more interesting if a live run proved them
> wrong.
>
> Repo, live demo and full write-up in the comments. Built with FastAPI, scikit-learn, uv and ruff.
> MIT licensed.
>
> If you're running an LLM judge in production and have never measured it: the whole battery runs
> in about four minutes, with no API key, on a laptop.
>
> #LLM #MachineLearning #MLEngineering #AIEvaluation #AIAgents #Guardrails #MLOps

**First comment:**
> 🔗 Live demo: [YOUR-URL]
> 📦 Repo: github.com/RyanSingh0/judgeguard
> 📄 Full findings: github.com/RyanSingh0/judgeguard/blob/main/docs/findings.md
> ⚠️ What this method *can't* show: github.com/RyanSingh0/judgeguard/blob/main/docs/limitations.md

---

## Version B — short, punchy, high scroll-stop

> Everyone uses an LLM to grade their AI's output.
> Almost nobody checks whether the grader is any good.
>
> So I checked. Four judges, 7,500 test cases, confidence intervals on everything.
>
> **What they catch:** an answer that's obviously off-topic → 95–100%
> **What they miss:** one number quietly changed from 11.7 to 15.2 → up to 32% missed
>
> The correct value was printed in the source document I gave them.
>
> Judges catch errors that *look* wrong. They miss errors that *are* wrong.
>
> It gets worse with agents. I gave them a trajectory where the right tool was called with a wrong
> argument — perfect-looking transcript, one bad value.
>
> Missed 37–64% of the time.
>
> And when the agent reaches the *correct* answer through a broken path? Output-only evaluation
> scores that 100%. By construction. It cannot see the problem.
>
> Three more things that surprised me:
> • Padding a correct answer with filler raised its score by up to 1.45/10
> • Swapping the order of two options flipped the verdict up to 32% of the time
> • The best judge lost 14 points purely to where its accept threshold sat
>
> The trick that made it possible without annotators: don't collect labels, manufacture them.
> Break a known-good answer in a controlled way, and the correct ranking is known for free.
>
> Then I shipped the calibrated judge as an inline guardrail. 4ms. Blocks 53% of bad output at
> 100% block precision — it never blocked a correct answer.
>
> Repo + live demo in comments 👇
>
> #LLM #AIEvaluation #MLEngineering #AIAgents

---

## Version C — for the ML-research audience

> **Can you trust the judge?**
>
> LLM-as-a-judge is the default eval method for open-ended generation. Evaluating the judge itself
> looks like it needs human preference labels — which is why almost nobody does it.
>
> It doesn't. **Gold labelling by degradation:** take a known-good reference, apply a controlled
> perturbation that definitionally makes it worse, and the correct ranking is known by
> construction. Zero annotators.
>
> Three design choices carry the method:
>
> **1.** Reference answers are *composed* from an explicit fact list, so every perturbation is exact
> and its semantic footprint is enumerable. `"effect: 11.7 → 15.2"` is the audit trail.
>
> **2.** The source document contains the values, so every perturbation is verifiable from the
> prompt. I learned this the hard way: with a context that named the metrics but not their values,
> judges scored the *correct* reference 2/10 as unsupported, and discrimination would have collapsed
> to chance for entirely the wrong reason.
>
> **3.** One degradation introduces no error at all — it pads the reference with content-free
> filler. Correct score change: exactly zero. Whatever the judge does instead is verbosity bias,
> measured in its own points. (Result: up to +1.45/10.)
>
> **The contribution is the transfer.** Degradation-based gold labels exist for text. I applied the
> same construction one level up — to an agent's *process* rather than its output. Four
> deterministic tools, so the correct trajectory is unambiguous without a human. Then: undefined
> tool calls, wrong arguments passed to correct tools, and correct answers reached via corrupted
> paths.
>
> **Argument blindness was the worst class for every judge: 37.2%–64.4% missed.** Detection
> accuracy 62.8% [55.6, 69.4] down to 35.6% [28.9, 42.8]. The declared action space is printed in
> the prompt, so phantom tools are a lookup — and they're the *best*-detected class. Wrong arguments
> require an execution model the judge doesn't have.
>
> Statistical treatment, because the field needs more of it: BCa bootstrap intervals on every
> number, paired bootstrap for judge comparisons, McNemar for paired binary decisions,
> Krippendorff's α for the 5-rerun self-consistency setting, Holm–Bonferroni across the config
> grid, and sample sizes chosen by power analysis rather than by budget.
>
> Two corollaries I didn't anticipate.
>
> Distillation inherits the teacher's *calibration* error along with its decisions — the student's
> probability means "the teacher would accept this", not "this is acceptable" (ECE 0.193 against
> ground truth). Manufactured gold labels are free, so recalibrating on a held-out slice is free:
> **ECE 0.193 → 0.015**.
>
> And distillation *over*-transfers blind spots. The student matches the teacher on hedging
> (100/100) and scores 0.0% on numeric substitution where the teacher gets 57.9%. That's structural
> for a bag-of-n-grams model. Hence a cascade — escalate 75% of traffic, match judge-only accuracy
> (94.5% vs 94.0%) at 75% of the cost — rather than a replacement.
>
> Full disclosure: the committed results come from a deterministic simulator, with parameters
> written as explicit falsifiable priors and every figure watermarked. The harness, the statistics
> and the latency measurements are real. A live run that contradicts the priors would be the most
> interesting outcome available, and it's one environment variable away.
>
> Workshop paper draft, code, committed results and a reproducible eval gate are all in the repo.
> Happy to be told where the method is wrong.
>
> #MachineLearning #NLP #LLMEvaluation #AIAgents #Research

---

## Posting notes

**Do**
- Post 8–10am Tue/Wed/Thu in your timezone.
- Attach the image natively. Links in the **first comment**, not the body.
- Reply to every comment in the first two hours — that window decides reach.
- Keep the first two lines strong; everything after is behind "…see more".

**Don't**
- Don't open with "I built a project". Open with what you found.
- Don't hide the simulated-results caveat. Stating it plainly is a credibility *gain*.
- Don't use more than ~7 hashtags.

**LinkedIn Featured, in this order:** live demo URL → `docs/findings.md` → the repo.

**Résumé bullets (real numbers from `results/`):**

> Built an LLM judge-reliability harness using degradation-generated gold labels, quantifying
> position, verbosity and self-enhancement bias across 4 judge configurations and 4 model families
> over 7,500 gold pairs with BCa bootstrap CIs and Holm-corrected paired tests; showed fixed-order
> evaluation over-reports accuracy by up to 12.1 pp [11.1, 13.2].

> Extended the harness to agent trajectories, measuring failure classes invisible to output-only
> scoring: judges missed malformed tool arguments in 37.2–64.4% of cases and undefined tool calls
> in 10.0–43.9%, with 95% CIs on every estimate.

> Distilled the calibrated judge into a 5 ms classifier serving as an inline guardrail at p99
> 7.2 ms behind FastAPI on Docker, blocking 53% of bad output at 100% block precision and a 0%
> false-block rate; cut expected calibration error 0.193 → 0.015 via free recalibration on
> manufactured gold labels, and showed a student-judge cascade matching judge-only accuracy
> (94.5% [90.5, 97.0] vs 94.0% [90.0, 96.5]) at 75% of the cost, with an eval suite gating CI.
