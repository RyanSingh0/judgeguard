# Portfolio Plan: Two Projects That Actually Differentiate

Aryan Meena | July 2026

---

## 0. The finding that changes the plan

I researched what's currently in public portfolios before designing these. Two things you need to know:

**A RAG demo signals nothing in 2026.** LangChain + Chroma + RAGAS over a document corpus is the single most-built portfolio project of the last two years. ResearchRAG as originally scoped will not move you ahead of anyone.

**A generic uplift project is also saturated.** There are public repos benchmarking 7 meta-learners across 5 datasets with bootstrap confidence intervals, and separate repos shipping S/T/X/R learners with a Qini evaluation and a FastAPI decision endpoint. "I built uplift models on Criteo" is now table stakes, not a differentiator.

So building the obvious version of either gap gets you parity, not advantage.

**Where the advantage actually is.** Both of your gaps are the same underlying skill wearing two costumes: *deciding whether a system's own evaluation can be trusted.*

- LLM side: the judge scores the model. But who scores the judge? Judge scores are known to be systematically optimistic, and published bias studies show position bias, verbosity bias and self-enhancement bias are real and measurable.
- Causal side: the uplift model estimates a treatment effect you can never observe for any individual. Qini and AUUC are themselves noisy estimators, and most portfolios report a single Qini number as if it were a fact.

That skill is already your strongest habit and it's visible across your existing work:

| Existing work | The instinct |
|---|---|
| WikiFlow | Trained NN to AUC 0.909, then benchmarked against logistic at 0.845 to test whether the complexity earned its cost |
| Drowsiness | LSTM hit validation F1 1.000, you diagnosed memorization and shipped the RF that held 0.923 across all splits |
| DXY | Re-tested every result at 0, 1 and 2 bps because a 219% return usually means the test is wrong |
| Drowsiness Task 2 | Custom loss making the model predict its own confidence |
| RetinaFace stage | Reported mAP 0.6123 (WIDER) vs 0.9528 (FDDB) and explained the domain shift rather than quoting the better number |

Nobody is going to out-compete you on that instinct. Build the two projects that put it on display in the two domains where you have no evidence yet.

---

# PROJECT 1 — JudgeGuard
### Can you trust the judge? Measuring LLM evaluator reliability, then shipping the calibrated judge as a live guardrail

**Target roles:** ML Engineer, AI Engineer, Applied Scientist
**Closes:** the LLM / transformer / evals / guardrails gap that blocked you at Quora
**Realistic time:** 2.5 to 3 weeks part-time. Not a weekend. I was wrong when I said that earlier.

## 1.1 The pitch in one paragraph

Everyone building with LLMs uses an LLM as a judge to score outputs. Almost nobody measures whether that judge is any good. JudgeGuard is a harness that (a) manufactures ground truth without human annotators, (b) runs a battery of controlled bias probes against multiple judge configurations, (c) reports every result with confidence intervals and paired significance tests, and (d) distills the winning judge configuration into a cheap fast classifier served as an inline guardrail with a real latency budget.

Half of it is a research contribution. Half of it is a deployed service. That combination is the thing your portfolio is missing.

## 1.2 The methodological trick that makes this possible solo

The obvious objection to any judge-evaluation project: *you need human raters to know if the judge is right, and you don't have any.*

The answer is **gold-labelling by degradation**, a principle from recent judge-reliability research. Instead of collecting human preferences, you start from a known-good response and apply a controlled perturbation that definitionally makes it worse. Now you have a gold pair with no humans involved:

```
reference response  (known good)
        │
        ├── degrade: drop a required fact          → known worse
        ├── degrade: inject a plausible fabrication → known worse
        ├── degrade: pad to 2x length, no new info  → known worse (verbosity probe)
        ├── degrade: swap a numeric value           → known worse
        └── degrade: answer a neighbouring question → known worse
```

If a judge cannot reliably rank the reference above a degraded variant, that's a measured, reportable failure. Severity is a knob, so you can plot judge accuracy against degradation magnitude and find where the judge goes blind. **That curve is your headline figure.** It is the equivalent of your WikiFlow baseline comparison: it converts a vague claim ("judges are unreliable") into a measurement.

## 1.3 What to measure

### Tier 1: single-response scoring
- **Discrimination**: can the judge separate reference from degraded? Report accuracy per degradation type and per severity level.
- **Verbosity bias**: pad the *reference* with filler that adds no information. Does the score go up? Known documented bias; you're quantifying its size.
- **Position bias**: run every pairwise comparison in both orderings. Disagreement rate between (A,B) and (B,A) is a direct bias measurement. The standard mitigation is swapping and averaging, so implement it and report the improvement.
- **Self-enhancement bias**: if judge model X evaluates output from model X vs model Y, is it biased toward its own? Needs at least two model families to test.
- **Rubric specificity**: vague prompt vs explicit rubric with score bands vs chain-of-thought rubric. Published work says CoT judging improves human alignment; test whether it improves *your* degradation-detection accuracy.

### Tier 2: agent trajectory scoring
This is where you separate from anyone else doing eval projects, and it maps directly onto job descriptions that mention agentic workflows. Build a small toy agent (3 to 5 tools: search, calculator, lookup, format), generate reference trajectories, and degrade the *trajectory* rather than the text:

- **Trajectory-length bias**: insert redundant but harmless steps. Does the judge prefer the longer path?
- **Hallucination blindness**: insert a call to a tool that was never defined. Does the judge notice?
- **Argument blindness**: keep the tool correct but pass a wrong argument value. Does the judge notice?
- **Silent failure**: correct final answer reached via a wrong path.

Argument blindness and hallucination blindness are the two most interesting because they are exactly the failures that matter in production and exactly the ones output-only evaluation cannot catch.

### Tier 3: the statistics layer (your edge)
This is where you do what almost no portfolio project does, and it's a straight transfer of the DXY methodology.

- **BCa bootstrap 95% CIs on every accuracy number.** No bare point estimates anywhere.
- **Paired bootstrap significance test** between judge configurations, so you can state which differences are real and which are noise.
- **McNemar's test** for paired binary judge decisions.
- **Cohen's kappa / Krippendorff's alpha** for agreement between judge configurations, and between judge runs at temperature > 0 (self-consistency).
- **Spearman correlation** between judge score and degradation severity. A good judge should be monotonic in severity; where the correlation breaks is where the judge stops discriminating.
- **Cost per correct judgment**, not just accuracy. A judge that's 3 points better at 30x the price is usually the wrong choice, and saying so out loud is a senior signal.

## 1.4 The engineering half

Research alone gets you a nice repo. The engineering half is what converts it into an MLE hire signal.

**Distillation.** Take the best-performing judge configuration, run it over a few thousand examples, and train a small classifier (start with logistic regression or gradient boosting on embeddings; a small fine-tuned encoder if you want to stretch) on its labels. Report the accuracy you lose and the cost and latency you gain. The current industry pattern is exactly this: cheap distilled evaluators scoring high volumes of production traffic, with the expensive judge reserved for deep verification. You will be implementing a pattern that's current, not a tutorial from two years ago.

**Guardrail vs evaluator, and why the distinction matters.** These are not interchangeable and confusing them is a real production failure:
- An **async evaluator** scores after the fact for dashboards and regression tracking. Latency is irrelevant.
- An **inline guardrail** blocks a bad output before the user sees it. It has a hard millisecond budget.

Build both paths and measure them. Set an explicit p99 budget for the guardrail (say 150 ms), show which configurations meet it, and show what accuracy you trade to get there. This one design discussion is worth more in an interview than the whole modeling section.

**Serving.**
```
FastAPI
  POST /evaluate    → full judge, async path, returns score + reasoning + CI
  POST /guard       → distilled classifier, inline path, returns allow/block + p99 latency header
  GET  /report      → the bias battery results, served as JSON
Docker (multi-stage, slim base)
Deploy: Fly.io or Railway (cheapest for an always-on demo) or HF Spaces (best for a clickable UI)
Observability: structured logs → a simple dashboard showing score distributions and block rate
CI: GitHub Actions running the eval suite on every push, failing the build on regression
```

That last line matters. **Eval-driven development with a failing CI gate is a genuinely current practice** and almost no portfolio has it.

**Front end.** A small Streamlit or React page: paste a response, see the judge score, see the guardrail verdict, see the latency. Plus a static results page with the bias battery charts. Recruiters click; they do not clone.

## 1.5 Repo layout

```
judgeguard/
├── README.md                  # results first, not setup first
├── judgeguard/
│   ├── degrade/               # perturbation generators, text + trajectory
│   ├── judges/                # judge configs: vague, rubric, CoT, pairwise, swapped
│   ├── agent/                 # toy 3-5 tool agent + trajectory recorder
│   ├── stats/                 # BCa bootstrap, paired tests, kappa, McNemar
│   ├── distill/               # label generation + student model training
│   └── serve/                 # FastAPI app, guardrail path, latency instrumentation
├── experiments/               # one script per figure, all reproducible
├── results/                   # committed JSON + charts so the README renders without a run
├── docs/
│   ├── methodology.md
│   ├── findings.md            # the paper-shaped writeup
│   └── limitations.md         # what degradation-based gold labels cannot tell you
├── tests/
├── Dockerfile
└── .github/workflows/eval.yml # CI gate
```

## 1.6 Build order

**Week 1 — measurement spine**
1. Pick a task and a source of reference responses. Public QA or summarization data is fine; you need reference answers, not human preferences.
2. Implement 5 to 6 text degradations with a severity parameter.
3. Implement 3 judge configurations (vague, explicit rubric, CoT rubric).
4. Run the discrimination experiment. First real chart: accuracy vs severity.
5. Wire in bootstrap CIs from the start so you never report a bare number.

**Week 2 — bias battery and agents**
6. Position bias with order swapping, verbosity bias with filler padding, self-enhancement if you have two model families available.
7. Build the toy agent and record reference trajectories.
8. Implement the four trajectory degradations. Measure hallucination blindness and argument blindness.
9. Run all paired significance tests. Write findings.md as you go, not at the end.

**Week 3 — ship it**
10. Distill the winning configuration. Measure the accuracy, cost and latency triangle.
11. FastAPI with both paths, latency instrumentation, Docker.
12. Deploy. Get the URL working.
13. CI gate. Front end. README with results at the top.

## 1.7 The failure mode to avoid

Do not let this become "I evaluated some LLMs." The project is about **judge reliability**, and every README sentence should reinforce that. If a reader comes away thinking you built a benchmark, you built the common thing. If they come away thinking you built a way to know when your evaluation is lying to you, you built the rare thing.

## 1.8 Resume bullets (fill in your real numbers)

> Built an evaluation harness measuring LLM judge reliability using degradation-generated gold labels, quantifying position, verbosity and self-enhancement bias across N judge configurations with BCa bootstrap CIs and paired significance tests; order-swapping reduced position-bias disagreement from X% to Y%.

> Extended the harness to agent trajectories, where judges failed to detect malformed tool arguments in X% of cases and undefined tool calls in Y%, failures invisible to output-only evaluation.

> Distilled the calibrated judge into a classifier retaining X% of accuracy at 1/Nth the cost, served as an inline guardrail at p99 latency of X ms behind FastAPI on Docker, with an eval suite gating CI.

## 1.9 The paper option

This is publishable at a workshop, the same way WikiFlow went to CSECS. The novel angle is not the bias taxonomy (that exists) but the **systematic transfer of degradation-based gold labelling from text to agent trajectories with a full statistical treatment**. If your Tier 2 results are interesting, write it up. You have done this once already and you know the pipeline.

---

# PROJECT 2 — LiftCheck
### Uplift modeling where the contribution is knowing which differences are real

**Target roles:** Data Scientist, Decision Scientist, Marketing/Product Analytics
**Closes:** the causal inference and uplift gap flagged across your DS applications
**Realistic time:** 10 to 14 days part-time

## 2.1 The pitch

Standard uplift portfolios train S/T/X/R learners, report a Qini number, and stop. The problem they ignore is that **you never observe the counterfactual for any individual**, so Qini and AUUC are noisy estimators and the ranking between methods is often statistically meaningless. Published benchmarks find that on true RCT data, propensity-aware methods carry no advantage and simpler better-regularized methods win, and that bootstrap CIs for top methods overlap heavily.

LiftCheck estimates treatment effects *and* quantifies how much of the method ranking is signal. Then it turns the estimate into a budget-constrained targeting decision served behind an API.

## 2.2 Data

| Dataset | Size | Notes |
|---|---|---|
| **Hillstrom** | ~64K rows, 8 features | Email marketing RCT. Auto-downloadable. Start here. |
| **Criteo-UPLIFT v2** | ~14M rows (25M raw), 12 features | Advertising incrementality RCT, binary treatment, visit and conversion labels. Sample it; you don't need all of it. |
| **Synthetic confounded DGP** | your choice | Essential. Real RCTs have known treatment assignment, so confounding-robust methods can't demonstrate their advantage. A confounded simulator with known ground-truth CATE is the only place you can measure actual estimation error. |

The synthetic component is not a shortcut, it's the scientific control. Say that explicitly in the README.

## 2.3 Methods

Use `causalml` and `econml` rather than reimplementing, with one exception below.

- S-learner, T-learner, X-learner, R-learner
- Doubly robust / DR-learner
- Causal forest
- Class transformation

**Implement the R-learner from scratch** and cross-check it against the library version by rank correlation. One from-scratch implementation proves you understand the machinery; seven proves only that you have time.

## 2.4 The differentiating layer

1. **BCa bootstrap CIs on Qini and AUUC for every method.** Then state plainly which methods are statistically indistinguishable. Most candidates cannot bring themselves to write "the difference between my best two models is noise." Writing it is the signal.
2. **Paired bootstrap tests between method pairs.**
3. **Propensity overlap diagnostics.** Where common support fails, CATE estimates are unreliable, and showing you check this separates you from people who don't know it's a requirement.
4. **Sensitivity analysis.** This is your DXY move transplanted: vary sample size, base learner, and confounding strength in the synthetic DGP, then show where each method breaks. Cross-fit methods are known to fail at small sample sizes because nuisance estimation goes unstable, so you should be able to reproduce that.
5. **Policy value, not just ranking metrics.** Qini is an ordering metric. The business question is "if I can treat 20% of users, how much incremental outcome do I get versus random targeting and versus treat-everyone?" Report policy value at several budget tiers with CIs.

## 2.5 The deployed half

```
FastAPI
  POST /score   → {uplift, ci_low, ci_high, decision, reason}
  POST /policy  → given budget b, return the treat/don't-treat set and expected incremental value
Streamlit
  budget slider → live Qini frontier, live expected-value readout
```

The `reason` field matters. Returning "treat: estimated uplift 0.12, CI [0.04, 0.19], above threshold at current budget" instead of a bare number is what a decision system looks like versus a model output.

## 2.6 Resume bullets

> Benchmarked 7 uplift meta-learners across Hillstrom, a Criteo-UPLIFT sample and a confounded synthetic DGP, reporting Qini and policy value with BCa bootstrap CIs; found the top N methods statistically indistinguishable on RCT data while cross-fit learners degraded below X samples.

> Implemented the R-learner from scratch, validated at X rank-correlation against econml and ~0 ATE error on simulated data with known ground-truth effects.

> Shipped a budget-constrained targeting API returning per-user uplift with confidence intervals and a treat/hold decision, plus a Streamlit interface for sweeping budget against the Qini frontier.

---

# 3. Sequencing, and what happens to the old roadmap

Your existing plan was DataForge → ResearchRAG → SignalServe. Here's my honest revision.

### Build order

**1. JudgeGuard.** First, and it isn't close.

The argument is marginal value. Your DS evidence is already deep: fraud classifier, hospital readmission, NHIS, DXY, jet engine RUL, MBTA graph work. Your LLM evidence is **zero**, and that zero is what disqualified you from a stated minimum requirement at a role paying $107K to $153K. Adding a seventh DS project moves you from strong to slightly stronger. Adding your first LLM project moves you from ineligible to eligible for an entire salary band.

It's also the least saturated of everything on this list, which means it's the one that can carry a workshop paper.

**2. LiftCheck.** Second. Real gap, genuinely recurring across your DS applications, and 10 to 14 days. But it's an upgrade to an already-strong portfolio rather than the removal of a blocker.

**3. SignalServe.** Third, and reduced in scope. JudgeGuard's guardrail service already gives you "model behind a live API, containerized, latency-instrumented." What SignalServe still adds uniquely is **drift monitoring and Terraform IaC**, which is a real MLOps signal. Keep it, build it after the first two, and consider trimming it to just the drift + Terraform piece rather than rebuilding a serving layer you'll already have.

**4. DataForge.** Last, and only if you re-target Data Engineering. Every posting you've brought me lately has been DS, MLE or analyst. A serverless Glue/Athena ETL project is excellent DE evidence and near-zero signal for the roles you're actually applying to. Don't spend two weeks on it while the LLM gap is open.

### ResearchRAG: cancel it

Absorb it into JudgeGuard. If you want retrieval in the portfolio, make the JudgeGuard evaluation task a RAG task, so faithfulness and context-relevance become two of your degradation dimensions. You get the RAG keyword coverage for free inside a project that actually differentiates, instead of spending a week on a demo that thousands of applicants have already built.

### What this does to your resume

Once JudgeGuard ships, the MLE resume becomes: WikiFlow (distributed DL at scale, published), JudgeGuard (LLM evaluation, guardrails, serving), Drowsiness (multi-stage pipeline, generalization investigation), MedAI (3D deep learning, team lead). That is a genuinely strong four-project new-grad ML portfolio, and every one of them has a real finding rather than a metric.

Once LiftCheck ships, the DS resume gets its causal inference answer and you stop having to name that gap in cover letters.

---

# 4. How to show it off

**README structure.** Results first. A recruiter or hiring manager gives you 30 seconds. Open with the headline chart and three findings in plain language, then architecture, then setup. Setup instructions at the top is the most common portfolio mistake.

**One clickable link per project.** A live URL beats a repo by a wide margin because it's verifiable in ten seconds. Budget for keeping it cheap: scale-to-zero hosting, cached results so the demo works even if an API key expires.

**Write the findings document.** For both projects. This is the thing you already know how to do from WikiFlow and the CS 767 report, and it's what makes the work look like research rather than a tutorial follow-along. It's also what you send when someone asks "tell me about a project."

**Post about the finding, not the project.** "I measured how often LLM judges miss malformed tool arguments" is a post people read. "I built an eval harness" is not.

**Update LinkedIn Featured as each ships.** Live demo link, then repo, then findings doc.

---

# 5. Honest risks

**JudgeGuard costs money in API calls.** Budget for it. Cache aggressively, keep sample sizes modest and justified by your power analysis rather than by budget, and say in the README that sample sizes were chosen for statistical power. Use cheaper models for the bulk runs and reserve the expensive judge for the comparisons that need it.

**Three weeks is a real three weeks.** If your interview pipeline heats up, JudgeGuard is the one to protect and LiftCheck is the one to postpone.

**Don't let the research half eat the engineering half.** Given your instincts, the risk is that you build a beautiful bias study and never ship the service. The deployed guardrail with a latency number is what makes it an *engineering* portfolio piece. Timebox the analysis and force yourself into week 3.

**Scope discipline on the agent.** Three to five tools. Not a framework, not a general agent. It exists solely to generate trajectories you can degrade.
