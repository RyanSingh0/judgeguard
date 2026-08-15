# Where JudgeGuard fits, and how to write it up

Working notes for the job hunt. Not part of the project docs.

---

## Read this before using anything below

**Every number in the repo right now came from a simulator.** README quotes 20
percentages, `docs/findings.md` quotes 59, `LINKEDIN_POST.md` quotes 12. None of
them describe a real model yet.

That LinkedIn draft opens with "I asked four LLM judges to review an AI agent's
work." I haven't. Posting that before the live battery finishes is a claim I
can't back, and the repo says so out loud: `provenance: simulated` is stamped on
every artefact. Anyone who clones it sees that in about ten seconds.

So the order is: **finish phase 1 → replace the numbers → then post, then
apply.** Roughly 4-5 days of daily runs.

What I *can* say truthfully today, if I need something now: "building an
evaluation harness for LLM judges; measurement in progress." That's a
conversation starter and it's accurate.

---

## Role fit, honestly

| Role | Fit | How to use it |
|---|---|---|
| **MLE** | Strongest | Lead project. Most of the work is MLE work. |
| **DS** | Strong | Lead or second project. The stats are the differentiator. |
| **QR** | Partial | One supporting line. Be ready to justify the connection. |
| **DA** | Weak-moderate | One line, reframed. Won't be what gets the interview. |
| **BA** | Weak | Probably leave it off. |

**Why MLE is the strongest fit.** Distillation, calibration, a latency budget
that's measured rather than asserted, a serving path, Docker, CI gates, a
provider abstraction with retry and rate-limit handling. That's the job.

**Why DS is close behind.** Bootstrap CIs on every estimate, a power analysis
that sized the sample before the run, McNemar for paired comparisons,
Krippendorff's alpha for agreement, stratified splits. Most portfolio projects
report a single accuracy number with no interval. This one can't, by design.

**Why QR is only partial.** The transferable part is measurement discipline:
isolating a confound (option ordering) and quantifying how much it inflates a
result, refusing to report a point estimate without an interval, sizing a sample
for power before spending the budget. That's real and it's QR-shaped. What's
missing is everything QR usually screens on — no market data, no time series, no
stochastic processes, no low-latency C++. Use it as evidence of rigour, not as a
finance project, and don't oversell it.

**Why DA/BA are weak.** These roles hire on SQL, stakeholder communication,
dashboards and business framing. A deep technical eval harness reads as
off-target and can suggest I'm applying to DA while wanting MLE. If I include
it, one line only, framed around the decision it supports rather than the method.

**A note to self on strategy:** five role families is too many for one resume.
DA/BA and MLE/QR are different hiring processes with different screens. Two
variants minimum — one technical (MLE/DS/QR), one business (DA/BA) — and
JudgeGuard is the headline on the first and a footnote on the second.

---

## Resume bullets

Square brackets are placeholders. Fill them from `live_numbers.txt` after phase
1. Do not fill them from the current results files.

### MLE version (3 bullets, lead project)

- Built an evaluation harness measuring LLM-judge reliability across four model
  families (Gemini, Llama 3.3 70B, Qwen 3.6 27B, GPT-OSS 20B) over ~32K API
  calls, generating ground-truth labels through controlled degradation instead
  of human annotation.
- Distilled the best-performing judge into a hashed n-gram classifier serving
  under a measured p99 ≤ 150 ms budget, with an isotonic calibration step that
  cut expected calibration error from [X] to [X]; ported inference to JavaScript
  with 1e-6 parity against scikit-learn, gated in CI.
- Rebuilt the runner around free-tier constraints: per-provider token-bucket
  limiter learned from response headers, quota-vs-rate-limit classification, and
  a resumable disk cache — cutting the battery from 64K to 32K calls and making
  a multi-day run restartable at zero repeat cost.

### DS version (3 bullets, lead or second)

- Measured how reliably LLM judges detect six classes of injected error,
  reporting BCa bootstrap confidence intervals on every estimate and sizing the
  sample with a power analysis before spending the API budget.
- Isolated option ordering as a confound by running every pairwise comparison in
  both orderings; a fixed-order harness over-reported accuracy by [X] points
  [CI], and the weakest judge reversed its own verdict on [X]% of comparisons.
- Showed a judge can rank well and still be miscalibrated as a decision: moving
  the threshold recovered [X] points from the identical signal, and recalibrating
  on manufactured gold labels cut ECE from [X] to [X].

### QR version (1 line, supporting)

- Designed and ran a controlled experiment isolating a confound in an automated
  scoring system, quantifying [X] points of bias from a fixed presentation order
  and reporting every estimate with a bootstrap confidence interval.

### DA version (1 line, only if there's room)

- Built an automated quality-check pipeline that flags unreliable machine-graded
  output before it reaches a decision, and quantified how often the grader itself
  is wrong.

---

## LinkedIn

The existing `LINKEDIN_POST.md` structure is fine — leads with the finding
rather than the project, which is right. Two things to fix before posting:

1. **Every number has to be real.** Rewrite after phase 1.
2. **The voice.** It reads polished in a way my own writing doesn't. Shorter
   sentences, fewer rhetorical turns, and cut the em-dashes.

### What to actually post

Post once, after the numbers are real, with the headline figure attached
natively and links in the first comment.

Lead with the concrete thing: an agent called the right tool with a wrong
argument, and the judges missed it [X]% of the time. Then the uncomfortable
detail — the correct value was printed in the source document the judge was
given. Then the method in one paragraph: don't collect labels, manufacture them
by breaking a known-good answer in a controlled way. Then one honest limitation,
because it's what makes the rest credible.

**Say the method is synthetic.** Degradation-generated labels are not human
labels, and someone will point that out in the comments. Better to get there
first — it reads as knowing the limits of my own method rather than being caught
not knowing.

### What not to do

- Don't post before the run finishes.
- Don't post three versions or repost the same finding weekly.
- Don't claim the judges are "broken." The finding is more specific and more
  interesting than that: they catch errors that *look* wrong and miss errors
  that *are* wrong.

---

## Interview answers worth having ready

- **"Why manufactured labels instead of a benchmark?"** Because I wanted the
  ground truth to be known by construction, and because human annotation wasn't
  available. The trade-off is that synthetic degradations may not match the
  error distribution of real model output — that's in `docs/limitations.md`.
- **"What went wrong?"** Good question to be honest on. The first live run died
  33 calls in because I treated a daily quota as a rate limit. Then I found
  llama-3.3-70b was producing 100% unparseable output because my token budget
  was 320 and it needed 380 — every call returned HTTP 200, so nothing raised.
  That one would have cost four days of quota. I wrote a preflight check for it.
- **"How do you know your numbers are right?"** There's a claim-checker that
  re-derives every quoted figure from the results files and fails CI on drift.
  It's how I caught two places where the docs had drifted from the data.
