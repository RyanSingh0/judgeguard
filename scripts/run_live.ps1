<#
.SYNOPSIS
    Run the bias battery against REAL models, sized to survive free-tier quotas.

.DESCRIPTION
    The old version of this script assumed the whole thing would finish in an
    afternoon. It can't. Counted against the providers' published limits the
    full battery is ~64,000 API calls, and the free tiers give about 1,000
    requests per model per day. That's two and a half weeks, and the first
    attempt died 33 calls in on a Gemini daily quota.

    Three changes:

    1. Runs in phases. Phase 1 covers the four experiments the three headline
       findings rest on, so once it's done every headline number is real.
       Phase 2 fills in the supporting tables.
    2. Sample sizes cut where the severity sweep bought resolution nothing
       quotes. Only experiment 01 plots a severity curve, so the rest run one
       mid severity. Saves about half the calls. The grid used is recorded in
       every results file.
    3. Running out of quota is a normal outcome now. An experiment that hits a
       judge's daily limit exits 42, writes nothing, and tells you to come back
       tomorrow. Work already done stays cached.

    RESUMING: just run the script again. Calls are cached on
    (provider, model, prompt, params, replicate), so completed work is free to
    skip. Do not delete .cache/.

.PARAMETER Phase
    1 = headline experiments only (default). 2 = supporting experiments.
    all = both, in order.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_live.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_live.ps1 -Phase all
#>

param(
  [ValidateSet("1", "2", "all")]
  [string]$Phase = "1"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:JUDGEGUARD_PROVIDER_MODE = "live"
# Concurrency 8 was part of what killed the first run. Groq's free tier gives
# qwen3.6-27b 8,000 tokens/min and a reasoning judgement costs ~2,000, so the
# real ceiling is about four calls a minute. The limiter enforces that anyway,
# but eight threads just queue up behind it.
if (-not $env:JUDGEGUARD_MAX_CONCURRENCY) { $env:JUDGEGUARD_MAX_CONCURRENCY = "4" }

$EXIT_QUOTA = 42

Write-Host ""
Write-Host "=== JudgeGuard LIVE battery - phase $Phase ===" -ForegroundColor Cyan
Write-Host "  Interrupted or quota-stopped runs resume for free. Do not delete .cache/."
Write-Host ""

uv run judgeguard status
if ($LASTEXITCODE -ne 0) { throw "judgeguard status failed" }

uv run python scripts/quota_status.py

# Twelve calls checking every judge emits a parseable score at the budget it
# will actually get. Cheap insurance against a run that finishes, looks fine,
# and produced nothing usable.
uv run python scripts/preflight.py
if ($LASTEXITCODE -ne 0) { throw "preflight failed - fix the judges above before spending quota" }

Write-Host ""
Write-Host "  Every panel row above must say 'live API'. Ctrl+C now if any says 'simulator'." -ForegroundColor Yellow
Start-Sleep -Seconds 6

# ---------------------------------------------------------------- phase plans
# Phase 1 covers all three headline findings:
#   01 -> "judges are blindest to the errors that look most fluent"
#   02 -> "swapping the options changes the answer"
#   08 -> "on agent trajectories the gap becomes a hole"
#   09 -> the distilled guardrail, the cascade, and the calibration result
$phase1 = @(
  @("dataset",           "00_build_dataset.py",        @("--items", "150")),
  @("discrimination",    "01_discrimination.py",       @("--items", "150")),
  @("position bias",     "02_position_bias.py",        @("--items", "100", "--severities", "0.5")),
  @("trajectory blind.", "08_trajectory_blindness.py", @("--trajectories", "60")),
  # 300 items rather than 150. Experiment 09 judges prompts experiment 01
  # already paid for, so most of it is cache hits. Doubling it costs 332 extra
  # calls on the teacher and doubles the per-class sample behind the student's
  # blind-spot table (n=6 to n=12).
  @("distillation",      "09_distill.py",              @("--items", "300")),
  @("cost / accuracy",   "07_cost_accuracy.py",        @()),
  @("latency",           "10_latency_bench.py",        @("--requests", "1200"))
)

$phase2 = @(
  @("verbosity bias",    "03_verbosity_bias.py",       @("--items", "100")),
  @("self-enhancement",  "04_self_enhancement.py",     @("--items", "100")),
  @("rubric ablation",   "05_rubric_ablation.py",      @("--items", "80", "--severities", "0.5")),
  @("self-consistency",  "06_self_consistency.py",     @("--items", "50", "--replicates", "5", "--severities", "0.5"))
)

$steps = switch ($Phase) {
  "1"   { $phase1 }
  "2"   { $phase2 }
  "all" { $phase1 + $phase2 }
}

$t0 = Get-Date
$i = 0
$quotaStopped = $false

foreach ($s in $steps) {
  $i++
  $name, $script, $extra = $s
  Write-Host ""
  Write-Host ("[{0}/{1}] {2}" -f $i, $steps.Count, $name) -ForegroundColor Green
  $started = Get-Date
  uv run python "experiments/$script" @extra
  $code = $LASTEXITCODE

  if ($code -eq $EXIT_QUOTA) {
    $quotaStopped = $true
    Write-Host ""
    Write-Host "  Daily allowance reached at '$name'. This is expected on a free tier." -ForegroundColor Yellow
    Write-Host "  Re-run this exact command tomorrow; everything already measured is cached." -ForegroundColor Yellow
    break
  }
  if ($code -ne 0) {
    Write-Host ""
    Write-Host "  !! $script failed for a reason that is NOT a quota." -ForegroundColor Red
    Write-Host "     Read the traceback above; completed work is cached and will not repeat." -ForegroundColor Red
    exit 1
  }
  Write-Host ("      done in {0:mm\:ss}" -f ((Get-Date) - $started)) -ForegroundColor DarkGray
}

Write-Host ""
uv run python scripts/quota_status.py

if ($quotaStopped) {
  Write-Host ""
  Write-Host ("Stopped on quota after {0:hh\:mm\:ss}. Run again tomorrow." -f ((Get-Date) - $t0)) -ForegroundColor Cyan
  exit 0
}

Write-Host ""
Write-Host "=== rebuilding the static demo ===" -ForegroundColor Cyan
uv run python scripts/export_web_model.py
uv run python scripts/build_site.py
uv run python experiments/make_figures.py

Write-Host ""
Write-Host "=== gate ===" -ForegroundColor Cyan
uv run python experiments/regression_suite.py --fail-under 0.70

Write-Host ""
Write-Host "=== which quoted numbers changed ===" -ForegroundColor Cyan
uv run python scripts/verify_claims.py
Write-Host ""
Write-Host "  Discrepancies above are EXPECTED while the docs still quote the simulated run." -ForegroundColor Yellow
Write-Host "  Send the output of the next command to update every document:" -ForegroundColor Yellow
Write-Host "     uv run python scripts/emit_numbers.py > live_numbers.txt" -ForegroundColor White
Write-Host ""
Write-Host ("Phase $Phase complete. Total elapsed: {0:hh\:mm\:ss}" -f ((Get-Date) - $t0)) -ForegroundColor Cyan
