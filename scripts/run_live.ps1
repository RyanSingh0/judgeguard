<#
.SYNOPSIS
    Run the full bias battery against REAL models and regenerate everything.

.DESCRIPTION
    Sample sizes are tuned so this finishes in an afternoon rather than a week,
    while keeping ~80% power to detect a 7-point accuracy difference.

    The binding constraint is qwen3.6-27b: it is a reasoning model and spends
    ~1,200 output tokens per judgement, about 4.7 s per call. Concurrency is
    raised to 8 to compensate. Experiments 05 and 06 multiply the call count by
    3 and 5 respectively, so they run on smaller item counts -- stated in the
    results JSON, not hidden.

    Every call is cached on (provider, model, prompt, params, replicate), so a
    crashed or interrupted run RESUMES FOR FREE. Re-running this script after a
    failure costs nothing for the work already done. Do not delete .cache/.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_live.ps1
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:JUDGEGUARD_PROVIDER_MODE = "live"
$env:JUDGEGUARD_MAX_CONCURRENCY = "8"

Write-Host ""
Write-Host "=== JudgeGuard LIVE battery ===" -ForegroundColor Cyan
Write-Host "  Results in results/ will be overwritten with real measurements."
Write-Host "  Interrupted runs resume for free (the cache is keyed on the call)."
Write-Host ""

uv run judgeguard status
if ($LASTEXITCODE -ne 0) { throw "judgeguard status failed" }

Write-Host ""
Write-Host "  Every panel row above must say 'live API'. Ctrl+C now if any says 'simulator'." -ForegroundColor Yellow
Start-Sleep -Seconds 6

# name, script, args, rough share of total runtime
$steps = @(
  @("dataset",            "00_build_dataset.py",        @("--items","150")),
  @("discrimination",     "01_discrimination.py",       @("--items","150")),
  @("position bias",      "02_position_bias.py",        @("--items","100")),
  @("verbosity bias",     "03_verbosity_bias.py",       @("--items","150")),
  @("self-enhancement",   "04_self_enhancement.py",     @("--items","100")),
  @("rubric ablation",    "05_rubric_ablation.py",      @("--items","80")),
  @("self-consistency",   "06_self_consistency.py",     @("--items","50","--replicates","5")),
  @("cost / accuracy",    "07_cost_accuracy.py",        @()),
  @("trajectory blind.",  "08_trajectory_blindness.py", @("--trajectories","60")),
  @("distillation",       "09_distill.py",              @("--items","150")),
  @("latency",            "10_latency_bench.py",        @("--requests","1200")),
  @("figures",            "make_figures.py",            @())
)

$t0 = Get-Date
$i = 0
foreach ($s in $steps) {
  $i++
  $name, $script, $extra = $s
  Write-Host ""
  Write-Host ("[{0}/{1}] {2}" -f $i, $steps.Count, $name) -ForegroundColor Green
  $started = Get-Date
  uv run python "experiments/$script" @extra
  if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  !! $script failed. Fix the cause and re-run this script;" -ForegroundColor Red
    Write-Host "     completed work is cached and will not be repeated." -ForegroundColor Red
    exit 1
  }
  Write-Host ("      done in {0:mm\:ss}" -f ((Get-Date) - $started)) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "=== rebuilding the static demo ===" -ForegroundColor Cyan
uv run python scripts/export_web_model.py
uv run python scripts/build_site.py

Write-Host ""
Write-Host "=== gate ===" -ForegroundColor Cyan
uv run python experiments/regression_suite.py --fail-under 0.70

Write-Host ""
Write-Host "=== which quoted numbers changed ===" -ForegroundColor Cyan
uv run python scripts/verify_claims.py
Write-Host ""
Write-Host "  Discrepancies above are EXPECTED: the docs still quote the simulated run." -ForegroundColor Yellow
Write-Host "  Send the output of the next command to update every document:" -ForegroundColor Yellow
Write-Host "     uv run python scripts/emit_numbers.py > live_numbers.txt" -ForegroundColor White
Write-Host ""
Write-Host ("Total elapsed: {0:hh\:mm\:ss}" -f ((Get-Date) - $t0)) -ForegroundColor Cyan
