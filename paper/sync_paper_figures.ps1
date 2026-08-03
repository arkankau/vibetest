# Copy verifier figures into paper/figures/ for a self-contained LaTeX build.
# Run from repo root:  powershell -File paper/sync_paper_figures.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $PSScriptRoot "figures"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$maps = @{
    "vibetest_evidence_verification.png" = @(
        (Join-Path $PSScriptRoot "figures/vibetest_evidence_verification.png")
    )
    "synthetic_verifier_pr_curve_combined.png" = @(
        (Join-Path $root "results/synthetic/figures/continuous/verifier_pr_curve_combined.png"),
        (Join-Path $root "results/synthetic/figures/verifier_pr_curve_combined.png")
    )
    "synthetic_verifier_f1_vs_threshold.png" = @(
        (Join-Path $root "results/synthetic/figures/continuous/verifier_f1_vs_threshold.png"),
        (Join-Path $root "results/synthetic/figures/verifier_f1_vs_threshold.png")
    )
    "real_verifier_pr_curve_combined.png" = @(
        (Join-Path $root "results/kaggle/figures/continuous/verifier_pr_curve_combined.png"),
        (Join-Path $root "results/kaggle/figures/verifier_pr_curve_combined.png")
    )
    "real_verifier_f1_vs_threshold.png" = @(
        (Join-Path $root "results/kaggle/figures/continuous/verifier_f1_vs_threshold.png"),
        (Join-Path $root "results/kaggle/figures/verifier_f1_vs_threshold.png")
    )
}

foreach ($name in $maps.Keys) {
    $target = Join-Path $dest $name
    $copied = $false
    foreach ($src in $maps[$name]) {
        if (Test-Path $src) {
            Copy-Item -Force $src $target
            Write-Host "Copied $src -> $target"
            $copied = $true
            break
        }
    }
    if (-not $copied) {
        Write-Warning "Missing source for $name"
    }
}
