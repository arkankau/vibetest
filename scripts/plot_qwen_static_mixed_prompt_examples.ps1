param(
  [string]$OutputPrefix = "results\synthetic\figures\synthetic_kaggle_qwen36_mixed_examples_scored_vllm_reviewer_inconclusive_as_pass_evidence_match_two_sided_t1_0p5",
  [double]$InconclusiveBandLow = 0.5
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$static0 = @(
  "results\synthetic\synthetic_kaggle_titanic_AT-Qwen-Qwen3.6-35B-A3B-FP8_case-score.jsonl",
  "results\synthetic\synthetic_kaggle_diabetic_AT-Qwen-Qwen3.6-35B-A3B-FP8_case-score.jsonl",
  "results\synthetic\synthetic_kaggle_nlp_AT-Qwen-Qwen3.6-35B-A3B-FP8_case-score.jsonl"
)

$dynamic = @(
  "results\synthetic\synthetic_kaggle_titanic_AT-Qwen-Qwen3.6-35B-A3B-FP8-dynamic_case-score.jsonl",
  "results\synthetic\synthetic_kaggle_diabetic_AT-Qwen-Qwen3.6-35B-A3B-FP8-dynamic_case-score.jsonl",
  "results\synthetic\synthetic_kaggle_nlp_AT-Qwen-Qwen3.6-35B-A3B-FP8-dynamic_case-score.jsonl"
)

$mixed10 = @(
  "results\synthetic\synthetic_kaggle_titanic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples10_scored-vllm-Qwen-Qwen3.6-35B-A3B-FP8.jsonl",
  "results\synthetic\synthetic_kaggle_diabetic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples10_scored-vllm-Qwen-Qwen3.6-35B-A3B-FP8.jsonl",
  "results\synthetic\synthetic_kaggle_nlp_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples10_scored-vllm-Qwen-Qwen3.6-35B-A3B-FP8.jsonl"
)

$mixed20 = @(
  "results\synthetic\synthetic_kaggle_titanic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples20_scored-vllm-Qwen-Qwen3.6-35B-A3B-FP8.jsonl",
  "results\synthetic\synthetic_kaggle_diabetic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples20_scored-vllm-Qwen-Qwen3.6-35B-A3B-FP8.jsonl",
  "results\synthetic\synthetic_kaggle_nlp_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples20_scored-vllm-Qwen-Qwen3.6-35B-A3B-FP8.jsonl"
)

$reviewerMode0 = @(
  "results\synthetic\synthetic_kaggle_titanic_baseline-reviewer-mode0-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl",
  "results\synthetic\synthetic_kaggle_diabetic_baseline-reviewer-mode0-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl",
  "results\synthetic\synthetic_kaggle_nlp_baseline-reviewer-mode0-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl"
)

$reviewerMode1 = @(
  "results\synthetic\synthetic_kaggle_titanic_baseline-reviewer-mode1-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl",
  "results\synthetic\synthetic_kaggle_diabetic_baseline-reviewer-mode1-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl",
  "results\synthetic\synthetic_kaggle_nlp_baseline-reviewer-mode1-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl"
)

$reviewerMode2 = @(
  "results\synthetic\synthetic_kaggle_titanic_baseline-reviewer-mode2-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl",
  "results\synthetic\synthetic_kaggle_diabetic_baseline-reviewer-mode2-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl",
  "results\synthetic\synthetic_kaggle_nlp_baseline-reviewer-mode2-static-Qwen-Qwen3.6-35B-A3B-FP8_mapper-gpt-5.4-mini.jsonl"
)

$allInputs = @($static0 + $dynamic + $mixed10 + $mixed20 + $reviewerMode0 + $reviewerMode1 + $reviewerMode2)
foreach ($path in $allInputs) {
  if (-not (Test-Path $path)) {
    throw "Missing result file needed for Adam-style mixed plot: $path"
  }
}

& ".venv\Scripts\python.exe" scripts\plot_synthetic_threshold_tradeoffs.py `
  --output-prefix $OutputPrefix `
  --target evidence-match `
  --inconclusive-band-low $InconclusiveBandLow `
  --series "VibeTest static (0 examples)=$($static0 -join ',')" `
  --series "VibeTest static + mixed ex10=$($mixed10 -join ',')" `
  --series "VibeTest static + mixed ex20=$($mixed20 -join ',')" `
  --series "VibeTest dynamic=$($dynamic -join ',')" `
  --series "Reviewer mode 0=$($reviewerMode0 -join ',')" `
  --series "Reviewer mode 1=$($reviewerMode1 -join ',')" `
  --series "Reviewer mode 2=$($reviewerMode2 -join ',')" `
  --inconclusive-as-pass-series "Reviewer mode 0" `
  --inconclusive-as-pass-series "Reviewer mode 1" `
  --inconclusive-as-pass-series "Reviewer mode 2"

Write-Host ""
Write-Host "Adam-style comparison plot:"
Write-Host "$($OutputPrefix)_covered_macro_f1_vs_coverage.png"
