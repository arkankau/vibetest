param(
  [string]$ScorerModel = "vllm/Qwen/Qwen3.6-35B-A3B-FP8",
  [string]$BaseUrl = "http://10.103.5.28:8001/v1",
  [string]$ApiKey = "brachiokey",
  [int]$ScorerConcurrency = 8
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$logPath = "logs\qwen_static_mixed_prompt_examples_score.transcript.log"
New-Item -ItemType Directory -Path "logs" -Force | Out-Null
Set-Content -Path $logPath -Value "Started mixed prompt-example scoring at $(Get-Date -Format o)"

function Write-Log {
  param([string]$Message)
  Write-Host $Message
  Add-Content -Path $logPath -Value $Message
}

function Safe-Model-Suffix {
  param([string]$Model)
  return ($Model -replace '^openai/', '' -replace '[^A-Za-z0-9_.-]', '-')
}

$scorerSuffix = Safe-Model-Suffix -Model $ScorerModel
$datasets = @("titanic", "diabetic", "nlp")

if ($ScorerModel -like "vllm/*") {
  $env:VLLM_BASE_URL = $BaseUrl
  $env:VLLM_API_KEY = $ApiKey
}

foreach ($examples in @(10, 20)) {
  foreach ($short in $datasets) {
    $input = "results\synthetic\synthetic_kaggle_${short}_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples${examples}.jsonl"
    $output = "results\synthetic\synthetic_kaggle_${short}_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples${examples}_scored-${scorerSuffix}.jsonl"
    if (-not (Test-Path $input)) {
      throw "Missing mixed result file to score: $input"
    }
    Write-Log ""
    Write-Log "Scoring mixed examples=$examples dataset=$short"
    Write-Log "Input: $input"
    Write-Log "Output: $output"

    & ".venv\Scripts\python.exe" experiments\synthetic.py `
      --rescore-results $input `
      --output-path $output `
      --scorer-model $ScorerModel `
      --scorer-concurrency $ScorerConcurrency
    if ($LASTEXITCODE -ne 0) {
      throw "Scoring failed with exit code $LASTEXITCODE for $input"
    }
    if (-not (Test-Path $output)) {
      throw "Scoring exited successfully but did not create output: $output"
    }
  }
}

Write-Log ""
Write-Log "Finished scoring static Qwen mixed prompt-example runs."
