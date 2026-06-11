param(
  [string]$Model = "openai/gpt-5-mini",
  [string]$EnvPath = "C:\Users\User\Downloads\vibetest-ml\vibetest\.env",
  [int]$MaxSamples = 4,
  [int]$MaxSandboxes = 4
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

New-Item -ItemType Directory -Path "logs" -Force | Out-Null
$logPath = "logs\real_kaggle_gpt5mini_static_prompt_examples.transcript.log"
Set-Content -Path $logPath -Value "Started real Kaggle GPT-5-mini static prompt examples at $(Get-Date -Format o)"

function Write-Log {
  param([string]$Message)
  Write-Host $Message
  Add-Content -Path $logPath -Value $Message
}

if (-not (Test-Path $EnvPath)) {
  throw "OpenAI env file not found: $EnvPath"
}

Get-Content $EnvPath | ForEach-Object {
  if ($_ -match '^\s*OPENAI_API_KEY\s*=\s*(.*)\s*$') {
    $env:OPENAI_API_KEY = $matches[1].Trim().Trim('"').Trim("'")
  }
}
if (-not $env:OPENAI_API_KEY) {
  throw "OPENAI_API_KEY not found in $EnvPath"
}

$datasets = @("titanic", "diabetic", "nlp")
$exampleCounts = @(0, 10, 20)

foreach ($examples in $exampleCounts) {
  foreach ($subset in $datasets) {
    $output = "results\kaggle_${subset}_AT-gpt-5-mini-static_examples${examples}.jsonl"
    Write-Log "Running subset=$subset examples=$examples output=$output"
    & ".venv\Scripts\python.exe" experiments\kaggle.py `
      --method vibetest `
      --static `
      --subset $subset `
      --model $Model `
      --correct-fail-examples $examples `
      --correct-fail-examples-path "results\synthetic\manual_audit_representative_fails_20.tsv" `
      --output-path $output `
      --max-samples $MaxSamples `
      --max-sandboxes $MaxSandboxes
    if ($LASTEXITCODE -ne 0) {
      throw "Run failed for subset=$subset examples=$examples with exit code $LASTEXITCODE"
    }
    Write-Log "Finished subset=$subset examples=$examples"
  }
}

Write-Log "Finished all real Kaggle GPT-5-mini static prompt-example runs at $(Get-Date -Format o)"
