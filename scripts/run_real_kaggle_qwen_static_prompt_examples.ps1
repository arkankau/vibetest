param(
  [string]$BaseUrl = "http://10.103.42.105:8001/v1",
  [string]$ApiKey = "brachiokey",
  [string]$InspectModel = "vllm/Qwen/Qwen3.6-35B-A3B-FP8",
  [string]$RawModel = "Qwen/Qwen3.6-35B-A3B-FP8",
  [int]$MaxSamples = 4,
  [int]$MaxSandboxes = 4,
  [int]$RepoLimit = 50,
  [int]$RepoOffset = 0,
  [int[]]$ExamplesList = @(0, 10, 20),
  [string[]]$Datasets = @("titanic", "diabetic", "nlp"),
  [string]$OutputSuffix = "",
  [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

New-Item -ItemType Directory -Path "logs" -Force | Out-Null
$logPath = "logs\real_kaggle_qwen_static_prompt_examples.transcript.log"
Set-Content -Path $logPath -Value "Started real Kaggle Qwen static prompt-example runs at $(Get-Date -Format o)"

function Write-Log {
  param([string]$Message)
  Write-Host $Message
  Add-Content -Path $logPath -Value $Message
}

function Test-VllmReachable {
  param([string]$Url)
  $uri = [Uri]$Url
  $hostName = $uri.Host
  $port = if ($uri.Port -gt 0) { $uri.Port } else { 80 }
  Write-Log "Preflight TCP check: ${hostName}:${port}"
  $tcp = Test-NetConnection -ComputerName $hostName -Port $port -WarningAction SilentlyContinue
  if (-not $tcp.TcpTestSucceeded) {
    throw "vLLM endpoint unreachable at ${hostName}:${port}"
  }
}

$env:VLLM_BASE_URL = $BaseUrl
$env:VLLM_API_KEY = $ApiKey

if (-not $SkipHealthCheck) {
  Test-VllmReachable -Url $BaseUrl
  $body = @"
{
  "model": "$RawModel",
  "messages": [
    {
      "role": "user",
      "content": "/no_think\nReply OK only."
    }
  ],
  "temperature": 0.0,
  "max_tokens": 32,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
"@
  Write-Log "Checking VLLM chat/completions: $BaseUrl"
  $health = Invoke-RestMethod `
    -Uri "$BaseUrl/chat/completions" `
    -Method Post `
    -Headers @{ Authorization = "Bearer $ApiKey" } `
    -ContentType "application/json" `
    -TimeoutSec 60 `
    -Body $body
  Write-Log "VLLM health check returned model: $($health.model)"
}

foreach ($examples in $ExamplesList) {
  foreach ($subset in $Datasets) {
    $examplesPath = if ($examples -le 10) { "results\synthetic\manual_audit_representative_fails_10.tsv" } else { "results\synthetic\manual_audit_representative_fails_20.tsv" }
    $output = "results\kaggle_${subset}_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_examples${examples}${OutputSuffix}.jsonl"
    Write-Log ""
    Write-Log "Running real Kaggle static VibeTest: subset=$subset examples=$examples output=$output"

    & ".venv\Scripts\python.exe" experiments\kaggle.py `
      --method vibetest `
      --static `
      --subset $subset `
      --model $InspectModel `
      --correct-fail-examples $examples `
      --correct-fail-examples-path $examplesPath `
      --output-path $output `
      --repo-limit $RepoLimit `
      --repo-offset $RepoOffset `
      --max-samples $MaxSamples `
      --max-sandboxes $MaxSandboxes
    if ($LASTEXITCODE -ne 0) {
      throw "Run failed for subset=$subset examples=$examples with exit code $LASTEXITCODE"
    }
    Write-Log "Finished subset=$subset examples=$examples"
  }
}

Write-Log ""
Write-Log "Finished all real Kaggle Qwen static prompt-example runs at $(Get-Date -Format o)"
