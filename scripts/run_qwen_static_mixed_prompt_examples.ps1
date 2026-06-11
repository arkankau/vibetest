param(
  [string]$BaseUrl = "http://10.103.5.28:8001/v1",
  [string]$ApiKey = "brachiokey",
  [string]$InspectModel = "vllm/Qwen/Qwen3.6-35B-A3B-FP8",
  [string]$RawModel = "Qwen/Qwen3.6-35B-A3B-FP8",
  [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$logPath = "logs\qwen_static_mixed_prompt_examples.transcript.log"
New-Item -ItemType Directory -Path "logs" -Force | Out-Null
Set-Content -Path $logPath -Value "Started mixed prompt-example batch at $(Get-Date -Format o)"

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
  Write-Log "VLLM health check returned model: $($health.model), finish_reason: $($health.choices[0].finish_reason)"
}

$datasets = @(
  @{ Name = "kaggle_titanic"; Label = "synth-data\injected\labels_kaggle_titanic.jsonl"; Short = "titanic" },
  @{ Name = "kaggle_diabetic"; Label = "synth-data\injected\labels_kaggle_diabetic.jsonl"; Short = "diabetic" },
  @{ Name = "kaggle_nlp"; Label = "synth-data\injected\labels_kaggle_nlp.jsonl"; Short = "nlp" }
)

foreach ($examples in @(10, 20)) {
  $examplesPath = "results\synthetic\manual_audit_mixed_fail_calibration_$examples.tsv"
  foreach ($dataset in $datasets) {
    $output = "results\synthetic\synthetic_kaggle_$($dataset.Short)_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples$examples.jsonl"
    Write-Log ""
    Write-Log "Running static VibeTest mixed calibration: $($dataset.Name), examples=$examples"
    Write-Log "Output: $output"

    & ".venv\Scripts\python.exe" experiments\synthetic.py `
      --method vibetest `
      --labels-path $dataset.Label `
      --datasets $dataset.Name `
      --model $InspectModel `
      --mixed-fail-examples $examples `
      --mixed-fail-examples-path $examplesPath `
      --skip-scoring `
      --sandbox docker `
      --output-path $output
    if ($LASTEXITCODE -ne 0) {
      throw "Mixed static run failed with exit code $LASTEXITCODE for $($dataset.Name), examples=$examples"
    }
    if (-not (Test-Path $output)) {
      throw "Mixed static run exited successfully but did not create output: $output"
    }
  }
}

Write-Log ""
Write-Log "Finished static Qwen mixed prompt-example runs."
