param(
  [string]$BaseUrl = "http://10.103.5.28:8001/v1",
  [string]$ApiKey = "brachiokey",
  [string]$InspectModel = "vllm/Qwen/Qwen3.6-35B-A3B-FP8",
  [string]$RawModel = "Qwen/Qwen3.6-35B-A3B-FP8",
  [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$logPath = "logs\qwen_static_onecase_mixed_check.transcript.log"
New-Item -ItemType Directory -Path "logs" -Force | Out-Null
Set-Content -Path $logPath -Value "Started mixed one-case check at $(Get-Date -Format o)"

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

$output = "results\synthetic\smoke_kaggle_titanic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_mixed_examples10_disablethinking_onecase.jsonl"
Write-Log "Running one-case static VibeTest mixed calibration smoke."
Write-Log "Output: $output"

& ".venv\Scripts\python.exe" experiments\synthetic.py `
  --method vibetest `
  --labels-path "synth-data\injected\labels_kaggle_titanic.jsonl" `
  --datasets kaggle_titanic `
  --repo-limit 1 `
  --model $InspectModel `
  --mixed-fail-examples 10 `
  --mixed-fail-examples-path "results\synthetic\manual_audit_mixed_fail_calibration_10.tsv" `
  --skip-scoring `
  --sandbox docker `
  --output-path $output
if ($LASTEXITCODE -ne 0) {
  throw "Mixed one-case Python run failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $output)) {
  throw "Mixed one-case Python run exited successfully but did not create output: $output"
}

Write-Log "Finished mixed one-case check."
