param(
  [string]$BaseUrl = "http://10.103.5.28:8001/v1",
  [string]$ApiKey = "brachiokey",
  [string]$InspectModel = "vllm/Qwen/Qwen3.6-35B-A3B-FP8",
  [string]$RawModel = "Qwen/Qwen3.6-35B-A3B-FP8",
  [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$logPath = "logs\qwen_static_onecase_check.transcript.log"
New-Item -ItemType Directory -Path "logs" -Force | Out-Null
Set-Content -Path $logPath -Value "Started one-case check at $(Get-Date -Format o)"

function Write-Log {
  param([string]$Message)
  Write-Host $Message
  Add-Content -Path $logPath -Value $Message
}

function Test-VllmReachable {
  param(
    [string]$Url
  )

  $uri = [Uri]$Url
  $hostName = $uri.Host
  $port = if ($uri.Port -gt 0) { $uri.Port } else { 80 }

  Write-Log "Preflight TCP check: ${hostName}:${port}"
  $tcp = Test-NetConnection -ComputerName $hostName -Port $port -WarningAction SilentlyContinue
  Write-Log "  PingSucceeded=$($tcp.PingSucceeded) TcpTestSucceeded=$($tcp.TcpTestSucceeded)"

  if (-not $tcp.TcpTestSucceeded) {
    Write-Log ""
    Write-Log "ERROR: Cannot reach vLLM at ${hostName}:${port} from this machine."
    Write-Log "  - Confirm the vLLM server is running on the host."
    Write-Log "  - Confirm you are on the lab/VPN network that can route to $hostName."
    Write-Log "  - If the IP/port changed, rerun with: -BaseUrl http://HOST:PORT/v1"
    Write-Log "  - Quick manual test:"
    Write-Log "      Test-NetConnection $hostName -Port $port"
    Write-Log "      Invoke-RestMethod $Url/models -Headers @{Authorization='Bearer $ApiKey'}"
    throw "vLLM endpoint unreachable at ${hostName}:${port}"
  }
}

$env:VLLM_BASE_URL = $BaseUrl
$env:VLLM_API_KEY = $ApiKey

$output = "results\synthetic\smoke_kaggle_titanic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_examples10_disablethinking_onecase.jsonl"
Remove-Item $output -ErrorAction SilentlyContinue

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

  Write-Log "Checking VLLM chat/completions with thinking disabled: $BaseUrl"
  $health = Invoke-RestMethod `
    -Uri "$BaseUrl/chat/completions" `
    -Method Post `
    -Headers @{ Authorization = "Bearer $ApiKey" } `
    -ContentType "application/json" `
    -TimeoutSec 60 `
    -Body $body
  Write-Log "VLLM health check returned model: $($health.model)"
  Write-Log "Health check content: $($health.choices[0].message.content)"
} else {
  Write-Log "Skipping health check (-SkipHealthCheck). VLLM must still be reachable for the run."
}

Write-Log ""
Write-Log "Running one-case static VibeTest check"
Write-Log "Output: $output"

& ".venv\Scripts\python.exe" experiments\synthetic.py `
  --method vibetest `
  --labels-path synth-data\injected\labels_kaggle_titanic.jsonl `
  --datasets kaggle_titanic `
  --repo-limit 1 `
  --model $InspectModel `
  --correct-fail-examples 10 `
  --correct-fail-examples-path results\synthetic\manual_audit_representative_fails_10.tsv `
  --skip-scoring `
  --sandbox docker `
  --output-path $output

Write-Log ""
Write-Log "Finished one-case static check."
