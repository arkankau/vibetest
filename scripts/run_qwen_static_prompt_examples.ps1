param(
  [string]$BaseUrl = "http://10.103.42.105:8001/v1",
  [string]$ApiKey = "brachiokey",
  [string]$InspectModel = "vllm/Qwen/Qwen3.6-35B-A3B-FP8",
  [string]$RawModel = "Qwen/Qwen3.6-35B-A3B-FP8",
  [string]$OutputSuffix = "",
  [int[]]$ExamplesList = @(10, 20),
  [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$logPath = "logs\qwen_static_prompt_examples.transcript.log"
New-Item -ItemType Directory -Path "logs" -Force | Out-Null
Set-Content -Path $logPath -Value "Started prompt-example batch at $(Get-Date -Format o)"

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

$datasets = @(
  @{ Name = "kaggle_titanic"; Label = "synth-data\injected\labels_kaggle_titanic.jsonl"; Short = "titanic"; ExamplesPath = "results\synthetic\manual_audit_representative_fails_10.tsv" },
  @{ Name = "kaggle_diabetic"; Label = "synth-data\injected\labels_kaggle_diabetic.jsonl"; Short = "diabetic"; ExamplesPath = "results\synthetic\manual_audit_representative_fails_20.tsv" },
  @{ Name = "kaggle_nlp"; Label = "synth-data\injected\labels_kaggle_nlp.jsonl"; Short = "nlp"; ExamplesPath = "results\synthetic\manual_audit_representative_fails_20.tsv" }
)

foreach ($examples in $ExamplesList) {
  foreach ($dataset in $datasets) {
    $examplesPath = if ($examples -le 10) { "results\synthetic\manual_audit_representative_fails_10.tsv" } else { "results\synthetic\manual_audit_representative_fails_20.tsv" }
    $output = "results\synthetic\synthetic_kaggle_$($dataset.Short)_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_examples$examples$OutputSuffix.jsonl"
    Write-Log ""
    Write-Log "Running static VibeTest: $($dataset.Name), examples=$examples"
    Write-Log "Output: $output"

    & ".venv\Scripts\python.exe" experiments\synthetic.py `
      --method vibetest `
      --labels-path $dataset.Label `
      --datasets $dataset.Name `
      --model $InspectModel `
      --correct-fail-examples $examples `
      --correct-fail-examples-path $examplesPath `
      --skip-scoring `
      --sandbox docker `
      --output-path $output
  }
}

Write-Log ""
Write-Log "Finished static Qwen prompt-example runs."
