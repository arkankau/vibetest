param(
  [string]$Main = "vibetest_colm_style"
)

$ErrorActionPreference = "Stop"

$requiredFiles = @("acl.sty", "acl_natbib.bst", "$Main.tex", "vibetest.bib")
$missing = @()
foreach ($file in $requiredFiles) {
  if (-not (Test-Path $file)) {
    $missing += $file
  }
}

if ($missing.Count -gt 0) {
  Write-Host "Missing required LaTeX/package files:"
  foreach ($file in $missing) {
    Write-Host "  - $file"
  }
  Write-Host ""
  Write-Host "Add the official ACL style files to this folder, then rerun:"
  Write-Host "  powershell -ExecutionPolicy Bypass -File .\compile_check.ps1"
  exit 1
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
  Write-Host "pdflatex is not available on PATH."
  Write-Host "Install a TeX distribution or run this package on a machine with pdflatex."
  exit 1
}

if (-not (Get-Command bibtex -ErrorAction SilentlyContinue)) {
  Write-Host "bibtex is not available on PATH."
  Write-Host "Install a TeX distribution or run this package on a machine with bibtex."
  exit 1
}

pdflatex -interaction=nonstopmode "$Main.tex"
bibtex $Main
pdflatex -interaction=nonstopmode "$Main.tex"
pdflatex -interaction=nonstopmode "$Main.tex"

Write-Host "Compile complete: $Main.pdf"
