#!/usr/bin/env bash
# Screening run: GPT-5.4 at low effort on n=50 cases (bg=100×, cyber).
# Goal: identify which cases are genuinely hard for GPT-5.4 so we can
# run the full effort-scaling experiment on just those.
#
# Usage: bash scripts/run_thinking_screen.sh
set -euo pipefail
source env.sh

MODEL="openai/gpt-5.4-2026-03-05"
SANDBOX="docker"
DOMAIN="cyber"
DECOMP=6
BG=100
CASES=50
SEED=42
EFFORT="low"

DS_NAME="safety_dm_${DOMAIN}_d${DECOMP}_bg${BG}_gpt54_screen50"
LOG="/tmp/thinking_screen.log"

echo "============================================================"
echo "GPT-5.4 Screening Run — Distributed Misuse"
echo "Model: ${MODEL}, effort: ${EFFORT}"
echo "Domain: ${DOMAIN}, decomp: ${DECOMP}, bg: ${BG}×, cases: ${CASES}"
echo "Dataset: ${DS_NAME}"
echo "Started at $(date)"
echo "============================================================"

.venv/bin/python experiments/safety.py \
    --safety-mode distributed-misuse \
    --distributed-misuse-domain "${DOMAIN}" \
    --distributed-misuse-decomp-level "${DECOMP}" \
    --distributed-misuse-background-multiplier "${BG}" \
    --cases "${CASES}" \
    --method vibetest \
    --model "${MODEL}" \
    --sandbox "${SANDBOX}" \
    --seed "${SEED}" \
    --reasoning-effort "${EFFORT}" \
    --dataset-name "${DS_NAME}" \
    > "${LOG}" 2>&1 || true

echo ""
echo "Done at $(date). Log: ${LOG}"
grep -E "Wrote.*rows|VibeTest summary" "${LOG}" || true

echo ""
echo "Results:"
ls -la results/safety_dm_*gpt54_screen50*.jsonl 2>/dev/null || echo "  (no results found)"
