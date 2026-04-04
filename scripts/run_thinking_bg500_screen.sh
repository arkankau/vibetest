#!/usr/bin/env bash
# Screening run: GPT-5.4 at low effort on n=10 cases, bg=500×.
# Goal: verify that bg=500× makes cases hard enough for GPT-5.4,
# then run full effort scaling on the same cases.
#
# Usage: bash scripts/run_thinking_bg500_screen.sh
set -euo pipefail
source env.sh

MODEL="openai/gpt-5.4-2026-03-05"
SANDBOX="docker"
DOMAIN="cyber"
DECOMP=6
BG=500
CASES=10
SEED=42

EFFORTS=("low" "medium" "high" "xhigh")

echo "============================================================"
echo "bg=500× Effort Scaling — Distributed Misuse"
echo "Model: ${MODEL}"
echo "Domain: ${DOMAIN}, decomp: ${DECOMP}, bg: ${BG}×, cases: ${CASES}"
echo "Efforts: ${EFFORTS[*]}"
echo "Started at $(date)"
echo "============================================================"

for EFFORT in "${EFFORTS[@]}"; do
    DS_NAME="safety_dm_${DOMAIN}_d${DECOMP}_bg${BG}_gpt54_effort_${EFFORT}"
    LOG="/tmp/thinking_bg500_${EFFORT}.log"

    echo ""
    echo "--- reasoning_effort=${EFFORT} ---"
    echo "  Dataset: ${DS_NAME}"

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

    echo "  Done at $(date). Log: ${LOG}"
    grep -E "Wrote.*rows|VibeTest summary" "${LOG}" || true
done

echo ""
echo "============================================================"
echo "All bg=500× experiments complete at $(date)"
echo "============================================================"
