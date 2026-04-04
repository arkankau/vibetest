#!/usr/bin/env bash
# Run distributed-misuse detection experiments with GPT-5.4 at varying reasoning_effort levels.
#
# Goal: test whether more inference compute (thinking) improves safety outcomes.
# Varies reasoning_effort={low,medium,high} with n=5 cases for plausibility.
#
# Usage: bash scripts/run_thinking_scaling.sh
set -euo pipefail
source env.sh

MODEL="openai/gpt-5.4-2026-03-05"
SANDBOX="docker"
DOMAIN="cyber"
DECOMP=6
BG=20
CASES=5
SEED=42

EFFORTS=("low" "medium" "high" "xhigh")

echo "============================================================"
echo "Thinking Scaling Experiments — Distributed Misuse"
echo "Model: ${MODEL}"
echo "Domain: ${DOMAIN}, decomp: ${DECOMP}, bg: ${BG}×, cases: ${CASES}"
echo "Reasoning efforts: ${EFFORTS[*]}"
echo "Started at $(date)"
echo "============================================================"

for EFFORT in "${EFFORTS[@]}"; do
    DS_NAME="safety_dm_${DOMAIN}_d${DECOMP}_bg${BG}_gpt54_effort_${EFFORT}"
    LOG="/tmp/thinking_scaling_${EFFORT}.log"

    echo ""
    echo "--- reasoning_effort=${EFFORT} ---"
    echo "  Dataset: ${DS_NAME}"
    echo "  Log: ${LOG}"

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

    echo "  Done. Log: ${LOG}"
    grep -E "Wrote.*rows|VibeTest summary" "${LOG}" || true
done

echo ""
echo "============================================================"
echo "All thinking scaling experiments complete at $(date)"
echo "============================================================"
echo ""
echo "Results files:"
ls -la results/safety_dm_*gpt54_effort_*.jsonl 2>/dev/null || echo "  (no results found)"
