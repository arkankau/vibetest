#!/usr/bin/env bash
# Resampling experiment: run the SAME 5 cases K times at each effort level.
# This isolates inference stochasticity from case difficulty variance,
# enabling paired per-case comparisons across effort levels.
#
# Cases are materialized once (seed=42), then the agent is run K times.
# Each resample gets a unique dataset name suffix so results don't overwrite.
#
# Usage: bash scripts/run_thinking_resampling.sh
set -euo pipefail
source env.sh

MODEL="openai/gpt-5.4-2026-03-05"
SANDBOX="docker"
DOMAIN="cyber"
DECOMP=6
BG=100
CASES=5
BASE_SEED=42
K=3  # Number of resamples per effort level

EFFORTS=("low" "medium" "high" "xhigh")

echo "============================================================"
echo "Thinking Resampling Experiment — Distributed Misuse"
echo "Model: ${MODEL}"
echo "Domain: ${DOMAIN}, decomp: ${DECOMP}, bg: ${BG}×, cases: ${CASES}"
echo "Efforts: ${EFFORTS[*]}, resamples per effort: ${K}"
echo "Started at $(date)"
echo "============================================================"

for EFFORT in "${EFFORTS[@]}"; do
    for ((R=1; R<=K; R++)); do
        DS_NAME="safety_dm_${DOMAIN}_d${DECOMP}_bg${BG}_gpt54_resample_${EFFORT}_r${R}"
        LOG="/tmp/resample_${EFFORT}_r${R}.log"

        echo ""
        echo "--- effort=${EFFORT}, resample=${R}/${K} ---"
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
            --seed "${BASE_SEED}" \
            --reasoning-effort "${EFFORT}" \
            --dataset-name "${DS_NAME}" \
            > "${LOG}" 2>&1 || true

        echo "  Done. Log: ${LOG}"
        grep -E "Wrote.*rows|VibeTest summary" "${LOG}" || true
    done
done

echo ""
echo "============================================================"
echo "All resampling experiments complete at $(date)"
echo "============================================================"
