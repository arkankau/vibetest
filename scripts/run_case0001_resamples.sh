#!/usr/bin/env bash
# Resample case_0001 to reach N=10 per effort level.
# Already have: low=4, medium=6, high=2, xhigh=2
# Need:         low=6, medium=4, high=8, xhigh=8
set -euo pipefail
source env.sh

MODEL="openai/gpt-5.4-2026-03-05"
SANDBOX="docker"
DOMAIN="cyber"
DECOMP=6
BG=100
SEED=42

run_case() {
    local EFFORT=$1
    local RUN_ID=$2
    DS_NAME="safety_dm_${DOMAIN}_d${DECOMP}_bg${BG}_gpt54_case0001_${EFFORT}_r${RUN_ID}"
    LOG="/tmp/case0001_${EFFORT}_r${RUN_ID}.log"

    echo "  ${EFFORT} r${RUN_ID} ..."

    .venv/bin/python experiments/safety.py \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain "${DOMAIN}" \
        --distributed-misuse-decomp-level "${DECOMP}" \
        --distributed-misuse-background-multiplier "${BG}" \
        --cases 2 \
        --case-ids "1" \
        --method vibetest \
        --model "${MODEL}" \
        --sandbox "${SANDBOX}" \
        --seed "${SEED}" \
        --reasoning-effort "${EFFORT}" \
        --dataset-name "${DS_NAME}" \
        > "${LOG}" 2>&1 || true

    grep -E "Wrote.*rows|VibeTest summary" "${LOG}" || true
}

echo "============================================================"
echo "case_0001 Resampling — target N=10 per effort"
echo "Started at $(date)"
echo "============================================================"

echo ""
echo "--- Low (need 6 more) ---"
for R in $(seq 1 6); do run_case low "$R"; done

echo ""
echo "--- Medium (need 4 more) ---"
for R in $(seq 1 4); do run_case medium "$R"; done

echo ""
echo "--- High (need 8 more) ---"
for R in $(seq 1 8); do run_case high "$R"; done

echo ""
echo "--- XHigh (need 8 more) ---"
for R in $(seq 1 8); do run_case xhigh "$R"; done

echo ""
echo "============================================================"
echo "All case_0001 resamples complete at $(date)"
echo "============================================================"
