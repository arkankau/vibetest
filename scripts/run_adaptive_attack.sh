#!/usr/bin/env bash
# Run adaptive attack experiments against the distributed-misuse agent defense.
#
# Usage:
#   bash scripts/run_adaptive_attack.sh [INJECTION_NAME] [NUM_CASES]
#
# Examples:
#   bash scripts/run_adaptive_attack.sh score_format 5
#   bash scripts/run_adaptive_attack.sh embedding_poison 5
#   bash scripts/run_adaptive_attack.sh combined 10
#   bash scripts/run_adaptive_attack.sh all 5          # run all strategies
set -euo pipefail

# Resolve the main repo root (works from worktree or main checkout)
MAIN_REPO="${MAIN_REPO:-/Users/davisbrown/vibetest}"
cd "${MAIN_REPO}"
source env.sh 2>/dev/null || true

INJECTION="${1:-score_format}"
CASES="${2:-5}"
MODEL="together/Qwen/Qwen3.5-397B-A17B-FP8"
SANDBOX="docker"
SOURCE_REPO="results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35_2pass"
CLUSTERING="coherence"
BASELINE_RESULT="results/dm_cyber_d6_bg100_qwen35_2pass_n10.jsonl"
VENV=".venv/bin/python"
# Use adaptive_attacks.py from the worktree
ATTACKS_SCRIPT="${MAIN_REPO}/.claude/worktrees/adaptive-attacks/experiments/adaptive_attacks.py"

STRATEGIES=(score_format embedding_poison topic_label_poison agent_instruction combined trace_scores_hijack)

run_one() {
    local inj="$1"
    local cases="$2"
    local dest="results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35_2pass_${inj}"
    local ds_name="safety_dm_cyber_d6_bg100_qwen35_2pass_${inj}"
    local log="/tmp/adaptive_attack_${inj}.log"

    echo ""
    echo "================================================================"
    echo "  Strategy: ${inj} | Cases: ${cases}"
    echo "================================================================"

    # Step 1: Create injected trace repo
    echo "--- Creating injected repo ---"
    ${VENV} "${ATTACKS_SCRIPT}" \
        --source-repo "${SOURCE_REPO}" \
        --injection "${inj}" \
        --dest-repo "${dest}" \
        --cases "${cases}"

    # Step 2: Run pipeline (scoring + clustering + agent)
    echo ""
    echo "--- Running pipeline ---"
    ${VENV} experiments/safety.py \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain cyber \
        --distributed-misuse-decomp-level 6 \
        --distributed-misuse-background-multiplier 100 \
        --distributed-misuse-clustering-strategy "${CLUSTERING}" \
        --cases "${cases}" \
        --method vibetest \
        --model "${MODEL}" \
        --sandbox "${SANDBOX}" \
        --seed 42 \
        --dataset-name "${ds_name}" \
        2>&1 | tee "${log}"

    echo ""
    echo "--- Results ---"
    # Find the output JSONL (vibetest writes to results/ by default)
    local result_file
    result_file=$(ls -t results/safety_dm_cyber_d6_bg100_qwen35_2pass_${inj}_AT-*.jsonl 2>/dev/null | head -1)
    if [ -n "${result_file}" ]; then
        ${VENV} "${ATTACKS_SCRIPT}" \
            --source-repo "${SOURCE_REPO}" \
            --injection "${inj}" \
            --analyze "${result_file}" \
            --compare-baseline "${BASELINE_RESULT}"
    else
        echo "No result JSONL found for ${inj}. Check ${log}."
    fi
}

if [ "${INJECTION}" = "all" ]; then
    for s in "${STRATEGIES[@]}"; do
        run_one "${s}" "${CASES}"
    done
else
    run_one "${INJECTION}" "${CASES}"
fi

echo ""
echo "Done at $(date)"
