#!/usr/bin/env bash
# Submit gradient prefix attack to Betty B200 cluster.
#
# Prerequisites:
#   1. SSH to login01.betty.parcc.upenn.edu
#   2. Create conda env: conda create -n gradient-attack python=3.11 pytorch torchvision -c pytorch -c nvidia
#   3. conda activate gradient-attack && pip install transformers sentence-transformers scikit-learn
#   4. Clone/sync this repo to the remote code root
#
# Usage:
#   bash scripts/submit_gradient_attack.sh [CASES] [STEPS] [PREFIX_LEN]

set -euo pipefail

CASES="${1:-50}"
STEPS="${2:-500}"
PREFIX_LEN="${3:-20}"

# Remote paths — update these for your setup
REMOTE_CODE_ROOT="${REMOTE_CODE_ROOT:-/vast/home/d/davisrbr/vibetest-adaptive-attacks}"
REMOTE_ARTIFACT_ROOT="${REMOTE_ARTIFACT_ROOT:-/vast/projects/exwong/brachiolab/davisrbr/vibetest-adaptive-attacks}"
CONDA_ENV="${CONDA_ENV:-gradient-attack}"

# Trace repo (must exist on remote)
TRACE_REPO="results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35_bayesian_boost"

RUN_NAME="gradient_prefix_attack_c${CASES}_s${STEPS}_p${PREFIX_LEN}"

echo "Submitting: ${RUN_NAME}"
echo "  Cases: ${CASES}, Steps: ${STEPS}, Prefix len: ${PREFIX_LEN}"
echo "  Code root: ${REMOTE_CODE_ROOT}"
echo "  Artifact root: ${REMOTE_ARTIFACT_ROOT}"

export PATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf

mkdir -p "${REMOTE_ARTIFACT_ROOT}/logs"

sbatch \
    --job-name="${RUN_NAME}" \
    --partition="dgx-b200" \
    --gres=gpu:1 \
    --time="6:00:00" \
    --account="exwong-brachiolab" \
    --mem="64G" \
    --output="${REMOTE_ARTIFACT_ROOT}/logs/${RUN_NAME}_%j.out" \
    --error="${REMOTE_ARTIFACT_ROOT}/logs/${RUN_NAME}_%j.err" \
    --wrap="cd ${REMOTE_CODE_ROOT} && \
        source activate ${CONDA_ENV} && \
        python experiments/gradient_prefix_attack.py \
            --trace-repo ${TRACE_REPO} \
            --cases ${CASES} \
            --output-dir ${REMOTE_ARTIFACT_ROOT}/gradient_prefixes \
            --model BAAI/bge-base-en-v1.5 \
            --steps ${STEPS} \
            --prefix-len ${PREFIX_LEN} \
            --topk 64 \
            --device cuda"

echo "Job submitted. Monitor with: squeue -u \$USER"
