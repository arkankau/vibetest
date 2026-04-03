#!/bin/bash
#SBATCH --job-name=tb2devaudit
#SBATCH --partition=genoa-std-mem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/vast/projects/exwong/brachiolab/davisrbr/vibetest_external_audit_runs/tb2_hard_devcheat_20260401a/slurm-%j.out

set -euo pipefail

source /vast/projects/exwong/brachiolab/davisrbr/.anthropic_vibetest.env

export TMPDIR=/tmp
export PIP_CACHE_DIR=/tmp/pip-cache-$USER
export HF_HOME=/tmp/hf-$USER
export HF_DATASETS_CACHE=/tmp/hf-$USER/datasets
export HUGGINGFACE_HUB_CACHE=/tmp/hf-$USER/hub
export XDG_CACHE_HOME=/tmp/xdg-$USER
export PYDEPS=/tmp/vibetest_pydeps_$SLURM_JOB_ID

mkdir -p "$PIP_CACHE_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$PYDEPS"

python3 -m pip install --break-system-packages --target "$PYDEPS" anthropic datasets
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"

cd /vast/projects/exwong/brachiolab/davisrbr/vibetest_external_audit_runs/tb2_hard_devcheat_20260401a

python3 monitor_external_trace_candidates.py \
  --dataset terminalbench-trajectories \
  --output-jsonl results.jsonl \
  --top-artifacts-dir top_traces \
  --top-k 15 \
  --limit 274 \
  --monitor-model claude-opus-4-6 \
  --max-input-chars 70000 \
  --max-output-tokens 700 \
  --success-only \
  --require-steps \
  --hard-tasks-only \
  --agent-substring terminus-3-3 \
  --agent-substring deepagent-harbor \
  --agent-substring judy \
  --agent-substring codex \
  --model-substring gemini-3.1-pro-preview \
  --model-substring claude-opus-4-6 \
  --model-substring claude-opus-4.6 \
  --model-substring gpt-5@openai \
  --model-substring gpt-5.2-codex@openai

rm -f /vast/projects/exwong/brachiolab/davisrbr/.anthropic_vibetest.env
