#!/bin/bash
# Betty note: keep all environments and caches off home; use node-local /tmp because
# personal home storage and inode quotas are tight.
# The Nebius rebench parquet is large, so do dataset caching and dependency installs
# entirely under /tmp and only persist the compact review bundle on project storage.
#SBATCH --job-name=nbrebt46
#SBATCH --partition=genoa-std-mem
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/vast/projects/exwong/brachiolab/davisrbr/vibetest_nebius_rebench_gitpeek_opus46_20260401a/slurm-%j.out

set -euo pipefail

set -a
source /vast/projects/exwong/brachiolab/davisrbr/vibetest-pipeline/.env
set +a

export TMPDIR=/tmp
export PIP_CACHE_DIR=/tmp/pip-cache-$USER
export HF_HOME=/tmp/hf-$USER
export HF_DATASETS_CACHE=/tmp/hf-$USER/datasets
export HUGGINGFACE_HUB_CACHE=/tmp/hf-$USER/hub
export XDG_CACHE_HOME=/tmp/xdg-$USER
export PYDEPS=/tmp/vibetest_pydeps_$SLURM_JOB_ID

mkdir -p "$PIP_CACHE_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$PYDEPS"

python3 -m pip install --break-system-packages --target "$PYDEPS" inspect_ai jsonlines datasets pyarrow anthropic
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd /vast/projects/exwong/brachiolab/davisrbr/vibetest_nebius_rebench_gitpeek_opus46_20260401a

python3 prepare_nebius_rebench_gitpeek_cases.py \
  --case-dir case \
  --selection-path results/nebius_rebench_gitpeek_selection.jsonl \
  --max-rows 5000 \
  --max-cases 48 \
  --max-per-repo 4 \
  --min-score 4 \
  --no-streaming

python3 monitor_nebius_rebench_gitpeek.py \
  --case-dir case \
  --review-path results/nebius_rebench_gitpeek_review.jsonl \
  --model anthropic/claude-opus-4-6 \
  --concurrency 8
