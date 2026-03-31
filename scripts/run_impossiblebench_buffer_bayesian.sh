#!/usr/bin/env bash
set -euo pipefail

# Runs buffer and Bayesian baselines for ImpossibleBench case sets derived from
# the Claude Opus 4.6 source traces.
#
# Defaults mirror the tracked ImpossibleBench result files:
# - 50 cases per traces-per-case bucket
# - traces-per-case sweep: 10,25,50,100
# - seed 42

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-uv run --active python}"
SEED="${SEED:-42}"
CASES="${CASES:-50}"
TPC_LIST="${TPC_LIST:-10,25,50,100}"
BUFFER_SIZE="${BUFFER_SIZE:-30}"
SOURCE_MODEL="${SOURCE_MODEL:-claude-opus-4.6}"
DATA_PATH="${DATA_PATH:-data/safety/impossiblebench}"
POSITIVE_RATE="${POSITIVE_RATE:-0.5}"
DATASET_NAME="safety_impossiblebench_${SOURCE_MODEL}"

MODELS=(
  "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
  "openai/gpt-5.4-mini"
)

model_slug() {
  local model="$1"
  if [[ "$model" == */* ]]; then
    echo "${model##*/}"
  else
    echo "$model"
  fi
}

for model in "${MODELS[@]}"; do
  buffer_out="results/${DATASET_NAME}_buffer-$(model_slug "$model").jsonl"
  bayesian_out="results/${DATASET_NAME}_bayesian-$(model_slug "$model").jsonl"

  echo
  if [[ -f "$buffer_out" ]]; then
    echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: buffer exists, skipping (${buffer_out}) ==="
  else
    echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: buffer ==="
    eval "$PYTHON_BIN experiments/safety.py" \
      --safety-mode impossiblebench \
      --data-path "$DATA_PATH" \
      --impossiblebench-source-model "$SOURCE_MODEL" \
      --method buffer \
      --buffer-size "$BUFFER_SIZE" \
      --model "$model" \
      --cases "$CASES" \
      --traces-per-case-list "$TPC_LIST" \
      --positive-rate "$POSITIVE_RATE" \
      --seed "$SEED"
  fi

  echo
  if [[ -f "$bayesian_out" ]]; then
    echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: bayesian exists, skipping (${bayesian_out}) ==="
  else
    echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: bayesian ==="
    eval "$PYTHON_BIN experiments/safety.py" \
      --safety-mode impossiblebench \
      --data-path "$DATA_PATH" \
      --impossiblebench-source-model "$SOURCE_MODEL" \
      --method bayesian \
      --model "$model" \
      --cases "$CASES" \
      --traces-per-case-list "$TPC_LIST" \
      --positive-rate "$POSITIVE_RATE" \
      --seed "$SEED"
  fi
done
