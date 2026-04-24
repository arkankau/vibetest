#!/usr/bin/env bash
set -euo pipefail

# Runs judge, Meerkat, buffer, Bayesian, and no-tools VibeTest baselines for
# ImpossibleBench case sets derived from the Claude Opus 4.6 source traces.
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
MAX_POSITIVE_TRACES_PER_CASE="${MAX_POSITIVE_TRACES_PER_CASE:-}"
MAX_POSITIVE_TRACES_PERCENT_PER_CASE="${MAX_POSITIVE_TRACES_PERCENT_PER_CASE:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
REMATERIALIZE="${REMATERIALIZE:-0}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_MEERKAT="${RUN_MEERKAT:-1}"
RUN_BUFFER="${RUN_BUFFER:-1}"
RUN_BAYESIAN="${RUN_BAYESIAN:-1}"
RUN_NAIVE="${RUN_NAIVE:-1}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-}"
JUDGE_DISABLE_THINKING="${JUDGE_DISABLE_THINKING:-}"
DATASET_NAME="safety_impossiblebench_${SOURCE_MODEL}"

if [[ -n "$MAX_POSITIVE_TRACES_PER_CASE" && -n "$MAX_POSITIVE_TRACES_PERCENT_PER_CASE" ]]; then
  echo "Set only one of MAX_POSITIVE_TRACES_PER_CASE and MAX_POSITIVE_TRACES_PERCENT_PER_CASE." >&2
  exit 1
fi

extra_positive_args=()
if [[ -n "$MAX_POSITIVE_TRACES_PER_CASE" ]]; then
  extra_positive_args+=(--max-positive-traces-per-case "$MAX_POSITIVE_TRACES_PER_CASE")
fi
if [[ -n "$MAX_POSITIVE_TRACES_PERCENT_PER_CASE" ]]; then
  extra_positive_args+=(--max-positive-traces-percent-per-case "$MAX_POSITIVE_TRACES_PERCENT_PER_CASE")
fi
if [[ "$REMATERIALIZE" == "1" ]]; then
  extra_positive_args+=(--rematerialize)
fi

judge_reasoning_args=()
if [[ -n "$JUDGE_REASONING_EFFORT" ]]; then
  judge_reasoning_args+=(--judge-reasoning-effort "$JUDGE_REASONING_EFFORT")
fi
if [[ "$JUDGE_DISABLE_THINKING" == "1" ]]; then
  judge_reasoning_args+=(--judge-disable-thinking)
fi

MODELS=(
  "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
  "openai/gpt-5.4-mini"
)

results_model_slug() {
  local model="$1"
  if [[ "$model" == */* ]]; then
    echo "${model#*/}" | tr '/' '-'
  else
    echo "$model"
  fi
}

for model in "${MODELS[@]}"; do
  judge_out="results/${DATASET_NAME}_llmjudge-$(results_model_slug "$model").jsonl"
  meerkat_out="results/${DATASET_NAME}_AT-$(results_model_slug "$model").jsonl"
  buffer_out="results/${DATASET_NAME}_buffer-$(results_model_slug "$model").jsonl"
  bayesian_out="results/${DATASET_NAME}_bayesian-$(results_model_slug "$model").jsonl"
  naive_out="results/${DATASET_NAME}_no_tools_AT-$(results_model_slug "$model").jsonl"

  if [[ "$RUN_JUDGE" == "1" ]]; then
    echo
    if [[ "$FORCE_RERUN" != "1" && -f "$judge_out" ]]; then
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: judge exists, skipping (${judge_out}) ==="
    else
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: judge ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode impossiblebench \
        --data-path "$DATA_PATH" \
        --impossiblebench-source-model "$SOURCE_MODEL" \
        --method judge \
        --judge-output-path "$judge_out" \
        --model "$model" \
        --cases "$CASES" \
        --traces-per-case-list "$TPC_LIST" \
        --positive-rate "$POSITIVE_RATE" \
        "${judge_reasoning_args[@]}" \
        "${extra_positive_args[@]}" \
        --seed "$SEED"
    fi
  fi

  if [[ "$RUN_MEERKAT" == "1" ]]; then
    echo
    if [[ "$FORCE_RERUN" != "1" && -f "$meerkat_out" ]]; then
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: meerkat exists, skipping (${meerkat_out}) ==="
    else
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: meerkat ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode impossiblebench \
        --data-path "$DATA_PATH" \
        --impossiblebench-source-model "$SOURCE_MODEL" \
        --method vibetest \
        --judge-output-path "$judge_out" \
        --vibetest-output-path "$meerkat_out" \
        --model "$model" \
        --cases "$CASES" \
        --traces-per-case-list "$TPC_LIST" \
        --positive-rate "$POSITIVE_RATE" \
        "${judge_reasoning_args[@]}" \
        "${extra_positive_args[@]}" \
        --seed "$SEED"
    fi
  fi

  if [[ "$RUN_BUFFER" == "1" ]]; then
    echo
    if [[ "$FORCE_RERUN" != "1" && -f "$buffer_out" ]]; then
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: buffer exists, skipping (${buffer_out}) ==="
    else
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: buffer ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode impossiblebench \
        --data-path "$DATA_PATH" \
        --impossiblebench-source-model "$SOURCE_MODEL" \
        --method buffer \
        --buffer-size "$BUFFER_SIZE" \
        --vibetest-output-path "$buffer_out" \
        --model "$model" \
        --cases "$CASES" \
        --traces-per-case-list "$TPC_LIST" \
        --positive-rate "$POSITIVE_RATE" \
        "${extra_positive_args[@]}" \
        --seed "$SEED"
    fi
  fi

  if [[ "$RUN_BAYESIAN" == "1" ]]; then
    echo
    if [[ "$FORCE_RERUN" != "1" && -f "$bayesian_out" ]]; then
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: bayesian exists, skipping (${bayesian_out}) ==="
    else
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: bayesian ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode impossiblebench \
        --data-path "$DATA_PATH" \
        --impossiblebench-source-model "$SOURCE_MODEL" \
        --method bayesian \
        --vibetest-output-path "$bayesian_out" \
        --model "$model" \
        --cases "$CASES" \
        --traces-per-case-list "$TPC_LIST" \
        --positive-rate "$POSITIVE_RATE" \
        "${extra_positive_args[@]}" \
        --seed "$SEED"
    fi
  fi

  if [[ "$RUN_NAIVE" == "1" ]]; then
    echo
    if [[ "$FORCE_RERUN" != "1" && -f "$naive_out" ]]; then
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: vibetest --no-tools exists, skipping (${naive_out}) ==="
    else
      echo "=== ImpossibleBench source=${SOURCE_MODEL} model=${model} :: vibetest --no-tools ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode impossiblebench \
        --data-path "$DATA_PATH" \
        --impossiblebench-source-model "$SOURCE_MODEL" \
        --method vibetest \
        --no-tools \
        --vibetest-output-path "$naive_out" \
        --model "$model" \
        --cases "$CASES" \
        --traces-per-case-list "$TPC_LIST" \
        --positive-rate "$POSITIVE_RATE" \
        "${extra_positive_args[@]}" \
        --seed "$SEED"
    fi
  fi
done
