#!/usr/bin/env bash
set -euo pipefail

# Runs the distributed-misuse buffer and Bayesian baselines for the
# domain/model/background settings currently used in the DM paper table.
#
# Case counts are matched to the currently selected Meerkat result files:
# - Cyber, gpt-5.4-mini: bg=20 -> 3 cases, bg=100 -> 15 cases
# - Cyber, Qwen-3.5:     bg=20 -> 20 cases, bg=100 -> 50 cases
# - Bio,   gpt-5.4-mini: bg=20 -> 3 cases, bg=100 -> 3 cases
# - Bio,   Qwen-3.5:     bg=20 -> 20 cases, bg=100 -> 50 cases

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-uv run --active python}"
SEED="${SEED:-42}"
DECOMP_LEVEL="${DECOMP_LEVEL:-6}"
BUFFER_SIZE="${BUFFER_SIZE:-30}"

model_slug() {
  local model="$1"
  case "$model" in
    *Qwen3.5*|*Qwen-3.5*) echo "qwen35" ;;
    */*) echo "${model##*/}" ;;
    *) echo "$model" ;;
  esac
}

result_path() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local method="$4"
  printf 'results/dm_%s_d%s_bg%s_%s_%s.jsonl' \
    "$domain" "$DECOMP_LEVEL" "$bg" "$(model_slug "$model")" "$method"
}

run_setting() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local cases="$4"
  local buffer_out
  local bayesian_out
  buffer_out="$(result_path "$domain" "$bg" "$model" buffer)"
  bayesian_out="$(result_path "$domain" "$bg" "$model" bayesian)"

  echo
  if [[ -f "$buffer_out" ]]; then
    echo "=== ${domain} bg=${bg} model=${model} :: buffer exists, skipping (${buffer_out}) ==="
  else
    echo "=== ${domain} bg=${bg} model=${model} cases=${cases} :: buffer ==="
    eval "$PYTHON_BIN experiments/safety.py" \
      --safety-mode distributed-misuse \
      --distributed-misuse-domain "$domain" \
      --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
      --distributed-misuse-background-multiplier "$bg" \
      --method buffer \
      --buffer-size "$BUFFER_SIZE" \
      --model "$model" \
      --cases "$cases" \
      --seed "$SEED"
  fi

  echo
  if [[ -f "$bayesian_out" ]]; then
    echo "=== ${domain} bg=${bg} model=${model} :: bayesian exists, skipping (${bayesian_out}) ==="
  else
    echo "=== ${domain} bg=${bg} model=${model} cases=${cases} :: bayesian ==="
    eval "$PYTHON_BIN experiments/safety.py" \
      --safety-mode distributed-misuse \
      --distributed-misuse-domain "$domain" \
      --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
      --distributed-misuse-background-multiplier "$bg" \
      --method bayesian \
      --model "$model" \
      --cases "$cases" \
      --seed "$SEED"
  fi
}

run_setting cyber 20 "openai/gpt-5.4-mini" 3
run_setting cyber 100 "openai/gpt-5.4-mini" 15
run_setting cyber 20 "vllm/Qwen/Qwen3.5-397B-A17B-FP8" 20
run_setting cyber 100 "vllm/Qwen/Qwen3.5-397B-A17B-FP8" 50

run_setting bio 20 "openai/gpt-5.4-mini" 3
run_setting bio 100 "openai/gpt-5.4-mini" 3
run_setting bio 20 "vllm/Qwen/Qwen3.5-397B-A17B-FP8" 20
run_setting bio 100 "vllm/Qwen/Qwen3.5-397B-A17B-FP8" 50
