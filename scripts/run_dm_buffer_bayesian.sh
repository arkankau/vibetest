#!/usr/bin/env bash
set -euo pipefail

# Runs the distributed-misuse Meerkat, buffer, Bayesian, and no-tools VibeTest baselines
# for the domain/model/background settings currently used in the DM paper table.
#
# Case counts are inferred from the same Meerkat JSONLs that
# `scripts/analyze_distributed_misuse.py` currently uses for the paper figures.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-uv run --active python}"
read -r -a PYTHON_CMD <<<"$PYTHON_BIN"
SEED="${SEED:-42}"
DECOMP_LEVEL="${DECOMP_LEVEL:-6}"
BUFFER_SIZE="${BUFFER_SIZE:-30}"
CLUSTERING_STRATEGY="${CLUSTERING_STRATEGY:-coherence}"
MEERKAT_CLUSTERS_ONLY="${MEERKAT_CLUSTERS_ONLY:-0}"
FORCE_RERUN_JUDGE="${FORCE_RERUN_JUDGE:-0}"
FORCE_RERUN_MEERKAT="${FORCE_RERUN_MEERKAT:-1}"
FORCE_RERUN_NAIVE="${FORCE_RERUN_NAIVE:-1}"
RUN_ONLY="${RUN_ONLY:-all}"
SETTING_FILTER="${SETTING_FILTER:-}"
ENABLE_CYBER_BG200_QWEN35="${ENABLE_CYBER_BG200_QWEN35:-0}"
ENABLE_CYBER_BG1000_QWEN35="${ENABLE_CYBER_BG1000_QWEN35:-0}"
JUDGE_SOURCE_MODEL="${JUDGE_SOURCE_MODEL:-}"
EXTRA_CYBER_MODELS="${EXTRA_CYBER_MODELS:-}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-}"
JUDGE_DISABLE_THINKING="${JUDGE_DISABLE_THINKING:-}"

if [[ "$RUN_ONLY" != "all" ]]; then
  if [[ "$RUN_ONLY" != "judge" ]]; then
    FORCE_RERUN_JUDGE=0
  fi
  if [[ "$RUN_ONLY" != "meerkat" ]]; then
    FORCE_RERUN_MEERKAT=0
  fi
  if [[ "$RUN_ONLY" != "naive" && "$RUN_ONLY" != "naive-stale" ]]; then
    FORCE_RERUN_NAIVE=0
  fi
fi

echo "DM runner configuration:"
echo "  RUN_ONLY=${RUN_ONLY}"
echo "  SETTING_FILTER=${SETTING_FILTER:-<none>}"
echo "  JUDGE_SOURCE_MODEL=${JUDGE_SOURCE_MODEL:-<same as eval model>}"
echo "  EXTRA_CYBER_MODELS=${EXTRA_CYBER_MODELS:-<none>}"
echo "  JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-<default>}"
echo "  JUDGE_DISABLE_THINKING=${JUDGE_DISABLE_THINKING}"
echo "  MEERKAT_CLUSTERS_ONLY=${MEERKAT_CLUSTERS_ONLY}"
echo "  FORCE_RERUN_JUDGE=${FORCE_RERUN_JUDGE}"
echo "  FORCE_RERUN_MEERKAT=${FORCE_RERUN_MEERKAT}"
echo "  FORCE_RERUN_NAIVE=${FORCE_RERUN_NAIVE}"
echo

override_var_name() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local slug
  slug="$(model_slug "$model" | tr '[:lower:].-' '[:upper:]__' | tr -cd 'A-Z0-9_')"
  printf 'CASE_COUNT_OVERRIDE_%s_BG%s_%s' "$(printf '%s' "$domain" | tr '[:lower:]' '[:upper:]')" "$bg" "$slug"
}

case_count_override() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local var_name
  var_name="$(override_var_name "$domain" "$bg" "$model")"
  printf '%s' "${!var_name:-}"
}

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

naive_result_path() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  printf 'results/dm_%s_d%s_bg%s_%s_naive.jsonl' \
    "$domain" "$DECOMP_LEVEL" "$bg" "$(model_slug "$model")"
}

judge_result_path() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  printf 'results/dm_%s_d%s_bg%s_%s_llmjudge.jsonl' \
    "$domain" "$DECOMP_LEVEL" "$bg" "$(model_slug "$model")"
}

setting_reference_model() {
  local model="$1"
  if [[ -n "$JUDGE_SOURCE_MODEL" ]]; then
    printf '%s' "$JUDGE_SOURCE_MODEL"
  else
    printf '%s' "$model"
  fi
}

no_tools_dataset_name() {
  local dataset_name="$1"
  case "$dataset_name" in
    *_no_tools) echo "$dataset_name" ;;
    *) echo "${dataset_name}_no_tools" ;;
  esac
}

meerkat_result_path() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local slug
  slug="$(model_slug "$model")"

  case "$domain:$bg:$slug" in
    cyber:20:gpt-5.4-mini) printf 'results/dm_cyber_d6_bg20_v6.jsonl' ;;
    cyber:100:gpt-5.4-mini) printf 'results/dm_cyber_d6_bg100_n15.jsonl' ;;
    cyber:20:qwen35) printf 'results/dm_cyber_d6_bg20_qwen35_n20.jsonl' ;;
    cyber:100:qwen35) printf 'results/dm_cyber_d6_bg100_qwen35_n50.jsonl' ;;
    cyber:200:qwen35) printf 'results/dm_cyber_d6_bg200_qwen35_n50.jsonl' ;;
    cyber:1000:qwen35) printf 'results/dm_cyber_d6_bg1000_qwen35_n20.jsonl' ;;
    bio:20:gpt-5.4-mini) printf 'results/dm_bio_d6_bg20_n20.jsonl' ;;
    bio:100:gpt-5.4-mini) printf 'results/dm_bio_d6_bg100_n50.jsonl' ;;
    bio:20:qwen35) printf 'results/dm_bio_d6_bg20_qwen35_n20.jsonl' ;;
    bio:100:qwen35) printf 'results/dm_bio_d6_bg100_qwen35_n50.jsonl' ;;
    *) printf 'results/dm_%s_d%s_bg%s_%s.jsonl' "$domain" "$DECOMP_LEVEL" "$bg" "$slug" ;;
  esac
}

dataset_name_from_meerkat() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local reference_model="${4:-$model}"
  local meerkat_path
  local fallback
  local repo_path
  local dataset_name
  meerkat_path="$(meerkat_result_path "$domain" "$bg" "$reference_model")"
  fallback="safety_dm_${domain}_d${DECOMP_LEVEL}_bg${bg}_$(model_slug "$reference_model")"
  if [[ ! -f "$meerkat_path" ]]; then
    echo "$fallback"
    return 0
  fi
  repo_path="$(head -n 1 "$meerkat_path" | sed -n 's/.*"repo"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  dataset_name="$(printf '%s\n' "$repo_path" | sed -n 's#.*trace_repos/\([^/]*\)/.*#\1#p')"
  echo "${dataset_name:-$fallback}"
}

case_count_from_meerkat() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local meerkat_path
  meerkat_path="$(meerkat_result_path "$domain" "$bg" "$model")"
  if [[ ! -f "$meerkat_path" ]]; then
    echo "Missing Meerkat result file for domain=${domain} bg=${bg} model=${model}: ${meerkat_path}" >&2
    return 1
  fi
  wc -l < "$meerkat_path" | tr -d '[:space:]'
}

expected_case_count_fallback() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local slug
  slug="$(model_slug "$model")"

  case "$domain:$bg:$slug" in
    cyber:20:gpt-5.4-mini) echo "3" ;;
    cyber:100:gpt-5.4-mini) echo "15" ;;
    cyber:20:qwen35) echo "20" ;;
    cyber:100:qwen35) echo "50" ;;
    cyber:200:qwen35) echo "50" ;;
    cyber:1000:qwen35) echo "20" ;;
    bio:20:gpt-5.4-mini) echo "20" ;;
    bio:100:gpt-5.4-mini) echo "50" ;;
    bio:20:qwen35) echo "20" ;;
    bio:100:qwen35) echo "50" ;;
    *)
      echo "No case-count fallback for domain=${domain} bg=${bg} model=${model}" >&2
      return 1
      ;;
  esac
}

preferred_case_count() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local slug
  slug="$(model_slug "$model")"

  case "$domain:$bg:$slug" in
    bio:20:gpt-5.4-mini) echo "20" ;;
    bio:100:gpt-5.4-mini) echo "50" ;;
    *) echo "" ;;
  esac
}

case_count_for_setting() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local reference_model="${4:-$model}"
  local candidate
  local count

  count="$(case_count_override "$domain" "$bg" "$model")"
  if [[ -n "$count" ]]; then
    echo "$count"
    return 0
  fi

  count="$(preferred_case_count "$domain" "$bg" "$model")"
  if [[ -n "$count" ]]; then
    echo "$count"
    return 0
  fi

  if count="$(case_count_from_meerkat "$domain" "$bg" "$model" 2>/dev/null)"; then
    echo "$count"
    return 0
  fi

  for candidate in \
    "$(naive_result_path "$domain" "$bg" "$model")" \
    "$(result_path "$domain" "$bg" "$model" buffer)" \
    "$(result_path "$domain" "$bg" "$model" bayesian)" \
    "$(judge_result_path "$domain" "$bg" "$model")"
  do
    if [[ -f "$candidate" ]]; then
      result_case_count "$candidate"
      return 0
    fi
  done

  if [[ "$reference_model" != "$model" ]]; then
    count="$(preferred_case_count "$domain" "$bg" "$reference_model")"
    if [[ -n "$count" ]]; then
      echo "$count"
      return 0
    fi

    if count="$(case_count_from_meerkat "$domain" "$bg" "$reference_model" 2>/dev/null)"; then
      echo "$count"
      return 0
    fi

    for candidate in \
      "$(naive_result_path "$domain" "$bg" "$reference_model")" \
      "$(result_path "$domain" "$bg" "$reference_model" buffer)" \
      "$(result_path "$domain" "$bg" "$reference_model" bayesian)" \
      "$(judge_result_path "$domain" "$bg" "$reference_model")"
    do
      if [[ -f "$candidate" ]]; then
        result_case_count "$candidate"
        return 0
      fi
    done
  fi

  expected_case_count_fallback "$domain" "$bg" "$reference_model"
}

result_case_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo ""
    return 0
  fi
  wc -l < "$path" | tr -d '[:space:]'
}

should_skip_result() {
  local path="$1"
  local expected_cases="$2"
  local actual_cases
  if [[ ! -f "$path" ]]; then
    return 1
  fi
  actual_cases="$(result_case_count "$path")"
  [[ "$actual_cases" == "$expected_cases" ]]
}

should_run_method() {
  local method="$1"
  case "$RUN_ONLY" in
    all) return 0 ;;
    judge) [[ "$method" == "judge" ]] ;;
    naive) [[ "$method" == "naive" ]] ;;
    naive-stale) [[ "$method" == "naive" ]] ;;
    meerkat) [[ "$method" == "meerkat" ]] ;;
    buffer) [[ "$method" == "buffer" ]] ;;
    bayesian) [[ "$method" == "bayesian" ]] ;;
    baselines) [[ "$method" == "buffer" || "$method" == "bayesian" ]] ;;
    non-naive) [[ "$method" != "naive" ]] ;;
    *)
      echo "Unsupported RUN_ONLY=${RUN_ONLY}. Expected one of: all, judge, naive, naive-stale, meerkat, buffer, bayesian, baselines, non-naive" >&2
      return 1
      ;;
  esac
}

should_run_setting() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local filter
  local slug
  local desc
  slug="$(model_slug "$model")"
  desc="${domain}:bg${bg}:${slug}"
  if [[ -z "$SETTING_FILTER" ]]; then
    return 0
  fi
  IFS=',' read -r -a filters <<<"$SETTING_FILTER"
  for filter in "${filters[@]}"; do
    filter="${filter//[[:space:]]/}"
    [[ -z "$filter" ]] && continue
    if [[ "$desc" == "$filter" || "$domain:$slug" == "$filter" || "$slug" == "$filter" || "$domain" == "$filter" ]]; then
      return 0
    fi
  done
  return 1
}

naive_result_staleness_reason() {
  local path="$1"
  local expected_cases="$2"
  local dataset_name="$3"
  local actual_cases
  local expected_no_tools_dataset

  if [[ ! -f "$path" ]]; then
    echo ""
    return 0
  fi

  actual_cases="$(result_case_count "$path")"
  if [[ "$actual_cases" != "$expected_cases" ]]; then
    echo "case-count=${actual_cases} expected=${expected_cases}"
    return 0
  fi

  expected_no_tools_dataset="$(no_tools_dataset_name "$dataset_name")"
  NAIVE_PATH="$path" EXPECTED_NO_TOOLS_DATASET="$expected_no_tools_dataset" "${PYTHON_CMD[@]}" - <<'PY'
import json
import os
import re
from pathlib import Path

path = Path(os.environ["NAIVE_PATH"])
expected_dataset = os.environ["EXPECTED_NO_TOOLS_DATASET"]
reasons: list[str] = []

try:
    with path.open() as fh:
        first = json.loads(next(fh))
except Exception:
    print("unreadable-jsonl")
    raise SystemExit(0)

repo = str(first.get("repo") or "")
match = re.search(r"trace_repos/([^/]+)/", repo)
repo_dataset = match.group(1) if match else ""
if repo_dataset != expected_dataset:
    reasons.append(f"repo-dataset={repo_dataset or 'missing'} expected={expected_dataset}")

tests = first.get("tests") or []
metadata = (tests[0].get("metadata") or {}) if tests else {}
test_description = str(metadata.get("test_description") or "")
if "initial_scores.tsv" in test_description or "clusters.txt" in test_description or "clusters.json" in test_description:
    reasons.append("prompt mentions initial_scores.tsv/clusters artifact")

print("; ".join(reasons))
PY
}

meerkat_result_staleness_reason() {
  local path="$1"
  local expected_cases="$2"

  if [[ ! -f "$path" ]]; then
    echo ""
    return 0
  fi

  MEERKAT_PATH="$path" EXPECTED_CASES="$expected_cases" EXPECT_CLUSTERS_ONLY="$MEERKAT_CLUSTERS_ONLY" "${PYTHON_CMD[@]}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["MEERKAT_PATH"])
expected_cases = os.environ["EXPECTED_CASES"]
reasons: list[str] = []

actual_cases = sum(1 for _ in path.open())
if str(actual_cases) != expected_cases:
    reasons.append(f"case-count={actual_cases} expected={expected_cases}")

try:
    with path.open() as fh:
        first = json.loads(next(fh))
except Exception:
    print("unreadable-jsonl")
    raise SystemExit(0)

tests = first.get("tests") or []
metadata = (tests[0].get("metadata") or {}) if tests else {}
test_description = str(metadata.get("test_description") or "")
execution_log = str(tests[0].get("execution_log") or "") if tests else ""
repo = Path(str(first.get("repo") or ""))
expect_clusters_only = os.environ.get("EXPECT_CLUSTERS_ONLY") == "1"

if "search_aids/clusters.json" not in test_description:
    reasons.append("prompt-missing-search_aids-clusters.json")
if "clusters.txt" in test_description:
    reasons.append("prompt-still-mentions-clusters.txt")
if expect_clusters_only:
    if "search_aids/initial_scores.tsv" in test_description:
        reasons.append("prompt-still-mentions-search_aids-initial_scores.tsv")
    if "search_aids/initial_scores_prompt.txt" in test_description:
        reasons.append("prompt-still-mentions-search_aids-initial_scores_prompt.txt")
else:
    if "search_aids/initial_scores.tsv" not in test_description:
        reasons.append("prompt-missing-search_aids-initial_scores.tsv")
    if "search_aids/initial_scores_prompt.txt" not in test_description:
        reasons.append("prompt-missing-search_aids-initial_scores_prompt.txt")
if "search_aids/clusters.json" not in execution_log:
    reasons.append("execution-log-missing-search_aids-clusters.json")
if "clusters.txt" in execution_log:
    reasons.append("execution-log-still-mentions-clusters.txt")
if expect_clusters_only:
    initial_scores_path = repo / "search_aids" / "initial_scores.tsv"
    if initial_scores_path.exists():
        reasons.append("repo-still-has-search_aids-initial_scores.tsv")
    initial_scores_prompt_path = repo / "search_aids" / "initial_scores_prompt.txt"
    if initial_scores_prompt_path.exists():
        reasons.append("repo-still-has-search_aids-initial_scores_prompt.txt")
else:
    initial_scores_prompt_path = repo / "search_aids" / "initial_scores_prompt.txt"
    if not initial_scores_prompt_path.exists():
        reasons.append("repo-missing-search_aids-initial_scores_prompt.txt")
    repeated_anchor_threads_path = repo / "search_aids" / "repeated_anchor_threads.tsv"
    if not repeated_anchor_threads_path.exists():
        reasons.append("repo-missing-search_aids-repeated_anchor_threads.tsv")
legacy_tracker_path = repo / "search_aids" / "hypothesis_tracker.py"
if legacy_tracker_path.exists():
    reasons.append("repo-still-has-search_aids-hypothesis_tracker.py")
clusters_json_path = repo / "search_aids" / "clusters.json"
if not clusters_json_path.exists():
    reasons.append("repo-missing-search_aids-clusters.json")
else:
    try:
        clusters_payload = json.loads(clusters_json_path.read_text())
    except Exception:
        reasons.append("repo-unreadable-search_aids-clusters.json")
    else:
        construction = str(clusters_payload.get("construction") or "")
        if "fixed-k k-means" not in construction or "n // 8" not in construction:
            reasons.append("repo-stale-cluster-construction")
        clusters = clusters_payload.get("clusters") or []
        required_fields = {"label", "summary", "avg_distance_to_centroid", "avg_silhouette", "traces"}
        if not clusters:
            reasons.append("repo-empty-clusters")
        elif not all(required_fields.issubset(cluster) for cluster in clusters):
            reasons.append("repo-clusters-missing-fields")

print("; ".join(reasons))
PY
}

run_setting() {
  local domain="$1"
  local bg="$2"
  local model="$3"
  local judge_model
  local cases
  local buffer_out
  local bayesian_out
  local naive_out
  local meerkat_out
  local judge_out
  local judge_reasoning_args
  local dataset_name
  local meerkat_initial_scores_arg
  if ! should_run_setting "$domain" "$bg" "$model"; then
    echo "=== ${domain} bg=${bg} model=${model} :: skipping setting because SETTING_FILTER=${SETTING_FILTER} ==="
    return 0
  fi
  judge_model="$(setting_reference_model "$model")"
  meerkat_initial_scores_arg=""
  if [[ "$MEERKAT_CLUSTERS_ONLY" == "1" ]]; then
    meerkat_initial_scores_arg=" --no-safety-repo-initial-scores"
  fi
  meerkat_out="$(meerkat_result_path "$domain" "$bg" "$model")"
  cases="$(case_count_for_setting "$domain" "$bg" "$model" "$judge_model")"
  dataset_name="$(dataset_name_from_meerkat "$domain" "$bg" "$model" "$judge_model")"
  buffer_out="$(result_path "$domain" "$bg" "$model" buffer)"
  bayesian_out="$(result_path "$domain" "$bg" "$model" bayesian)"
  naive_out="$(naive_result_path "$domain" "$bg" "$model")"
  judge_out="$(judge_result_path "$domain" "$bg" "$judge_model")"
  judge_reasoning_args=()
  if [[ -n "$JUDGE_REASONING_EFFORT" ]]; then
    judge_reasoning_args+=(--judge-reasoning-effort "$JUDGE_REASONING_EFFORT")
  fi
  if [[ "$JUDGE_DISABLE_THINKING" == "1" ]]; then
    judge_reasoning_args+=(--judge-disable-thinking)
  elif [[ "$JUDGE_DISABLE_THINKING" == "0" ]]; then
    judge_reasoning_args+=(--no-judge-disable-thinking)
  fi

  echo
  if should_run_method judge; then
    if [[ "$FORCE_RERUN_JUDGE" == "1" && -f "$judge_out" ]]; then
      echo "=== ${domain} bg=${bg} model=${model} :: forcing llmjudge rerun, removing (${judge_out}) ==="
      rm -f "$judge_out"
    fi
    if should_skip_result "$judge_out" "$cases"; then
      echo "=== ${domain} bg=${bg} model=${model} :: llmjudge exists with ${cases} cases, skipping (${judge_out}) ==="
    else
      if [[ -f "$judge_out" ]]; then
        echo "=== ${domain} bg=${bg} model=${model} :: llmjudge has stale case count $(result_case_count "$judge_out"), rerunning (${judge_out}) ==="
        rm -f "$judge_out"
      fi
      echo "=== ${domain} bg=${bg} model=${model} judge_model=${judge_model} cases=${cases} dataset=${dataset_name} :: llmjudge (${judge_out}) ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain "$domain" \
        --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
        --distributed-misuse-background-multiplier "$bg" \
        --dataset-name "$dataset_name" \
        --method judge \
        --judge-output-path "$judge_out" \
        --model "$judge_model" \
        --cases "$cases" \
        "${judge_reasoning_args[@]}" \
        --seed "$SEED"
    fi
  else
    echo "=== ${domain} bg=${bg} model=${model} :: skipping llmjudge because RUN_ONLY=${RUN_ONLY} ==="
  fi

  echo
  if should_run_method meerkat; then
    local meerkat_stale_reason
    meerkat_stale_reason="$(meerkat_result_staleness_reason "$meerkat_out" "$cases")"
    if [[ "$FORCE_RERUN_MEERKAT" == "1" && -f "$meerkat_out" ]]; then
      echo "=== ${domain} bg=${bg} model=${model} :: forcing Meerkat rerun, removing (${meerkat_out}) ==="
      rm -f "$meerkat_out"
    elif [[ -n "$meerkat_stale_reason" ]]; then
      echo "=== ${domain} bg=${bg} model=${model} :: stale Meerkat result detected (${meerkat_stale_reason}), rerunning (${meerkat_out}) ==="
      rm -f "$meerkat_out"
    fi
    if should_skip_result "$meerkat_out" "$cases"; then
      echo "=== ${domain} bg=${bg} model=${model} :: Meerkat exists with ${cases} cases, skipping (${meerkat_out}) ==="
    else
      if [[ -f "$meerkat_out" ]]; then
        echo "=== ${domain} bg=${bg} model=${model} :: Meerkat has stale case count $(result_case_count "$meerkat_out"), rerunning (${meerkat_out}) ==="
        rm -f "$meerkat_out"
      fi
      echo "=== ${domain} bg=${bg} model=${model} judge_model=${judge_model} cases=${cases} dataset=${dataset_name} :: Meerkat (${meerkat_out}) ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain "$domain" \
        --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
        --distributed-misuse-background-multiplier "$bg" \
        --distributed-misuse-clustering-strategy "$CLUSTERING_STRATEGY" \
        --dataset-name "$dataset_name" \
        --method vibetest \
        --judge-output-path "$judge_out" \
        --vibetest-output-path "$meerkat_out" \
        --model "$model" \
        --cases "$cases" \
        --seed "$SEED" \
        $meerkat_initial_scores_arg
    fi
  else
    echo "=== ${domain} bg=${bg} model=${model} :: skipping Meerkat because RUN_ONLY=${RUN_ONLY} ==="
  fi

  echo
  if should_run_method buffer; then
    if should_skip_result "$buffer_out" "$cases"; then
      echo "=== ${domain} bg=${bg} model=${model} :: buffer exists with ${cases} cases, skipping (${buffer_out}) ==="
    else
      if [[ -f "$buffer_out" ]]; then
        echo "=== ${domain} bg=${bg} model=${model} :: buffer has stale case count $(result_case_count "$buffer_out"), rerunning (${buffer_out}) ==="
        rm -f "$buffer_out"
      fi
      echo "=== ${domain} bg=${bg} model=${model} cases=${cases} dataset=${dataset_name} :: buffer (matched to ${meerkat_out}) ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain "$domain" \
        --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
        --distributed-misuse-background-multiplier "$bg" \
        --dataset-name "$dataset_name" \
        --method buffer \
        --buffer-size "$BUFFER_SIZE" \
        --model "$model" \
        --cases "$cases" \
        --seed "$SEED"
    fi
  else
    echo "=== ${domain} bg=${bg} model=${model} :: skipping buffer because RUN_ONLY=${RUN_ONLY} ==="
  fi

  echo
  if should_run_method bayesian; then
    if should_skip_result "$bayesian_out" "$cases"; then
      echo "=== ${domain} bg=${bg} model=${model} :: bayesian exists with ${cases} cases, skipping (${bayesian_out}) ==="
    else
      if [[ -f "$bayesian_out" ]]; then
        echo "=== ${domain} bg=${bg} model=${model} :: bayesian has stale case count $(result_case_count "$bayesian_out"), rerunning (${bayesian_out}) ==="
        rm -f "$bayesian_out"
      fi
      echo "=== ${domain} bg=${bg} model=${model} cases=${cases} dataset=${dataset_name} :: bayesian (matched to ${meerkat_out}) ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain "$domain" \
        --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
        --distributed-misuse-background-multiplier "$bg" \
        --dataset-name "$dataset_name" \
        --method bayesian \
        --model "$model" \
        --cases "$cases" \
        --seed "$SEED"
    fi
  else
    echo "=== ${domain} bg=${bg} model=${model} :: skipping bayesian because RUN_ONLY=${RUN_ONLY} ==="
  fi

  echo
  if should_run_method naive; then
    local naive_stale_reason
    naive_stale_reason="$(naive_result_staleness_reason "$naive_out" "$cases" "$dataset_name")"
    if [[ "$RUN_ONLY" == "naive-stale" ]]; then
      if [[ ! -f "$naive_out" ]]; then
        echo "=== ${domain} bg=${bg} model=${model} :: naive result missing, skipping because RUN_ONLY=naive-stale only reruns existing stale outputs (${naive_out}) ==="
      elif [[ -z "$naive_stale_reason" ]]; then
        echo "=== ${domain} bg=${bg} model=${model} :: naive result is current, skipping (${naive_out}) ==="
      else
        echo "=== ${domain} bg=${bg} model=${model} :: stale naive result detected (${naive_stale_reason}), rerunning (${naive_out}) ==="
        rm -f "$naive_out"
        echo "=== ${domain} bg=${bg} model=${model} cases=${cases} dataset=${dataset_name} :: vibetest --no-tools (matched to ${meerkat_out}) ==="
        eval "$PYTHON_BIN experiments/safety.py" \
          --safety-mode distributed-misuse \
          --distributed-misuse-domain "$domain" \
          --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
          --distributed-misuse-background-multiplier "$bg" \
          --dataset-name "$dataset_name" \
          --method vibetest \
          --no-tools \
          --vibetest-output-path "$naive_out" \
          --model "$model" \
          --cases "$cases" \
          --seed "$SEED"
      fi
    elif [[ "$FORCE_RERUN_NAIVE" == "1" && -f "$naive_out" ]]; then
      echo "=== ${domain} bg=${bg} model=${model} :: forcing vibetest --no-tools rerun, removing (${naive_out}) ==="
      rm -f "$naive_out"
      echo "=== ${domain} bg=${bg} model=${model} cases=${cases} dataset=${dataset_name} :: vibetest --no-tools (matched to ${meerkat_out}) ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain "$domain" \
        --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
        --distributed-misuse-background-multiplier "$bg" \
        --dataset-name "$dataset_name" \
        --method vibetest \
        --no-tools \
        --vibetest-output-path "$naive_out" \
        --model "$model" \
        --cases "$cases" \
        --seed "$SEED"
    elif should_skip_result "$naive_out" "$cases" && [[ -z "$naive_stale_reason" ]]; then
      echo "=== ${domain} bg=${bg} model=${model} :: vibetest --no-tools exists with ${cases} cases, skipping (${naive_out}) ==="
    else
      if [[ -f "$naive_out" ]]; then
        if [[ -n "$naive_stale_reason" ]]; then
          echo "=== ${domain} bg=${bg} model=${model} :: vibetest --no-tools is stale (${naive_stale_reason}), rerunning (${naive_out}) ==="
        else
          echo "=== ${domain} bg=${bg} model=${model} :: vibetest --no-tools has stale case count $(result_case_count "$naive_out"), rerunning (${naive_out}) ==="
        fi
        rm -f "$naive_out"
      fi
      echo "=== ${domain} bg=${bg} model=${model} cases=${cases} dataset=${dataset_name} :: vibetest --no-tools (matched to ${meerkat_out}) ==="
      eval "$PYTHON_BIN experiments/safety.py" \
        --safety-mode distributed-misuse \
        --distributed-misuse-domain "$domain" \
        --distributed-misuse-decomp-level "$DECOMP_LEVEL" \
        --distributed-misuse-background-multiplier "$bg" \
        --dataset-name "$dataset_name" \
        --method vibetest \
        --no-tools \
        --vibetest-output-path "$naive_out" \
        --model "$model" \
        --cases "$cases" \
        --seed "$SEED"
    fi
  else
    echo "=== ${domain} bg=${bg} model=${model} :: skipping vibetest --no-tools because RUN_ONLY=${RUN_ONLY} ==="
  fi
}

run_setting cyber 20 "openai/gpt-5.4-mini"
run_setting cyber 100 "openai/gpt-5.4-mini"
run_setting cyber 20 "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
run_setting cyber 100 "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
if [[ "$ENABLE_CYBER_BG200_QWEN35" == "1" ]]; then
  run_setting cyber 200 "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
fi
if [[ "$ENABLE_CYBER_BG1000_QWEN35" == "1" ]] || should_run_setting cyber 1000 "vllm/Qwen/Qwen3.5-397B-A17B-FP8"; then
  run_setting cyber 1000 "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
fi
if [[ -n "$EXTRA_CYBER_MODELS" ]]; then
  IFS=',' read -r -a extra_cyber_models <<<"$EXTRA_CYBER_MODELS"
  for model in "${extra_cyber_models[@]}"; do
    model="${model//[[:space:]]/}"
    [[ -z "$model" ]] && continue
    run_setting cyber 20 "$model"
    run_setting cyber 100 "$model"
  done
fi

run_setting bio 20 "openai/gpt-5.4-mini"
run_setting bio 100 "openai/gpt-5.4-mini"
run_setting bio 20 "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
run_setting bio 100 "vllm/Qwen/Qwen3.5-397B-A17B-FP8"
