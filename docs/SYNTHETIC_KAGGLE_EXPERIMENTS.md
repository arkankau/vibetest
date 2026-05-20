# Synthetic Kaggle Experiments

This note describes the ML synthetic bug dataset and how to run VibeTest on it.
The dataset contains Kaggle notebook repositories with synthetic ML pipeline
bugs injected into otherwise clean repos.

## Data Archive

The share archive is `synthetic-kaggle-data.tar.gz`. The corrected archive has
SHA256:

```text
9c6862a559ff4c35ce3ca43115298aca064b8d90d77a2dee480d8628fb07349f
```

Extract it from the repo root:

```sh
tar -xzf /path/to/synthetic-kaggle-data.tar.gz
```

After extraction, the expected files are:

- `synth-data/ml-bugs/clean_repos.jsonl`
- `synth-data/ml-bugs/bug_examples.jsonl`
- `synth-data/ml-bugs/summary.json`
- `synth-data/injected/labels_kaggle_titanic.jsonl`
- `synth-data/injected/labels_kaggle_diabetic.jsonl`
- `synth-data/injected/labels_kaggle_nlp.jsonl`
- `synth-data/injected/repos/ml-bugs/`
- `synth-data/injected/injection_run_summary.json`

The label files are the source of truth for experiment inputs. Each label row
contains `output_repo_path`, `ground_truth_property_labels`,
`ground_truth_violation_descriptions`, and `ground_truth_by_property`.

Current archive contents:

| Dataset | Label file | Rows | Injection model |
| --- | --- | ---: | --- |
| Titanic | `synth-data/injected/labels_kaggle_titanic.jsonl` | 25 | `openai/gpt-5.2-codex` |
| Diabetic retinopathy | `synth-data/injected/labels_kaggle_diabetic.jsonl` | 25 | `openai/gpt-5.2-codex` |
| NLP | `synth-data/injected/labels_kaggle_nlp.jsonl` | 25 | `openai/gpt-5.2-codex` |

## Setup

Install dependencies and configure model credentials:

```sh
uv sync
cp .env.example .env
```

Add the needed API keys to `.env`, typically `OPENAI_API_KEY` for the commands
below. Docker must be available for sandboxed VibeTest runs.

Optional quick validation after extracting the archive:

```sh
uv run --active python scripts/view_synthetic_bugs.py \
  --labels synth-data/injected/labels_kaggle_titanic.jsonl \
  --list
```

## Run VibeTest

Run the synthetic VibeTest method on one Kaggle split:

```sh
uv run --active python experiments/synthetic.py \
  --method vibetest \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --datasets kaggle_titanic \
  --model openai/gpt-5-mini \
  --sandbox docker \
  --scorer-model openai/gpt-5-mini
```

Run all three Kaggle splits:

```sh
for dataset in kaggle_titanic kaggle_diabetic kaggle_nlp; do
  uv run --active python experiments/synthetic.py \
    --method vibetest \
    --labels-path "synth-data/injected/labels_${dataset}.jsonl" \
    --datasets "$dataset" \
    --model openai/gpt-5-mini \
    --sandbox docker \
    --scorer-model openai/gpt-5-mini
done
```

The loop above writes standardized JSONL files under `results/synthetic/`.

To run the Codex review baseline:

```sh
uv run --active python experiments/synthetic.py \
  --method codex \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --datasets kaggle_titanic \
  --model openai/gpt-5-mini \
  --codex-cmd codex \
  --codex-model inspect \
  --codex-prompt /review \
  --sandbox docker \
  --review-mapper-model openai/gpt-5-mini \
  --scorer-model openai/gpt-5-mini
```

To run TrainCheck on an ML split:

```sh
uv run --active python experiments/synthetic.py \
  --method traincheck \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --datasets kaggle_titanic \
  --traincheck-reference data/traincheck/mnist.py \
  --traincheck-timeout 1200 \
  --review-mapper-model openai/gpt-5-mini \
  --scorer-model openai/gpt-5-mini
```

## Analyze Results

Compute verifier-backed precision, recall, F1, response rate, and cost summaries:

```sh
uv run --active python scripts/analyze_synthetic_metrics.py \
  --results-dir results/synthetic \
  --glob 'synthetic_kaggle_*.jsonl' \
  --output-csv results/synthetic/synthetic_kaggle_metrics.csv \
  --output-md results/synthetic/synthetic_kaggle_metrics.md
```

Scoring treats the positive class as ground-truth property failures. A predicted
FAIL is a true positive only when the verifier grades the evidence as matching
the injected violation. `INCONCLUSIVE` and `NOT APPLICABLE` are abstentions.

## Regenerating Injections

The current shared archive already contains injected repos and labels. To create
new injections instead, use `scripts/create_synthetic_bug_dataset.py`:

```sh
uv run --active python scripts/create_synthetic_bug_dataset.py inject \
  --plan-path synth-data/injection_plan_25_per_subdataset.jsonl \
  --output-root synth-data/injected \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --domains ml-bugs \
  --datasets kaggle_titanic \
  --model openai/gpt-5.2-codex \
  --sandbox docker \
  --confirm
```

Regeneration is expensive and will overwrite or create new injected data under
`synth-data/injected`; use a separate output root if preserving the current
archive contents matters.
