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

Synthetic VibeTest runs place injected repos under `/kaggle/repo`, matching the
real Kaggle experiment sandbox. Titanic runs mount `titanic-kaggle-data/` at
`/kaggle/input/titanic/`, and NLP runs mount `nlp-kaggle-data/` at
`/kaggle/input/nlp-getting-started/`, when those local data directories are
present. Diabetic synthetic runs currently match the real Diabetic runner and do
not mount an additional local data directory.

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
  --traincheck-invariants results/traincheck_qwen36_mapper/simple/trace/invariants.json \
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

## Verify FAIL Evidence

To run a clean-context evidence verifier on existing VibeTest FAIL results
without rerunning the original VibeTest agent, use:

```sh
uv run --active python scripts/verify_vibetest_fail_evidence.py \
  results/synthetic/synthetic_kaggle_titanic_AT-gpt-5-mini.jsonl \
  --model openai/gpt-5-mini \
  --sandbox docker
```

This writes `*_evidence_verified.jsonl` by default. Use `--in-place` to annotate
the input file directly. The verifier adds `metadata.evidence_verifier` to each
FAIL test with a score from 0.0 to 1.0, a reason, and an evidence assessment.
The score measures only how clearly the original evidence supports the original
FAIL verdict; it is independent of synthetic ground truth correctness.

By default the verifier also looks for saved base-agent evidence bundles under
`evidence-dumps/<model>/evidence-<sample-id>.tar.gz`, extracts them, and makes
them available to the clean-context verifier at `/evidence`. Current synthetic
Kaggle VibeTest sample IDs are stored in test metadata and include the dataset
name, for example `kaggle_titanic_row0_kaggle_p3`. Older result files used
`row<synthetic_row_index>_<property_id>` names. Use `--evidence-root` if the
evidence bundles live somewhere else, `--evidence-model` if the model folder
differs from the result metadata, or `--fail-on-missing-evidence` to require
every FAIL item to have a bundle.

## Iterative VibeTest + Verifier

The standard synthetic experiment runs VibeTest once. To enable iterative
feedback, add `--iterative-verifier` to the `experiments/synthetic.py` VibeTest
command:

```sh
uv run --active python experiments/synthetic.py \
  --method vibetest \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --datasets kaggle_titanic \
  --model openai/gpt-5-mini \
  --iterative-verifier \
  --max-iterations 3 \
  --verifier-threshold 0.7
```

In this mode, each VibeTest FAIL is checked by the clean-context verifier. If
the verifier score is below the threshold, VibeTest is rerun for that property
with the verifier's feedback in `extra_instructions`. `--max-iterations` is the
maximum number of total VibeTest attempts, including the initial attempt. The
default is 3, and iterative behavior is disabled unless `--iterative-verifier`
is passed.

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
