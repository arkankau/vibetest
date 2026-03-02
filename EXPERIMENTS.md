## Real Bug Experiments

The experiment scripts are located under `experiments/`. The main experiments
can be run as the following to run vibetest and the baselines can be run by
changing the `--method` parameter.

```sh
# two sec. vuln. datasets
uv run experiments/vulnerabilities.py --method vibetest --dataset cwe-bench
uv run experiments/vulnerabilities.py --method vibetest --dataset bibifi
# code review baseline (Codex)
uv run experiments/vulnerabilities.py --method codex --dataset cwe-bench

# three kaggle datasets for ml properties
uv run experiments/kaggle.py --method vibetest --dataset nlp
uv run experiments/kaggle.py --method vibetest --dataset diabetic
uv run experiments/kaggle.py --method vibetest --dataset titanic
# TrainCheck + code review baseline
uv run experiments/kaggle.py --method traincheck --dataset nlp
uv run experiments/kaggle.py --method codex --dataset nlp

# iclr dataset for ml properties
uv run experiments/iclr2026.py --method vibetest

# citation hallucinations
uv run experiments/hallucination.py --method vibetest --paper-list data/hallucination/sample_100.txt
```

## Synthetic (Injected) Bug Experiments

To overcome the problem with a lack of ground truth with the real bug
experiments above, we create a synthetic bug dataset seeded by the real bugs
from above. We take clean code with no known bugs and inject different property
failures into the code based on some examples of failures from the above
experiments. This gives us a dataset of code labelled with exactly the bugs
which it contains allowing us to automatically measure a tool's precision and
recall without any costly human annotation.

### Dataset Creation
1. For each of the domains (security vuln, ml bugs, citation hallucinations) we
collect all the "clean" examples as those which are not labelled with any "FAIL"
or "INCONCLUSIVE" labels by the vibetest method. This produces the "clean" sample set.
2. For each domain, we then collect ~10 examples of property failures for each
of the properties based on the verified FAIL labeled from the vibetest method.
These create the bug examples for each domain.
3. The clean samples and bug examples for each domain should be stored in the
`synth-data/` directory to be manually inspected and verified before using them for
the rest of the experiments.
4. For each clean sample of each domain we will then sample k~Unif(0,
num-properties) and then choose a random combination of k properties which we
will inject bugs for. We will use an LLM call plus the relevant bug examples of
the property failure to inject the bug into the code and then store the buggy
code for future use. Note that the code is a full repo, so the LLM call will
need to be a call to an agent given the code repo and bug examples with a prompt
to inject a bug. Also have the LLM produce an explanation of what the property
failure that was injected was and save this as the "label" for this repo.
6. The final data should be a directory of repos with injected bugs along with a
file containing all the labels consisting of where the bug was injected and a
description of the bug.

#### Implementation Details / Assumptions (Current)
- Implementation script: `scripts/create_synthetic_bug_dataset.py`
- Output root: `synth-data/` (no writes under `data/`)
- Subcommands:
  - `collect`: builds domain-specific `clean_repos.jsonl` and `bug_examples.jsonl`
  - `plan`: samples per-clean-repo property combinations into `synth-data/injection_plan.jsonl`
  - `inject`: runs agentic bug injection from `injection_plan.jsonl` (requires explicit `--confirm`)
- Current "verified FAIL" sources used by `collect`:
  - Security vulnerabilities: `*_AT-*_recall_scored.jsonl` with `verification.verified_fail=true`
  - ML bugs (Kaggle): human annotation `C` labels for `kaggle_*_AT-*` in `results/human-annotations/human_annotations_long.csv`
  - Citation hallucinations:
    - Primary source is `results/hallucination_refchecker.jsonl`
    - Verified FAILs come from human annotation `C` labels for `hallucination_refchecker` in `results/human-annotations/human_annotations_long.csv`
    - Otherwise optional fallback (`--allow-unverified-hallucination-fallback`) uses unverified RefChecker FAIL examples to avoid empty bug-example sets
- Current "clean" operationalization used by `collect`:
  - For domains with verification labels: no verified FAIL and no INCONCLUSIVE
  - Citation hallucinations (RefChecker-verified): no verified FAIL (INCONCLUSIVE is not treated as disqualifying due sparse verification on FAIL outputs)
  - For hallucination when no RefChecker verification labels are available: strict all-PASS and no INCONCLUSIVE (may produce zero clean repos)
- `collect` supports domain filtering via `--domains` (e.g., run only `security-vuln` and `ml-bugs`)
- `collect` supports subdataset balancing via `--equalize-clean-repos-per-subdataset`
  - This balances *clean repo counts* per subdataset (using the `dataset` field) by random downsampling to the minimum available count
  - Deterministic with `--seed`
- For `security-vuln` / `cwe-bench`, repo paths are resolved against numeric-prefixed directories (e.g., `23_<repo_slug>`) under `data/vuln/cwe-bench/repos/`.
- `plan` supports fixed-size planning per subdataset via `--samples-per-subdataset`
  - If set to `N>0`, planner emits exactly `N` rows per subdataset (`dataset` field), sampling clean repos with replacement
  - If unset/0, planner emits one row per clean repo (previous behavior)
  - Property sampling is dataset-scoped: each clean repo only samples from bug exemplars with the same `dataset` value (e.g., `cwe-bench` repos only use `data/vuln/cwe-bench/properties.md`-derived exemplars, not `bibifi`).
- Bug examples are capped per property by `--max-examples-per-property` (e.g., `3`)
- `inject` is now implemented in `scripts/create_synthetic_bug_dataset.py` and requires explicit `--confirm`
  - Reads `synth-data/injection_plan.jsonl`
  - Runs a VibeTest agent per planned row to perform code edits inside sandbox
  - Executes all selected planned rows in a single `execute_tests(...)` call (Inspect handles internal scheduling)
  - Supports filtering planned rows by domain/dataset for one-split-at-a-time runs:
    - `--domains ml-bugs` and/or `--datasets kaggle_titanic`
  - Requires agent to emit:
    - `/evidence/artifacts/injection_report.json`
    - `/evidence/artifacts/repo.tar.gz` (full modified repo contents from repo root; tar command uses `-C /workspace/repo .`, not a nested subdirectory)
  - Injection prompt explicitly requires in-place edits only (no `.bak`/backup/renamed-clean-copy artifacts)
  - Extracts modified repos into `synth-data/injected/repos/<domain>/<row>_<repo>/<repo_code>/`
    - Example: `synth-data/injected/repos/security-vuln/000001_49/49/`
    - Diff/artifact metadata stays one level above the repo code directory.
    - Extractor preserves tar root layout exactly (no post-extraction flattening), and stores tar member previews in labels metadata for debugging.
  - Computes and stores source-vs-injected diffs per sample:
    - `synth-data/injected/repos/<domain>/<row>_<repo>/injection.diff.patch`
    - `synth-data/injected/repos/<domain>/<row>_<repo>/injection.diff.json`
    - For modified `.ipynb` files, diff text uses `nbdime` (`nbdiff`) when available; otherwise falls back to unified text diff and records fallback notes in diff metadata.
  - Writes labels and metadata to `synth-data/injected/labels.jsonl`
    - `output_repo_path` points to the code directory (`.../<row>_<repo>/<repo_code>/`)
    - `output_sample_path` points to the sample directory (`.../<row>_<repo>/`)
    - Includes explicit per-property ground-truth fields per injected sample:
      - `ground_truth_property_labels`: map `{property_id: 0|1}` where `1=FAIL (bug injected)` and `0=PASS`
      - `ground_truth_violation_descriptions`: map `{property_id: description}` for failed properties
      - `ground_truth_by_property`: row-friendly list with `(property_id, verdict, gt_violation_description)`
  - Writes run summary to `synth-data/injected/injection_run_summary.json`
  - Supports `--limit`, `--offset`, `--model`, `--sandbox`, `--static`, and `--stop-on-error`

##### Inject One Subdataset At A Time (Example: Kaggle Titanic)
```sh
uv run --active python scripts/create_synthetic_bug_dataset.py inject \
  --plan-path synth-data/injection_plan_25_per_subdataset.jsonl \
  --output-root synth-data/injected \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --domains ml-bugs \
  --datasets kaggle_titanic \
  --model openai/gpt-5-mini \
  --sandbox docker \
  --confirm
```

### Experiment
Using the data created as described above, we then run the vibetester as well as
the baselines on the data with an automated LLM judge which determines if the
output (verdict, evidence) is correct by comparing it to the ground truth.

#### Synthetic Experiment Runner (Current)
- Implementation script: `experiments/synthetic.py`
- Supported methods:
  - `vibetest` (AT)
  - `codex` (generic code review + LLM property mapping)
  - `traincheck` (ML bugs only)
  - `codeql` (security-vuln only)
  - `refchecker` (citation-hallucinations only)
- Input:
  - `--labels-path synth-data/injected/labels_<subdataset>.jsonl`
- Output:
  - Standardized JSONL in `results/synthetic/` by default (or `--output-path` override)
  - Includes predictions plus per-property synthetic scoring metadata

##### Synthetic Scoring Semantics
- Binary verdict check:
  - PASS/FAIL prediction must match `ground_truth_property_labels[property_id]`
  - PASS/FAIL mismatch is scored incorrect
- FAIL evidence verification:
  - If prediction is FAIL and GT is FAIL, an LLM verifier checks whether predicted FAIL evidence matches `ground_truth_violation_descriptions[property_id]`
  - Only verifier grade `C` is scored correct
- Abstentions:
  - `INCONCLUSIVE` / `NOT APPLICABLE` are treated as abstentions (not correct)
- Verified FAIL classification metrics:
  - Positive class: GT FAIL labels (`ground_truth_label = 1`)
  - Candidate positive prediction: any FAIL prediction (`predicted_verdict = FAIL`)
  - True positive: candidate FAIL prediction with verifier grade `C` on a GT FAIL
  - False positive: candidate FAIL prediction not counted as TP (including verifier grade `I`)
  - True negative: GT PASS with predicted PASS (abstentions are not counted as TN)
  - Precision / Recall / F1 are computed from these verifier-backed TP/FP/FN definitions

##### Example Commands
Run synthetic AT on one injected subdataset:
```sh
uv run --active python experiments/synthetic.py \
  --method vibetest \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --datasets kaggle_titanic \
  --model openai/gpt-5-mini \
  --sandbox docker \
  --scorer-model openai/gpt-5-mini
```

Run synthetic Codex baseline on one injected subdataset:
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

Run synthetic TrainCheck baseline (ML bugs only):
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

Run synthetic RefChecker baseline (citation hallucinations only):
```sh
uv run --active python experiments/synthetic.py \
  --method refchecker \
  --labels-path synth-data/injected/labels_hallucination.jsonl \
  --domains citation-hallucinations \
  --datasets hallucination \
  --refchecker-timeout 1200 \
  --review-mapper-model openai/gpt-5-mini \
  --scorer-model openai/gpt-5-mini
```

Compute precision/recall/F1 from saved synthetic results:
```sh
uv run --active python scripts/analyze_synthetic_metrics.py \
  results/synthetic/synthetic_kaggle_titanic_AT-gpt-5-mini.jsonl \
  --output-csv results/synthetic/synthetic_metrics.csv
```
