# Commit Boundaries

This workspace contains both paper-ready artifacts and older exploratory outputs. To keep the handoff clean, commits should be split by purpose.

## Commit 1: Paper Docs and Headline Tables

Include:

- `docs/PAPER_RESULTS_DRAFT.md`
- `docs/PAPER_METHODS_EVAL_DRAFT.md`
- `docs/PAPER_NEXT_STEPS.md`
- `docs/KAGGLE_BENCHMARK_RESULTS.md`
- `docs/AUDIT_EXAMPLES.md`
- `docs/SUBMISSION_PACKAGE.md`
- `results/paper_headline_table.md`
- `results/paper_headline_table.csv`
- `scripts/make_paper_headline_table.py`

Purpose:

Document the current results story, methods framing, audit examples, and paper headline table.

## Commit 2: Canonical Summaries and Sensitivity Tables

Include:

- `results/benchmark_manifest.json`
- `results/benchmark_summary.md`
- `results/benchmark_summary.csv`
- `results/real_kaggle_full_context_sensitivity_summary.md`
- `results/real_kaggle_full_context_sensitivity_summary.csv`
- `scripts/summarize_benchmark_manifest.py`
- `scripts/summarize_real_kaggle_full_context_sensitivity.py`

Purpose:

Track reproducible summary artifacts and the scripts that regenerate them.

## Commit 3: Plot Script Updates

Include:

- `scripts/plot_real_kaggle_qwen_f1_coverage.py`
- `scripts/plot_real_kaggle_qwen_dual_threshold_f1_coverage.py`

Purpose:

Make conservative real Kaggle audit metrics the main plotted result and keep full-context adjusted values as sensitivity only.

## Commit 4: Audit Prompt and Synthetic Audit Utilities

Include only if we want to preserve the verifier/audit workflow changes:

- `docs/synthetic_ground_truth_audit_prompt.md`
- `scripts/analyze_synthetic_high_conf_errors.py`
- `scripts/run_openrouter_synthetic_gt_audit.py`
- `scripts/analyze_synthetic_accepted_predictions.py`
- `results/synthetic/synthetic_qwen_high_confidence_errors_for_audit.csv`
- `results/synthetic/openrouter_synthetic_gt_disagreement_audit_15pct_neutral.csv`
- `results/synthetic/synthetic_qwen_accepted_predictions_for_audit.csv`

Purpose:

Preserve the synthetic ground-truth audit workflow and sampled audit data.

## Commit 5: TrainCheck Attempt Logs

Include only if provenance matters:

- `vibetest/baselines/traincheck.py`
- `results/synthetic/synthetic_kaggle_titanic_traincheck_smoke*.jsonl`
- `results/traincheck_synthetic_smoke*/`

Purpose:

Document the TrainCheck attempted baseline and execution issues. This should be kept separate because it contains environment-specific smoke-test output.

## Leave Out of Main Paper Commit

Leave older exploratory plots, temporary smoke CSVs, and partial OpenRouter smoke outputs unstaged unless they are needed for reproducibility. The submission package should point reviewers to canonical figures, summaries, and audit files only.
