# Submission Package

This file indexes the artifacts that are currently ready for PI/paper review.

## Main Draft Docs

| Artifact | Purpose |
|---|---|
| `docs/PAPER_RESULTS_DRAFT.md` | Main draft results section with tables, figure references, and interpretation. |
| `docs/PAPER_METHODS_EVAL_DRAFT.md` | Draft methods/evaluation wording for verdicts, scoring, selective evaluation, audits, and baselines. |
| `docs/KAGGLE_BENCHMARK_RESULTS.md` | Concise benchmark result summary. |
| `docs/AUDIT_EXAMPLES.md` | Representative synthetic and real audit examples. |
| `docs/PAPER_NEXT_STEPS.md` | Claim stack, figure captions, risk register, and remaining work. |

## Main Figures

| Figure | Use |
|---|---|
| `results/synthetic/figures/synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label_covered_macro_f1_vs_coverage.png` | Main synthetic F1-vs-coverage result comparing Qwen examples, reviewers, and TrainCheck. |
| `results/synthetic/figures/synthetic_kaggle_qwen36_symmetric_max_selective_error_vs_coverage_log_y_inverted.png` | Synthetic max-error selective plot. |
| `figures/direct_property_vs_vibetest_titanic_repo5_summary.png` | Sampled direct-property baseline probe compared with VibeTest on the same five Titanic synthetic repositories. |
| `results/figures/real_kaggle_qwen36_k_examples_conservative_macro_f1_vs_coverage.png` | Main conservative real Kaggle F1-vs-coverage result. |
| `results/synthetic/figures/synthetic_kaggle_qwen36_case_score_bands_histogram.png` | Synthetic Qwen score histogram. |
| `results/figures/real_kaggle_qwen36_case_score_histogram.png` | Real Kaggle Qwen score histogram. |

## Appendix/Sensitivity Artifacts

| Artifact | Use |
|---|---|
| `results/figures/real_kaggle_qwen36_conservative_dual_threshold_f1_vs_coverage.png` | Optional real Kaggle dual-threshold selective plot. |
| `results/real_kaggle_full_context_sensitivity_summary.md` | Full-context false-fail reaudit sensitivity table. |
| `results/real_kaggle_full_context_sensitivity_summary.csv` | Machine-readable sensitivity table. |
| `results/synthetic/figures/synthetic_kaggle_qwen36_case_score_bands_exact_score_frequency.png` | More detailed synthetic score-frequency figure. |

## Tables and Reproducibility

| Artifact | Use |
|---|---|
| `results/paper_headline_table.md` | Paper-ready headline table. |
| `results/paper_headline_table.csv` | Machine-readable headline table. |
| `results/benchmark_summary.md` | Canonical generated benchmark summary. |
| `results/benchmark_summary.csv` | Machine-readable canonical benchmark summary. |
| `results/benchmark_manifest.json` | Canonical input file manifest for the summary. |
| `supporting_docs/direct_property_vs_vibetest_titanic_repo5_summary.csv` | Machine-readable direct-property baseline probe summary. |
| `supporting_docs/direct_property_vs_vibetest_titanic_repo5_report.md` | Markdown direct-property baseline probe report. |
| `supporting_docs/direct_property_qwen_flash_vs_vibetest_titanic_repo5_summary.csv` | Machine-readable Qwen3.6 Flash direct-property probe summary on the matched Titanic subset. |
| `supporting_docs/direct_property_qwen_flash_vs_vibetest_titanic_repo5_report.md` | Markdown Qwen3.6 Flash direct-property probe report. |
| `supporting_docs/synthetic_kaggle_titanic_codex_reviewer_openrouter-gpt-4.1-mini_repo5.jsonl` | Codex reviewer GPT-4.1-mini practical-tool probe on the same five Titanic synthetic repositories. |
| `scripts/make_paper_headline_table.py` | Regenerates the headline table from `results/benchmark_summary.csv`. |
| `scripts/summarize_benchmark_manifest.py` | Regenerates the benchmark summary from the manifest. |
| `scripts/summarize_real_kaggle_full_context_sensitivity.py` | Regenerates the full-context sensitivity summary. |

## Audit Inputs

| Artifact | Use |
|---|---|
| `results/synthetic/openrouter_synthetic_gt_disagreement_audit_15pct_neutral.csv` | Synthetic high-confidence disagreement audit. |
| `results/openrouter_fail_human_audit_sample15_gpt5mini.csv` | Conservative real Kaggle fail audit. |
| `results/openrouter_false_fail_reaudit_full_context.csv` | Full-context reaudit of conservative false fails. |

## Not Mainline

The workspace still contains older smoke-test outputs, exploratory figures, and intermediate audit files. They are useful for provenance but should not be presented as main paper artifacts unless explicitly moved into the package above.
