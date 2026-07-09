# Paper Figure Inventory

| Figure file | Purpose | Data source | Main-text use |
| --- | --- | --- | --- |
| `paper_synthetic_selective_f1_coverage.{png,pdf}` | Synthetic selective macro F1 vs coverage for VibeTest and reviewer modes. TrainCheck has zero usable coverage and is reported in the table rather than as a visible curve. | `results/synthetic/figures/synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label.csv` | Main result figure for synthetic benchmark. |
| `paper_synthetic_max_error_coverage.{png,pdf}` | Worst pass/fail selective error vs coverage on a linear y-axis. | `results/synthetic/figures/synthetic_kaggle_qwen36_symmetric_max_selective_error_vs_coverage_log_y_inverted.csv` | Reliability/error analysis. |
| `paper_probe_baseline_metrics.{png,pdf}` | Matched five-repository probe comparing direct prompting, Codex review, and VibeTest. | Values from the probe table in the manuscript. | Baseline comparison figure. |
| `paper_real_conservative_f1_coverage.{png,pdf}` | Conservative real Kaggle macro F1 vs coverage under two-sided thresholds. | Real Qwen static outputs plus `openrouter_fail_human_audit_sample15_gpt5mini.csv` | Real benchmark figure. |
| `paper_case_score_ecdf.{png,pdf}` | Empirical score distributions for synthetic and real ex0/ex10/ex20. | Synthetic and real Qwen static JSONL files. | Calibration/score-spread figure. |
| `agentic_testing_evidence_verification.png` | Workflow overview figure from the earlier EMNLP-style writeup. | Existing package asset. | Method overview. |

All paper figures are mirrored to `vibetest_qwen_kaggle_emnlp2026_package/figures/` and the underlying CSVs are mirrored to `supporting_docs/` where applicable.
