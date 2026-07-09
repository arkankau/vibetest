# Paper Figure Inventory

| Figure file | Purpose | Data source | Main-text use |
| --- | --- | --- | --- |
| `paper_synthetic_selective_f1_coverage.{png,pdf}` | Synthetic selective macro F1 vs coverage for VibeTest and reviewer modes, plus a single Codex-reviewer full-run point (coverage 0.055, F1 0.419) at bottom-left. TrainCheck has zero usable coverage and is in the table only. Regenerated via `regen_paper_figures.py`. | `supporting_docs/paper_synthetic_selective_f1_coverage.csv` + Codex point from `results/synthetic/synthetic_metrics.csv` | Main result figure for synthetic benchmark. |
| `paper_synthetic_max_error_coverage.{png,pdf}` | Worst pass/fail selective error vs coverage, linear y. Coverage <0.15 omitted (noisy). Regenerated via `regen_paper_figures.py`. | `supporting_docs/paper_synthetic_max_error_coverage.csv` | Reliability/error analysis. |
| `paper_probe_baseline_metrics.{png,pdf}` | Matched five-repository probe (direct prompting, Codex review, VibeTest). NOTE: cut from the current draft (redundant with the probe table); PNG retained for reference. | Values from the probe table in the manuscript. | Not used in main text. |
| `paper_real_conservative_f1_coverage.{png,pdf}` | Conservative real Kaggle macro F1 vs coverage under two-sided thresholds. Coverage <0.30 omitted (unstable audit estimate). Regenerated via `regen_paper_figures.py`. | `supporting_docs/paper_real_conservative_f1_coverage.csv` | Real benchmark figure. |
| `paper_case_score_ecdf.{png,pdf}` | Empirical score distributions for synthetic and real ex0/ex10/ex20. | Synthetic and real Qwen static JSONL files. | Calibration/score-spread figure. |
| `agentic_testing_evidence_verification.png` | Workflow overview figure from the earlier EMNLP-style writeup. | Existing package asset. | Method overview. |

All paper figures are mirrored to `vibetest_qwen_kaggle_emnlp2026_package/figures/` and the underlying CSVs are mirrored to `supporting_docs/` where applicable.
