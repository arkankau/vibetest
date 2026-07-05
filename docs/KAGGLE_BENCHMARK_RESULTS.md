# Kaggle Benchmark Results

This note summarizes the current benchmark story for the Qwen3.6 VibeTest runs, the real Kaggle audit, and the attempted TrainCheck baseline. The headline numbers use the conservative labels currently in `results/benchmark_summary.md`.

## Synthetic Benchmark Results

On the synthetic Kaggle benchmark, Qwen3.6 VibeTest improves over the reviewer baselines under the current selective-evaluation setup.

| Method | Tests | Coverage | Best macro F1 |
|---|---:|---:|---:|
| VibeTest static ex0 | 1125 | 0.773 | 0.802 |
| VibeTest static ex10 | 1125 | 0.903 | 0.813 |
| VibeTest static ex20 | 1125 | 0.798 | 0.837 |
| Reviewer mode 0 | 1125 | 0.264 | 0.669 |
| Reviewer mode 1 | 1125 | 0.321 | 0.722 |
| Reviewer mode 2 | 1125 | 0.308 | 0.711 |
| TrainCheck | 1125 | 0.000 | n/a |

The ex10 prompt gives the strongest coverage among the Qwen example prompts, while ex20 gives the best synthetic macro F1. The important pattern is that adding representative examples improves the VibeTest curve without collapsing coverage.

Core figures:

- `results/synthetic/figures/synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label_covered_macro_f1_vs_coverage.png`
- `results/synthetic/figures/synthetic_kaggle_qwen36_symmetric_max_selective_error_vs_coverage_log_y_inverted.png`
- `results/synthetic/figures/synthetic_kaggle_qwen36_case_score_bands_histogram.png`
- `results/synthetic/figures/synthetic_kaggle_qwen36_case_score_bands_exact_score_frequency.png`

## Real Kaggle Results

For the real Kaggle benchmark, the conservative fail audit gives roughly 0.93 macro F1 across all example sizes.

| Method | Tests | Coverage | Conservative macro F1 |
|---|---:|---:|---:|
| VibeTest static ex0 | 1920 | 0.822 | 0.931 |
| VibeTest static ex10 | 1920 | 0.869 | 0.933 |
| VibeTest static ex20 | 1920 | 0.871 | 0.938 |

The real-data result should be presented as strong but not near-perfect. A full-context spot check suggests that some false-fail penalties in the conservative audit were too harsh, but that analysis should stay as sensitivity analysis rather than the headline number.

Core figures and sensitivity artifacts:

- `results/figures/real_kaggle_qwen36_k_examples_conservative_macro_f1_vs_coverage.png`
- `results/figures/real_kaggle_qwen36_conservative_dual_threshold_f1_vs_coverage.png`
- `results/figures/real_kaggle_qwen36_case_score_histogram.png`
- `results/real_kaggle_full_context_sensitivity_summary.md`

## Synthetic Ground-Truth Audit

The synthetic benchmark is useful, but the current labels appear pessimistic/noisy. In the sampled high-confidence disagreements, 53 cases were usable after removing one parse error:

| Audit outcome | Count |
|---|---:|
| Real Qwen miss | 29 |
| Ground-truth/property-definition mismatch | 24 |

So a little over half of the sampled disagreements are real Qwen misses, but a large minority are cases where the synthetic ground truth or property definition does not cleanly match the evidence standard used by VibeTest. This means the synthetic F1 should be treated as a conservative stress-test number, not as a perfectly clean estimate of model quality.

The main interpretation is not that the synthetic benchmark is unusable. It is that raw synthetic F1 mixes two effects: actual model mistakes and dataset/property-definition mismatch. That helps explain why synthetic error can remain high even at low coverage, while real Kaggle appears stronger after audit.

Audit artifact:

- `results/synthetic/openrouter_synthetic_gt_disagreement_audit_15pct_neutral.csv`

## TrainCheck Attempted Baseline

TrainCheck was attempted as a baseline on the synthetic benchmark, but the canonical files in this workspace are all `INCONCLUSIVE`, giving 0 coverage. Local smoke runs got past several Windows/runtime issues, but target notebooks can still fail before useful invariant checks are reached.

TrainCheck should therefore be reported as an attempted baseline with execution fragility on this benchmark, not as a central competitor unless we rerun it under a more controlled Linux/container setup.

## Takeaways

The current story is:

- Qwen3.6 VibeTest beats the reviewer-style baselines on synthetic Kaggle under the current selective-evaluation setup.
- Prompt examples help, especially ex10 for coverage and ex20 for synthetic macro F1.
- Real Kaggle conservative audit is around 0.93 macro F1, with ex20 slightly ahead.
- Full-context spot checks suggest the conservative real audit may understate performance, but the paper should avoid claiming near-perfect F1 from the small sensitivity sample.
- Synthetic F1 is still useful, but it is pessimistic because sampled disagreements include both real Qwen misses and ground-truth/property-definition mismatches.
