# Benchmark Summary

Canonical result summary generated from `results/benchmark_manifest.json`.

- Manifest: `results/benchmark_manifest.json`
- CSV: `results/benchmark_summary.csv`
- Missing manifest files: 0

## Results

| Section | Name | Tests | Coverage | F1 / Best F1 | Notes |
|---|---:|---:|---:|---:|---|
| synthetic | qwen_static_ex0 | 1125 | 0.773 | 0.802 | raw synthetic GT |
| synthetic | qwen_static_ex10 | 1125 | 0.903 | 0.813 | raw synthetic GT |
| synthetic | qwen_static_ex20 | 1125 | 0.798 | 0.837 | raw synthetic GT |
| synthetic | reviewer_mode_0 | 1125 | 0.264 | 0.669 | INCONCLUSIVE treated as PASS for baseline |
| synthetic | reviewer_mode_1 | 1125 | 0.321 | 0.722 | INCONCLUSIVE treated as PASS for baseline |
| synthetic | reviewer_mode_2 | 1125 | 0.308 | 0.711 | INCONCLUSIVE treated as PASS for baseline |
| synthetic | traincheck | 1125 | 0.000 |  | all inconclusive in canonical files |
| real_kaggle | qwen_static_ex0 | 1920 | 0.822 | 0.931 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |
| real_kaggle | qwen_static_ex10 | 1920 | 0.869 | 0.933 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |
| real_kaggle | qwen_static_ex20 | 1920 | 0.871 | 0.938 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |
| audit | synthetic_gt_disagreement | 54 |  |  | {'GT_CORRECT_QWEN_WRONG': 29, 'GT_WRONG_QWEN_CORRECT': 24, 'PARSE_ERROR': 1} |
| audit | real_fail_sample | 152 |  |  | {'TRUE_FAIL': 122, 'FALSE_FAIL': 30} |

## Audit Files

- Synthetic GT disagreement audit: `results/synthetic/openrouter_synthetic_gt_disagreement_audit_15pct_neutral.csv`
- Real fail audit: `results/openrouter_fail_human_audit_sample15_gpt5mini.csv`
- Real full-context false-fail reaudit: `results/openrouter_false_fail_reaudit_full_context.csv`

## TrainCheck Note

The canonical TrainCheck synthetic files are all `INCONCLUSIVE` in this workspace. Local smoke runs got past several Windows/runtime issues, but target notebooks can still crash before useful invariant checks, so TrainCheck should be reported as an execution-fragile attempted baseline rather than a central competitor.
