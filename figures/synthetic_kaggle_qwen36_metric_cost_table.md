# Synthetic Kaggle Qwen3.6 Metric/Cost Table

F1 is best macro-F1 over the plotted thresholds. Response rate is the non-INCONCLUSIVE rate at that threshold. Cost is saved agent-token usage per synthetic repo/case; Qwen USD pricing is not stored in these results. Mapper/scorer usage is not included in saved agent-token totals.

## Standard abstention semantics

| Method | Best threshold | Macro F1 | FAIL F1 | PASS F1 | Response rate | Avg agent tokens/case | Relative cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| VibeTest static | 0.75 | 0.674 | 0.643 | 0.705 | 0.849 | 3.28M | 1.00x |
| VibeTest dynamic | 0.75 | 0.710 | 0.651 | 0.769 | 0.975 | 13.11M | 4.00x |
| TrainCheck | 0.00 | N/A | N/A | N/A | 0.000 | 0 | 0.00x |
| Reviewer mode 0 | 0.00 | 0.268 | 0.504 | 0.033 | 0.264 | 379.9K | 0.12x |
| Reviewer mode 1 | 0.00 | 0.295 | 0.579 | 0.010 | 0.321 | 2.44M | 0.74x |
| Reviewer mode 2 | 0.00 | 0.314 | 0.585 | 0.043 | 0.308 | 5.28M | 1.61x |

## Reviewer INCONCLUSIVE counted as PASS

| Method | Best threshold | Macro F1 | FAIL F1 | PASS F1 | Response rate | Avg agent tokens/case | Relative cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| VibeTest static | 0.75 | 0.674 | 0.643 | 0.705 | 0.849 | 3.28M | 1.00x |
| VibeTest dynamic | 0.75 | 0.710 | 0.651 | 0.769 | 0.975 | 13.11M | 4.00x |
| TrainCheck | 0.00 | N/A | N/A | N/A | 0.000 | 0 | 0.00x |
| Reviewer mode 0 | 0.00 | 0.631 | 0.504 | 0.758 | 1.000 | 379.9K | 0.12x |
| Reviewer mode 1 | 0.00 | 0.677 | 0.579 | 0.776 | 1.000 | 2.44M | 0.74x |
| Reviewer mode 2 | 0.00 | 0.681 | 0.585 | 0.777 | 1.000 | 5.28M | 1.61x |
