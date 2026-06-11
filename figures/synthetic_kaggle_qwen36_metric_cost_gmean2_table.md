# Synthetic Kaggle Qwen3.6 Metric/Cost/G-Mean Table

For VibeTest and TrainCheck, INCONCLUSIVE responses are abstentions and are excluded from TPR/TNR. For reviewer baselines only, INCONCLUSIVE is counted as PASS. g-mean^2 is TPR * TNR, where TPR is evidence-matched FAIL recall and TNR is GT-PASS recall. Cost is saved agent-token usage per synthetic repo/case; Qwen USD pricing and mapper/scorer usage are not stored in these JSONLs.

| Method | Best threshold | Macro F1 | Response rate | TPR | TNR | g-mean^2 | Avg agent tokens/case | Relative cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VibeTest static | 0.75 | 0.674 | 0.849 | 0.641 | 0.832 | 0.533 | 3.28M | 1.00x |
| VibeTest dynamic | 0.75 | 0.710 | 0.975 | 0.602 | 0.840 | 0.506 | 13.11M | 4.00x |
| TrainCheck | 0.00 | N/A | 0.000 | N/A | N/A | N/A | 0 | 0.00x |
| Reviewer mode 0 | 0.00 | 0.631 | 1.000 | 0.386 | 0.919 | 0.354 | 379.9K | 0.12x |
| Reviewer mode 1 | 0.00 | 0.677 | 1.000 | 0.483 | 0.892 | 0.431 | 2.44M | 0.74x |
| Reviewer mode 2 | 0.00 | 0.681 | 1.000 | 0.474 | 0.910 | 0.431 | 5.28M | 1.61x |
