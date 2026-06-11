# Synthetic Kaggle Qwen3.6 Selective Classification Summary

Primary metric is abstention-penalized macro F1: INCONCLUSIVE counts as a missed classification for the true class. Response rate is PASS/FAIL coverage. Selective balanced accuracy is computed only on non-INCONCLUSIVE outputs, so it measures reliability when the method answers. FAIL correctness requires evidence-match. Cost is saved agent-token usage per synthetic repo/case; Qwen USD pricing and mapper/scorer usage are not stored in these JSONLs.

| Method | Interpretation | Macro F1 | Response rate | Selective bal. acc. | FAIL precision | FAIL recall | Avg tokens/case | Relative cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VibeTest static | standard abstention | 0.674 | 0.849 | 0.736 | 0.741 | 0.567 | 3.28M | 1.00x |
| VibeTest dynamic | standard abstention | 0.710 | 0.975 | 0.721 | 0.726 | 0.590 | 13.11M | 4.00x |
| TrainCheck | standard abstention | N/A | 0.000 | N/A | N/A | 0.000 | 0 | 0.00x |
| Reviewer mode 0 | reviewer abstaining | 0.268 | 0.264 | 0.517 | 0.725 | 0.386 | 379.9K | 0.12x |
| Reviewer mode 1 | reviewer abstaining | 0.295 | 0.321 | 0.461 | 0.723 | 0.483 | 2.44M | 0.74x |
| Reviewer mode 2 | reviewer abstaining | 0.314 | 0.308 | 0.550 | 0.764 | 0.474 | 5.28M | 1.61x |
| Reviewer mode 0 | no finding = PASS | 0.631 | 1.000 | 0.652 | 0.725 | 0.386 | 379.9K | 0.12x |
| Reviewer mode 1 | no finding = PASS | 0.677 | 1.000 | 0.687 | 0.723 | 0.483 | 2.44M | 0.74x |
| Reviewer mode 2 | no finding = PASS | 0.681 | 1.000 | 0.692 | 0.764 | 0.474 | 5.28M | 1.61x |
