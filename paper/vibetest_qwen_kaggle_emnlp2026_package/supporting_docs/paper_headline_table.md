# Paper Headline Table

Headline metrics use conservative labels. Full-context real Kaggle reaudit is excluded from the main table and reported only as sensitivity analysis.

| Benchmark | Method | Examples | Tests | Coverage | Macro F1 | Out/VF | Audit basis | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic Kaggle | VibeTest static | 0 | 1125 | 0.560 | 0.802 | 19.5K | Raw synthetic ground truth | raw synthetic GT |
| Synthetic Kaggle | VibeTest static | 10 | 1125 | 0.643 | 0.813 | 24.5K | Raw synthetic ground truth | raw synthetic GT |
| Synthetic Kaggle | VibeTest static | 20 | 1125 | 0.562 | 0.837 | 21.8K | Raw synthetic ground truth | raw synthetic GT |
| Synthetic Kaggle | Reviewer Mode 0 |  | 1125 | 0.973 | 0.669 | 3.8K | Raw synthetic ground truth | INCONCLUSIVE treated as PASS for baseline |
| Synthetic Kaggle | Reviewer Mode 1 |  | 1125 | 0.963 | 0.722 | 14.8K | Raw synthetic ground truth | INCONCLUSIVE treated as PASS for baseline |
| Synthetic Kaggle | Reviewer Mode 2 |  | 1125 | 0.982 | 0.711 | 23.1K | Raw synthetic ground truth | INCONCLUSIVE treated as PASS for baseline |
| Synthetic Kaggle | Codex reviewer |  | 1125 | 0.055 | 0.419 |  | Raw synthetic ground truth | sample probe |
| Synthetic Kaggle | TrainCheck |  | 1125 | 0.000 |  |  | Raw synthetic ground truth | all inconclusive in canonical files |
| Real Kaggle | VibeTest static | 0 | 1920 | 0.822 | 0.931 |  | Conservative 15% fail audit | Full-context reaudit is sensitivity only |
| Real Kaggle | VibeTest static | 10 | 1920 | 0.869 | 0.933 |  | Conservative 15% fail audit | Full-context reaudit is sensitivity only |
| Real Kaggle | VibeTest static | 20 | 1920 | 0.871 | 0.938 |  | Conservative 15% fail audit | Full-context reaudit is sensitivity only |
