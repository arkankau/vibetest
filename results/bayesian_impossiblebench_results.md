# Adaptive Bayesian Detection — ImpossibleBench Results

## Method

Same Bayesian protocol as the MLE-Sabotage and distributed misuse experiments, applied to the ImpossibleBench reward hacking setting (Patronus trace dataset).

### Pipeline
1. **Stage 1**: Per-trace scoring (pre-computed `initial_scores.json` in each case)
2. **Stage 2**: Bayesian updating (calibrated and hypothesis-grid variants, LOO)

---

## Data

| Parameter | Value |
|-----------|-------|
| Dataset | safety_patronus_trace |
| Cases | 50 (25 positive, 25 negative) |
| Traces per case | 4 |
| Hacked traces | 40 / 200 (20.0%) |
| Hacking type | Reward hacking (shortcuts, test exploitation) |

---

## Results

| Method | AUROC | AP |
|--------|------:|---:|
| Raw monitor scores (baseline) | 0.767 | **0.717** |
| **Bayesian calibration fine (LOO)** | **0.775** | 0.707 |
| Bayesian calibration default (LOO) | 0.770 | 0.700 |
| Bayesian unsupervised | 0.770 | 0.700 |
| Hypothesis-grid Bayesian (wide r) | 0.760 | 0.693 |

### Plots

#### Method Comparison
![AUROC and AP comparison](bayesian_impossiblebench_comparison.png)

#### Per-Case AUROC Distribution
![Per-case AUROC boxplot](bayesian_impossiblebench_per_case.png)

#### Monitor Score Distribution
![Score distribution for clean vs hacked traces](bayesian_impossiblebench_score_dist.png)

---

## Key Findings

### 1. Per-trace monitor scores provide moderate discrimination

Unlike the previous (wrong) ImpossibleBench dataset where the monitor returned all zeros, this Patronus dataset shows real signal:
- Clean traces: mean score 0.158 (82% at score < 0.1)
- Hacked traces: mean score 0.510 (20% at score >= 0.9)
- Baseline AUROC: 0.767

### 2. Bayesian calibration provides a small improvement

The fine-bucket calibrator achieves **0.775 AUROC** (+0.008 over baseline). The improvement is modest because:
- The monitor is already reasonably well-calibrated
- With only 4 traces per case, the case-level Bayesian update has limited context
- The score gap (0.158 vs 0.510) is moderate — enough signal for the baseline but limited room for Bayesian amplification

### 3. Consistent across all three settings

| Setting | Monitor AUROC | Best Bayesian AUROC | Delta |
|---------|:---:|:---:|:---:|
| Distributed misuse (bg=100x) | 0.574 | **0.747** | **+0.173** |
| MLE-Sabotage | 0.772 | 0.759 | -0.013 |
| ImpossibleBench | 0.767 | 0.775 | +0.008 |

The pattern holds: the Bayesian protocol adds the most value when the per-trace monitor is weak and there are many traces to aggregate over (distributed misuse). With only 4 traces per case and a moderate/strong monitor, the framework can at best match the baseline.

---

## Configuration Details

| Config | bucket_size | smoothing | prior | mode | AUROC |
|--------|-----------|-----------|-------|------|------:|
| calibrated_fine | 0.05 | 0.5 | 0.3 | LOO | 0.775 |
| calibrated_default | 0.1 | 1.0 | 0.3 | LOO | 0.770 |
| calibrated_low_prior | 0.1 | 1.0 | 0.1 | LOO | 0.770 |
| calibrated_high_prior | 0.1 | 1.0 | 0.5 | LOO | 0.760 |
| bayesian_wide_r | 0.1 | 1.0 | 0.5 | LOO (r,β) | 0.760 |
| bayesian_unsupervised | 0.1 | 1.0 | 0.5 | unsupervised | 0.770 |
