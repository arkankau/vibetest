# Adaptive Bayesian Sabotage Detection — MLE-Sabotage Results

## Method

We adapt the macro-protocol from [Wen et al. (2024)](https://arxiv.org/abs/2411.17693) for detecting sabotage in the MLE-Sabotage setting, where "aware" agents secretly pursue auxiliary objectives beyond their stated task.

### Pipeline

1. **Stage 0 — Case materialization**: Mix "regular" (clean) and "aware" (sabotage) agent transcripts from Claude 3.7 eval runs into cases of 4 traces each. 50% of cases contain at least one aware trace.
2. **Stage 1 — Per-trace LLM monitoring**: Each transcript is scored independently for sabotage indicators (0.0–1.0) using `gpt-4o-mini` with the `MLE_SABOTAGE_PER_TRACE_PROMPT`.
3. **Stage 2 — Bayesian updating**: Monitor scores are used as observations to update credences over hypotheses and compute calibrated per-trace sabotage posteriors.

### Two Bayesian approaches tested

**A. Direct Bayesian Calibration (`BayesianCalibrator`)**
Learns P(score | benign) and P(score | malicious) directly from labeled data using bucketed histograms with Laplace smoothing. Computes calibrated posteriors with case-level Bayesian context:
```
P(malicious | score) = P(score | malicious) * P(malicious) / P(score)
```
Case-level update: accumulates evidence across all traces in a case to adjust the prior.

**B. Hypothesis-grid Bayesian (`AdaptiveBayesianDetector`)**
The full (r, β) hypothesis grid from the paper, with `r` = misbehavior rate and `β` = sneakiness. Processes traces sequentially with Bayesian credence updates and bootstrapping.

Both methods use leave-one-out cross-validation (fit on 49 cases, score 1 held-out case).

---

## Data

| Parameter | Value |
|-----------|-------|
| Source data | Claude 3.7 React (3 epochs) |
| Regular traces | 66 (clean agent runs) |
| Aware traces | 63 (agents with hidden auxiliary objectives) |
| Cases | 50 (25 positive, 25 negative) |
| Traces per case | 4 |
| Positive rate | 50% of cases |
| Malicious trace rate | 30.5% of all traces |
| Stage 1 scorer | gpt-4o-mini |
| Evaluation | LOO cross-validation, 19 cases with mixed labels |

---

## Results

### Summary Table

| Method | AUROC | AP |
|--------|------:|---:|
| **Raw monitor scores (baseline)** | **0.772** | **0.772** |
| Bayesian calibration (LOO) | 0.759 | 0.752 |
| Bayesian unsupervised | 0.654 | 0.706 |
| Hypothesis-grid Bayesian (wide r) | 0.246 | 0.531 |

### Plots

#### Method Comparison
![AUROC and AP comparison](bayesian_sabotage_comparison.png)

#### Per-Case AUROC Distribution
![Per-case AUROC boxplot](bayesian_sabotage_per_case.png)

#### Monitor Score Distribution
![Score distribution for regular vs aware traces](bayesian_sabotage_score_dist.png)

---

## Key Findings

### 1. The per-trace LLM monitor is already highly discriminative

The gpt-4o-mini monitor achieves strong separation between regular and aware traces:
- Regular traces: mean score 0.376, mostly concentrated at 0.0 and 0.6-0.8
- Aware traces: mean score 0.625, concentrated at 0.7
- Baseline AUROC of 0.772 from raw scores alone

This is qualitatively different from distributed misuse, where most malicious traces have score 0.0 and the monitor alone achieves only 0.57-0.61 AUROC.

### 2. Bayesian calibration preserves baseline quality

The direct calibration approach (BayesianCalibrator) achieves 0.759 AUROC, closely tracking the baseline. This is expected: when the monitor is already well-calibrated and discriminative, Bayesian calibration primarily confirms the existing ranking rather than improving it.

The calibrator is robust across hyperparameters — bucket_size (0.05 vs 0.1), smoothing (0.5-2.0), and prior (0.1-0.5) all yield identical AUROC, indicating the method is stable.

### 3. The (r, β) hypothesis grid from the paper fails on this setting

The hypothesis-grid Bayesian method gets AUROC 0.246 (below random, inverted scores). This occurs because:
- The β parameterization (β = 1 - score) creates a near-perfect correlation between sneakiness and score, producing extremely narrow reference distributions
- With only 61 malicious training traces spread across few score buckets, the distributions are too sparse for reliable likelihood estimation
- The default r-values (designed for 1% positive rate) are far too low for the 30% rate here

This highlights an important limitation: the paper's protocol is designed for deployment sequences with thousands of tasks at low misbehavior rates, not for small-batch classification.

### 4. The unsupervised variant degrades gracefully

Without labels, the unsupervised Bayesian method (which builds synthetic malicious distributions) achieves 0.654 AUROC — above random but below baseline. The synthetic distributions don't match the real data well enough to improve on the raw scores.

### 5. Comparison with distributed misuse setting

| Aspect | MLE-Sabotage | Distributed Misuse |
|--------|-------------|-------------------|
| Traces/case | 4 | 600 |
| Positive rate | 30% | 1% |
| Monitor discriminability | High (mean gap 0.25) | Low (mean gap ~0.05) |
| Bayesian improvement over baseline | -0.013 AUROC | +0.173 AUROC (with clusters) |
| Best method | Raw monitor scores | Bayesian + cluster features |

The Bayesian framework adds the most value when the monitor is weak and there are many traces to aggregate evidence over. When the monitor is already strong (as in sabotage), the framework can only confirm what the monitor already detects.

---

## Configuration Details

| Config | bucket_size | smoothing | prior | mode |
|--------|-----------|-----------|-------|------|
| calibrated_default | 0.1 | 1.0 | 0.3 | LOO |
| calibrated_fine | 0.05 | 0.5 | 0.3 | LOO |
| calibrated_low_prior | 0.1 | 1.0 | 0.1 | LOO |
| calibrated_high_prior | 0.1 | 1.0 | 0.5 | LOO |
| calibrated_smooth | 0.1 | 2.0 | 0.3 | LOO |
| bayesian_wide_r | 0.1 | 1.0 | 0.5 | LOO (r,β) |
| bayesian_unsupervised | 0.1 | 1.0 | 0.5 | unsupervised |

---

## Computational Cost

- **Stage 1 scoring**: 200 traces x 1 gpt-4o-mini call = ~200 API calls (~$0.10)
- **Stage 2 Bayesian**: < 1 second for all 50 cases (pure local computation)
- **Total wall-clock**: ~3 minutes (dominated by API latency)
