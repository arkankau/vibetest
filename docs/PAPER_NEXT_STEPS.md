# Paper Next Steps

This document turns the current Kaggle benchmark artifacts into a paper-facing work plan. The goal is to keep the story defensible: strong results, clear caveats, and no overclaiming from small audit samples.

## Current Claim Stack

### Claim 1: VibeTest improves over reviewer-style baselines on synthetic Kaggle.

Evidence:

- Qwen3.6 VibeTest static ex0/ex10/ex20 reaches best synthetic macro F1 of 0.802/0.813/0.837.
- Reviewer modes 0/1/2 reach 0.669/0.722/0.711 under the same canonical summary.
- TrainCheck is currently an attempted baseline with 0 usable coverage in the canonical files.

Main figure:

- `results/synthetic/figures/synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label_covered_macro_f1_vs_coverage.png`

Caption draft:

> Selective macro F1 versus coverage on the synthetic Kaggle benchmark. VibeTest with Qwen3.6 outperforms reviewer-style baselines across the reported operating points. Adding representative prompt examples improves the VibeTest curve, with ex20 achieving the highest synthetic macro F1 and ex10 achieving the highest coverage among example prompts.

### Claim 2: Synthetic Kaggle is useful but pessimistic because the labels are noisy.

Evidence:

- In a sampled audit of high-confidence synthetic disagreements, 53 cases were usable.
- 29/53 were real Qwen misses.
- 24/53 were ground-truth/property-definition mismatches.
- This means raw synthetic F1 mixes model errors with benchmark-label disagreement.

Main artifact:

- `results/synthetic/openrouter_synthetic_gt_disagreement_audit_15pct_neutral.csv`

Caption/table draft:

> Audit of sampled high-confidence disagreements on synthetic Kaggle. A majority of sampled disagreements are genuine Qwen misses, but a large minority come from mismatches between the synthetic ground truth/property definition and the evidence standard used by VibeTest. This makes synthetic F1 a useful but conservative signal.

### Claim 3: Real Kaggle conservative audit gives around 0.93 macro F1.

Evidence:

- Conservative real Kaggle macro F1 is 0.931/0.933/0.938 for ex0/ex10/ex20.
- Coverage is 0.822/0.869/0.871.
- Full-context spot checks suggest some conservative false-fail penalties were too harsh, but those should stay sensitivity-only.

Main figure:

- `results/figures/real_kaggle_qwen36_k_examples_conservative_macro_f1_vs_coverage.png`

Caption draft:

> Conservative macro F1 versus coverage on the real Kaggle benchmark. VibeTest remains above 0.93 macro F1 across prompt-example settings under the conservative 15% fail audit. Full-context reaudit is reported separately as sensitivity analysis rather than used as the headline metric.

### Claim 4: Score calibration improved enough to support selective plots, but should still be shown.

Evidence:

- Synthetic and real score histograms show how Qwen case scores distribute across example sizes.
- These plots explain why threshold curves behave the way they do.

Main figures:

- `results/synthetic/figures/synthetic_kaggle_qwen36_case_score_bands_histogram.png`
- `results/figures/real_kaggle_qwen36_case_score_histogram.png`

Caption draft:

> Distribution of Qwen case scores across prompt-example settings. The score spread supports threshold-based selective evaluation, while also showing that the model still concentrates many outputs into a limited set of score regions.

## Recommended Paper Structure

1. Introduction
   - Problem: agentic ML pipelines fail in subtle ways that are hard to catch with ordinary static review.
   - Contribution: property-driven agentic testing that maps repository behavior to concrete pass/fail/inconclusive judgments with evidence.

2. Method
   - Property generation or property set.
   - Repository inspection and evidence collection.
   - Verdict and case-score output.
   - Selective evaluation using coverage and thresholding.

3. Synthetic Kaggle Benchmark
   - Explain injected/property-labeled benchmark.
   - Present Qwen ex0/ex10/ex20 versus reviewer baselines.
   - Report TrainCheck as attempted but execution-fragile.

4. Real Kaggle Benchmark
   - Explain conservative fail audit.
   - Present conservative F1 and coverage.
   - Include full-context reaudit only as sensitivity.

5. Ground-Truth Audit
   - Explain why synthetic labels are imperfect.
   - Report 29/53 real Qwen misses and 24/53 GT/property-definition mismatches.
   - Use this to interpret synthetic F1 as pessimistic/noisy.

6. Discussion and Limitations
   - Audit sample sizes are limited.
   - Real Kaggle labels rely on verifier-assisted human-style audit.
   - TrainCheck comparison needs a cleaner controlled execution environment.
   - Synthetic labels need a clearer distinction between strict injected-bug labels and evidence-based property violations.

## Remaining Work

### Must Do

- Write a short methods subsection explaining the verdict schema, especially `PASS`, `FAIL`, `INCONCLUSIVE`, and `case_score`.
- Add a table that combines synthetic, real Kaggle, reviewer baselines, and TrainCheck attempted baseline in one place.
- Add a short paragraph explaining why conservative real Kaggle F1 is the headline, not full-context adjusted F1.
- Add 2-3 concrete synthetic audit examples showing GT/property-definition mismatch.
- Add 2-3 concrete real Kaggle audit examples showing true fail versus false fail.

### Should Do

- Generate one compact CSV/table for all paper headline numbers.
- Add figure captions directly near each plot in a paper draft.
- Re-run or document TrainCheck in a Linux/container setup if we want it to be a stronger baseline.
- Run a larger synthetic GT audit sample if time allows.

### Do Later

- Clean old exploratory figures from the working tree or move them to an archive folder.
- Prepare a PR branch once write access to the Penn-Agentic-Lab repo is settled.
- Convert this note into the actual paper results section.

## Current Risk Register

| Risk | Why it matters | Current mitigation |
|---|---|---|
| Synthetic labels are noisy | Raw synthetic F1 may understate VibeTest quality | Report GT audit and call synthetic F1 conservative/noisy |
| Real Kaggle labels are audited, not exhaustive | Conservative F1 depends on sampled fail audit | Use conservative audit as headline and full-context only as sensitivity |
| Full-context adjusted F1 can look too high | Near-perfect scores are not defensible from a small spot check | Do not headline full-context adjusted numbers |
| TrainCheck has 0 canonical coverage | Baseline comparison may look incomplete | Report as attempted execution-fragile baseline |
| Prompt/example effects are not monotonic | ex20 has best synthetic F1 but ex10 has higher coverage | Report both and avoid saying more examples always help |

## Immediate Next Artifact

The next best artifact is a single paper-headline table with columns:

- Benchmark
- Method
- Example count
- Tests
- Coverage
- Macro F1
- Audit basis
- Notes

This table should be generated from `results/benchmark_summary.csv` plus the real full-context sensitivity file, but the main rows should use conservative metrics.
