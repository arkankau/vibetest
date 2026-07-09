# Frozen Claims and Canonical Results

This document pins the paper story so individual plots do not keep changing the thesis.

## Main Claim

VibeTest turns natural-language ML-pipeline properties into evidence-backed repository tests. The method should be evaluated with abstention, score-threshold curves, and audits of property interpretation.

## Claims We Can Defend

- VibeTest with Qwen3.6 produces usable selective predictions on synthetic and real Kaggle-style ML repositories.
- Prompt examples change the coverage/accuracy tradeoff: ex10 gives the strongest synthetic coverage, while ex20 gives the strongest synthetic macro F1 point estimate. NOTE: 95% bootstrap CIs for ex0/ex10/ex20 macro F1 ([0.77,0.83]/[0.78,0.84]/[0.81,0.87]) overlap, so the ex ranking is NOT statistically distinguishable — do not claim ex20 is significantly best. VibeTest vs reviewer-mode CIs do not overlap (gain over reviewers is robust). Computed by bootstrap_cis.py.
- Real Kaggle performance should be reported conservatively because labels come from sampled fail audits, not exhaustive ground truth.
- Synthetic raw F1 is useful but noisy because injected labels and evidence-based property judgments can disagree.
- The direct-property and Codex-style runs are useful probes, but the same-model direct-property baseline remains the cleanest missing comparison.
- Codex reviewer, run on the full synthetic benchmark, is execution/format-fragile at scale: it commits a parseable verdict on only 5.5% of cases (macro F1 0.419 on that sliver, mostly by predicting FAIL). This is a full-benchmark baseline row, not just a probe. Its higher apparent coverage on the 5-repo probe (0.44) comes from the reviewer inconclusive->pass folding used there.
- There is no full-benchmark direct-property run (probe only), and no real-Kaggle Codex/direct numbers (real GT is a fail audit of VibeTest's own predictions).

## Claims We Should Not Overstate

- Do not claim complete real-Kaggle recall; the real audit samples predicted failures.
- Do not claim the full-context real audit proves near-perfect F1; it is a sensitivity check.
- Do not claim TrainCheck is intrinsically worse; our available canonical files show execution-fragile zero coverage.
- Do not claim every synthetic disagreement is a model error; the audit shows many are property-boundary mismatches.

## Note on coverage convention (paper table vs. this doc)

The `coverage` column below is the **base non-inconclusive rate** (threshold 0),
and `best_selective_f1` is the **best macro F1 over the score-threshold sweep**,
which occurs at a *lower* coverage. These are two different operating points.

The submission draft (`vibetest_qwen_kaggle_emnlp2026_pips_style.tex`, Table 1)
was corrected to avoid pairing them misleadingly: it now reports coverage AND
macro F1 at the **single peak-selective-F1 operating point** (the apex of each
risk-coverage curve). Peak-F1 coverage there is 0.560 / 0.643 / 0.562 for
ex0/ex10/ex20 (vs. base coverage 0.773 / 0.903 / 0.798 here), and reviewer
coverage is 0.973 / 0.963 / 0.982 (the figure convention, folding INCONCLUSIVE
into PASS), not the native 0.264 / 0.321 / 0.308. F1 values are unchanged.

## Provenance of per-case data (IMPORTANT)

There are TWO FP8 Qwen3.6 synthetic runs, both real, ~85% verdict agreement:
1. **Headline run** = `..._static_examples{0,10,20}_case_score_bands.jsonl` on branch **`ml-vibetest-a`**. The ex0 bands files reproduce the paper's 0.773 base coverage EXACTLY. Use these for RQ1, coverage, selective F1, and `tab:models` (Qwen native: titanic 0.720/0.762, nlp 0.835/0.795, diabetic 0.765/0.749, mean 0.773/0.769; ~2.9k out-tok/prop). Reproduced by `model_dataset_cost.py`.
2. **Evidence-verified run** = `..._case-score.jsonl` on branch `qwen36-prompt-examples` (has `scoring.fail_evidence_verified_*`, scorer `openai/gpt-5.4-mini`). This is the ONLY run with per-verdict evidence checks, so RQ4 evidence quality (306/335 = 0.913) comes from here. It is a separate FP8 pass, same model/benchmark.

Correction made: `tab:models` originally used run #2, which over-counts output tokens ~2x (5.9k vs the correct 2.9k). Fixed to run #1. The abstract's "half the output-token cost" claim was wrong and was removed (backbones are comparable, ~3k each).

## Canonical Headline Rows

| section | name | tests | coverage | best_selective_f1 | audited_macro_f1 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic | qwen_static_ex0 | 1125 | 0.7733333333333333 | 0.8021978021978022 |  | raw synthetic GT |
| synthetic | qwen_static_ex10 | 1125 | 0.9031111111111112 | 0.813498535481294 |  | raw synthetic GT |
| synthetic | qwen_static_ex20 | 1125 | 0.7982222222222223 | 0.8371875772518509 |  | raw synthetic GT |
| synthetic | reviewer_mode_0 | 1125 | 0.264 | 0.6691030464485489 |  | INCONCLUSIVE treated as PASS for baseline |
| synthetic | reviewer_mode_1 | 1125 | 0.3208888888888889 | 0.7223738653841989 |  | INCONCLUSIVE treated as PASS for baseline |
| synthetic | reviewer_mode_2 | 1125 | 0.3075555555555556 | 0.7111322569851096 |  | INCONCLUSIVE treated as PASS for baseline |
| synthetic | codex_reviewer_full | 1125 | 0.0551 | 0.4194 |  | full run; native committed operating point (commits on 5.5% of cases, mostly predicting FAIL); computed from results/synthetic/synthetic_metrics.csv codex kaggle rows |
| synthetic | traincheck | 1125 | 0.0 |  |  | all inconclusive in canonical files |
| real_kaggle | qwen_static_ex0 | 1920 | 0.821875 |  | 0.930516647854022 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |
| real_kaggle | qwen_static_ex10 | 1920 | 0.86875 |  | 0.9332050589839254 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |
| real_kaggle | qwen_static_ex20 | 1920 | 0.8713541666666667 |  | 0.937603764292305 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |

## Audit Rows

| name | tests | notes |
| --- | --- | --- |
| synthetic_gt_disagreement | 54 | {'GT_CORRECT_QWEN_WRONG': 29, 'GT_WRONG_QWEN_CORRECT': 24, 'PARSE_ERROR': 1} |
| real_fail_sample | 152 | {'TRUE_FAIL': 122, 'FALSE_FAIL': 30} |
