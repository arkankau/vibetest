# Frozen Claims and Canonical Results

This document pins the paper story so individual plots do not keep changing the thesis.

## Main Claim

VibeTest turns natural-language ML-pipeline properties into evidence-backed repository tests. The method should be evaluated with abstention, score-threshold curves, and audits of property interpretation.

## Claims We Can Defend

- VibeTest with Qwen3.6 produces usable selective predictions on synthetic and real Kaggle-style ML repositories.
- Prompt examples change the coverage/accuracy tradeoff: ex10 gives the strongest synthetic coverage, while ex20 gives the strongest synthetic macro F1.
- Real Kaggle performance should be reported conservatively because labels come from sampled fail audits, not exhaustive ground truth.
- Synthetic raw F1 is useful but noisy because injected labels and evidence-based property judgments can disagree.
- The direct-property and Codex-style runs are useful probes, but the same-model direct-property baseline remains the cleanest missing comparison.

## Claims We Should Not Overstate

- Do not claim complete real-Kaggle recall; the real audit samples predicted failures.
- Do not claim the full-context real audit proves near-perfect F1; it is a sensitivity check.
- Do not claim TrainCheck is intrinsically worse; our available canonical files show execution-fragile zero coverage.
- Do not claim every synthetic disagreement is a model error; the audit shows many are property-boundary mismatches.

## Canonical Headline Rows

| section | name | tests | coverage | best_selective_f1 | audited_macro_f1 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic | qwen_static_ex0 | 1125 | 0.7733333333333333 | 0.8021978021978022 |  | raw synthetic GT |
| synthetic | qwen_static_ex10 | 1125 | 0.9031111111111112 | 0.813498535481294 |  | raw synthetic GT |
| synthetic | qwen_static_ex20 | 1125 | 0.7982222222222223 | 0.8371875772518509 |  | raw synthetic GT |
| synthetic | reviewer_mode_0 | 1125 | 0.264 | 0.6691030464485489 |  | INCONCLUSIVE treated as PASS for baseline |
| synthetic | reviewer_mode_1 | 1125 | 0.3208888888888889 | 0.7223738653841989 |  | INCONCLUSIVE treated as PASS for baseline |
| synthetic | reviewer_mode_2 | 1125 | 0.3075555555555556 | 0.7111322569851096 |  | INCONCLUSIVE treated as PASS for baseline |
| synthetic | traincheck | 1125 | 0.0 |  |  | all inconclusive in canonical files |
| real_kaggle | qwen_static_ex0 | 1920 | 0.821875 |  | 0.930516647854022 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |
| real_kaggle | qwen_static_ex10 | 1920 | 0.86875 |  | 0.9332050589839254 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |
| real_kaggle | qwen_static_ex20 | 1920 | 0.8713541666666667 |  | 0.937603764292305 | conservative 15% fail audit; full-context reaudit kept as sensitivity only |

## Audit Rows

| name | tests | notes |
| --- | --- | --- |
| synthetic_gt_disagreement | 54 | {'GT_CORRECT_QWEN_WRONG': 29, 'GT_WRONG_QWEN_CORRECT': 24, 'PARSE_ERROR': 1} |
| real_fail_sample | 152 | {'TRUE_FAIL': 122, 'FALSE_FAIL': 30} |
