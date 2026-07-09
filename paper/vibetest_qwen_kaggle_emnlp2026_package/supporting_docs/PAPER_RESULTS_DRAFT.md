# Results Draft

## Evaluation Setup

We evaluate VibeTest on ML pipeline repositories using repository-property cases. For each case, VibeTest returns a structured verdict: `PASS`, `FAIL`, or `INCONCLUSIVE`, along with evidence and a continuous `case_score`. Lower scores indicate confidence that the repository satisfies the property; higher scores indicate confidence that the repository violates it. We report macro F1 over covered cases and coverage as the fraction of cases that receive a non-inconclusive decision.

We compare three Qwen3.6 static-prompt settings: no examples in the prompt (`ex0`), 10 representative examples (`ex10`), and 20 representative examples (`ex20`). We also compare against reviewer-style baselines on the synthetic benchmark. To answer the "is this just asking an agent the property directly?" concern, we ran cost-controlled direct-property agent baselines on the first five Titanic synthetic repositories, including an OpenRouter Qwen3.6 Flash probe. We also ran a small Codex reviewer GPT-4.1-mini practical-tool probe. TrainCheck was attempted as an additional baseline, but the canonical TrainCheck outputs in this workspace are all inconclusive, so we report it as an execution-fragile attempted baseline rather than a central competitor.

## Synthetic Kaggle Results

On synthetic Kaggle, VibeTest with Qwen3.6 outperforms the reviewer-style baselines under the current selective-evaluation setup. The best synthetic macro F1 values are 0.802 for ex0, 0.813 for ex10, and 0.837 for ex20. The reviewer baselines are lower, with best macro F1 of 0.669, 0.722, and 0.711 for reviewer modes 0, 1, and 2.

| Method | Examples | Tests | Coverage | Best macro F1 |
|---|---:|---:|---:|---:|
| VibeTest static | 0 | 1125 | 0.773 | 0.802 |
| VibeTest static | 10 | 1125 | 0.903 | 0.813 |
| VibeTest static | 20 | 1125 | 0.798 | 0.837 |
| Reviewer mode 0 |  | 1125 | 0.264 | 0.669 |
| Reviewer mode 1 |  | 1125 | 0.321 | 0.722 |
| Reviewer mode 2 |  | 1125 | 0.308 | 0.711 |
| TrainCheck |  | 1125 | 0.000 | n/a |

The example prompts do not produce a strictly monotonic pattern, which is worth reporting honestly. The ex10 prompt gives the highest coverage among the Qwen example settings, while ex20 gives the highest synthetic macro F1. The main result is that representative examples improve the VibeTest curve without collapsing coverage, and that all Qwen VibeTest settings outperform the reviewer-style baselines on macro F1.

### Direct-Property Baseline Probe

The direct-property baseline uses the same repository/tool access and structured output fields as VibeTest, but with a minimal prompt that asks the model to judge one property directly. The Qwen3.6 Flash run is the controlled prompt-only comparison because it keeps the model family close to the main Qwen runs while removing VibeTest's rubric/examples. The GPT-4.1-mini direct-property and Codex-reviewer runs are practical probes, not controlled model-family comparisons. We ran these on the same first five Titanic synthetic repositories because the ReAct-style direct agent is slow: 75 property checks took 25m19s with OpenRouter Qwen3.6 Flash and 35m43s with GPT-4.1-mini.

On this subset, direct prompting has weaker failure recall than VibeTest with examples. The practical Codex reviewer probe is even lower coverage and lower recall. That is the distinction we want the baseline stack to test: simple direct prompting and generic code review can produce useful decisions, but the VibeTest rubric/examples are better at catching injected failures.

| Method | Cases | Coverage | Macro F1 | FAIL precision | FAIL recall |
|---|---:|---:|---:|---:|---:|
| Direct property Qwen3.6 Flash | 75 | 0.840 | 0.620 | 0.750 | 0.486 |
| Direct property GPT-4.1-mini | 75 | 0.973 | 0.610 | 0.824 | 0.378 |
| Codex reviewer GPT-4.1-mini | 75 | 0.440 | 0.349 | 0.625 | 0.135 |
| VibeTest static ex0 | 75 | 0.840 | 0.681 | 0.840 | 0.568 |
| VibeTest static ex10 | 75 | 0.920 | 0.791 | 0.833 | 0.811 |
| VibeTest static ex20 | 75 | 0.893 | 0.775 | 0.794 | 0.730 |

Artifacts:

- `supporting_docs/direct_property_vs_vibetest_titanic_repo5_summary.csv`
- `supporting_docs/direct_property_vs_vibetest_titanic_repo5_report.md`
- `results/synthetic/direct_property_qwen_flash_vs_vibetest_titanic_repo5_summary.csv`
- `results/synthetic/direct_property_qwen_flash_vs_vibetest_titanic_repo5_report.md`
- `results/synthetic/synthetic_kaggle_titanic_codex_reviewer_openrouter-gpt-4.1-mini_repo5.jsonl`
- `figures/direct_property_vs_vibetest_titanic_repo5_summary.png`

Main figure:

- `results/synthetic/figures/synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label_covered_macro_f1_vs_coverage.png`

Caption draft:

> Selective macro F1 versus coverage on synthetic Kaggle. VibeTest with Qwen3.6 outperforms reviewer-style baselines across the reported operating points. Representative prompt examples improve the VibeTest curve, with ex20 achieving the strongest synthetic macro F1 and ex10 achieving the strongest coverage among Qwen example settings.

## Synthetic Ground-Truth Audit

The synthetic benchmark is useful, but the audit suggests that its labels are noisy under an evidence-based property interpretation. We sampled high-confidence disagreements between Qwen predictions and synthetic labels. Of 53 usable audited disagreements, 29 were real Qwen misses and 24 were ground-truth or property-definition mismatches.

| Audit outcome | Count | Interpretation |
|---|---:|---|
| Real Qwen miss | 29 | Synthetic label was correct; VibeTest made the wrong call. |
| Ground-truth/property-definition mismatch | 24 | VibeTest was more consistent with the evidence standard than the synthetic label. |

This means synthetic F1 is still meaningful, but it is pessimistic. The raw score mixes true model errors with cases where the synthetic label or property definition does not cleanly match the evidence required by VibeTest.

This is the direct answer to Adam's concern about high residual synthetic error and real F1 looking higher than synthetic F1. The synthetic curve is not only measuring model mistakes; it also penalizes cases where the injected label and the evidence-based property interpretation disagree. The real Kaggle audit, by contrast, asks whether the surfaced evidence supports the decision, so it can look higher even without implying that the synthetic benchmark is useless.

Representative examples:

| Type | Dataset | Repository | Property | Audit finding |
|---|---|---|---|---|
| Real Qwen miss | diabetic | `SYN-D1` | all parameters updated during training | The notebook freezes all model parameters before an initial `fit_one_cycle`; VibeTest incorrectly treated later unfreezing as enough to pass. |
| GT/property mismatch | diabetic | `SYN-D1` | avoid Python loops when matrix ops apply | The only loops were file printing, path construction, and shape-control logic; numerical computation used tensor operations, so the synthetic fail was too broad. |
| GT/property mismatch | diabetic | `SYN-D2` | augmentation only on train data | The same stochastic transform with `RandomHorizontalFlip` was used for train and validation loaders; VibeTest correctly flagged a violation despite a synthetic pass label. |

Audit artifact:

- `results/synthetic/openrouter_synthetic_gt_disagreement_audit_15pct_neutral.csv`

## Real Kaggle Results

On real Kaggle repositories, the conservative audit gives roughly 0.93 macro F1 across all Qwen prompt settings. The ex20 setting is slightly strongest, with 0.938 conservative macro F1 at 0.871 coverage.

| Method | Examples | Tests | Coverage | Conservative macro F1 |
|---|---:|---:|---:|---:|
| VibeTest static | 0 | 1920 | 0.822 | 0.931 |
| VibeTest static | 10 | 1920 | 0.869 | 0.933 |
| VibeTest static | 20 | 1920 | 0.871 | 0.938 |

The real Kaggle headline should use these conservative audit numbers. A full-context reaudit suggests that some conservative false-fail penalties were too harsh, but that analysis is a smaller sensitivity check and should not be used to claim near-perfect performance.

Main figure:

- `results/figures/real_kaggle_qwen36_k_examples_conservative_macro_f1_vs_coverage.png`

Caption draft:

> Conservative macro F1 versus coverage on real Kaggle. VibeTest with Qwen3.6 reaches around 0.93 macro F1 across prompt-example settings under the conservative fail audit. Full-context reaudit is reported separately as sensitivity analysis.

## Real Kaggle Audit Examples

The conservative real Kaggle audit sampled predicted failures and labeled 122 as true failures and 30 as false failures. Full-context reaudit of the 30 conservative false failures found that many were rejected only because the verifier lacked relevant notebook cells; 26 of the 30 were accepted as true failures under full context.

Representative examples:

| Type | Dataset | Repository | Property | Audit finding |
|---|---|---|---|---|
| True fail | Titanic | `REAL-T1` | validation/test loaders should not shuffle | Both training and validation loaders were constructed with `shuffle=True`, directly violating the property. |
| True fail | Titanic | `REAL-T2` | metrics use exact definitions | The notebook computed ROC-AUC from hard class predictions instead of probabilities or continuous scores. |
| Conservative false fail flipped by context | Titanic | `REAL-T3` | avoid replaceable Python loops | Conservative audit lacked the cited cells; full-context audit found the loop-based `predict` function and accepted the failure. |
| Conservative false fail flipped by context | Titanic | `REAL-T4` | consistent device placement | Conservative audit lacked the cited cells; full-context audit found mixed `net.to(device)`, hardcoded `.cuda()`, `.cpu()`, and CPU tensor usage. |

Sensitivity artifact:

- `results/real_kaggle_full_context_sensitivity_summary.md`

## Score Distributions and Selective Behavior

The score histograms show how Qwen uses the `case_score` range under each prompt-example setting. These plots are useful for interpreting the threshold curves: the model produces enough score spread to support selective evaluation, but many outputs still concentrate in a limited set of score regions.

Score figures:

- `results/synthetic/figures/synthetic_kaggle_qwen36_case_score_bands_histogram.png`
- `results/synthetic/figures/synthetic_kaggle_qwen36_case_score_bands_exact_score_frequency.png`
- `results/figures/real_kaggle_qwen36_case_score_histogram.png`

## Baselines and TrainCheck

Reviewer-style baselines are weaker than VibeTest on synthetic Kaggle in the current result package. The direct-property baseline probes suggest that the gains are not only from giving an LLM tool access to the repository: plain direct prompts miss many injected failures. The Codex reviewer practical-tool probe has low coverage and low failure recall on the same subset. TrainCheck was attempted but produced all-inconclusive canonical outputs, giving 0 coverage. We should describe TrainCheck as an attempted execution-fragile baseline unless a cleaner controlled rerun is completed.

This caveat matters because TrainCheck is not being beaten as a fully operational baseline here; rather, it did not produce usable predictions under the current setup.

## Takeaways

The current evidence supports four careful claims. First, Qwen3.6 VibeTest outperforms reviewer-style baselines on synthetic Kaggle. Second, a small direct-property baseline probe suggests the full evidence rubric/examples matter for failure recall. Third, conservative real Kaggle audit results are strong, around 0.93 macro F1, with ex20 slightly ahead. Fourth, synthetic F1 is useful but pessimistic because high-confidence disagreements include both real Qwen misses and ground-truth/property-definition mismatches.

The paper should avoid claiming near-perfect real Kaggle F1 from the full-context sensitivity check. A safer framing is that the conservative real Kaggle estimate is strong, and the full-context analysis suggests that this estimate may understate performance in some cases.
