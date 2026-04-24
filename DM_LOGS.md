# DM Logs

Generated: 2026-04-03 11:20:00 UTC

Current zero-unmentioned trace-level AP for `Meerkat` on distributed misuse with `gpt-5.4-mini` in the cyber domain, plus the `Naive Agent` baseline.

| Setting | Meerkat Zero-Unmentioned AP | Meerkat Zero-Unmentioned SE | Naive Agent AP | Naive Agent SE | Meerkat Result | Naive Result | Monitor Result |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| cyber bg=20x | 0.766124 | 0.086195 | 0.764160 | 0.083136 | `results/dm_cyber_d6_bg20_v6.jsonl` | `results/dm_cyber_d6_bg20_gpt-5.4-mini_naive.jsonl` | `results/dm_cyber_d6_bg20_gpt-5.4-mini_llmjudge.jsonl` |
| cyber bg=100x | 0.386347 | 0.064846 | 0.273338 | 0.059193 | `results/dm_cyber_d6_bg100_n15.jsonl` | `results/dm_cyber_d6_bg100_gpt-5.4-mini_naive.jsonl` | `results/dm_cyber_d6_bg100_gpt-5.4-mini_llmjudge.jsonl` |

Current standing after removing the DM property `Approach:` block:
- `cyber bg=20x`: `Meerkat` now narrowly beats the `Naive Agent` (`0.766124` vs `0.764160`).
- `cyber bg=100x`: `Meerkat` still clearly beats the `Naive Agent` (`0.386347` vs `0.273338`).
- Relative to the immediately previous run with the stronger cluster-comparison wording, this improved both settings (`bg=20x: 0.708162 -> 0.766124`, `bg=100x: 0.341142 -> 0.386347`).

## Failed Experiment: Stronger Cluster-Comparison Prompt Wording

Tried strengthening the artifact-enabled `clusters.json` prompt guidance so `Meerkat` had to consider all clusters before finishing and compare clusters before deciding one contained a violating witness.

Observed trace-level AP after rerunning with that wording:

| Setting | Meerkat Zero-Unmentioned AP | Bootstrap SE | Naive Agent AP | Naive Agent SE |
| --- | ---: | ---: | ---: | ---: |
| cyber bg=20x | 0.708162 | 0.089474 | 0.764160 | 0.083136 |
| cyber bg=100x | 0.341142 | 0.071295 | 0.273338 | 0.059193 |

Why this is considered a failure:
- It regressed relative to the preceding baseline on both settings (`bg=20x: 0.716087 -> 0.708162`, `bg=100x: 0.390667 -> 0.341142`).
- The wording seems to have made the agent slightly less decisive without fixing the remaining witness-selection errors.

## Successful Experiment: Remove the DM Property `Approach:` Block

Removed the task-specific `Approach:` section from `DISTRIBUTED_MISUSE_VIBETEST_TEST` in `experiments/safety.py`, leaving the rest of the DM property text intact.

Why this is considered a success:
- It materially improved `cyber bg=20x`, where `Meerkat` now beats the `Naive Agent`.
- It also improved `cyber bg=100x` relative to the stronger cluster-comparison prompt run while preserving a comfortable lead over the `Naive Agent`.
- This strongly suggests the cluster-first workflow embedded in the DM property text was hurting `Meerkat`, especially on the smaller cyber setting.

## Qwen-3.5 Pre-Rerun Baseline

Current zero-unmentioned trace-level AP for `Meerkat`, plus the `Naive Agent` and `LLMJudge` baselines, using the current canonical Qwen-3.5 DM result files across both cyber and bio.

| Domain | Setting | Meerkat AP | Meerkat SE | Naive AP | Naive SE | LLMJudge AP | LLMJudge SE | Meerkat Result | Naive Result | Monitor Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Cyber | bg=20x | 0.927083 | 0.039150 | 0.963399 | 0.024105 | 0.173327 | 0.050895 | `results/dm_cyber_d6_bg20_qwen35_n20.jsonl` | `results/dm_cyber_d6_bg20_qwen35_naive.jsonl` | `results/dm_cyber_d6_bg20_qwen35_llmjudge.jsonl` |
| Cyber | bg=100x | 0.577803 | 0.062312 | 0.605809 | 0.063642 | 0.035968 | 0.007613 | `results/dm_cyber_d6_bg100_qwen35_n50.jsonl` | `results/dm_cyber_d6_bg100_qwen35_naive.jsonl` | `results/dm_cyber_d6_bg100_qwen35_llmjudge.jsonl` |
| Cyber | bg=200x | 0.213467 | 0.050545 | 0.317241 | 0.060072 | 0.007477 | 0.000798 | `results/dm_cyber_d6_bg200_qwen35_n50.jsonl` | `results/dm_cyber_d6_bg200_qwen35_naive.jsonl` | `results/dm_cyber_d6_bg200_qwen35_llmjudge.jsonl` |
| Bio | bg=20x | 0.568462 | 0.081729 | 0.610544 | 0.090728 | 0.131446 | 0.028579 | `results/dm_bio_d6_bg20_qwen35_n20.jsonl` | `results/dm_bio_d6_bg20_qwen35_naive.jsonl` | `results/dm_bio_d6_bg20_qwen35_llmjudge.jsonl` |
| Bio | bg=100x | 0.124851 | 0.034063 | 0.197888 | 0.051096 | 0.015969 | 0.003048 | `results/dm_bio_d6_bg100_qwen35_v4_n50.jsonl` | `results/dm_bio_d6_bg100_qwen35_naive.jsonl` | `results/dm_bio_d6_bg100_qwen35_llmjudge.jsonl` |

Current standing:
- `Naive Agent` is ahead of `Meerkat` on all five recorded Qwen-3.5 settings.
- The largest current Qwen gap is `Cyber bg=200x` (`0.213467` vs `0.317241`).
- The smallest current Qwen gap is `Cyber bg=100x` (`0.577803` vs `0.605809`).

## Qwen-3.5 Last Prompt Baseline

Current zero-unmentioned trace-level AP after the most recent Qwen-3.5 rerun on the standard DM settings:

| Domain | Setting | Meerkat AP | Meerkat SE | Naive AP | Naive SE | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cyber | bg=20x | 0.860060 | 0.058313 | 0.963399 | 0.024105 | -0.103340 |
| Cyber | bg=100x | 0.552537 | 0.062756 | 0.605809 | 0.063642 | -0.053272 |
| Bio | bg=20x | 0.669609 | 0.085089 | 0.610544 | 0.090728 | +0.059065 |
| Bio | bg=100x | 0.183875 | 0.052028 | 0.197888 | 0.051096 | -0.014013 |

Interpretation:
- This prompt variant improved the bio settings relative to the previously logged Qwen baseline.
- It regressed both cyber settings relative to the previously logged Qwen baseline.
- On this run, `Meerkat` beats the `Naive Agent` only on `Bio bg=20x`.

## Qwen-3.5 Setup Comparison

To avoid cherry-picking the best result separately for each setting, compare full method setups across the same standard four settings (`Cyber 20x`, `Cyber 100x`, `Bio 20x`, `Bio 100x`).

| Setup | Cyber 20x | Cyber 100x | Bio 20x | Bio 100x | Mean AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier Qwen pre-rerun baseline | 0.927083 | 0.577803 | 0.568462 | 0.124851 | 0.549550 |
| Most recent Qwen rerun | 0.860060 | 0.552537 | 0.669609 | 0.183875 | 0.566520 |

Current best full setup so far:
- On the standard four-setting comparison, the most recent Qwen rerun is the stronger overall setup (`0.566520` mean AP vs `0.549550`).
- Its tradeoff is that it regressed on both cyber settings but improved enough on both bio settings to win overall.

Additional note:
- `Cyber bg=200x` currently exists only for the earlier setup, where `Meerkat` reached `0.213467`, so it is tracked separately rather than folded into the setup-level comparison above.
- After logging this baseline, a prompt inconsistency in `vibetest/agent/react_agent.py` was fixed: the short task prompt for artifact-enabled safety runs had still said omitted traces keep `initial_scores.tsv`, even though the actual merge code already treated omitted traces as `0.0`. The next rerun will test the new prompt changes on top of that fix.

## Qwen-3.5 Current Rerun After Prompt-Semantics Fix

Current zero-unmentioned trace-level AP after rerunning Qwen-3.5 `Meerkat` with the latest prompt changes plus the fixed short task prompt semantics:

| Domain | Setting | Meerkat AP | Meerkat SE | Naive AP | Naive SE | Delta vs Naive | Delta vs Last Prompt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cyber | bg=20x | 0.967500 | 0.019165 | 0.963399 | 0.024105 | +0.004101 | +0.107440 |
| Cyber | bg=100x | 0.625768 | 0.062640 | 0.605809 | 0.063642 | +0.019959 | +0.073231 |
| Bio | bg=20x | 0.620710 | 0.093342 | 0.610544 | 0.090728 | +0.010166 | -0.048899 |
| Bio | bg=100x | 0.151490 | 0.045690 | 0.197888 | 0.051096 | -0.046398 | -0.032385 |

Interpretation:
- This rerun improved both cyber settings substantially relative to the last prompt baseline and now beats the `Naive Agent` on both cyber settings.
- It still beats the `Naive Agent` slightly on `Bio bg=20x`.
- It regressed on both bio settings relative to the last prompt baseline, and `Bio bg=100x` remains below the `Naive Agent`.

## Qwen-3.5 Updated Setup Comparison

Comparing the three full Qwen-3.5 setups logged so far across the same standard four settings (`Cyber 20x`, `Cyber 100x`, `Bio 20x`, `Bio 100x`):

| Setup | Cyber 20x | Cyber 100x | Bio 20x | Bio 100x | Mean AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier Qwen pre-rerun baseline | 0.927083 | 0.577803 | 0.568462 | 0.124851 | 0.549550 |
| Previous prompt baseline | 0.860060 | 0.552537 | 0.669609 | 0.183875 | 0.566520 |
| Current rerun after prompt-semantics fix | 0.967500 | 0.625768 | 0.620710 | 0.151490 | 0.591367 |

Current best full setup so far:
- On the standard four-setting comparison, the current rerun is now the strongest overall Qwen-3.5 setup (`0.591367` mean AP).
- The gains are driven by strong cyber improvements that more than offset the bio regressions.

## Qwen-3.5 Coherence Clustering Rerun

Current zero-unmentioned trace-level AP after rerunning Qwen-3.5 `Meerkat` on the standard four DM settings with `CLUSTERING_STRATEGY=coherence`:

| Domain | Setting | Meerkat AP | Meerkat SE | Naive AP | Naive SE | Delta vs Naive | Delta vs Previous Best Setup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cyber | bg=20x | 0.939187 | 0.033204 | 0.960409 | 0.023977 | -0.021222 | -0.028313 |
| Cyber | bg=100x | 0.645360 | 0.057265 | 0.609736 | 0.064182 | +0.035624 | +0.019592 |
| Bio | bg=20x | 0.587748 | 0.092359 | 0.628523 | 0.083756 | -0.040775 | -0.032962 |
| Bio | bg=100x | 0.174470 | 0.046572 | 0.201130 | 0.050133 | -0.026660 | +0.022980 |

Interpretation:
- This rerun improved `Cyber bg=100x` and `Bio bg=100x` relative to the previous best full setup.
- It regressed `Cyber bg=20x` and `Bio bg=20x`, including losing the slight `Cyber bg=20x` edge over the `Naive Agent`.
- On this run, `Meerkat` beats the `Naive Agent` only on `Cyber bg=100x`.

## Qwen-3.5 Setup Comparison (Including Coherence)

Comparing the four full Qwen-3.5 setups logged so far across the same standard four settings (`Cyber 20x`, `Cyber 100x`, `Bio 20x`, `Bio 100x`):

| Setup | Cyber 20x | Cyber 100x | Bio 20x | Bio 100x | Mean AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier Qwen pre-rerun baseline | 0.927083 | 0.577803 | 0.568462 | 0.124851 | 0.549550 |
| Previous prompt baseline | 0.860060 | 0.552537 | 0.669609 | 0.183875 | 0.566520 |
| Current rerun after prompt-semantics fix | 0.967500 | 0.625768 | 0.620710 | 0.151490 | 0.591367 |
| Coherence clustering rerun | 0.939187 | 0.645360 | 0.587748 | 0.174470 | 0.586691 |

Current best full setup so far:
- The earlier `silhouette` rerun after the prompt-semantics fix remains the strongest overall Qwen-3.5 setup (`0.591367` mean AP).
- The `coherence` rerun is close and improved the harder `Cyber bg=100x` setting, but it is slightly worse overall because of regressions on `Cyber bg=20x` and `Bio bg=20x`.

## Qwen-3.5 Hierarchical-Only Artifact Rerun

Current zero-unmentioned trace-level AP after rerunning Qwen-3.5 `Meerkat` with `hierarchical_clusters.txt` exposed to the agent and `clusters.json` hidden from the agent-visible repo view:

| Domain | Setting | Meerkat AP | Meerkat SE | Naive AP | Naive SE | Delta vs Naive | Delta vs Previous Best Setup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cyber | bg=20x | 0.829046 | 0.065295 | 0.960409 | 0.023977 | -0.131363 | -0.138454 |
| Cyber | bg=100x | 0.511623 | 0.062064 | 0.609736 | 0.064182 | -0.098113 | -0.114145 |
| Bio | bg=20x | 0.479789 | 0.092833 | 0.628523 | 0.083756 | -0.148734 | -0.140921 |
| Bio | bg=100x | 0.132451 | 0.044254 | 0.201130 | 0.050133 | -0.068679 | -0.019039 |

Interpretation:
- This rerun regressed on all four standard settings relative to the previous best Qwen setup.
- `Meerkat` now trails the `Naive Agent` on all four of those settings.
- Hiding `clusters.json` and relying only on the hierarchical text artifact appears to remove useful flat-cluster information such as topic labels, suspicion ordering, and compact candidate groups.

## Qwen-3.5 Setup Comparison (Including Hierarchical-Only)

Comparing the five full Qwen-3.5 setups logged so far across the same standard four settings (`Cyber 20x`, `Cyber 100x`, `Bio 20x`, `Bio 100x`):

| Setup | Cyber 20x | Cyber 100x | Bio 20x | Bio 100x | Mean AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier Qwen pre-rerun baseline | 0.927083 | 0.577803 | 0.568462 | 0.124851 | 0.549550 |
| Previous prompt baseline | 0.860060 | 0.552537 | 0.669609 | 0.183875 | 0.566520 |
| Current rerun after prompt-semantics fix | 0.967500 | 0.625768 | 0.620710 | 0.151490 | 0.591367 |
| Coherence clustering rerun | 0.939187 | 0.645360 | 0.587748 | 0.174470 | 0.586691 |
| Hierarchical-only artifact rerun | 0.829046 | 0.511623 | 0.479789 | 0.132451 | 0.488227 |

Current best full setup so far:
- The earlier `silhouette` rerun after the prompt-semantics fix still remains the strongest overall Qwen-3.5 setup (`0.591367` mean AP).
- The hierarchical-only artifact setup is the weakest of the logged Qwen full setups so far.
