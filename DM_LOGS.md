# DM Logs

Generated: 2026-04-02 04:39:56 UTC

Current zero-unmentioned trace-level AP for `Meerkat` on distributed misuse with `gpt-5.4-mini` in the cyber domain, plus the `Naive Agent` baseline.

| Setting | Meerkat Zero-Unmentioned AP | Meerkat Zero-Unmentioned SE | Naive Agent AP | Naive Agent SE | Meerkat Result | Naive Result | Monitor Result |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| cyber bg=20x | 0.716087 | 0.095432 | 0.764160 | 0.083136 | `results/dm_cyber_d6_bg20_v6.jsonl` | `results/dm_cyber_d6_bg20_gpt-5.4-mini_naive.jsonl` | `results/dm_cyber_d6_bg20_gpt-5.4-mini_llmjudge.jsonl` |
| cyber bg=100x | 0.390667 | 0.068201 | 0.273338 | 0.059193 | `results/dm_cyber_d6_bg100_n15.jsonl` | `results/dm_cyber_d6_bg100_gpt-5.4-mini_naive.jsonl` | `results/dm_cyber_d6_bg100_gpt-5.4-mini_llmjudge.jsonl` |

Current standing before the new cluster-comparison prompt rerun:
- `cyber bg=20x`: `Meerkat` is still behind the `Naive Agent` (`0.716087` vs `0.764160`).
- `cyber bg=100x`: `Meerkat` is now ahead of the `Naive Agent` (`0.390667` vs `0.273338`).
- The remaining gap is setting-specific: `Meerkat` has recovered strongly on the larger `bg=100x` cyber run, but it still loses some high-impact `bg=20x` cases by choosing the wrong cluster or abstaining even when a clean violating cluster exists.

## Failed Experiment: `initial_rank.csv` Prior

Tried replacing the visible `initial_scores.tsv` Meerkat prior with `initial_rank.csv`, where rank `1` is most suspicious and ties share the same rank. This was a failed experiment and was reverted.

Observed trace-level AP after rerunning with the rank-only prior:

| Setting | Trace AP | Bootstrap SE |
| --- | ---: | ---: |
| cyber bg=20x | 0.589408 | 0.208846 |
| cyber bg=100x | 0.130379 | 0.028405 |

Why this is considered a failure:
- The main `cyber bg=20x` setting regressed materially, from `0.770569` down to `0.589408`.
- Converting scores to ranks removed magnitude information from the monitor prior.
- Ties collapsed many traces onto the same rank, which made the prior less discriminative for navigation.
- Meerkat appears to benefit from the calibrated score magnitude, not just the ordering.

## Failed Experiment: General Objective "Search Comprehensively"

Tried changing the general Meerkat safety-agent objective so it explicitly said to search comprehensively for witnesses rather than stopping at the first plausible one. In the same round of edits, I also simplified the artifact-enabled `safety_rules` guidance, reducing the more detailed search procedure down to a shorter description focused on `initial_scores.tsv` and `clusters.json`. This was also a failed experiment and was reverted.

Observed trace-level AP after rerunning with the modified objective:

| Setting | Trace AP | Bootstrap SE |
| --- | ---: | ---: |
| cyber bg=20x | 0.479642 | 0.213744 |
| cyber bg=100x | 0.126954 | 0.048887 |

Why this is considered a failure:
- The main `cyber bg=20x` setting regressed again, from the original `0.770569` down to `0.479642`.
- The broader search objective appears to have made the agent less decisive without improving witness selection.
- Simplifying `safety_rules` likely removed useful procedural guidance about hierarchical search and witness reassessment.
- The change was too general and did not specifically address the real failure mode of choosing the wrong narrow cluster.

## Failed Experiment: Zero-Unmentioned Meerkat Scoring + `silhouette` Rerun

Tried a new Meerkat experiment where artifact-enabled runs still use `initial_scores.tsv` as a search prior, but the prompt and merge logic were changed so that `TRACE_SCORES` is treated as a fresh final list and any unmentioned traces default to `0.0`. In the same experiment round, the DM runner was changed back from `coherence` to `silhouette` for Meerkat.

Changes relative to `HEAD` before the rerun:
- `vibetest/agent/react_agent.py`: artifact-enabled Meerkat prompt now says omitted traces default to `0.0` and `initial_scores.tsv` is only a search prior.
- `experiments/safety.py`: `_merge_trace_scores(...)` now zero-initializes all traces instead of inheriting `initial_scores.tsv` for omitted traces.
- `scripts/run_dm_buffer_bayesian.sh`: DM Meerkat invocation switched back to `--distributed-misuse-clustering-strategy silhouette`.

Observed `cyber bg=100x` results after rerunning:

| Metric | Trace AP | Bootstrap SE |
| --- | ---: | ---: |
| Meerkat (max merge, actual reported metric) | 0.086452 | 0.024960 |
| Meerkat (overwrite merge, counterfactual) | 0.080638 | 0.025789 |
| Meerkat (zero-unmentioned, counterfactual) | 0.294172 | 0.116979 |
| Naive Agent | 0.332140 | 0.114411 |
| Previous committed Meerkat max-merge baseline (`HEAD`) | 0.102373 | 0.019941 |

Why this is considered a failure:
- The actual reported Meerkat metric is still worse than the Naive Agent (`0.086452` vs `0.332140`).
- The actual reported Meerkat metric also regressed relative to the committed baseline (`0.086452` vs `0.102373`).
- The counterfactual zero-unmentioned variant did improve substantially (`0.294172`), which supports the hypothesis that inherited initial-score false positives are harming ranking quality.
- But even that zero-unmentioned variant still did not beat the Naive Agent on this setting.

## Decision: Keep Zero-Unmentioned Scoring

Even though the combined `cyber bg=100x` rerun above was still worse than the Naive Agent, the zero-unmentioned scoring semantics are materially better than inherited-score max-merge and are worth keeping.

Key observation:
- `cyber bg=20x`: zero-unmentioned `0.944444` vs max-merge `0.783770`
- `cyber bg=100x`: zero-unmentioned `0.294172` vs max-merge `0.086452`

Interpretation:
- Letting omitted traces inherit `initial_scores.tsv` keeps many stale false positives alive in the final ranking.
- Treating `TRACE_SCORES` as the fresh final score list is a better fit for how Meerkat actually reasons about witnesses.
- The remaining gap to the Naive Agent appears to be a witness-selection problem, not a score-merging problem.
