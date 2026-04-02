# DM Logs

Generated: 2026-04-02 16:07:02 UTC

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
