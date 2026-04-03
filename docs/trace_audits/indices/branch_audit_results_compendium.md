# Branch Audit Results Compendium

This note records how the old audit branches were consolidated into this unified archive.

## Canonical source branches

| Source branch | Imported here | Canonical write-ups in this tree | Notes |
|---|---|---|---|
| `terminalbench-reward-hacking` | yes | [Terminal-Bench Forge Answer Injection](../developer_cheating/terminalbench_forge_answer_injection.md), [BountyBench](../task_gaming/bountybench_vulnerability_detection.md), [CyBench](../task_gaming/cybench_external_writeups.md) | primary source for Terminal-Bench, BountyBench, and CyBench positives |
| `hal-usaco-review` | yes | [HAL USACO Top-10](../developer_cheating/hal_usaco_top10_trace_review.md), [HAL USACO Multimodel](../developer_cheating/hal_usaco_multimodel_replication.md), [KernelBench Top-50 Strong Hits](../task_gaming/kernelbench_top50_strong_hits.md) | primary source for HAL developer-cheating results and the broader KernelBench manual pass |
| `eval-trace-audit-plan-local` | yes | [Task-Gaming Compendium](task_gaming_compendium.md), [MLE-Bench](../task_gaming/mlebench_public_examples.md), [Nebius Git-Peek](../task_gaming/nebius_rebench_gitpeek_review.md), [SWE-bench Bash Git-History](../task_gaming/swebench_bash_git_history_cases.md), [SWE-bench Verified Tools Git-History](../task_gaming/swebench_verified_tools_git_history_cases.md), [TB2 reviews](../task_gaming/tb2_task_gaming_review.md), [SWE-smith](../task_gaming/swe_smith_sift4_hardcoding.md), [KernelBench family review](../negative_results/kernelbench_family_top19_review.md) | canonical synthesis layer for cross-dataset docs and curated evidence |
| `terminalbench-leaderboard-review` | yes | [TerminalBench Leaderboard Review](../negative_results/terminalbench_leaderboard_review.md) | imported only the unique leaderboard slice and monitor outputs |
| `terminalbench-yoonholee-review` | yes | [TerminalBench Yoonholee Review](../negative_results/terminalbench_yoonholee_review.md) | imported because it contains two clean `expected.csv` task-gaming positives despite low value for developer-cheating claims |
| `swe-rebench-openhands-top50-dataset` | yes | [SWE-rebench OpenHands Top-50 Review](../task_gaming/swe_rebench_openhands_top50_review.md) | reclassified from “cheating” to task gaming / overclaiming |
| `taubench-banking-top50-review` | yes | [TAU-Bench Banking Top-50 Review](../negative_results/tau_bench_banking_top50_review.md) | negative result retained to document a checked but low-yield corpus |

## Intentionally excluded

| Branch | Why excluded |
|---|---|
| `atlassian-rovo-top38-dataset` | user explicitly asked not to import it after the claim was downgraded |

## Alias or notes-only branches not needed as merge sources

These branches did not add distinct evidence beyond the canonical sources above, so they were not imported as separate content lines:

- `agent-trace-audit-notes`
- `audit-more-agent-traces`
- `audit-new-evals`
- `cheating-audit-new-evals`
- `eval-trace-audit-plan`
- `swe-rebench-openhands-top50-audit`
- `swe-rebench-openhands-top50-audit-branch`
- `the-agent-company-experiments-audit`
- `trace-acquisition-plan`
- `trace-acquisition-plan-local`

## Coverage notes

- The unified archive keeps the strongest affirmative write-ups separate from screening / negative-result reviews.
- `task gaming` is the umbrella term for model cheating and the older reward-hacking label in this tree.
- `developer cheating` is reserved for scaffold- or submission-level answer injection or same-instance leakage.
- `betty_pull/` was treated as a transport directory, not a canonical storage location. Relevant artifacts were rehomed under:
  - `data/safety/trace_audits/`
  - `results/external_audit/`
- Old exploratory files that were strictly superseded by cleaner descendants were not duplicated again. The main example is the older KernelBench and MLE review notes, whose substantive content is preserved by the curated case docs plus the broader family review.

## Confidence statement

I am >90% confident that this tree captures the meaningful developer-cheating and task-gaming write-ups from the relevant audit branches, with one intentional omission:

- `atlassian-rovo-top38-dataset`, excluded by user request

The remaining gaps are limited to:

- stale or duplicated exploratory notes that were superseded by cleaner imported write-ups
- branch aliases that pointed at the same historical commits rather than distinct content
