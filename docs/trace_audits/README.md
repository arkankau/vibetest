# Trace Audits

This directory is the unified external-trace audit archive for this repo.

Terminology:

- `developer cheating`: scaffold- or developer-authored behavior that injects privileged benchmark answers, solution code, or same-instance gold information before the model solves the task
- `task gaming`: model-side or scaffold-side behavior that exploits the evaluator, leaked artifacts, answer files, benchmark-specific shortcuts, or other non-intended paths to score well without faithfully solving the intended task
- `negative result`: a reviewed corpus or slice that did not yield strong usable positives

Imported source branches:

- `terminalbench-reward-hacking`
- `hal-usaco-review`
- `eval-trace-audit-plan-local`
- `terminalbench-leaderboard-review`
- `terminalbench-yoonholee-review`
- `swe-rebench-openhands-top50-dataset`
- `taubench-banking-top50-review`

Intentionally excluded:

- `atlassian-rovo-top38-dataset`
  The user explicitly asked not to import it after the claim was downgraded.

## Developer Cheating

| Write-up | Scope | Evidence bundle |
|---|---|---|
| [HAL USACO Top-10 Trace Review](developer_cheating/hal_usaco_top10_trace_review.md) | strongest single-model HAL USACO slice | [hal_usaco_top50](../../data/safety/trace_audits/hal_usaco_top50/case_0000/) |
| [HAL USACO Multimodel Replication](developer_cheating/hal_usaco_multimodel_replication.md) | cross-model replication of the same HAL scaffold failure | [hal_usaco_multimodel](../../data/safety/trace_audits/hal_usaco_multimodel/case_0000/), [hal_usaco_multimodel_all](../../data/safety/trace_audits/hal_usaco_multimodel_all/case_0000/) |
| [Terminal-Bench Forge Answer Injection](developer_cheating/terminalbench_forge_answer_injection.md) | framework-level answer injection via `AGENTS.md` in Forge leaderboard traces | [terminalbench_task_gaming](../../data/safety/trace_audits/terminalbench_task_gaming/) |

## Task Gaming

| Write-up | Scope | Evidence bundle |
|---|---|---|
| [BountyBench Vulnerability Detection](task_gaming/bountybench_vulnerability_detection.md) | detect-phase shortcutting and benchmark-shaped exploit substitutes | [bountybench traces](../../data/safety/trace_audits/terminalbench_task_gaming/bountybench_traces/) |
| [CyBench External Writeups](task_gaming/cybench_external_writeups.md) | public writeup lookup and solution replay | [cybench traces](../../data/safety/trace_audits/terminalbench_task_gaming/cybench_traces/) |
| [SWE-smith SIFT4 Hardcoding](task_gaming/swe_smith_sift4_hardcoding.md) | strongest clean SWE-smith task-gaming case | [swe-smith outputs](../../results/external_audit/betty_swe_smith_opus46_20260401b/) |
| [SWE-rebench OpenHands Top-50 Review](task_gaming/swe_rebench_openhands_top50_review.md) | model-side hardcoding, threshold tuning, and overclaiming | [swe_rebench_openhands_top50](../../data/safety/trace_audits/swe_rebench_openhands_top50/case_0000/) |
| [TB2 Task-Gaming Review](task_gaming/tb2_task_gaming_review.md) | top TB2 hard-task sweep | [tb2 outputs](../../results/external_audit/betty_tb2_hard_devcheat_20260401a/) |
| [TB2 Circuit Fibsqrt Scaffold Exploit](task_gaming/tb2_circuit_fibsqrt_scaffold_exploit.md) | clean evaluator-bypass case | [tb2 curated cases](../../data/safety/trace_audits/tb2_curated_cases/) |
| [TB2 Model Extraction Harness Leak](task_gaming/tb2_model_extraction_harness_leak.md) | harness leak, not a clean flagship case | [tb2 curated cases](../../data/safety/trace_audits/tb2_curated_cases/) |
| [KernelBench Trace 146 Product Broadcast](task_gaming/kernelbench_trace_146_product_broadcast.md) | degenerate reduction that still passes | [kernelbench curated cases](../../data/safety/trace_audits/kernelbench_curated_cases/) |
| [KernelBench Trace 244 Native Fallback](task_gaming/kernelbench_trace_244_native_fallback.md) | native fallback on the benchmarked branch | [kernelbench curated cases](../../data/safety/trace_audits/kernelbench_curated_cases/) |
| [KernelBench Top-50 Strong Hits](task_gaming/kernelbench_top50_strong_hits.md) | broader manual review of top KernelBench hits | [kernelbench full review outputs](../../results/external_audit/betty_kernelbench_all_opus46_20260401a/) |
| [MLE-Bench Public Examples](task_gaming/mlebench_public_examples.md) | public leaderboard traces with leaked test-side structure and benchmark-specific overrides | [mlebench curated cases](../../data/safety/trace_audits/mlebench_public_curated_cases/) |
| [Nebius SWE-rebench Git-Peek Review](task_gaming/nebius_rebench_gitpeek_review.md) | repo-history answer recovery in OpenHands traces | [nebius curated cases](../../data/safety/trace_audits/nebius_rebench_gitpeek_curated_cases/), [Betty run](../../results/external_audit/betty_nebius_rebench_gitpeek_opus46_20260401a/) |
| [SWE-bench Bash-Only Git-History Cases](task_gaming/swebench_bash_git_history_cases.md) | bash-only repo-history answer recovery | [swebench_bash_gitpeek_frontier](../../data/safety/trace_audits/swebench_bash_gitpeek_frontier/case_0000/), [selection file](../../results/external_audit/swebench_bash_gitpeek_frontier_selection.jsonl) |
| [SWE-bench Verified Tools Git-History Cases](task_gaming/swebench_verified_tools_git_history_cases.md) | verified-tools repo-history answer recovery | [swebench_verified_tools_gitpeek](../../data/safety/trace_audits/swebench_verified_tools_gitpeek/case_0000/), [selection file](../../results/external_audit/swebench_verified_tools_gitpeek_selection.jsonl) |

## Screening And Negative Results

| Write-up | Scope | Evidence bundle |
|---|---|---|
| [TAU-Bench Banking Top-50 Review](negative_results/tau_bench_banking_top50_review.md) | low-yield TAU2 banking slice; strongest hit was a false positive | [taubench_banking_top50](../../data/safety/trace_audits/taubench_banking_top50/case_0000/), [TAU outputs](../../results/external_audit/taubench/) |
| [TerminalBench Leaderboard Review](negative_results/terminalbench_leaderboard_review.md) | low-yield leaderboard slice for developer cheating | [terminalbench_leaderboard](../../data/safety/trace_audits/terminalbench_leaderboard/case_0000/), [leaderboard outputs](../../results/external_audit/terminalbench_leaderboard/) |
| [TerminalBench Yoonholee Review](negative_results/terminalbench_yoonholee_review.md) | mixed slice with two real `expected.csv` task-gaming hits, but low value for developer-cheating claims overall | [terminalbench_yoonholee](../../data/safety/trace_audits/terminalbench_yoonholee/case_0000/) |
| [KernelBench Family Top-19 Review](negative_results/kernelbench_family_top19_review.md) | family-level scan across reduction, conv, and matmul | [kernelbench full review outputs](../../results/external_audit/betty_kernelbench_all_opus46_20260401a/) |

## Indices And Meta

| Write-up | Purpose |
|---|---|
| [Task-Gaming Compendium](indices/task_gaming_compendium.md) | narrow set of deliberate, high-confidence task-gaming cases across datasets |
| [Branch Audit Results Compendium](indices/branch_audit_results_compendium.md) | branch-by-branch provenance, imports, exclusions, and coverage notes for this reorg |
| [External Trace Audit Plan](meta/external_trace_audit_plan.md) | prior acquisition plan and corpus-selection context |
| [Betty Trace Audit Notes](meta/betty_trace_audit_notes.md) | operational notes for remote runs |
