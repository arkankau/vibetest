# TAU-Bench / TAU2 trace viability check

Date: 2026-04-02

## Sources checked

- `/tmp/davisrbr/tau-bench/historical_trajectories`
  - `gpt-4o-airline.json`
  - `gpt-4o-retail.json`
  - `sonnet-35-new-airline.json`
  - `sonnet-35-new-retail.json`
- `/tmp/davisrbr/tau2-bench/web/leaderboard/public/submissions`
- Downloaded public TAU2 trajectories to `/tmp/davisrbr/tau2-audit`
  - `gpt-5-2_sierra_2026-02-26` airline / retail / telecom / banking_knowledge
  - `claude-opus-4-5_sierra_2026-02-26` airline / retail / telecom / banking_knowledge
  - `qwen3-max_qwen_2025-10-30` airline / retail / telecom
  - `distyl-buttonagent_distyl_2026-03-25` banking_knowledge

## Main findings

- No assistant-emitted control tokens found in the checked public traces: no `###STOP###`, `###TRANSFER###`, or `###OUT-OF-SCOPE###` from the assistant side.
- No assistant-side references to task ids, assertions, reward fields, or evaluator structure in the checked standard traces.
- Sierra banking retrieval looked benchmark-clean:
  - GPT-5.2 banking had `19062` shell calls with `0` hits for task json files, reward terms, benchmark terms, control tokens, or policy files.
  - Claude Opus 4.5 banking had `8202` shell calls with the same `0` suspicious hits.
- Distyl ButtonAgent did not use shell-like retrieval in the released traces; it used `KB_search`.
  - `6039` KB queries, `0` hits for task files, reward terms, control tokens, policy files, or task ids.
  - The small number of benchmark-term regex hits were benign uses of the ordinary word `submission` in banking workflow language.
- Short high-reward airline transfers appear legitimate, not exploitative.
  - Example: task `46` is explicitly an unsupported insurance-only refund request; full reward is consistent with immediate transfer.
  - Example: task `13` explicitly expects transfer because the user requests a destination change on an existing booking.

## Verdict

TAU-Bench / TAU2 public trajectories look low-yield for cheating or reward hacking under our current rubric.

The standard Sierra submissions mostly expose normal policy-following behavior plus documented retrieval in banking. The strongest custom submission checked here, Distyl ButtonAgent, still looked like ordinary knowledge-base retrieval rather than benchmark leakage.

Unless we want to spend more time on the Google-Drive-only `gpt-5.4_sierra_2026-03-25` banking file or voice submissions, TAU-Bench does not currently look like a high-priority cheating dataset.
