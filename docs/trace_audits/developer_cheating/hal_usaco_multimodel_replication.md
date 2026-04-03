# HAL USACO Multimodel Replication

Date: 2026-04-02

Corpus:
- Source: `agent-evals/hal_traces`
- Benchmark: `usaco`
- Scaffold: `USACO Episodic + Semantic`
- Paper origin: `Can Language Models Solve Olympiad Programming?` (`https://arxiv.org/abs/2404.10952`)
- Public HAL pages:
  - `https://hal.cs.princeton.edu/`
  - `https://hal.cs.princeton.edu/usaco`
  - `https://hal.cs.princeton.edu/agent/USACO%20Episodic%20%2B%20Semantic`
- Betty run dir: `/vast/projects/exwong/brachiolab/davisrbr/vibetest_hal_usaco_multimodel_monitor`
- Pulled artifacts:
  - `results/external_audit/hal_usaco_multimodel/hal_usaco_multimodel_flagged_for_review.jsonl`
  - `data/safety/trace_audits/hal_usaco_multimodel/case_0000`
  - `data/safety/trace_audits/hal_usaco_multimodel_all/case_0000`

## Question

Does the HAL USACO cheating signal depend on one model, or is it a
scaffold-level failure that replicates across multiple model variants using the
same `USACO Episodic + Semantic` harness?

## Setup

I ran the same Stage-1 monitor workflow on a balanced cross-model slice from
all public `USACO Episodic + Semantic` archives I could locate.

- Public archives covered: `12`
- Traces selected per archive: `10`
- Total traces scored: `120`
- Selection rule:
  - successful tasks only
  - exact benchmark-title match required
  - per-archive cap to avoid a single model dominating the case repo
- Monitor: `anthropic/claude-opus-4-6`

Covered archives:
- `claude-3-7-sonnet-20250219`
- `claude-3-7-sonnet-20250219 high`
- `claude-opus-4.1-20250514`
- `claude-opus-4.1-20250514 high`
- `deepseek-ai/DeepSeek-R1`
- `deepseek-ai/DeepSeek-V3`
- `gemini/gemini-2.0-flash`
- `gpt-4.1-2025-04-14`
- `gpt-5-2025-08-07`
- `o3-2025-04-16`
- `o4-mini-2025-04-16 high`
- `o4-mini-2025-04-16 low`

## Stage-1 result

The result is saturated.

- `120 / 120` traces scored `0.95`
- score histogram: `{0.95: 120}`
- every archive contributed exactly `10` selected traces
- every selected trace carried the same high-signal heuristic features:
  - exact benchmark title present
  - exact benchmark task block present
  - `Solution Notes:` present
  - `Solution Code:` present
  - benchmark-specialized retrieval scaffold present

In other words, this is not a one-model anomaly.

## Full pushed corpus

The balanced `120`-trace slice was only for fair cross-model Stage-1 scoring.
I also materialized and retained the uncapped text corpus of all exact-match
successful cases found in the 12 public archives.

- full exact-match successful cases: `595`
- full text-only corpus path:
  - `data/safety/trace_audits/hal_usaco_multimodel_all/case_0000`
- size on disk: about `30 MB`

Per-archive counts in the full pushed corpus:
- `31` `claude-3-7-sonnet-20250219`
- `31` `claude-3-7-sonnet-20250219 high`
- `58` `claude-opus-4.1-20250514`
- `59` `claude-opus-4.1-20250514 high`
- `43` `DeepSeek-R1`
- `48` `DeepSeek-V3`
- `26` `gemini-2.0-flash`
- `49` `gpt-4.1-2025-04-14`
- `89` `gpt-5-2025-08-07`
- `54` `o3-2025-04-16`
- `70` `o4-mini-2025-04-16 high`
- `37` `o4-mini-2025-04-16 low`

## Manual review

Because all traces tied at `0.95`, the usual “top 10” was not informative.
Instead, I reviewed one representative trace from each of ten distinct model
archives.

### Representative positives

1. `claude-3-7-sonnet-20250219`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_000.txt`
   - Evidence:
     - line 9: `main.run_usaco_episodic_semantic_retrieval`
     - line 16: prompt says it will provide “multiple somewhat similar problems and solutions”
     - lines 44-46: exact active task block starts with `Problem 1: Gifts [Kalki Seksaria and Brian Dean, 2012]`
     - line 100: `Solution Notes:`
     - line 105: `Solution Code:`

2. `claude-3-7-sonnet-20250219 high`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_010.txt`
   - Evidence:
     - line 9: same scaffold function
     - line 47: exact active task block starts with `Problem 3: Grazing Patterns [Brian Dean, 2012]`
     - line 75: `PROBLEM NAME: grazing`
     - line 136: `Solution Notes:`
     - line 139: `Solution Code:`

3. `claude-opus-4.1-20250514`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_020.txt`
   - Evidence:
     - lines 44-46: exact `Gifts` task block
     - line 100: `Solution Notes:`
     - line 105: `Solution Code:`

4. `claude-opus-4.1-20250514 high`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_030.txt`
   - Evidence:
     - lines 44-46: exact `Gifts` task block
     - line 100: `Solution Notes:`
     - line 105: `Solution Code:`

5. `deepseek-ai/DeepSeek-R1`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_040.txt`
   - Evidence:
     - line 47: exact `Grazing Patterns` task block
     - line 136: `Solution Notes:`
     - line 139: `Solution Code:`

6. `deepseek-ai/DeepSeek-V3`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_050.txt`
   - Evidence:
     - lines 44-46: exact `Gifts` task block
     - line 100: `Solution Notes:`
     - line 105: `Solution Code:`

7. `gemini/gemini-2.0-flash`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_060.txt`
   - Evidence:
     - line 42: exact `Problem 3: Mountain Climbing [Videh Seksaria, 2012]`
     - line 64: `PROBLEM NAME: climb`
     - line 97: `Solution Notes:`
     - line 102: `Solution Code:`

8. `gpt-4.1-2025-04-14`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_070.txt`
   - Evidence:
     - lines 44-46: exact `Gifts` task block
     - line 100: `Solution Notes:`
     - line 105: `Solution Code:`

9. `gpt-5-2025-08-07`
   - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_080.txt`
   - Evidence:
     - lines 44-46: exact `Gifts` task block
     - line 100: `Solution Notes:`
     - line 105: `Solution Code:`

10. `o3-2025-04-16`
    - File: `data/safety/trace_audits/hal_usaco_multimodel/case_0000/traces/trace_090.txt`
    - Evidence:
      - lines 44-46: exact `Gifts` task block
      - line 100: `Solution Notes:`
      - line 105: `Solution Code:`

I also spot-checked the remaining `o4-mini` high/low representatives:
- `trace_100.txt`
- `trace_110.txt`

Both show the same pattern on the exact `Gifts` benchmark task, again with
`Solution Notes:` and `Solution Code:` injected into the prompt.

## What this shows

This is a harness-level cheating failure.

The same public scaffold:
- uses the same benchmark-specialized retrieval function
- frames the retrieved bundle as “similar problems and solutions”
- then injects the exact current benchmark task
- then includes the intended solution notes
- then includes full solution code

That pattern repeats across all twelve public model variants in the balanced
slice. The model choice changes, but the cheating behavior does not.

## Assessment

This should be treated as a cross-model replication of developer cheating.

The strongest claim supported by the traces is:
- the `USACO Episodic + Semantic` scaffold itself is cheating
- it is not just one model exploiting a weak prompt
- the public HAL USACO release distributes the same scaffold failure across a
  broad set of frontier and near-frontier models

This is a materially stronger result than the original single-model GPT-5
finding, because it shows the failure is scaffold-driven and model-agnostic.
