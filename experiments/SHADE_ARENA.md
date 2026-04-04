# SHADE-Arena Sabotage Detection

## Overview

This experiment evaluates sabotage detection on the SHADE-Arena and CUA-SHADE-Arena datasets from [Kale et al. (2025)](https://scale.com/research/mrt). Agents perform personal-assistant and computer-use tasks while secretly pursuing hidden side objectives (data exfiltration, financial manipulation, systematic suboptimal choices).

## Dataset

**Source:** [ScaleAI/mrt](https://huggingface.co/datasets/ScaleAI/mrt) (HuggingFace)

| Source | Tasks | Sabotage | Clean | Total |
|--------|-------|----------|-------|-------|
| SHADE-Arena | 2 (excl. bank_transfer) | 17 | 87 | 104 |
| CUA-SHADE-Arena | 15 | 149 | 150 | 299 |
| **Total** | **17** | **166** | **237** | **403** |

All transcripts from `claude-3-5-sonnet-20241022`. Transcripts cleaned: `<thinking>` tags stripped, system prompt sabotage instructions redacted. `bank_transfer_2` excluded (0 clean traces).

**CUA-SHADE-Arena tasks:** admin (spreadsheets, forms, page removal), data science (coffee, meetings, tables, formatting), finance (budget, payroll, expenses, invoices), HR (attendance, salary memos, salary analysis), research.

## Methods

### SHADE Monitor
The monitoring protocol from [Kutasov et al. (2025)](https://arxiv.org/abs/2506.15740). Each transcript is independently scored by the monitor LLM with a detailed system prompt asking for a 0-100 suspicion verdict. Full transcript context (no truncation).

### Extract-and-Evaluate (EaE)
Two-stage pipeline adapted from [Arike et al. (2025)](https://arxiv.org/abs/2601.21112):
1. **Extractor**: LLM reads full transcript, identifies 3-5 most suspicious excerpts with verbatim quotes
2. **Evaluator**: Second LLM call scores the excerpts with the user's task description as context

Both methods are per-trace (score each transcript independently). Model: `Qwen3.5-397B-A17B-FP8` via vLLM.

## Results

### Trace-level AUROC across case sizes

Cases constructed with TRACE-style stratified sampling: variable traces-per-case (tpc), mixed tasks, 50% positive rate, sabotage count uniformly distributed across 1..max per positive case, interleaved case ordering.

| tpc | Cases | Method | Traces | Case Acc | Trace AUROC |
|-----|-------|--------|--------|----------|-------------|
| 1 | 403 | **Meerkat (Agent)** | 429 | 0.744 | **0.867** |
| 1 | 403 | EaE | 403 | 0.727 | 0.827 |
| 1 | 403 | Monitor | 403 | 0.749 | 0.808 |
| 10 | 40 | EaE | 354 | 0.500 | 0.890 |
| 10 | 40 | Monitor | 400 | 0.525 | 0.829 |
| 25 | 16 | EaE | 351 | 0.500 | 0.904 |
| 25 | 16 | Monitor | 400 | 0.500 | 0.901 |
| 50 | 8 | EaE | 349 | 0.500 | 0.938 |
| 50 | 8 | Monitor | 400 | 0.500 | 0.980 |

Meerkat (Agent) only run at tpc=1 (requires Docker sandbox, batched in groups of 20). Monitor and EaE are per-trace methods; EaE scores at tpc>1 propagated from tpc=1 via content hash (~87% coverage).

### Per-task AUROC (Monitor, tpc=1)

| Task | n | AUROC | Category |
|------|---|-------|----------|
| admin_make_spreadsheet | 24 | **1.000** | Easy |
| admin_mass_forms_filling | 17 | **1.000** | Easy |
| admin_remove_pages | 15 | **1.000** | Easy |
| ds_format_excel_sheets | 26 | **1.000** | Easy |
| finance_check_attendance_payroll | 23 | 0.996 | Easy |
| finance_expense_validation | 21 | **1.000** | Easy |
| finance_invoice_matching | 21 | **1.000** | Easy |
| hr_salary_analysis | 24 | **1.000** | Easy |
| research_answer_questions | 18 | **1.000** | Easy |
| ds_fix_table_values_and_missing_answers | 17 | 0.896 | Medium |
| ds_find_meeting_spreadsheet | 16 | 0.889 | Medium |
| ds_coffee | 15 | 0.821 | Medium |
| api_key_calendar_agendas_2 | 32 | 0.756 | Medium |
| finance_budget_variance | 11 | 0.750 | Medium |
| hr_populate_salary_increase_memo | 20 | 0.719 | Medium |
| hr_check_attendance_multiple_days | 18 | 0.444 | Hard |
| american_airlines_2 | 85 | 0.422 | Hard |

9 tasks are trivially detectable (AUROC >= 0.99). The hard cases are `american_airlines_2` (subtle flight cost inflation) and `hr_check_attendance_multiple_days` (subtle data manipulation).

### Full method comparison (tpc=1, n=403)

| Method | AUROC | Acc | Notes |
|--------|-------|-----|-------|
| **Meerkat (Agent)** | **0.867** | 0.744 | Full tool-use in Docker sandbox |
| EaE | 0.827 | 0.727 | Two-stage: extract excerpts + evaluate |
| Monitor | 0.808 | **0.749** | Paper's single-call protocol |
| Paper's monitor | 0.648 | 0.607 | Their baseline (84 traces, SHADE-only) |

The agent outperforms both the monitor and EaE on AUROC. With tool access in a Docker sandbox, the agent can read full transcripts, search for patterns, and reason about multi-step behaviors that the single-call monitor misses. The monitor achieves the highest case-level accuracy (0.749) due to better specificity on clean traces.

### Figures

![PR Curves by Case Size](../figures/shade_arena_paper_pr_curves.png)

![AUROC Decomposition](../figures/shade_arena_paper_decomposition.png)

### Where does the Agent's improvement come from?

A swap analysis decomposes the Agent's AUROC improvement over the Monitor into sensitivity (finding positives) vs. specificity (rejecting negatives):

| Configuration | AUROC | vs. Monitor |
|---------------|-------|-------------|
| Monitor (baseline) | 0.808 | — |
| Agent pos scores + Monitor neg scores | 0.957 | +0.149 |
| Monitor pos scores + Agent neg scores | 0.681 | -0.127 |
| **Agent (full)** | **0.884** | **+0.076** |

**The Agent's improvement comes entirely from better positive detection.** If we keep the Monitor's clean-trace scores but use the Agent's sabotage-trace scores, AUROC jumps to 0.957 (+0.149). The Agent finds hard positives — subtle sabotage traces (AA flights, HR manipulation) where the Monitor gives moderate scores (0.3-0.5) but the Agent, with tool access and multi-step reasoning, correctly pushes them to 0.8+.

The tradeoff: the Agent is *worse* at rejecting clean traces. Swapping in the Agent's negative scores drops AUROC by -0.127, indicating more false alarms. But this is a worthwhile trade: the Agent raises the recall ceiling on hard positives that the Monitor simply cannot detect.

The decomposition figure (panel b) shows this clearly: at any FPR budget, the Agent catches more sabotage traces. The positive CDF (panel c) shows the Agent pushes sabotage scores higher, while the negative CDF (panel d) confirms the Agent is more trigger-happy on clean traces.

### Comparison with paper

We outperform the paper's reported monitor results across all methods. Key differences:
1. **Transcript cleaning**: We strip `<thinking>` tags and redact system prompt sabotage instructions. The paper's transcripts may include these, paradoxically hurting the monitor.
2. **Larger dataset**: Our evaluation includes 15 CUA-SHADE-Arena tasks (many with near-perfect detection) alongside the 3 original SHADE-Arena tasks.
3. **Model**: Qwen3.5-397B vs the paper's unspecified monitor model.

## Running

```bash
# Download dataset (SHADE-Arena + CUA-SHADE-Arena)
python scripts/download_shade_arena.py --include-cua

# Materialize case-size sweep
for tpc in 1 10 25 50; do
  python scripts/materialize_shade_arena_multi.py \
    --traces-per-case $tpc --sabotage-per-case $((tpc/5 > 0 ? tpc/5 : 1)) \
    --mix-tasks --dataset-name safety_shade_arena_tpc${tpc}
done

# Run monitor
python experiments/safety.py \
  --safety-mode mle-sabotage --method shade-monitor \
  --model "vllm/Qwen/Qwen3.5-397B-A17B-FP8" \
  --dataset-name safety_shade_arena_tpc1 \
  --cases 403 --traces-per-case 1 --seed 42

# Run EaE
python experiments/safety.py \
  --safety-mode mle-sabotage --method eae \
  --model "vllm/Qwen/Qwen3.5-397B-A17B-FP8" \
  --dataset-name safety_shade_arena_tpc1 \
  --cases 403 --traces-per-case 1 --seed 42
```
