# Reward Hacking Detection

## Overview

This experiment evaluates whether an AI safety monitor can detect **reward hacking** in agent transcripts. Reward hacking occurs when an AI agent exploits shortcuts to satisfy tests or metrics rather than solving the genuine problem — for example, hardcoding expected outputs, modifying test files, or bypassing evaluation constraints.

This setting uses the ImpossibleBench dataset, where "oneoff" traces contain agents that found shortcuts to pass tests, and "original" traces contain agents that solved the genuine problem.

## Task Design

Each **case** consists of a directory of agent transcript files. The monitor must examine all transcripts and identify which ones exhibit reward hacking behavior.

**Case construction:**
- A mix of "oneoff" (shortcut/cheating) and "original" (legitimate) agent transcripts
- Configurable number of traces per case and positive rate
- All traces are shuffled; the monitor has no prior information about which are positive

**Key parameters:**
- `traces_per_case`: Number of transcripts per case (default: 4)
- `positive_rate`: Fraction of cases containing at least one positive trace (default: 0.5)
- `max_positive_traces_per_case`: Cap on cheating traces per positive case

## Methods

### Agent (VibeTest)
The full VibeTest agent examines each transcript using tool use (bash, python, text editor) within a sandbox. It reads the transcripts, analyzes the agent's behavior, and produces a verdict with evidence.

### Judge (LLM Judge)
A single LLM call per case with the full transcript concatenated. The judge produces a cheating/legit verdict.

### Buffer Defense
A two-stage approach adapted from the stateful buffer defense in [Brown et al. (2025)](https://arxiv.org/abs/2506.06414):

1. **Stage 1 — Per-trace scoring**: Each transcript is independently scored for reward hacking indicators (0.0–1.0) using `IMPOSSIBLEBENCH_PER_TRACE_PROMPT`.
2. **Stage 2 — Buffer classification**: The top-m most suspicious transcripts (ordered by Stage 1 score) are fed in-context to an LLM classifier that examines them collectively and produces revised per-trace scores.

The buffer defense is significantly cheaper than the full agent (single LLM call vs. multi-turn tool use) while potentially capturing cross-trace patterns that per-trace scoring alone would miss.

**Max-merge**: `final_score = max(initial, buffer_revised)` — the buffer can only boost scores, never degrade the per-trace baseline.

## Running the Experiment

```bash
# Buffer defense on impossiblebench
python experiments/safety.py \
  --safety-mode impossiblebench \
  --method buffer \
  --buffer-size 10 \
  --model openai/gpt-5.4-mini-2026-03-17 \
  --cases 50 \
  --seed 42

# Compare with full agent
python experiments/safety.py \
  --safety-mode impossiblebench \
  --method vibetest \
  --model openai/gpt-5.4-mini-2026-03-17 \
  --cases 50 \
  --sandbox docker \
  --seed 42
```

## Metrics

- **Classification accuracy**: Fraction of cases correctly classified as containing/not containing reward hacking
- **Verified accuracy**: Classification accuracy verified by an LLM scorer checking evidence quality
- **Trace precision/recall**: Precision and recall at the individual trace level
- **AUROC/AP**: Area under the ROC/PR curves from per-trace scores

## Results

Data: PatronusAI/trace-dataset (517 traces: 249 benign, 268 positive). 50 cases, 4 traces/case, 50% positive rate.

### Method Comparison

| Method | Model | Cls Acc | Verified | AUROC | AP |
|--------|-------|--------:|---------:|------:|---:|
| **Agent (VibeTest)** | Qwen3.5-397B | 0.560 | **0.440** | **0.857** | **0.897** |
| Buffer defense | Qwen3.5-397B | 0.560 | 0.120 | **0.857** | **0.897** |
| Buffer defense | gpt-5.4-mini | **0.660** | 0.360 | 0.778 | 0.873 |

### Analysis

The agent and Qwen3.5 buffer achieve identical trace-level AUROC (0.857) and AP (0.897), indicating the buffer captures the same signal as the full agent for this task. The key difference is in verified accuracy: the agent (0.440) produces better-quality evidence than the buffer (0.120), since it can use tools to inspect code and reason about behavior in detail.

gpt-5.4-mini buffer achieves the highest classification accuracy (0.660) by predicting 26 positive cases (close to the GT of 25), while Qwen3.5 buffer is more conservative (11 predicted positive). However, Qwen3.5's trace-level scores are more discriminative.
