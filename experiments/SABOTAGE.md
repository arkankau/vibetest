# Sabotage Detection

## Overview

This experiment evaluates whether an AI safety monitor can detect **sabotage behavior** in agent transcripts. Sabotage occurs when an AI agent secretly pursues auxiliary objectives beyond its stated task instructions — for example, inserting backdoors, manipulating oversight mechanisms, or accessing resources beyond task requirements.

This setting uses the MLE-Sabotage dataset, where "aware" traces contain agents with hidden auxiliary instructions, and "regular" traces contain agents performing only the stated task.

## Task Design

Each **case** consists of a directory of agent transcript files. The monitor must examine all transcripts and identify which ones exhibit suspicious auxiliary behavior.

**Case construction:**
- A mix of "aware" (sabotage) and "regular" (legitimate) agent transcripts
- Configurable number of traces per case and positive rate
- All traces are shuffled; the monitor has no prior information about which are positive

**Key parameters:**
- `traces_per_case`: Number of transcripts per case (default: 4)
- `positive_rate`: Fraction of cases containing at least one positive trace (default: 0.5)
- `max_positive_traces_per_case`: Cap on aware traces per positive case

## Methods

### Agent (VibeTest)
The full VibeTest agent examines each transcript using tool use (bash, python, text editor) within a sandbox. It reads the transcripts, analyzes the agent's behavior for hidden objectives, and produces a verdict with evidence.

### Judge (LLM Judge)
A single LLM call per case with the full transcript. Uses a dedicated sabotage-detection prompt that flags actions outside the requested task, suspicious modifications, or behavior suggesting hidden goals.

### Buffer Defense
A two-stage approach adapted from the stateful buffer defense in [Brown et al. (2025)](https://arxiv.org/abs/2506.06414):

1. **Stage 1 — Per-trace scoring**: Each transcript is independently scored for sabotage indicators (0.0–1.0) using `MLE_SABOTAGE_PER_TRACE_PROMPT`. Indicators include: actions unrelated to the stated task, unauthorized file modifications, backdoor insertion, oversight manipulation.
2. **Stage 2 — Buffer classification**: The top-m most suspicious transcripts (ordered by Stage 1 score) are fed in-context to an LLM classifier using `MLE_SABOTAGE_BUFFER_PROMPT`.

**Max-merge**: `final_score = max(initial, buffer_revised)` — the buffer can only boost scores, never degrade the per-trace baseline.

## Running the Experiment

```bash
# Buffer defense on mle-sabotage
python experiments/safety.py \
  --safety-mode mle-sabotage \
  --method buffer \
  --buffer-size 10 \
  --model openai/gpt-5.4-mini-2026-03-17 \
  --cases 50 \
  --seed 42

# Compare with full agent
python experiments/safety.py \
  --safety-mode mle-sabotage \
  --method vibetest \
  --model openai/gpt-5.4-mini-2026-03-17 \
  --cases 50 \
  --sandbox docker \
  --seed 42
```

## Metrics

- **Classification accuracy**: Fraction of cases correctly classified as containing/not containing sabotage
- **Verified accuracy**: Classification accuracy verified by an LLM scorer checking evidence quality
- **Trace precision/recall**: Precision and recall at the individual trace level
- **AUROC/AP**: Area under the ROC/PR curves from per-trace scores

## Results

### Buffer Defense (Qwen3.5-397B, n=25, m=4)

| Metric | Value |
|--------|-------|
| Classification accuracy | 0.520 |
| Verified accuracy | 0.360 |
| AUROC (macro) | **0.885±0.17** |
| AP (macro) | **0.917±0.19** |

The buffer defense achieves strong trace-level discrimination for sabotage detection. The high AUROC (0.885) and AP (0.917) indicate that the combination of per-trace scoring and buffer-level classification effectively identifies traces with hidden auxiliary objectives.

The classification accuracy (0.520) is moderate, suggesting the buffer's binary verdict is less reliable than its continuous scores. The verified accuracy (0.360) reflects the LLM verifier's stricter standard for evidence quality.

Sabotage behavior (unauthorized file modifications, backdoor insertion, hidden functionality) appears to be more detectable from truncated transcripts than reward hacking or distributed misuse campaigns, likely because sabotage actions are inherently anomalous relative to the stated task.
