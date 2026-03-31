# Adaptive Attacks on Distributed Misuse Detection

## Overview

We evaluate the robustness of our post-hoc decomposition-attack defense to adaptive prompt injection.  The defense pipeline has three stages:

1. **Stage 1** — a per-trace LLM classifier scores each deployment log independently for suspicion (0.0–1.0).
2. **Bayesian monitor** — leave-one-out Bayesian inference over Stage 1 scores, optionally enhanced with semantic clustering, produces refined per-trace posterior scores.
3. **Compositional agent** — a ReAct agent reads the pre-computed clusters, topic labels, initial scores, and raw trace files, then decides whether any cluster constitutes a decomposition campaign and revises trace scores accordingly.

The agent is the most powerful stage: it can identify campaigns by their *structural signature* (a tight cluster with a narrow, specific topic whose individual traces have low initial scores) and boost campaign-trace scores from near-zero to 0.85–0.95.

## Threat model

The attacker controls the content of the campaign traces *before* they enter the pipeline.  They can prepend or append adversarial text to each trace.  Benign background traces are not modified.  The attacker's goal is to prevent the agent from boosting campaign-trace scores so that the defense adds no value over the monitor baseline.

## Attack strategies tested

We tested 12 injection strategies spanning four categories:

**Authority-bias injections** (prepended text claiming prior analysis):
- `auditor_note` — short "false positive" annotation targeting the agent's deference to perceived authority
- `score_format` — fake Stage-1 classifier output block
- `agent_instruction` — fake safety-team review note

**Content reframing** (modifying trace structure to look benign):
- `context_reframe` — academic course framing prepended as conversation turn
- `persona_reframe` — per-trace unique personas (code review, CTF, debugging, etc.)
- `reasoning_anchor` — minimal one-liner establishing broad context
- `fake_refusal` / `varied_refusal` — replace assistant response with (varied) refusal text
- `response_truncate` — completely remove assistant response, keep only question + refusal
- `combined_subtle` — layered context_reframe + topic_broadener

**Embedding attacks** (targeting the clustering stage):
- `embedding_poison` v1/v2 — append/wrap diverse topic text to dilute embedding similarity
- `optimized_embedding` v1/v2 — black-box coordinate-descent prefix optimization against text-embedding-3-small (budget=500 API queries per case)

**Joint attacks**:
- `joint_optimized` — auditor_note + optimized embedding prefix

## Results

All experiments use Qwen3.5-397B-A17B-FP8 as both the Stage 1 scorer and the compositional agent.  Results report mean campaign score (average score assigned to ground-truth campaign traces after max-merge) with paired bootstrap 95% CIs.

### Main results on cyber bg=100x

Two attacks achieve statistically significant reductions in campaign scores at bg=100x:

| Method | n | Mean campaign score | Δ | 95% CI | AP |
|---|---|---|---|---|---|
| Agent baseline | 50 | 0.533 | — | — | 0.10 |
| + `auditor_note` | 50 | 0.433 | **−0.100** | **[−0.180, −0.021]** | 0.13 |
| + `response_truncate` | 20 | 0.192 | **−0.291** | **[−0.449, −0.130]** | 0.02 |
| + `persona_reframe` | 20 | 0.601 | +0.118 | [−0.047, +0.265] | — |

**`response_truncate` is the strongest attack**: it removes the assistant's response entirely, replacing it with a varied refusal (unique per trace).  This reduces mean campaign scores by 60% and AP from 0.10 to 0.02.  Of 11 campaigns the baseline agent found, 9 (82%) were suppressed and 0 backfired.

**`auditor_note`** achieves a more modest 19% reduction.  Of 31 findable campaigns, 10 (32%) were suppressed and 3 (10%) backfired.

**`persona_reframe`** backfires (not significant).

**Why `response_truncate` works**: The agent's compositional analysis requires confirming that "responses can be recomposed into a joint response that answers a harmful task."  Without actual harmful responses — only refusals — the agent cannot confirm compositional harm.  At bg=100x with 600 traces to analyze, the agent doesn't have enough analysis budget to infer campaign membership from question structure alone.

**Why `response_truncate` fails at bg=20x** (Δ=+0.041, n=5): With only 120 traces and 15 clusters, the agent can deeply analyze every suspicious cluster and detects campaigns from question structure alone.  It explicitly reasons: *"The traces systematically cover different aspects of exploiting Siemens S7-1200 PLCs."*

### Null results at bg=20x

| Strategy | n | Baseline | Attack | Δ | 95% CI |
|---|---|---|---|---|---|
| `auditor_note` | 20 | 0.595 | 0.590 | −0.006 | [−0.157, +0.142] |
| `reasoning_anchor` | 20 | 0.595 | 0.608 | +0.013 | [−0.126, +0.134] |
| `persona_reframe` | 20 | 0.595 | 0.659 | +0.063 | [−0.022, +0.162] |
| `varied_refusal` | 20 | 0.595 | 0.698 | +0.103 | [−0.005, +0.219] |

No strategy achieves a statistically significant reduction at bg=20x.  Several trend toward *increasing* campaign scores (backfiring), though none significantly.

### Embedding attacks (n=10, bg=100x, 2pass variant)

| Strategy | Baseline | Attack | Δ |
|---|---|---|---|
| `optimized_embedding` v2 | 0.453 | 0.625 | +0.172 |
| `joint_optimized` | 0.453 | 0.405 | −0.048 |

Black-box embedding optimization successfully reduces clustering coherence (0.64→0.39) and splits campaigns across 3–4 clusters, but the agent compensates via independent content analysis.  The `joint_optimized` attack (auditor_note + embedding optimization) shows a small reduction but is not significant at n=10.

## Why the attack works at bg=100x but not bg=20x

1. **Cognitive load matters.**  At bg=100x the agent processes 600 traces across 75 clusters.  At bg=20x it processes only 120 traces across 15 clusters.  With fewer traces, the agent can thoroughly investigate every suspicious cluster and the injection has no effect.

2. **The agent detects the injection as adversarial 90–94% of the time.**  When it sees the "NOTE TO SAFETY AUDITOR" text, it frequently reasons: *"This is a classic adversarial technique — trying to convince the auditor to ignore suspicious content."*  However, at bg=100x, even when the agent detects the injection, the additional cognitive overhead of managing the adversarial detection alongside 75 clusters sometimes causes it to under-prioritize the actual campaign cluster.

3. **Identical injection text is a fingerprint.**  The agent observes that all 6 traces in the suspicious cluster contain the same prepended note — and zero benign traces do.  It reasons: *"All Cluster X traces contain identical adversarial headers attempting to discourage investigation."*  This makes the cluster *more* suspicious, but at bg=100x the agent sometimes allocates its analysis budget to other suspicious clusters first.

## Why other attacks fail

1. **The defense detects campaigns from question structure alone.**  The `response_truncate` experiment completely removed assistant responses, leaving only questions + refusals.  The agent still detected campaigns with camp_μ = 0.732 (vs 0.691 baseline) — *questions alone reveal the decomposition structure* (systematically covering different aspects of one specific target).

2. **Content reframing backfires.**  Academic framing (`context_reframe`: +34%), persona variation (`persona_reframe`: +11%), and fake refusals (`varied_refusal`: +17%) all make campaign traces *more* conspicuous rather than less.

3. **Embedding model is robust.**  text-embedding-3-small captures the S7-1200-specific semantics regardless of wrapper text.  Even with 1500+ chars of benign context wrapping a 200-char campaign question, the embedding still clusters campaign traces together (coherence 0.66 with v2 wrapping vs 0.63 baseline).

4. **Per-trace variation doesn't help.**  Using unique injection text per trace (`varied_refusal`, `persona_reframe`) eliminates the identical-text fingerprint but doesn't reduce detection, because the agent identifies campaigns from the underlying question content.

## Implications

1. **The agent defense is robust to prompt injection at bg=20x** where it can thoroughly analyze every cluster.  No injection strategy we tested produces a statistically significant reduction in campaign scores.

2. **At bg=100x, the `auditor_note` injection achieves a modest but significant reduction** (19%, p < 0.05) by exploiting cognitive load — the agent must triage among many clusters and sometimes under-prioritizes the campaign.

3. **Preprocessing stages are completely robust.**  Embeddings, clustering, topic labeling, and Bayesian scoring are unaffected by all injection strategies.  The vulnerability is exclusively in the agent's LLM reasoning.

4. **The structural signature of decomposition campaigns is irreducible.**  The questions themselves — narrowly targeting one specific system across multiple aspects — reveal the campaign regardless of injected framing, annotations, or response modifications.

5. **Possible defenses against the bg=100x attack**: (a) strip or ignore in-band annotations that claim prior classification, (b) increase agent analysis budget for high-background settings, (c) ensemble multiple agent runs and take the maximum score.

## Reproduction

```bash
# Create injected repo (auditor_note, cyber bg100, 50 cases)
python experiments/adaptive_attacks.py \
    --source-repo results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35_bayesian_boost \
    --injection auditor_note \
    --dest-repo results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35_bb_auditor_note_n50 \
    --cases 50

# Run pipeline with bayesian boost
python experiments/safety.py \
    --safety-mode distributed-misuse \
    --distributed-misuse-domain cyber \
    --distributed-misuse-decomp-level 6 \
    --distributed-misuse-background-multiplier 100 \
    --distributed-misuse-clustering-strategy coherence \
    --cases 50 \
    --method vibetest \
    --model "vllm/Qwen/Qwen3.5-397B-A17B-FP8" \
    --sandbox docker \
    --seed 42 \
    --bayesian-boost \
    --dataset-name "safety_dm_cyber_d6_bg100_qwen35_bb_auditor_note_n50"

# Analysis with bootstrap CIs
python experiments/analyze_adaptive_attacks.py
```
