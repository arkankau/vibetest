# VibeTest Research Closure Plan

## One-Sentence Research Goal

Show that an LLM agent can test messy ML repositories against natural-language pipeline properties with evidence-backed `PASS` / `FAIL` / `INCONCLUSIVE` decisions, while also showing that evaluation of this setting must account for property-operationalization mismatch rather than treating every disagreement as a model error.

## Core Research Question

Can agentic testing turn informal ML pipeline requirements into useful repository-level audits, and how should we evaluate the result when the same natural-language property can be interpreted differently by the benchmark, the model, and the verifier?

## What We Have

### Method

- `VibeTest`: repository-property evaluation with `PASS`, `FAIL`, `INCONCLUSIVE`, evidence, and `case_score`.
- Static Qwen3.6 runs with prompt-example settings:
  - `ex0`: no examples.
  - `ex10`: 10 representative examples.
  - `ex20`: 20 representative examples.
- Selective evaluation using coverage and macro F1.
- Score-based threshold plots and score histograms.
- Conservative real-data audit plus smaller full-context sensitivity analysis.

### Synthetic Kaggle Results

From `results/benchmark_summary.md`:

| Setting | Tests | Coverage | Best macro F1 |
|---|---:|---:|---:|
| Qwen static ex0 | 1125 | 0.773 | 0.802 |
| Qwen static ex10 | 1125 | 0.903 | 0.813 |
| Qwen static ex20 | 1125 | 0.798 | 0.837 |
| Reviewer mode 0 | 1125 | 0.264 | 0.669 |
| Reviewer mode 1 | 1125 | 0.321 | 0.722 |
| Reviewer mode 2 | 1125 | 0.308 | 0.711 |
| TrainCheck | 1125 | 0.000 | n/a |

Interpretation:

- VibeTest beats reviewer-style baselines on synthetic Kaggle.
- Examples help, but not monotonically:
  - `ex10` gives highest coverage.
  - `ex20` gives highest macro F1.
- TrainCheck should be included only as an attempted execution-fragile baseline, not as a strong defeated baseline.
- Small matched-subset probes suggest that the VibeTest rubric/examples matter: direct prompting and generic Codex review miss many injected failures.

### Real Kaggle Results

From `results/benchmark_summary.md`:

| Setting | Tests | Coverage | Conservative macro F1 |
|---|---:|---:|---:|
| Qwen static ex0 | 1920 | 0.822 | 0.931 |
| Qwen static ex10 | 1920 | 0.869 | 0.933 |
| Qwen static ex20 | 1920 | 0.871 | 0.938 |

Interpretation:

- Conservative real Kaggle results are strong, around 0.93 macro F1.
- `ex20` is slightly strongest.
- Full-context reaudit suggests the conservative audit may understate performance, but those adjusted values should stay sensitivity-only.

### Synthetic Ground-Truth / Property Audit

From `results/synthetic/synthetic_gt_mismatch_analysis.md`:

- Total audited rows: 54.
- Usable rows after parse errors: 53.
- Real Qwen misses: 29.
- Synthetic GT / property-definition mismatches: 24.
- 45.3% of usable high-confidence disagreements were property-operationalization mismatches.

Interpretation:

- Synthetic Kaggle is useful as a controlled stress test.
- But raw synthetic F1 is pessimistic/noisy because some labels encode a different interpretation of the property than the evidence standard used by VibeTest.
- This is not just "bad labels"; it is the core phenomenon of natural-language testing.

### Figures We Have

Main paper figures:

- `figures/agentic_testing_evidence_verification.png`
- `figures/qwen_synthetic_f1_coverage.png`
- `figures/qwen_synthetic_max_error_coverage.png`
- `figures/qwen_real_conservative_f1_coverage.png`
- `figures/qwen_synthetic_score_histogram.png`
- `figures/qwen_real_score_histogram.png`

These support the current story:

- Workflow.
- Synthetic selective performance.
- Synthetic residual error / motivation for audit.
- Real conservative performance.
- Score spread.

## What Is Missing / Risky

### 1. Direct Property-Prompt Baseline

Status:

- Completed cost-controlled probes on the first five Titanic synthetic repositories.
- The baseline asks an agent to judge one property directly with the same repository/tool access and the same structured output fields, but without the full VibeTest evidence rubric or representative examples.
- Direct-property model endpoints used for the probes: OpenRouter `qwen/qwen3.6-flash` and `openai/gpt-4.1-mini`.
- Codex reviewer practical-tool probe used OpenRouter `openai/gpt-4.1-mini`.
- Runtime: 75 property checks took 25 minutes and 19 seconds with Qwen3.6 Flash, 35 minutes and 43 seconds with GPT-4.1-mini direct prompting, and 6 minutes and 53 seconds with the Codex reviewer probe.
- Follow-up full synthetic direct-property Qwen3.6 Flash runs are now in progress in the active experiment thread:
  - Titanic full run is complete.
  - NLP full run is partially written.
  - Diabetic full run is not yet available in the current package snapshot.

Why it matters:

- Reviewer baselines are useful, but a skeptical reader may ask whether VibeTest is better than a simple direct prompt.

Observed result:

| Method | Cases | Coverage | Macro F1 | FAIL precision | FAIL recall |
|---|---:|---:|---:|---:|---:|
| Direct property Qwen3.6 Flash | 75 | 0.840 | 0.620 | 0.750 | 0.486 |
| Direct property GPT-4.1-mini | 75 | 0.973 | 0.610 | 0.824 | 0.378 |
| Codex reviewer GPT-4.1-mini | 75 | 0.440 | 0.349 | 0.625 | 0.135 |
| VibeTest static ex0 | 75 | 0.840 | 0.681 | 0.840 | 0.568 |
| VibeTest static ex10 | 75 | 0.920 | 0.791 | 0.833 | 0.811 |
| VibeTest static ex20 | 75 | 0.893 | 0.775 | 0.794 | 0.730 |

Recommendation:

- Use this as a sampled baseline probe, not a full benchmark.
- The defensible claim is that direct prompting and generic Codex review are substantially weaker on failure recall in this subset.
- If the full direct-property runs complete cleanly, refresh the baseline table and text from the full synthetic files. Until then, keep the package claim scoped to the five-repository matched probe.

### 2. Stronger Baseline Framing for TrainCheck

Missing:

- A clean successful TrainCheck run on Kaggle notebooks.

Why it matters:

- TrainCheck is a natural ML debugging baseline.

Current status:

- Existing canonical TrainCheck files have 0 coverage / all inconclusive.
- Smoke tests reproduced execution fragility.

Recommendation:

- Do not spend more time forcing TrainCheck onto Kaggle notebooks unless we move to a clean Linux/container setup.
- In the paper, keep TrainCheck as an attempted baseline with 0 coverage for transparency.

### 3. Broader Audit of Real Passes

Missing:

- The real Kaggle audit focuses on predicted fails.
- We do not have full recall over all real bugs.

Why it matters:

- The current real F1 is based on conservative sampled fail audit / proxy assumptions, not exhaustive real ground truth.

Recommendation:

- Be explicit: real Kaggle results estimate precision/quality of surfaced decisions under audit, not complete bug recall.
- Avoid overclaiming real-world recall.

### 4. Synthetic Corrected-Label Table

Missing:

- A clean table translating the 29/24 audit into a corrected or sensitivity-adjusted synthetic estimate.

Why it matters:

- Adam's concern was that real F1 looked higher than synthetic F1, suggesting synthetic labels may be noisy.

Recommendation:

- Keep raw synthetic as the headline benchmark number.
- Add the disagreement audit as the explanation for why synthetic is pessimistic.
- Do not fully "correct" synthetic F1 unless we audit a larger and more representative sample.

### 5. Compile-Ready ACL Package

Missing:

- `acl.sty` and `acl_natbib.bst` are not included in the package.

Recommendation:

- Before sending externally, place official ACL style files in the package and compile once.

## Defendable Storyline

### Short Version

Agentic testing is promising for ML repositories because many important ML pipeline requirements are naturally written in English, not as formal specifications. VibeTest can operationalize those properties over real notebooks and produce evidence-backed decisions. It outperforms reviewer-style baselines on synthetic Kaggle and performs strongly on real Kaggle under conservative audit. However, the hardest part is not just model accuracy: natural-language properties themselves have ambiguous operational boundaries. The synthetic audit shows that many high-confidence "errors" are actually disagreements between the benchmark label and an evidence-based interpretation of the property.

### Paper Arc

1. **Problem**
   ML pipeline bugs often violate natural-language intent: no leakage, correct splits, train-only augmentation, deterministic evaluation, sensible metrics.

2. **Gap**
   Formal/static tools need formalized properties. Reviewer-style agents can find bugs, but they are not property-directed and can produce unsupported findings.

3. **Method**
   VibeTest treats each `(repo, property)` pair as a test case. The agent returns `PASS`, `FAIL`, or `INCONCLUSIVE`, evidence, and a continuous score.

4. **Result 1: VibeTest Works Better Than Reviewer Baselines**
   On synthetic Kaggle, VibeTest with Qwen3.6 beats reviewer-style baselines on macro F1 and coverage. A small direct-property baseline probe also suggests that simply asking an agent the property directly is not enough to recover the same failure recall.

5. **Result 2: Real Kaggle Looks Strong Under Conservative Audit**
   On real Kaggle, conservative sampled fail audit gives around 0.93 macro F1.

6. **Result 3: Synthetic Is Pessimistic Because NL Properties Are Ambiguous**
   In 53 usable high-confidence synthetic disagreements, 29 are real Qwen misses and 24 are property-operationalization mismatches.

7. **Core Takeaway**
   The research contribution is not simply "VibeTest gets high F1." It is that natural-language repository testing requires evidence-backed verdicts and evaluation protocols that separate model mistakes from property-definition mismatch.

## What We Should Claim

Strong claims:

- VibeTest is a viable framework for repository-level testing against natural-language ML pipeline properties.
- Qwen3.6 VibeTest outperforms reviewer-style baselines on synthetic Kaggle.
- A sampled direct-property baseline has high coverage but weaker failure recall, suggesting that the evidence rubric and example calibration matter.
- Real Kaggle conservative audit suggests strong precision/F1 for surfaced decisions.
- Synthetic ground truth is useful but pessimistic because natural-language property labels can mismatch evidence-based interpretation.

Careful claims:

- VibeTest does not solve general ML debugging.
- Real Kaggle recall is not fully established.
- The direct-property baseline is sampled, not a full benchmark.
- TrainCheck was attempted but did not provide usable coverage in this notebook setting.
- Full-context adjusted real F1 values should be sensitivity analysis, not headline claims.

Claims to avoid:

- "VibeTest reaches near-perfect F1 on real Kaggle."
- "TrainCheck is beaten as a fully operational baseline."
- "Synthetic labels are wrong in general."
- "The verifier definitively fixes evaluation."

## Immediate Next Steps

1. Add a compact "Research Question and Contributions" section to the paper draft.
2. Add a paragraph explicitly answering Adam's concern: real F1 can look higher than synthetic because synthetic raw F1 mixes true model misses with property-operationalization mismatch.
3. When the full direct-property Qwen3.6 Flash baseline finishes, update:
   - `vibetest_qwen_kaggle_emnlp2026.tex`
   - `vibetest_qwen_kaggle_emnlp2026_pips_style.tex`
   - `supporting_docs/PAPER_RESULTS_DRAFT.md`
   - `supporting_docs/RESEARCH_CLOSURE_PLAN.md`
   - `supporting_docs/SUBMISSION_PACKAGE.md`
   - the package zip.
4. Add an appendix table with the 29/24 synthetic audit breakdown by dataset/property.
5. Add ACL style files and compile the package.

## Current Best Title

`VibeTest: Evidence-Backed Agentic Testing for Natural-Language ML Pipeline Properties`

Alternative:

`Agentic Testing for Natural-Language ML Pipeline Properties`

The first title is better if we want the system name to be central. The second is better if we want the paper to feel more like a general research contribution.
