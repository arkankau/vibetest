# Evidence verifier notes

## Evaluation scope (real Kaggle)
- Human evaluation is intentionally on **100 sampled FAIL cases** (from ~380 FAIL-verdict predictions), not all 380.
- Re-audit CSV: `results/human-annotations/reaudit_100_fail_pass_inconclusive.csv`
- Match JSONL tests by `(dataset, repo_name, normalized test_prompt)` ↔ `metadata.test_description`

## Label columns in the 100-case re-audit
- `corrected_label` (Fail/Pass/Inconclusive): stricter hand re-audit → `results/kaggle/reaudit_verifier_*`
- `original_label` (Correct/Incorrect): original C/I annotation on those 100 → `results/kaggle/original_label_verifier_*`

## Original-label results (n=100, intentional sample)
- Baseline precision: **0.64** (64 correct / 36 incorrect by original_label)
- Verifier AUROC: **0.47**, AP: **0.64**
- @ threshold 0.7: P=0.59, recall=0.20
- Figures: `results/kaggle/figures/original_label/`

- Verifier prompt uses **continuous scoring** in [0.0, 1.0] (no fixed anchor rubric); prefer fine-grained scores.
- All **100/100** re-audit cases have verifier scores in the kaggle JSONL files.
- Remaining ~280 FAIL-verdict cases were **not** part of the human eval sample; no need to verify them for paper metrics.

## Real Kaggle verifier runs
- Local repo checkouts under `data/kaggle/*` are often missing; use `--allow-missing-repo --no-evidence-artifacts` for text-only verification against cited evidence in the JSONL description.
- Filter to re-audit subset with `--reaudit-csv ...` (100 items vs ~922 total FAILs).
- Docker can exhaust network pools after many Inspect runs; `docker network prune -f` and smaller `--batch-size` (3–5) help.
- `--skip-existing` skips any row with `metadata.evidence_verifier`; clear entries with `score: null` before retry.

## Re-audit results (combined, n=100)
- Baseline precision among audited FAILs: **0.40** (40 valid / 60 invalid)
- Verifier AUROC: **0.42**, AP: **0.39**
- @ threshold 0.7: P=0.36, recall of valid fails=0.20 (too aggressive)
- Outputs: `results/kaggle/reaudit_verifier_*.csv`, figures in `results/kaggle/figures/`

## Real Kaggle aggregate stats (for paper)
- **121** notebooks (50 Titanic / 50 NLP / 21 Diabetic), **16** tests each → **380** FAIL predictions
- Full manual audit precision: **66.1%** (251/380 correct FAIL evidence)
- Experiment-required tests across 121 notebooks (PASS/FAIL/INC):
  - `randomized_label_sanity`: 0 / 44 / 77
  - `outperforms_baseline`: 0 / 23 / 98
  - `training_loss_behavior`: 0 / 7 / 114
  - `tiny_batch_overfit`: 1 / 13 / 107
  - **87** experiment-required FAILs total; re-audit sample: 21/22 → Inconclusive under strict rubric

## Synthetic aggregate stats (for paper)
- **75** injected notebooks (25 × 3 domains), **15** properties each → **1,125** notebook–property pairs
- Combined Vibetest FAIL precision vs GT: **58.8%** (280 TP / 476 FAIL preds); recall **55.6%**
- Verifier AUROC on predicted FAILs: **0.50**; @ τ=0.7 filtered precision **71.8%**, recall **59.1%**

## Paper framing note
- Verifier currently near-chance as human-label classifier; stronger as methodology contribution + PR-curve analysis
- Re-audit (n=100) validates label-source concern: strict rubric → 40% valid FAILs vs 64% in original sample / 66.1% full audit
- **`paper/vibetest_verifier_emnlp2026.tex`**: 8-page target; full property list in appendix (`app:properties`); main-body tables inline via `\captionof{table}` (no forced `[H]` floats) to avoid whitespace gaps

## Verifier fix roadmap (priority order)
1. **Structured rubric** (`--rubric structured`): FAIL / INCONCLUSIVE / PASS aligned with strict re-audit; maps to scores 0.95 / 0.15 / 0.05 for PR curves. Implemented in `vibetest/verifier/rubric.py` + `EvidenceVerifierAgent(rubric=...)`.
2. **Cross-model verifier**: rerun with `--model openai/gpt-4.1` or Claude (≠ Vibetest GPT-5-mini) to break shared bias.
3. **Re-run re-audit 100** → `results/kaggle/structured/` → compare AUROC vs continuous 0.46 baseline.
4. **Future loop**: map INCONCLUSIVE → revise Vibetest verdict (not silent drop).

## Structured smoke (n=10, Titanic re-audit subset, 2026-05-25)
- Output: `results/kaggle/structured/kaggle_titanic_AT-gpt-5-mini_verified.jsonl`
- Verifier states: 9× INCONCLUSIVE (0.15), 1× FAIL (0.95)
- vs `corrected_label`: 4/4 Inconclusive→INCONCLUSIVE; 3/3 Pass→INCONCLUSIVE (good filter); 2/2 Fail→INCONCLUSIVE (too conservative); 1 Inconclusive→FAIL (FP)

## Structured re-audit 100 (complete, 2026-05-25)
- Outputs: `results/kaggle/structured/*_verified.jsonl`, `reaudit_verifier_*.csv`, figures `results/kaggle/figures/structured/`
- States: 95 INCONCLUSIVE, 5 FAIL (same-backbone GPT-5-mini verifier)
- **ALL (n=100)**: baseline P=0.40; AUROC **0.521** (vs continuous **0.458**); @ τ=0.7 P=**0.60** (3/5), R=0.075 (vs continuous @0.7 P=0.353, 6/17)
- Cross-tab (human → verifier): Pass→INC 14, Pass→FAIL 1; Inc→INC 44, Inc→FAIL 1; Fail→INC 37, Fail→FAIL 3
- Interpretation: structured rubric aligns with strict re-audit on Pass/Inconclusive (58/60 → INC) but over-downgrades valid Fail (37/40 → INC); high-τ filter raises precision on tiny kept set

## Synthetic structured run (GT comparison, 2026-05-25)
- **476** predicted FAILs across 3 splits (142/166/168); dry-run repos present, evidence artifacts partial
- Smoke (Titanic n=10): 8 FAIL, 2 INCONCLUSIVE; all 10 were GT-PASS false-positive FAILs (verifier agrees they're unsupported)
- Full run in progress → `results/synthetic/structured/*_verified.jsonl`
- Compare vs continuous baseline: ALL AUROC **0.576**, @ τ=0.7 filtered P=**72.5%** (`results/synthetic/continuous/verifier_scores_summary.csv`)

Smoke command (re-audit 100):
```bash
uv run --active python scripts/verify_vibetest_fail_evidence.py \
  results/kaggle_titanic_AT-gpt-5-mini.jsonl \
  results/kaggle_nlp_AT-gpt-5-mini.jsonl \
  results/kaggle_diabetic_AT-gpt-5-mini.jsonl \
  --reaudit-csv results/human-annotations/reaudit_100_fail_pass_inconclusive.csv \
  --rubric structured --model openai/gpt-4.1 \
  --allow-missing-repo --no-evidence-artifacts --batch-size 5 \
  --output-path results/kaggle/structured/combined_verified.jsonl
```
(Titanic/diabetic/nlp need separate outputs or `--in-place` on copies — use per-file `*_verified.jsonl` under `results/kaggle/structured/`.)
