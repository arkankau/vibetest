# Claim Audit and Baseline Handoff

This note records which paper claims are already backed by package artifacts and what should be refreshed after the full direct-property Qwen3.6 Flash baseline finishes.

## Stable Headline Claims

| Claim in draft | Current value | Package artifact |
|---|---:|---|
| Synthetic Kaggle size | 1,125 repo-property cases | `supporting_docs/paper_headline_table.md` |
| Real Kaggle size | 1,920 repo-property cases | `supporting_docs/paper_headline_table.md` |
| Synthetic VibeTest ex0 | coverage 0.773, macro F1 0.802 | `supporting_docs/paper_headline_table.md`, `figures/qwen_synthetic_f1_coverage.png` |
| Synthetic VibeTest ex10 | coverage 0.903, macro F1 0.813 | `supporting_docs/paper_headline_table.md`, `figures/qwen_synthetic_f1_coverage.png` |
| Synthetic VibeTest ex20 | coverage 0.798, macro F1 0.837 | `supporting_docs/paper_headline_table.md`, `figures/qwen_synthetic_f1_coverage.png` |
| Synthetic reviewer modes | macro F1 0.669 / 0.722 / 0.711 | `supporting_docs/paper_headline_table.md` |
| TrainCheck attempted baseline | coverage 0.000 | `supporting_docs/paper_headline_table.md`, `supporting_docs/RESEARCH_CLOSURE_PLAN.md` |
| Real VibeTest ex0 | coverage 0.822, conservative macro F1 0.931 | `supporting_docs/paper_headline_table.md`, `figures/qwen_real_conservative_f1_coverage.png` |
| Real VibeTest ex10 | coverage 0.869, conservative macro F1 0.933 | `supporting_docs/paper_headline_table.md`, `figures/qwen_real_conservative_f1_coverage.png` |
| Real VibeTest ex20 | coverage 0.871, conservative macro F1 0.938 | `supporting_docs/paper_headline_table.md`, `figures/qwen_real_conservative_f1_coverage.png` |
| Synthetic disagreement audit | 53 usable rows: 29 model misses, 24 operationalization mismatches | `supporting_docs/PAPER_RESULTS_DRAFT.md`, `supporting_docs/AUDIT_EXAMPLES.md` |
| Real conservative fail audit | 122 true failures, 30 false failures | `supporting_docs/PAPER_RESULTS_DRAFT.md`, `supporting_docs/AUDIT_EXAMPLES.md` |
| Real full-context sensitivity | 26 of 30 conservative false failures accepted under full context | `supporting_docs/PAPER_RESULTS_DRAFT.md`, `supporting_docs/AUDIT_EXAMPLES.md` |

## Baseline Probe Claims

These are intentionally scoped as matched five-repository probes until the full direct-property run is complete.

| Method | Cases | Coverage | Macro F1 | FAIL precision | FAIL recall | Package artifact |
|---|---:|---:|---:|---:|---:|---|
| Direct property Qwen3.6 Flash | 75 | 0.840 | 0.620 | 0.750 | 0.486 | `supporting_docs/direct_property_qwen_flash_vs_vibetest_titanic_repo5_report.md` |
| Direct property GPT-4.1-mini | 75 | 0.973 | 0.610 | 0.824 | 0.378 | `supporting_docs/direct_property_vs_vibetest_titanic_repo5_report.md` |
| Codex reviewer GPT-4.1-mini | 75 | 0.440 | 0.349 | 0.625 | 0.135 | `supporting_docs/PAPER_RESULTS_DRAFT.md`, `supporting_docs/synthetic_kaggle_titanic_codex_reviewer_openrouter-gpt-4.1-mini_repo5.jsonl` |
| VibeTest ex10 on same subset | 75 | 0.920 | 0.791 | 0.833 | 0.811 | `supporting_docs/direct_property_qwen_flash_vs_vibetest_titanic_repo5_report.md` |

## Claims To Keep Scoped

- The direct-property and Codex reviewer baselines are sampled probes in this package, not full benchmarks.
- Real Kaggle results are based on conservative sampled fail audit, not exhaustive real ground truth.
- Full-context real Kaggle values are sensitivity analysis only.
- TrainCheck was attempted but did not yield usable coverage in the canonical workspace files.
- Synthetic labels should be described as useful but pessimistic, not simply wrong.

## After Full Direct Baseline Completes

Refresh these files if the full Qwen3.6 Flash direct-property baseline is complete and clean:

1. `vibetest_qwen_kaggle_emnlp2026.tex`
2. `vibetest_qwen_kaggle_emnlp2026_pips_style.tex`
3. `supporting_docs/PAPER_RESULTS_DRAFT.md`
4. `supporting_docs/RESEARCH_CLOSURE_PLAN.md`
5. `supporting_docs/SUBMISSION_PACKAGE.md`
6. `supporting_docs/CLAIM_AUDIT_AND_BASELINE_HANDOFF.md`
7. Refresh the adjacent package zip after editing the folder contents.

Specific text/table updates:

- Replace "cost-controlled matched-subset probe" language only if all three synthetic subdatasets are complete.
- Add a full direct-property baseline row/table with dataset scope, total cases, coverage, macro F1, FAIL precision, and FAIL recall.
- Keep the five-repository table as either an appendix/provenance table or remove it from the main draft if superseded.
- If only Titanic and NLP complete, keep full-run results out of headline claims and describe them as partial/full-by-subdataset diagnostics.

## Mechanical Checks Already Done

- `supporting_docs/SUBMISSION_PACKAGE.md` was changed to reference package-local paths only.
- All indexed package paths in `supporting_docs/SUBMISSION_PACKAGE.md` exist.
- Citation keys in both TeX drafts resolve against `vibetest.bib`.
