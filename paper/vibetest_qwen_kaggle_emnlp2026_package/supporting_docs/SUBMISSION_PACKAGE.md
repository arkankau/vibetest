# Submission Package

This file indexes the artifacts that are currently ready for PI/paper review.

## Main Draft Docs

| Artifact | Purpose |
|---|---|
| `vibetest_qwen_kaggle_emnlp2026.tex` | Main ACL-style manuscript draft. |
| `vibetest_qwen_kaggle_emnlp2026_pips_style.tex` | Alternate method-first rewrite inspired by the attached PIPS paper's presentation style. |
| `supporting_docs/PAPER_RESULTS_DRAFT.md` | Main draft results section with tables, figure references, and interpretation. |
| `supporting_docs/AUDIT_EXAMPLES.md` | Representative synthetic and real audit examples. |
| `supporting_docs/RESEARCH_CLOSURE_PLAN.md` | Claim stack, risk register, and remaining work. |
| `supporting_docs/REFERENCE_STYLE_NOTES.md` | Notes on how the PIPS reference was used as presentation context. |
| `supporting_docs/CLAIM_AUDIT_AND_BASELINE_HANDOFF.md` | Numeric-claim audit and checklist for updating the package after the full direct-property baseline finishes. |

## Main Figures

| Figure | Use |
|---|---|
| `figures/qwen_synthetic_f1_coverage.png` | Main synthetic F1-vs-coverage result comparing Qwen examples, reviewers, and TrainCheck. |
| `figures/qwen_synthetic_max_error_coverage.png` | Synthetic max-error selective plot. |
| `figures/direct_property_vs_vibetest_titanic_repo5_summary.png` | Sampled direct-property baseline probe compared with VibeTest on the same five Titanic synthetic repositories. |
| `figures/qwen_real_conservative_f1_coverage.png` | Main conservative real Kaggle F1-vs-coverage result. |
| `figures/qwen_synthetic_score_histogram.png` | Synthetic Qwen score histogram. |
| `figures/qwen_real_score_histogram.png` | Real Kaggle Qwen score histogram. |
| `figures/agentic_testing_evidence_verification.png` | Workflow diagram for evidence-backed agentic testing. |

## Appendix/Sensitivity Artifacts

| Artifact | Use |
|---|---|
| `supporting_docs/RESEARCH_CLOSURE_PLAN.md` | Records sensitivity-analysis interpretation and remaining optional work. |
| `supporting_docs/AUDIT_EXAMPLES.md` | Includes real full-context sensitivity examples and caveats. |

## Tables and Reproducibility

| Artifact | Use |
|---|---|
| `supporting_docs/paper_headline_table.md` | Paper-ready headline table. |
| `supporting_docs/direct_property_vs_vibetest_titanic_repo5_summary.csv` | Machine-readable direct-property baseline probe summary. |
| `supporting_docs/direct_property_vs_vibetest_titanic_repo5_report.md` | Markdown direct-property baseline probe report. |
| `supporting_docs/direct_property_qwen_flash_vs_vibetest_titanic_repo5_summary.csv` | Machine-readable Qwen3.6 Flash direct-property probe summary on the matched Titanic subset. |
| `supporting_docs/direct_property_qwen_flash_vs_vibetest_titanic_repo5_report.md` | Markdown Qwen3.6 Flash direct-property probe report. |
| `supporting_docs/synthetic_kaggle_titanic_codex_reviewer_openrouter-gpt-4.1-mini_repo5.jsonl` | Codex reviewer GPT-4.1-mini practical-tool probe on the same five Titanic synthetic repositories. |

## Audit Inputs

| Artifact | Use |
|---|---|
| `supporting_docs/AUDIT_EXAMPLES.md` | Human-readable synthetic and real audit examples. |
| `supporting_docs/PAPER_RESULTS_DRAFT.md` | Contains the synthetic disagreement-audit and real-audit aggregate numbers used in the paper. |

## Not Mainline

The workspace still contains older smoke-test outputs, exploratory figures, and intermediate audit files. They are useful for provenance but should not be presented as main paper artifacts unless explicitly moved into the package above.
