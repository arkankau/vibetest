# Archived drafts (do not compile as the main document)

These are superseded earlier drafts, moved here so the package has exactly one
compilable main document at its root
(`vibetest_qwen_kaggle_emnlp2026_pips_style.tex`). Keeping multiple
`\begin{document}` files that all wrote to the same `output.aux` was causing an
Overleaf `\@LN@col`/"Missing \begin{document}" compile failure.

- `vibetest_qwen_kaggle_emnlp2026_pips_style.tex` — the previous recommended
  draft (PIPS-style presentation), superseded by the COLM-style rewrite
  `vibetest_colm_style.tex` at the package root. All of its content (results,
  figures, flags) was carried into the rewrite.
- `vibetest_qwen_kaggle_emnlp2026.tex` — older EMNLP-style draft (also `[review]{acl}`).
- `agentic_testing_icml2026.tex` — earlier ICML-style draft (plain `article`).

Note: their relative paths (`figures/`, `vibetest.bib`, `acl.sty`) point at the
parent folder, so to compile one of these again you would need to move it back to
the package root or fix the paths.
