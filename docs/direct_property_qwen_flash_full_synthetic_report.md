# Direct-Property Qwen3.6 Flash Full Synthetic Baseline

This run evaluates the direct-property baseline on all three synthetic Kaggle datasets using OpenRouter `qwen/qwen3.6-flash`.

The baseline receives a compact repository snapshot plus one natural-language property and emits `PASS`, `FAIL`, or `INCONCLUSIVE`.
It does not use the full VibeTest evidence rubric, representative examples, or ReAct/tool loop.

| Dataset | Tests | Coverage | Macro F1 | FAIL P | FAIL R | FAIL F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Titanic | 375 | 0.464 | 0.702 | 0.877 | 0.532 | 0.662 | $1.090 |
| NLP | 375 | 0.608 | 0.754 | 0.745 | 0.693 | 0.718 | $1.174 |
| Diabetic | 375 | 0.323 | 0.851 | 0.879 | 0.823 | 0.850 | $1.251 |
| Combined | 1125 | 0.465 | 0.760 | 0.818 | 0.665 | 0.734 | $3.515 |

Takeaway: direct prompting is a useful baseline but is much more abstention-heavy than VibeTest.
The combined coverage is 0.465, with FAIL recall 0.665.
This supports the paper claim that the VibeTest rubric/examples improve coverage and failure discovery beyond simply asking Qwen to judge each property from a repository snapshot.

Canonical result files:

- `results/synthetic/synthetic_kaggle_titanic_direct-property-baseline-openrouter-qwen3.6-flash_full.jsonl`
- `results/synthetic/synthetic_kaggle_nlp_direct-property-baseline-openrouter-qwen3.6-flash_full.jsonl`
- `results/synthetic/synthetic_kaggle_diabetic_direct-property-baseline-openrouter-qwen3.6-flash_full.jsonl`
