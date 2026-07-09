# Running Qwen3.6-35B via OpenRouter (no GPU)

The tester's model layer is Inspect AI, which supports OpenRouter natively.
This unblocks the same-checkpoint Qwen work without self-hosting vLLM.

- Model on OpenRouter: `qwen/qwen3.6-35b-a3b-20260415` (35B MoE, snapshot 2026-04-15).
  Pricing ~ $0.14/M in, $1.00/M out. **Caveat:** likely not the exact FP8 quant you
  self-hosted, and it is the April snapshot -- report it as
  "Qwen3.6-35B-A3B via OpenRouter (2026-04-15)".
- The agent still runs code in a **local Docker sandbox** (no GPU, just Docker Desktop).

## 1. One-time setup

Add to `vibetest/.env`:

```
OPENROUTER_API_KEY=<your-openrouter-key>
OPENAI_API_KEY=<your-openai-key>
```

Make sure Docker Desktop is running.

## 2. Re-run the canonical Qwen AT (fixes provenance flags + gives seed data)

Run once per dataset (swap `--labels-path` and `--datasets`):

```sh
cd vibetest
uv run --active python experiments/synthetic.py \
  --method vibetest \
  --labels-path synth-data/injected/labels_kaggle_titanic.jsonl \
  --datasets kaggle_titanic \
  --model openrouter/qwen/qwen3.6-35b-a3b-20260415 \
  --sandbox docker \
  --scorer-model openai/gpt-5.4-mini
# repeat with labels_kaggle_nlp.jsonl / kaggle_nlp
# repeat with labels_kaggle_diabetic.jsonl / kaggle_diabetic
```

Cost: ~$1.7 per dataset run, ~$5 for all three.

Then score:

```sh
uv run --active python scripts/analyze_synthetic_metrics.py \
  results/synthetic/synthetic_kaggle_titanic_AT-*qwen3.6*.jsonl \
  --output-csv results/synthetic/synthetic_metrics_openrouter.csv
```

## 3. Seeds (settle the ex0/ex10/ex20 CI + enable the cluster bootstrap)

Run step 2 three to five times into different output dirs (or with an Inspect
`--seed` if the runner exposes it) and compute the spread of macro F1.
Caveat: OpenRouter forwards `seed` best-effort per provider; treat reruns as a
variance estimate, not bitwise-deterministic.

Cost: ~$5/pass x 5 = ~$25.

## 4. Still needs building (not just a command)

- **Same-checkpoint direct-property baseline**: its runner is not in the repo.
  It must be reconstructed (a single-prompt "does R satisfy p?" agent, no evidence
  rubric) from the probe output schema before it can run through OpenRouter.
- **ex10/ex20 example counts**: `synthetic.py` has no obvious example-count flag;
  confirm how the 10/20 representative examples are injected before reproducing
  those settings.

## 5. What each run fixes in the paper

| Run | Clears |
| --- | --- |
| Canonical Qwen AT (step 2) | provenance flags on RQ4, `fig:worked-example`, `tab:models` |
| + evidence scorer (gpt-5.4-mini) | recomputes RQ4 evidence-quality on the canonical run |
| Seeds (step 3) | the "ex ranking not significant" caveat; per-case data for the repo-cluster bootstrap |
| Direct-property baseline (step 4) | the `\needsinput` same-checkpoint-baseline flag |
