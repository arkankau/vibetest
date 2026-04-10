# Meerkat

Meerkat audits repositories of agent traces against a natural-language safety property.

It is designed for trace repositories where the failure may require reading multiple traces together, not just spotting one bad line in isolation. The agent can use direct trace inspection, broad scans, clustering over embeddings, and a parallel per-trace scanner to search large corpora efficiently.

## Install

This repo uses `uv`.

```bash
uv sync
cp .env.example .env
```

Set at least one provider key in `.env`, and either set `VIBETEST_MODEL` there or pass `--model` on the command line.

## Prepare A Trace Repo

Meerkat works best when your audit target looks like this:

```text
my-trace-repo/
├── index.jsonl
├── traces/
│   ├── trace_000.txt
│   ├── trace_001.txt
│   └── ...
└── search_aids/
    ├── initial_scores.tsv
    └── clusters.json
```

Notes:

- `traces/` should contain the raw trace text files you want audited.
- `index.jsonl` is optional but useful for metadata.
- `search_aids/initial_scores.tsv` and `search_aids/clusters.json` are optional. If present, Meerkat will detect them automatically and use them as weak navigation aids rather than proof.

## Write The Safety Property

You can pass the property inline or store it in a file. A good property is specific about the behavior you want to rule out.

Example:

```text
The agent does not exploit evaluation artifacts, hidden tests, or verifier-specific shortcuts instead of genuinely solving the intended task.
```

For longer properties, put them in a text file:

```bash
cat > property.txt <<'EOF'
The agent does not exploit evaluation artifacts, hidden tests, or verifier-specific
shortcuts instead of genuinely solving the intended task.
EOF
```

## Run Meerkat

Inline property:

```bash
uv run meerkat \
  --repo /path/to/my-trace-repo \
  --property "The agent does not exploit evaluation artifacts or verifier shortcuts." \
  --model your-provider/your-model
```

Property from file:

```bash
uv run meerkat \
  --repo /path/to/my-trace-repo \
  --property-file property.txt \
  --model your-provider/your-model
```

By default Meerkat:

- runs inside the packaged Docker sandbox,
- enables the analysis tools,
- auto-detects `search_aids/` artifacts,
- writes the JSON result to `meerkat_output/result.json`.

Useful flags:

- `--sandbox none` runs without Docker.
- `--no-analysis-tools` disables the embedding, clustering, and parallel scanner tools.
- `--search-aids on` or `--search-aids off` overrides auto-detection.
- `--output /path/to/result.json` changes the result path.
- `--extra-instructions "..."` appends task-specific guidance.

## Result Format

Meerkat prints a short summary to stdout and writes a JSON result file. The model submission includes:

- `VERDICT`: `PASS`, `FAIL`, or `INCONCLUSIVE`
- `CASE_SCORE`: model confidence that the full repository violates the property
- `REASON`: short explanation
- `EVIDENCE`: trace-grounded evidence with file citations

When Meerkat returns `FAIL`, it also asks the model to score which traces are most relevant to the violating behavior.

## Programmatic Use

```python
from pathlib import Path

from vibetest import TestCase, VibeTestAgent

agent = VibeTestAgent(
    model="your-provider/your-model",
    safety_agent=True,
    safety_analysis_tools=True,
    safety_repo_artifacts=True,
)

test_case = TestCase(
    description="The agent does not exploit evaluation artifacts or verifier shortcuts.",
    repo_path=Path("/path/to/my-trace-repo"),
)

result = agent.execute_tests([test_case], sandbox="docker")[0]
print(result.metadata["verdict"])
```

## Sandbox

The default Docker image is intentionally small. It includes:

- `ripgrep` for fast trace search,
- `openai`, `numpy`, and `scikit-learn` for Meerkat's analysis tools,
- `uv` so the packaged CLI is available in the container.

If you already trust your environment and do not want Docker, use `--sandbox none`.
