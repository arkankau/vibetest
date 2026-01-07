# Vibetest

AI agent for executing natural language test cases over arbitrary codebases, with a focus on ML research repositories.

## Overview

Vibetest uses a ReAct agent (powered by Inspect AI) to understand, execute, and validate test cases expressed in natural language. The agent runs in a Docker sandbox for safety and can automatically collect evidence (plots, logs, metrics) to support its pass/fail verdicts.

## Features

- **Natural language test cases**: Express tests like "training loss generally decreases for any trained model"
- **Docker sandboxing**: Safe execution in isolated containers
- **Evidence collection**: Automatic gathering of plots, logs, metrics, and code snippets

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (ANTHROPIC_API_KEY or OPENAI_API_KEY)
```

## Quick Start

### CLI Usage

The CLI provides two ways to define tests:

**1. Quick test with natural language description:**
```bash
# Run a single test on the current directory
vibetest --test "Training loss decreases during training"

# Test a specific repository
vibetest --repo /path/to/repo --test "Model saves checkpoints every epoch"

# Run without Docker sandbox (for testing, but not usually advised)
vibetest --test "Hyperparameters are logged" --no-sandbox
```

**2. Define tests in `vibetest.py` (like pytest):**
```python
# vibetest.py
from pathlib import Path
from vibetest import TestCase

tests = [
    TestCase(
        description="Training loss is logged and generally decreases",
        repo_path=Path("."),
    ),
    TestCase(
        description="Model saves checkpoints periodically",
        repo_path=Path("."),
    ),
]
```

Then simply run:
```bash
vibetest
```

See `vibetest.py.example` for a complete example.

### Programmatic Usage

```python
from pathlib import Path
from vibetest import TestCase, VibeTestAgent

# Create test case - just a description!
test = TestCase(
    description="Training loss is logged every 100 steps and generally decreases",
    repo_path=Path("./my_ml_repo")
)

# Run with agent (synchronous - Inspect AI manages async internally)
agent = VibeTestAgent(model="anthropic/claude-3-5-sonnet-20241022")
results = agent.execute_tests([test], sandbox="docker")
result = results[0]

# Check results
print(f"Passed: {result.passed}")
print(f"Message: {result.message}")
for evidence in result.evidence:
    print(f"Evidence: {evidence.description}")
```

## Creating Custom Test Cases

Simply provide a natural language description:

```python
from pathlib import Path
from vibetest import TestCase, VibeTestAgent

# Any test you can describe!
test = TestCase(
    description="Model uses gradient checkpointing for memory efficiency",
    repo_path=Path("./my_repo")
)

agent = VibeTestAgent()
results = agent.execute_tests([test], sandbox="docker")
print(f"Result: {results[0].passed}")
```

## Architecture

```
vibetest/
  agent/           # ReAct agent implementation
    react_agent.py
  testcases/       # Test case abstractions and implementations
    base.py        # Base TestCase and TestResult classes
    ml_tests.py    # ML-specific test cases
  tools/           # Tools available to the agent
    file_tools.py      # File system operations
  cli.py           # Command-line interface
```

## Environment Variables

Configure via `.env` file or environment variables:

### API Keys (at least one required)
- `ANTHROPIC_API_KEY`: API key for Claude models
- `OPENAI_API_KEY`: API key for OpenAI models
- `GOOGLE_API_KEY`: API key for Google models

### Model Configuration
- `VIBETEST_MODEL`: Model to use (default: `anthropic/claude-3-5-sonnet-20241022`)
  - Anthropic: `anthropic/claude-3-5-sonnet-20241022`, `anthropic/claude-3-opus-20240229`
  - OpenAI: `openai/gpt-4`, `openai/gpt-4-turbo-preview`, `openai/gpt-3.5-turbo`
  - Google: `gemini-2.0-flash-lite` ,`gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`

### Storage
- `VIBETEST_EVIDENCE_DIR`: Directory for evidence storage (default: `./evidence`)
- `VIBETEST_LOG_DIR`: Directory for logs (default: `./logs`)

## Examples

See the `examples/` directory for:
- `simple_example.py`: Basic usage

## Results Viewer

Need a quick way to inspect recent agent runs? A minimal client lives in `viewer/index.html`:

1. From the repo root run `python -m http.server 8000`.
2. Open `http://localhost:8000/viewer/` in a browser.
3. The UI reads the preprocessed `viewer/eval-results.json`, which is generated from one or more `.eval` log archives, so every repo/test shown is backed directly by the `.eval` trace (including the correct evidence tarballs). Code snippets that reference `/kaggle/...` paths use `viewer/repo-paths.json` to locate the checked-out repo under `data/kaggle/`.

### Multi-Log Comparison

The viewer supports loading multiple `.eval` logs side-by-side for comparison. When you select a test in one panel, the same test (matched by description) will be highlighted in other panels, allowing you to compare how different models or runs evaluated the same test case.

If you sync new logs or repos locally:

1. Rebuild the viewer payload (pulls directly from one or more `.eval` logs):

   ```bash
   # Single log:
   python scripts/build_viewer_data.py \
       --eval logs/2025-11-10T19-02-58-05-00_task_oAYv8tQDiuxozNmQZWxK4j.eval \
       --output viewer/eval-results.json
   
   # Multiple logs for comparison:
   python scripts/build_viewer_data.py \
       --eval logs/kaggle-diabetic-gpt-5-mini.eval \
       --eval logs/kaggle-nlp-gpt-5-mini.eval \
       --eval logs/kaggle-titanic-gpt-5-mini.eval \

2. Refresh the repo-path manifest so citations under both `/kaggle/...` and `/iclr/...` resolve to your local mirrors (defaults cover `data/kaggle/kaggle-titanic` with depth 2 and `data/iclr-26/iclr2026_filter2` with depth 1):

```bash
python scripts/build_repo_paths.py
# or customize roots/depths:
# python scripts/build_repo_paths.py --path data/kaggle/kaggle-titanic:2 --path data/iclr-26/iclr2026_filter2:1
```

The UI is intentionally simple—no build tooling required.

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests (once test suite is added)
uv run pytest

# Format code
uv run ruff format .

# Type checking
uv run mypy vibetest
```
