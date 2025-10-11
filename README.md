# Vibetest

AI agent for executing natural language test cases over arbitrary codebases, with a focus on ML research repositories.

## Overview

Vibetest uses a ReAct agent (powered by Inspect AI) to understand, execute, and validate test cases expressed in natural language. The agent runs in a Docker sandbox for safety and can automatically collect evidence (plots, logs, metrics) to support its pass/fail verdicts.

## Features

- **Natural language test cases**: Express tests like "training loss is logged every K steps and generally decreases"
- **ReAct agent**: Iterative reasoning and action using Inspect AI
- **Docker sandboxing**: Safe execution in isolated containers
- **Evidence collection**: Automatic gathering of plots, logs, metrics, and code snippets
- **Extensible architecture**: Easy to add custom test cases and tools
- **Built for ML**: Pre-built test cases for common ML validation scenarios

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (ANTHROPIC_API_KEY or OPENAI_API_KEY)

# Build Docker image for sandboxed execution
docker build -t vibetest .
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

# Run without Docker sandbox (for testing)
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
    exec_tools.py      # Code execution
    analysis_tools.py  # Plotting and analysis
  evidence/        # Evidence collection system
    collector.py
  cli.py           # Command-line interface
```

## Environment Variables

Configure via `.env` file or environment variables:

### API Keys (at least one required)
- `ANTHROPIC_API_KEY`: API key for Claude models
- `OPENAI_API_KEY`: API key for OpenAI models

### Model Configuration
- `VIBETEST_MODEL`: Model to use (default: `anthropic/claude-3-5-sonnet-20241022`)
  - Anthropic: `anthropic/claude-3-5-sonnet-20241022`, `anthropic/claude-3-opus-20240229`
  - OpenAI: `openai/gpt-4`, `openai/gpt-4-turbo-preview`, `openai/gpt-3.5-turbo`

### Storage
- `VIBETEST_EVIDENCE_DIR`: Directory for evidence storage (default: `./evidence`)
- `VIBETEST_LOG_DIR`: Directory for logs (default: `./logs`)

## Docker Sandbox

The Docker sandbox provides isolation for safe code execution:

```dockerfile
# Build the image
docker build -t vibetest .

# Run tests in sandbox
docker run --rm \
  -v /path/to/repo:/workspace/repos/target \
  -v /path/to/output:/workspace/evidence \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  vibetest
```

## Examples

See the `examples/` directory for:
- `simple_example.py`: Basic usage
- `custom_test.py`: Creating custom test cases

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
