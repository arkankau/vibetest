# Vibetest Quick Start

## 5-Minute Setup

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure API Keys
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key(s)
# ANTHROPIC_API_KEY=your-anthropic-key-here
# OPENAI_API_KEY=your-openai-key-here
```

### 3. Build Docker Image
```bash
docker build -t vibetest .
```

### 4. Run a Test
```bash
uv run vibetest /path/to/your/ml/repo --test training_loss
```

## Quick Examples

### Example 1: Training Loss Test (CLI)
```bash
uv run vibetest ./my_ml_project \
    --test training_loss \
    --logging-interval 100 \
    --output-dir ./results
```

### Example 2: Programmatic Usage
```python
import asyncio
from pathlib import Path
from vibetest import TestCase, VibeTestAgent

async def main():
    # Just describe what you want to test!
    test = TestCase(
        description="Training loss is logged every 100 steps and generally decreases",
        repo_path=Path("./my_ml_project")
    )

    agent = VibeTestAgent()
    result = await agent.execute_test(test, sandbox="docker")

    print(f"Passed: {result.passed}")
    print(result.message)

asyncio.run(main())
```

### Example 3: Custom Test Case
```python
import asyncio
from pathlib import Path
from vibetest import TestCase, VibeTestAgent

async def main():
    # Creating custom tests is just describing them!
    test = TestCase(
        description="Model uses gradient checkpointing for memory efficiency",
        repo_path=Path("./my_repo")
    )

    agent = VibeTestAgent()
    result = await agent.execute_test(test, sandbox="docker")

    print(f"Passed: {result.passed}")

asyncio.run(main())
```

### Example 4: Add Custom Tool
```python
from typing import Annotated
from inspect_ai.tool import tool
from vibetest import VibeTestAgent

@tool
def check_dependencies(requirements_file: Annotated[str, "Path to requirements.txt"]) -> str:
    """Check if specific dependencies are present."""
    try:
        with open(requirements_file) as f:
            content = f.read()
        return f"Dependencies:\n{content}"
    except Exception as e:
        return f"Error: {e}"

agent = VibeTestAgent()
agent.add_tool(check_dependencies)
```

## Common Workflows

### Test ML Training Script
1. Point to ML repo
2. Run `training_loss` test
3. Check generated plot in output directory
4. Review verdict and evidence

### Debug Failed Test
1. Run with `--no-sandbox` for faster iteration
2. Check execution log in results
3. Examine evidence files
4. Adjust test case or repo

### Create Test Suite
```python
from vibetest import TestCase
from vibetest.testcases.ml_tests import training_loss_test, hyperparameter_logging_test

tests = [
    training_loss_test(repo_path, logging_interval=100),
    hyperparameter_logging_test(repo_path),
    TestCase(description="Model uses attention mechanisms", repo_path=repo_path),
]

for test in tests:
    result = await agent.execute_test(test, sandbox="docker")
    print(f"{test.description}: {'PASS' if result.passed else 'FAIL'}")
```

## Troubleshooting

### "Docker not found"
- Install Docker: https://docs.docker.com/get-docker/
- Or use `--no-sandbox` flag (less safe)

### "API key not set"
```bash
# Option 1: Use .env file (recommended)
cp .env.example .env
# Edit .env and add your key

# Option 2: Export environment variable
export ANTHROPIC_API_KEY='sk-ant-...'
# or
export OPENAI_API_KEY='sk-...'
```

### "Which model should I use?"
- **Claude 3.5 Sonnet** (default): Best balance of speed and capability
- **GPT-4**: Alternative if you prefer OpenAI
- Set in `.env`: `VIBETEST_MODEL=openai/gpt-4`

### "Test times out"
- Increase timeout in config
- Check if code hangs
- Verify repo is accessible

### "Agent gives wrong verdict"
- Review execution log
- Check collected evidence
- Adjust test prompt for clarity
- Add more specific tools

## File Locations

- **Results**: `vibetest_output/` (default)
- **Evidence**: `evidence/` (default)
- **Logs**: `logs/` (Inspect AI logs)
- **Config**: Set via environment variables or code

## Tips

1. **Start simple**: Use built-in tests first
2. **Check evidence**: Always review collected evidence
3. **Iterate prompts**: Refine test prompts based on results
4. **Add tools**: Create domain-specific tools for better results
5. **Use sandbox**: Always use Docker for untrusted code

## Resources

- Full docs: `README.md`
- Architecture: `docs/ARCHITECTURE.md`
- Contributing: `docs/CONTRIBUTING.md`
- Examples: `examples/`

## Next Steps

1. Try the examples in `examples/`
2. Create your first custom test case
3. Add domain-specific tools
4. Run tests on your ML projects
5. Contribute improvements!
