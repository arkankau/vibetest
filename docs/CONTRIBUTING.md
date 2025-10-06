# Contributing to Vibetest

Thank you for your interest in contributing to Vibetest! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for sandbox testing)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd vibetest

# Install dependencies
uv sync

# Verify installation
uv run python main.py
```

## Project Structure

```
vibetest/
├── vibetest/           # Main package
│   ├── agent/          # Agent implementations
│   ├── testcases/      # Test case classes
│   ├── tools/          # Tools for the agent
│   ├── evidence/       # Evidence collection
│   ├── config.py       # Configuration
│   └── cli.py          # CLI interface
├── examples/           # Usage examples
├── docs/              # Documentation
├── tests/             # Test suite (future)
├── Dockerfile         # Docker sandbox image
└── pyproject.toml     # Project metadata
```

## Contributing Guidelines

### Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Document all public APIs with docstrings
- Keep functions focused and modular

### Adding a New Test Case

1. Create a class in `vibetest/testcases/` (or create a new file for a category)
2. Inherit from `TestCase`
3. Implement required methods:

```python
from pathlib import Path
from vibetest.testcases.base import TestCase, TestResult
from vibetest.agent.react_agent import VibeTestAgent

class MyNewTest(TestCase):
    def __init__(self, repo_path: Path, **kwargs):
        description = "What this test validates"
        super().__init__(description, repo_path)
        # Store additional parameters
        self.my_param = kwargs.get('my_param')

    async def execute(self) -> TestResult:
        """Execute the test."""
        agent = VibeTestAgent()
        return await agent.execute_test(self, sandbox="docker")

    def get_prompt(self) -> str:
        """Customize the agent prompt."""
        base_prompt = super().get_prompt()

        specific_instructions = """
        Additional instructions for the agent...

        VERDICT: [PASS/FAIL]
        REASON: [Explanation]
        EVIDENCE: [Evidence description]
        """
        return base_prompt + specific_instructions
```

4. Add tests for your test case
5. Update documentation

### Adding a New Tool

1. Create tool in `vibetest/tools/` (or add to existing category file)
2. Use `@tool` decorator from Inspect AI:

```python
from typing import Annotated
from inspect_ai.tool import tool

@tool
def my_new_tool(
    param1: Annotated[str, "Description of param1"],
    param2: Annotated[int, "Description of param2"] = 10,
) -> str:
    """Tool description that the agent will see.

    Args:
        param1: More detailed description
        param2: Another description

    Returns:
        Description of what's returned
    """
    try:
        # Implementation
        result = f"Processed {param1} with {param2}"
        return result
    except Exception as e:
        return f"Error in my_new_tool: {str(e)}"
```

3. Export from `vibetest/tools/__init__.py`
4. Add to default tools in `VibeTestAgent._setup_tools()` if generally useful
5. Document the tool and add examples

### Adding Evidence Types

1. Add to `EvidenceType` enum in `vibetest/testcases/base.py`:

```python
class EvidenceType(str, Enum):
    # ... existing types
    MY_NEW_TYPE = "my_new_type"
```

2. Add collector method in `vibetest/evidence/collector.py`:

```python
def add_my_evidence(
    self,
    data: MyDataType,
    description: str,
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Add my custom evidence type."""
    # Process and store data as needed
    evidence = Evidence(
        type=EvidenceType.MY_NEW_TYPE,
        description=description,
        data=data,
        metadata=metadata or {},
    )
    self.evidence_items.append(evidence)
    return evidence
```

### Testing

Currently, testing is done manually. In the future, we'll add:

```bash
# Run test suite (future)
uv run pytest

# Type checking (future)
uv run mypy vibetest

# Linting (future)
uv run ruff check .
```

For now, please manually test:
1. Your changes work with the CLI
2. Docker sandbox execution works
3. Evidence is collected correctly
4. Results are formatted properly

### Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Add docstrings to all new code
- Include examples for new features

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Test thoroughly
4. Update documentation
5. Commit with clear messages
6. Push and create a pull request
7. Ensure all checks pass

### Commit Messages

Use clear, descriptive commit messages:

```
Add TrainingLossTest for ML validation

- Implements test case for loss logging validation
- Creates plot generation for loss evidence
- Adds example usage in examples/
```

## Design Philosophy

When contributing, keep in mind:

1. **Extensibility**: Make it easy for users to add custom behavior
2. **Safety**: Default to sandboxed execution
3. **Simplicity**: Minimize boilerplate for common cases
4. **Observability**: Collect evidence and logs thoroughly
5. **Modularity**: Keep components loosely coupled

## Ideas for Contributions

### High Priority
- Add more ML-specific test cases
- Improve error handling and reporting
- Add comprehensive test suite
- Performance optimizations
- Better evidence visualization

### Medium Priority
- Support for batch test execution
- Test suite abstraction
- CI/CD integration examples
- Alternative sandbox implementations
- Caching layer for tool results

### Future Ideas
- Multi-agent evaluation
- Interactive test debugging
- Web UI for results
- Test case templates
- Auto-generated test suggestions

## Questions?

Feel free to:
- Open an issue for bugs or feature requests
- Start a discussion for questions
- Submit a draft PR for early feedback

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
