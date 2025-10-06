# Vibetest Architecture

## Overview

Vibetest is built on a modular, extensible architecture designed for running AI agents that validate codebases against natural language test cases.

## Core Components

### 1. Test Cases (`vibetest/testcases/`)

**Base Classes:**
- `TestCase`: Abstract base for all test cases
  - Holds test description and repository path
  - Provides `execute()` method (must be implemented)
  - Generates prompts via `get_prompt()`

- `TestResult`: Encapsulates test outcomes
  - Binary pass/fail verdict
  - Message explaining the result
  - Evidence list
  - Execution log and metadata

- `Evidence`: Represents collected evidence
  - Type (plot, log, metrics, code snippet, etc.)
  - Description and data
  - Extensible metadata

**Concrete Implementations:**
- `TrainingLossTest`: Validates loss logging and decrease
- `HyperparameterLoggingTest`: Checks hyperparameter logging

### 2. Agent (`vibetest/agent/`)

**VibeTestAgent:**
- Wraps Inspect AI's ReAct implementation
- Configurable model and tools
- Executes tests in sandboxed environments
- Parses agent output into TestResults

**Key Methods:**
- `execute_test(test_case, sandbox)`: Run a test case
- `add_tool(tool)`: Dynamically add tools
- `remove_tool(name)`: Remove tools

### 3. Tools (`vibetest/tools/`)

Tools are the agent's interface to the environment. All tools are Inspect AI `@tool` decorated functions.

**File Tools:**
- `read_file`: Read file contents
- `list_directory`: List directory contents
- `find_files`: Search for files by pattern

**Execution Tools:**
- `run_python`: Execute Python code
- `run_command`: Run shell commands

**Analysis Tools:**
- `create_plot`: Generate plots from data
- `parse_logs`: Extract information from logs using regex

**Extensibility:**
- Tools can be added via `VibeTestAgent.add_tool()`
- Create custom tools with `@tool` decorator
- Tools are automatically documented for the agent

### 4. Evidence Collection (`vibetest/evidence/`)

**EvidenceCollector:**
- Manages evidence storage
- Organizes files in output directory
- Generates evidence manifests
- Supports multiple evidence types

**Methods:**
- `add_plot()`: Store plot files
- `add_log()`: Store log text
- `add_metrics()`: Store metric dictionaries
- `add_code_snippet()`: Store relevant code
- `save_manifest()`: Create JSON manifest

### 5. Configuration (`vibetest/config.py`)

**VibeTestConfig:**
- Centralized configuration using Pydantic
- Environment variable integration
- Default values for all settings

**Key Settings:**
- Model selection
- Sandbox type
- Execution timeouts
- Storage directories
- API keys

### 6. CLI (`vibetest/cli.py`)

Command-line interface for running tests:
- Argument parsing
- Test case instantiation
- Result formatting and display
- Output file management

## Data Flow

```
1. User creates TestCase
   ↓
2. TestCase.execute() called
   ↓
3. VibeTestAgent.execute_test() runs
   ↓
4. Inspect AI Task created with:
   - Test prompt
   - Tool list
   - Sandbox config
   ↓
5. ReAct loop executes:
   - Agent reasons about task
   - Calls tools (read files, run code, etc.)
   - Iterates until conclusion
   ↓
6. Agent output parsed
   ↓
7. TestResult created with:
   - Pass/fail verdict
   - Message
   - Evidence
   ↓
8. Results returned to user
```

## Sandbox Integration

Vibetest uses Inspect AI's sandbox abstraction:

- **Docker Sandbox**: Isolated container execution
  - Safe code execution
  - Clean environment per test
  - Volume mounting for repo access

- **Configuration**: Via `sandbox` parameter in `execute_test()`
- **Dockerfile**: Defines the execution environment

## Extension Points

### 1. Custom Test Cases
Subclass `TestCase` and implement:
- `__init__`: Set description and parameters
- `execute()`: Define execution logic
- `get_prompt()`: Customize agent instructions

### 2. Custom Tools
Create new tools with `@tool` decorator:
```python
from inspect_ai.tool import tool

@tool
def my_tool(param: str) -> str:
    """Tool description for the agent."""
    # Implementation
    return result
```

### 3. Custom Evidence Types
Add to `EvidenceType` enum and create collector methods:
```python
class EvidenceType(str, Enum):
    # ... existing types
    CUSTOM_TYPE = "custom_type"

# In EvidenceCollector:
def add_custom_evidence(self, data, description):
    evidence = Evidence(
        type=EvidenceType.CUSTOM_TYPE,
        description=description,
        data=data
    )
    self.evidence_items.append(evidence)
    return evidence
```

### 4. Custom Agent Behavior
Modify `VibeTestAgent._create_solver()`:
- Change system prompt
- Adjust max_attempts
- Add custom solvers to pipeline

### 5. Alternative Models
Set via config or environment:
```python
agent = VibeTestAgent(model="openai/gpt-4")
# or
export VIBETEST_MODEL="openai/gpt-4"
```

## Design Principles

1. **Extensibility**: Easy to add test cases, tools, and evidence types
2. **Modularity**: Clear separation of concerns
3. **Safety**: Sandboxed execution by default
4. **Observability**: Comprehensive logging and evidence collection
5. **Configurability**: Override defaults via code or environment
6. **Simplicity**: Minimal boilerplate for common use cases

## Future Architecture Considerations

Potential additions:
- Parallel test execution
- Test suite abstraction
- Result aggregation and reporting
- CI/CD integration hooks
- Multi-agent evaluation
- Caching layer for tool results
- Progressive evidence collection during execution
