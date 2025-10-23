"""ReAct agent implementation using Inspect AI."""

import os
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes, scorer, Score
from inspect_ai.tool import Tool, bash_session, python, text_editor, bash, web_search
from inspect_ai.util import sandbox, SandboxEnvironmentSpec
from inspect_ai.scorer import Target, accuracy
from inspect_ai.solver import TaskState

from vibetest.testcases.base import TestCase, TestResult
from vibetest.config import get_package_root


@scorer(metrics=[accuracy()])
def save_evidence_tar(out_dir: str | os.PathLike = "./evidence-dumps", *, dir_to_save="/evidence"):
    """
    Creates /tmp/evidence-<sample>.tar.gz inside the sandbox, pulls it out,
    and writes it under out_dir on the host as evidence-<sample>.tar.gz.
    """
    async def _score(state: TaskState, target: Target) -> Score:
        env = sandbox()  # SandboxEnvironment for the current sample

        # Name artifact using sample_id from the TaskState
        sample_id = state.sample_id if hasattr(state, "sample_id") else "unknown"
        tar_in_sandbox = f"/tmp/evidence-{sample_id}.tar.gz"

        # Best-effort: tar up the directory if it exists (don't fail if it's missing)
        # -C / makes the archive paths absolute-looking but rooted properly
        await env.exec(["bash", "-lc", f"if [ -d '{dir_to_save}' ]; then tar -czf '{tar_in_sandbox}' -C / '{dir_to_save.lstrip('/')}' ; fi || true"])

        # Try to read the tarball back; if it wasn't created, just return a benign score
        try:
            blob = await env.read_file(tar_in_sandbox, text=False)  # returns bytes
        except Exception:
            return Score(value=True, explanation=f"No evidence found at {dir_to_save}")

        # Write to the host filesystem (e.g., alongside your logs)
        out_base = Path(out_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        out_path = out_base / f"evidence-{sample_id}.tar.gz"
        out_path.write_bytes(blob)
        
        # Set proper permissions (0o644 = rw-r--r--)
        os.chmod(out_path, 0o644)

        return Score(value=True, explanation=f"Saved evidence to {out_path}")
    return _score


def get_files(test_case: TestCase, sandbox_prefix="/workspace/repos/") -> dict[str, str]:
    """Get files from the test case repository.

    Args:
        test_case: Test case containing the repository path
    Returns:        Dictionary mapping file path in the sandbox to file path
    """
    files = {}
    repo_path = test_case.repo_path
    if repo_path and os.path.isdir(repo_path):
        for root, _, filenames in os.walk(repo_path):
            if ".venv" in root or "__pycache__" in root:
                continue  # Skip virtual environments and cache directories
            for filename in filenames:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, repo_path)
                sandbox_path = os.path.join(sandbox_prefix, repo_path, relative_path)
                files[sandbox_path] = full_path
    return files


def cleanup_docker_sandbox() -> None:
    """Clean up temporary Docker configuration files."""
    temp_dir = Path.cwd() / ".vibetest_tmp"
    if temp_dir.exists():
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Best effort cleanup


def setup_docker_sandbox() -> str | SandboxEnvironmentSpec:
    """Setup Docker sandbox configuration to use vibetest's Dockerfile and compose.yaml.

    Returns:
        SandboxEnvironmentSpec configured to use vibetest's Docker configuration.
        Creates a temporary compose.yaml in the current directory with correct paths.
    """
    import tempfile
    import yaml

    package_root = get_package_root()
    dockerfile_path = package_root / "Dockerfile"
    source_compose_path = package_root / "compose.yaml"

    if not dockerfile_path.exists():
        raise FileNotFoundError(
            f"Dockerfile not found at {dockerfile_path}. "
            "Please ensure vibetest is properly installed."
        )

    # Create a temporary compose.yaml in the current directory with correct build context
    # Inspect AI will look for compose.yaml in the config directory
    temp_dir = Path.cwd() / ".vibetest_tmp"
    temp_dir.mkdir(exist_ok=True)
    temp_compose_path = temp_dir / "compose.yaml"

    # Read the source compose.yaml and update the build context
    if source_compose_path.exists():
        with open(source_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)

        # Update build context to point to package root
        if 'services' in compose_config and 'default' in compose_config['services']:
            if isinstance(compose_config['services']['default'].get('build'), dict):
                compose_config['services']['default']['build']['context'] = str(package_root)
                compose_config['services']['default']['build']['dockerfile'] = "Dockerfile"
            elif compose_config['services']['default'].get('build') == '.':
                compose_config['services']['default']['build'] = {
                    'context': str(package_root),
                    'dockerfile': 'Dockerfile'
                }

        # Write temporary compose.yaml
        with open(temp_compose_path, 'w') as f:
            yaml.dump(compose_config, f)
    else:
        # Create a basic compose.yaml if source doesn't exist
        compose_config = {
            'services': {
                'default': {
                    'build': {
                        'context': str(package_root),
                        'dockerfile': 'Dockerfile'
                    },
                    'init': True,
                    'command': 'tail -f /dev/null',
                    'deploy': {
                        'resources': {
                            'reservations': {
                                'devices': [{
                                    'driver': 'nvidia',
                                    'count': 1,
                                    'capabilities': ['gpu']
                                }]
                            }
                        }
                    }
                }
            }
        }
        with open(temp_compose_path, 'w') as f:
            yaml.dump(compose_config, f)

    # Return the path to the temporary compose.yaml file
    # Inspect AI expects the file path, not the directory
    return SandboxEnvironmentSpec(type="docker", config=str(temp_compose_path))


class VibeTestAgent:
    """ReAct agent for executing natural language test cases.

    This agent uses Inspect AI's `react()` agent which implements the ReAct pattern

    The agent iteratively:
    1. Reasons about what to do next (thinking step-by-step)
    2. Takes an action using a tool
    3. Observes the result
    4. Repeats until it has enough evidence to submit an answer

    This is a proper ReAct implementation, not just tool-calling.
    """

    def __init__(
        self,
        model=None,
        max_attempts: int = 20,
        additional_tools: list[Tool] | None = None,
    ):
        """Initialize the agent.

        Args:
            model: Model to use (e.g., "anthropic/claude-3-5-sonnet-20241022")
            max_attempts: Maximum reasoning/action attempts
            additional_tools: Extra tools to add beyond defaults
        """
        self.model_name = model or os.getenv(
            "VIBETEST_MODEL", "openai/gpt-5-mini"
        )
        self.max_attempts = max_attempts
        self.tools = self._setup_tools(additional_tools)

    def _setup_tools(self, additional_tools: list[Tool] | None = None) -> list[Tool]:
        """Setup tools available to the agent.

        Args:
            additional_tools: Additional tools to include

        Returns:
            List of all tools
        """
        base_tools = [
            # bash_session(),
            bash(),
            python(),
            text_editor(),
            web_search(),
        ]

        if additional_tools:
            base_tools.extend(additional_tools)

        return base_tools

    def _create_solver(self):
        """Create the ReAct solver for the agent.

        Uses Inspect AI's react() agent

        The agent alternates between:
        1. Reasoning about what to do next
        2. Taking an action with a tool
        3. Observing the result
        4. Repeating until it submits an answer via the submit() tool

        Returns:
            Solver configured with ReAct pattern
        """
        instructions = """# Role
You are an expert software testing agent that evaluates existing codebases against natural-language test criteria. You gather objective evidence and return a binary verdict.

# Objective
Determine whether the repository PASSes or FAILs the specified test case, and produce verifiable evidence.

# Inputs:
- TEST_CASE (text): Natural-language condition(s) to evaluate.
- REPO_ROOT (path): Filesystem path to the repository.

# Operating Rules
1. Evaluate, don't rewrite. Avoid substantial new code. Prefer instrumentation (logging, flags, CLI args, small patches ≤ ~50 lines total). Record all edits as diffs.
2. Evidence over opinion. Prefer runtime traces, logs, metrics, file hashes, config snapshots, git SHAs, and small data extracts.
3. Data availability. Check for required datasets locally (e.g., /kaggle, mounted volumes) before downloading. If data is missing, look for directions for downloading it.
4. Environment setup. Set up an environment (uv is installed) and install any necessary dependencies.
5. Determinism where possible. Capture python -V, CUDA/cuDNN, pip freeze/conda list, git rev-parse HEAD, and relevant seeds.
6. Use default parameters. Run code with default settings unless the test case requires otherwise.

# PASS/FAIL Rubric
- PASS: You found direct evidence satisfying the TEST_CASE in the target repo (e.g., successful run producing expected metrics/logs/behaviors) without violating the rules above.
- FAIL: You found contradictory evidence, or you cannot obtain required evidence after reasonable attempts (missing code, irreparable errors, non-reproducible steps, unresolvable dependencies, or the repo implements the opposite behavior). Explain why.

# Workflow
## Phase 0 — Initialize
- Normalize TEST_CASE into concrete checks (files, entry points, commands, artifacts).  

## Phase 1 — Recon
- Map the repo: `README`, `requirements*`, `environment.yml`, `pyproject.toml`, entry points (`main.py`, `train.py`, `eval.py`), notebooks, configs.  
- Search for relevant code fragments (e.g., "loss", "evaluation", flags).  
- Locate data locally before downloading.

## Phase 2 — Setup
- Create isolated environment and install dependencies. Make sure to use the correct python version which can be configured with uv or conda. (uv is pre-installed)
- Convert notebooks via `jupyter nbconvert --to script`.  
- Apply minimal patches if necessary.

## Phase 3 — Execute & Instrument
- Run the **minimal** reproducible command(s) to demonstrate or refute the property. Do not change default hyperparameters such as epochs, batch size, or learning rate unless the test case explicitly requires it.
- Capture logs, metrics, and outputs in `/evidence/artifacts/`.

## Phase 4 — Corroborate
- Cross-check signals (logs + metrics + artifacts).  
- Prefer runtime evidence over static inspection.

## Phase 5 — Decide
- Apply the PASS/FAIL rubric and cite concrete artifact-based evidence.

# Output Format
When you have enough evidence to make a determination, call the submit() tool with your final answer in this format:

VERDICT: [PASS/FAIL]
REASON: [Brief explanation of why]
EVIDENCE: [Description of evidence collected]

Remember: You MUST use the submit() tool to report your final answer."""

        # Create the ReAct agent with built-in submit() tool
        # The react() function returns a solver that can be used directly
        agent = react(
            prompt=instructions,
            tools=self.tools,
            attempts=self.max_attempts,
            submit=True,  # Explicitly enable submit tool (True by default)
        )

        return agent

    def _create_prompt(self, test_case: TestCase) -> str:
        """Create the prompt for the agent to execute a test.

        Args:
            test_case: Test case to create prompt for

        Returns:
            Formatted prompt string
        """
        prompt = f"""Here is the test case and the repository to evaluate:
Test: {test_case.description}
Repository: /workspace/repos/{test_case.repo_path}"""
        return prompt

    def execute_tests(
        self, test_cases: list[TestCase], sandbox: str | None = None
    ) -> list[TestResult]:
        """Execute a test case using the agent.

        Args:
            test_case: Test case to execute
            sandbox: Sandbox environment type (e.g., "docker"). If "docker" is specified,
                    vibetest will automatically use its packaged Dockerfile.

        Returns:
            TestResult with verdict and evidence

        Note:
            This is a synchronous function even though it runs async operations internally.
            Inspect AI's eval() manages its own event loop, so we don't use async/await.
        """
        # Create a mapping from sample ID to test case to maintain order
        # Since samples may be executed in parallel and returned out of order
        id_to_test_case = {}

        # Setup sandbox configuration
        sandbox_config = None
        if sandbox == "docker":
            sandbox_config = setup_docker_sandbox()
        elif sandbox is not None:
            sandbox_config = sandbox

        # Create Inspect task with a scorer
        # The scorer is required when using submit() tool in react() agent
        # We use includes() to accept any submission that contains "VERDICT"
        samples = []
        for idx, test_case in enumerate(test_cases):
            sample_id = f"{Path(test_case.repo_path).name}_{idx}"
            id_to_test_case[sample_id] = test_case
            samples.append(Sample(
                input=self._create_prompt(test_case),
                id=sample_id,
                files=get_files(test_case)
            ))

        task = Task(
            dataset=samples,
            solver=self._create_solver(),
            scorer=save_evidence_tar(f"./evidence-dumps/{self.model_name.split('/')[1]}"), #includes(),  # Accept any answer containing the target
            sandbox=sandbox_config,
        )

        # Run evaluation
        try:
            results = eval(
                tasks=task,
                model=self.model_name,
                log_dir="./logs",  # Must be string, not Path
                retry_on_error=2,
                fail_on_error=False,
            )

            # Parse results
            return self._parse_results(results, id_to_test_case)
        finally:
            # Clean up temporary Docker configuration
            if sandbox == "docker":
                cleanup_docker_sandbox()

    def _parse_results(self, results, id_to_test_case: dict[str, TestCase]) -> list[TestResult]:
        """Parse Inspect AI results into TestResults.

        Args:
            results: Results from Inspect eval
            id_to_test_case: Mapping from sample ID to test case

        Returns:
            List of parsed TestResults in the original test case order
        """
        # Create a mapping from sample ID to parsed result
        sample_id_to_result = {}

        # Extract results for each sample
        if results and len(results) > 0:
            eval_result = results[0]
            if eval_result.samples:
                # Iterate through samples and match to test cases using IDs
                for sample in eval_result.samples:
                    sample_id = sample.id
                    
                    # Skip if we don't have a matching test case
                    if sample_id not in id_to_test_case:
                        continue
                    
                    test_case = id_to_test_case[sample_id]

                    # Get the submitted answer (from basic_agent's submit tool)
                    output = ""
                    if sample.output and sample.output.completion:
                        output = sample.output.completion

                    # Also check messages for the final submission
                    if sample.messages:
                        for msg in reversed(sample.messages):
                            if hasattr(msg, 'text') and msg.text:
                                output = msg.text
                                break

                    # Parse verdict from output
                    passed = "VERDICT: PASS" in output.upper()
                    message = output

                    # Build execution log from message history
                    execution_log = ""
                    if sample.messages:
                        log_parts = []
                        for msg in sample.messages:
                            role = getattr(msg, 'role', 'unknown')
                            content = getattr(msg, 'text', '') or getattr(msg, 'content', '')
                            if content:
                                log_parts.append(f"[{role}] {content[:200]}...")
                        execution_log = "\n".join(log_parts)

                    sample_id_to_result[sample_id] = TestResult(
                        test_case=test_case,
                        passed=passed,
                        message=message,
                        execution_log=execution_log,
                        metadata={
                            "model": self.model_name,
                            "test_description": test_case.description,
                            "score": sample.score.value if sample.score else None,
                        },
                    )

        # Build final results list in the original test case order
        test_results = []
        for sample_id, test_case in id_to_test_case.items():
            if sample_id in sample_id_to_result:
                test_results.append(sample_id_to_result[sample_id])
            else:
                # Add fallback result for missing test case
                test_results.append(TestResult(
                    test_case=test_case,
                    passed=False,
                    message="Failed to execute test or parse results",
                    metadata={
                        "error": "Execution or parsing failure",
                        "test_description": test_case.description
                    },
                ))

        return test_results
