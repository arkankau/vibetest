"""Baseline ReAct agent implementation using Inspect AI."""

import os
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.tool import Tool, bash_session, python, text_editor

from vibetest.testcases.base import TestCase, TestResult
from vibetest.agent.react_agent import get_files


class BaselineAgent:
    """Baseline ReAct agent for examining repositories for bugs.

    This agent uses a simple, unmodified prompt that asks to examine the repository
    and determine if there are any bugs. It serves as a baseline for comparison
    with more sophisticated agents.

    The agent uses Inspect AI's `react()` agent which implements the ReAct pattern:
    1. Reasons about what to do next (thinking step-by-step)
    2. Takes an action using a tool
    3. Observes the result
    4. Repeats until it has enough evidence to submit an answer
    """

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        max_attempts: int = 20,
        additional_tools: list[Tool] | None = None,
    ):
        """Initialize the baseline agent.

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
            bash_session(),
            python(),
            text_editor(),
        ]

        if additional_tools:
            base_tools.extend(additional_tools)

        return base_tools

    def _create_solver(self):
        """Create the ReAct solver for the agent.

        Uses Inspect AI's react() agent with minimal instructions.

        Returns:
            Solver configured with ReAct pattern
        """
        instructions = """You are an agent that examines repositories to determine if there are any bugs.

When you have completed your examination, call the submit() tool with your findings in this format:

VERDICT: [BUGS FOUND/NO BUGS FOUND]
EXPLANATION: [Your explanation of what you found]"""

        # Create the ReAct agent with built-in submit() tool
        agent = react(
            prompt=instructions,
            tools=self.tools,
            attempts=self.max_attempts,
            submit=True,
        )

        return agent

    def _create_prompt(self, test_case: TestCase) -> str:
        """Create the prompt for the agent to execute a test.

        Args:
            test_case: Test case to create prompt for

        Returns:
            Formatted prompt string - simply the repo name with the query
        """
        prompt = f"""Examine the repository and determine if there are any bugs.
Repository: /workspace/repos/{test_case.repo_path}"""
        return prompt

    def execute_tests(
        self, test_cases: list[TestCase], sandbox: str | None = None
    ) -> list[TestResult]:
        """Execute test cases using the baseline agent.

        Args:
            test_cases: List of test cases to execute
            sandbox: Sandbox environment type (e.g., "docker")

        Returns:
            List of TestResults with verdicts and evidence

        Note:
            This is a synchronous function even though it runs async operations internally.
            Inspect AI's eval() manages its own event loop, so we don't use async/await.
        """
        # Create Inspect task with a simple scorer that accepts any submission with VERDICT
        task = Task(
            dataset=[Sample(
                input=self._create_prompt(test_case), 
                target="VERDICT", 
                id=f"{Path(test_case.repo_path).name}_{idx % 9}",
                files=get_files(test_case)
            ) for idx, test_case in enumerate(test_cases)],
            solver=self._create_solver(),
            scorer=includes(),  # Accept any submission containing the target
            sandbox=sandbox,
        )

        # Run evaluation
        results = eval(
            tasks=task,
            model=self.model_name,
            log_dir="./logs",
            retry_on_error=2,
            fail_on_error=False,
        )

        # Parse results
        return self._parse_results(results, test_cases)

    def _parse_results(self, results, test_cases: list[TestCase]) -> list[TestResult]:
        """Parse Inspect AI results into TestResults.

        Args:
            results: Results from Inspect eval
            test_cases: Original test cases

        Returns:
            List of parsed TestResults
        """
        test_results = []

        # Extract results for each test case
        if results and len(results) > 0:
            eval_result = results[0]
            if eval_result.samples:
                # Iterate through samples and corresponding test cases
                for idx, sample in enumerate(eval_result.samples):
                    if idx >= len(test_cases):
                        break

                    test_case = test_cases[idx]

                    # Get the submitted answer
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
                    # For baseline agent, "NO BUGS FOUND" means passed, "BUGS FOUND" means failed
                    passed = "NO BUGS FOUND" in output.upper()
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

                    test_results.append(TestResult(
                        test_case=test_case,
                        passed=passed,
                        message=message,
                        execution_log=execution_log,
                        metadata={
                            "model": self.model_name,
                            "test_description": test_case.description,
                            "score": sample.score.value if sample.score else None,
                        },
                    ))

        # If no results were parsed, add fallback results for each test case
        if not test_results:
            for test_case in test_cases:
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
