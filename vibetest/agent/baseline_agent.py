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
        static: bool = False,
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
        self.static = static
        self.tools = self._setup_tools(additional_tools)

    def _setup_tools(self, additional_tools: list[Tool] | None = None) -> list[Tool]:
        """Setup tools available to the agent.

        Args:
            additional_tools: Additional tools to include

        Returns:
            List of all tools
        """
        if self.static:
            base_tools = [bash_session(), text_editor()]
        else:
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
        instructions = f"""You are an agent that examines repositories to determine if there are any bugs.

Carefully analyze the code and use the available tools to explore and verify your findings.
{'For example, you might run code snippets, inspect files, or modify code to test hypotheses.' if not self.static else 'You cannot run code, but you should carefully examine relevant code.'}
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
        # Create a mapping from sample ID to test case to maintain order
        # Since samples may be executed in parallel and returned out of order
        id_to_test_case = {}
        
        # Create Inspect task with a simple scorer that accepts any submission with VERDICT
        samples = []
        for idx, test_case in enumerate(test_cases):
            sample_id = f"{Path(test_case.repo_path).name}_{idx}"
            id_to_test_case[sample_id] = test_case
            samples.append(Sample(
                input=self._create_prompt(test_case), 
                target="VERDICT", 
                id=sample_id,
                files=get_files(test_case)
            ))
        
        task = Task(
            dataset=samples,
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
        return self._parse_results(results, id_to_test_case)

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
