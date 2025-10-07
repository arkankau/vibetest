"""ReAct agent implementation using Inspect AI."""

import os
from pathlib import Path
from typing import Optional, List

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.agent import react, as_solver
from inspect_ai.scorer import match, includes
from inspect_ai.tool import Tool, bash_session, python, text_editor

from vibetest.testcases.base import TestCase, TestResult
# from vibetest.tools import (
#     read_file,
#     list_directory,
#     find_files,
#     run_python,
#     run_command,
#     create_plot,
#     parse_logs,
# )


class VibeTestAgent:
    """ReAct agent for executing natural language test cases.

    This agent uses Inspect AI's `react()` agent which implements the ReAct
    pattern from the paper "ReAct: Synergizing Reasoning and Acting in Language
    Models" (https://arxiv.org/abs/2210.03629).

    The agent iteratively:
    1. Reasons about what to do next (thinking step-by-step)
    2. Takes an action using a tool
    3. Observes the result
    4. Repeats until it has enough evidence to submit an answer

    This is a proper ReAct implementation, not just tool-calling.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_attempts: int = 20,
        additional_tools: Optional[list[Tool]] = None,
    ):
        """Initialize the agent.

        Args:
            model: Model to use (e.g., "anthropic/claude-3-5-sonnet-20241022")
            max_attempts: Maximum reasoning/action attempts
            additional_tools: Extra tools to add beyond defaults
        """
        self.model_name = model or os.getenv(
            "VIBETEST_MODEL", "openai/gpt-5-nano"
        )
        self.max_attempts = max_attempts
        self.tools = self._setup_tools(additional_tools)

    def _setup_tools(self, additional_tools: Optional[list[Tool]] = None) -> list[Tool]:
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
            # read_file,
            # list_directory,
            # find_files,
            # run_python,
            # run_command,
            # create_plot,
            # parse_logs,
        ]

        if additional_tools:
            base_tools.extend(additional_tools)

        return base_tools

    def _create_solver(self):
        """Create the ReAct solver for the agent.

        Uses Inspect AI's react() agent which implements the ReAct pattern
        from the paper "ReAct: Synergizing Reasoning and Acting in Language Models"
        (https://arxiv.org/abs/2210.03629).

        The agent alternates between:
        1. Reasoning about what to do next
        2. Taking an action with a tool
        3. Observing the result
        4. Repeating until it submits an answer via the submit() tool

        Returns:
            Solver configured with ReAct pattern
        """
        instructions = """You are an expert testing agent that evaluates codebases against natural language test criteria.

Your approach should be:
1. Understand what needs to be tested
2. Systematically explore the codebase
3. Execute necessary code to gather evidence (you must install necessary dependencies and data)
4. Analyze results objectively
5. Provide a clear binary verdict (PASS or FAIL) with supporting evidence

Available tools let you:
- Read files and explore directories
- Run Python code and shell commands
- Create plots and parse logs

Think step-by-step about what information you need, then use tools to gather evidence.

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
        prompt = f"""You are a testing agent evaluating a codebase against the following test case:

Test: {test_case.description}
Repository: {test_case.repo_path}

Your task is to:
1. Understand the codebase structure
2. Execute necessary code to gather evidence
3. Analyze the results against the test criteria
4. Provide a binary pass/fail verdict with supporting evidence

Use the available tools to explore the repository, run code, and collect evidence.
Be thorough and systematic in your analysis.

When you reach a conclusion, respond with:
VERDICT: [PASS/FAIL]
REASON: [Brief explanation]
EVIDENCE: [Description of evidence collected]
"""
        return prompt

    def execute_tests(
        self, test_cases: List[TestCase], sandbox: Optional[str] = None
    ) -> TestResult:
        """Execute a test case using the agent.

        Args:
            test_case: Test case to execute
            sandbox: Sandbox environment type (e.g., "docker")

        Returns:
            TestResult with verdict and evidence

        Note:
            This is a synchronous function even though it runs async operations internally.
            Inspect AI's eval() manages its own event loop, so we don't use async/await.
        """
        # Create Inspect task with a scorer
        # The scorer is required when using submit() tool in react() agent
        # We use includes() to accept any submission that contains "VERDICT"
        task = Task(
            dataset=[Sample(input=self._create_prompt(test_case), target="VERDICT") for test_case in test_cases],
            solver=self._create_solver(),
            scorer=includes(),  # Accept any answer containing the target
            sandbox=sandbox,
        )

        # Run evaluation
        results = eval(
            tasks=task,
            model=self.model_name,
            log_dir="./logs",  # Must be string, not Path
        )

        # Parse results
        return self._parse_results(results, test_cases)

    def _parse_results(self, results, test_cases: List[TestCase]) -> List[TestResult]:
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

                    test_results.append(TestResult(
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
                    passed=False,
                    message="Failed to execute test or parse results",
                    metadata={
                        "error": "Execution or parsing failure",
                        "test_description": test_case.description
                    },
                ))

        return test_results

    def add_tool(self, tool: Tool) -> None:
        """Add a new tool to the agent.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool by name.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if removed, False if not found
        """
        initial_len = len(self.tools)
        self.tools = [t for t in self.tools if t.name != tool_name]
        return len(self.tools) < initial_len
