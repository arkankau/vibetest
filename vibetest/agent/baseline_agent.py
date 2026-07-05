"""Baseline ReAct agent implementation using Inspect AI."""

import os
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.agent import AgentAttempts, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.tool import Tool, bash_session, python, text_editor

from vibetest.testcases.base import TestCase, TestResult
from vibetest.agent.react_agent import (
    _parse_submission_output,
    cleanup_archive_temps,
    create_files_archive,
)
from vibetest.usage import usage_payload_from_sample


def _always_continue_score_value(_value) -> float:
    return 0.0


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
        max_sandboxes: int = 10,
        reviewer_reprompts: int = 0,
        findings_json: bool = False,
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
        self.max_sandboxes = max_sandboxes
        self.reviewer_reprompts = max(0, reviewer_reprompts)
        self.findings_json = findings_json
        self.tools = self._setup_tools(additional_tools)

    def _model_extra_body(self) -> dict | None:
        model_name = (self.model_name or "").lower()
        if "qwen" in model_name:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    def _model_prompt_prefix(self) -> str:
        model_name = (self.model_name or "").lower()
        if "qwen" in model_name:
            return "/no_think\n"
        return ""

    def _setup_tools(self, additional_tools: list[Tool] | None = None) -> list[Tool]:
        """Setup tools available to the agent.

        Args:
            additional_tools: Additional tools to include

        Returns:
            List of all tools
        """
        if self.static:
            base_tools = [bash_session(timeout=210), text_editor()]
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
        analysis_instruction = (
            "Use static inspection only. Do not run repository code, notebooks, tests, training scripts, or user code. "
            "You may use shell/text tools only to list, search, and read files."
            if self.static
            else "Carefully analyze the code and use the available tools to explore and verify your findings. "
            "For example, you might run code snippets, inspect files, or modify code to test hypotheses."
        )
        if self.findings_json:
            instructions = f"""You are a code review agent that examines repositories to identify bugs affecting the code's correctness.

{analysis_instruction}
Output all findings that the original author would fix if they knew about it. If there is no finding that a person would definitely love to see and fix, prefer outputting no findings. Do not stop at the first qualifying finding. Continue until you've listed every qualifying finding.

When you have completed your examination, call the submit() tool with only compact JSON in this format:

{{"findings":[{{"title":"Short bug title","body":"Why this is a bug, with concrete file/line/function citations.","confidence":0.0}}]}}

Rules:
- Each finding must describe a concrete bug, not a best-practice suggestion.
- confidence must be a float from 0.0 to 1.0 for how confident you are that the finding is a real bug.
- If there are no qualifying findings, submit {{"findings":[]}}."""
        else:
            instructions = f"""You are a code review agent that examines repositories to determine if there are any bugs affecting the code's correctness.

{analysis_instruction}
After examining the code, make a final verdict based on if any bugs were found. The verdict should be PASS if there are no bugs impacting correctness, FAIL if there is at least one bug impacting code correctness, or INCONCLUSIVE if additional information is required to conclusively determine the code's correctness.
When you have completed your examination, call the submit() tool with your findings in this format:

VERDICT: [PASS/FAIL/INCONCLUSIVE]
EXPLANATION: [Your explanation of what you found]"""

        attempts: int | AgentAttempts
        if self.reviewer_reprompts:
            attempts = AgentAttempts(
                attempts=self.reviewer_reprompts + 1,
                incorrect_message=(
                    "Do not stop yet. Keep looking and recheck your result. "
                    "Do not run repository code, notebooks, tests, training scripts, or user code. "
                    "When you are done, submit your updated review."
                ),
                score_value=_always_continue_score_value,
            )
        else:
            attempts = self.max_attempts

        # Create the ReAct agent with built-in submit() tool
        agent = react(
            prompt=instructions,
            tools=self.tools,
            attempts=attempts,
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
        repo_root = f"{test_case.sandbox_path.rstrip('/')}/repo"
        context = test_case.description.strip() if test_case.description else ""
        context_block = f"\n\nReview context and important guidelines:\n{context}" if context else ""
        prompt = f"""Examine the repository and determine if there are any bugs.
Repository: {repo_root}{context_block}"""
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
        target = "findings" if self.findings_json else "VERDICT"
        for idx, test_case in enumerate(test_cases):
            sample_id = f"{Path(test_case.repo_path).name}_{idx}"
            id_to_test_case[sample_id] = test_case
            files_dict, setup_script = create_files_archive(
                test_case, sandbox_prefix=test_case.sandbox_path
            )
            samples.append(Sample(
                input=self._create_prompt(test_case),
                target=target,
                id=sample_id,
                files=files_dict,
                setup=setup_script,
            ))

        task = Task(
            dataset=samples,
            solver=self._create_solver(),
            scorer=includes(),  # Accept any submission containing the target
            sandbox=sandbox,
        )

        try:
            # Run evaluation
            results = eval(
                tasks=task,
                model=self.model_name,
                extra_body=self._model_extra_body(),
                log_dir="./logs",
                retry_on_error=2,
                fail_on_error=False,
                max_sandboxes=self.max_sandboxes,
            )

            # Parse results
            return self._parse_results(results, id_to_test_case)
        finally:
            cleanup_archive_temps()

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
                            "total_time": getattr(sample, "total_time", None),
                            "working_time": getattr(sample, "working_time", None),
                            **usage_payload_from_sample(sample),
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


class DirectPropertyAgent(BaselineAgent):
    """Per-property tool-using baseline with no VibeTest calibration examples."""

    def _create_solver(self):
        analysis_instruction = (
            "Use static inspection only. Do not run repository code, notebooks, tests, training scripts, or user code. "
            "You may use shell/text tools only to list, search, and read files."
            if self.static
            else "Carefully analyze the repository with the available tools. You may run focused commands when useful."
        )
        instructions = f"""You are a direct property-testing agent. Evaluate exactly one natural-language property for one repository.

{analysis_instruction}

Return FAIL only when you find concrete repository evidence that the property is violated. Return PASS when the inspected repository satisfies the property, or when the property is conditional and the triggering condition is absent with no violation evidence. For example, if a property says "if there is model selection or hyperparameter tuning" and you find no model selection or hyperparameter tuning, that condition is not a reason for INCONCLUSIVE. Return INCONCLUSIVE only when the relevant repository code/data is missing, too weak, ambiguous, or not inspectable enough to judge the property.

Evaluate only the current property. Repositories may violate nearby or related properties, but do not return FAIL unless the evidence directly violates the property above. If you find a bug that is better described by a different property, mention it only if it also directly violates the current property.

case_score is a float between 0.0 and 1.0 representing your confidence that the repository violates the property. Higher means more confident the repository violates the property; lower means more confident the repository satisfies the property. If completely confident the verdict is PASS, use 0.0 to 0.2. Use 0.2 to 0.5 if it seems like PASS but the evidence is incomplete, so the verdict is probably INCONCLUSIVE. Use 0.5 to 0.8 if it seems like FAIL but the evidence is incomplete, so the verdict is probably INCONCLUSIVE. If sure the verdict is FAIL, use 0.8 to 1.0.

When finished, call submit() with exactly this format:

VERDICT: [PASS/FAIL/INCONCLUSIVE]
CASE_SCORE: [0.00-1.00]
EVIDENCE_STRENGTH: [0.00-1.00]
REASON: [One to three concise sentences.]
EVIDENCE: [Concrete file/function/cell/line references or commands inspected.]"""

        return react(
            prompt=instructions,
            tools=self.tools,
            attempts=self.max_attempts,
            submit=True,
        )

    def _create_prompt(self, test_case: TestCase) -> str:
        repo_root = f"{test_case.sandbox_path.rstrip('/')}/repo"
        return self._model_prompt_prefix() + f"""Repository: {repo_root}

Property to evaluate:
{test_case.description.strip()}

Inspect the repository and decide whether this one property is satisfied. Start by inspecting the exact repository path above, for example with `ls -la {repo_root}`; do not infer repository contents from the shell's current working directory."""

    def _parse_results(self, results, id_to_test_case: dict[str, TestCase]) -> list[TestResult]:
        sample_id_to_result = {}

        if results and len(results) > 0:
            eval_result = results[0]
            if eval_result.samples:
                for sample in eval_result.samples:
                    sample_id = sample.id
                    if sample_id not in id_to_test_case:
                        continue

                    test_case = id_to_test_case[sample_id]
                    output = ""
                    if sample.output and sample.output.completion:
                        output = sample.output.completion
                    if sample.messages:
                        for msg in reversed(sample.messages):
                            if hasattr(msg, "text") and msg.text:
                                output = msg.text
                                break

                    verdict, case_score, evidence_strength, reason_text, evidence_text = (
                        _parse_submission_output(output)
                    )
                    if not verdict:
                        verdict = "INCONCLUSIVE"

                    execution_log = ""
                    if sample.messages:
                        log_parts = []
                        for msg in sample.messages:
                            role = getattr(msg, "role", "unknown")
                            content = getattr(msg, "text", "") or getattr(msg, "content", "")
                            if content:
                                log_parts.append(f"[{role}] {content[:200]}...")
                        execution_log = "\n".join(log_parts)

                    sample_id_to_result[sample_id] = TestResult(
                        test_case=test_case,
                        passed=verdict == "PASS",
                        message=output,
                        execution_log=execution_log,
                        metadata={
                            "model": self.model_name,
                            "test_description": test_case.description,
                            "verdict": verdict,
                            "case_score": case_score,
                            "fail_support_score": case_score,
                            "evidence_strength": evidence_strength,
                            "reason_text": reason_text,
                            "evidence_text": evidence_text,
                            "score": sample.score.value if sample.score else None,
                            "total_time": getattr(sample, "total_time", None),
                            "working_time": getattr(sample, "working_time", None),
                            **usage_payload_from_sample(sample),
                        },
                    )

        test_results = []
        for sample_id, test_case in id_to_test_case.items():
            if sample_id in sample_id_to_result:
                test_results.append(sample_id_to_result[sample_id])
            else:
                test_results.append(
                    TestResult(
                        test_case=test_case,
                        passed=False,
                        message="Failed to execute test or parse results",
                        metadata={
                            "error": "Execution or parsing failure",
                            "verdict": "INCONCLUSIVE",
                            "test_description": test_case.description,
                        },
                    )
                )

        return test_results
