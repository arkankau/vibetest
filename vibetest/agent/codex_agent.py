"""Codex code review agent using Inspect AI agent bridge."""

from __future__ import annotations

import os
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.agent import AgentState, agent, sandbox_agent_bridge
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, user_prompt
from inspect_ai.scorer import includes
from inspect_ai.util import sandbox

from vibetest.agent.react_agent import get_files
from vibetest.testcases.base import TestCase, TestResult


@agent
def codex_review_agent(
    *,
    codex_cmd: str = "codex",
    codex_model: str | None = "inspect",
    codex_prompt: str = "/review",
    repo_root: str = "/workspace/repo",
    timeout_s: int = 1200,
    skip_git_check: bool = True,
):
    """Codex CLI agent that runs inside the sandbox and bridges via Inspect AI."""

    async def execute(state: AgentState) -> AgentState:
        prompt_text = user_prompt(state.messages).text or ""
        review_prompt = (
            f"{codex_prompt}\n\n"
            f"Repository: {repo_root}\n"
            "Focus on correctness bugs and risky behavior.\n"
        )
        if prompt_text:
            review_prompt += f"\nAdditional context:\n{prompt_text}\n"

        args = [codex_cmd, "exec"]
        if skip_git_check:
            args.append("--skip-git-repo-check")
        if codex_model:
            args.extend(["-m", codex_model])
        args.append(review_prompt)

        async with sandbox_agent_bridge(state) as bridge:
            env = {
                "OPENAI_BASE_URL": f"http://localhost:{bridge.port}/v1",
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "inspect"),
            }
            result = await sandbox().exec(args, env=env, timeout=timeout_s)
            if not result.success:
                raise RuntimeError(
                    f"Codex CLI failed: {result.returncode}\n{result.stderr or result.stdout}"
                )

            # If the bridge did not capture a completion, fall back to stdout.
            if not getattr(bridge.state, "output", None) and result.stdout:
                bridge.state.output = ModelOutput(
                    model="codex",
                    completion=result.stdout.strip(),
                )

            return bridge.state

    return execute


class CodexReviewAgent:
    """Agent wrapper that runs Codex review via Inspect AI sandbox bridge."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-5-mini",
        codex_cmd: str = "codex",
        codex_model: str | None = "inspect",
        codex_prompt: str = "/review",
        timeout_s: int = 1200,
        skip_git_check: bool = True,
    ):
        self.model_name = model or os.getenv("VIBETEST_MODEL", "openai/gpt-5-mini")
        self.codex_cmd = codex_cmd
        self.codex_model = codex_model
        self.codex_prompt = codex_prompt
        self.timeout_s = timeout_s
        self.skip_git_check = skip_git_check

    def _create_solver(self):
        return codex_review_agent(
            codex_cmd=self.codex_cmd,
            codex_model=self.codex_model,
            codex_prompt=self.codex_prompt,
            timeout_s=self.timeout_s,
            skip_git_check=self.skip_git_check,
        )

    def _create_prompt(self, test_case: TestCase) -> str:
        repo_root = "/workspace/repo"
        return f"Review the repository at {repo_root}."

    def execute_tests(
        self,
        test_cases: list[TestCase],
        *,
        sandbox: str | None = None,
    ) -> list[TestResult]:
        id_to_test_case: dict[str, TestCase] = {}
        samples = []
        for idx, test_case in enumerate(test_cases):
            sample_id = f"{Path(test_case.repo_path).name}_{idx}"
            id_to_test_case[sample_id] = test_case
            samples.append(
                Sample(
                    input=self._create_prompt(test_case),
                    target="REVIEW",
                    id=sample_id,
                    files=get_files(test_case),
                )
            )

        task = Task(
            dataset=samples,
            solver=self._create_solver(),
            scorer=includes(),
            sandbox=sandbox,
        )

        results = eval(
            tasks=task,
            model=self.model_name,
            log_dir="./logs",
            retry_on_error=1,
            fail_on_error=False,
        )

        return self._parse_results(results, id_to_test_case)

    def _parse_results(self, results, id_to_test_case: dict[str, TestCase]) -> list[TestResult]:
        sample_id_to_result: dict[str, TestResult] = {}

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

                    execution_log = ""
                    if sample.messages:
                        log_parts = []
                        for msg in sample.messages:
                            role = getattr(msg, "role", "unknown")
                            content = getattr(msg, "text", "") or getattr(msg, "content", "")
                            if content:
                                log_parts.append(f"[{role}] {str(content)[:200]}...")
                        execution_log = "\n".join(log_parts)

                    sample_id_to_result[sample_id] = TestResult(
                        test_case=test_case,
                        passed=bool(output.strip()),
                        message=output or "No review output captured.",
                        execution_log=execution_log,
                        metadata={
                            "model": self.model_name,
                            "codex_cmd": self.codex_cmd,
                            "codex_model": self.codex_model,
                            "codex_prompt": self.codex_prompt,
                        },
                    )

        test_results: list[TestResult] = []
        for sample_id, test_case in id_to_test_case.items():
            if sample_id in sample_id_to_result:
                test_results.append(sample_id_to_result[sample_id])
            else:
                test_results.append(
                    TestResult(
                        test_case=test_case,
                        passed=False,
                        message="Failed to execute Codex review or parse results",
                        metadata={
                            "error": "Execution or parsing failure",
                            "test_description": test_case.description,
                        },
                    )
                )

        return test_results
