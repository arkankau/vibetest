"""Claude Code agents backed by inspect_swe's Claude Code integration."""

from __future__ import annotations

from inspect_swe import claude_code

from vibetest.agent.codex_agent import (
    _CODEX_REVIEW_INSTRUCTIONS,
    _CodexAgentBase,
    _build_execution_log,
    _extract_output,
)
from vibetest.agent.react_agent import (
    _parse_submission_output,
    build_agent_instructions,
    build_test_case_prompt,
)
from vibetest.testcases.base import TestCase, TestResult
from vibetest.usage import usage_payload_from_sample


class _ClaudeCodeAgentBase(_CodexAgentBase):
    def _backend_agent(self, system_prompt: str):
        return claude_code(
            name=self.__class__.__name__,
            system_prompt=system_prompt,
            model=self.model_name,
            cwd="/workspace/repo",
            version="auto",
            attempts=1,
            sandbox=None,
        )

    def _backend_name(self) -> str:
        return "inspect_swe.claude_code"

    def _backend_label(self) -> str:
        return self.model_name


class ClaudeCodeReviewAgent(_ClaudeCodeAgentBase):
    """Agent wrapper that runs Claude Code review via Inspect AI sandbox bridge."""

    def __init__(
        self,
        *,
        model: str = "anthropic/claude-sonnet-4.5",
        codex_cmd: str = "claude",
        codex_model: str | None = None,
        codex_prompt: str = "/review",
        timeout_s: int = 1200,
        skip_git_check: bool = True,
        max_files: int | None = 2000,
        max_total_bytes: int | None = 10 * 1024 * 1024,
        log_dir: str = "./logs",
    ):
        super().__init__(
            model=model,
            codex_cmd=codex_cmd,
            codex_model=codex_model,
            timeout_s=timeout_s,
            skip_git_check=skip_git_check,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
            log_dir=log_dir,
        )
        self.codex_prompt = codex_prompt

    def _instructions(self) -> str:
        return _CODEX_REVIEW_INSTRUCTIONS

    def _create_prompt(self, test_case: TestCase) -> str:
        return (
            f"{self._instructions()}\n\n"
            f"{self.codex_prompt}\n\n"
            "Repository: /workspace/repo\nReview the repository."
        )

    def _target(self) -> str:
        return "REVIEW"

    def _parse_result(self, *, sample, test_case: TestCase) -> TestResult:
        result = super()._parse_result(sample=sample, test_case=test_case)
        result.metadata["codex_prompt"] = self.codex_prompt
        return result


class ClaudeCodeVibeTestAgent(_ClaudeCodeAgentBase):
    """Claude Code-backed property testing agent with the same prompt surface as VibeTestAgent."""

    def __init__(
        self,
        *,
        model: str = "anthropic/claude-sonnet-4.5",
        codex_cmd: str = "claude",
        codex_model: str | None = None,
        timeout_s: int = 1200,
        skip_git_check: bool = True,
        max_files: int | None = 2000,
        max_total_bytes: int | None = 10 * 1024 * 1024,
        log_dir: str = "./logs",
        static: bool = False,
        safety_agent: bool = False,
        safety_analysis_tools: bool = False,
        safety_repo_artifacts: bool = True,
    ):
        super().__init__(
            model=model,
            codex_cmd=codex_cmd,
            codex_model=codex_model,
            timeout_s=timeout_s,
            skip_git_check=skip_git_check,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
            log_dir=log_dir,
        )
        self.static = static
        self.safety_agent = safety_agent
        self.safety_analysis_tools = safety_analysis_tools
        self.safety_repo_artifacts = safety_repo_artifacts

    def _instructions(self) -> str:
        return build_agent_instructions(
            static=self.static,
            safety_agent=self.safety_agent,
            safety_analysis_tools=self.safety_analysis_tools,
            safety_repo_artifacts=self.safety_repo_artifacts,
        )

    def _create_prompt(self, test_case: TestCase) -> str:
        return (
            f"{self._instructions()}\n\n"
            f"{build_test_case_prompt(
                test_case,
                safety_agent=self.safety_agent,
                safety_repo_artifacts=self.safety_repo_artifacts,
            )}"
        )

    def _parse_result(self, *, sample, test_case: TestCase) -> TestResult:
        output = _extract_output(sample)
        verdict, case_score, reason_text, evidence_text = _parse_submission_output(output)
        if not verdict:
            verdict = "INCONCLUSIVE"

        return TestResult(
            test_case=test_case,
            passed=verdict == "PASS",
            message=output or "No Claude Code testing output captured.",
            execution_log=_build_execution_log(sample),
            metadata={
                "model": self.model_name,
                "codex_cmd": self.codex_cmd,
                "codex_model": self._backend_label(),
                "codex_backend": self._backend_name(),
                "verdict": verdict,
                "case_score": case_score,
                "reason_text": reason_text,
                "evidence_text": evidence_text,
                "total_time": getattr(sample, "total_time", None),
                "working_time": getattr(sample, "working_time", None),
                **usage_payload_from_sample(sample),
            },
        )
