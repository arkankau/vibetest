"""Clean-context Inspect/ReAct verifier for VibeTest FAIL evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import CompactionAuto
from inspect_ai.scorer import includes
from inspect_ai.tool import bash, python, text_editor, update_plan

from vibetest.agent.react_agent import (
    cleanup_archive_temps,
    cleanup_docker_sandbox,
    create_files_archive,
    setup_docker_sandbox,
)
from vibetest.testcases.base import TestCase, TestResult
from vibetest.usage import usage_payload_from_sample


_SCORE_RE = re.compile(r"SCORE:\s*(1(?:\.0+)?|0(?:\.\d+)?|\.\d+)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.*?)(?:\nEVIDENCE_ASSESSMENT:|\Z)", re.IGNORECASE | re.DOTALL)
_ASSESSMENT_RE = re.compile(r"EVIDENCE_ASSESSMENT:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)


_VERIFIER_INSTRUCTIONS = """You are an independent evidence verifier.

You will receive:
- a repository,
- a natural-language property,
- the original VibeTest FAIL verdict,
- the original reason and evidence claimed to support that FAIL verdict.

Your job is only to judge whether the provided evidence clearly supports the
claimed property failure. Do not decide whether the repository actually passes
or fails from scratch. If the original evidence is weak, vague, miscited,
unsupported by cited files, or does not logically support the stated property
failure, give a low score even if an independent investigation could find a
different valid failure.

Important: absence of evidence is not evidence of failure. Do not give a high
score merely because the repository lacks an optional best practice, framework
feature, helper, explicit policy, or diagnostic check. A FAIL verdict needs
positive evidence that the property is violated, or clear evidence that the
property explicitly requires something and the repository's stated behavior
omits it. If the evidence only shows "no DataLoader", "no shuffle=True",
"no .to(device)", "no CUDA/autocast", "no explicit dtype policy", "no baseline
comparison", "no loss logging", or similar absence-only findings, score it low
unless the original reason also demonstrates why that absence creates an actual
property failure in this repository.

Be especially careful with best-practice-shaped ML properties:
- Full-batch training has no minibatch order to shuffle; absence of a DataLoader
  or shuffle setting alone does not establish that training data is improperly
  ordered.
- CPU-only/default-device tensor code can be internally consistent; absence of
  explicit device or mixed-precision policy alone does not establish a device or
  dtype failure.
- A missing optional sanity check, baseline, overfit test, or logging trace is
  weaker than evidence that the implemented training/evaluation behavior fails
  the property.
- A statement that a more sophisticated model does not beat another model is not
  automatically evidence that the final training procedure fails a simple
  baseline property unless the claimed baseline and evaluation comparison are
  clearly established.

Use the repository only to check the cited evidence and the local context needed
to understand it. Prefer direct file inspection over speculation.

Score rubric:
- 1.0: the evidence is concrete, cited or otherwise directly checkable, and
  clearly establishes the claimed property failure.
- 0.7: the evidence mostly supports the failure, but has minor gaps or weak
  citations.
- 0.4: the evidence is suggestive but incomplete, indirect, or only partially
  supports the failure.
- 0.1: the evidence is mostly unsupported, vague, miscited, absence-only,
  best-practice-only, or does not address the property.
- 0.0: the evidence is absent or contradicts the claimed failure.

Required final output. End your answer with exactly these fields:

SCORE: [single number from 0.0 to 1.0]
REASON: [brief explanation of why the evidence does or does not support the FAIL verdict]
EVIDENCE_ASSESSMENT: [specific notes about the cited evidence, including any checked files/lines when relevant]
"""


def _extract_text_from_message(msg: Any) -> str:
    text = getattr(msg, "text", "") or ""
    if text:
        return str(text)

    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            block_text = str(block.get("text") or "").strip()
            if block_text:
                parts.append(block_text)
        return "\n".join(parts)
    return str(content or "")


def _extract_output(sample: Any) -> str:
    output = ""
    if sample.output and sample.output.completion:
        output = str(sample.output.completion)
        if "SCORE:" in output.upper():
            return output

    if sample.messages:
        for msg in reversed(sample.messages):
            text = _extract_text_from_message(msg)
            if text and "SCORE:" in text.upper():
                return text

    return output


def parse_verifier_output(output: str) -> tuple[float | None, str, str]:
    """Parse verifier score, reason, and assessment from final output."""
    text = output or ""
    score: float | None = None
    match = _SCORE_RE.search(text)
    if match:
        try:
            score = max(0.0, min(1.0, float(match.group(1))))
        except Exception:
            score = None

    reason_match = _REASON_RE.search(text)
    reason = reason_match.group(1).strip() if reason_match else ""
    assessment_match = _ASSESSMENT_RE.search(text)
    assessment = assessment_match.group(1).strip() if assessment_match else ""
    return score, reason, assessment


def _build_execution_log(sample: Any) -> str:
    if not sample.messages:
        return ""

    log_parts: list[str] = []
    for msg in sample.messages:
        role = getattr(msg, "role", "unknown")
        text = _extract_text_from_message(msg)
        if text:
            log_parts.append(f"[{role}] {text[:200]}...")
    return "\n".join(log_parts)


class EvidenceVerifierAgent:
    """Inspect/ReAct agent that scores support for an existing FAIL verdict."""

    def __init__(
        self,
        model: str | None = None,
        *,
        static: bool = False,
        log_dir: str = "./logs",
    ):
        self.model_name = model or "openai/gpt-5-mini"
        self.static = static
        self.log_dir = log_dir

    def _create_solver(self):
        tools = [bash(timeout=120), text_editor(), update_plan()]
        if not self.static:
            tools.insert(1, python(timeout=120))
        return react(
            prompt=_VERIFIER_INSTRUCTIONS,
            tools=tools,
            submit=True,
            compaction=CompactionAuto(),
        )

    def _create_prompt(self, test_case: TestCase) -> str:
        metadata = test_case.metadata or {}
        original_reason = str(metadata.get("original_reason") or "").strip()
        original_evidence = str(metadata.get("original_evidence") or "").strip()
        original_output = str(metadata.get("original_output") or "").strip()
        artifact_note = (
            "Saved evidence artifacts from the original VibeTest run are available under /evidence. "
            "If the original evidence cites /evidence/artifacts/..., inspect those files there."
            if metadata.get("evidence_artifacts_available")
            else "No saved /evidence artifact bundle is available for this verifier run."
        )

        return f"""Repository: {test_case.sandbox_path}/repo

{artifact_note}

Property:
{test_case.description}

Original VibeTest verdict: FAIL

Original reason:
{original_reason or "(not provided separately)"}

Original evidence:
{original_evidence or "(not provided separately)"}

Original full output:
{original_output[:20000] if original_output else "(not provided)"}

Judge only whether the original reason/evidence supports the FAIL verdict for
the property. Inspect repository files as needed to verify citations and code
context. Do not search for unrelated new failures.
"""

    def execute_tests(
        self,
        test_cases: list[TestCase],
        *,
        sandbox: str | None = None,
    ) -> list[TestResult]:
        if not test_cases:
            return []

        id_to_test_case: dict[str, TestCase] = {}
        samples: list[Sample] = []
        sandbox_config = None
        temp_dir_to_cleanup: Path | None = None

        if sandbox == "docker":
            sandbox_config, temp_dir_to_cleanup = setup_docker_sandbox()
        elif sandbox is not None:
            sandbox_config = sandbox

        try:
            for test_case in test_cases:
                sample_id = test_case.name
                id_to_test_case[sample_id] = test_case
                files_dict, setup_script = create_files_archive(
                    test_case,
                    sandbox_prefix=test_case.sandbox_path,
                )
                samples.append(
                    Sample(
                        id=sample_id,
                        input=self._create_prompt(test_case),
                        target="SCORE",
                        files=files_dict,
                        setup=setup_script,
                    )
                )

            task = Task(
                dataset=samples,
                solver=self._create_solver(),
                scorer=includes(),
                sandbox=sandbox_config,
            )
            results = eval(
                tasks=task,
                model=self.model_name,
                reasoning_effort="medium",
                reasoning_summary="auto",
                log_dir=self.log_dir,
                retry_on_error=2,
                fail_on_error=False,
            )
            return self._parse_results(results, id_to_test_case)
        finally:
            if sandbox == "docker" and temp_dir_to_cleanup is not None:
                cleanup_docker_sandbox(temp_dir_to_cleanup)
            cleanup_archive_temps()

    def _parse_results(self, results: Any, id_to_test_case: dict[str, TestCase]) -> list[TestResult]:
        sample_id_to_result: dict[str, TestResult] = {}
        if results and len(results) > 0:
            eval_result = results[0]
            if eval_result.samples:
                for sample in eval_result.samples:
                    sample_id = sample.id
                    if sample_id not in id_to_test_case:
                        continue
                    output = _extract_output(sample)
                    score, reason, assessment = parse_verifier_output(output)
                    sample_id_to_result[sample_id] = TestResult(
                        test_case=id_to_test_case[sample_id],
                        passed=score is not None,
                        message=output,
                        execution_log=_build_execution_log(sample),
                        metadata={
                            "verifier_model": self.model_name,
                            "evidence_support_score": score,
                            "reason_text": reason,
                            "evidence_assessment": assessment,
                            "score": sample.score.value if sample.score else None,
                            "total_time": getattr(sample, "total_time", None),
                            "working_time": getattr(sample, "working_time", None),
                            **usage_payload_from_sample(sample),
                        },
                    )

        out: list[TestResult] = []
        for sample_id, test_case in id_to_test_case.items():
            if sample_id in sample_id_to_result:
                out.append(sample_id_to_result[sample_id])
            else:
                out.append(
                    TestResult(
                        test_case=test_case,
                        passed=False,
                        message="Evidence verifier failed to execute or parse SCORE.",
                        metadata={
                            "verifier_model": self.model_name,
                            "evidence_support_score": None,
                            "error": "Execution or parsing failure",
                        },
                    )
                )
        return out
