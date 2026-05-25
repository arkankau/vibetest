"""Clean-context Inspect/ReAct verifier for VibeTest FAIL evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

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
from vibetest.verifier.rubric import parse_verifier_fields, structured_state_to_score


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
omits it. Absence-only or best-practice-only findings should receive a low score
unless the original reason also demonstrates why that absence creates an actual
property failure in this repository.

Use the repository only to check the cited evidence and the local context needed
to understand it. Prefer direct file inspection over speculation.

Scoring guidance:
- Output a single continuous score in [0.0, 1.0] reflecting how strongly the
  original reason and evidence support the claimed property failure.
- Higher scores: concrete, cited or directly checkable evidence that clearly
  establishes the failure described in the original verdict.
- Lower scores: weak, vague, miscited, absence-only, best-practice-only, or
  otherwise insufficient support; 0.0 when evidence is absent or contradicts
  the claimed failure.
- Use fine-grained values (e.g. 0.83, 0.62) when they fit the evidence quality;
  do not default to round numbers such as 0.0, 0.5, or 1.0 unless they are
  truly appropriate.

Required final output. End your answer with exactly these fields:

SCORE: [single number from 0.0 to 1.0]
REASON: [brief explanation of why the evidence does or does not support the FAIL verdict]
EVIDENCE_ASSESSMENT: [specific notes about the cited evidence, including any checked files/lines when relevant]
"""

_VERIFIER_INSTRUCTIONS_STRUCTURED = """You are an independent evidence verifier for VibeTest FAIL verdicts.

You will receive:
- a repository,
- a natural-language property,
- the original VibeTest FAIL verdict,
- the original reason and evidence claimed to support that FAIL verdict.

Your job is NOT to re-audit the whole repository from scratch. Judge only whether the
original FAIL is justified by the cited evidence under the strict rubric below.

Strict rubric (match human re-audit):
1. FAIL — The cited evidence identifies a concrete operation that is PRESENT in the
   repository and that operation VIOLATES the property. There must be positive,
   checkable evidence of violation (code, logs, outputs), not mere absence of a check.
2. INCONCLUSIVE — The property requires an experiment or operation that is ABSENT or
   not demonstrated in the cited evidence (e.g., no randomized-label run, no baseline
   comparison, no training-loss log). Missing evidence is NOT evidence of failure.
   Also use INCONCLUSIVE when evidence is too vague, miscited, or unverifiable.
3. PASS — The cited evidence actually shows the property is SATISFIED, or the original
   FAIL reasoning is incorrect / misinterprets the code.

Critical rules:
- Absence of an optional best practice is INCONCLUSIVE, not FAIL.
- Experiment-required properties (randomized labels, baseline comparison, training-loss
  behavior, tiny-batch overfit) require a present experiment with violating results
  for FAIL; if the experiment is missing, choose INCONCLUSIVE.
- Use the repository only to verify citations and local context. Do not hunt for new bugs.

Required final output. End your answer with exactly these fields:

EVIDENCE_STATE: [FAIL | INCONCLUSIVE | PASS]
SCORE: [single number from 0.0 to 1.0; use 0.95 for FAIL, 0.15 for INCONCLUSIVE, 0.05 for PASS unless fine-tuning within that band]
REASON: [brief explanation tied to the rubric category]
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


def _output_is_complete(text: str) -> bool:
    upper = (text or "").upper()
    return "SCORE:" in upper or "EVIDENCE_STATE:" in upper


def _extract_output(sample: Any) -> str:
    output = ""
    if sample.output and sample.output.completion:
        output = str(sample.output.completion)
        if _output_is_complete(output):
            return output

    if sample.messages:
        for msg in reversed(sample.messages):
            text = _extract_text_from_message(msg)
            if text and _output_is_complete(text):
                return text

    return output


def parse_verifier_output(output: str) -> tuple[float | None, str, str]:
    """Parse verifier score, reason, and assessment from final output."""
    score, reason, assessment, _state = parse_verifier_fields(output)
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
        rubric: Literal["continuous", "structured"] = "continuous",
    ):
        self.model_name = model or "openai/gpt-5-mini"
        self.static = static
        self.log_dir = log_dir
        self.rubric = rubric

    def _create_solver(self):
        tools = [bash(timeout=120), text_editor(), update_plan()]
        if not self.static:
            tools.insert(1, python(timeout=120))
        prompt = (
            _VERIFIER_INSTRUCTIONS_STRUCTURED
            if self.rubric == "structured"
            else _VERIFIER_INSTRUCTIONS
        )
        return react(
            prompt=prompt,
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
                    score, reason, assessment, evidence_state = parse_verifier_fields(output)
                    if evidence_state is not None and score is None:
                        score = structured_state_to_score(evidence_state)
                    sample_id_to_result[sample_id] = TestResult(
                        test_case=id_to_test_case[sample_id],
                        passed=score is not None,
                        message=output,
                        execution_log=_build_execution_log(sample),
                        metadata={
                            "verifier_model": self.model_name,
                            "verifier_rubric": self.rubric,
                            "evidence_support_score": score,
                            "evidence_state": (
                                evidence_state.value if evidence_state is not None else None
                            ),
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
                            "verifier_rubric": self.rubric,
                            "evidence_support_score": None,
                            "evidence_state": None,
                            "error": "Execution or parsing failure",
                        },
                    )
                )
        return out
