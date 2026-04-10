from pathlib import Path

from vibetest.cli import _detect_search_aids
from vibetest.agent.react_agent import (
    _parse_submission_output,
    build_agent_instructions,
)
from vibetest.testcases.base import TestCase as AuditCase


def test_parse_submission_output_preserves_case_score() -> None:
    verdict, case_score, reason, evidence = _parse_submission_output(
        """VERDICT: FAIL
CASE_SCORE: 0.85
REASON: clear violation
EVIDENCE: concrete trace evidence
"""
    )

    assert verdict == "FAIL"
    assert case_score == 0.85
    assert reason == "clear violation"
    assert evidence == "concrete trace evidence"


def test_safety_prompt_includes_repository_layout_and_search_aids() -> None:
    prompt = build_agent_instructions(
        static=False,
        safety_agent=True,
        safety_analysis_tools=True,
        safety_repo_artifacts=True,
    )

    assert "## Repository Layout" in prompt
    assert "## Workflow and Guidance" in prompt
    assert "search_aids/clusters.json" in prompt
    assert "TRACE_SCORES_START" in prompt


def test_test_case_name_defaults() -> None:
    case = AuditCase(
        description="Audit this trace repo.",
        repo_path=Path("."),
    )

    assert case.name.startswith("audit-")


def test_detect_search_aids(tmp_path: Path) -> None:
    assert not _detect_search_aids(tmp_path)

    search_aids = tmp_path / "search_aids"
    search_aids.mkdir()
    (search_aids / "clusters.json").write_text("{}", encoding="utf-8")

    assert _detect_search_aids(tmp_path)
