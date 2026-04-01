from pathlib import Path
from types import SimpleNamespace

from vibetest.agent.codex_agent import CodexVibeTestAgent
from vibetest.agent.react_agent import VibeTestAgent
from vibetest.testcases.base import TestCase


def _test_case() -> TestCase:
    return TestCase(
        name="sample-case",
        description="sample description",
        repo_path=Path("."),
    )


def test_react_resume_parser_keeps_case_score() -> None:
    output = "VERDICT: FAIL\nCASE_SCORE: 0.75\nREASON: found issue\nEVIDENCE: trace"
    sample = SimpleNamespace(
        id="case-1",
        output=SimpleNamespace(completion=output),
        messages=[],
        score=None,
        total_time=None,
        working_time=None,
        model_usage=None,
    )
    eval_log = SimpleNamespace(samples=[sample])
    agent = SimpleNamespace(model_name="openai/gpt-5-mini")

    results = VibeTestAgent._parse_results_from_log(
        agent,
        eval_log,
        {"case-1": _test_case()},
        {"case-1"},
    )

    assert len(results) == 1
    assert results[0].metadata["verdict"] == "FAIL"
    assert results[0].metadata["case_score"] == 0.75
    assert results[0].metadata["reason_text"] == "found issue"
    assert results[0].metadata["evidence_text"] == "trace"


def test_codex_parser_keeps_case_score() -> None:
    output = "VERDICT: PASS\nCASE_SCORE: 0.25\nREASON: looks good\nEVIDENCE: none"
    sample = SimpleNamespace(
        output=SimpleNamespace(completion=output),
        messages=[],
        total_time=None,
        working_time=None,
        model_usage=None,
    )
    agent = SimpleNamespace(
        model_name="openai/gpt-5-mini",
        codex_cmd="codex",
    )

    result = CodexVibeTestAgent._parse_result(agent, sample=sample, test_case=_test_case())

    assert result.metadata["verdict"] == "PASS"
    assert result.metadata["case_score"] == 0.25
    assert result.metadata["reason_text"] == "looks good"
    assert result.metadata["evidence_text"] == "none"
