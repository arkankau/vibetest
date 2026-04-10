from pathlib import Path

import numpy as np

from meerkat.agent.meerkat_agent import _parse_submission_output, build_agent_instructions
from meerkat.search_aids import list_trace_files, prepare_search_aids
from meerkat.testcases.base import TestCase as AuditCase


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
    prompt = build_agent_instructions()

    assert "## Repository Layout" in prompt
    assert "## Workflow and Guidance" in prompt
    assert "search_aids/clusters.json" in prompt
    assert "TRACE_SCORES_START" in prompt
    assert "run_parallel_llm_scanner" not in prompt


def test_test_case_name_defaults() -> None:
    case = AuditCase(
        description="Audit this trace repo.",
        repo_path=Path("."),
    )

    assert case.name.startswith("audit-")


def test_list_trace_files_requires_traces_directory(tmp_path: Path) -> None:
    try:
        list_trace_files(tmp_path)
    except ValueError as exc:
        assert "traces/" in str(exc)
    else:
        raise AssertionError("Expected list_trace_files() to require traces/")


def test_prepare_search_aids_creates_directory_and_artifacts(tmp_path: Path, monkeypatch) -> None:
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    (traces_dir / "trace_000.txt").write_text("first trace", encoding="utf-8")
    (traces_dir / "trace_001.txt").write_text("second trace", encoding="utf-8")

    async def fake_score_traces_parallel(trace_contents, **kwargs):
        assert sorted(trace_contents) == ["traces/trace_000.txt", "traces/trace_001.txt"]
        return {
            "traces/trace_000.txt": 0.2,
            "traces/trace_001.txt": 0.8,
        }

    def fake_embed_traces(repo_path, trace_files, **kwargs):
        assert repo_path == tmp_path
        return (
            trace_files,
            ["first trace", "second trace"],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        )

    async def fake_label_clusters_parallel(repo_path, payload, **kwargs):
        assert repo_path == tmp_path
        for index, cluster in enumerate(payload["clusters"], start=1):
            cluster["label"] = f"cluster-{index}"
            cluster["summary"] = f"summary-{index}"
        return payload

    monkeypatch.setattr(
        "meerkat.search_aids._score_traces_parallel",
        fake_score_traces_parallel,
    )
    monkeypatch.setattr("meerkat.search_aids._embed_traces", fake_embed_traces)
    monkeypatch.setattr(
        "meerkat.search_aids._label_clusters_parallel",
        fake_label_clusters_parallel,
    )

    artifacts = prepare_search_aids(
        tmp_path,
        "The agent does not exploit verifier shortcuts.",
        scoring_model="openai/gpt-5-mini",
    )

    search_aids_dir = tmp_path / "search_aids"
    assert search_aids_dir.is_dir()
    assert artifacts["directory"] == str(search_aids_dir)

    initial_scores = (search_aids_dir / "initial_scores.tsv").read_text(encoding="utf-8")
    assert "traces/trace_000.txt\t0.200000" in initial_scores
    assert "traces/trace_001.txt\t0.800000" in initial_scores

    clusters_json = (search_aids_dir / "clusters.json").read_text(encoding="utf-8")
    assert '"label": "cluster-1"' in clusters_json

    clusters_txt = (search_aids_dir / "clusters.txt").read_text(encoding="utf-8")
    assert "Flat clustering summary" in clusters_txt
