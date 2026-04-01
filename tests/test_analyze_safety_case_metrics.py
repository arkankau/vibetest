import json

from scripts import analyze_safety_case_metrics as safety_metrics


def test_interp_precision_at_recalls_uses_upper_envelope() -> None:
    curve = [
        {"recall": 0.0, "precision": 1.0},
        {"recall": 0.5, "precision": 0.4},
        {"recall": 1.0, "precision": 0.8},
    ]

    precisions = safety_metrics._interp_precision_at_recalls(curve, [0.25, 0.75, 1.0])

    assert precisions == [0.8, 0.8, 0.8]


def test_judge_vs_at_ap_points_uses_model_specific_at_score_mode(tmp_path) -> None:
    judge_path = tmp_path / "safety_impossiblebench_claude-opus-4.6_llmjudge-gpt-5.4-mini.jsonl"
    at_path = tmp_path / "safety_impossiblebench_claude-opus-4.6_AT-gpt-5.4-mini.jsonl"
    buffer_path = tmp_path / "safety_impossiblebench_claude-opus-4.6_buffer-gpt-5.4-mini.jsonl"

    judge_row = {
        "case_id": "case-1",
        "ground_truth": {"hacked_trace_files": ["p1", "p2"]},
        "scoring": {"trace_scores": {"p1": 0.9, "p2": 0.8, "n1": 0.1}},
    }
    at_row = {
        "case_id": "case-1",
        "repo": str(tmp_path / "repo"),
        "ground_truth": {"hacked_trace_files": ["p1", "p2"]},
        "scoring": {"trace_scores": {"p1": 0.9, "p2": 0.1, "n1": 0.05}},
    }

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "initial_scores.tsv").write_text("p1\t0.9\np2\t0.8\nn1\t0.1\n", encoding="utf-8")
    judge_path.write_text(json.dumps(judge_row) + "\n", encoding="utf-8")
    at_path.write_text(json.dumps(at_row) + "\n", encoding="utf-8")
    buffer_path.write_text("", encoding="utf-8")

    rows = [
        {
            "file": str(judge_path),
            "score_mode": "original",
        },
        {
            "file": str(at_path),
            "score_mode": "max-merge",
        },
        {
            "file": str(buffer_path),
            "score_mode": "original",
        },
    ]

    points = safety_metrics._judge_vs_at_ap_points(
        rows,
        {"impossiblebench_claude-opus-4.6": "ImpossibleBench"},
    )

    dataset_points = points["impossiblebench_claude-opus-4.6"]
    assert len(dataset_points) == 1
    assert dataset_points[0]["judge_ap"] == 1.0
    assert dataset_points[0]["at_ap"] == 1.0
