import json
from pathlib import Path

from scripts import analyze_distributed_misuse as dm


def test_meerkat_vs_monitor_ap_points_falls_back_to_initial_scores(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    repo_dir = tmp_path / "repo_case_1"
    repo_dir.mkdir()
    (repo_dir / "initial_scores.tsv").write_text("p1\t0.8\nn1\t0.1\n", encoding="utf-8")

    row = {
        "case_id": "case_0001",
        "repo": str(repo_dir),
        "ground_truth": {
            "has_distributed_misuse": True,
            "harmful_trace_files": ["p1"],
            "trace_files": ["p1", "n1"],
        },
        "scoring": {
            "trace_scores": {
                "p1": 0.8,
                "n1": 0.9,
            }
        },
    }
    agent_path = results_dir / "dm_bio_d6_bg20_v6.jsonl"
    agent_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    points = dm._meerkat_vs_monitor_ap_points(results_dir, [agent_path])

    assert len(points) == 1
    point = points[0]
    assert point["domain"] == "bio"
    assert point["model_label"] == "gpt-5.4-mini"
    assert point["monitor_ap"] == 1.0
    assert point["meerkat_ap"] == 0.5


def test_paper_curve_candidates_prefer_direct_bayesian_and_buffer_files(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    direct_bayesian = results_dir / "dm_cyber_d6_bg20_qwen35_bayesian.jsonl"
    legacy_bayesian = results_dir / "dm_cyber_d6_bg20_qwen35_bayesian_no_cluster.jsonl"
    direct_buffer = results_dir / "dm_cyber_d6_bg100_qwen35_buffer.jsonl"
    legacy_buffer = results_dir / "dm_cyber_d6_bg100_buffer_qwen35.jsonl"
    for path in (direct_bayesian, legacy_bayesian, direct_buffer, legacy_buffer):
        path.write_text("", encoding="utf-8")

    candidates = dm._paper_curve_candidates(results_dir, [], domain="cyber")
    selected: dict[tuple[str, str, int], Path] = {
        (model_label, method_label, bg): path for path, model_label, method_label, bg in candidates
    }

    assert selected[("Qwen-3.5", "Bayesian", 20)] == direct_bayesian
    assert selected[("Qwen-3.5", "Buffer", 100)] == direct_buffer


def test_trace_ap_ignores_stale_saved_pr_curve(tmp_path) -> None:
    path = tmp_path / "dm_cyber_d6_bg100_qwen35_naive.jsonl"
    rows = [
        {
            "case_id": "case_0000",
            "ground_truth": {
                "has_distributed_misuse": True,
                "harmful_trace_files": ["p1"],
            },
            "scoring": {
                "trace_scores": {
                    "p1": 0.9,
                    "n1": 0.2,
                },
                # Deliberately stale/wrong curve: AP would be 0.25 if trusted.
                "pr_curve": [
                    {"recall": 0.0, "precision": 1.0},
                    {"recall": 1.0, "precision": 0.25},
                ],
            },
        },
        {
            "case_id": "case_0001",
            "ground_truth": {
                "has_distributed_misuse": True,
                "harmful_trace_files": ["p2"],
            },
            "scoring": {
                "trace_scores": {
                    "p2": 0.8,
                    "n2": 0.1,
                },
                "pr_curve": [
                    {"recall": 0.0, "precision": 1.0},
                    {"recall": 1.0, "precision": 0.1},
                ],
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    trace_ap, _ = dm._method_metric_summary(
        path,
        method_label="Naive Agent",
        monitor_path=None,
        metric_key="trace_ap",
        bootstrap_samples=10,
        seed=42,
    )
    runs = dm._paper_curve_runs(tmp_path, [path], domain="cyber", bootstrap_samples=10)

    assert trace_ap == 1.0
    assert len(runs) == 1
    assert runs[0].average_precision == 1.0
