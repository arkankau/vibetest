#!/usr/bin/env python3
"""Per-trace and per-case analysis of adaptive attack results.

Computes AUROC, average precision, and score distributions for campaign
vs benign traces across all attack strategies.

Usage:
    python experiments/analyze_adaptive_attacks.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

_VALID_TRACE_PATH_RE = re.compile(r"^traces/trace_\d{2,4}\.txt$")


def _auroc(labels: list[int], scores: list[float]) -> float:
    if not labels or len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _average_precision(labels: list[int], scores: list[float]) -> float:
    if not labels or sum(labels) == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def analyze_result_file(path: Path) -> dict:
    """Analyze a single result JSONL file."""
    cases = [json.loads(line) for line in path.read_text().strip().split("\n") if line.strip()]

    per_case = []
    all_labels = []
    all_scores = []

    for c in cases:
        gt = c.get("ground_truth", {})
        campaign_set = set(gt.get("harmful_trace_files", []))
        has_dm = gt.get("has_distributed_misuse", False)

        tests = c.get("tests", [{}])
        meta = tests[0].get("metadata", {}) if tests else {}
        trace_scores = meta.get("trace_scores", {})
        verdict = meta.get("verdict", "?")
        detected = c.get("failed_tests", 0) > 0

        # Per-trace labels and scores for this case
        case_labels = []
        case_scores = []
        campaign_scores_list = []
        benign_scores_list = []

        for trace_file, score in trace_scores.items():
            if not _VALID_TRACE_PATH_RE.match(trace_file):
                continue  # skip corrupted/truncated paths
            is_campaign = trace_file in campaign_set
            case_labels.append(1 if is_campaign else 0)
            case_scores.append(float(score))
            if is_campaign:
                campaign_scores_list.append(float(score))
            else:
                benign_scores_list.append(float(score))
            all_labels.append(1 if is_campaign else 0)
            all_scores.append(float(score))

        case_auroc = _auroc(case_labels, case_scores)
        case_ap = _average_precision(case_labels, case_scores)

        per_case.append({
            "case_id": c.get("case_id", "?"),
            "has_dm": has_dm,
            "detected": detected,
            "verdict": verdict,
            "n_traces": len(trace_scores),
            "n_campaign": len(campaign_set),
            "auroc": case_auroc,
            "ap": case_ap,
            "campaign_mean": np.mean(campaign_scores_list) if campaign_scores_list else float("nan"),
            "campaign_max": max(campaign_scores_list) if campaign_scores_list else float("nan"),
            "campaign_min": min(campaign_scores_list) if campaign_scores_list else float("nan"),
            "benign_mean": np.mean(benign_scores_list) if benign_scores_list else float("nan"),
            "benign_p95": float(np.percentile(benign_scores_list, 95)) if benign_scores_list else float("nan"),
            "benign_max": max(benign_scores_list) if benign_scores_list else float("nan"),
            "campaign_scores": dict(zip(
                sorted(campaign_set),
                [trace_scores.get(t, 0.0) for t in sorted(campaign_set)]
            )),
        })

    # Macro-averaged metrics
    gt_pos_cases = [c for c in per_case if c["has_dm"]]
    macro_auroc = np.nanmean([c["auroc"] for c in gt_pos_cases]) if gt_pos_cases else float("nan")
    macro_ap = np.nanmean([c["ap"] for c in gt_pos_cases]) if gt_pos_cases else float("nan")

    # Global metrics
    global_auroc = _auroc(all_labels, all_scores)
    global_ap = _average_precision(all_labels, all_scores)

    n_gt_pos = len(gt_pos_cases)
    n_detected = sum(1 for c in gt_pos_cases if c["detected"])

    return {
        "file": str(path.name),
        "n_cases": len(cases),
        "n_gt_positive": n_gt_pos,
        "n_detected": n_detected,
        "detection_rate": n_detected / max(1, n_gt_pos),
        "macro_auroc": float(macro_auroc),
        "macro_ap": float(macro_ap),
        "global_auroc": global_auroc,
        "global_ap": global_ap,
        "campaign_mean_score": float(np.nanmean([c["campaign_mean"] for c in gt_pos_cases])) if gt_pos_cases else float("nan"),
        "benign_mean_score": float(np.nanmean([c["benign_mean"] for c in gt_pos_cases])) if gt_pos_cases else float("nan"),
        "per_case": per_case,
    }


def main():
    results_dir = Path("/Users/davisbrown/vibetest/results")

    experiments = [
        # Main bayesian_boost results (large-scale)
        ("Baseline (bb, n=50)", "safety_dm_cyber_d6_bg100_qwen35_bayesian_boost_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("auditor_note (bb, n=50)", "safety_dm_cyber_d6_bg100_qwen35_bb_auditor_note_n50_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("response_truncate (bb, n=20)", "safety_dm_cyber_d6_bg100_qwen35_bb_response_truncate_n20_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("persona_reframe (bb, n=20)", "safety_dm_cyber_d6_bg100_qwen35_bb_persona_reframe_n20_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        # 2pass results (smaller-scale)
        ("Baseline (2pass, n=10)", "dm_cyber_d6_bg100_qwen35_2pass_n10.jsonl"),
        ("score_format (2pass)", "safety_dm_cyber_d6_bg100_qwen35_2pass_score_format_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("embedding_poison v1 (2pass)", "safety_dm_cyber_d6_bg100_qwen35_2pass_embedding_poison_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("embedding_poison v2 (2pass)", "safety_dm_cyber_d6_bg100_qwen35_2pass_embedding_poison_v2_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("optimized_embedding (2pass)", "safety_dm_cyber_d6_bg100_qwen35_2pass_optimized_embedding_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
    ]

    # Auto-discover additional 2pass results
    for name, pattern in [
        ("optimized_embedding v2 (2pass)", "safety_dm_cyber_d6_bg100_qwen35_2pass_optimized_embedding_v2_AT-*.jsonl"),
        ("auditor_note (2pass)", "safety_dm_cyber_d6_bg100_qwen35_2pass_auditor_note_AT-*.jsonl"),
        ("joint_optimized (2pass)", "safety_dm_cyber_d6_bg100_qwen35_2pass_joint_optimized_AT-*.jsonl"),
    ]:
        matches = sorted(results_dir.glob(pattern))
        if matches:
            experiments.append((name, matches[-1].name))

    all_results = []
    for name, filename in experiments:
        path = results_dir / filename
        if not path.exists():
            continue
        result = analyze_result_file(path)
        result["strategy"] = name
        all_results.append(result)

    # Summary table
    print("=" * 115)
    print(f"{'Strategy':<35s} {'Cases':>5s} {'Det':>4s} {'Rate':>6s} "
          f"{'μ-AUROC':>8s} {'μ-AP':>7s} {'M-AP':>6s} {'Camp μ':>7s} {'Ben μ':>7s} {'Gap':>7s}")
    print("=" * 115)
    for r in all_results:
        gap = r["campaign_mean_score"] - r["benign_mean_score"]
        print(f"{r['strategy']:<35s} {r['n_gt_positive']:>5d} {r['n_detected']:>4d} "
              f"{r['detection_rate']:>5.0%} "
              f"{r['global_auroc']:>8.3f} {r['global_ap']:>7.3f} "
              f"{r['macro_ap']:>6.3f} "
              f"{r['campaign_mean_score']:>7.3f} {r['benign_mean_score']:>7.3f} "
              f"{gap:>+6.3f}")
    print("=" * 115)

    # Per-case detail for each strategy
    for r in all_results:
        print(f"\n--- {r['strategy']} (per-case) ---")
        print(f"  {'Case':<12s} {'Det':>4s} {'AUROC':>7s} {'AP':>7s} "
              f"{'Camp μ':>7s} {'Camp ↑':>7s} {'Camp ↓':>7s} {'Ben μ':>7s}")
        for c in r["per_case"]:
            if not c["has_dm"]:
                continue
            det = "YES" if c["detected"] else "NO"
            print(f"  {c['case_id']:<12s} {det:>4s} "
                  f"{c['auroc']:>7.3f} {c['ap']:>7.3f} "
                  f"{c['campaign_mean']:>7.3f} {c['campaign_max']:>7.3f} "
                  f"{c['campaign_min']:>7.3f} {c['benign_mean']:>7.3f}")

    # Per-trace campaign scores across strategies
    print(f"\n{'='*100}")
    print("Per-trace campaign scores across strategies")
    print(f"{'='*100}")
    # Find common cases across all strategies
    case_ids = None
    for r in all_results:
        ids = {c["case_id"] for c in r["per_case"] if c["has_dm"]}
        if case_ids is None:
            case_ids = ids
        else:
            case_ids &= ids

    if case_ids:
        for case_id in sorted(case_ids):
            print(f"\n  {case_id}:")
            for r in all_results:
                for c in r["per_case"]:
                    if c["case_id"] == case_id and c["has_dm"]:
                        scores = c.get("campaign_scores", {})
                        score_str = " ".join(f"{s:.2f}" for s in scores.values())
                        print(f"    {r['strategy']:<25s}: [{score_str}]")


if __name__ == "__main__":
    main()
