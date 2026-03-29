#!/usr/bin/env python3
"""Per-trace and per-case analysis of adaptive attack results.

Computes AUROC, average precision, and score distributions for campaign
vs benign traces across all attack strategies.

Usage:
    python experiments/analyze_adaptive_attacks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC from binary labels and continuous scores."""
    if not labels or len(set(labels)) < 2:
        return float("nan")
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    tp_prev = fp_prev = 0
    auc = 0.0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    prev_score = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - fp_prev) * (tp + tp_prev) / 2
            tp_prev = tp
            fp_prev = fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - fp_prev) * (tp + tp_prev) / 2
    return auc / (n_pos * n_neg)


def _average_precision(labels: list[int], scores: list[float]) -> float:
    """Compute average precision from binary labels and continuous scores."""
    if not labels or sum(labels) == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    ap = 0.0
    for rank, (score, label) in enumerate(pairs, 1):
        if label == 1:
            tp += 1
            ap += tp / rank
    return ap / sum(labels)


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
        ("Baseline (2pass)", "dm_cyber_d6_bg100_qwen35_2pass_n10.jsonl"),
        ("score_format", "safety_dm_cyber_d6_bg100_qwen35_2pass_score_format_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("embedding_poison v1", "safety_dm_cyber_d6_bg100_qwen35_2pass_embedding_poison_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("embedding_poison v2", "safety_dm_cyber_d6_bg100_qwen35_2pass_embedding_poison_v2_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
        ("optimized_embedding", "safety_dm_cyber_d6_bg100_qwen35_2pass_optimized_embedding_AT-Qwen3.5-397B-A17B-FP8.jsonl"),
    ]

    # Auto-discover additional results
    for name, pattern in [
        ("optimized_embedding v2", "safety_dm_cyber_d6_bg100_qwen35_2pass_optimized_embedding_v2_AT-*.jsonl"),
        ("auditor_note", "safety_dm_cyber_d6_bg100_qwen35_2pass_auditor_note_AT-*.jsonl"),
        ("joint_optimized", "safety_dm_cyber_d6_bg100_qwen35_2pass_joint_optimized_AT-*.jsonl"),
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
    print("=" * 100)
    print(f"{'Strategy':<25s} {'Cases':>5s} {'Det':>4s} {'Rate':>6s} "
          f"{'AUROC':>7s} {'AP':>7s} {'Camp μ':>7s} {'Ben μ':>7s} {'Gap':>7s}")
    print("=" * 100)
    for r in all_results:
        gap = r["campaign_mean_score"] - r["benign_mean_score"]
        print(f"{r['strategy']:<25s} {r['n_gt_positive']:>5d} {r['n_detected']:>4d} "
              f"{r['detection_rate']:>5.0%} "
              f"{r['macro_auroc']:>7.3f} {r['macro_ap']:>7.3f} "
              f"{r['campaign_mean_score']:>7.3f} {r['benign_mean_score']:>7.3f} "
              f"{gap:>+6.3f}")
    print("=" * 100)

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
