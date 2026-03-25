#!/usr/bin/env python3
"""
Unified Bayesian Detection Evaluation across all settings.

Runs detection methods on four settings (MLE-Sabotage, ImpossibleBench,
DM Cyber bg100, DM Bio bg100), computes extended metrics (AUROC, AP,
threshold sweep, benign-only FPR), and saves unified results JSON.

Usage:
    python experiments/run_unified_bayesian.py [--settings ...] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# --- Direct module import (avoid vibetest.__init__ dependency chain) ---
_project_root = Path(__file__).parent.parent
_mod_path = str(_project_root / "vibetest" / "baselines" / "adaptive_bayesian.py")
_spec = importlib.util.spec_from_file_location("adaptive_bayesian", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["adaptive_bayesian"] = _mod
_spec.loader.exec_module(_mod)

AdaptiveBayesianDetector = _mod.AdaptiveBayesianDetector
BayesianCalibrator = _mod.BayesianCalibrator
CaseData = _mod.CaseData
compute_extended_metrics = _mod.compute_extended_metrics
load_dataset = _mod.load_dataset
run_loo_evaluation = _mod.run_loo_evaluation
run_calibrated_loo = _mod.run_calibrated_loo


# ---------------------------------------------------------------------------
# Setting definitions
# ---------------------------------------------------------------------------

TRACE_REPOS_BASE = Path("results/safety/trace_repos")

SETTINGS = {
    "mle_sabotage": {
        "name": "MLE-Sabotage",
        "repo": "safety_mle_sabotage",
        "sets_subdir": "mle-sabotage-sets",
        "traces_per_case": 4,
        "loader": "standard",
        "methods": {
            "initial_scores": {"desc": "Raw monitor scores", "mode": "raw"},
            "calibrated_default": {
                "desc": "Bayesian calibrator (LOO)",
                "mode": "calibrated_loo",
                "kwargs": {"bucket_size": 0.1, "smoothing": 1.0, "trace_prior_positive": 0.3},
            },
            "bayesian_wide_r": {
                "desc": "Hypothesis-grid Bayesian (LOO)",
                "mode": "loo",
                "kwargs": {
                    "r_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                },
            },
            "bayesian_unsupervised": {
                "desc": "Unsupervised Bayesian",
                "mode": "unsupervised",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                },
            },
        },
    },
    "impossiblebench": {
        "name": "ImpossibleBench",
        "repo": "safety_patronus_trace",
        "sets_subdir": "impossiblebench-sets",
        "traces_per_case": 4,
        "loader": "standard",
        "methods": {
            "initial_scores": {"desc": "Raw monitor scores", "mode": "raw"},
            "calibrated_default": {
                "desc": "Bayesian calibrator (LOO)",
                "mode": "calibrated_loo",
                "kwargs": {"bucket_size": 0.1, "smoothing": 1.0, "trace_prior_positive": 0.3},
            },
            "bayesian_wide_r": {
                "desc": "Hypothesis-grid Bayesian (LOO)",
                "mode": "loo",
                "kwargs": {
                    "r_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                },
            },
            "bayesian_unsupervised": {
                "desc": "Unsupervised Bayesian",
                "mode": "unsupervised",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                },
            },
        },
    },
    "dm_cyber_bg100": {
        "name": "DM Cyber bg100",
        "repo": "safety_dm_cyber_d6_bg100_qwen35",
        "sets_subdir": "distributed-misuse-sets",
        "traces_per_case": 600,
        "loader": "dm",
        "agent_results": "results/dm_cyber_d6_bg100_qwen35_n50.jsonl",
        "methods": {
            "initial_scores": {"desc": "Raw monitor scores", "mode": "raw"},
            "bayesian_default": {
                "desc": "Bayesian default (LOO)",
                "mode": "loo",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                    "use_cluster_features": False,
                },
            },
            "bayesian_cluster": {
                "desc": "Bayesian + clusters (LOO)",
                "mode": "loo",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                    "use_cluster_features": True,
                },
            },
            "bayesian_unsupervised": {
                "desc": "Unsupervised Bayesian",
                "mode": "unsupervised",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                    "use_cluster_features": False,
                },
            },
        },
    },
    "dm_bio_bg100": {
        "name": "DM Bio bg100",
        "repo": "safety_dm_bio_d6_bg100_qwen35",
        "sets_subdir": "distributed-misuse-sets",
        "traces_per_case": 600,
        "loader": "dm",
        "agent_results": "results/dm_bio_d6_bg100_qwen35_v4_n50.jsonl",
        "methods": {
            "initial_scores": {"desc": "Raw monitor scores", "mode": "raw"},
            "bayesian_default": {
                "desc": "Bayesian default (LOO)",
                "mode": "loo",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                    "use_cluster_features": False,
                },
            },
            "bayesian_cluster": {
                "desc": "Bayesian + clusters (LOO)",
                "mode": "loo",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                    "use_cluster_features": True,
                },
            },
            "bayesian_unsupervised": {
                "desc": "Unsupervised Bayesian",
                "mode": "unsupervised",
                "kwargs": {
                    "bucket_size": 0.1, "smoothing": 1.0,
                    "n_bootstraps": 5, "prior_benign": 0.5,
                    "use_cluster_features": False,
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_standard_cases(
    cases_root: Path,
    max_cases: int | None = None,
) -> list[CaseData]:
    """Load cases from MLE-sabotage or ImpossibleBench format."""
    meta_dir = cases_root / "_metadata"
    cases = []
    for mf in sorted(meta_dir.glob("case_*.json")):
        cid = mf.stem
        cd = cases_root / cid
        if not cd.exists():
            continue
        with open(mf) as f:
            meta = json.load(f)
        tfs = meta.get("trace_files", [])
        gt = set(meta.get("gt_hacked_trace_files", []))
        sp = cd / "initial_scores.json"
        if sp.exists():
            with open(sp) as f:
                ms = json.load(f)
        else:
            ms = {t: 0.0 for t in tfs}
        cases.append(CaseData(case_id=cid, trace_ids=tfs, monitor_scores=ms, gt_malicious=gt))
        if max_cases and len(cases) >= max_cases:
            break
    return cases


def load_agent_results(results_path: str) -> dict[str, dict[str, float]]:
    """Load per-trace scores from agent results JSONL."""
    result = {}
    path = Path(results_path)
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = row.get("case_id", row.get("repo_name", ""))
            scoring = row.get("scoring", {})
            trace_scores = scoring.get("trace_scores", {})
            if trace_scores:
                result[case_id] = trace_scores
    return result


def load_cases_for_setting(
    setting_key: str,
    max_cases: int | None = None,
) -> list[CaseData]:
    """Load cases for a given setting."""
    cfg = SETTINGS[setting_key]
    repo_path = TRACE_REPOS_BASE / cfg["repo"]
    cases_root = repo_path / cfg["sets_subdir"]

    if cfg["loader"] == "dm":
        return load_dataset(repo_path, max_cases=max_cases)
    else:
        return load_standard_cases(cases_root, max_cases=max_cases)


# ---------------------------------------------------------------------------
# Method execution
# ---------------------------------------------------------------------------

def run_method(
    method_key: str,
    method_cfg: dict,
    cases: list[CaseData],
    setting_cfg: dict,
) -> dict[str, dict[str, float]]:
    """Run a single detection method and return scored dict."""
    mode = method_cfg["mode"]
    kwargs = method_cfg.get("kwargs", {})

    if mode == "raw":
        return {c.case_id: dict(c.monitor_scores) for c in cases}

    elif mode == "agent":
        agent_path = setting_cfg.get("agent_results", "")
        return load_agent_results(agent_path)

    elif mode == "calibrated_loo":
        return run_calibrated_loo(cases, calibrator_kwargs=kwargs)

    elif mode == "loo":
        return run_loo_evaluation(cases, detector_kwargs=kwargs)

    elif mode == "unsupervised":
        det = AdaptiveBayesianDetector(**kwargs)
        det.fit_unsupervised(cases)
        return {c.case_id: det.score_case(c) for c in cases}

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_setting(
    setting_key: str,
    max_cases: int | None = None,
) -> dict:
    """Run all methods for a single setting, return results dict."""
    cfg = SETTINGS[setting_key]
    print(f"\n{'='*70}")
    print(f"Setting: {cfg['name']} ({setting_key})")
    print(f"{'='*70}")

    cases = load_cases_for_setting(setting_key, max_cases)
    print(f"  Loaded {len(cases)} cases")

    if not cases:
        print("  No cases found, skipping.")
        return {"setting": setting_key, "error": "no cases"}

    total_traces = sum(len(c.trace_ids) for c in cases)
    total_mal = sum(len(c.gt_malicious or set()) for c in cases)
    n_positive = sum(1 for c in cases if len(c.gt_malicious or set()) > 0)
    n_benign = sum(1 for c in cases if len(c.gt_malicious or set()) == 0)
    print(f"  Total traces: {total_traces}, malicious: {total_mal}")
    print(f"  Positive cases: {n_positive}, benign-only cases: {n_benign}")

    # Score distribution
    benign_scores, mal_scores = [], []
    for c in cases:
        gt = c.gt_malicious or set()
        for tid, s in c.monitor_scores.items():
            (mal_scores if tid in gt else benign_scores).append(s)

    result = {
        "setting": setting_key,
        "name": cfg["name"],
        "n_cases": len(cases),
        "n_positive_cases": n_positive,
        "n_benign_cases": n_benign,
        "total_traces": total_traces,
        "total_malicious": total_mal,
        "traces_per_case": cfg["traces_per_case"],
        "score_dist": {
            "benign_mean": sum(benign_scores) / max(len(benign_scores), 1),
            "malicious_mean": sum(mal_scores) / max(len(mal_scores), 1),
            "benign_nonzero": sum(1 for s in benign_scores if s > 0),
            "malicious_nonzero": sum(1 for s in mal_scores if s > 0),
        },
        "methods": {},
    }

    # Run each method
    for method_key, method_cfg in cfg["methods"].items():
        print(f"\n  [{method_key}] {method_cfg['desc']}...")
        random.seed(42)
        np.random.seed(42)
        t0 = time.time()

        try:
            scored = run_method(method_key, method_cfg, cases, cfg)

            # Filter cases to those that were scored
            scored_cases = [c for c in cases if c.case_id in scored]
            if not scored_cases:
                print(f"    WARNING: no cases scored, skipping")
                result["methods"][method_key] = {
                    "desc": method_cfg["desc"],
                    "error": "no scored cases",
                }
                continue

            metrics = compute_extended_metrics(scored_cases, scored)
            elapsed = time.time() - t0

            result["methods"][method_key] = {
                "desc": method_cfg["desc"],
                "elapsed_s": round(elapsed, 1),
                **metrics,
            }

            # Print summary
            auroc = metrics.get("macro_auroc", 0)
            ap = metrics.get("macro_ap", 0)
            n_ben = metrics.get("n_benign_cases", 0)
            print(f"    AUROC: {auroc:.4f}  AP: {ap:.4f}  ({elapsed:.1f}s)")
            if n_ben > 0:
                # Show benign-only FPR at threshold 0.5
                bo = [b for b in metrics.get("benign_only", []) if b["threshold"] == 0.5]
                if bo:
                    print(f"    Benign-only FPR@0.5: {bo[0]['fpr']:.3f} "
                          f"({bo[0]['n_flagged']}/{bo[0]['n_benign_cases']} flagged)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ERROR ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            result["methods"][method_key] = {
                "desc": method_cfg["desc"],
                "error": str(e),
            }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified Bayesian Detection Evaluation"
    )
    parser.add_argument("--settings", nargs="*", default=None,
                        help="Settings to run (default: all)")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    print("=" * 70)
    print("Unified Bayesian Detection Evaluation")
    print("=" * 70)

    settings_to_run = args.settings or list(SETTINGS.keys())
    all_results = {}

    for setting_key in settings_to_run:
        if setting_key not in SETTINGS:
            print(f"Unknown setting: {setting_key}")
            continue
        result = run_setting(setting_key, max_cases=args.max_cases)
        all_results[setting_key] = result

    # Save unified results
    output_path = Path(args.output_dir) / "unified_bayesian_results.json"

    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=clean)
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Setting':<20} {'Method':<30} {'AUROC':>7} {'AP':>7} "
          f"{'Cases':>6} {'Ben':>5} {'FPR@0.5':>8}")
    print("-" * 85)

    for setting_key, result in all_results.items():
        for method_key, method_data in result.get("methods", {}).items():
            if "error" in method_data:
                continue
            auroc = method_data.get("macro_auroc", 0)
            ap = method_data.get("macro_ap", 0)
            n_cases = method_data.get("n_cases", 0)
            n_ben = method_data.get("n_benign_cases", 0)
            fpr = ""
            bo = [b for b in method_data.get("benign_only", []) if b["threshold"] == 0.5]
            if bo:
                fpr = f"{bo[0]['fpr']:.3f}"
            print(f"{setting_key:<20} {method_key:<30} {auroc:>7.4f} {ap:>7.4f} "
                  f"{n_cases:>6} {n_ben:>5} {fpr:>8}")

    return all_results


if __name__ == "__main__":
    main()
