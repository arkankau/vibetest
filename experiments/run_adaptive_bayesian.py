#!/usr/bin/env python3
"""
Run Adaptive Bayesian Sabotage Detection experiments.

Loads existing distributed-misuse case data (initial_scores.json + clusters.json)
and runs the Bayesian protocol from Wen et al. (2411.17693), adapted for detection.

Compares multiple configurations against the raw initial_scores baseline and
the full agent pipeline results.

Usage:
    python experiments/run_adaptive_bayesian.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import the module directly to avoid pulling in all vibetest dependencies
import importlib.util

_project_root = Path(__file__).parent.parent
_mod_path = str(_project_root / "vibetest" / "baselines" / "adaptive_bayesian.py")
_spec = importlib.util.spec_from_file_location("adaptive_bayesian", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["adaptive_bayesian"] = _mod
_spec.loader.exec_module(_mod)

AdaptiveBayesianDetector = _mod.AdaptiveBayesianDetector
CaseData = _mod.CaseData
compute_metrics = _mod.compute_metrics
load_dataset = _mod.load_dataset
run_loo_evaluation = _mod.run_loo_evaluation


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

TRACE_REPOS_BASE = Path("results/safety/trace_repos")

DATASETS = {
    "cyber_bg100": {
        "repo": "safety_dm_cyber_d6_bg100_qwen35",
        "domain": "cyber",
        "bg": 100,
        "agent_results": "results/dm_cyber_d6_bg100_qwen35_n50.jsonl",
    },
    "cyber_bg20": {
        "repo": "safety_dm_cyber_d6_bg20_qwen35",
        "domain": "cyber",
        "bg": 20,
        "agent_results": "results/dm_cyber_d6_bg20_qwen35_n20.jsonl",
    },
    "bio_bg100": {
        "repo": "safety_dm_bio_d6_bg100_qwen35",
        "domain": "bio",
        "bg": 100,
        "agent_results": "results/dm_bio_d6_bg100_qwen35_v4_n50.jsonl",
    },
    "bio_bg20": {
        "repo": "safety_dm_bio_d6_bg20_qwen35",
        "domain": "bio",
        "bg": 20,
        "agent_results": "results/dm_bio_d6_bg20_qwen35_v4_n20.jsonl",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agent_results(results_path: str) -> dict[str, dict[str, float]]:
    """Load per-trace scores from an existing agent results JSONL file.

    Returns {case_id: {trace_id: score}}.
    """
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


def compute_baseline_metrics(cases: list[CaseData]) -> dict:
    """Compute metrics using raw initial_scores as the baseline."""
    scored = {}
    for case in cases:
        scored[case.case_id] = dict(case.monitor_scores)
    return compute_metrics(cases, scored)


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "bayesian_default": {
        "desc": "Default Bayesian (LOO, supervised, bucket=0.1, boot=5)",
        "kwargs": {
            "bucket_size": 0.1,
            "smoothing": 1.0,
            "n_bootstraps": 5,
            "prior_benign": 0.5,
            "use_cluster_features": False,
        },
        "mode": "loo",
    },
    "bayesian_fine_bucket": {
        "desc": "Fine buckets (0.05) for more precise likelihood",
        "kwargs": {
            "bucket_size": 0.05,
            "smoothing": 0.5,
            "n_bootstraps": 5,
            "prior_benign": 0.5,
            "use_cluster_features": False,
        },
        "mode": "loo",
    },
    "bayesian_cluster": {
        "desc": "Bayesian + cluster features",
        "kwargs": {
            "bucket_size": 0.1,
            "smoothing": 1.0,
            "n_bootstraps": 5,
            "prior_benign": 0.5,
            "use_cluster_features": True,
        },
        "mode": "loo",
    },
    "bayesian_low_prior": {
        "desc": "Low benign prior (0.3) — more suspicious",
        "kwargs": {
            "bucket_size": 0.1,
            "smoothing": 1.0,
            "n_bootstraps": 5,
            "prior_benign": 0.3,
            "use_cluster_features": False,
        },
        "mode": "loo",
    },
    "bayesian_high_prior": {
        "desc": "High benign prior (0.7) — less suspicious",
        "kwargs": {
            "bucket_size": 0.1,
            "smoothing": 1.0,
            "n_bootstraps": 5,
            "prior_benign": 0.7,
            "use_cluster_features": False,
        },
        "mode": "loo",
    },
    "bayesian_unsupervised": {
        "desc": "Unsupervised (no labels)",
        "kwargs": {
            "bucket_size": 0.1,
            "smoothing": 1.0,
            "n_bootstraps": 5,
            "prior_benign": 0.5,
            "use_cluster_features": False,
        },
        "mode": "unsupervised",
    },
    "bayesian_cluster_fine": {
        "desc": "Cluster features + fine buckets",
        "kwargs": {
            "bucket_size": 0.05,
            "smoothing": 0.5,
            "n_bootstraps": 5,
            "prior_benign": 0.5,
            "use_cluster_features": True,
        },
        "mode": "loo",
    },
    "bayesian_wide_r": {
        "desc": "Wider r grid with more attack rates",
        "kwargs": {
            "r_values": [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.15],
            "bucket_size": 0.1,
            "smoothing": 1.0,
            "n_bootstraps": 5,
            "prior_benign": 0.5,
            "use_cluster_features": False,
        },
        "mode": "loo",
    },
}


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    dataset_key: str,
    config_keys: list[str] | None = None,
    max_cases: int | None = None,
) -> dict:
    """Run all Bayesian configurations on a dataset.

    Returns a dict with results for each configuration plus baselines.
    """
    ds_info = DATASETS[dataset_key]
    repo_path = TRACE_REPOS_BASE / ds_info["repo"]

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_key} ({ds_info['domain']}, bg={ds_info['bg']}x)")
    print(f"Repo: {repo_path}")
    print(f"{'='*70}")

    # Load cases
    cases = load_dataset(repo_path, max_cases=max_cases)
    print(f"Loaded {len(cases)} cases")

    if not cases:
        print("  No cases found, skipping.")
        return {}

    # Print data summary
    total_traces = sum(len(c.trace_ids) for c in cases)
    total_malicious = sum(len(c.gt_malicious or set()) for c in cases)
    print(f"  Total traces: {total_traces}")
    print(f"  Total malicious: {total_malicious} ({100*total_malicious/total_traces:.2f}%)")

    all_scores = []
    for c in cases:
        all_scores.extend(c.monitor_scores.values())
    nonzero = sum(1 for s in all_scores if s > 0)
    print(f"  Non-zero scores: {nonzero}/{len(all_scores)} ({100*nonzero/len(all_scores):.1f}%)")

    results = {"dataset": dataset_key, "n_cases": len(cases), "methods": {}}

    # Baseline: raw initial_scores
    print("\n  [Baseline] Raw initial scores...")
    baseline_metrics = compute_baseline_metrics(cases)
    results["methods"]["initial_scores"] = {
        "desc": "Raw per-trace LLM monitor scores",
        **baseline_metrics,
    }
    print(f"    AUROC: {baseline_metrics['macro_auroc']:.4f}  AP: {baseline_metrics['macro_ap']:.4f}")

    # Agent results (if available)
    agent_results_path = ds_info.get("agent_results", "")
    if agent_results_path and Path(agent_results_path).exists():
        print(f"  [Baseline] Agent pipeline ({agent_results_path})...")
        agent_scores = load_agent_results(agent_results_path)
        if agent_scores:
            agent_metrics = compute_metrics(cases, agent_scores)
            results["methods"]["agent_pipeline"] = {
                "desc": "Full agent pipeline (Stage 1+2+3)",
                **agent_metrics,
            }
            print(f"    AUROC: {agent_metrics['macro_auroc']:.4f}  AP: {agent_metrics['macro_ap']:.4f}")

    # Run Bayesian configurations
    if config_keys is None:
        config_keys = list(CONFIGS.keys())

    for config_key in config_keys:
        config = CONFIGS[config_key]
        print(f"\n  [{config_key}] {config['desc']}...")

        random.seed(42)
        np.random.seed(42)

        try:
            if config["mode"] == "loo":
                scored = run_loo_evaluation(cases, detector_kwargs=config["kwargs"])
            elif config["mode"] == "unsupervised":
                det = AdaptiveBayesianDetector(**config["kwargs"])
                det.fit_unsupervised(cases)
                scored = {}
                for case in cases:
                    scored[case.case_id] = det.score_case(case)
            else:
                # Default: fit on all, score all (for quick testing)
                det = AdaptiveBayesianDetector(**config["kwargs"])
                det.fit(cases)
                scored = {}
                for case in cases:
                    scored[case.case_id] = det.score_case(case)

            metrics = compute_metrics(cases, scored)
            results["methods"][config_key] = {
                "desc": config["desc"],
                **metrics,
            }
            print(f"    AUROC: {metrics['macro_auroc']:.4f}  AP: {metrics['macro_ap']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["methods"][config_key] = {"desc": config["desc"], "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results: dict, output_dir: str = "results") -> list[str]:
    """Generate comparison plots.

    Returns list of generated plot file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    plots = []

    # --- Plot 1: AUROC comparison bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, metric_key in enumerate(["macro_auroc", "macro_ap"]):
        ax = axes[ax_idx]
        metric_name = "AUROC" if metric_key == "macro_auroc" else "Average Precision"

        datasets_to_plot = sorted(all_results.keys())
        methods_to_plot = set()
        for ds_results in all_results.values():
            methods_to_plot.update(ds_results.get("methods", {}).keys())
        methods_to_plot = sorted(methods_to_plot)

        x = np.arange(len(datasets_to_plot))
        width = 0.8 / max(1, len(methods_to_plot))

        colors = plt.cm.tab20(np.linspace(0, 1, len(methods_to_plot)))

        for i, method in enumerate(methods_to_plot):
            vals = []
            for ds in datasets_to_plot:
                m = all_results[ds].get("methods", {}).get(method, {})
                vals.append(m.get(metric_key, 0.0))
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                         label=method.replace("_", " "), color=colors[i])

        ax.set_ylabel(metric_name)
        ax.set_title(f"Macro {metric_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([ds.replace("_", "\n") for ds in datasets_to_plot],
                           fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="random")
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Adaptive Bayesian Sabotage Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = str(output_dir / "adaptive_bayesian_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(plot_path)
    print(f"\nSaved: {plot_path}")

    # --- Plot 2: Per-case AUROC distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ds_idx, ds_key in enumerate(sorted(all_results.keys())):
        if ds_idx >= 4:
            break
        ax = axes[ds_idx]
        ds_results = all_results[ds_key]

        methods = ds_results.get("methods", {})
        method_names = []
        method_aurocs = []

        for method_name, method_data in sorted(methods.items()):
            if "per_case" not in method_data:
                continue
            aurocs = [v["auroc"] for v in method_data["per_case"].values()]
            if aurocs:
                method_names.append(method_name.replace("_", "\n"))
                method_aurocs.append(aurocs)

        if method_aurocs:
            bp = ax.boxplot(method_aurocs, labels=method_names, patch_artist=True)
            for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(method_aurocs)))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(ds_key.replace("_", " "), fontsize=11)
        ax.set_ylabel("Per-case AUROC")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Case AUROC Distribution by Method", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = str(output_dir / "adaptive_bayesian_per_case.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(plot_path)
    print(f"Saved: {plot_path}")

    # --- Plot 3: ROC curves for best configs ---
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ds_idx, ds_key in enumerate(sorted(all_results.keys())):
        ax = axes[ds_idx]
        ds_results = all_results[ds_key]
        methods = ds_results.get("methods", {})

        for method_name, method_data in sorted(methods.items()):
            if "per_case" not in method_data:
                continue
            auroc = method_data.get("macro_auroc", 0)
            label = f"{method_name} (AUROC={auroc:.3f})"
            # Plot micro ROC if we have pooled data
            # For now just show AUROC as a horizontal line
            ax.axhline(y=auroc, label=label, alpha=0.7)

        ax.set_title(f"{ds_key}", fontsize=11)
        ax.set_xlabel("Method")
        ax.set_ylabel("AUROC")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("Method AUROC Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = str(output_dir / "adaptive_bayesian_roc_summary.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(plot_path)
    print(f"Saved: {plot_path}")

    # --- Plot 4: Improvement over baseline heatmap ---
    datasets = sorted(all_results.keys())
    methods_set = set()
    for ds_results in all_results.values():
        methods_set.update(ds_results.get("methods", {}).keys())
    methods_list = sorted(m for m in methods_set if m != "initial_scores")

    if methods_list and datasets:
        fig, ax = plt.subplots(figsize=(max(10, len(methods_list) * 1.5), max(4, len(datasets) * 1.2)))

        heatmap_data = np.zeros((len(datasets), len(methods_list)))
        for i, ds in enumerate(datasets):
            baseline_auroc = all_results[ds].get("methods", {}).get(
                "initial_scores", {}
            ).get("macro_auroc", 0.5)
            for j, method in enumerate(methods_list):
                method_auroc = all_results[ds].get("methods", {}).get(
                    method, {}
                ).get("macro_auroc", 0.0)
                heatmap_data[i, j] = method_auroc - baseline_auroc

        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto",
                       vmin=-0.2, vmax=0.2)
        ax.set_xticks(range(len(methods_list)))
        ax.set_xticklabels([m.replace("_", "\n") for m in methods_list],
                          fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=9)

        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(methods_list)):
                val = heatmap_data[i, j]
                color = "white" if abs(val) > 0.1 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                       fontsize=8, color=color)

        fig.colorbar(im, ax=ax, label="AUROC improvement over baseline")
        ax.set_title("AUROC Improvement Over Initial Scores Baseline",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot_path = str(output_dir / "adaptive_bayesian_improvement_heatmap.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(plot_path)
        print(f"Saved: {plot_path}")

    return plots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_combined_evaluation(
    dataset_keys: list[str] | None = None,
    bayesian_config_key: str = "bayesian_cluster",
    max_cases: int | None = None,
) -> dict:
    """Offline combined evaluation: Bayesian + Agent max-merge.

    Uses existing Bayesian LOO scores and existing agent results to compute
    the hypothetical three-way max-merge performance without re-running the agent.
    This validates the ceiling performance of the combined pipeline.

    Returns dict of {dataset_key: {method: metrics}}.
    """
    from scipy.stats import wilcoxon

    if dataset_keys is None:
        dataset_keys = list(DATASETS.keys())

    config = CONFIGS[bayesian_config_key]
    all_results = {}

    for ds_key in dataset_keys:
        ds_info = DATASETS[ds_key]
        repo_path = TRACE_REPOS_BASE / ds_info["repo"]
        if not repo_path.exists():
            print(f"  Skipping {ds_key}: repo not found")
            continue

        print(f"\n{'='*70}")
        print(f"Combined eval: {ds_key} ({ds_info['domain']}, bg={ds_info['bg']}x)")
        print(f"{'='*70}")

        cases = load_dataset(repo_path, max_cases=max_cases)
        if not cases:
            print("  No cases found, skipping.")
            continue
        print(f"  Loaded {len(cases)} cases")

        # 1. Baseline: raw initial scores
        baseline_scored = {c.case_id: dict(c.monitor_scores) for c in cases}
        baseline_metrics = compute_metrics(cases, baseline_scored)

        # 2. Bayesian LOO scores
        random.seed(42)
        np.random.seed(42)
        bayesian_scored = run_loo_evaluation(cases, detector_kwargs=config["kwargs"])
        bayesian_metrics = compute_metrics(cases, bayesian_scored)

        # 3. Agent scores
        agent_results_path = ds_info.get("agent_results", "")
        agent_scored = load_agent_results(agent_results_path) if agent_results_path else {}
        agent_metrics = compute_metrics(cases, agent_scored) if agent_scored else None

        # 4. Three-way max-merge: max(initial, bayesian, agent)
        combined_scored = {}
        for case in cases:
            merged = {}
            for tid in case.trace_ids:
                s_initial = case.monitor_scores.get(tid, 0.0)
                s_bayesian = bayesian_scored.get(case.case_id, {}).get(tid, 0.0)
                s_agent = agent_scored.get(case.case_id, {}).get(tid, 0.0)
                merged[tid] = max(s_initial, s_bayesian, s_agent)
            combined_scored[case.case_id] = merged
        combined_metrics = compute_metrics(cases, combined_scored)

        # 5. Also compute bayesian+agent only (no initial) for comparison
        ba_scored = {}
        for case in cases:
            merged = {}
            for tid in case.trace_ids:
                s_bayesian = bayesian_scored.get(case.case_id, {}).get(tid, 0.0)
                s_agent = agent_scored.get(case.case_id, {}).get(tid, 0.0)
                merged[tid] = max(s_bayesian, s_agent)
            ba_scored[case.case_id] = merged
        ba_metrics = compute_metrics(cases, ba_scored)

        # Print comparison
        print(f"\n  {'Method':<35} {'AUROC':>8} {'AP':>8}")
        print(f"  {'-'*55}")
        print(f"  {'Initial scores':<35} {baseline_metrics['macro_auroc']:>8.4f} {baseline_metrics['macro_ap']:>8.4f}")
        print(f"  {'Bayesian+clusters (LOO)':<35} {bayesian_metrics['macro_auroc']:>8.4f} {bayesian_metrics['macro_ap']:>8.4f}")
        if agent_metrics:
            print(f"  {'Agent pipeline':<35} {agent_metrics['macro_auroc']:>8.4f} {agent_metrics['macro_ap']:>8.4f}")
        print(f"  {'Max(bayesian, agent)':<35} {ba_metrics['macro_auroc']:>8.4f} {ba_metrics['macro_ap']:>8.4f}")
        print(f"  {'Max(initial, bayesian, agent)':<35} {combined_metrics['macro_auroc']:>8.4f} {combined_metrics['macro_ap']:>8.4f}")

        # Improvement over best standalone
        best_standalone = max(
            bayesian_metrics["macro_auroc"],
            agent_metrics["macro_auroc"] if agent_metrics else 0.0,
        )
        delta = combined_metrics["macro_auroc"] - best_standalone
        print(f"\n  Combined AUROC improvement over best standalone: {delta:+.4f}")

        # Statistical test: paired Wilcoxon on per-case AUROCs
        combined_pc = combined_metrics.get("per_case", {})
        bayesian_pc = bayesian_metrics.get("per_case", {})
        agent_pc = agent_metrics.get("per_case", {}) if agent_metrics else {}

        common_cases = sorted(set(combined_pc) & set(bayesian_pc))
        if len(common_cases) >= 5:
            combined_aurocs = [combined_pc[c]["auroc"] for c in common_cases]
            bayesian_aurocs = [bayesian_pc[c]["auroc"] for c in common_cases]
            diffs = [c - b for c, b in zip(combined_aurocs, bayesian_aurocs)]
            if any(d != 0 for d in diffs):
                stat, p = wilcoxon(diffs)
                n_improved = sum(1 for d in diffs if d > 0)
                print(f"  vs Bayesian: Wilcoxon p={p:.4f}, {n_improved}/{len(common_cases)} cases improved")

        if agent_pc:
            common_cases_a = sorted(set(combined_pc) & set(agent_pc))
            if len(common_cases_a) >= 5:
                combined_aurocs_a = [combined_pc[c]["auroc"] for c in common_cases_a]
                agent_aurocs_a = [agent_pc[c]["auroc"] for c in common_cases_a]
                diffs_a = [c - a for c, a in zip(combined_aurocs_a, agent_aurocs_a)]
                if any(d != 0 for d in diffs_a):
                    stat_a, p_a = wilcoxon(diffs_a)
                    n_improved_a = sum(1 for d in diffs_a if d > 0)
                    print(f"  vs Agent:    Wilcoxon p={p_a:.4f}, {n_improved_a}/{len(common_cases_a)} cases improved")

        all_results[ds_key] = {
            "n_cases": len(cases),
            "methods": {
                "initial_scores": baseline_metrics,
                "bayesian_cluster": bayesian_metrics,
                "agent_pipeline": agent_metrics or {},
                "max_bayesian_agent": ba_metrics,
                "combined_3way": combined_metrics,
            },
        }

    # Print final summary
    print(f"\n{'='*80}")
    print("COMBINED EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Initial':>8} {'Bayesian':>8} {'Agent':>8} {'Combined':>8} {'Delta':>8}")
    print(f"{'-'*80}")
    for ds_key in sorted(all_results.keys()):
        methods = all_results[ds_key]["methods"]
        init_a = methods["initial_scores"].get("macro_auroc", 0)
        bay_a = methods["bayesian_cluster"].get("macro_auroc", 0)
        agt_a = methods["agent_pipeline"].get("macro_auroc", 0)
        comb_a = methods["combined_3way"].get("macro_auroc", 0)
        best = max(bay_a, agt_a)
        delta = comb_a - best
        print(f"{ds_key:<20} {init_a:>8.4f} {bay_a:>8.4f} {agt_a:>8.4f} {comb_a:>8.4f} {delta:>+8.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Adaptive Bayesian Sabotage Detection experiments"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to run (default: all available)"
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Configurations to run (default: all)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Maximum cases per dataset"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Run offline combined evaluation (Bayesian + Agent max-merge)"
    )
    args = parser.parse_args()

    datasets = args.datasets or list(DATASETS.keys())
    # Filter to datasets that actually exist
    datasets = [
        d for d in datasets
        if (TRACE_REPOS_BASE / DATASETS[d]["repo"]).exists()
    ]

    if not datasets:
        print("No datasets found. Check that results/safety/trace_repos/ has data.")
        sys.exit(1)

    # Combined evaluation mode: offline Bayesian + Agent max-merge
    if args.combined:
        combined_results = run_combined_evaluation(
            dataset_keys=datasets,
            max_cases=args.max_cases,
        )
        output_path = Path(args.output_dir) / "combined_bayesian_agent_results.json"
        def clean_for_json(obj):
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
            json.dump(combined_results, f, indent=2, default=clean_for_json)
        print(f"\nCombined results saved to: {output_path}")
        return combined_results

    print(f"Running experiments on {len(datasets)} datasets: {datasets}")
    print(f"Configurations: {args.configs or 'all'}")

    all_results = {}
    for ds_key in datasets:
        results = run_experiment(ds_key, args.configs, args.max_cases)
        all_results[ds_key] = results

    # Save raw results
    output_path = Path(args.output_dir) / "adaptive_bayesian_results.json"
    # Convert non-serializable values
    def clean_for_json(obj):
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
        json.dump(all_results, f, indent=2, default=clean_for_json)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<20} {'Method':<30} {'AUROC':>8} {'AP':>8} {'N cases':>8}")
    print("-" * 80)
    for ds_key in sorted(all_results.keys()):
        ds_results = all_results[ds_key]
        for method_name, method_data in sorted(ds_results.get("methods", {}).items()):
            auroc = method_data.get("macro_auroc", 0)
            ap = method_data.get("macro_ap", 0)
            n = method_data.get("n_cases", 0)
            print(f"{ds_key:<20} {method_name:<30} {auroc:>8.4f} {ap:>8.4f} {n:>8}")
        print()

    # Generate plots
    if not args.no_plots:
        plot_files = plot_results(all_results, args.output_dir)
        print(f"\nGenerated {len(plot_files)} plots.")

    return all_results


if __name__ == "__main__":
    main()
