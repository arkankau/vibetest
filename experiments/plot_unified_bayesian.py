#!/usr/bin/env python3
"""
Plot and generate write-up from unified Bayesian detection results.

Reads results/unified_bayesian_results.json and generates:
- Cross-setting AUROC/AP comparison table
- FPR-at-threshold curves
- Case-level classification bar charts
- Score distribution plots
- Markdown write-up

Usage:
    python experiments/plot_unified_bayesian.py [--input FILE] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Plot 1: Cross-setting AUROC/AP comparison
# ---------------------------------------------------------------------------

def plot_auroc_ap_comparison(all_results: dict, output_dir: Path, plt) -> str:
    """Bar chart comparing AUROC and AP across settings and methods."""
    settings = sorted(all_results.keys())
    # Collect all method keys
    all_methods = set()
    for result in all_results.values():
        for mk in result.get("methods", {}):
            if "error" not in result["methods"][mk]:
                all_methods.add(mk)
    methods = sorted(all_methods)

    if not methods or not settings:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(settings) * 3), 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for ax_idx, (metric_key, metric_name) in enumerate([
        ("macro_auroc", "AUROC"), ("macro_ap", "Average Precision")
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(settings))
        width = 0.8 / max(1, len(methods))

        for i, method in enumerate(methods):
            vals = []
            for s in settings:
                md = all_results[s].get("methods", {}).get(method, {})
                vals.append(md.get(metric_key, 0.0))
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=method.replace("_", " "), color=colors[i])

        ax.set_ylabel(f"Macro {metric_name}")
        ax.set_title(f"Macro {metric_name}")
        ax.set_xticks(x)
        setting_labels = [all_results[s].get("name", s) for s in settings]
        ax.set_xticklabels(setting_labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Unified Bayesian Detection: AUROC & AP", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = str(output_dir / "unified_bayesian_auroc_ap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ---------------------------------------------------------------------------
# Plot 2: FPR-at-threshold curves
# ---------------------------------------------------------------------------

def plot_fpr_curves(all_results: dict, output_dir: Path, plt) -> str:
    """FPR vs threshold curves, one subplot per setting, methods overlaid."""
    settings = sorted(all_results.keys())
    n = len(settings)
    if n == 0:
        return ""

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, setting_key in enumerate(settings):
        ax = axes[idx // cols][idx % cols]
        result = all_results[setting_key]
        methods = result.get("methods", {})

        color_idx = 0
        for method_key, method_data in sorted(methods.items()):
            if "error" in method_data or "threshold_metrics" not in method_data:
                continue
            tm = method_data["threshold_metrics"]
            thresholds = [m["threshold"] for m in tm]
            trace_fprs = [m["trace_fpr"] for m in tm]
            case_fprs = [m["case_fpr"] for m in tm]

            label = method_key.replace("_", " ")
            ax.plot(thresholds, trace_fprs, "o-", color=colors[color_idx % 10],
                    label=f"{label} (trace)", linewidth=1.5, markersize=4)
            ax.plot(thresholds, case_fprs, "s--", color=colors[color_idx % 10],
                    label=f"{label} (case)", linewidth=1, markersize=3, alpha=0.7)
            color_idx += 1

        ax.set_xlabel("Threshold")
        ax.set_ylabel("False Positive Rate")
        ax.set_title(result.get("name", setting_key))
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("FPR vs Threshold", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = str(output_dir / "unified_bayesian_fpr_curves.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ---------------------------------------------------------------------------
# Plot 3: Case-level classification bar charts
# ---------------------------------------------------------------------------

def plot_case_classification(all_results: dict, output_dir: Path, plt) -> str:
    """Case-level accuracy/F1 at thresholds 0.5 and 0.7."""
    settings = sorted(all_results.keys())
    target_thresholds = [0.5, 0.7]

    fig, axes = plt.subplots(1, len(target_thresholds),
                             figsize=(8 * len(target_thresholds), 6))
    if len(target_thresholds) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax_idx, target_t in enumerate(target_thresholds):
        ax = axes[ax_idx]
        # Collect (setting, method, accuracy, f1)
        labels = []
        accs = []
        f1s = []
        bar_colors = []

        color_idx = 0
        for setting_key in settings:
            result = all_results[setting_key]
            setting_name = result.get("name", setting_key)
            for method_key, method_data in sorted(result.get("methods", {}).items()):
                if "error" in method_data or "threshold_metrics" not in method_data:
                    continue
                tm = method_data["threshold_metrics"]
                match = [m for m in tm if abs(m["threshold"] - target_t) < 0.01]
                if not match:
                    continue
                m = match[0]
                labels.append(f"{setting_name}\n{method_key}")
                accs.append(m["case_accuracy"])
                f1s.append(m["case_f1"])
                bar_colors.append(colors[color_idx % 10])
                color_idx += 1

        if not labels:
            continue

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, accs, w, label="Accuracy", color="steelblue", alpha=0.8)
        ax.bar(x + w / 2, f1s, w, label="F1", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Case-Level Classification @ threshold={target_t}")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Case-Level Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = str(output_dir / "unified_bayesian_case_classification.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ---------------------------------------------------------------------------
# Plot 4: Benign-only FPR analysis
# ---------------------------------------------------------------------------

def plot_benign_fpr(all_results: dict, output_dir: Path, plt) -> str:
    """Benign-only case FPR vs threshold for each setting."""
    settings = sorted(all_results.keys())
    # Only settings with benign-only cases
    settings_with_benign = [
        s for s in settings
        if any(
            method_data.get("n_benign_cases", 0) > 0
            for method_data in all_results[s].get("methods", {}).values()
            if "error" not in method_data
        )
    ]
    if not settings_with_benign:
        return ""

    n = len(settings_with_benign)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, setting_key in enumerate(settings_with_benign):
        ax = axes[idx // cols][idx % cols]
        result = all_results[setting_key]
        methods = result.get("methods", {})

        color_idx = 0
        for method_key, method_data in sorted(methods.items()):
            if "error" in method_data or "benign_only" not in method_data:
                continue
            bo = method_data["benign_only"]
            if not bo or bo[0].get("n_benign_cases", 0) == 0:
                continue
            thresholds = [b["threshold"] for b in bo]
            fprs = [b["fpr"] for b in bo]
            label = method_key.replace("_", " ")
            ax.plot(thresholds, fprs, "o-", color=colors[color_idx % 10],
                    label=label, linewidth=1.5, markersize=5)
            color_idx += 1

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Benign-Only Case FPR")
        ax.set_title(f"{result.get('name', setting_key)} ({result.get('n_benign_cases', '?')} benign cases)")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Benign-Only Case False Positive Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = str(output_dir / "unified_bayesian_benign_fpr.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_markdown(all_results: dict, plots: list[str], output_dir: Path) -> str:
    """Generate unified markdown write-up."""
    lines = []
    lines.append("# Unified Bayesian Detection Evaluation\n")
    lines.append("Cross-setting comparison of Bayesian sabotage/misuse detectors ")
    lines.append("with false positive analysis on benign-only cases.\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Setting | Cases | Traces/Case | Positive | Benign-Only |")
    lines.append("|---------|-------|-------------|----------|-------------|")
    for setting_key in sorted(all_results.keys()):
        r = all_results[setting_key]
        lines.append(
            f"| {r.get('name', setting_key)} | {r.get('n_cases', 0)} | "
            f"{r.get('traces_per_case', '?')} | {r.get('n_positive_cases', 0)} | "
            f"{r.get('n_benign_cases', 0)} |"
        )
    lines.append("")

    # AUROC/AP table
    lines.append("## Detection Performance (AUROC / AP)\n")
    lines.append("| Setting | Method | Macro AUROC | Macro AP | Micro AUROC | N Cases |")
    lines.append("|---------|--------|-------------|----------|-------------|---------|")
    for setting_key in sorted(all_results.keys()):
        r = all_results[setting_key]
        setting_name = r.get("name", setting_key)
        for method_key, md in sorted(r.get("methods", {}).items()):
            if "error" in md:
                continue
            lines.append(
                f"| {setting_name} | {method_key} | "
                f"{md.get('macro_auroc', 0):.4f} | {md.get('macro_ap', 0):.4f} | "
                f"{md.get('micro_auroc', 0):.4f} | {md.get('n_cases', 0)} |"
            )
    lines.append("")

    # FPR table at key thresholds
    lines.append("## False Positive Rates\n")
    lines.append("### Trace-Level FPR\n")
    lines.append("| Setting | Method | FPR@0.3 | FPR@0.5 | FPR@0.7 | FPR@0.9 |")
    lines.append("|---------|--------|---------|---------|---------|---------|")
    for setting_key in sorted(all_results.keys()):
        r = all_results[setting_key]
        setting_name = r.get("name", setting_key)
        for method_key, md in sorted(r.get("methods", {}).items()):
            if "error" in md or "threshold_metrics" not in md:
                continue
            tm = {m["threshold"]: m for m in md["threshold_metrics"]}
            fprs = []
            for t in [0.3, 0.5, 0.7, 0.9]:
                if t in tm:
                    fprs.append(f"{tm[t]['trace_fpr']:.4f}")
                else:
                    fprs.append("—")
            lines.append(
                f"| {setting_name} | {method_key} | "
                f"{' | '.join(fprs)} |"
            )
    lines.append("")

    # Benign-only FPR
    lines.append("### Benign-Only Case FPR\n")
    lines.append("| Setting | Method | FPR@0.3 | FPR@0.5 | FPR@0.7 | FPR@0.9 |")
    lines.append("|---------|--------|---------|---------|---------|---------|")
    for setting_key in sorted(all_results.keys()):
        r = all_results[setting_key]
        setting_name = r.get("name", setting_key)
        for method_key, md in sorted(r.get("methods", {}).items()):
            if "error" in md or "benign_only" not in md:
                continue
            bo = {b["threshold"]: b for b in md["benign_only"]}
            if not bo or list(bo.values())[0].get("n_benign_cases", 0) == 0:
                continue
            fprs = []
            for t in [0.3, 0.5, 0.7, 0.9]:
                if t in bo:
                    fprs.append(f"{bo[t]['fpr']:.3f}")
                else:
                    fprs.append("—")
            lines.append(
                f"| {setting_name} | {method_key} | "
                f"{' | '.join(fprs)} |"
            )
    lines.append("")

    # Case-level classification
    lines.append("## Case-Level Classification @ threshold=0.5\n")
    lines.append("| Setting | Method | Accuracy | F1 | Precision | TPR | FPR |")
    lines.append("|---------|--------|----------|-----|-----------|-----|-----|")
    for setting_key in sorted(all_results.keys()):
        r = all_results[setting_key]
        setting_name = r.get("name", setting_key)
        for method_key, md in sorted(r.get("methods", {}).items()):
            if "error" in md or "threshold_metrics" not in md:
                continue
            tm = {m["threshold"]: m for m in md["threshold_metrics"]}
            if 0.5 not in tm:
                continue
            m = tm[0.5]
            lines.append(
                f"| {setting_name} | {method_key} | "
                f"{m['case_accuracy']:.3f} | {m['case_f1']:.3f} | "
                f"{m['case_precision']:.3f} | {m['case_tpr']:.3f} | "
                f"{m['case_fpr']:.3f} |"
            )
    lines.append("")

    # Plots
    lines.append("## Figures\n")
    for p in plots:
        if p:
            rel = Path(p).name
            lines.append(f"![{rel}]({rel})\n")

    md_text = "\n".join(lines)
    md_path = output_dir / "bayesian_detection_results.md"
    md_path.write_text(md_text)
    print(f"Saved: {md_path}")
    return str(md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot unified Bayesian detection results"
    )
    parser.add_argument("--input", default="results/unified_bayesian_results.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print("Run experiments/run_unified_bayesian.py first.")
        return

    with open(input_path) as f:
        all_results = json.load(f)

    print("=" * 70)
    print("Plotting Unified Bayesian Detection Results")
    print("=" * 70)
    print(f"Settings: {list(all_results.keys())}")

    plt = setup_matplotlib()

    plots = []
    plots.append(plot_auroc_ap_comparison(all_results, output_dir, plt))
    plots.append(plot_fpr_curves(all_results, output_dir, plt))
    plots.append(plot_case_classification(all_results, output_dir, plt))
    plots.append(plot_benign_fpr(all_results, output_dir, plt))

    md_path = generate_markdown(all_results, plots, output_dir)

    print(f"\nDone. Write-up: {md_path}")
    print(f"Plots: {[p for p in plots if p]}")


if __name__ == "__main__":
    main()
