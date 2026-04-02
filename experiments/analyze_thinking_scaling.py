"""Analyze thinking scaling experiments for distributed misuse detection.

Loads results from experiments run at different reasoning_effort levels and
produces metrics + plots showing how inference compute affects safety outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonlines
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

EFFORT_ORDER = ["low", "medium", "high", "xhigh"]
EFFORT_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "XHigh"}
EFFORT_COLORS = {"low": "#4878d0", "medium": "#ee854a", "high": "#6acc64", "xhigh": "#d65f5f"}


def load_results(paths: list[Path]) -> dict[str, list[dict]]:
    """Load JSONL result files, grouped by reasoning_effort."""
    by_effort: dict[str, list[dict]] = {}
    for p in paths:
        with jsonlines.open(str(p)) as reader:
            for row in reader:
                effort = row.get("reasoning_effort", "medium")
                by_effort.setdefault(effort, []).append(row)
    return by_effort


def per_case_auroc(row: dict) -> float | None:
    """Compute per-case AUROC from trace scores and ground truth."""
    scoring = row.get("scoring") or {}
    trace_scores = scoring.get("trace_scores") or {}
    if not trace_scores:
        # Fall back to test metadata
        test = (row.get("tests") or [{}])[0]
        trace_scores = (test.get("metadata") or {}).get("trace_scores") or {}
    if not trace_scores:
        return None

    gt = row.get("ground_truth") or {}
    gt_positive = set()
    for key in ("harmful_trace_files", "hacked_trace_files", "suspicious_trace_files"):
        if key in gt:
            gt_positive.update(gt[key])
    if not gt_positive:
        # Check has_distributed_misuse
        if not gt.get("has_distributed_misuse", False):
            return None  # All-negative case, AUROC undefined

    scores_list = []
    labels_list = []
    for trace_path, score in trace_scores.items():
        # Normalize path
        p = trace_path.strip()
        for prefix in ("/workspace/repo/", "/workspace/", "workspace/repo/", "workspace/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        scores_list.append(float(score))
        labels_list.append(1 if p in gt_positive else 0)

    if sum(labels_list) == 0 or sum(labels_list) == len(labels_list):
        return None  # Can't compute AUROC with single class

    # Manual AUROC computation (avoid sklearn dependency)
    pairs = sorted(zip(scores_list, labels_list), reverse=True)
    tp = 0
    fp = 0
    tp_prev = 0
    fp_prev = 0
    auc = 0.0
    n_pos = sum(labels_list)
    n_neg = len(labels_list) - n_pos
    prev_score = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            tp_prev = tp
            fp_prev = fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
    if n_pos > 0 and n_neg > 0:
        return auc / (n_pos * n_neg)
    return None


def per_case_ap(row: dict) -> float | None:
    """Compute per-case Average Precision from trace scores."""
    scoring = row.get("scoring") or {}
    trace_scores = scoring.get("trace_scores") or {}
    if not trace_scores:
        test = (row.get("tests") or [{}])[0]
        trace_scores = (test.get("metadata") or {}).get("trace_scores") or {}
    if not trace_scores:
        return None

    gt = row.get("ground_truth") or {}
    gt_positive = set()
    for key in ("harmful_trace_files", "hacked_trace_files", "suspicious_trace_files"):
        if key in gt:
            gt_positive.update(gt[key])
    if not gt_positive:
        return None

    scores_list = []
    labels_list = []
    for trace_path, score in trace_scores.items():
        p = trace_path.strip()
        for prefix in ("/workspace/repo/", "/workspace/", "workspace/repo/", "workspace/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        scores_list.append(float(score))
        labels_list.append(1 if p in gt_positive else 0)

    if sum(labels_list) == 0:
        return None

    # Sort by score descending
    pairs = sorted(zip(scores_list, labels_list), reverse=True)
    tp = 0
    ap = 0.0
    for i, (_, label) in enumerate(pairs):
        if label == 1:
            tp += 1
            ap += tp / (i + 1)
    return ap / sum(labels_list) if sum(labels_list) > 0 else 0.0


def extract_usage(row: dict) -> dict:
    """Extract token usage from a result row."""
    usage = row.get("usage") or {}
    totals = usage.get("usage_totals") or {}
    return {
        "input_tokens": int(totals.get("input_tokens", 0)),
        "output_tokens": int(totals.get("output_tokens", 0)),
        "total_tokens": int(totals.get("total_tokens", 0)),
        "reasoning_tokens": int(totals.get("reasoning_tokens", 0)),
    }


def compute_metrics(by_effort: dict[str, list[dict]]) -> dict[str, dict]:
    """Compute per-effort aggregate metrics."""
    results = {}
    for effort in EFFORT_ORDER:
        rows = by_effort.get(effort, [])
        if not rows:
            continue

        aurocs = [per_case_auroc(r) for r in rows]
        aurocs = [a for a in aurocs if a is not None]
        aps = [per_case_ap(r) for r in rows]
        aps = [a for a in aps if a is not None]
        usages = [extract_usage(r) for r in rows]

        # Per-case metrics from scoring block
        precisions = []
        recalls = []
        coverages = []
        for r in rows:
            s = r.get("scoring") or {}
            if "trace_precision" in s:
                precisions.append(float(s["trace_precision"]))
            if "trace_recall" in s:
                recalls.append(float(s["trace_recall"]))
            if "campaign_coverage" in s:
                coverages.append(float(s["campaign_coverage"]))

        total_tokens = [u["total_tokens"] for u in usages]
        reasoning_tokens = [u["reasoning_tokens"] for u in usages]
        output_tokens = [u["output_tokens"] for u in usages]

        results[effort] = {
            "n_cases": len(rows),
            "auroc_mean": np.mean(aurocs) if aurocs else float("nan"),
            "auroc_std": np.std(aurocs) if aurocs else float("nan"),
            "aurocs": aurocs,
            "ap_mean": np.mean(aps) if aps else float("nan"),
            "ap_std": np.std(aps) if aps else float("nan"),
            "aps": aps,
            "precision_mean": np.mean(precisions) if precisions else float("nan"),
            "recall_mean": np.mean(recalls) if recalls else float("nan"),
            "coverage_mean": np.mean(coverages) if coverages else float("nan"),
            "coverage_std": np.std(coverages) if coverages else float("nan"),
            "coverages": coverages,
            "total_tokens_mean": np.mean(total_tokens) if total_tokens else 0,
            "total_tokens_std": np.std(total_tokens) if total_tokens else 0,
            "reasoning_tokens_mean": np.mean(reasoning_tokens) if reasoning_tokens else 0,
            "reasoning_tokens_std": np.std(reasoning_tokens) if reasoning_tokens else 0,
            "output_tokens_mean": np.mean(output_tokens) if output_tokens else 0,
            "total_tokens_all": total_tokens,
            "reasoning_tokens_all": reasoning_tokens,
        }
    return results


def plot_metrics_vs_effort(metrics: dict[str, dict], out_dir: Path) -> None:
    """Create a 2x2 figure: AUROC, AP, Campaign Coverage, Token Usage vs effort."""
    efforts = [e for e in EFFORT_ORDER if e in metrics]
    if not efforts:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # --- AUROC ---
    ax = axes[0, 0]
    means = [metrics[e]["auroc_mean"] for e in efforts]
    stds = [metrics[e]["auroc_std"] for e in efforts]
    colors = [EFFORT_COLORS[e] for e in efforts]
    ax.bar([EFFORT_LABELS[e] for e in efforts], means, yerr=stds,
           color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    # Scatter individual points
    for i, e in enumerate(efforts):
        ax.scatter([i] * len(metrics[e]["aurocs"]), metrics[e]["aurocs"],
                   color="black", s=15, zorder=5, alpha=0.6)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Case AUROC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # --- AP ---
    ax = axes[0, 1]
    means = [metrics[e]["ap_mean"] for e in efforts]
    stds = [metrics[e]["ap_std"] for e in efforts]
    ax.bar([EFFORT_LABELS[e] for e in efforts], means, yerr=stds,
           color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    for i, e in enumerate(efforts):
        ax.scatter([i] * len(metrics[e]["aps"]), metrics[e]["aps"],
                   color="black", s=15, zorder=5, alpha=0.6)
    ax.set_ylabel("Average Precision")
    ax.set_title("Per-Case AP")
    ax.set_ylim(0, 1.05)

    # --- Campaign Coverage ---
    ax = axes[1, 0]
    means = [metrics[e]["coverage_mean"] for e in efforts]
    stds = [metrics[e]["coverage_std"] for e in efforts]
    ax.bar([EFFORT_LABELS[e] for e in efforts], means, yerr=stds,
           color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    for i, e in enumerate(efforts):
        ax.scatter([i] * len(metrics[e]["coverages"]), metrics[e]["coverages"],
                   color="black", s=15, zorder=5, alpha=0.6)
    ax.set_ylabel("Campaign Coverage")
    ax.set_title("Campaign Coverage")
    ax.set_ylim(0, 1.05)

    # --- Token Usage ---
    ax = axes[1, 1]
    total_means = [metrics[e]["total_tokens_mean"] for e in efforts]
    total_stds = [metrics[e]["total_tokens_std"] for e in efforts]
    reasoning_means = [metrics[e]["reasoning_tokens_mean"] for e in efforts]
    x = np.arange(len(efforts))
    width = 0.35
    ax.bar(x - width / 2, total_means, width, yerr=total_stds,
           label="Total", color=[EFFORT_COLORS[e] for e in efforts],
           capsize=4, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, reasoning_means, width,
           label="Reasoning", color=[EFFORT_COLORS[e] for e in efforts],
           alpha=0.5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Tokens")
    ax.set_title("Token Usage per Case")
    ax.set_xticks(x)
    ax.set_xticklabels([EFFORT_LABELS[e] for e in efforts])
    ax.legend()

    fig.suptitle("Inference Compute Scaling — Distributed Misuse Detection (GPT-5.4)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "thinking_scaling_metrics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tokens_vs_metrics(metrics: dict[str, dict], out_dir: Path) -> None:
    """Scatter plot of per-case metrics vs. tokens used (natural variance)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for effort in EFFORT_ORDER:
        if effort not in metrics:
            continue
        m = metrics[effort]
        color = EFFORT_COLORS[effort]
        label = EFFORT_LABELS[effort]
        tokens = m["total_tokens_all"]

        # AUROC vs tokens
        if len(m["aurocs"]) == len(tokens):
            axes[0].scatter(tokens, m["aurocs"], color=color, label=label,
                            s=30, alpha=0.8, edgecolor="black", linewidth=0.3)
        # Coverage vs tokens
        if len(m["coverages"]) == len(tokens):
            axes[1].scatter(tokens, m["coverages"], color=color, label=label,
                            s=30, alpha=0.8, edgecolor="black", linewidth=0.3)

    axes[0].set_xlabel("Total Tokens")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("AUROC vs. Inference Compute")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].set_xlabel("Total Tokens")
    axes[1].set_ylabel("Campaign Coverage")
    axes[1].set_title("Coverage vs. Inference Compute")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    fig.suptitle("Per-Case Metrics vs. Token Usage", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "thinking_scaling_tokens_scatter.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_bg_comparison(
    metrics_by_bg: dict[str, dict[str, dict]], out_dir: Path,
) -> None:
    """Create a 2x3 figure comparing metrics across bg settings and effort levels."""
    bg_labels = sorted(metrics_by_bg.keys())
    if not bg_labels:
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))

    for col, bg_label in enumerate(bg_labels):
        metrics = metrics_by_bg[bg_label]
        efforts = [e for e in EFFORT_ORDER if e in metrics]
        if not efforts:
            continue
        colors = [EFFORT_COLORS[e] for e in efforts]
        x_labels = [EFFORT_LABELS[e] for e in efforts]

        # Row 0: AUROC
        ax = axes[0, col]
        means = [metrics[e]["auroc_mean"] for e in efforts]
        stds = [metrics[e]["auroc_std"] for e in efforts]
        ax.bar(x_labels, means, yerr=stds, color=colors, capsize=5,
               edgecolor="black", linewidth=0.5)
        for i, e in enumerate(efforts):
            ax.scatter([i] * len(metrics[e]["aurocs"]), metrics[e]["aurocs"],
                       color="black", s=15, zorder=5, alpha=0.6)
        ax.set_ylabel("AUROC" if col == 0 else "")
        ax.set_title(f"bg={bg_label}")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        # Row 1: AP
        ax = axes[1, col]
        means = [metrics[e]["ap_mean"] for e in efforts]
        stds = [metrics[e]["ap_std"] for e in efforts]
        ax.bar(x_labels, means, yerr=stds, color=colors, capsize=5,
               edgecolor="black", linewidth=0.5)
        for i, e in enumerate(efforts):
            ax.scatter([i] * len(metrics[e]["aps"]), metrics[e]["aps"],
                       color="black", s=15, zorder=5, alpha=0.6)
        ax.set_ylabel("Average Precision" if col == 0 else "")
        ax.set_xlabel("Reasoning Effort")
        ax.set_ylim(0, 1.05)

    # Third column: token usage comparison across bg settings
    ax_tok = axes[0, 2]
    x = np.arange(len(EFFORT_ORDER))
    width = 0.25
    for i, bg_label in enumerate(bg_labels):
        metrics = metrics_by_bg[bg_label]
        efforts = [e for e in EFFORT_ORDER if e in metrics]
        means = [metrics.get(e, {}).get("total_tokens_mean", 0) for e in EFFORT_ORDER]
        stds = [metrics.get(e, {}).get("total_tokens_std", 0) for e in EFFORT_ORDER]
        offset = (i - (len(bg_labels) - 1) / 2) * width
        ax_tok.bar(x + offset, means, width, yerr=stds,
                   label=f"bg={bg_label}", capsize=3, edgecolor="black", linewidth=0.5)
    ax_tok.set_ylabel("Total Tokens")
    ax_tok.set_title("Token Usage")
    ax_tok.set_xticks(x)
    ax_tok.set_xticklabels([EFFORT_LABELS[e] for e in EFFORT_ORDER])
    ax_tok.legend(fontsize=7)

    ax_reason = axes[1, 2]
    for i, bg_label in enumerate(bg_labels):
        metrics = metrics_by_bg[bg_label]
        means = [metrics.get(e, {}).get("reasoning_tokens_mean", 0) for e in EFFORT_ORDER]
        stds = [metrics.get(e, {}).get("reasoning_tokens_std", 0) for e in EFFORT_ORDER]
        offset = (i - (len(bg_labels) - 1) / 2) * width
        ax_reason.bar(x + offset, means, width, yerr=stds,
                      label=f"bg={bg_label}", capsize=3, edgecolor="black", linewidth=0.5)
    ax_reason.set_ylabel("Reasoning Tokens")
    ax_reason.set_xlabel("Reasoning Effort")
    ax_reason.set_xticks(x)
    ax_reason.set_xticklabels([EFFORT_LABELS[e] for e in EFFORT_ORDER])
    ax_reason.legend(fontsize=7)

    fig.suptitle(
        "Inference Compute Scaling — Distributed Misuse Detection (GPT-5.4, cyber, d=6)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = out_dir / "thinking_scaling_combined.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary_table(metrics: dict[str, dict], label: str = "") -> None:
    """Print a summary table to stdout."""
    print()
    if label:
        print(f"  [{label}]")
    print("=" * 90)
    print(f"{'Effort':<10} {'N':>3} {'AUROC':>12} {'AP':>12} {'Coverage':>12} "
          f"{'Total Tok':>12} {'Reason Tok':>12}")
    print("-" * 90)
    for effort in EFFORT_ORDER:
        if effort not in metrics:
            continue
        m = metrics[effort]
        print(
            f"{effort:<10} {m['n_cases']:>3} "
            f"{m['auroc_mean']:>5.3f}±{m['auroc_std']:.3f} "
            f"{m['ap_mean']:>5.3f}±{m['ap_std']:.3f} "
            f"{m['coverage_mean']:>5.3f}±{m['coverage_std']:.3f} "
            f"{m['total_tokens_mean']:>10.0f}±{m['total_tokens_std']:.0f} "
            f"{m['reasoning_tokens_mean']:>10.0f}±{m['reasoning_tokens_std']:.0f}"
        )
    print("=" * 90)
    print()


def _infer_bg(row: dict) -> str:
    """Infer background multiplier from dataset name or trace count."""
    ds = row.get("dataset", "")
    if "bg100" in ds:
        return "100×"
    if "bg20" in ds:
        return "20×"
    # Fallback: infer from traces_per_case
    tpc = row.get("traces_per_case", 0)
    if tpc > 100:
        return "100×"
    return "20×"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_files",
        nargs="*",
        type=Path,
        help="JSONL result files. If empty, auto-discovers results/safety_dm_*gpt54_effort_*.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for plots and JSON summary.",
    )
    args = parser.parse_args()

    if args.result_files:
        paths = args.result_files
    else:
        paths = sorted(Path("results").glob("safety_dm_*gpt54_effort_*.jsonl"))
        if not paths:
            raise SystemExit(
                "No result files found. Pass paths explicitly or run experiments first."
            )
    print(f"Loading {len(paths)} result files: {[p.name for p in paths]}")

    # Load and split by bg setting
    all_rows: list[dict] = []
    for p in paths:
        with jsonlines.open(str(p)) as reader:
            for row in reader:
                all_rows.append(row)

    by_bg: dict[str, dict[str, list[dict]]] = {}
    for row in all_rows:
        bg = _infer_bg(row)
        effort = row.get("reasoning_effort", "medium")
        by_bg.setdefault(bg, {}).setdefault(effort, []).append(row)

    # Compute metrics for each bg setting
    metrics_by_bg: dict[str, dict[str, dict]] = {}
    for bg_label in sorted(by_bg.keys()):
        by_effort = by_bg[bg_label]
        print(f"\n--- Background: {bg_label} ---")
        for effort, rows in sorted(by_effort.items()):
            print(f"  {effort}: {len(rows)} cases")
        metrics = compute_metrics(by_effort)
        metrics_by_bg[bg_label] = metrics
        print_summary_table(metrics, label=bg_label)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-bg plots
    for bg_label, metrics in metrics_by_bg.items():
        suffix = bg_label.replace("×", "x")
        plot_metrics_vs_effort(metrics, args.out_dir)
        # Rename to include bg
        src = args.out_dir / "thinking_scaling_metrics.png"
        dst = args.out_dir / f"thinking_scaling_metrics_bg{suffix}.png"
        if src.exists():
            src.rename(dst)
            print(f"Renamed: {dst}")
        plot_tokens_vs_metrics(metrics, args.out_dir)
        src = args.out_dir / "thinking_scaling_tokens_scatter.png"
        dst = args.out_dir / f"thinking_scaling_tokens_scatter_bg{suffix}.png"
        if src.exists():
            src.rename(dst)
            print(f"Renamed: {dst}")

    # Combined plot
    if len(metrics_by_bg) > 1:
        plot_combined_bg_comparison(metrics_by_bg, args.out_dir)

    # Save JSON summary
    summary_path = args.out_dir / "thinking_scaling_summary.json"
    summary: dict = {}
    for bg_label, metrics in metrics_by_bg.items():
        summary[bg_label] = {}
        for effort, m in metrics.items():
            summary[bg_label][effort] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in m.items()
                if not isinstance(v, list)
            }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
