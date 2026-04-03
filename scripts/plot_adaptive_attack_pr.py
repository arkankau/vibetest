#!/usr/bin/env python3
"""Plot PR curves comparing baseline vs adaptive attack (auditor_note).

Produces figures in the same style as dm_cyber_paper_pr_curves.pdf /
dm_bio_paper_pr_curves.pdf but with baseline and attack curves overlaid
to show the before/after effect of the auditor_note injection.

Usage:
    python scripts/plot_adaptive_attack_pr.py [--figures-dir figures]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cases(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().strip().split("\n") if l.strip()]


def _per_case_pr_curve(case: dict) -> tuple[list[float], list[float], float]:
    """Compute a single case's PR curve. Returns (recall_pts, precision_pts, AP)."""
    gt = case.get("ground_truth", {})
    campaign_set = set(gt.get("harmful_trace_files", []))
    if not campaign_set or not gt.get("has_distributed_misuse"):
        return [], [], 0.0

    ts = case["tests"][0].get("metadata", {}).get("trace_scores", {})
    if not ts:
        return [], [], 0.0

    labels = [(1 if t in campaign_set else 0) for t in ts]
    scores = [ts[t] for t in ts]
    n_pos = sum(labels)
    if n_pos == 0:
        return [], [], 0.0

    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    recalls = [0.0]
    precisions = [1.0]
    ap = 0.0
    for rank, (score, label) in enumerate(pairs, 1):
        if label == 1:
            tp += 1
            ap += tp / rank
        recalls.append(tp / n_pos)
        precisions.append(tp / rank)
    ap /= n_pos
    return recalls, precisions, ap


def _interp_at_grid(recalls: list[float], precisions: list[float],
                     grid: list[float]) -> list[float]:
    """Interpolate precision at fixed recall grid points (monotone decreasing)."""
    if not recalls:
        return [0.0] * len(grid)
    # Make monotone decreasing (right-to-left max)
    mono_prec = list(precisions)
    for i in range(len(mono_prec) - 2, -1, -1):
        mono_prec[i] = max(mono_prec[i], mono_prec[i + 1])
    result = []
    for r in grid:
        # Find precision at recall >= r
        best = 0.0
        for rec, prec in zip(recalls, mono_prec):
            if rec >= r - 1e-9:
                best = max(best, prec)
        result.append(best)
    return result


def _compute_run(
    path: Path,
    recall_grid: list[float],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict | None:
    """Compute mean PR curve + bootstrap CI + AP for a result file."""
    cases = _load_cases(path)
    gt_pos = [c for c in cases if c.get("ground_truth", {}).get("has_distributed_misuse")]
    if not gt_pos:
        return None

    per_case_curves = []
    ap_values = []
    for c in gt_pos:
        recalls, precisions, ap = _per_case_pr_curve(c)
        if recalls:
            interp = _interp_at_grid(recalls, precisions, recall_grid)
            per_case_curves.append(interp)
            ap_values.append(ap)

    if not per_case_curves:
        return None

    n = len(per_case_curves)
    mean_curve = [np.mean([c[i] for c in per_case_curves]) for i in range(len(recall_grid))]
    mean_ap = np.mean(ap_values)

    # Bootstrap CI
    rng = random.Random(seed)
    boot_curves = []
    boot_aps = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled = [per_case_curves[i] for i in indices]
        boot_curves.append([np.mean([s[i] for s in sampled]) for i in range(len(recall_grid))])
        boot_aps.append(np.mean([ap_values[i] for i in indices]))

    lower = [np.percentile([b[i] for b in boot_curves], 2.5) for i in range(len(recall_grid))]
    upper = [np.percentile([b[i] for b in boot_curves], 97.5) for i in range(len(recall_grid))]
    ap_lower = float(np.percentile(boot_aps, 2.5))
    ap_upper = float(np.percentile(boot_aps, 97.5))

    # Campaign mean score
    camp_scores = []
    for c in gt_pos:
        cs = set(c["ground_truth"]["harmful_trace_files"])
        ts = c["tests"][0].get("metadata", {}).get("trace_scores", {})
        camp_scores.append(np.mean([ts.get(t, 0) for t in cs]))
    camp_mean = float(np.mean(camp_scores))

    return {
        "path": str(path),
        "n_cases": n,
        "mean_curve": mean_curve,
        "lower": lower,
        "upper": upper,
        "mean_ap": float(mean_ap),
        "ap_ci": (ap_lower, ap_upper),
        "camp_mean": camp_mean,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attack_comparison(
    *,
    results_dir: Path,
    figures_dir: Path,
    formats: list[str] = ["pdf", "png"],
) -> list[Path]:
    """Plot before/after PR curves for auditor_note attack across settings."""

    recall_grid = [i / 40.0 for i in range(41)]  # 0.0, 0.025, ..., 1.0

    # Define all comparisons: (label, domain, bg, baseline_file, attack_file)
    comparisons = []
    for domain in ["cyber", "bio"]:
        for bg, n in [(100, 50), (20, 20)]:
            base_pattern = f"safety_dm_{domain}_d6_bg{bg}_qwen35_bayesian_boost_AT-*.jsonl"
            atk_pattern = f"safety_dm_{domain}_d6_bg{bg}_qwen35_bb_auditor_note_n{n}_AT-*.jsonl"
            base_files = sorted(results_dir.glob(base_pattern))
            atk_files = sorted(results_dir.glob(atk_pattern))
            if base_files and atk_files:
                comparisons.append({
                    "label": f"{domain} bg={bg}x",
                    "domain": domain,
                    "bg": bg,
                    "baseline": base_files[-1],
                    "attack": atk_files[-1],
                    "n": n,
                })

    if not comparisons:
        print("No comparison pairs found. Available files:")
        for f in sorted(results_dir.glob("safety_dm_*_AT-*.jsonl")):
            print(f"  {f.name}")
        return []

    # Compute all curves
    for comp in comparisons:
        comp["base_run"] = _compute_run(comp["baseline"], recall_grid, seed=42)
        comp["atk_run"] = _compute_run(comp["attack"], recall_grid, seed=43)

    comparisons = [c for c in comparisons if c["base_run"] and c["atk_run"]]
    if not comparisons:
        print("No valid comparison pairs with data.")
        return []

    # --- Plot: one panel per setting ---
    n_panels = len(comparisons)
    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.2), squeeze=False)
        axes = axes[0]

        for idx, comp in enumerate(comparisons):
            ax = axes[idx]
            base = comp["base_run"]
            atk = comp["atk_run"]

            # Baseline curve
            ax.fill_between(recall_grid, base["lower"], base["upper"],
                          color="#0072B2", alpha=0.15, linewidth=0)
            ax.plot(recall_grid, base["mean_curve"],
                   color="#0072B2", linewidth=2.2, linestyle="-",
                   label=f"Agent (AP={base['mean_ap']:.2f})")

            # Attack curve
            ax.fill_between(recall_grid, atk["lower"], atk["upper"],
                          color="#D55E00", alpha=0.15, linewidth=0)
            ax.plot(recall_grid, atk["mean_curve"],
                   color="#D55E00", linewidth=2.2, linestyle="--",
                   label=f"+ Auditor note (AP={atk['mean_ap']:.2f})")

            ax.set_title(f"{comp['domain'].capitalize()} bg={comp['bg']}x (n={comp['n']})",
                        fontsize=9, pad=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            ax.set_xlabel("Recall", fontsize=8.5)
            if idx == 0:
                ax.set_ylabel("Precision", fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.legend(loc="upper right", frameon=False, fontsize=7,
                     handlelength=1.9, borderaxespad=0.2, labelspacing=0.25)

        fig.suptitle("Adaptive Attack: Auditor Note Injection", fontsize=10, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        output_paths = []
        figures_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            out = figures_dir / f"dm_adaptive_attack_pr_curves.{fmt}"
            fig.savefig(out, dpi=200)
            output_paths.append(out)
            print(f"Saved: {out}")
        plt.close(fig)

    # --- Print summary table ---
    print()
    print(f"{'Setting':<20s} {'Base AP':>8s} {'Atk AP':>8s} {'Base camp':>10s} {'Atk camp':>10s} {'Δcamp':>8s}")
    print("-" * 60)
    for comp in comparisons:
        b = comp["base_run"]
        a = comp["atk_run"]
        delta = a["camp_mean"] - b["camp_mean"]
        print(f"{comp['label']:<20s} {b['mean_ap']:>8.3f} {a['mean_ap']:>8.3f} "
              f"{b['camp_mean']:>10.3f} {a['camp_mean']:>10.3f} {delta:>+7.3f}")

    return output_paths


def main():
    p = argparse.ArgumentParser(description="Plot adaptive attack PR curves.")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = p.parse_args()
    plot_attack_comparison(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
