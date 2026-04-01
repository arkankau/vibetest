"""Generate paper figures for SHADE-Arena sabotage detection experiments.

Produces:
  figures/shade_arena_paper_pr_curves.{png,pdf}

Requires: results/shade_arena_tpc1_{monitor,eae,agent_full}.jsonl

Usage:
    python scripts/plot_shade_arena.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_pooled_scores(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(l) for l in open(path)]
    scores, labels = [], []
    for r in rows:
        ts = (r.get("scoring") or {}).get("trace_scores", {})
        gt_files = set(r.get("ground_truth", {}).get("suspicious_trace_files", []))
        for k, v in ts.items():
            scores.append(v)
            labels.append(1 if k in gt_files else 0)
    return np.array(scores), np.array(labels)


def pooled_pr_with_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    recall_grid: list[float],
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[list[float], list[float], list[float], float]:
    rng = np.random.RandomState(seed)
    n = len(scores)

    def compute_pr(s, l):
        thresholds = np.sort(np.unique(s))[::-1]
        recs, precs = [0.0], [1.0]
        for t in thresholds:
            pred = s >= t
            tp = np.sum(pred & (l == 1))
            fp = np.sum(pred & (l == 0))
            fn = np.sum(~pred & (l == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recs.append(rec)
            precs.append(prec)
        return np.array(recs), np.array(precs)

    def interp(recs, precs, grid):
        return [float(np.max(precs[recs >= r])) if (recs >= r).any() else 0.0 for r in grid]

    recs, precs = compute_pr(scores, labels)
    mean_curve = interp(recs, precs, recall_grid)
    ap = float(np.sum([precs[i] * (recs[i] - recs[i - 1]) for i in range(1, len(recs)) if recs[i] > recs[i - 1]]))

    boot_curves = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bs, bl = scores[idx], labels[idx]
        if bl.sum() == 0 or bl.sum() == n:
            continue
        r, p = compute_pr(bs, bl)
        boot_curves.append(interp(r, p, recall_grid))
    if boot_curves:
        bc = np.array(boot_curves)
        lower = np.percentile(bc, 2.5, axis=0).tolist()
        upper = np.percentile(bc, 97.5, axis=0).tolist()
    else:
        lower = mean_curve[:]
        upper = mean_curve[:]
    return mean_curve, lower, upper, ap


def main() -> None:
    recall_grid = [i / 40.0 for i in range(41)]
    method_order = ["Meerkat", "Monitor", "EaE"]
    method_colors = {"Meerkat": "#D55E00", "Monitor": "#CC79A7", "EaE": "#0072B2"}
    method_linestyles = {"Meerkat": "-", "Monitor": "-.", "EaE": ":"}
    method_files = {"Meerkat": "agent_full", "Monitor": "monitor", "EaE": "eae"}

    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    ):
        fig = plt.figure(figsize=(5.0, 2.4))
        ax = fig.add_subplot(111)
        legend_labels = []
        for method in method_order:
            path = f"results/shade_arena_tpc1_{method_files[method]}.jsonl"
            try:
                scores, labels = load_pooled_scores(path)
            except FileNotFoundError:
                print(f"Skipping {method}: {path} not found")
                continue
            if len(scores) == 0:
                continue
            mean_curve, lower, upper, ap = pooled_pr_with_bootstrap(scores, labels, recall_grid)
            ax.fill_between(
                recall_grid, lower, upper,
                color=method_colors[method], alpha=0.14, linewidth=0.0, zorder=1,
            )
            ax.plot(
                recall_grid, mean_curve,
                color=method_colors[method], linestyle=method_linestyles[method],
                linewidth=2.2, zorder=2,
            )
            legend_labels.append(f"{method} (AP={ap:.2f})")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.25)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=9.0)
        ax.set_xlabel("Recall", fontsize=10.5)
        ax.set_ylabel("Precision", fontsize=10.5)
        handles = [
            Line2D([0], [0], color=method_colors[m], linestyle=method_linestyles[m], linewidth=2.2)
            for m in method_order
        ]
        ax.legend(
            handles, legend_labels, loc="lower left", frameon=False,
            fontsize=8.0, handlelength=1.9, borderaxespad=0.3, labelspacing=0.25,
        )
        fig.subplots_adjust(top=0.96, left=0.12, right=0.98, bottom=0.17)

        Path("figures").mkdir(exist_ok=True)
        fig.savefig("figures/shade_arena_paper_pr_curves.png", dpi=300)
        fig.savefig("figures/shade_arena_paper_pr_curves.pdf")
        print("Saved figures/shade_arena_paper_pr_curves.{png,pdf}")
        plt.close(fig)


if __name__ == "__main__":
    main()
