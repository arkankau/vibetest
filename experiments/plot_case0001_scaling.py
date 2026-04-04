"""Plot inference-compute scaling for case_0001 (hardest case).

Uses all available GPT-5.4 runs of case_0001 at bg=100×, across effort levels.
Bootstrap CIs are computed over the repeated runs at each effort level.
"""

from __future__ import annotations

from pathlib import Path

import jsonlines
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Paper style ────────────────────────────────────────────────────────────

RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "figure.dpi": 300,
}

TITLE_SIZE = 9.0
LABEL_SIZE = 8.5
TICK_SIZE = 7.0
LEGEND_SIZE = 7.5

EFFORT_ORDER = ["low", "medium", "high", "xhigh"]
EFFORT_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "XHigh"}
EFFORT_COLORS = {
    "low": "#0072B2",
    "medium": "#D55E00",
    "high": "#009E73",
    "xhigh": "#CC79A7",
}

FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")


# ── Metric helpers ─────────────────────────────────────────────────────────

def _normalize(p: str) -> str:
    p = p.strip()
    for pfx in ("/workspace/repo/", "/workspace/", "workspace/repo/", "workspace/"):
        if p.startswith(pfx):
            return p[len(pfx):]
    return p


def auroc_from_row(row: dict) -> float | None:
    scoring = row.get("scoring") or {}
    ts = scoring.get("trace_scores") or {}
    if not ts:
        ts = ((row.get("tests") or [{}])[0].get("metadata") or {}).get("trace_scores") or {}
    if not ts:
        return None
    gt = row.get("ground_truth") or {}
    gt_pos = set()
    for k in ("harmful_trace_files", "hacked_trace_files", "suspicious_trace_files"):
        if k in gt:
            gt_pos.update(gt[k])
    if not gt_pos:
        return None
    scores, labels = [], []
    for tp, sc in ts.items():
        scores.append(float(sc))
        labels.append(1 if _normalize(tp) in gt_pos else 0)
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = tp_prev = fp_prev = 0
    auc = 0.0
    prev_s = None
    for s, l in pairs:
        if s != prev_s and prev_s is not None:
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            tp_prev, fp_prev = tp, fp
        tp += l
        fp += 1 - l
        prev_s = s
    auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
    return auc / (n_pos * n_neg)


def usage_from_row(row: dict) -> dict:
    u = (row.get("usage") or {}).get("usage_totals") or {}
    return {
        "total_tokens": int(u.get("total_tokens", 0)),
        "reasoning_tokens": int(u.get("reasoning_tokens", 0)),
    }


# ── Data loading ───────────────────────────────────────────────────────────

def load_case0001() -> dict[str, list[dict]]:
    """Load all GPT-5.4 bg=100× runs for case_0001, grouped by effort."""
    files = sorted(RESULTS_DIR.glob("safety_dm_cyber_d6_bg100*gpt54*AT-gpt-5.4*.jsonl"))
    by_effort: dict[str, list[dict]] = {}

    for f in files:
        with jsonlines.open(str(f)) as reader:
            for row in reader:
                if row.get("case_id") != "case_0001":
                    continue
                effort = row.get("reasoning_effort", "medium")
                a = auroc_from_row(row)
                u = usage_from_row(row)
                if a is not None and u["total_tokens"] > 100:
                    by_effort.setdefault(effort, []).append({
                        "auroc": a,
                        "reasoning_tokens": u["reasoning_tokens"],
                        "total_tokens": u["total_tokens"],
                        "source": f.name,
                    })
    return by_effort


# ── Bootstrap ──────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi)."""
    arr = np.array(values)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(arr)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ── Plotting ───────────────────────────────────────────────────────────────

def main():
    by_effort = load_case0001()

    for e in EFFORT_ORDER:
        runs = by_effort.get(e, [])
        print(f"{e}: {len(runs)} runs")
        for r in runs:
            print(f"  AUROC={r['auroc']:.3f}  reason_tok={r['reasoning_tokens']:>6,}")

    FIG_DIR.mkdir(exist_ok=True)
    efforts = [e for e in EFFORT_ORDER if e in by_effort]

    with mpl.rc_context(RCPARAMS):
        fig, (ax_abs, ax_scatter) = plt.subplots(
            1, 2, figsize=(6.5, 2.8), gridspec_kw={"wspace": 0.38}
        )

        # ── Panel (a): AUROC vs reasoning tokens with bootstrap CIs ───
        for e in efforts:
            runs = by_effort[e]
            aurocs = [r["auroc"] for r in runs]
            rtokens = [r["reasoning_tokens"] for r in runs]

            auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(aurocs)
            rtok_mean, rtok_lo, rtok_hi = bootstrap_ci(rtokens)

            ax_abs.errorbar(
                rtok_mean, auroc_mean,
                xerr=[[rtok_mean - rtok_lo], [rtok_hi - rtok_mean]],
                yerr=[[auroc_mean - auroc_lo], [auroc_hi - auroc_mean]],
                fmt="o", color=EFFORT_COLORS[e], markersize=7,
                capsize=3, capthick=1.2, elinewidth=1.2,
                markeredgecolor="black", markeredgewidth=0.4,
                label=f"{EFFORT_LABELS[e]} (n={len(runs)})", zorder=5,
            )

        # Connect means
        xs = []
        ys = []
        for e in efforts:
            runs = by_effort[e]
            xs.append(np.mean([r["reasoning_tokens"] for r in runs]))
            ys.append(np.mean([r["auroc"] for r in runs]))
        ax_abs.plot(xs, ys, color="grey", ls="--", lw=0.8, alpha=0.5, zorder=1)

        ax_abs.set_xscale("log")
        ax_abs.set_xlabel("Reasoning tokens", fontsize=LABEL_SIZE)
        ax_abs.set_ylabel("AUROC", fontsize=LABEL_SIZE)
        ax_abs.set_title("(a)  case_0001: bootstrap 95% CI of mean",
                         fontsize=TITLE_SIZE, fontweight="bold")
        ax_abs.set_ylim(0.15, 1.0)
        ax_abs.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax_abs.tick_params(labelsize=TICK_SIZE)
        ax_abs.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper left",
                      handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

        # ── Panel (b): All individual runs as scatter ──────────────────
        for e in efforts:
            runs = by_effort[e]
            for r in runs:
                ax_scatter.scatter(
                    r["reasoning_tokens"], r["auroc"],
                    color=EFFORT_COLORS[e], s=30, zorder=3,
                    edgecolors="black", linewidth=0.3, alpha=0.8,
                )
            # Add invisible point for legend
            ax_scatter.scatter([], [], color=EFFORT_COLORS[e], s=30,
                               edgecolors="black", linewidth=0.3,
                               label=f"{EFFORT_LABELS[e]} (n={len(runs)})")

        ax_scatter.set_xscale("log")
        ax_scatter.set_xlabel("Reasoning tokens", fontsize=LABEL_SIZE)
        ax_scatter.set_ylabel("AUROC", fontsize=LABEL_SIZE)
        ax_scatter.set_title("(b)  case_0001: all individual runs",
                             fontsize=TITLE_SIZE, fontweight="bold")
        ax_scatter.set_ylim(0.15, 1.0)
        ax_scatter.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax_scatter.tick_params(labelsize=TICK_SIZE)
        ax_scatter.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper left",
                          handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = FIG_DIR / f"thinking_scaling_case0001.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=300)
            print(f"Saved: {p}")
        plt.close(fig)

    # ── Print summary ──────────────────────────────────────────────────
    print("\nBootstrap summary:")
    print(f"{'Effort':<8} {'N':>3} {'AUROC mean':>10} {'95% CI':>18} "
          f"{'Reason tok mean':>16} {'95% CI':>22}")
    print("-" * 85)
    for e in efforts:
        runs = by_effort[e]
        aurocs = [r["auroc"] for r in runs]
        rtokens = [r["reasoning_tokens"] for r in runs]
        a_m, a_lo, a_hi = bootstrap_ci(aurocs)
        t_m, t_lo, t_hi = bootstrap_ci(rtokens)
        print(f"{e:<8} {len(runs):>3} {a_m:>10.3f} [{a_lo:.3f}, {a_hi:.3f}]"
              f"   {t_m:>12.0f}   [{t_lo:.0f}, {t_hi:.0f}]")


if __name__ == "__main__":
    main()
