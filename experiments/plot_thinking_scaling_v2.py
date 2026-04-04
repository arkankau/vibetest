"""Improved inference-compute scaling figures for distributed misuse detection.

Key insight: between-case variance dominates (~10× larger than within-effort),
so absolute-AUROC plots have wide CIs that obscure the scaling signal.

This script produces:
1. Paired delta scaling curve: ΔAUROC (vs low baseline) per effort level,
   with bootstrap CIs that remove case-difficulty noise.
2. Slopegraph: each case's AUROC trajectory across effort levels,
   showing that case difficulty dominates but there IS a consistent shift.
3. Combined 2-panel figure for the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonlines
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Paper style (matching DM/SHADE-Arena figures) ──────────────────────────

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
# Colorblind-safe palette (same as DM paper figures)
EFFORT_COLORS = {
    "low": "#0072B2",
    "medium": "#D55E00",
    "high": "#009E73",
    "xhigh": "#CC79A7",
}

OUT_DIR = Path("results")
FIG_DIR = Path("figures")


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
        "output_tokens": int(u.get("output_tokens", 0)),
        "input_tokens": int(u.get("input_tokens", 0)),
    }


# ── Data loading ───────────────────────────────────────────────────────────

def load_main_runs() -> dict[str, list[dict]]:
    """Load main effort runs, keyed by effort level."""
    by_effort: dict[str, list[dict]] = {}
    for effort in EFFORT_ORDER:
        pattern = f"safety_dm_cyber_d6_bg100_gpt54_effort_{effort}_AT-*.jsonl"
        files = sorted(OUT_DIR.glob(pattern))
        rows = []
        for f in files:
            with jsonlines.open(str(f)) as reader:
                for row in reader:
                    row["_source_file"] = f.name
                    rows.append(row)
        if rows:
            by_effort[effort] = rows
    return by_effort


def load_resample_runs() -> dict[str, dict[str, list[dict]]]:
    """Load resampling runs, keyed by effort -> case_id -> [rows]."""
    by_effort_case: dict[str, dict[str, list[dict]]] = {}
    pattern = "safety_dm_cyber_d6_bg100_gpt54_resample_*_AT-*.jsonl"
    for f in sorted(OUT_DIR.glob(pattern)):
        with jsonlines.open(str(f)) as reader:
            for row in reader:
                e = row.get("reasoning_effort", "medium")
                c = row.get("case_id", "?")
                by_effort_case.setdefault(e, {}).setdefault(c, []).append(row)
    return by_effort_case


# ── Build paired data structure ────────────────────────────────────────────

def build_paired_data(by_effort: dict[str, list[dict]]) -> dict:
    """Build per-case, per-effort AUROC and token data.

    Returns dict with:
      case_ids: list of case IDs
      efforts: list of effort levels present
      auroc[effort][case_id]: float
      reasoning_tokens[effort][case_id]: int
      total_tokens[effort][case_id]: int
    """
    # Get case_ids from low (baseline)
    if "low" not in by_effort:
        raise ValueError("No 'low' effort data for baseline")

    case_ids = sorted(set(r.get("case_id", "?") for r in by_effort["low"]))
    efforts = [e for e in EFFORT_ORDER if e in by_effort]

    auroc = {}
    reasoning_tokens = {}
    total_tokens = {}

    for e in efforts:
        auroc[e] = {}
        reasoning_tokens[e] = {}
        total_tokens[e] = {}
        for row in by_effort[e]:
            cid = row.get("case_id", "?")
            a = auroc_from_row(row)
            u = usage_from_row(row)
            if a is not None:
                auroc[e][cid] = a
                reasoning_tokens[e][cid] = u["reasoning_tokens"]
                total_tokens[e][cid] = u["total_tokens"]

    return {
        "case_ids": case_ids,
        "efforts": efforts,
        "auroc": auroc,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def bootstrap_paired_delta(
    data: dict,
    baseline: str = "low",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Bootstrap paired ΔAUROC and token ratios relative to baseline.

    Returns per-effort:
      delta_mean, delta_ci_lo, delta_ci_hi (AUROC)
      token_mean, token_ci_lo, token_ci_hi (reasoning tokens)
      abs_auroc_mean, abs_auroc_ci_lo, abs_auroc_ci_hi
    """
    rng = np.random.default_rng(seed)
    results = {}

    case_ids = data["case_ids"]
    n_cases = len(case_ids)

    for e in data["efforts"]:
        # Get paired values (only cases present in both baseline and this effort)
        paired_cases = [c for c in case_ids
                        if c in data["auroc"].get(baseline, {})
                        and c in data["auroc"].get(e, {})]
        if not paired_cases:
            continue

        base_aurocs = np.array([data["auroc"][baseline][c] for c in paired_cases])
        this_aurocs = np.array([data["auroc"][e][c] for c in paired_cases])
        this_rtokens = np.array([data["reasoning_tokens"][e][c] for c in paired_cases])
        deltas = this_aurocs - base_aurocs

        # Bootstrap over cases (paired)
        boot_deltas = []
        boot_abs_aurocs = []
        boot_rtokens = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(paired_cases), size=len(paired_cases))
            boot_deltas.append(np.mean(deltas[idx]))
            boot_abs_aurocs.append(np.mean(this_aurocs[idx]))
            boot_rtokens.append(np.mean(this_rtokens[idx]))

        boot_deltas = np.array(boot_deltas)
        boot_abs_aurocs = np.array(boot_abs_aurocs)
        boot_rtokens = np.array(boot_rtokens)

        results[e] = {
            "delta_mean": float(np.mean(deltas)),
            "delta_ci_lo": float(np.percentile(boot_deltas, 2.5)),
            "delta_ci_hi": float(np.percentile(boot_deltas, 97.5)),
            "abs_auroc_mean": float(np.mean(this_aurocs)),
            "abs_auroc_ci_lo": float(np.percentile(boot_abs_aurocs, 2.5)),
            "abs_auroc_ci_hi": float(np.percentile(boot_abs_aurocs, 97.5)),
            "token_mean": float(np.mean(this_rtokens)),
            "token_ci_lo": float(np.percentile(boot_rtokens, 2.5)),
            "token_ci_hi": float(np.percentile(boot_rtokens, 97.5)),
            "n_cases": len(paired_cases),
            "per_case_deltas": {c: float(d) for c, d in zip(paired_cases, deltas)},
        }

    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_combined(data: dict, boot: dict, savedir: Path = FIG_DIR):
    """Two-panel figure:
    (a) Absolute AUROC vs reasoning tokens — paired bootstrap CIs
    (b) ΔAUROC (vs Low baseline) vs reasoning tokens — tighter CIs
    """
    savedir.mkdir(exist_ok=True)
    efforts = data["efforts"]

    with mpl.rc_context(RCPARAMS):
        fig, (ax_abs, ax_delta) = plt.subplots(
            1, 2, figsize=(6.5, 2.8), gridspec_kw={"wspace": 0.38}
        )

        # ── Panel (a): Absolute AUROC vs reasoning tokens ──────────────
        for e in efforts:
            if e not in boot:
                continue
            b = boot[e]
            ax_abs.errorbar(
                b["token_mean"], b["abs_auroc_mean"],
                xerr=[[b["token_mean"] - b["token_ci_lo"]],
                      [b["token_ci_hi"] - b["token_mean"]]],
                yerr=[[b["abs_auroc_mean"] - b["abs_auroc_ci_lo"]],
                      [b["abs_auroc_ci_hi"] - b["abs_auroc_mean"]]],
                fmt="o", color=EFFORT_COLORS[e], markersize=7,
                capsize=3, capthick=1.2, elinewidth=1.2,
                markeredgecolor="black", markeredgewidth=0.4,
                label=EFFORT_LABELS[e], zorder=5,
            )
            # Scatter individual cases faintly
            for cid in data["case_ids"]:
                if cid in data["auroc"].get(e, {}) and cid in data["reasoning_tokens"].get(e, {}):
                    ax_abs.scatter(
                        data["reasoning_tokens"][e][cid],
                        data["auroc"][e][cid],
                        color=EFFORT_COLORS[e], s=12, alpha=0.25, zorder=2,
                        edgecolors="none",
                    )

        # Connect means with dashed line
        xs = [boot[e]["token_mean"] for e in efforts if e in boot]
        ys = [boot[e]["abs_auroc_mean"] for e in efforts if e in boot]
        ax_abs.plot(xs, ys, color="grey", ls="--", lw=0.8, alpha=0.5, zorder=1)

        ax_abs.set_xscale("log")
        ax_abs.set_xlabel("Reasoning tokens (per case)", fontsize=LABEL_SIZE)
        ax_abs.set_ylabel("AUROC", fontsize=LABEL_SIZE)
        ax_abs.set_title("(a)  Absolute AUROC", fontsize=TITLE_SIZE, fontweight="bold")
        ax_abs.set_ylim(0.25, 1.05)
        ax_abs.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax_abs.tick_params(labelsize=TICK_SIZE)
        ax_abs.text(0.03, 0.03, "Bars: bootstrap 95% CI",
                    transform=ax_abs.transAxes, fontsize=5.5, color="grey",
                    ha="left", va="bottom")
        ax_abs.legend(
            fontsize=LEGEND_SIZE, frameon=False, loc="lower right",
            handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
        )

        # ── Panel (b): ΔAUROC vs reasoning tokens ─────────────────────
        for e in efforts:
            if e not in boot:
                continue
            b = boot[e]
            ax_delta.errorbar(
                b["token_mean"], b["delta_mean"],
                xerr=[[b["token_mean"] - b["token_ci_lo"]],
                      [b["token_ci_hi"] - b["token_mean"]]],
                yerr=[[b["delta_mean"] - b["delta_ci_lo"]],
                      [b["delta_ci_hi"] - b["delta_mean"]]],
                fmt="o", color=EFFORT_COLORS[e], markersize=7,
                capsize=3, capthick=1.2, elinewidth=1.2,
                markeredgecolor="black", markeredgewidth=0.4,
                label=EFFORT_LABELS[e], zorder=5,
            )

        # Connect means
        xs = [boot[e]["token_mean"] for e in efforts if e in boot]
        ys = [boot[e]["delta_mean"] for e in efforts if e in boot]
        ax_delta.plot(xs, ys, color="grey", ls="--", lw=0.8, alpha=0.5, zorder=1)

        ax_delta.set_xscale("log")
        ax_delta.axhline(0.0, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax_delta.set_xlabel("Reasoning tokens (per case)", fontsize=LABEL_SIZE)
        ax_delta.set_ylabel("\u0394AUROC (vs Low)", fontsize=LABEL_SIZE)
        ax_delta.set_title("(b)  Paired improvement", fontsize=TITLE_SIZE, fontweight="bold")
        ax_delta.tick_params(labelsize=TICK_SIZE)
        ax_delta.text(0.03, 0.97, "Bars: paired bootstrap 95% CI",
                      transform=ax_delta.transAxes, fontsize=5.5, color="grey",
                      ha="left", va="top")
        ax_delta.legend(
            fontsize=LEGEND_SIZE, frameon=False, loc="center right",
            handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
        )

        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = savedir / f"thinking_scaling_paired.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=300)
            print(f"Saved: {p}")
        plt.close(fig)


def bootstrap_mean_ci_per_effort(
    data: dict, n_boot: int = 5000, seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap 95% CI of the mean AUROC at each effort level.

    Returns {effort: (mean, ci_lo, ci_hi)}.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for e in data["efforts"]:
        vals = np.array([data["auroc"][e][c] for c in data["case_ids"]
                         if c in data["auroc"].get(e, {})])
        if len(vals) == 0:
            continue
        boot_means = np.array([
            np.mean(rng.choice(vals, size=len(vals), replace=True))
            for _ in range(n_boot)
        ])
        result[e] = (
            float(np.mean(vals)),
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        )
    return result


def plot_slopegraph(data: dict, boot: dict, savedir: Path = FIG_DIR):
    """Slopegraph: each case's AUROC trajectory across effort levels,
    with bootstrap 95% CI band on the mean trajectory."""
    savedir.mkdir(exist_ok=True)
    efforts = data["efforts"]
    case_ids = data["case_ids"]

    mean_cis = bootstrap_mean_ci_per_effort(data)

    # Case colors from a qualitative palette
    cmap = plt.cm.Set2
    case_colors = {cid: cmap(i / max(len(case_ids) - 1, 1))
                   for i, cid in enumerate(case_ids)}

    with mpl.rc_context(RCPARAMS):
        fig, ax = plt.subplots(figsize=(3.8, 3.0))

        x_positions = list(range(len(efforts)))

        for cid in case_ids:
            xs, ys = [], []
            for i, e in enumerate(efforts):
                if cid in data["auroc"].get(e, {}):
                    xs.append(i)
                    ys.append(data["auroc"][e][cid])
            if xs:
                ax.plot(xs, ys, color=case_colors[cid], lw=1.2, alpha=0.6, zorder=2)
                ax.scatter(xs, ys, color=case_colors[cid], s=20, zorder=3,
                           edgecolors="black", linewidth=0.3,
                           label=cid.replace("case_", "Case "))

        # Bootstrap CI band on mean
        mean_ys = [mean_cis[e][0] for e in efforts if e in mean_cis]
        ci_lo = [mean_cis[e][1] for e in efforts if e in mean_cis]
        ci_hi = [mean_cis[e][2] for e in efforts if e in mean_cis]
        xs_band = [i for i, e in enumerate(efforts) if e in mean_cis]
        ax.fill_between(xs_band, ci_lo, ci_hi, color="black", alpha=0.12, zorder=3,
                        label="95% CI of mean")
        ax.plot(xs_band, mean_ys, color="black", lw=2.5, ls="-", alpha=0.85,
                zorder=4, label="Mean")
        ax.scatter(xs_band, mean_ys, color="black", s=40, zorder=5,
                   edgecolors="black", linewidth=0.5)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([EFFORT_LABELS[e] for e in efforts], fontsize=TICK_SIZE)
        ax.set_ylabel("AUROC", fontsize=LABEL_SIZE)
        ax.set_xlabel("Reasoning effort", fontsize=LABEL_SIZE)
        ax.set_title("Per-case AUROC trajectories", fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_ylim(0.25, 1.05)
        ax.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(
            fontsize=LEGEND_SIZE - 0.5, frameon=False, loc="lower left",
            handletextpad=0.3, borderpad=0.2, labelspacing=0.2,
            ncol=2,
        )

        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = savedir / f"thinking_scaling_slopegraph.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=300)
            print(f"Saved: {p}")
        plt.close(fig)


def plot_three_panel(data: dict, boot: dict, savedir: Path = FIG_DIR):
    """Three-panel figure combining absolute, delta, and slopegraph."""
    savedir.mkdir(exist_ok=True)
    efforts = data["efforts"]
    case_ids = data["case_ids"]

    cmap = plt.cm.Set2
    case_colors = {cid: cmap(i / max(len(case_ids) - 1, 1))
                   for i, cid in enumerate(case_ids)}

    with mpl.rc_context(RCPARAMS):
        fig, (ax_abs, ax_delta, ax_slope) = plt.subplots(
            1, 3, figsize=(9.5, 2.8),
            gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1, 0.9]},
        )

        # ── (a) Absolute AUROC ────────────────────────────────────────
        for e in efforts:
            if e not in boot:
                continue
            b = boot[e]
            ax_abs.errorbar(
                b["token_mean"], b["abs_auroc_mean"],
                xerr=[[b["token_mean"] - b["token_ci_lo"]],
                      [b["token_ci_hi"] - b["token_mean"]]],
                yerr=[[b["abs_auroc_mean"] - b["abs_auroc_ci_lo"]],
                      [b["abs_auroc_ci_hi"] - b["abs_auroc_mean"]]],
                fmt="o", color=EFFORT_COLORS[e], markersize=6,
                capsize=2.5, capthick=1.0, elinewidth=1.0,
                markeredgecolor="black", markeredgewidth=0.4,
                label=EFFORT_LABELS[e], zorder=5,
            )
        xs = [boot[e]["token_mean"] for e in efforts if e in boot]
        ys = [boot[e]["abs_auroc_mean"] for e in efforts if e in boot]
        ax_abs.plot(xs, ys, color="grey", ls="--", lw=0.8, alpha=0.5, zorder=1)
        ax_abs.set_xscale("log")
        ax_abs.set_xlabel("Reasoning tokens", fontsize=LABEL_SIZE)
        ax_abs.set_ylabel("AUROC", fontsize=LABEL_SIZE)
        ax_abs.set_title("(a)  Absolute", fontsize=TITLE_SIZE, fontweight="bold")
        ax_abs.set_ylim(0.25, 1.05)
        ax_abs.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax_abs.tick_params(labelsize=TICK_SIZE)
        ax_abs.legend(fontsize=LEGEND_SIZE, frameon=False, loc="lower right",
                      handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

        # ── (b) ΔAUROC ────────────────────────────────────────────────
        for e in efforts:
            if e not in boot:
                continue
            b = boot[e]
            ax_delta.errorbar(
                b["token_mean"], b["delta_mean"],
                xerr=[[b["token_mean"] - b["token_ci_lo"]],
                      [b["token_ci_hi"] - b["token_mean"]]],
                yerr=[[b["delta_mean"] - b["delta_ci_lo"]],
                      [b["delta_ci_hi"] - b["delta_mean"]]],
                fmt="o", color=EFFORT_COLORS[e], markersize=6,
                capsize=2.5, capthick=1.0, elinewidth=1.0,
                markeredgecolor="black", markeredgewidth=0.4,
                label=EFFORT_LABELS[e], zorder=5,
            )
        xs = [boot[e]["token_mean"] for e in efforts if e in boot]
        ys = [boot[e]["delta_mean"] for e in efforts if e in boot]
        ax_delta.plot(xs, ys, color="grey", ls="--", lw=0.8, alpha=0.5, zorder=1)
        ax_delta.set_xscale("log")
        ax_delta.axhline(0.0, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax_delta.set_xlabel("Reasoning tokens", fontsize=LABEL_SIZE)
        ax_delta.set_ylabel("\u0394AUROC (vs Low)", fontsize=LABEL_SIZE)
        ax_delta.set_title("(b)  Paired \u0394", fontsize=TITLE_SIZE, fontweight="bold")
        ax_delta.tick_params(labelsize=TICK_SIZE)
        ax_delta.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper left",
                        handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

        # ── (c) Slopegraph with bootstrap CI band ─────────────────────
        mean_cis = bootstrap_mean_ci_per_effort(data)
        x_positions = list(range(len(efforts)))
        for cid in case_ids:
            xs, ys = [], []
            for i, e in enumerate(efforts):
                if cid in data["auroc"].get(e, {}):
                    xs.append(i)
                    ys.append(data["auroc"][e][cid])
            if xs:
                ax_slope.plot(xs, ys, color=case_colors[cid], lw=1.0, alpha=0.55, zorder=2)
                ax_slope.scatter(xs, ys, color=case_colors[cid], s=14, zorder=3,
                                 edgecolors="black", linewidth=0.2)

        # Bootstrap CI band on mean
        mean_ys = [mean_cis[e][0] for e in efforts if e in mean_cis]
        ci_lo = [mean_cis[e][1] for e in efforts if e in mean_cis]
        ci_hi = [mean_cis[e][2] for e in efforts if e in mean_cis]
        xs_band = [i for i, e in enumerate(efforts) if e in mean_cis]
        ax_slope.fill_between(xs_band, ci_lo, ci_hi, color="black", alpha=0.12, zorder=3)
        ax_slope.plot(xs_band, mean_ys, color="black", lw=2.0, alpha=0.85, zorder=4)
        ax_slope.scatter(xs_band, mean_ys, color="black", s=30, zorder=5,
                         edgecolors="black", linewidth=0.5)

        ax_slope.set_xticks(x_positions)
        ax_slope.set_xticklabels([EFFORT_LABELS[e] for e in efforts], fontsize=TICK_SIZE)
        ax_slope.set_ylabel("AUROC", fontsize=LABEL_SIZE)
        ax_slope.set_title("(c)  Per-case trajectories", fontsize=TITLE_SIZE, fontweight="bold")
        ax_slope.set_ylim(0.25, 1.05)
        ax_slope.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.4)
        ax_slope.tick_params(labelsize=TICK_SIZE)

        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = savedir / f"thinking_scaling_v2.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=300)
            print(f"Saved: {p}")
        plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    by_effort = load_main_runs()
    print(f"Loaded main runs: {', '.join(f'{e}={len(rows)}' for e, rows in by_effort.items())}")

    data = build_paired_data(by_effort)
    boot = bootstrap_paired_delta(data)

    # Print summary
    print(f"\nPaired bootstrap results (vs Low baseline, n_boot=5000):")
    print(f"{'Effort':<8} {'AUROC':>8} {'95% CI':>16} {'ΔAUROC':>8} {'95% CI':>16} "
          f"{'Reason Tok':>12} {'95% CI':>20}")
    print("-" * 100)
    for e in EFFORT_ORDER:
        if e not in boot:
            continue
        b = boot[e]
        print(f"{e:<8} {b['abs_auroc_mean']:>8.3f} [{b['abs_auroc_ci_lo']:.3f}, {b['abs_auroc_ci_hi']:.3f}]"
              f"   {b['delta_mean']:>+8.3f} [{b['delta_ci_lo']:.3f}, {b['delta_ci_hi']:.3f}]"
              f"   {b['token_mean']:>10.0f}  [{b['token_ci_lo']:.0f}, {b['token_ci_hi']:.0f}]")

    # Plots
    plot_combined(data, boot)
    plot_slopegraph(data, boot)
    plot_three_panel(data, boot)

    # Save bootstrap results as JSON
    out = {}
    for e, b in boot.items():
        out[e] = {k: v for k, v in b.items() if k != "per_case_deltas"}
        out[e]["per_case_deltas"] = b["per_case_deltas"]
    (OUT_DIR / "thinking_scaling_v2_bootstrap.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(f"\nSaved: {OUT_DIR / 'thinking_scaling_v2_bootstrap.json'}")


if __name__ == "__main__":
    main()
