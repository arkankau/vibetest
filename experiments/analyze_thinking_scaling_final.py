"""Final analysis of thinking scaling experiments for distributed misuse detection.

Produces:
1. Summary table (main runs + resampling)
2. Inference scaling curve (metrics vs reasoning tokens)
3. Per-case paired comparison (low vs medium, with resampling CIs)
4. Token usage distributions
5. Combined figure for the paper
"""

from __future__ import annotations

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
EFFORT_LABELS = {"low": "Low", "medium": "Med", "high": "High", "xhigh": "XHigh"}
EFFORT_COLORS = {"low": "#4878d0", "medium": "#ee854a", "high": "#6acc64", "xhigh": "#d65f5f"}
EFFORT_MARKERS = {"low": "o", "medium": "s", "high": "D", "xhigh": "^"}

OUT_DIR = Path("results")


# ── Metric helpers ──────────────────────────────────────────────────────────

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


def ap_from_row(row: dict) -> float | None:
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

    if sum(labels) == 0:
        return None

    pairs = sorted(zip(scores, labels), reverse=True)
    tp_cum = 0
    ap = 0.0
    for i, (_, l) in enumerate(pairs):
        if l == 1:
            tp_cum += 1
            ap += tp_cum / (i + 1)
    return ap / sum(labels)


def usage_from_row(row: dict) -> dict:
    u = (row.get("usage") or {}).get("usage_totals") or {}
    return {
        "total_tokens": int(u.get("total_tokens", 0)),
        "reasoning_tokens": int(u.get("reasoning_tokens", 0)),
        "output_tokens": int(u.get("output_tokens", 0)),
        "input_tokens": int(u.get("input_tokens", 0)),
    }


def coverage_from_row(row: dict) -> float | None:
    s = row.get("scoring") or {}
    v = s.get("campaign_coverage")
    return float(v) if v is not None else None


# ── Data loading ────────────────────────────────────────────────────────────

def load_all() -> tuple[list[dict], list[dict]]:
    """Returns (main_rows, resample_rows)."""
    main_files = sorted(OUT_DIR.glob("safety_dm_cyber_d6_bg100_gpt54_effort_*_AT-*.jsonl"))
    resample_files = sorted(OUT_DIR.glob("safety_dm_cyber_d6_bg100_gpt54_resample_*_AT-*.jsonl"))

    def _load(files):
        rows = []
        for f in files:
            with jsonlines.open(str(f)) as reader:
                for row in reader:
                    row["_source_file"] = f.name
                    rows.append(row)
        return rows

    return _load(main_files), _load(resample_files)


# ── Analysis ────────────────────────────────────────────────────────────────

def main():
    main_rows, resample_rows = load_all()
    print(f"Main runs: {len(main_rows)} rows from {len(set(r['_source_file'] for r in main_rows))} files")
    print(f"Resamples: {len(resample_rows)} rows from {len(set(r['_source_file'] for r in resample_rows))} files")

    # ── 1. Main run summary ─────────────────────────────────────────────────
    main_by_effort: dict[str, list[dict]] = {}
    for r in main_rows:
        e = r.get("reasoning_effort", "medium")
        main_by_effort.setdefault(e, []).append(r)

    print("\n" + "=" * 100)
    print("MAIN RUNS (bg=100×, n=5 unique cases per effort, single run)")
    print("=" * 100)
    print(f"{'Effort':<8} {'N':>3} {'AUROC':>14} {'AP':>14} {'Coverage':>14} "
          f"{'Total Tok':>14} {'Reason Tok':>14}")
    print("-" * 100)

    effort_summary = {}
    for effort in EFFORT_ORDER:
        rows = main_by_effort.get(effort, [])
        if not rows:
            continue
        aurocs = [auroc_from_row(r) for r in rows]
        aurocs = [a for a in aurocs if a is not None]
        aps = [ap_from_row(r) for r in rows]
        aps = [a for a in aps if a is not None]
        covs = [coverage_from_row(r) for r in rows]
        covs = [c for c in covs if c is not None]
        usages = [usage_from_row(r) for r in rows]
        total_tok = [u["total_tokens"] for u in usages]
        reason_tok = [u["reasoning_tokens"] for u in usages]

        effort_summary[effort] = {
            "aurocs": aurocs, "aps": aps, "coverages": covs,
            "total_tokens": total_tok, "reasoning_tokens": reason_tok,
        }
        print(f"{effort:<8} {len(rows):>3} "
              f"{np.mean(aurocs):>6.3f}±{np.std(aurocs):.3f}  "
              f"{np.mean(aps):>6.3f}±{np.std(aps):.3f}  "
              f"{np.mean(covs):>6.3f}±{np.std(covs):.3f}  "
              f"{np.mean(total_tok):>10.0f}±{np.std(total_tok):.0f}  "
              f"{np.mean(reason_tok):>10.0f}±{np.std(reason_tok):.0f}")
    print("=" * 100)

    # ── 2. Resampling analysis ──────────────────────────────────────────────
    resample_by_effort_case: dict[str, dict[str, list[dict]]] = {}
    for r in resample_rows:
        e = r.get("reasoning_effort", "medium")
        c = r.get("case_id", "?")
        resample_by_effort_case.setdefault(e, {}).setdefault(c, []).append(r)

    print("\n" + "=" * 100)
    print("RESAMPLING (bg=100×, same 5 cases, K=3 resamples each)")
    print("=" * 100)

    for effort in EFFORT_ORDER:
        if effort not in resample_by_effort_case:
            continue
        cases = resample_by_effort_case[effort]
        k = len(list(cases.values())[0])
        print(f"\n  {effort.upper()} (K={k}):")
        print(f"  {'Case':<12} {'AUROC (mean±std)':>18} {'Range':>14} "
              f"{'Tok (mean±std)':>18} {'Reason (mean±std)':>20}")
        print(f"  {'-'*86}")
        for cid in sorted(cases.keys()):
            runs = cases[cid]
            aurocs = [auroc_from_row(r) for r in runs]
            aurocs = [a for a in aurocs if a is not None]
            toks = [usage_from_row(r)["total_tokens"] for r in runs]
            rtoks = [usage_from_row(r)["reasoning_tokens"] for r in runs]
            if aurocs:
                print(f"  {cid:<12} {np.mean(aurocs):>8.3f}±{np.std(aurocs):.3f}  "
                      f"[{min(aurocs):.3f}-{max(aurocs):.3f}]  "
                      f"{np.mean(toks):>10.0f}±{np.std(toks):.0f}  "
                      f"{np.mean(rtoks):>10.0f}±{np.std(rtoks):.0f}")

    # ── 3. Paired comparison (low vs medium, per case) ──────────────────────
    print("\n" + "=" * 100)
    print("PAIRED COMPARISON: LOW vs MEDIUM (resampling means per case)")
    print("=" * 100)

    if "low" in resample_by_effort_case and "medium" in resample_by_effort_case:
        low_cases = resample_by_effort_case["low"]
        med_cases = resample_by_effort_case["medium"]
        common = sorted(set(low_cases) & set(med_cases))

        deltas = []
        print(f"  {'Case':<12} {'Low AUROC':>12} {'Med AUROC':>12} {'Δ':>8} {'Low Tok':>10} {'Med Tok':>10} {'Tok Ratio':>10}")
        print(f"  {'-'*78}")
        for cid in common:
            low_aurocs = [auroc_from_row(r) for r in low_cases[cid]]
            low_aurocs = [a for a in low_aurocs if a is not None]
            med_aurocs = [auroc_from_row(r) for r in med_cases[cid]]
            med_aurocs = [a for a in med_aurocs if a is not None]
            low_tok = np.mean([usage_from_row(r)["total_tokens"] for r in low_cases[cid]])
            med_tok = np.mean([usage_from_row(r)["total_tokens"] for r in med_cases[cid]])

            low_m = np.mean(low_aurocs)
            med_m = np.mean(med_aurocs)
            d = med_m - low_m
            deltas.append(d)
            ratio = med_tok / low_tok if low_tok > 0 else float("inf")
            print(f"  {cid:<12} {low_m:>12.3f} {med_m:>12.3f} {d:>+8.3f} {low_tok:>10.0f} {med_tok:>10.0f} {ratio:>9.1f}×")

        print(f"  {'-'*78}")
        print(f"  {'Mean Δ':<12} {'':>12} {'':>12} {np.mean(deltas):>+8.3f}")
        print(f"  {'Std Δ':<12} {'':>12} {'':>12} {np.std(deltas):>8.3f}")
        wins = sum(1 for d in deltas if d > 0)
        print(f"  Medium wins: {wins}/{len(deltas)} cases")

    # ── 4. Inference scaling curve ──────────────────────────────────────────
    # Combine main runs + resamples for a richer scatter
    all_rows = main_rows + resample_rows
    all_by_effort: dict[str, list[dict]] = {}
    for r in all_rows:
        e = r.get("reasoning_effort", "medium")
        all_by_effort.setdefault(e, []).append(r)

    # ── FIGURE 1: Main 2x2 (AUROC bars, AP bars, scaling curve, token dist) ──
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    efforts_present = [e for e in EFFORT_ORDER if e in effort_summary]

    # (0,0) AUROC bar chart with individual points
    ax = axes[0, 0]
    for i, e in enumerate(efforts_present):
        m = effort_summary[e]
        ax.bar(i, np.mean(m["aurocs"]), yerr=np.std(m["aurocs"]),
               color=EFFORT_COLORS[e], capsize=5, edgecolor="black", linewidth=0.5,
               width=0.7)
        ax.scatter([i] * len(m["aurocs"]), m["aurocs"],
                   color="black", s=18, zorder=5, alpha=0.6)
    ax.set_xticks(range(len(efforts_present)))
    ax.set_xticklabels([EFFORT_LABELS[e] for e in efforts_present])
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Case AUROC (n=5)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.5, alpha=0.5)

    # (0,1) AP bar chart
    ax = axes[0, 1]
    for i, e in enumerate(efforts_present):
        m = effort_summary[e]
        ax.bar(i, np.mean(m["aps"]), yerr=np.std(m["aps"]),
               color=EFFORT_COLORS[e], capsize=5, edgecolor="black", linewidth=0.5,
               width=0.7)
        ax.scatter([i] * len(m["aps"]), m["aps"],
                   color="black", s=18, zorder=5, alpha=0.6)
    ax.set_xticks(range(len(efforts_present)))
    ax.set_xticklabels([EFFORT_LABELS[e] for e in efforts_present])
    ax.set_ylabel("Average Precision")
    ax.set_title("Per-Case AP (n=5)")
    ax.set_ylim(0, 1.05)

    # (1,0) Inference scaling curve: AUROC vs mean reasoning tokens
    ax = axes[1, 0]
    for e in efforts_present:
        m = effort_summary[e]
        mean_rtok = np.mean(m["reasoning_tokens"])
        mean_auroc = np.mean(m["aurocs"])
        std_auroc = np.std(m["aurocs"])
        ax.errorbar(mean_rtok, mean_auroc, yerr=std_auroc,
                     fmt=EFFORT_MARKERS[e], color=EFFORT_COLORS[e],
                     markersize=10, capsize=5, label=EFFORT_LABELS[e],
                     markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    # Also scatter all individual points (main + resamples) faintly
    for e in EFFORT_ORDER:
        if e not in all_by_effort:
            continue
        for r in all_by_effort[e]:
            a = auroc_from_row(r)
            u = usage_from_row(r)
            if a is not None:
                ax.scatter(u["reasoning_tokens"], a,
                           color=EFFORT_COLORS[e], s=10, alpha=0.25, zorder=2)
    ax.set_xlabel("Reasoning Tokens (per case)")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs. Inference Compute")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    # (1,1) Token breakdown stacked bar
    ax = axes[1, 1]
    x = np.arange(len(efforts_present))
    reason_means = [np.mean(effort_summary[e]["reasoning_tokens"]) for e in efforts_present]
    total_means = [np.mean(effort_summary[e]["total_tokens"]) for e in efforts_present]
    other_means = [t - r for t, r in zip(total_means, reason_means)]
    ax.bar(x, other_means, color=[EFFORT_COLORS[e] for e in efforts_present],
           edgecolor="black", linewidth=0.5, label="I/O tokens")
    ax.bar(x, reason_means, bottom=other_means,
           color=[EFFORT_COLORS[e] for e in efforts_present],
           alpha=0.4, edgecolor="black", linewidth=0.5, hatch="//", label="Reasoning tokens")
    ax.set_xticks(x)
    ax.set_xticklabels([EFFORT_LABELS[e] for e in efforts_present])
    ax.set_ylabel("Tokens per Case")
    ax.set_title("Token Breakdown")
    ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "Inference Compute Scaling — Distributed Misuse Detection\n"
        "GPT-5.4, cyber domain, decomp=6, bg=100×",
        fontsize=12,
    )
    fig.tight_layout()
    p = OUT_DIR / "thinking_scaling_final.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {p}")

    # ── FIGURE 2: Paired resampling comparison ──────────────────────────────
    if "low" in resample_by_effort_case and "medium" in resample_by_effort_case:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        low_cases = resample_by_effort_case["low"]
        med_cases = resample_by_effort_case["medium"]
        common = sorted(set(low_cases) & set(med_cases))
        case_colors = plt.cm.Set2(np.linspace(0, 1, len(common)))

        # (a) Paired AUROC: low vs medium per case
        ax = axes[0]
        for i, cid in enumerate(common):
            low_a = [auroc_from_row(r) for r in low_cases[cid]]
            low_a = [a for a in low_a if a is not None]
            med_a = [auroc_from_row(r) for r in med_cases[cid]]
            med_a = [a for a in med_a if a is not None]
            # Draw lines connecting each resample pair
            for la in low_a:
                for ma in med_a:
                    ax.plot([0, 1], [la, ma], color=case_colors[i], alpha=0.15, lw=0.8)
            ax.scatter([0] * len(low_a), low_a, color=case_colors[i], s=25, zorder=5,
                       edgecolor="black", linewidth=0.3, label=cid if i < 5 else None)
            ax.scatter([1] * len(med_a), med_a, color=case_colors[i], s=25, zorder=5,
                       edgecolor="black", linewidth=0.3)
            # Connect means with solid line
            ax.plot([0, 1], [np.mean(low_a), np.mean(med_a)],
                    color=case_colors[i], lw=2, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low", "Medium"])
        ax.set_ylabel("AUROC")
        ax.set_title("Paired: Low vs Medium")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, loc="lower left")

        # (b) Token variance within effort (box plot)
        ax = axes[1]
        box_data = []
        box_labels = []
        box_colors = []
        for effort in ["low", "medium"]:
            if effort not in resample_by_effort_case:
                continue
            all_toks = []
            for cid, runs in resample_by_effort_case[effort].items():
                for r in runs:
                    all_toks.append(usage_from_row(r)["total_tokens"])
            box_data.append(all_toks)
            box_labels.append(EFFORT_LABELS[effort])
            box_colors.append(EFFORT_COLORS[effort])

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("Total Tokens")
        ax.set_title("Token Variance (K=3 resamples)")

        # (c) Per-case AUROC improvement (delta) with CI from resampling
        ax = axes[2]
        case_ids = []
        delta_means = []
        delta_cis = []
        for cid in common:
            low_a = [a for a in [auroc_from_row(r) for r in low_cases[cid]] if a is not None]
            med_a = [a for a in [auroc_from_row(r) for r in med_cases[cid]] if a is not None]
            # Bootstrap delta
            rng = np.random.default_rng(42)
            boot_deltas = []
            for _ in range(1000):
                l = rng.choice(low_a)
                m = rng.choice(med_a)
                boot_deltas.append(m - l)
            case_ids.append(cid)
            delta_means.append(np.mean(boot_deltas))
            delta_cis.append((np.percentile(boot_deltas, 2.5), np.percentile(boot_deltas, 97.5)))

        y = np.arange(len(case_ids))
        for i, (dm, (lo, hi)) in enumerate(zip(delta_means, delta_cis)):
            color = "#6acc64" if dm > 0 else "#d65f5f"
            ax.barh(i, dm, color=color, edgecolor="black", linewidth=0.5, height=0.6)
            ax.plot([lo, hi], [i, i], color="black", lw=1.5)
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(case_ids, fontsize=8)
        ax.set_xlabel("ΔAUROC (Medium − Low)")
        ax.set_title("Per-Case Improvement\n(bootstrap 95% CI)")

        fig.suptitle("Resampling Analysis — Low vs Medium (bg=100×, K=3)", fontsize=12)
        fig.tight_layout()
        p = OUT_DIR / "thinking_scaling_paired.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    # ── FIGURE 3: Inference scaling curve (clean, paper-style) ──────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for e in efforts_present:
        m = effort_summary[e]
        mean_rtok = np.mean(m["reasoning_tokens"])
        mean_auroc = np.mean(m["aurocs"])
        std_auroc = np.std(m["aurocs"])
        ax.errorbar(mean_rtok, mean_auroc, yerr=std_auroc,
                     fmt=EFFORT_MARKERS[e], color=EFFORT_COLORS[e],
                     markersize=12, capsize=6, label=EFFORT_LABELS[e],
                     markeredgecolor="black", markeredgewidth=0.7, zorder=5,
                     capthick=1.5)
    # Connect with line
    rtoks = [np.mean(effort_summary[e]["reasoning_tokens"]) for e in efforts_present]
    aurocs = [np.mean(effort_summary[e]["aurocs"]) for e in efforts_present]
    ax.plot(rtoks, aurocs, color="gray", ls="--", lw=1, alpha=0.5, zorder=1)

    ax.set_xlabel("Mean Reasoning Tokens per Case")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Inference Compute Scaling (GPT-5.4, bg=100×)")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    p = OUT_DIR / "thinking_scaling_curve.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # ── Save JSON summary ───────────────────────────────────────────────────
    summary = {}
    for e in efforts_present:
        m = effort_summary[e]
        summary[e] = {
            "n_cases": len(m["aurocs"]),
            "auroc_mean": float(np.mean(m["aurocs"])),
            "auroc_std": float(np.std(m["aurocs"])),
            "ap_mean": float(np.mean(m["aps"])),
            "ap_std": float(np.std(m["aps"])),
            "coverage_mean": float(np.mean(m["coverages"])),
            "total_tokens_mean": float(np.mean(m["total_tokens"])),
            "reasoning_tokens_mean": float(np.mean(m["reasoning_tokens"])),
        }
    (OUT_DIR / "thinking_scaling_final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved: {OUT_DIR / 'thinking_scaling_final_summary.json'}")


if __name__ == "__main__":
    main()
