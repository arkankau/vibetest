"""Regenerate the paper figures in a camera-ready publication style.

Design choices (grounded in standard LaTeX/matplotlib publication guidance):
- Serif fonts (Times New Roman + STIX math) so figures match the ACL body text.
  We do NOT use text.usetex, so this runs without a local TeX toolchain.
- Figures are sized to the *actual* ACL column / text width in inches, so a
  \\includegraphics[width=\\linewidth]{...} places them 1:1 and the in-figure
  font really is ~8pt in the compiled paper.
- Okabe-Ito colourblind-safe palette. The method (VibeTest) uses the three vivid
  hues; reviewer baselines are muted so the method visually leads.
- Top/right spines removed, thin axes, light grid, small markers, no chartjunk.
- Vector PDF with TrueType-embedded fonts (pdf.fonttype=42) to avoid Type-3
  fonts, which some venues reject. A PNG is emitted alongside for previews.

Outputs overwrite figures/*.{pdf,png}.
"""
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "supporting_docs")
FIGS = os.path.join(HERE, "figures")

# ACL two-column geometry (a4paper, margin 2.5cm, columnsep 0.6cm):
# textwidth = 21 - 5 = 16cm; columnwidth = (16 - 0.6)/2 = 7.7cm.
INCH = 1 / 2.54
COL = 7.7 * INCH        # ~3.03 in, single column
TEXT = 16.0 * INCH      # ~6.30 in, full width (figure*)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.30,
    "grid.color": "0.6",
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.0,
    "legend.handletextpad": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito palette. VibeTest = vivid trio; reviewers = muted; Codex = black.
C_BLUE, C_ORANGE, C_GREEN = "#0072B2", "#E69F00", "#009E73"
STYLE = {
    "VibeTest static":        (C_BLUE,   "o", "VibeTest"),
    "VibeTest static + ex10": (C_ORANGE, "s", "VibeTest+ex10"),
    "VibeTest static + ex20": (C_GREEN,  "^", "VibeTest+ex20"),
    "Reviewer mode 0":        ("#B47EB3", "P", "Reviewer 0"),
    "Reviewer mode 1":        ("#D55E00", "X", "Reviewer 1"),
    "Reviewer mode 2":        ("#8C8C8C", "v", "Reviewer 2"),
}


def _key(series):
    return series.split(" (")[0]


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)


def regen_selective_f1():
    df = pd.read_csv(os.path.join(DOCS, "paper_synthetic_selective_f1_coverage.csv"))
    df = df[~df["series"].str.contains("TrainCheck")]
    fig, ax = plt.subplots(figsize=(COL, 3.15))
    for series, g in df.groupby("series", sort=False):
        if g.empty:
            continue
        color, marker, lab = STYLE.get(_key(series), ("0.3", "o", _key(series)))
        g = g.sort_values("coverage")
        muted = "Reviewer" in lab
        ax.plot(g["coverage"], g["covered_macro_f1"], marker=marker, color=color,
                markersize=3.2, linewidth=1.0 if muted else 1.4,
                alpha=0.85 if muted else 1.0, markeredgewidth=0, label=lab)
    ax.scatter([0.055], [0.419], marker="*", s=90, color="black", zorder=6,
               edgecolors="white", linewidths=0.4, label="Codex (full)")
    ax.set_xlabel("Coverage (non-inconclusive rate)")
    ax.set_ylabel("Macro F1 on covered cases")
    ax.set_xlim(0.0, 1.0)
    _despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              borderaxespad=0.0)
    _save(fig, "paper_synthetic_selective_f1_coverage")
    print("selective_f1: done")


def regen_max_error():
    df = pd.read_csv(os.path.join(DOCS, "paper_synthetic_max_error_coverage.csv"))
    min_cov = 0.15
    df = df[(df["coverage"] >= min_cov) & (~df["series"].str.contains("TrainCheck"))]
    fig, ax = plt.subplots(figsize=(COL, 3.15))
    for series, g in df.groupby("series", sort=False):
        if g.empty:
            continue
        color, marker, lab = STYLE.get(_key(series), ("0.3", "o", _key(series)))
        g = g.sort_values("coverage")
        muted = "Reviewer" in lab
        ax.plot(g["coverage"], g["max_error"], marker=marker, color=color,
                markersize=3.2, linewidth=1.0 if muted else 1.4,
                alpha=0.85 if muted else 1.0, markeredgewidth=0, label=lab)
    ax.set_xlabel("Coverage (accepted pass + fail rate)")
    ax.set_ylabel("Worst-class error rate")
    ax.set_ylim(0.0, None)
    ax.set_xlim(min_cov - 0.03, 1.0)
    _despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              borderaxespad=0.0)
    _save(fig, "paper_synthetic_max_error_coverage")
    print(f"max_error: done (coverage >= {min_cov})")


def regen_probe_baselines():
    df = pd.read_csv(os.path.join(DOCS, "paper_probe_baseline_metrics.csv"))
    method_labels = {
        "Direct Qwen Flash": "Direct Qwen",
        "Direct GPT-4.1 mini": "Direct GPT-4.1",
        "Codex reviewer": "Codex reviewer",
        "VibeTest ex0": "VibeTest ex0",
        "VibeTest ex10": "VibeTest ex10",
        "VibeTest ex20": "VibeTest ex20",
    }
    method_colors = {
        "Direct Qwen Flash": "#6F6F6F",
        "Direct GPT-4.1 mini": "#A0A0A0",
        "Codex reviewer": "#333333",
        "VibeTest ex0": C_BLUE,
        "VibeTest ex10": C_ORANGE,
        "VibeTest ex20": C_GREEN,
    }
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.55), sharex=True,
                             sharey=True)
    panels = [("macro_f1", "Macro F1"), ("fail_recall", "FAIL recall")]
    for ax, (metric, title) in zip(axes, panels):
        for _, row in df.iterrows():
            method = row["method"]
            ax.scatter(row["coverage"], row[metric], s=42,
                       color=method_colors[method], edgecolors="white",
                       linewidths=0.55, zorder=3)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Coverage")
        ax.set_xlim(0.38, 1.0)
        ax.set_ylim(0.08, 0.86)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        _despine(ax)
    axes[0].set_ylabel("Metric value")
    method_handles = [
        plt.Line2D([0], [0], marker="o", color=method_colors[m],
                   linestyle="", markersize=4.8, label=method_labels[m])
        for m in df["method"]
    ]
    fig.legend(handles=method_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.1,
               handletextpad=0.45)
    fig.subplots_adjust(bottom=0.31, left=0.09, right=0.99, top=0.86,
                        wspace=0.12)
    _save(fig, "paper_probe_baseline_metrics")
    print("probe_baselines: done")


def regen_real_f1(min_cov=0.30):
    df = pd.read_csv(os.path.join(DOCS, "paper_real_conservative_f1_coverage.csv"))
    df = df[df["coverage"] >= min_cov]
    ex = {0: (C_BLUE, "o", "ex0"), 10: (C_ORANGE, "s", "ex10"),
          20: (C_GREEN, "^", "ex20")}
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    for e, (color, marker, lab) in ex.items():
        g = df[df["examples"] == e].sort_values("coverage")
        ax.plot(g["coverage"], g["macro_f1"], marker=marker, color=color,
                markersize=3.2, markeredgewidth=0, label=lab)
    ax.set_xlabel("Coverage (accepted pass + fail rate)")
    ax.set_ylabel("Estimated macro F1")
    ax.set_xlim(min_cov - 0.03, 1.0)
    ax.set_ylim(0.88, 1.0)
    _despine(ax)
    ax.legend(loc="lower left", ncol=3, columnspacing=0.8)
    _save(fig, "paper_real_conservative_f1_coverage")
    print(f"real_f1: done (coverage >= {min_cov})")


def regen_ecdf():
    df = pd.read_csv(os.path.join(DOCS, "paper_case_score_histograms.csv"))
    ex = {"ex0": (C_BLUE, "ex0"), "ex10": (C_ORANGE, "ex10"),
          "ex20": (C_GREEN, "ex20")}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.5), sharey=True)
    for ax, source in zip(axes, ["Synthetic", "Real"]):
        sub = df[df["source"] == source]
        for key, (color, lab) in ex.items():
            s = sub[sub["examples"] == key]["case_score"].sort_values().values
            if len(s) == 0:
                continue
            y = (pd.Series(range(1, len(s) + 1)) / len(s)).values
            ax.step(s, y, where="post", color=color, linewidth=1.3, label=lab)
        ax.set_title(source)
        ax.set_xlabel("case\\_score" if False else "case score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _despine(ax)
    axes[0].set_ylabel("Cumulative fraction")
    axes[1].legend(loc="lower right")
    _save(fig, "paper_case_score_ecdf")
    print("ecdf: done")


if __name__ == "__main__":
    regen_selective_f1()
    regen_probe_baselines()
    regen_max_error()
    regen_real_f1()
    regen_ecdf()
    print("done")
