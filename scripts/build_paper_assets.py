from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent / "vibetest_qwen_kaggle_emnlp2026_package"
OUT = ROOT / "results" / "paper_assets"
PACKAGE_FIGURES = PACKAGE / "figures"
PACKAGE_DOCS = PACKAGE / "supporting_docs"

SYNTHETIC_CURVE_CSV = (
    ROOT
    / "results"
    / "synthetic"
    / "figures"
    / "synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label.csv"
)
SYNTHETIC_ERROR_CSV = (
    ROOT
    / "results"
    / "synthetic"
    / "figures"
    / "synthetic_kaggle_qwen36_symmetric_max_selective_error_vs_coverage_log_y_inverted.csv"
)
SUMMARY_CSV = ROOT / "results" / "benchmark_summary.csv"
REAL_AUDIT_CSV = ROOT / "results" / "openrouter_fail_human_audit_sample15_gpt5mini.csv"
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
DATASETS = ("titanic", "diabetic", "nlp")
EXAMPLES = (0, 10, 20)

PROBE_ROWS = [
    {"method": "Direct Qwen Flash", "macro_f1": 0.620, "coverage": 0.840, "fail_recall": 0.486},
    {"method": "Direct GPT-4.1 mini", "macro_f1": 0.610, "coverage": 0.973, "fail_recall": 0.378},
    {"method": "Codex reviewer", "macro_f1": 0.349, "coverage": 0.440, "fail_recall": 0.135},
    {"method": "VibeTest ex0", "macro_f1": 0.681, "coverage": 0.840, "fail_recall": 0.568},
    {"method": "VibeTest ex10", "macro_f1": 0.791, "coverage": 0.920, "fail_recall": 0.811},
    {"method": "VibeTest ex20", "macro_f1": 0.775, "coverage": 0.893, "fail_recall": 0.730},
]

PALETTE = {
    "VibeTest static": "#3B6EA8",
    "VibeTest static + ex10": "#D18F00",
    "VibeTest static + ex20": "#00856F",
    "TrainCheck": "#6F6F6F",
    "Reviewer mode 0": "#C99BC8",
    "Reviewer mode 1": "#B75D9A",
    "Reviewer mode 2": "#7D3F63",
}
MARKERS = {
    "VibeTest static": "o",
    "VibeTest static + ex10": "s",
    "VibeTest static + ex20": "^",
    "TrainCheck": "D",
    "Reviewer mode 0": "P",
    "Reviewer mode 1": "X",
    "Reviewer mode 2": "v",
}


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "figure.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        },
    )


def base_series(label: str) -> str:
    return label.split(" (", 1)[0]


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PACKAGE_FIGURES.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=320)
    fig.savefig(pdf)
    plt.close(fig)
    shutil.copy2(png, PACKAGE_FIGURES / png.name)
    shutil.copy2(pdf, PACKAGE_FIGURES / pdf.name)


def copy_csv(path: Path, stem: str) -> None:
    target = OUT / f"{stem}.csv"
    shutil.copy2(path, target)
    PACKAGE_DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, PACKAGE_DOCS / target.name)


def markdown_table(df: pd.DataFrame) -> str:
    rows = df.fillna("").astype(str).to_dict("records")
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in headers) + " |")
    return "\n".join(lines)


def plot_synthetic_f1() -> None:
    df = pd.read_csv(SYNTHETIC_CURVE_CSV)
    df = df[df["covered_macro_f1"].notna()].copy()
    df["base_series"] = df["series"].map(base_series)

    fig, ax = plt.subplots(figsize=(7.0, 3.65))
    for label, group in df.groupby("series", sort=False):
        group = group.sort_values("coverage")
        base = base_series(label)
        alpha = 1.0 if base.startswith("VibeTest") else 0.72
        linewidth = 2.4 if base.startswith("VibeTest") else 1.7
        ax.plot(
            group["coverage"],
            group["covered_macro_f1"],
            label=label,
            color=PALETTE.get(base),
            marker=MARKERS.get(base, "o"),
            markersize=3.8,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_xlabel("Coverage (non-inconclusive rate)")
    ax.set_ylabel("Macro F1 on covered cases")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.56, 0.86)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )
    sns.despine(ax=ax)
    fig.subplots_adjust(bottom=0.34)
    save_figure(fig, "paper_synthetic_selective_f1_coverage")
    copy_csv(SYNTHETIC_CURVE_CSV, "paper_synthetic_selective_f1_coverage")


def plot_synthetic_error() -> None:
    df = pd.read_csv(SYNTHETIC_ERROR_CSV)
    df = df[df["max_error"].notna() & (df["coverage"] > 0)].copy()
    df["base_series"] = df["series"].map(base_series)

    fig, ax = plt.subplots(figsize=(7.0, 3.65))
    for label, group in df.groupby("series", sort=False):
        group = group.sort_values("coverage")
        base = base_series(label)
        alpha = 1.0 if base.startswith("VibeTest") else 0.72
        linewidth = 2.4 if base.startswith("VibeTest") else 1.7
        ax.plot(
            group["coverage"],
            group["max_error"],
            label=label,
            color=PALETTE.get(base),
            marker=MARKERS.get(base, "o"),
            markersize=3.8,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_xlabel("Coverage (accepted pass + fail rate)")
    ax.set_ylabel("Worst class error rate")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 0.65)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )
    sns.despine(ax=ax)
    fig.subplots_adjust(bottom=0.34)
    save_figure(fig, "paper_synthetic_max_error_coverage")
    copy_csv(SYNTHETIC_ERROR_CSV, "paper_synthetic_max_error_coverage")


def result_path(dataset: str, examples: int) -> Path:
    return ROOT / "results" / f"kaggle_{dataset}_AT-{MODEL}-static_examples{examples}.jsonl"


def case_score(test: dict[str, Any]) -> float | None:
    raw = (test.get("metadata") or {}).get("case_score")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return max(0.0, min(1.0, value))


def load_real_full_items(examples: int) -> list[dict[str, float]]:
    items = []
    for dataset in DATASETS:
        path = result_path(dataset, examples)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for test in row.get("tests") or []:
                    score = case_score(test)
                    if score is None:
                        continue
                    items.append({"dataset": dataset, "examples": examples, "case_score": score})
    return items


def load_real_audit_items(examples: int) -> list[dict[str, Any]]:
    out = []
    with REAL_AUDIT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["examples"]) != examples:
                continue
            try:
                score = float(row["case_score"])
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "case_score": max(0.0, min(1.0, score)),
                    "true_fail": row["audited_outcome"] == "TRUE_FAIL",
                }
            )
    return out


def real_macro_f1(pass_count: float, fail_count: float, true_fail_rate: float) -> float:
    tp = fail_count * true_fail_rate
    fp = fail_count * (1.0 - true_fail_rate)
    tn = pass_count
    fn = 0.0
    fail_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    pass_f1 = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return (fail_f1 + pass_f1) / 2.0


def real_curve_for_examples(examples: int) -> list[dict[str, Any]]:
    full = load_real_full_items(examples)
    audit = load_real_audit_items(examples)
    total = len(full)
    thresholds = [
        0.51,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.91,
        0.92,
        0.93,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    ]
    global_true_rate = (
        sum(1 for item in audit if item["true_fail"]) / len(audit) if audit else 0.0
    )
    rows = []
    for threshold in thresholds:
        pass_cutoff = 1.0 - threshold
        accepted_pass = [item for item in full if item["case_score"] <= pass_cutoff]
        accepted_fail = [item for item in full if item["case_score"] >= threshold]
        accepted_audit = [item for item in audit if item["case_score"] >= threshold]
        true_rate = (
            sum(1 for item in accepted_audit if item["true_fail"]) / len(accepted_audit)
            if accepted_audit
            else global_true_rate
        )
        rows.append(
            {
                "examples": examples,
                "threshold": threshold,
                "pass_cutoff": pass_cutoff,
                "coverage": (len(accepted_pass) + len(accepted_fail)) / total if total else 0.0,
                "macro_f1": real_macro_f1(len(accepted_pass), len(accepted_fail), true_rate),
                "audit_n": len(accepted_audit),
                "true_rate": true_rate,
            }
        )
    return rows


def plot_real_f1() -> None:
    rows = []
    for examples in EXAMPLES:
        rows.extend(real_curve_for_examples(examples))
    df = pd.DataFrame(rows)
    csv_path = OUT / "paper_real_conservative_f1_coverage.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    PACKAGE_DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, PACKAGE_DOCS / csv_path.name)

    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    colors = {0: PALETTE["VibeTest static"], 10: PALETTE["VibeTest static + ex10"], 20: PALETTE["VibeTest static + ex20"]}
    labels = {0: "ex0", 10: "ex10", 20: "ex20"}
    for examples, group in df.groupby("examples", sort=True):
        group = group.sort_values("coverage")
        ax.plot(
            group["coverage"],
            group["macro_f1"],
            label=labels[examples],
            color=colors[examples],
            marker=MARKERS["VibeTest static"],
            markersize=3.7,
            linewidth=2.4,
        )
        best = group.loc[group["macro_f1"].idxmax()]
        ax.scatter(
            [best["coverage"]],
            [best["macro_f1"]],
            s=45,
            color=colors[examples],
            edgecolor="#222222",
            linewidth=0.7,
            zorder=5,
        )

    ax.set_xlabel("Coverage (accepted pass + fail rate)")
    ax.set_ylabel("Estimated macro F1")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.45, 1.01)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, borderpad=0.35)
    sns.despine(ax=ax)
    save_figure(fig, "paper_real_conservative_f1_coverage")


def plot_probe_baselines() -> None:
    df = pd.DataFrame(PROBE_ROWS)
    csv_path = OUT / "paper_probe_baseline_metrics.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    PACKAGE_DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, PACKAGE_DOCS / csv_path.name)

    long = df.melt(
        id_vars="method",
        value_vars=["macro_f1", "coverage", "fail_recall"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "macro_f1": "Macro F1",
        "coverage": "Coverage",
        "fail_recall": "FAIL recall",
    }
    metric_colors = {
        "macro_f1": "#3B6EA8",
        "coverage": "#D18F00",
        "fail_recall": "#C0392B",
    }
    methods = [row["method"] for row in PROBE_ROWS]
    y_positions = {method: idx for idx, method in enumerate(reversed(methods))}

    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    offsets = {"macro_f1": -0.18, "coverage": 0.0, "fail_recall": 0.18}
    for metric in ["macro_f1", "coverage", "fail_recall"]:
        subset = long[long["metric"] == metric]
        ax.scatter(
            subset["value"],
            [y_positions[method] + offsets[metric] for method in subset["method"]],
            s=42,
            color=metric_colors[metric],
            label=metric_labels[metric],
            zorder=3,
        )
        for _, row in subset.iterrows():
            ax.plot(
                [0, row["value"]],
                [y_positions[row["method"]] + offsets[metric]] * 2,
                color=metric_colors[metric],
                alpha=0.18,
                linewidth=1.4,
                zorder=1,
            )

    ax.set_yticks([y_positions[method] for method in reversed(methods)])
    ax.set_yticklabels(list(reversed(methods)))
    ax.set_xlabel("Metric value")
    ax.set_xlim(0.0, 1.02)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
    )
    sns.despine(ax=ax, left=False)
    fig.subplots_adjust(bottom=0.25)
    save_figure(fig, "paper_probe_baseline_metrics")


def synthetic_score_rows() -> list[dict[str, Any]]:
    rows = []
    for examples in EXAMPLES:
        for dataset in DATASETS:
            path = (
                ROOT
                / "results"
                / "synthetic"
                / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples{examples}_case_score_bands.jsonl"
            )
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    for test in entry.get("tests") or []:
                        score = case_score(test)
                        if score is not None:
                            rows.append(
                                {
                                    "source": "Synthetic",
                                    "dataset": dataset,
                                    "examples": f"ex{examples}",
                                    "case_score": score,
                                }
                            )
    return rows


def real_score_rows() -> list[dict[str, Any]]:
    rows = []
    for examples in EXAMPLES:
        for item in load_real_full_items(examples):
            rows.append(
                {
                    "source": "Real",
                    "dataset": item["dataset"],
                    "examples": f"ex{examples}",
                    "case_score": item["case_score"],
                }
            )
    return rows


def plot_score_histograms() -> None:
    rows = synthetic_score_rows() + real_score_rows()
    df = pd.DataFrame(rows)
    csv_path = OUT / "paper_case_score_histograms.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    PACKAGE_DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, PACKAGE_DOCS / csv_path.name)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)
    hist_colors = {
        "ex0": PALETTE["VibeTest static"],
        "ex10": PALETTE["VibeTest static + ex10"],
        "ex20": PALETTE["VibeTest static + ex20"],
    }
    for ax, source in zip(axes, ("Synthetic", "Real"), strict=True):
        subset = df[df["source"] == source]
        for examples in ("ex0", "ex10", "ex20"):
            vals = subset[subset["examples"] == examples]["case_score"].sort_values().to_list()
            if not vals:
                continue
            y = [(idx + 1) / len(vals) for idx in range(len(vals))]
            ax.step(
                vals,
                y,
                where="post",
                linewidth=1.8,
                color=hist_colors[examples],
                label=examples,
            )
        ax.set_title(source)
        ax.set_xlabel("case_score")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Cumulative fraction")
    axes[1].legend(loc="upper left", frameon=True, framealpha=0.95, borderpad=0.35)
    save_figure(fig, "paper_case_score_ecdf")


def write_claims_doc() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    headline = summary[summary["section"].isin(["synthetic", "real_kaggle"])].copy()
    audit = summary[summary["section"] == "audit"].copy()

    lines = [
        "# Frozen Claims and Canonical Results",
        "",
        "This document pins the paper story so individual plots do not keep changing the thesis.",
        "",
        "## Main Claim",
        "",
        "VibeTest turns natural-language ML-pipeline properties into evidence-backed repository tests. "
        "The method should be evaluated with abstention, score-threshold curves, and audits of property interpretation.",
        "",
        "## Claims We Can Defend",
        "",
        "- VibeTest with Qwen3.6 produces usable selective predictions on synthetic and real Kaggle-style ML repositories.",
        "- Prompt examples change the coverage/accuracy tradeoff: ex10 gives the strongest synthetic coverage, while ex20 gives the strongest synthetic macro F1.",
        "- Real Kaggle performance should be reported conservatively because labels come from sampled fail audits, not exhaustive ground truth.",
        "- Synthetic raw F1 is useful but noisy because injected labels and evidence-based property judgments can disagree.",
        "- The direct-property and Codex-style runs are useful probes, but the same-model direct-property baseline remains the cleanest missing comparison.",
        "",
        "## Claims We Should Not Overstate",
        "",
        "- Do not claim complete real-Kaggle recall; the real audit samples predicted failures.",
        "- Do not claim the full-context real audit proves near-perfect F1; it is a sensitivity check.",
        "- Do not claim TrainCheck is intrinsically worse; our available canonical files show execution-fragile zero coverage.",
        "- Do not claim every synthetic disagreement is a model error; the audit shows many are property-boundary mismatches.",
        "",
        "## Canonical Headline Rows",
        "",
        markdown_table(
            headline[
                [
                    "section",
                    "name",
                    "tests",
                    "coverage",
                    "best_selective_f1",
                    "audited_macro_f1",
                    "notes",
                ]
            ]
        ),
        "",
        "## Audit Rows",
        "",
        markdown_table(audit[["name", "tests", "notes"]]),
        "",
    ]
    path = ROOT / "docs" / "PAPER_CLAIMS_AND_RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    PACKAGE_DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, PACKAGE_DOCS / path.name)


def write_figure_manifest() -> None:
    lines = [
        "# Paper Figure Inventory",
        "",
        "| Figure file | Purpose | Data source | Main-text use |",
        "| --- | --- | --- | --- |",
        "| `paper_synthetic_selective_f1_coverage.{png,pdf}` | Synthetic selective macro F1 vs coverage for VibeTest and reviewer modes. TrainCheck has zero usable coverage and is reported in the table rather than as a visible curve. | `results/synthetic/figures/synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label.csv` | Main result figure for synthetic benchmark. |",
        "| `paper_synthetic_max_error_coverage.{png,pdf}` | Worst pass/fail selective error vs coverage on a linear y-axis. | `results/synthetic/figures/synthetic_kaggle_qwen36_symmetric_max_selective_error_vs_coverage_log_y_inverted.csv` | Reliability/error analysis. |",
        "| `paper_probe_baseline_metrics.{png,pdf}` | Matched five-repository probe comparing direct prompting, Codex review, and VibeTest. | Values from the probe table in the manuscript. | Baseline comparison figure. |",
        "| `paper_real_conservative_f1_coverage.{png,pdf}` | Conservative real Kaggle macro F1 vs coverage under two-sided thresholds. | Real Qwen static outputs plus `openrouter_fail_human_audit_sample15_gpt5mini.csv` | Real benchmark figure. |",
        "| `paper_case_score_ecdf.{png,pdf}` | Empirical score distributions for synthetic and real ex0/ex10/ex20. | Synthetic and real Qwen static JSONL files. | Calibration/score-spread figure. |",
        "| `agentic_testing_evidence_verification.png` | Workflow overview figure from the earlier EMNLP-style writeup. | Existing package asset. | Method overview. |",
        "",
        "All paper figures are mirrored to `vibetest_qwen_kaggle_emnlp2026_package/figures/` and the underlying CSVs are mirrored to `supporting_docs/` where applicable.",
        "",
    ]
    path = ROOT / "docs" / "PAPER_FIGURE_INVENTORY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    PACKAGE_DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, PACKAGE_DOCS / path.name)


def write_error_analysis_doc() -> None:
    lines = [
        "# Error Analysis",
        "",
        "## Synthetic Disagreements",
        "",
        "The synthetic audit should be read as an audit of property operationalization, not only as an audit of Qwen. "
        "Among 53 usable sampled high-confidence disagreements, 29 were real Qwen misses and 24 were property-definition mismatches.",
        "",
        "The main mismatch pattern is that the injected bug label sometimes encodes a broader or narrower version of the natural-language property than the evidence rubric. "
        "For example, a synthetic `avoid replaceable loops` label can treat benign file/path loops as failures, while the VibeTest/verifier rubric treats the property as targeting active numerical loops that should be vectorized. "
        "Similarly, properties around parameter freezing or augmentation depend on whether earlier training stages, validation transforms, or stated intent count as evidence.",
        "",
        "## Real Kaggle Audit",
        "",
        "The conservative real audit labels sampled predicted failures without always having full repository context. "
        "When the verifier was given fuller source context, many conservative false failures became supported failures. "
        "That is why the paper should headline conservative F1 and report full-context adjusted values only as sensitivity.",
        "",
        "## Interpretation",
        "",
        "The clean story is not that synthetic is bad or real is easy. "
        "The clean story is that natural-language ML properties have boundaries, and benchmark labels are one operationalization of those boundaries. "
        "VibeTest exposes these boundaries because it produces evidence and abstentions rather than only binary labels.",
        "",
    ]
    path = ROOT / "docs" / "PAPER_ERROR_ANALYSIS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    PACKAGE_DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, PACKAGE_DOCS / path.name)


def main() -> None:
    configure_style()
    plot_synthetic_f1()
    plot_synthetic_error()
    plot_probe_baselines()
    plot_real_f1()
    plot_score_histograms()
    write_claims_doc()
    write_figure_manifest()
    write_error_analysis_doc()
    print(f"Wrote paper assets to {OUT}")
    print(f"Mirrored figures/docs to {PACKAGE}")


if __name__ == "__main__":
    main()
