"""Plot precision-recall curves from analyze_evidence_verifier_scores.py output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _short_label(file_path: str) -> str:
    name = Path(str(file_path).replace("\\", "/")).name
    if name == "ALL":
        return "Combined"
    for key, label in (
        ("titanic", "Titanic"),
        ("diabetic", "Diabetic"),
        ("nlp", "NLP"),
    ):
        if key in name.lower():
            return label
    return name


def _load_curve_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_summary_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["file"], row["metric"])] = row
    return out


def _curve_points(rows: list[dict[str, str]], file_key: str) -> list[tuple[float, float, float, float]]:
    points: list[tuple[float, float, float, float]] = []
    for row in rows:
        if row["file"] != file_key:
            continue
        if not row.get("precision") or not row.get("recall"):
            continue
        recall = float(row["recall"])
        precision = float(row["precision"])
        threshold = float(row["threshold"])
        if row.get("f1"):
            f1 = float(row["f1"])
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        points.append((recall, precision, threshold, f1))
    points.sort(key=lambda item: item[0])
    return points


def _dedupe_step(points: list[tuple[float, float, float, float]]) -> tuple[list[float], list[float]]:
    recalls: list[float] = []
    precisions: list[float] = []
    seen: set[tuple[float, float]] = set()
    for recall, precision, _, _ in points:
        key = (round(recall, 6), round(precision, 6))
        if key in seen:
            continue
        seen.add(key)
        recalls.append(recall)
        precisions.append(precision)
    return recalls, precisions


def _metric_value(row: dict[str, str] | None, *keys: str) -> float | None:
    if not row:
        return None
    for key in keys:
        raw = row.get(key)
        if raw not in (None, ""):
            return float(raw)
    return None


def _plot_pr_curves(
    *,
    curve_rows: list[dict[str, str]],
    summary_rows: dict[tuple[str, str], dict[str, str]],
    file_keys: list[str],
    out_path: Path,
    title: str,
    mark_threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    colors = plt.get_cmap("tab10").colors

    for idx, file_key in enumerate(file_keys):
        points = _curve_points(curve_rows, file_key)
        if not points:
            continue
        recalls, precisions = _dedupe_step(points)
        label = _short_label(file_key)
        color = colors[idx % len(colors)]
        ax.step(recalls, precisions, where="post", linewidth=2.0, color=color, label=label)

        baseline = summary_rows.get((file_key, "baseline"))
        baseline_recall = _metric_value(baseline, "recall", "recall_of_labeled_correct_fails")
        baseline_precision = _metric_value(baseline, "precision")
        if baseline_recall is not None and baseline_precision is not None:
            ax.scatter(
                [baseline_recall],
                [baseline_precision],
                marker="o",
                s=70,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )

        threshold_row = summary_rows.get((file_key, f"threshold_{mark_threshold:g}"))
        threshold_recall = _metric_value(threshold_row, "recall", "recall_of_labeled_correct_fails")
        threshold_precision = _metric_value(threshold_row, "precision")
        if threshold_recall is not None and threshold_precision is not None:
            ax.scatter(
                [threshold_recall],
                [threshold_precision],
                marker="x",
                s=70,
                color=color,
                linewidths=2.0,
                zorder=5,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall (fraction of valid FAILs kept)")
    ax.set_ylabel("Precision (fraction of kept FAILs that are valid)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.extend(
        [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="Baseline (no filter)"),
            Line2D([0], [0], marker="x", color="gray", markeredgecolor="gray", linestyle="None", markersize=8, label=f"Threshold {mark_threshold:g}"),
        ]
    )
    ax.legend(handles=handles, loc="lower left", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_f1_vs_threshold(
    *,
    curve_rows: list[dict[str, str]],
    file_keys: list[str],
    out_path: Path,
    mark_threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = plt.get_cmap("tab10").colors

    for idx, file_key in enumerate(file_keys):
        points = _curve_points(curve_rows, file_key)
        if not points:
            continue
        thresholds = [item[2] for item in points]
        f1s = [item[3] for item in points]
        order = sorted(range(len(thresholds)), key=lambda i: thresholds[i], reverse=True)
        thresholds = [thresholds[i] for i in order]
        f1s = [f1s[i] for i in order]
        ax.plot(thresholds, f1s, linewidth=2.0, color=colors[idx % len(colors)], label=_short_label(file_key))

    ax.axvline(mark_threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Verifier score threshold")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs verifier threshold")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-curve-csv",
        type=Path,
        default=Path("results/synthetic/verifier_pr_curve.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/synthetic/verifier_scores_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/synthetic/figures"),
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--title-prefix", default="", help="Optional prefix for plot titles.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    curve_rows = _load_curve_rows(args.pr_curve_csv)
    summary_rows = _load_summary_rows(args.summary_csv)

    file_keys = sorted({row["file"] for row in curve_rows}, key=lambda value: (value != "ALL", value))
    split_keys = [key for key in file_keys if key != "ALL"]
    combined_keys = [key for key in file_keys if key == "ALL"]

    out_dir = args.output_dir
    title_prefix = f"{args.title_prefix} " if args.title_prefix else ""
    per_split = out_dir / "verifier_pr_curves_by_split.png"
    combined = out_dir / "verifier_pr_curve_combined.png"
    f1_plot = out_dir / "verifier_f1_vs_threshold.png"

    _plot_pr_curves(
        curve_rows=curve_rows,
        summary_rows=summary_rows,
        file_keys=split_keys,
        out_path=per_split,
        title=f"{title_prefix}Verifier threshold tradeoff by split".strip(),
        mark_threshold=args.threshold,
    )
    if combined_keys:
        _plot_pr_curves(
            curve_rows=curve_rows,
            summary_rows=summary_rows,
            file_keys=combined_keys,
            out_path=combined,
            title=f"{title_prefix}Verifier threshold tradeoff (combined)".strip(),
            mark_threshold=args.threshold,
        )
    _plot_f1_vs_threshold(
        curve_rows=curve_rows,
        file_keys=split_keys + combined_keys,
        out_path=f1_plot,
        mark_threshold=args.threshold,
    )

    print(f"Wrote {per_split}")
    if combined_keys:
        print(f"Wrote {combined}")
    print(f"Wrote {f1_plot}")


if __name__ == "__main__":
    main()
