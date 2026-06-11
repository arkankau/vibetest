from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_DIR = Path("results/synthetic")
FIG_DIR = RESULT_DIR / "figures"
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
SUFFIX = "_case_score_bands"
DATASETS = ("titanic", "diabetic", "nlp")
LOG_EPSILON = 1e-3


@dataclass(frozen=True)
class Item:
    label: int
    score: float


def example_paths(examples: int) -> list[Path]:
    return [
        RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples{examples}{SUFFIX}.jsonl"
        for dataset in DATASETS
    ]


def reviewer_paths(mode: int) -> list[Path]:
    return [
        RESULT_DIR
        / f"synthetic_kaggle_{dataset}_baseline-reviewer-mode{mode}-static-{MODEL}_mapper-gpt-5.4-mini.jsonl"
        for dataset in DATASETS
    ]


def traincheck_paths() -> list[Path]:
    return [
        RESULT_DIR / f"synthetic_kaggle_{dataset}_traincheck_mapper-{MODEL}.jsonl"
        for dataset in DATASETS
    ]


def output_tokens_per_property(paths: list[Path]) -> float | None:
    output_tokens = 0
    total_tests = 0
    saw_usage = False
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_tests += int(row.get("total_tests") or len(row.get("tests") or []))
                totals = (row.get("usage") or {}).get("usage_totals") or {}
                if totals.get("output_tokens") is not None:
                    saw_usage = True
                    output_tokens += int(totals.get("output_tokens") or 0)
    if not saw_usage or total_tests <= 0:
        return None
    return output_tokens / total_tests


def label_with_tokens(name: str, paths: list[Path]) -> str:
    value = output_tokens_per_property(paths)
    if value is None:
        return name
    return f"{name} ({value / 1000:.2f}K out/prop)"


def load_items(paths: list[Path]) -> list[Item]:
    items: list[Item] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                labels = row.get("ground_truth_property_labels") or {}
                for test in row.get("tests") or []:
                    meta = test.get("metadata") or {}
                    synthetic_score = meta.get("synthetic_score") or {}
                    prop = str(meta.get("property_id") or synthetic_score.get("property_id") or "")
                    if synthetic_score.get("ground_truth_label") is not None:
                        label = 1 if int(synthetic_score["ground_truth_label"]) else 0
                    elif prop in labels:
                        label = 1 if int(labels[prop]) else 0
                    else:
                        continue
                    raw_score = meta.get("case_score")
                    if raw_score is None:
                        raw_score = meta.get("fail_support_score")
                    if not isinstance(raw_score, (int, float)):
                        continue
                    items.append(Item(label=label, score=float(raw_score)))
    return items


def safe_ratio(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


def evaluate(items: list[Item], threshold: float) -> dict:
    low = 1.0 - threshold
    tp = fp = tn = fn = 0
    covered = pred_pass = pred_fail = inconclusive = 0
    for item in items:
        if item.score <= low:
            pred = 0
            pred_pass += 1
        elif item.score >= threshold:
            pred = 1
            pred_fail += 1
        else:
            inconclusive += 1
            continue

        covered += 1
        if pred == 1 and item.label == 1:
            tp += 1
        elif pred == 1 and item.label == 0:
            fp += 1
        elif pred == 0 and item.label == 0:
            tn += 1
        elif pred == 0 and item.label == 1:
            fn += 1

    fail_error = safe_ratio(fp, tp + fp)
    pass_error = safe_ratio(fn, tn + fn)
    max_error = None
    if fail_error is not None and pass_error is not None:
        max_error = max(fail_error, pass_error)

    return {
        "threshold": threshold,
        "low_threshold": low,
        "high_threshold": threshold,
        "total": len(items),
        "coverage": covered / len(items) if items else 0.0,
        "fail_error": fail_error,
        "pass_error": pass_error,
        "max_error": max_error,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_pass": pred_pass,
        "predicted_fail": pred_fail,
        "inconclusive": inconclusive,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, list[Path]]] = []
    specs.append((label_with_tokens("VibeTest static", example_paths(0)), example_paths(0)))
    specs.append((label_with_tokens("VibeTest static + ex10", example_paths(10)), example_paths(10)))
    specs.append((label_with_tokens("VibeTest static + ex20", example_paths(20)), example_paths(20)))
    specs.append((label_with_tokens("TrainCheck", traincheck_paths()), traincheck_paths()))
    for mode in (0, 1, 2):
        paths = reviewer_paths(mode)
        specs.append((label_with_tokens(f"Reviewer mode {mode}", paths), paths))

    missing = [str(path) for _, paths in specs for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    colors = {
        "VibeTest static": "#9ecae1",
        "VibeTest static + ex10": "#1f77b4",
        "VibeTest static + ex20": "#08306b",
        "TrainCheck": "#666666",
        "Reviewer mode 0": "#DDA0C6",
        "Reviewer mode 1": "#CC79A7",
        "Reviewer mode 2": "#8F4A73",
    }
    markers = {
        "VibeTest static": "o",
        "VibeTest static + ex10": "s",
        "VibeTest static + ex20": "^",
        "TrainCheck": "D",
        "Reviewer mode 0": "o",
        "Reviewer mode 1": "o",
        "Reviewer mode 2": "o",
    }

    rows: list[dict] = []
    plt.figure(figsize=(9.0, 6.2))
    for label, paths in specs:
        base_label = label.split(" (", 1)[0]
        items = load_items(paths)
        series_rows = []
        for step in range(50, 101):
            row = evaluate(items, step / 100)
            row["series"] = label
            series_rows.append(row)
            rows.append(row)
        points = [
            row
            for row in series_rows
            if row["max_error"] is not None and row["coverage"] > 0
        ]
        points.sort(key=lambda row: row["coverage"])
        plt.plot(
            [row["coverage"] for row in points],
            [max(row["max_error"], LOG_EPSILON) for row in points],
            marker=markers.get(base_label, "o"),
            linewidth=2,
            color=colors.get(base_label),
            label=label,
        )

    plt.xlabel("Coverage (non-INCONCLUSIVE rate)")
    plt.ylabel("max(FP/(TP+FP), FN/(TN+FN)) (log scale)")
    plt.title("Synthetic Kaggle Qwen3.6: Max Selective Error vs Coverage")
    plt.xlim(0, 1.02)
    plt.yscale("log")
    plt.ylim(1.02, LOG_EPSILON)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    prefix = FIG_DIR / "synthetic_kaggle_qwen36_symmetric_max_selective_error_vs_coverage_log_y_inverted"
    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")
    csv_path = prefix.with_suffix(".csv")
    plt.savefig(png, dpi=180)
    plt.savefig(pdf)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv_path}")
    for label in [label for label, _ in specs]:
        valid = [
            row
            for row in rows
            if row["series"] == label and row["max_error"] is not None
        ]
        if not valid:
            print(f"{label}: no valid points")
            continue
        best = min(valid, key=lambda row: row["max_error"])
        print(
            f"{label}: min max_error={best['max_error']:.3f} "
            f"coverage={best['coverage']:.3f} t={best['threshold']:.2f} "
            f"fail_error={best['fail_error']:.3f} pass_error={best['pass_error']:.3f}"
        )


if __name__ == "__main__":
    main()
