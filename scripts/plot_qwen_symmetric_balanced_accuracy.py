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
EXAMPLES = (0, 10, 20)


@dataclass(frozen=True)
class Item:
    label: int
    score: float


def load_items(examples: int) -> list[Item]:
    items: list[Item] = []
    for dataset in DATASETS:
        path = RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples{examples}{SUFFIX}.jsonl"
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                labels = row.get("ground_truth_property_labels") or {}
                for test in row.get("tests") or []:
                    meta = test.get("metadata") or {}
                    prop = str(meta.get("property_id") or "")
                    raw_score = meta.get("case_score")
                    if raw_score is None:
                        raw_score = meta.get("fail_support_score")
                    if prop not in labels or not isinstance(raw_score, (int, float)):
                        continue
                    items.append(Item(label=1 if int(labels[prop]) else 0, score=float(raw_score)))
    return items


def safe_ratio(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


def evaluate(items: list[Item], threshold: float) -> dict:
    low = 1.0 - threshold
    pass_total = fail_total = 0
    pass_correct = fail_correct = 0
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
        if item.label == 0:
            pass_total += 1
            if pred == 0:
                pass_correct += 1
        else:
            fail_total += 1
            if pred == 1:
                fail_correct += 1

    pass_recall = safe_ratio(pass_correct, pass_total)
    fail_recall = safe_ratio(fail_correct, fail_total)
    balanced = None
    if pass_recall is not None and fail_recall is not None:
        balanced = (pass_recall + fail_recall) / 2

    return {
        "threshold": threshold,
        "low_threshold": low,
        "high_threshold": threshold,
        "total": len(items),
        "coverage": covered / len(items) if items else 0.0,
        "balanced_selective_accuracy": balanced,
        "pass_recall": pass_recall,
        "fail_recall": fail_recall,
        "predicted_pass": pred_pass,
        "predicted_fail": pred_fail,
        "inconclusive": inconclusive,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {0: "#6b7280", 10: "#2563eb", 20: "#dc2626"}
    markers = {0: "o", 10: "s", 20: "^"}
    all_rows: list[dict] = []

    plt.figure(figsize=(7.5, 5.4))
    for examples in EXAMPLES:
        items = load_items(examples)
        rows = []
        for step in range(50, 101):
            row = evaluate(items, step / 100)
            row["examples"] = examples
            rows.append(row)
            all_rows.append(row)
        points = [
            row
            for row in rows
            if row["balanced_selective_accuracy"] is not None and row["coverage"] > 0
        ]
        points.sort(key=lambda row: row["coverage"])
        plt.plot(
            [row["coverage"] for row in points],
            [row["balanced_selective_accuracy"] for row in points],
            marker=markers[examples],
            linewidth=2,
            color=colors[examples],
            label=f"{examples} examples",
        )

    plt.xlabel("Coverage (non-INCONCLUSIVE rate)")
    plt.ylabel("Balanced selective accuracy")
    plt.title("Symmetric Threshold Balanced Selective Accuracy vs Coverage")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    png = FIG_DIR / "synthetic_kaggle_qwen36_symmetric_balanced_selective_accuracy_vs_coverage.png"
    pdf = FIG_DIR / "synthetic_kaggle_qwen36_symmetric_balanced_selective_accuracy_vs_coverage.pdf"
    csv_path = FIG_DIR / "synthetic_kaggle_qwen36_symmetric_balanced_selective_accuracy_vs_coverage.csv"
    plt.savefig(png, dpi=180)
    plt.savefig(pdf)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv_path}")
    for examples in EXAMPLES:
        valid = [
            row
            for row in all_rows
            if row["examples"] == examples and row["balanced_selective_accuracy"] is not None
        ]
        best = max(valid, key=lambda row: row["balanced_selective_accuracy"])
        print(
            f"{examples} examples: best balanced_acc={best['balanced_selective_accuracy']:.3f} "
            f"coverage={best['coverage']:.3f} t={best['threshold']:.2f} "
            f"pass_recall={best['pass_recall']:.3f} fail_recall={best['fail_recall']:.3f}"
        )


if __name__ == "__main__":
    main()
