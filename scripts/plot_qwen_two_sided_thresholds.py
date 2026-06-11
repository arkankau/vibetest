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
                    if prop not in labels or not isinstance(raw_score, (int, float)):
                        continue
                    items.append(Item(label=1 if int(labels[prop]) else 0, score=float(raw_score)))
    return items


def class_f1(tp: int, fp: int, fn: int) -> float | None:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return None
    return 2 * tp / denom


def evaluate(items: list[Item], low: float, high: float) -> dict:
    fail_tp = fail_fp = fail_fn = 0
    pass_tp = pass_fp = pass_fn = 0
    covered = 0
    pred_pass = pred_fail = inconclusive = 0

    for item in items:
        if item.score <= low:
            pred = 0
            pred_pass += 1
        elif item.score >= high:
            pred = 1
            pred_fail += 1
        else:
            inconclusive += 1
            continue

        covered += 1
        if pred == 1 and item.label == 1:
            fail_tp += 1
        elif pred == 1 and item.label == 0:
            fail_fp += 1
            pass_fn += 1
        elif pred == 0 and item.label == 0:
            pass_tp += 1
        elif pred == 0 and item.label == 1:
            pass_fp += 1
            fail_fn += 1

    fail_f1 = class_f1(fail_tp, fail_fp, fail_fn)
    pass_f1 = class_f1(pass_tp, pass_fp, pass_fn)
    covered_macro_f1 = None
    if fail_f1 is not None and pass_f1 is not None:
        covered_macro_f1 = (fail_f1 + pass_f1) / 2

    # Abstention-penalized: abstentions count as incorrect for both relevant class recall.
    total_fail = sum(1 for item in items if item.label == 1)
    total_pass = len(items) - total_fail
    abstain_fail = total_fail - (fail_tp + pass_fp)
    abstain_pass = total_pass - (pass_tp + fail_fp)
    end_fail_f1 = class_f1(fail_tp, fail_fp, fail_fn + abstain_fail)
    end_pass_f1 = class_f1(pass_tp, pass_fp, pass_fn + abstain_pass)
    macro_f1 = None
    if end_fail_f1 is not None and end_pass_f1 is not None:
        macro_f1 = (end_fail_f1 + end_pass_f1) / 2

    return {
        "low_threshold": low,
        "high_threshold": high,
        "total": len(items),
        "coverage": covered / len(items) if items else 0.0,
        "inconclusive_rate": inconclusive / len(items) if items else 0.0,
        "predicted_pass": pred_pass,
        "predicted_fail": pred_fail,
        "inconclusive": inconclusive,
        "covered_macro_f1": covered_macro_f1,
        "macro_f1": macro_f1,
        "fail_f1": fail_f1,
        "pass_f1": pass_f1,
    }


def frontier(rows: list[dict], metric: str) -> list[dict]:
    valid = [row for row in rows if row[metric] is not None]
    # Keep the best metric value for each rounded coverage level, then the
    # non-dominated upper envelope.
    best_by_cov: dict[float, dict] = {}
    for row in valid:
        cov = round(float(row["coverage"]), 4)
        prev = best_by_cov.get(cov)
        if prev is None or row[metric] > prev[metric]:
            best_by_cov[cov] = row
    out: list[dict] = []
    best_so_far = -1.0
    for row in sorted(best_by_cov.values(), key=lambda r: r["coverage"]):
        if row[metric] >= best_so_far - 1e-12:
            out.append(row)
            best_so_far = max(best_so_far, row[metric])
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    curves: dict[str, list[dict]] = {}

    grid = [i / 100 for i in range(0, 101)]
    for examples in EXAMPLES:
        items = load_items(examples)
        rows = []
        for low in grid:
            for high in grid:
                if low >= high:
                    continue
                row = evaluate(items, low, high)
                row["series"] = f"{examples} examples"
                rows.append(row)
                all_rows.append(row)
        curves[f"{examples} examples"] = rows

    csv_path = FIG_DIR / "synthetic_kaggle_qwen36_case_score_bands_two_sided_thresholds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(all_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    colors = {"0 examples": "#6b7280", "10 examples": "#2563eb", "20 examples": "#dc2626"}
    markers = {"0 examples": "o", "10 examples": "s", "20 examples": "^"}

    for metric, ylabel, title, suffix in (
        (
            "covered_macro_f1",
            "Macro F1",
            "Two-Sided Selective F1 vs Coverage",
            "covered_macro_f1_vs_coverage",
        ),
        (
            "macro_f1",
            "Abstention-penalized macro F1",
            "Two-Sided End-to-End F1 vs Coverage",
            "macro_f1_vs_coverage",
        ),
    ):
        plt.figure(figsize=(7.5, 5.4))
        for label, rows in curves.items():
            pts = frontier(rows, metric)
            pts = sorted(pts, key=lambda row: row["coverage"])
            plt.plot(
                [row["coverage"] for row in pts],
                [row[metric] for row in pts],
                marker=markers[label],
                linewidth=2,
                label=label,
                color=colors[label],
            )
        plt.xlabel("Coverage (non-INCONCLUSIVE rate)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        out = FIG_DIR / f"synthetic_kaggle_qwen36_case_score_bands_two_sided_{suffix}.png"
        plt.savefig(out, dpi=180)
        plt.savefig(out.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote {out}")

        plt.figure(figsize=(7.5, 5.4))
        for label, rows in curves.items():
            valid = [row for row in rows if row[metric] is not None]
            plt.scatter(
                [row["coverage"] for row in valid],
                [row[metric] for row in valid],
                s=8,
                alpha=0.14,
                color=colors[label],
                edgecolors="none",
            )
            pts = sorted(frontier(rows, metric), key=lambda row: row["coverage"])
            plt.plot(
                [row["coverage"] for row in pts],
                [row[metric] for row in pts],
                marker=markers[label],
                linewidth=2,
                label=label,
                color=colors[label],
            )
        plt.xlabel("Coverage (non-INCONCLUSIVE rate)")
        plt.ylabel(ylabel)
        plt.title(title + " (scatter + frontier)")
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        scatter_out = FIG_DIR / f"synthetic_kaggle_qwen36_case_score_bands_two_sided_{suffix}_scatter.png"
        plt.savefig(scatter_out, dpi=180)
        plt.savefig(scatter_out.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote {scatter_out}")

    print(f"Wrote {csv_path}")
    for label, rows in curves.items():
        valid = [row for row in rows if row["covered_macro_f1"] is not None]
        best = max(valid, key=lambda row: row["covered_macro_f1"])
        print(
            f"{label}: best selective_f1={best['covered_macro_f1']:.3f} "
            f"coverage={best['coverage']:.3f} low={best['low_threshold']:.2f} high={best['high_threshold']:.2f}"
        )


if __name__ == "__main__":
    main()
