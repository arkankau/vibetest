"""Plot dual-threshold F1 vs coverage for real Kaggle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
DATASETS = ("titanic", "diabetic", "nlp")
EXAMPLES = (0, 10, 20)
RESULT_DIR = Path("results")
AUDIT_CSV = RESULT_DIR / "openrouter_fail_human_audit_sample15_gpt5mini.csv"
FIG_DIR = RESULT_DIR / "figures"


def result_path(dataset: str, examples: int) -> Path:
    return RESULT_DIR / f"kaggle_{dataset}_AT-{MODEL}-static_examples{examples}.jsonl"


def case_score(test: dict) -> float | None:
    raw = (test.get("metadata") or {}).get("case_score")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return max(0.0, min(1.0, value))


def load_full_items(examples: int) -> list[dict]:
    items: list[dict] = []
    for dataset in DATASETS:
        with result_path(dataset, examples).open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for test in row.get("tests") or []:
                    score = case_score(test)
                    if score is None:
                        continue
                    items.append({"dataset": dataset, "examples": examples, "case_score": score})
    return items


def load_audit_items(examples: int) -> list[dict]:
    out = []
    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
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


def f1_score(accepted_pass_count: float, accepted_fail_count: float, true_fail_rate: float) -> float:
    tp = accepted_fail_count * true_fail_rate
    fp = accepted_fail_count * (1.0 - true_fail_rate)
    tn = accepted_pass_count
    fn = 0.0
    fail_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    pass_f1 = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return (fail_f1 + pass_f1) / 2


def curve_for_examples(examples: int) -> list[dict]:
    full = load_full_items(examples)
    audit = load_audit_items(examples)
    total = len(full)

    thresholds = sorted({0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99})
    rows = []
    global_true_rate = sum(1 for item in audit if item["true_fail"]) / len(audit) if audit else 0.0
    for t in thresholds:
        pass_cutoff = 1.0 - t
        accepted_pass = [item for item in full if item["case_score"] <= pass_cutoff]
        accepted_fail = [item for item in full if item["case_score"] >= t]
        accepted_audit = [item for item in audit if item["case_score"] >= t]
        true_rate = (
            sum(1 for item in accepted_audit if item["true_fail"]) / len(accepted_audit)
            if accepted_audit
            else global_true_rate
        )
        coverage = (len(accepted_pass) + len(accepted_fail)) / total if total else 0.0
        rows.append(
            {
                "threshold": t,
                "pass_cutoff": pass_cutoff,
                "coverage": coverage,
                "macro_f1": f1_score(len(accepted_pass), len(accepted_fail), true_rate),
                "audit_n": len(accepted_audit),
                "true_rate": true_rate,
            }
        )
    return rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {0: "#0072B2", 10: "#E69F00", 20: "#009E73"}
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    for examples in EXAMPLES:
        rows = curve_for_examples(examples)
        xs = [row["coverage"] for row in rows]
        ys = [row["macro_f1"] for row in rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5,
            linewidth=2.5,
            color=colors[examples],
            label=f"ex={examples}",
        )
        best = max(rows, key=lambda row: row["macro_f1"])
        ax.scatter([best["coverage"]], [best["macro_f1"]], s=95, color=colors[examples], edgecolor="black", zorder=5)
        ax.annotate(
            f"ex={examples}\nt={best['threshold']:.2f}",
            (best["coverage"], best["macro_f1"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10,
        )
        print(
            f"ex={examples:<2} best_f1={best['macro_f1']:.3f} "
            f"coverage={best['coverage']:.3f} t={best['threshold']:.2f} "
            f"pass<= {best['pass_cutoff']:.2f} fail>= {best['threshold']:.2f} "
            f"audit_n_at_threshold={best['audit_n']} true_rate={best['true_rate']:.3f}"
        )

    ax.set_title("Real Kaggle Qwen3.6: Dual-Threshold Selective F1 vs Coverage", fontsize=17, pad=12)
    ax.set_xlabel("Coverage (accepted PASS + accepted FAIL rate)", fontsize=13)
    ax.set_ylabel("Macro F1 (log scale)", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out = FIG_DIR / "real_kaggle_qwen36_dual_threshold_f1_vs_coverage_log_y.png"
    fig.savefig(out, dpi=180)
    print(out.resolve())


if __name__ == "__main__":
    main()
