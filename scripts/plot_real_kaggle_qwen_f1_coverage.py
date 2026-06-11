"""Plot F1 vs coverage for real Kaggle Qwen example sizes."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
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


def verdict(test: dict) -> str:
    meta = test.get("metadata") or {}
    value = str(meta.get("verdict") or "").strip().upper()
    if value:
        return value
    if test.get("passed") is True:
        return "PASS"
    if test.get("passed") is False:
        return "FAIL"
    return "INCONCLUSIVE"


def summarize_result(dataset: str, examples: int) -> dict[str, float]:
    counts: Counter[str] = Counter()
    path = result_path(dataset, examples)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for test in row.get("tests") or []:
                counts[verdict(test)] += 1
    total = sum(counts.values())
    covered = counts["PASS"] + counts["FAIL"]
    return {
        "total": total,
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "inconclusive": counts["INCONCLUSIVE"],
        "coverage": covered / total if total else 0.0,
    }


def load_false_fail_rates() -> dict[tuple[str, int], dict[str, float]]:
    counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], int(row["examples"]))
            counts[key][row["audited_outcome"]] += 1

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, counter in counts.items():
        total = counter["TRUE_FAIL"] + counter["FALSE_FAIL"]
        out[key] = {
            "sample_total": total,
            "true_rate": counter["TRUE_FAIL"] / total if total else 0.0,
            "false_rate": counter["FALSE_FAIL"] / total if total else 0.0,
        }
    return out


def f1_score(pass_count: float, fail_count: float, true_rate: float) -> float:
    tp = fail_count * true_rate
    fp = fail_count * (1.0 - true_rate)
    tn = pass_count
    fn = 0.0
    fail_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    pass_f1 = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return (fail_f1 + pass_f1) / 2


def main() -> None:
    rates = load_false_fail_rates()
    rows = []
    for examples in EXAMPLES:
        agg = Counter()
        weighted_true = 0.0
        weighted_fail = 0.0
        sample_total = 0
        for dataset in DATASETS:
            summary = summarize_result(dataset, examples)
            rate = rates[(dataset, examples)]
            agg["total"] += summary["total"]
            agg["pass"] += summary["pass"]
            agg["fail"] += summary["fail"]
            agg["inconclusive"] += summary["inconclusive"]
            weighted_true += summary["fail"] * rate["true_rate"]
            weighted_fail += summary["fail"]
            sample_total += int(rate["sample_total"])

        coverage = (agg["pass"] + agg["fail"]) / agg["total"]
        true_rate = weighted_true / weighted_fail if weighted_fail else 0.0
        macro_f1 = f1_score(agg["pass"], agg["fail"], true_rate)
        rows.append(
            {
                "examples": examples,
                "coverage": coverage,
                "macro_f1": macro_f1,
                "true_rate": true_rate,
                "sample_total": sample_total,
                "out_per_prop": 0.0,
            }
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    xs = [r["coverage"] for r in rows]
    ys = [r["macro_f1"] for r in rows]
    labels = [f"ex={r['examples']} (n={r['sample_total']})" for r in rows]

    color = "#0072B2"
    ax.plot(xs, ys, marker="o", linewidth=3, markersize=9, color=color, label="VibeTest static Qwen3.6")
    for x, y, label in zip(xs, ys, labels, strict=True):
        ax.annotate(label, (x, y), xytext=(8, 7), textcoords="offset points", fontsize=11)

    ax.set_title("Real Kaggle Qwen3.6: F1 vs Coverage", fontsize=17, pad=12)
    ax.set_xlabel("Coverage (non-INCONCLUSIVE rate)", fontsize=13)
    ax.set_ylabel("Macro F1", fontsize=13)
    ax.grid(alpha=0.25)
    ax.set_xlim(max(0.75, min(xs) - 0.025), min(0.95, max(xs) + 0.025))
    ax.set_ylim(max(0.75, min(ys) - 0.025), min(1.0, max(ys) + 0.025))
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()

    out = FIG_DIR / "real_kaggle_qwen36_f1_vs_coverage.png"
    fig.savefig(out, dpi=180)
    print(out.resolve())
    for row in rows:
        print(
            f"ex={row['examples']:<2} coverage={row['coverage']:.3f} "
            f"f1={row['macro_f1']:.3f} true_fail_rate={row['true_rate']:.3f} "
            f"audit_n={row['sample_total']}"
        )


if __name__ == "__main__":
    main()
