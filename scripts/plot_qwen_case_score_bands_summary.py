from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_DIR = Path("results/synthetic")
FIG_DIR = RESULT_DIR / "figures"
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
SUFFIX = "_case_score_bands"
DATASETS = ("titanic", "diabetic", "nlp")
EXAMPLES = (0, 10, 20)


def load_items(dataset: str, examples: int) -> list[dict]:
    path = RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples{examples}{SUFFIX}.jsonl"
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            labels = row.get("ground_truth_property_labels") or {}
            for test in row.get("tests") or []:
                meta = test.get("metadata") or {}
                prop = meta.get("property_id")
                label = labels.get(prop)
                if label is None:
                    continue
                verdict = str(meta.get("verdict") or "UNKNOWN").upper()
                items.append(
                    {
                        "dataset": dataset,
                        "examples": examples,
                        "label": int(label),
                        "verdict": verdict,
                        "case_score": meta.get("case_score"),
                        "empty_reason": not bool(str(meta.get("reason_text") or "").strip()),
                    }
                )
    return items


def class_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def metrics(items: list[dict]) -> dict:
    counts = Counter(item["verdict"] for item in items)
    total = len(items)
    coverage = (total - counts["INCONCLUSIVE"]) / total if total else 0.0
    # Treat INCONCLUSIVE as PASS/negative for raw macro-F1, matching the earlier
    # "inconclusive as pass" convention. This is not verifier evidence-match F1.
    tp = fp = tn = fn = 0
    for item in items:
        pred = 1 if item["verdict"] == "FAIL" else 0
        label = item["label"]
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    fail_f1 = class_f1(tp, fp, fn)
    pass_f1 = class_f1(tn, fn, fp)
    scores = [float(item["case_score"]) for item in items if isinstance(item.get("case_score"), (int, float))]
    return {
        "n": total,
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "inconclusive": counts["INCONCLUSIVE"],
        "unknown": counts["UNKNOWN"],
        "coverage": coverage,
        "raw_macro_f1": (fail_f1 + pass_f1) / 2,
        "fail_f1": fail_f1,
        "pass_f1": pass_f1,
        "avg_case_score": sum(scores) / len(scores) if scores else None,
        "missing_score": total - len(scores),
        "empty_reason": sum(1 for item in items if item["empty_reason"]),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in DATASETS:
        for examples in EXAMPLES:
            row = {"dataset": dataset, "examples": examples}
            row.update(metrics(load_items(dataset, examples)))
            rows.append(row)

    csv_path = FIG_DIR / "synthetic_kaggle_qwen36_case_score_bands_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {0: "#6b7280", 10: "#2563eb", 20: "#dc2626"}
    markers = {0: "o", 10: "s", 20: "^"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for examples in EXAMPLES:
        subset = [r for r in rows if r["examples"] == examples]
        xs = list(range(len(DATASETS)))
        axes[0].plot(
            xs,
            [r["raw_macro_f1"] for r in subset],
            marker=markers[examples],
            linewidth=2,
            color=colors[examples],
            label=f"{examples} examples",
        )
        axes[1].plot(
            xs,
            [r["coverage"] for r in subset],
            marker=markers[examples],
            linewidth=2,
            color=colors[examples],
            label=f"{examples} examples",
        )

    for ax, title, ylabel in (
        (axes[0], "Raw Macro-F1", "Macro-F1"),
        (axes[1], "Coverage", "Non-inconclusive rate"),
    ):
        ax.set_title(title)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([d.title() for d in DATASETS])
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel(ylabel)

    width = 0.24
    offsets = {0: -width, 10: 0.0, 20: width}
    for examples in EXAMPLES:
        subset = [r for r in rows if r["examples"] == examples]
        xs = [i + offsets[examples] for i in range(len(DATASETS))]
        bottoms = [0] * len(subset)
        for key, label, color in (
            ("pass", "PASS", "#16a34a"),
            ("fail", "FAIL", "#dc2626"),
            ("inconclusive", "INCONCLUSIVE", "#f59e0b"),
        ):
            vals = [r[key] / r["n"] for r in subset]
            axes[2].bar(xs, vals, width, bottom=bottoms, color=color, alpha=0.85, label=label if examples == 0 else None)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
    axes[2].set_title("Verdict Mix")
    axes[2].set_xticks(range(len(DATASETS)))
    axes[2].set_xticklabels([d.title() for d in DATASETS])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of tests")
    axes[2].grid(True, axis="y", alpha=0.3)

    axes[0].legend(loc="lower left")
    axes[2].legend(loc="lower right")
    fig.suptitle("Qwen Static Synthetic Kaggle: Prompt Examples 0 vs 10 vs 20", fontsize=14)
    fig.tight_layout()

    png_path = FIG_DIR / "synthetic_kaggle_qwen36_case_score_bands_summary.png"
    pdf_path = FIG_DIR / "synthetic_kaggle_qwen36_case_score_bands_summary.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            f"{row['dataset']} ex={row['examples']} "
            f"macro_f1={row['raw_macro_f1']:.3f} coverage={row['coverage']:.3f} "
            f"PASS={row['pass']} FAIL={row['fail']} INC={row['inconclusive']} "
            f"missing={row['missing_score']} empty={row['empty_reason']}"
        )


if __name__ == "__main__":
    main()
