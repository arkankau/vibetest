from __future__ import annotations

from collections import Counter
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt


RUNS = [
    ("Titanic", 0, Path("results/kaggle_titanic_AT-gpt-5-mini-static_examples0.jsonl")),
    ("Diabetic", 0, Path("results/kaggle_diabetic_AT-gpt-5-mini-static_examples0.jsonl")),
    ("NLP", 0, Path("results/kaggle_nlp_AT-gpt-5-mini-static_examples0.jsonl")),
    ("Titanic", 10, Path("results/kaggle_titanic_AT-gpt-5-mini-static_examples10.jsonl")),
    ("Diabetic", 10, Path("results/kaggle_diabetic_AT-gpt-5-mini-static_examples10.jsonl")),
]


def verdict(test: dict) -> str:
    meta = test.get("metadata") or {}
    value = str(meta.get("verdict") or "").strip().upper()
    if value:
        return value
    desc = str(test.get("description") or "")
    if test.get("passed") is True:
        return "PASS"
    if "INCONCLUSIVE" in desc.upper():
        return "INCONCLUSIVE"
    if "NOT APPLICABLE" in desc.upper():
        return "NOT APPLICABLE"
    return "FAIL"


def summarize(path: Path) -> dict[str, float | int]:
    counts: Counter[str] = Counter()
    total = 0
    repos = 0
    with jsonlines.open(str(path)) as reader:
        for row in reader:
            repos += 1
            for test in row.get("tests") or []:
                total += 1
                counts[verdict(test)] += 1
    coverage = (counts["PASS"] + counts["FAIL"]) / total if total else 0.0
    return {
        "repos": repos,
        "total": total,
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "inconclusive": counts["INCONCLUSIVE"],
        "coverage": coverage,
        "pass_rate": counts["PASS"] / total if total else 0.0,
        "fail_rate": counts["FAIL"] / total if total else 0.0,
        "inconclusive_rate": counts["INCONCLUSIVE"] / total if total else 0.0,
    }


def main() -> None:
    rows = []
    for dataset, examples, path in RUNS:
        if not path.exists():
            continue
        rows.append((dataset, examples, summarize(path)))

    labels = [f"{dataset}\nex={examples}" for dataset, examples, _ in rows]
    pass_rates = [s["pass_rate"] for _, _, s in rows]
    fail_rates = [s["fail_rate"] for _, _, s in rows]
    inc_rates = [s["inconclusive_rate"] for _, _, s in rows]
    coverages = [s["coverage"] for _, _, s in rows]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.8),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )

    x = list(range(len(rows)))
    ax1.bar(x, pass_rates, color="#4C78A8", label="PASS")
    ax1.bar(x, fail_rates, bottom=pass_rates, color="#E45756", label="FAIL")
    bottoms = [p + f for p, f in zip(pass_rates, fail_rates, strict=True)]
    ax1.bar(x, inc_rates, bottom=bottoms, color="#BAB0AC", label="INCONCLUSIVE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of tests")
    ax1.set_title("Verdict mix for completed real Kaggle slices")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3, frameon=False)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(x, coverages, marker="o", linewidth=2.5, color="#2F7D32")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Coverage = PASS + FAIL")
    ax2.set_title("Interim coverage summary")
    ax2.grid(axis="y", alpha=0.25)
    for xi, cov in zip(x, coverages, strict=True):
        ax2.text(xi, min(cov + 0.035, 0.98), f"{cov:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "Real Kaggle static prompt-example run: interim results\n"
        "F1 is not plotted because these real Kaggle result files do not include ground-truth labels.",
        fontsize=13,
        fontweight="bold",
    )

    out = Path("results/real_kaggle_interim_coverage_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out.resolve())


if __name__ == "__main__":
    main()
