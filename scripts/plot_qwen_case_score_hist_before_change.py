from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_DIR = Path("results/synthetic")
FIG_DIR = RESULT_DIR / "figures"
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
DATASETS = ("titanic", "diabetic", "nlp")
SERIES = {
    "0 examples": [
        RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}_case-score.jsonl"
        for dataset in DATASETS
    ],
    "10 examples": [
        RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples10.jsonl"
        for dataset in DATASETS
    ],
    "20 examples": [
        RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples20.jsonl"
        for dataset in DATASETS
    ],
}
VERDICT_COLORS = {
    "PASS": "#16a34a",
    "INCONCLUSIVE": "#f59e0b",
    "FAIL": "#dc2626",
}


def load_scores(paths: list[Path]) -> dict[str, list[float]]:
    by_verdict: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for test in row.get("tests") or []:
                    meta = test.get("metadata") or {}
                    score = meta.get("case_score")
                    if score is None:
                        score = meta.get("fail_support_score")
                    if not isinstance(score, (int, float)):
                        continue
                    verdict = str(meta.get("verdict") or "UNKNOWN").upper()
                    if verdict not in VERDICT_COLORS:
                        continue
                    by_verdict[verdict].append(round(float(score), 2))
    return by_verdict


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for paths in SERIES.values() for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    bins = [i / 100 for i in range(101)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, (label, paths) in zip(axes, SERIES.items(), strict=True):
        by_verdict = load_scores(paths)
        verdicts = ["PASS", "INCONCLUSIVE", "FAIL"]
        data = [by_verdict.get(v, []) for v in verdicts]
        colors = [VERDICT_COLORS[v] for v in verdicts]
        ax.hist(data, bins=bins, stacked=True, label=verdicts, color=colors, alpha=0.85)
        ax.axvspan(0.0, 0.2, color="#16a34a", alpha=0.06)
        ax.axvspan(0.2, 0.5, color="#f59e0b", alpha=0.05)
        ax.axvspan(0.5, 0.8, color="#f59e0b", alpha=0.08)
        ax.axvspan(0.8, 1.0, color="#dc2626", alpha=0.06)
        ax.set_title(label)
        ax.set_xlabel("CASE_SCORE")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)
        total = sum(len(v) for v in by_verdict.values())
        unique = len({score for vals in by_verdict.values() for score in vals})
        ax.text(
            0.03,
            0.95,
            f"n={total}\nunique={unique}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
        )

    axes[0].set_ylabel("Count")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.suptitle("Qwen Static Synthetic Kaggle CASE_SCORE Spread Before Prompt Change", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.92, 1))

    png = FIG_DIR / "synthetic_kaggle_qwen36_case_score_before_change_histogram.png"
    pdf = FIG_DIR / "synthetic_kaggle_qwen36_case_score_before_change_histogram.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
