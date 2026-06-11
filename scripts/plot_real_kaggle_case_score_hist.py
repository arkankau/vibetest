"""Plot real Kaggle Qwen CASE_SCORE histograms by prompt example size."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_DIR = Path("results")
FIG_DIR = RESULT_DIR / "figures"
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
DATASETS = ("titanic", "diabetic", "nlp")
EXAMPLES = (0, 10, 20)
VERDICT_COLORS = {
    "PASS": "#16a34a",
    "INCONCLUSIVE": "#f59e0b",
    "FAIL": "#dc2626",
}


def result_path(dataset: str, examples: int) -> Path:
    return RESULT_DIR / f"kaggle_{dataset}_AT-{MODEL}-static_examples{examples}.jsonl"


def load_scores(examples: int) -> dict[str, list[float]]:
    by_verdict: dict[str, list[float]] = defaultdict(list)
    for dataset in DATASETS:
        path = result_path(dataset, examples)
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
    missing = [
        str(result_path(dataset, examples))
        for examples in EXAMPLES
        for dataset in DATASETS
        if not result_path(dataset, examples).exists()
    ]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    bins = [i / 100 for i in range(101)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)

    for ax, examples in zip(axes, EXAMPLES, strict=True):
        by_verdict = load_scores(examples)
        verdicts = ["PASS", "INCONCLUSIVE", "FAIL"]
        data = [by_verdict.get(v, []) for v in verdicts]
        colors = [VERDICT_COLORS[v] for v in verdicts]
        ax.hist(data, bins=bins, stacked=True, label=verdicts, color=colors, alpha=0.85)
        ax.axvspan(0.0, 0.2, color="#16a34a", alpha=0.06)
        ax.axvspan(0.2, 0.5, color="#f59e0b", alpha=0.05)
        ax.axvspan(0.5, 0.8, color="#f59e0b", alpha=0.08)
        ax.axvspan(0.8, 1.0, color="#dc2626", alpha=0.06)
        ax.set_title(f"{examples} examples", fontsize=14)
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
    fig.suptitle("Real Kaggle Qwen3.6 Static CASE_SCORE Spread", fontsize=17)
    fig.tight_layout(rect=(0, 0, 0.92, 1))

    png = FIG_DIR / "real_kaggle_qwen36_case_score_histogram.png"
    pdf = FIG_DIR / "real_kaggle_qwen36_case_score_histogram.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    print(png.resolve())
    print(pdf.resolve())

    for examples in EXAMPLES:
        by_verdict = load_scores(examples)
        print(
            f"ex={examples:<2} "
            + " ".join(f"{v}={len(by_verdict.get(v, []))}" for v in ("PASS", "INCONCLUSIVE", "FAIL"))
            + f" unique={len({score for vals in by_verdict.values() for score in vals})}"
        )


if __name__ == "__main__":
    main()
