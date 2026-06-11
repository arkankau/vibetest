from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
DATASETS = ("titanic", "diabetic", "nlp")
EXAMPLES = (0, 10)
RESULT_DIR = Path("results")
FIG_DIR = RESULT_DIR / "figures"


def result_path(dataset: str, examples: int) -> Path:
    return RESULT_DIR / f"kaggle_{dataset}_AT-{MODEL}-static_examples{examples}.jsonl"


def verdict(test: dict) -> str:
    meta = test.get("metadata") or {}
    value = str(meta.get("verdict") or "").strip().upper()
    if value:
        return value
    desc = str(test.get("description") or "").upper()
    if test.get("passed") is True:
        return "PASS"
    if "INCONCLUSIVE" in desc:
        return "INCONCLUSIVE"
    if "NOT APPLICABLE" in desc:
        return "NOT APPLICABLE"
    return "FAIL"


def summarize(path: Path) -> dict:
    counts: Counter[str] = Counter()
    repos = 0
    total = 0
    scores: list[float] = []
    output_tokens = 0
    input_tokens = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            repos += 1
            usage = (row.get("usage") or {}).get("usage_totals") or {}
            output_tokens += int(usage.get("output_tokens") or 0)
            input_tokens += int(usage.get("input_tokens") or 0)
            for test in row.get("tests") or []:
                total += 1
                counts[verdict(test)] += 1
                score = (test.get("metadata") or {}).get("case_score")
                if isinstance(score, (int, float)):
                    scores.append(float(score))
    covered = counts["PASS"] + counts["FAIL"]
    return {
        "repos": repos,
        "total": total,
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "inconclusive": counts["INCONCLUSIVE"],
        "not_applicable": counts["NOT APPLICABLE"],
        "coverage": covered / total if total else 0.0,
        "pass_rate": counts["PASS"] / total if total else 0.0,
        "fail_rate": counts["FAIL"] / total if total else 0.0,
        "inconclusive_rate": counts["INCONCLUSIVE"] / total if total else 0.0,
        "scores": scores,
        "out_per_prop": output_tokens / total if total else 0.0,
        "in_per_prop": input_tokens / total if total else 0.0,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for examples in EXAMPLES:
        for dataset in DATASETS:
            path = result_path(dataset, examples)
            if path.exists():
                rows.append((dataset, examples, summarize(path)))

    if not rows:
        raise SystemExit("No completed Qwen real Kaggle result files found.")

    labels = [f"{dataset}\nex={examples}" for dataset, examples, _ in rows]
    x = list(range(len(rows)))
    pass_rates = [s["pass_rate"] for _, _, s in rows]
    fail_rates = [s["fail_rate"] for _, _, s in rows]
    inc_rates = [s["inconclusive_rate"] for _, _, s in rows]
    coverages = [s["coverage"] for _, _, s in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.bar(x, pass_rates, color="#4C78A8", label="PASS")
    ax1.bar(x, fail_rates, bottom=pass_rates, color="#E45756", label="FAIL")
    bottoms = [p + f for p, f in zip(pass_rates, fail_rates, strict=True)]
    ax1.bar(x, inc_rates, bottom=bottoms, color="#BAB0AC", label="INCONCLUSIVE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of properties")
    ax1.set_title("Verdict mix")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3, frameon=False)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(x, coverages, marker="o", linewidth=2.5, color="#2F7D32")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Coverage = PASS + FAIL")
    ax2.set_title("Coverage")
    ax2.grid(axis="y", alpha=0.25)
    for xi, cov in zip(x, coverages, strict=True):
        ax2.text(xi, min(cov + 0.035, 0.98), f"{cov:.3f}", ha="center", fontsize=9)

    for examples, color in [(0, "#4C78A8"), (10, "#F58518")]:
        scores = [score for _, ex, s in rows if ex == examples for score in s["scores"]]
        if scores:
            ax3.hist(scores, bins=[i / 20 for i in range(21)], alpha=0.55, label=f"ex={examples}", color=color)
    ax3.set_xlabel("case_score")
    ax3.set_ylabel("Count")
    ax3.set_title("Case score spread")
    ax3.legend(frameon=False)
    ax3.grid(axis="y", alpha=0.25)

    out_props = [s["out_per_prop"] / 1000 for _, _, s in rows]
    ax4.bar(x, out_props, color="#6F4E7C")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel("K output tokens / property")
    ax4.set_title("Output budget")
    ax4.grid(axis="y", alpha=0.25)
    for xi, val in zip(x, out_props, strict=True):
        ax4.text(xi, val + 0.04, f"{val:.2f}K", ha="center", fontsize=9)

    fig.suptitle(
        "Real Kaggle Qwen3.6 Static: ex0 vs ex10",
        fontsize=14,
        fontweight="bold",
    )

    out = FIG_DIR / "real_kaggle_qwen36_static_ex0_vs_ex10.png"
    fig.savefig(out, dpi=180)
    print(out.resolve())

    for dataset, examples, s in rows:
        print(
            f"{dataset:8s} ex={examples:<2d} repos={s['repos']:<2d} tests={s['total']:<4d} "
            f"PASS={s['pass']:<4d} FAIL={s['fail']:<4d} INC={s['inconclusive']:<4d} "
            f"coverage={s['coverage']:.3f} out/prop={s['out_per_prop']:.1f}"
        )


if __name__ == "__main__":
    main()
