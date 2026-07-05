from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from plot_real_kaggle_qwen_f1_coverage import DATASETS, EXAMPLES, RESULT_DIR, f1_score, summarize_result


AUDIT_CSV = RESULT_DIR / "openrouter_fail_human_audit_sample15_gpt5mini.csv"
REAUDIT_CSV = RESULT_DIR / "openrouter_false_fail_reaudit_full_context.csv"
OUT = RESULT_DIR / "real_kaggle_full_context_sensitivity_summary.csv"
OUT_MD = RESULT_DIR / "real_kaggle_full_context_sensitivity_summary.md"


def load_rates(*, use_reaudit: bool) -> dict[tuple[str, int], dict[str, float]]:
    overrides: dict[tuple[str, str, str, str, str], str] = {}
    if use_reaudit and REAUDIT_CSV.exists():
        with REAUDIT_CSV.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                overrides[
                    (
                        row["dataset"],
                        row["examples"],
                        row["repo_name"],
                        row["row_index"],
                        row["property_index"],
                    )
                ] = row["audited_outcome"]

    counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    with AUDIT_CSV.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], int(row["examples"]))
            override_key = (
                row["dataset"],
                row["examples"],
                row["repo_name"],
                row["row_index"],
                row["property_index"],
            )
            counts[key][overrides.get(override_key, row["audited_outcome"])] += 1

    rates: dict[tuple[str, int], dict[str, float]] = {}
    for key, counter in counts.items():
        total = counter["TRUE_FAIL"] + counter["FALSE_FAIL"]
        rates[key] = {
            "sample_total": total,
            "true_rate": counter["TRUE_FAIL"] / total if total else 0.0,
            "false_rate": counter["FALSE_FAIL"] / total if total else 0.0,
        }
    return rates


def summarize_examples(examples: int, *, use_reaudit: bool) -> dict[str, float]:
    rates = load_rates(use_reaudit=use_reaudit)
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
    true_rate = weighted_true / weighted_fail if weighted_fail else 0.0
    return {
        "examples": examples,
        "coverage": (agg["pass"] + agg["fail"]) / agg["total"] if agg["total"] else 0.0,
        "macro_f1": f1_score(agg["pass"], agg["fail"], true_rate),
        "true_fail_rate": true_rate,
        "audit_n": sample_total,
    }


def main() -> None:
    rows = []
    for examples in EXAMPLES:
        conservative = summarize_examples(examples, use_reaudit=False)
        adjusted = summarize_examples(examples, use_reaudit=True)
        rows.append(
            {
                "examples": examples,
                "coverage": conservative["coverage"],
                "conservative_macro_f1": conservative["macro_f1"],
                "full_context_adjusted_macro_f1": adjusted["macro_f1"],
                "conservative_true_fail_rate": conservative["true_fail_rate"],
                "full_context_true_fail_rate": adjusted["true_fail_rate"],
                "audit_n": conservative["audit_n"],
                "note": "Full-context adjusted values are sensitivity analysis only, not headline metrics.",
            }
        )

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Real Kaggle Full-Context Sensitivity",
        "",
        "Full-context false-fail reaudit values are reported only as sensitivity analysis. Headline metrics use the conservative audit.",
        "",
        "| Examples | Coverage | Conservative F1 | Full-Context Adjusted F1 | Audit n |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['examples']} | {row['coverage']:.3f} | "
            f"{row['conservative_macro_f1']:.3f} | "
            f"{row['full_context_adjusted_macro_f1']:.3f} | {row['audit_n']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.resolve())
    print(OUT_MD.resolve())
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
