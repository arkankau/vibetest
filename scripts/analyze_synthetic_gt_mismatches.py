from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "results" / "synthetic" / "openrouter_synthetic_gt_disagreement_audit_15pct_neutral.csv"
OUT_MD = ROOT / "results" / "synthetic" / "synthetic_gt_mismatch_analysis.md"
OUT_CSV = ROOT / "results" / "synthetic" / "synthetic_gt_mismatch_property_counts.csv"


def short(text: str, limit: int = 260) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    usable = [r for r in rows if r["audited_outcome"] != "PARSE_ERROR"]
    gt_wrong = [r for r in usable if r["audited_outcome"] == "GT_WRONG_QWEN_CORRECT"]
    qwen_wrong = [r for r in usable if r["audited_outcome"] == "GT_CORRECT_QWEN_WRONG"]

    by_dataset = Counter(r["dataset"] for r in gt_wrong)
    by_examples = Counter(r["examples"] for r in gt_wrong)
    by_property = Counter(r["property"] for r in gt_wrong)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["count", "property"])
        writer.writeheader()
        for prop, count in by_property.most_common():
            writer.writerow({"count": count, "property": prop})

    examples_by_property: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in gt_wrong:
        if len(examples_by_property[row["property"]]) < 3:
            examples_by_property[row["property"]].append(row)

    lines = [
        "# Synthetic Ground-Truth Mismatch Analysis",
        "",
        "This analysis focuses only on audited cases where the synthetic ground truth was judged wrong and the Qwen/VibeTest prediction was judged correct.",
        "",
        "## Headline",
        "",
        f"- Total audited rows: {len(rows)}",
        f"- Usable rows after parse errors: {len(usable)}",
        f"- Real Qwen misses: {len(qwen_wrong)}",
        f"- GT/property-definition mismatches: {len(gt_wrong)}",
        f"- Share of usable disagreements that are GT/property-definition mismatches: {len(gt_wrong) / len(usable):.1%}",
        "",
        "Interpretation: this is large enough that synthetic F1 should be treated as a conservative/noisy stress-test metric, not a clean measurement of model quality.",
        "",
        "## Breakdown by Dataset",
        "",
        "| Dataset | GT mismatch count |",
        "|---|---:|",
    ]
    for dataset, count in by_dataset.most_common():
        lines.append(f"| {dataset} | {count} |")

    lines.extend([
        "",
        "## Breakdown by Prompt Example Count",
        "",
        "| Examples | GT mismatch count |",
        "|---:|---:|",
    ])
    for ex, count in sorted(by_examples.items(), key=lambda item: int(item[0])):
        lines.append(f"| {ex} | {count} |")

    lines.extend([
        "",
        "## Breakdown by Property",
        "",
        "| Count | Property |",
        "|---:|---|",
    ])
    for prop, count in by_property.most_common():
        lines.append(f"| {count} | {prop} |")

    lines.extend([
        "",
        "## Representative Mismatch Patterns",
        "",
    ])
    for prop, count in by_property.most_common():
        lines.append(f"### {prop}")
        lines.append("")
        lines.append(f"Count: {count}")
        lines.append("")
        for row in examples_by_property[prop]:
            lines.append(f"- `{row['dataset']}/{row['repo_name']}` examples={row['examples']} score={row['case_score']}")
            lines.append(f"  - Synthetic label: `{row['ground_truth_label']}`; VibeTest prediction: `{row['prediction']}`")
            lines.append(f"  - Audit assessment: {short(row['ground_truth_assessment'], 420)}")
            lines.append(f"  - Evidence: {short(row['source_evidence_check'], 420)}")
        lines.append("")

    lines.extend([
        "## Paper-Relevant Framing",
        "",
        "The GT-mismatch slice should not replace the main agentic-testing story. It should explain why synthetic benchmark numbers are lower/noisier than real audited numbers.",
        "",
        "The strongest framing is:",
        "",
        "> Synthetic Kaggle is useful as a controlled stress test, but its labels sometimes encode a different interpretation of a property than the evidence standard used by VibeTest. Therefore, raw synthetic F1 is conservative and should be paired with a disagreement audit.",
        "",
    ])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
