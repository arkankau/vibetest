from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results" / "benchmark_summary.csv"
OUT_CSV = ROOT / "results" / "paper_headline_table.csv"
OUT_MD = ROOT / "results" / "paper_headline_table.md"


def fmt_float(value: str, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def example_count(name: str) -> str:
    for suffix in ("ex0", "ex10", "ex20"):
        if name.endswith(suffix):
            return suffix.removeprefix("ex")
    return ""


def method_label(section: str, name: str) -> str:
    if name.startswith("qwen_static"):
        return "VibeTest static"
    if name.startswith("reviewer_mode"):
        return name.replace("_", " ").title()
    if name == "traincheck":
        return "TrainCheck"
    return name


def row_for_summary(row: dict[str, str]) -> dict[str, str] | None:
    section = row["section"]
    name = row["name"]

    if section == "audit":
        return None

    if section == "synthetic":
        macro_f1 = row["best_selective_f1"]
        audit_basis = "Raw synthetic ground truth"
        notes = row["notes"]
    elif section == "real_kaggle":
        macro_f1 = row["audited_macro_f1"]
        audit_basis = "Conservative 15% fail audit"
        notes = "Full-context reaudit is sensitivity only"
    else:
        return None

    return {
        "benchmark": "Synthetic Kaggle" if section == "synthetic" else "Real Kaggle",
        "method": method_label(section, name),
        "example_count": example_count(name),
        "tests": row["tests"],
        "coverage": fmt_float(row["coverage"]),
        "macro_f1": fmt_float(macro_f1),
        "audit_basis": audit_basis,
        "notes": notes,
    }


def write_markdown(rows: list[dict[str, str]]) -> None:
    headers = [
        "Benchmark",
        "Method",
        "Examples",
        "Tests",
        "Coverage",
        "Macro F1",
        "Audit basis",
        "Notes",
    ]
    keys = [
        "benchmark",
        "method",
        "example_count",
        "tests",
        "coverage",
        "macro_f1",
        "audit_basis",
        "notes",
    ]

    lines = [
        "# Paper Headline Table",
        "",
        "Headline metrics use conservative labels. Full-context real Kaggle reaudit is excluded from the main table and reported only as sensitivity analysis.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[key] for key in keys) + " |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        rows = [row_for_summary(row) for row in csv.DictReader(f)]

    headline_rows = [row for row in rows if row is not None]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "method",
                "example_count",
                "tests",
                "coverage",
                "macro_f1",
                "audit_basis",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(headline_rows)

    write_markdown(headline_rows)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
