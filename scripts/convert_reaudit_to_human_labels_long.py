"""Convert a re-audit CSV into long-format human labels for analyze_evidence_verifier_human_labels.py."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_property(text: str) -> str:
    return _normalize_whitespace(text).lower()


def _label_from_row(row: dict[str, str], *, label_column: str) -> str | None:
    if label_column == "original_label":
        raw = _normalize_whitespace(str(row.get("original_label") or "")).lower()
        if raw == "correct":
            return "C"
        if raw == "incorrect":
            return "I"
        return None

    corrected = _normalize_whitespace(str(row.get("corrected_label") or "")).lower()
    if corrected == "fail":
        return "C"
    if corrected in {"pass", "inconclusive"}:
        return "I"
    return None


def convert(
    reaudit_csv: Path,
    *,
    output_csv: Path,
    method_suffix: str,
    label_column: str,
) -> int:
    rows_out: list[dict[str, str]] = []
    with reaudit_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset = _normalize_whitespace(str(row.get("dataset") or ""))
            repo_name = _normalize_whitespace(str(row.get("repo") or ""))
            prop = _normalize_property(str(row.get("test_prompt") or ""))
            label = _label_from_row(row, label_column=label_column)
            if not dataset or not repo_name or not prop or label is None:
                continue
            rows_out.append(
                {
                    "method_run_name": f"kaggle_{dataset}_{method_suffix}",
                    "repo_name": repo_name,
                    "property": prop,
                    "human_label": label,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method_run_name", "repo_name", "property", "human_label"],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reaudit-csv",
        type=Path,
        default=Path("results/human-annotations/reaudit_100_fail_pass_inconclusive.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/human-annotations/human_annotations_long_from_reaudit_original_100.csv"),
    )
    parser.add_argument("--method-suffix", default="AT-gpt-5-mini")
    parser.add_argument(
        "--label-column",
        choices=["original_label", "corrected_label"],
        default="original_label",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    count = convert(
        args.reaudit_csv,
        output_csv=args.output_csv,
        method_suffix=args.method_suffix,
        label_column=args.label_column,
    )
    print(f"Wrote {count} label row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
