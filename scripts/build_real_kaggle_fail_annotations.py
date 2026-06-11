"""Create a fail-annotation CSV that adds all ex20 FAILs to the existing CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


INPUT_CSV = Path("results/real_kaggle_qwen_ex0_ex10_fail_annotations.csv")
EX20_JSONLS = {
    "titanic": Path("results/kaggle_titanic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_examples20.jsonl"),
    "diabetic": Path("results/kaggle_diabetic_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_examples20.jsonl"),
    "nlp": Path("results/kaggle_nlp_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_examples20.jsonl"),
}
OUTPUT_CSV = Path("results/real_kaggle_qwen_ex0_ex10_ex20_fail_annotations.csv")


def main() -> None:
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else [
            "audited_outcome",
            "test_prompt",
            "reason",
            "dataset",
            "examples",
            "repo_name",
            "verdict",
            "case_score",
            "evidence_strength",
            "evidence",
            "result_file",
            "row_index",
            "property_index",
        ]

    added_by_dataset: dict[str, int] = {}
    for dataset, ex20_jsonl in EX20_JSONLS.items():
        added = 0
        with ex20_jsonl.open(encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                if not line.strip():
                    continue
                entry = json.loads(line)
                repo_name = str(entry.get("repo_name") or "")
                for property_index, test in enumerate(entry.get("tests") or []):
                    metadata = test.get("metadata") or {}
                    verdict = str(metadata.get("verdict") or "").upper()
                    if verdict != "FAIL":
                        continue
                    rows.append(
                        {
                            "audited_outcome": "",
                            "test_prompt": metadata.get("test_description") or "",
                            "reason": metadata.get("reason_text") or "",
                            "dataset": dataset,
                            "examples": "20",
                            "repo_name": repo_name,
                            "verdict": "FAIL",
                            "case_score": metadata.get("case_score") or "",
                            "evidence_strength": metadata.get("evidence_strength") or "",
                            "evidence": metadata.get("evidence_text") or "",
                            "result_file": str(ex20_jsonl),
                            "row_index": str(row_index),
                            "property_index": str(property_index),
                        }
                    )
                    added += 1
        added_by_dataset[dataset] = added

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Read {len(rows) - sum(added_by_dataset.values())} existing rows from {INPUT_CSV}")
    for dataset, added in added_by_dataset.items():
        print(f"Added {added} {dataset} ex20 FAIL rows from {EX20_JSONLS[dataset]}")
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
