from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RESULT_DIR = Path("results/synthetic")
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
SUFFIX = "_case_score_bands"
DATASETS = ("titanic", "diabetic", "nlp")
THRESHOLDS = {
    0: 0.83,
    10: 0.95,
    20: 0.90,
}


def path_for(dataset: str, examples: int) -> Path:
    return RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples{examples}{SUFFIX}.jsonl"


def as_label(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return 1 if int(value) else 0
    except (TypeError, ValueError):
        return None


def compact(text: Any, limit: int = 600) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit] + ("..." if len(value) > limit else "")


def load_accepted_predictions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for examples, threshold in THRESHOLDS.items():
        low = 1.0 - threshold
        for dataset in DATASETS:
            result_file = path_for(dataset, examples)
            with result_file.open(encoding="utf-8", errors="replace") as handle:
                for row_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    labels = entry.get("ground_truth_property_labels") or {}
                    violation_desc = entry.get("ground_truth_violation_descriptions") or {}
                    for test_index, test in enumerate(entry.get("tests") or []):
                        meta = test.get("metadata") or {}
                        score_meta = meta.get("synthetic_score") or {}
                        prop_id = str(meta.get("property_id") or score_meta.get("property_id") or "")
                        label = as_label(score_meta.get("ground_truth_label"))
                        if label is None and prop_id in labels:
                            label = as_label(labels[prop_id])

                        raw_score = meta.get("case_score")
                        if raw_score is None:
                            raw_score = meta.get("fail_support_score")
                        if label is None or not isinstance(raw_score, (int, float)):
                            continue

                        score = max(0.0, min(1.0, float(raw_score)))
                        if score <= low:
                            prediction = 0
                        elif score >= threshold:
                            prediction = 1
                        else:
                            continue

                        qwen_correctness = "correct" if prediction == label else "wrong"
                        rows.append(
                            {
                                "examples": examples,
                                "threshold": threshold,
                                "dataset": dataset,
                                "repo_name": entry.get("repo_name") or Path(str(entry.get("repo") or "")).name,
                                "source_repo_path": entry.get("source_repo_path") or "",
                                "injected_repo_path": entry.get("injected_repo_path") or entry.get("repo") or "",
                                "row_index": row_index,
                                "property_index": test_index,
                                "property_id": prop_id,
                                "ground_truth_label": label,
                                "prediction": prediction,
                                "qwen_correctness": qwen_correctness,
                                "error_type": "NONE" if qwen_correctness == "correct" else ("FP" if prediction == 1 else "FN"),
                                "case_score": score,
                                "property": compact(
                                    meta.get("test_description")
                                    or meta.get("property_text")
                                    or score_meta.get("property")
                                    or test.get("description"),
                                    700,
                                ),
                                "ground_truth_violation_description": compact(violation_desc.get(prop_id), 800),
                                "reason": compact(meta.get("reason_text") or test.get("description"), 900),
                                "evidence": compact(meta.get("evidence_text"), 900),
                                "result_file": str(result_file),
                            }
                        )
    return rows


def main() -> None:
    rows = load_accepted_predictions()
    out = RESULT_DIR / "synthetic_qwen_accepted_predictions_for_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["examples"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} accepted predictions: {out}")
    totals: dict[tuple[int, str, str], int] = {}
    for row in rows:
        key = (int(row["examples"]), str(row["dataset"]), str(row["qwen_correctness"]))
        totals[key] = totals.get(key, 0) + 1
    for key, count in sorted(totals.items()):
        print(f"  ex={key[0]:<2} {key[1]:<8} {key[2]:<7} {count}")


if __name__ == "__main__":
    main()
