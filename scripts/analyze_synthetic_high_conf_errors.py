from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
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


def compact(text: Any, limit: int = 360) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit] + ("..." if len(value) > limit else "")


def load_errors() -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for examples, threshold in THRESHOLDS.items():
        low = 1.0 - threshold
        for dataset in DATASETS:
            path = path_for(dataset, examples)
            with path.open(encoding="utf-8", errors="replace") as handle:
                for row_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    labels = row.get("ground_truth_property_labels") or {}
                    violation_desc = row.get("ground_truth_violation_descriptions") or {}
                    for test_index, test in enumerate(row.get("tests") or []):
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
                            pred = 0
                        elif score >= threshold:
                            pred = 1
                        else:
                            continue
                        if pred == label:
                            continue

                        errors.append(
                            {
                                "examples": examples,
                                "threshold": threshold,
                                "dataset": dataset,
                                "repo_name": row.get("repo_name") or Path(str(row.get("repo") or "")).name,
                                "source_repo_path": row.get("source_repo_path") or "",
                                "injected_repo_path": row.get("injected_repo_path") or row.get("repo") or "",
                                "row_index": row_index,
                                "property_index": test_index,
                                "property_id": prop_id,
                                "ground_truth_label": label,
                                "prediction": pred,
                                "qwen_correctness": "wrong",
                                "error_type": "FP" if pred == 1 else "FN",
                                "case_score": score,
                                "property": compact(
                                    meta.get("test_description")
                                    or meta.get("property_text")
                                    or score_meta.get("property")
                                    or test.get("description"),
                                    520,
                                ),
                                "ground_truth_violation_description": compact(
                                    violation_desc.get(prop_id),
                                    700,
                                ),
                                "reason": compact(meta.get("reason_text") or test.get("description"), 700),
                                "evidence": compact(meta.get("evidence_text"), 700),
                                "result_file": str(path),
                            }
                        )
    return errors


def main() -> None:
    errors = load_errors()
    out = RESULT_DIR / "synthetic_qwen_high_confidence_errors_for_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0]) if errors else ["examples"])
        writer.writeheader()
        writer.writerows(errors)

    print(f"Wrote {len(errors)} high-confidence errors: {out}")
    print()
    print("By examples/error type")
    for key, count in sorted(Counter((e["examples"], e["error_type"]) for e in errors).items()):
        print(f"  ex={key[0]:<2} {key[1]}: {count}")
    print()
    print("By dataset/error type")
    for key, count in sorted(Counter((e["dataset"], e["error_type"]) for e in errors).items()):
        print(f"  {key[0]:<8} {key[1]}: {count}")
    print()
    print("Top properties by error count")
    by_prop: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for error in errors:
        by_prop[(error["property_id"], error["property"])].append(error)
    for (prop_id, prop), rows in sorted(by_prop.items(), key=lambda item: len(item[1]), reverse=True)[:12]:
        types = Counter(row["error_type"] for row in rows)
        datasets = Counter(row["dataset"] for row in rows)
        examples = Counter(row["examples"] for row in rows)
        print(f"  n={len(rows):<2} types={dict(types)} datasets={dict(datasets)} examples={dict(examples)}")
        print(f"    {prop_id}: {prop}")
    print()
    print("Representative errors")
    for error in errors[:12]:
        print(
            f"  ex={error['examples']} {error['dataset']} {error['repo_name']} "
            f"{error['error_type']} score={error['case_score']:.2f} gt={error['ground_truth_label']} pred={error['prediction']}"
        )
        print(f"    property: {error['property']}")
        print(f"    reason: {error['reason']}")


if __name__ == "__main__":
    main()
