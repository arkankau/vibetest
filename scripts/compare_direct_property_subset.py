"""Compare direct-property baseline against VibeTest on a shared synthetic subset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [json.loads(line) for line in f if line.strip()]


def verdict(test: dict[str, Any]) -> str:
    raw = str((test.get("metadata") or {}).get("verdict") or "").strip().upper()
    if raw in {"PASS", "FAIL", "INCONCLUSIVE"}:
        return raw
    passed = test.get("passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "INCONCLUSIVE"


def score(results: list[dict[str, Any]], row_ids: set[int] | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    cm: Counter[tuple[str, str]] = Counter()
    total = 0
    covered = 0
    correct = 0
    for row in results:
        row_id = int(row.get("synthetic_row_index", -1))
        if row_ids is not None and row_id not in row_ids:
            continue
        gt = {str(k): int(v) for k, v in (row.get("ground_truth_property_labels") or {}).items()}
        for test in row.get("tests") or []:
            meta = test.get("metadata") or {}
            pid = str(meta.get("property_id") or "")
            if pid not in gt:
                continue
            y = "FAIL" if gt[pid] else "PASS"
            pred = verdict(test)
            total += 1
            counts[pred] += 1
            cm[(y, pred)] += 1
            if pred != "INCONCLUSIVE":
                covered += 1
            if pred == y:
                correct += 1

    def class_f1(cls: str) -> tuple[float, float, float]:
        tp = cm[(cls, cls)]
        fp = sum(v for (y, p), v in cm.items() if p == cls and y != cls)
        fn = sum(v for (y, p), v in cm.items() if y == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1

    pass_p, pass_r, pass_f1 = class_f1("PASS")
    fail_p, fail_r, fail_f1 = class_f1("FAIL")
    return {
        "total": total,
        "covered": covered,
        "coverage": covered / total if total else 0.0,
        "accuracy_all": correct / total if total else 0.0,
        "accuracy_covered": correct / covered if covered else 0.0,
        "macro_f1": (pass_f1 + fail_f1) / 2,
        "pass_precision": pass_p,
        "pass_recall": pass_r,
        "pass_f1": pass_f1,
        "fail_precision": fail_p,
        "fail_recall": fail_r,
        "fail_f1": fail_f1,
        "verdict_counts": dict(counts),
        "confusion": {f"{y}->{p}": v for (y, p), v in sorted(cm.items())},
    }


def flatten(results: list[dict[str, Any]], row_ids: set[int]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in results:
        row_id = int(row.get("synthetic_row_index", -1))
        if row_id not in row_ids:
            continue
        gt = {str(k): int(v) for k, v in (row.get("ground_truth_property_labels") or {}).items()}
        for test in row.get("tests") or []:
            meta = test.get("metadata") or {}
            pid = str(meta.get("property_id") or "")
            if pid not in gt:
                continue
            out[(row_id, pid)] = {
                "row": row_id,
                "repo": row.get("repo_name"),
                "property_id": pid,
                "gt": "FAIL" if gt[pid] else "PASS",
                "verdict": verdict(test),
                "case_score": meta.get("case_score"),
                "reason": str(meta.get("reason_text") or test.get("description") or "").replace("\n", " "),
            }
    return out


def fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--vibetest", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    direct_rows = load_jsonl(args.direct)
    row_ids = {int(r["synthetic_row_index"]) for r in direct_rows}
    rows: list[dict[str, Any]] = []
    specs = [("Direct property agent", args.direct, direct_rows)]
    for label, path in zip(args.labels, args.vibetest):
        specs.append((label, path, load_jsonl(path)))

    for label, path, data in specs:
        metrics = score(data, row_ids)
        rows.append({"method": label, "path": str(path), **metrics})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "total",
            "coverage",
            "macro_f1",
            "accuracy_all",
            "accuracy_covered",
            "fail_precision",
            "fail_recall",
            "fail_f1",
            "pass_precision",
            "pass_recall",
            "pass_f1",
            "verdict_counts",
            "confusion",
            "path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    direct_flat = flatten(direct_rows, row_ids)
    comparison_blocks: list[str] = []
    for label, _, data in specs[1:]:
        other = flatten(data, row_ids)
        disagree = []
        for key, d in sorted(direct_flat.items()):
            o = other.get(key)
            if not o:
                continue
            if d["verdict"] != o["verdict"]:
                disagree.append((key, d, o))
        comparison_blocks.append(f"### Direct vs {label}\n")
        comparison_blocks.append(f"Disagreements: {len(disagree)} / {len(direct_flat)}\n")
        comparison_blocks.append("| row | property | GT | direct | VibeTest | direct reason |")
        comparison_blocks.append("|---:|---|---|---|---|---|")
        for _, d, o in disagree[:20]:
            reason = d["reason"][:180].replace("|", "/")
            comparison_blocks.append(
                f"| {d['row']} | {d['property_id']} | {d['gt']} | {d['verdict']} | {o['verdict']} | {reason} |"
            )
        comparison_blocks.append("")

    lines = [
        "# Direct Property Baseline Subset Comparison",
        "",
        f"Subset rows: {', '.join(str(x) for x in sorted(row_ids))}",
        "",
        "| method | n | coverage | macro F1 | acc all | acc covered | FAIL P | FAIL R | FAIL F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {total} | {coverage} | {macro_f1} | {accuracy_all} | {accuracy_covered} | {fail_precision} | {fail_recall} | {fail_f1} |".format(
                method=row["method"],
                total=row["total"],
                coverage=fmt(row["coverage"]),
                macro_f1=fmt(row["macro_f1"]),
                accuracy_all=fmt(row["accuracy_all"]),
                accuracy_covered=fmt(row["accuracy_covered"]),
                fail_precision=fmt(row["fail_precision"]),
                fail_recall=fmt(row["fail_recall"]),
                fail_f1=fmt(row["fail_f1"]),
            )
        )
    lines.extend(["", *comparison_blocks])
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
