from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "benchmark_manifest.json"
OUT_MD = ROOT / "results" / "benchmark_summary.md"
OUT_CSV = ROOT / "results" / "benchmark_summary.csv"

sys.path.insert(0, str(ROOT / "scripts"))

from plot_synthetic_threshold_tradeoffs import _inconclusive_as_pass, _load_items, _rows_for_group  # noqa: E402
from plot_real_kaggle_qwen_f1_coverage import f1_score, summarize_result  # noqa: E402


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def as_paths(paths: list[str]) -> list[Path]:
    return [(ROOT / path).resolve() for path in paths]


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verdict(test: dict[str, Any]) -> str:
    meta = test.get("metadata") or {}
    value = str(meta.get("verdict") or "").strip().upper()
    if value:
        return value
    if test.get("passed") is True:
        return "PASS"
    if test.get("passed") is False:
        return "FAIL"
    return "INCONCLUSIVE"


def count_outputs(paths: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows = 0
    tests = 0
    score_values = []
    traincheck_ok: Counter[str] = Counter()

    for path in paths:
        for row in iter_jsonl(path):
            rows += 1
            for test in row.get("tests") or []:
                tests += 1
                counts[verdict(test)] += 1
                meta = test.get("metadata") or {}
                if "traincheck_ok" in meta:
                    traincheck_ok[str(meta.get("traincheck_ok"))] += 1
                raw_score = meta.get("case_score", meta.get("fail_support_score"))
                try:
                    score_values.append(float(raw_score))
                except (TypeError, ValueError):
                    pass

    covered = counts["PASS"] + counts["FAIL"]
    return {
        "rows": rows,
        "tests": tests,
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "inconclusive": counts["INCONCLUSIVE"],
        "coverage": covered / tests if tests else 0.0,
        "score_count": len(score_values),
        "unique_scores_rounded_2dp": len({round(score, 2) for score in score_values}),
        "traincheck_ok": dict(traincheck_ok),
    }


def synthetic_curve_summary(paths: list[Path], *, inconclusive_as_pass: bool = False) -> dict[str, Any]:
    items = []
    for path in paths:
        items.extend(_load_items(path))
    if inconclusive_as_pass:
        items = _inconclusive_as_pass(items)
    rows = _rows_for_group(
        items,
        target="gt-label",
        threshold_pass_low_score=False,
        inconclusive_band_low=None,
    )
    valid = [row for row in rows if row["covered_macro_f1"] is not None]
    if not valid:
        return {"best_selective_f1": None, "best_coverage": None, "best_threshold": None}
    best = max(valid, key=lambda row: row["covered_macro_f1"])
    return {
        "best_selective_f1": best["covered_macro_f1"],
        "best_coverage": best["coverage"],
        "best_threshold": best["threshold"],
    }


def load_real_fail_audit_rates(*, use_full_context_reaudit: bool = False) -> dict[tuple[str, int], dict[str, float]]:
    manifest = load_manifest()
    audit_csv = ROOT / manifest["real_kaggle"]["fail_audit"]
    reaudit_csv = ROOT / manifest["real_kaggle"]["false_fail_full_context_reaudit"]

    overrides: dict[tuple[str, str, str, str, str], str] = {}
    if use_full_context_reaudit and reaudit_csv.exists():
        with reaudit_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
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

    counts: dict[tuple[str, int], Counter[str]] = {}
    with audit_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], int(row["examples"]))
            counts.setdefault(key, Counter())
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


def real_kaggle_summary(examples: int, *, use_full_context_reaudit: bool = False) -> dict[str, Any]:
    rates = load_real_fail_audit_rates(use_full_context_reaudit=use_full_context_reaudit)
    agg = Counter()
    weighted_true = 0.0
    weighted_fail = 0.0
    sample_total = 0

    for dataset in ("titanic", "diabetic", "nlp"):
        summary = summarize_result(dataset, examples)
        rate = rates[(dataset, examples)]
        agg["total"] += summary["total"]
        agg["pass"] += summary["pass"]
        agg["fail"] += summary["fail"]
        agg["inconclusive"] += summary["inconclusive"]
        weighted_true += summary["fail"] * rate["true_rate"]
        weighted_fail += summary["fail"]
        sample_total += int(rate["sample_total"])

    coverage = (agg["pass"] + agg["fail"]) / agg["total"] if agg["total"] else 0.0
    true_rate = weighted_true / weighted_fail if weighted_fail else 0.0
    return {
        "tests": agg["total"],
        "pass": agg["pass"],
        "fail": agg["fail"],
        "inconclusive": agg["inconclusive"],
        "coverage": coverage,
        "audited_macro_f1": f1_score(agg["pass"], agg["fail"], true_rate),
        "audited_true_fail_rate": true_rate,
        "audit_n": sample_total,
    }


def audit_counts(path: Path, field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts[row.get(field, "")] += 1
    return counts


def add_row(rows: list[dict[str, Any]], section: str, name: str, stats: dict[str, Any], notes: str = "") -> None:
    row = {"section": section, "name": name, **stats, "notes": notes}
    rows.append(row)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_outputs(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    missing = []
    for row in flatten_manifest_paths(manifest):
        path = ROOT / row
        if not path.exists():
            missing.append(row)

    lines = [
        "# Benchmark Summary",
        "",
        "Canonical result summary generated from `results/benchmark_manifest.json`.",
        "",
        f"- Manifest: `{rel(MANIFEST)}`",
        f"- CSV: `{rel(OUT_CSV)}`",
        f"- Missing manifest files: {len(missing)}",
        "",
    ]
    if missing:
        lines.append("## Missing Files")
        lines.extend(f"- `{path}`" for path in missing)
        lines.append("")

    lines.extend(
        [
            "## Results",
            "",
            "| Section | Name | Tests | Coverage | F1 / Best F1 | Notes |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        f1 = row.get("audited_macro_f1", row.get("best_selective_f1", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("section", "")),
                    str(row.get("name", "")),
                    fmt(row.get("tests", "")),
                    fmt(row.get("coverage", row.get("best_coverage", ""))),
                    fmt(f1),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Audit Files",
            "",
            f"- Synthetic GT disagreement audit: `{manifest['synthetic']['ground_truth_audit']}`",
            f"- Real fail audit: `{manifest['real_kaggle']['fail_audit']}`",
            f"- Real full-context false-fail reaudit: `{manifest['real_kaggle']['false_fail_full_context_reaudit']}`",
            "",
            "## TrainCheck Note",
            "",
            "The canonical TrainCheck synthetic files are all `INCONCLUSIVE` in this workspace. "
            "Local smoke runs got past several Windows/runtime issues, but target notebooks can still crash before useful invariant checks, so TrainCheck should be reported as an execution-fragile attempted baseline rather than a central competitor.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def flatten_manifest_paths(value: Any) -> list[str]:
    if isinstance(value, str) and (value.endswith(".jsonl") or value.endswith(".csv")):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_manifest_paths(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_manifest_paths(item))
        return out
    return []


def main() -> None:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []

    for examples, raw_paths in manifest["synthetic"]["qwen_static_examples"].items():
        paths = as_paths(raw_paths)
        stats = count_outputs(paths)
        stats.update(synthetic_curve_summary(paths))
        add_row(rows, "synthetic", f"qwen_static_ex{examples}", stats, "raw synthetic GT")

    for mode, raw_paths in manifest["synthetic"]["reviewer_modes"].items():
        paths = as_paths(raw_paths)
        stats = count_outputs(paths)
        stats.update(synthetic_curve_summary(paths, inconclusive_as_pass=True))
        add_row(rows, "synthetic", f"reviewer_mode_{mode}", stats, "INCONCLUSIVE treated as PASS for baseline")

    traincheck_paths = as_paths(manifest["synthetic"]["traincheck"])
    traincheck_stats = count_outputs(traincheck_paths)
    add_row(rows, "synthetic", "traincheck", traincheck_stats, "all inconclusive in canonical files")

    for examples in sorted(manifest["real_kaggle"]["qwen_static_examples"], key=int):
        stats = real_kaggle_summary(int(examples), use_full_context_reaudit=False)
        add_row(
            rows,
            "real_kaggle",
            f"qwen_static_ex{examples}",
            stats,
            "conservative 15% fail audit; full-context reaudit kept as sensitivity only",
        )

    synthetic_audit = audit_counts(ROOT / manifest["synthetic"]["ground_truth_audit"], "audited_outcome")
    add_row(rows, "audit", "synthetic_gt_disagreement", {"tests": sum(synthetic_audit.values())}, dict(synthetic_audit).__repr__())
    real_audit = audit_counts(ROOT / manifest["real_kaggle"]["fail_audit"], "audited_outcome")
    add_row(rows, "audit", "real_fail_sample", {"tests": sum(real_audit.values())}, dict(real_audit).__repr__())

    write_outputs(rows, manifest)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
