"""Compute synthetic precision/recall/F1 from scored synthetic result JSONL files.

Metric definition used:
- Positive class (ground truth): FAIL labels (`ground_truth_label == 1`)
- Candidate positive predictions: any predicted FAIL (`predicted_verdict == FAIL`)
- Verifier-backed TP: predicted FAIL with verifier grade `C` on a GT FAIL
- Verifier-backed FP: predicted FAIL not counted as TP (grade != `C` or GT not FAIL)
- FN: GT FAIL not counted as TP
- TN: GT PASS with predicted PASS (strict; INCONCLUSIVE is not TN)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL|INCONCLUSIVE|NOT\s+APPLICABLE)\b", re.IGNORECASE)
PRICING_USD_PER_M: dict[str, dict[str, float]] = {
    "gpt-5-mini": {
        "input_tokens": 0.25,
        "input_tokens_cache_read": 0.025,
        "input_tokens_cache_write": 0.25,
        "output_tokens": 2.00,
    },
    "gpt-5.2": {
        "input_tokens": 1.75,
        "input_tokens_cache_read": 0.175,
        "input_tokens_cache_write": 1.75,
        "output_tokens": 14.00,
    },
    "gpt-5.3-codex": {
        "input_tokens": 1.75,
        "input_tokens_cache_read": 0.175,
        "input_tokens_cache_write": 1.75,
        "output_tokens": 14.00,
    },
    "minimax-m2.5": {
        "input_tokens": 0.30,
        "input_tokens_cache_read": 0.03,
        "input_tokens_cache_write": 0.375,
        "output_tokens": 1.20,
    },
}


def _normalize_verdict(raw: Any, passed: Any = None) -> str:
    text = str(raw or "").strip().upper()
    if text in {"PASS", "FAIL", "INCONCLUSIVE", "NOT APPLICABLE"}:
        return text
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    if text.startswith("P"):
        return "PASS"
    if text.startswith("F"):
        return "FAIL"
    return "INCONCLUSIVE"


def _safe_ratio(n: int, d: int) -> float | None:
    if d <= 0:
        return None
    return n / d


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _fmt(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.4f}"


def _fmt_cost(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.6f}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _pricing_tier_for_model(model_name: str) -> str | None:
    name = (model_name or "").lower()
    if "minimax-m2.5" in name or "minimaxai/minimax-m2.5" in name:
        return "minimax-m2.5"
    if "gpt-5.3-codex" in name or "gpt-5.3" in name:
        return "gpt-5.3-codex"
    if "gpt-5.2" in name:
        return "gpt-5.2"
    if "gpt-5-mini" in name:
        return "gpt-5-mini"
    return None


def _estimate_cost_usd_from_usage_totals(usage_totals: dict[str, Any], *, model_name: str) -> float | None:
    tier = _pricing_tier_for_model(model_name)
    if tier is None:
        return None
    pricing = PRICING_USD_PER_M[tier]
    input_tokens = _to_int(usage_totals.get("input_tokens"))
    cached_input_tokens = _to_int(usage_totals.get("input_tokens_cache_read"))
    cache_write_tokens = _to_int(usage_totals.get("input_tokens_cache_write"))
    output_tokens = _to_int(usage_totals.get("output_tokens"))
    total_tokens = _to_int(usage_totals.get("total_tokens"))
    if total_tokens <= 0 and input_tokens <= 0 and output_tokens <= 0:
        return None
    uncached_input_tokens = max(input_tokens - cached_input_tokens - cache_write_tokens, 0)
    return (
        (uncached_input_tokens / 1_000_000.0) * pricing["input_tokens"]
        + (cached_input_tokens / 1_000_000.0) * pricing["input_tokens_cache_read"]
        + (cache_write_tokens / 1_000_000.0) * pricing.get("input_tokens_cache_write", pricing["input_tokens"])
        + (output_tokens / 1_000_000.0) * pricing["output_tokens"]
    )


def _entry_cost_usd(entry: dict[str, Any], *, fallback_model_name: str) -> float | None:
    usage = entry.get("usage") or {}
    explicit = _to_float(usage.get("cost_usd"))
    if explicit is not None:
        return explicit

    model_usage = usage.get("model_usage")
    if isinstance(model_usage, dict) and model_usage:
        total = 0.0
        has_any = False
        for model_name, model_totals_raw in model_usage.items():
            model_totals = model_totals_raw if isinstance(model_totals_raw, dict) else {}
            est = _estimate_cost_usd_from_usage_totals(model_totals, model_name=str(model_name))
            if est is None:
                continue
            total += est
            has_any = True
        if has_any:
            return total

    usage_totals = usage.get("usage_totals") or {}
    if isinstance(usage_totals, dict):
        return _estimate_cost_usd_from_usage_totals(usage_totals, model_name=fallback_model_name)

    return None


def _method_variant_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if stem.endswith("-dynamic") or "-dynamic_" in stem or "_dynamic" in stem:
        return "dynamic"
    return ""


def _gt_label_for_test(entry: dict[str, Any], test: dict[str, Any]) -> int | None:
    md = test.get("metadata") or {}
    score = md.get("synthetic_score") or {}
    if score.get("ground_truth_label") is not None:
        return 1 if int(score["ground_truth_label"]) else 0

    pid = str(md.get("property_id") or "").strip()
    labels = entry.get("ground_truth_property_labels") or {}
    if pid and pid in labels:
        return 1 if int(labels[pid]) else 0
    return None


def _verdict_from_text(text: Any) -> str:
    blob = str(text or "")
    m = _VERDICT_RE.search(blob)
    if not m:
        return ""
    return _normalize_verdict(m.group(1))


def _predicted_verdict(test: dict[str, Any]) -> str:
    md = test.get("metadata") or {}
    score = md.get("synthetic_score") or {}
    # Prefer explicit VERDICT in description (legacy synthetic files may have
    # metadata verdict collapsed to PASS/FAIL while description preserves INCONCLUSIVE).
    from_text = _verdict_from_text(test.get("description"))
    if from_text:
        return from_text
    raw_verdict = score.get("predicted_verdict") or md.get("verdict")
    return _normalize_verdict(raw_verdict, passed=test.get("passed"))


def _predicted_fail(test: dict[str, Any]) -> bool:
    return _predicted_verdict(test) == "FAIL"


def _is_verified_tp(test: dict[str, Any], gt_label: int) -> bool:
    if gt_label != 1:
        return False
    md = test.get("metadata") or {}
    score = md.get("synthetic_score") or {}
    verdict = _predicted_verdict(test)
    grade = str(score.get("evidence_match_grade") or "").strip().upper()
    return verdict == "FAIL" and grade == "C"


def _aggregate(path: Path) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    method_variant = _method_variant_from_path(path)

    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for entry in rows:
        key = (
            str(entry.get("domain") or ""),
            str(entry.get("dataset") or ""),
            str(entry.get("method") or ""),
            str(entry.get("method_model") or ""),
            method_variant,
        )
        g = grouped.setdefault(
            key,
            {
                "file": str(path),
                "domain": key[0],
                "dataset": key[1],
                "method": key[2],
                "method_model": key[3],
                "method_variant": key[4],
                "total_tests": 0,
                "non_inconclusive_predictions": 0,
                "fail_labels": 0,
                "fail_predictions": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "missing_gt_label": 0,
                "total_cost_usd": 0.0,
                "has_cost": False,
            },
        )

        entry_cost = _entry_cost_usd(entry, fallback_model_name=key[3])
        if entry_cost is not None:
            g["total_cost_usd"] += entry_cost
            g["has_cost"] = True

        for test in entry.get("tests") or []:
            gt = _gt_label_for_test(entry, test)
            if gt is None:
                g["missing_gt_label"] += 1
                continue

            pred_verdict = _predicted_verdict(test)
            pred_fail = pred_verdict == "FAIL"
            pred_pass = pred_verdict == "PASS"
            tp = _is_verified_tp(test, gt)
            g["total_tests"] += 1
            if pred_verdict in {"PASS", "FAIL"}:
                g["non_inconclusive_predictions"] += 1
            if gt == 1:
                g["fail_labels"] += 1
            if pred_fail:
                g["fail_predictions"] += 1

            if tp:
                g["tp"] += 1
            # Any predicted FAIL that is not a verifier-backed TP is FP.
            if pred_fail and not tp:
                g["fp"] += 1
            # Any GT FAIL that is not a verifier-backed TP is FN.
            if gt == 1 and not tp:
                g["fn"] += 1
            if gt == 0 and pred_pass:
                g["tn"] += 1

    out: list[dict[str, Any]] = []
    for g in grouped.values():
        p = _safe_ratio(g["tp"], g["tp"] + g["fp"])
        r = _safe_ratio(g["tp"], g["tp"] + g["fn"])
        f = _f1(p, r)
        response_rate = _safe_ratio(g["non_inconclusive_predictions"], g["total_tests"])
        g["precision"] = p
        g["recall"] = r
        g["f1"] = f
        g["response_rate"] = response_rate
        g["average_cost_usd"] = (g["total_cost_usd"] / g["total_tests"]) if g.get("has_cost") and g["total_tests"] > 0 else None
        if not g.get("has_cost"):
            g["total_cost_usd"] = None
        g.pop("has_cost", None)
        out.append(g)

    out.sort(
        key=lambda x: (
            x["domain"],
            x["dataset"],
            x["method"],
            x["method_model"],
            x.get("method_variant", ""),
            x["file"],
        )
    )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "domain",
        "dataset",
        "method",
        "method_model",
        "method_variant",
        "total_tests",
        "non_inconclusive_predictions",
        "fail_labels",
        "fail_predictions",
        "tp",
        "fp",
        "fn",
        "tn",
        "missing_gt_label",
        "total_cost_usd",
        "average_cost_usd",
        "response_rate",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            item = dict(row)
            item["total_cost_usd"] = _fmt_cost(item.get("total_cost_usd"))
            item["average_cost_usd"] = _fmt_cost(item.get("average_cost_usd"))
            item["response_rate"] = _fmt(item.get("response_rate"))
            item["precision"] = _fmt(item.get("precision"))
            item["recall"] = _fmt(item.get("recall"))
            item["f1"] = _fmt(item.get("f1"))
            w.writerow(item)


def _method_label(row: dict[str, Any]) -> str:
    method = str(row.get("method") or "")
    model = str(row.get("method_model") or "")
    variant = str(row.get("method_variant") or "")
    base = f"{method}-{model}" if model else method
    return f"{base}-{variant}" if variant else base


def _aggregate_for_markdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("domain") or ""),
            str(row.get("dataset") or ""),
            str(row.get("method") or ""),
            str(row.get("method_model") or ""),
            str(row.get("method_variant") or ""),
        )
        g = grouped.setdefault(
            key,
            {
                "domain": key[0],
                "dataset": key[1],
                "method": key[2],
                "method_model": key[3],
                "method_variant": key[4],
                "total_tests": 0,
                "non_inconclusive_predictions": 0,
                "fail_labels": 0,
                "fail_predictions": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "missing_gt_label": 0,
                "total_cost_usd": 0.0,
                "has_cost": False,
            },
        )
        for field in (
            "total_tests",
            "non_inconclusive_predictions",
            "fail_labels",
            "fail_predictions",
            "tp",
            "fp",
            "fn",
            "tn",
            "missing_gt_label",
        ):
            g[field] += int(row.get(field) or 0)
        row_cost = row.get("total_cost_usd")
        if row_cost is not None:
            g["total_cost_usd"] += float(row_cost)
            g["has_cost"] = True

    out: list[dict[str, Any]] = []
    for g in grouped.values():
        g["response_rate"] = _safe_ratio(g["non_inconclusive_predictions"], g["total_tests"])
        g["precision"] = _safe_ratio(g["tp"], g["tp"] + g["fp"])
        g["recall"] = _safe_ratio(g["tp"], g["tp"] + g["fn"])
        g["f1"] = _f1(g["precision"], g["recall"])
        g["average_cost_usd"] = (g["total_cost_usd"] / g["total_tests"]) if g.get("has_cost") and g["total_tests"] > 0 else None
        if not g.get("has_cost"):
            g["total_cost_usd"] = None
        out.append(g)

    out.sort(key=lambda r: (r["domain"], r["dataset"], _method_label(r)))
    return out


def _markdown_tables(rows: list[dict[str, Any]]) -> str:
    agg = _aggregate_for_markdown(rows)
    by_dataset: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in agg:
        key = (str(row["domain"]), str(row["dataset"]))
        by_dataset.setdefault(key, []).append(row)

    headers = [
        "Method",
        "Response Rate",
        "Precision",
        "Recall",
        "F1",
        "Avg Cost (USD)",
        "TP",
        "FP",
        "FN",
        "TN",
        "Fail Labels",
        "Fail Predictions",
    ]

    lines: list[str] = []
    for domain, dataset in sorted(by_dataset.keys()):
        lines.append(f"### {domain}/{dataset}")
        lines.append("")
        row_cells: list[list[str]] = []
        for row in sorted(by_dataset[(domain, dataset)], key=_method_label):
            row_cells.append(
                [
                    _method_label(row),
                    _fmt(row.get("response_rate")),
                    _fmt(row.get("precision")),
                    _fmt(row.get("recall")),
                    _fmt(row.get("f1")),
                    _fmt_cost(row.get("average_cost_usd")),
                    str(int(row.get("tp") or 0)),
                    str(int(row.get("fp") or 0)),
                    str(int(row.get("fn") or 0)),
                    str(int(row.get("tn") or 0)),
                    str(int(row.get("fail_labels") or 0)),
                    str(int(row.get("fail_predictions") or 0)),
                ]
            )

        widths = [len(h) for h in headers]
        for cells in row_cells:
            for i, cell in enumerate(cells):
                widths[i] = max(widths[i], len(cell))

        def _fmt_row(cells: list[str]) -> str:
            return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

        lines.append(_fmt_row(headers))
        lines.append(
            "| "
            + " | ".join("-" * widths[i] for i in range(len(headers)))
            + " |"
        )
        for cells in row_cells:
            lines.append(_fmt_row(cells))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _discover_results(results_dir: Path, pattern: str) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(path for path in results_dir.glob(pattern) if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        help="Optional synthetic results JSONL files. If omitted, files are auto-discovered.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/synthetic",
        help="Directory to scan when no positional results are provided.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="synthetic_*.jsonl",
        help="Glob pattern used for auto-discovery inside --results-dir.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        help="Optional markdown output path for dataset tables.",
    )
    args = parser.parse_args()

    result_paths: list[Path]
    if args.results:
        result_paths = [Path(raw) for raw in args.results]
    else:
        result_paths = _discover_results(Path(args.results_dir), args.glob)
        if not result_paths:
            raise SystemExit(
                f"No synthetic result files found in {args.results_dir} matching {args.glob}"
            )
        print(f"Auto-discovered {len(result_paths)} synthetic result file(s):")
        for path in result_paths:
            print(f"  - {path}")

    all_rows: list[dict[str, Any]] = []
    for path in result_paths:
        if not path.exists():
            raise SystemExit(f"Results file not found: {path}")
        all_rows.extend(_aggregate(path))

    if not all_rows:
        raise SystemExit("No rows to report.")

    print("=" * 100)
    print("Synthetic Metrics (positive class = GT FAIL, positives = predicted FAIL, verifier determines TP/FP)")
    print("=" * 100)
    for row in all_rows:
        print(
            f"{Path(row['file']).name} | "
            f"{row['domain']}/{row['dataset']} | "
            f"{_method_label(row)}"
        )
        print(
            f"  TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']} "
            f"| fail_labels={row['fail_labels']} fail_predictions={row['fail_predictions']} "
            f"response_rate={_fmt(row['response_rate'])} avg_cost_usd={_fmt_cost(row.get('average_cost_usd'))}"
        )
        print(
            f"  precision={_fmt(row['precision'])} recall={_fmt(row['recall'])} f1={_fmt(row['f1'])}"
        )

    markdown = _markdown_tables(all_rows)
    print("\nMarkdown tables by dataset:")
    print(markdown)

    if args.output_csv:
        out = Path(args.output_csv)
        _write_csv(out, all_rows)
        print(f"\nWrote CSV: {out}")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
        print(f"Wrote Markdown: {out_md}")


if __name__ == "__main__":
    main()
