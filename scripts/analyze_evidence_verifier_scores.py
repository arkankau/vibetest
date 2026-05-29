"""Analyze clean-context evidence verifier scores on synthetic VibeTest results.

The verifier score is evaluated only on predicted FAIL verdicts. A predicted
FAIL can be evaluated either against the synthetic ground-truth property label
or, preferably, against the synthetic scorer's evidence_match_grade. The
evidence-match target treats a predicted FAIL as positive only when the scorer
judged the FAIL evidence to match the ground-truth violation.

The script reports:
- baseline VibeTest precision/recall/F1 for predicting ground-truth FAIL,
- precision/recall/F1 after accepting only predicted FAILs whose verifier score
  is at least a threshold,
- AUROC/AP of verifier score for separating true-positive from false-positive
  FAIL verdicts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL|INCONCLUSIVE|NOT\s+APPLICABLE)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FailItem:
    file: str
    dataset: str
    method: str
    method_model: str
    row_index: int
    test_index: int
    property_id: str
    gt_label: int
    evidence_match_label: int
    predicted_verdict: str
    verifier_score: float | None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_verdict(raw: Any, *, passed: Any = None) -> str:
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


def _verdict_from_text(text: Any) -> str:
    match = _VERDICT_RE.search(str(text or ""))
    return _normalize_verdict(match.group(1)) if match else ""


def _predicted_verdict(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    synthetic_score = metadata.get("synthetic_score") or {}
    from_text = _verdict_from_text(test.get("description"))
    if from_text:
        return from_text
    return _normalize_verdict(
        synthetic_score.get("predicted_verdict") or metadata.get("verdict"),
        passed=test.get("passed"),
    )


def _gt_label(entry: dict[str, Any], test: dict[str, Any]) -> int | None:
    metadata = test.get("metadata") or {}
    synthetic_score = metadata.get("synthetic_score") or {}
    if synthetic_score.get("ground_truth_label") is not None:
        return 1 if int(synthetic_score["ground_truth_label"]) else 0
    property_id = str(metadata.get("property_id") or "").strip()
    labels = entry.get("ground_truth_property_labels") or {}
    if property_id and property_id in labels:
        return 1 if int(labels[property_id]) else 0
    return None


def _verifier_score(test: dict[str, Any]) -> float | None:
    metadata = test.get("metadata") or {}
    verifier = metadata.get("evidence_verifier") or {}
    raw = verifier.get("score")
    try:
        if raw is None:
            return None
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return max(0.0, min(1.0, score))


def _evidence_match_label(test: dict[str, Any]) -> int:
    metadata = test.get("metadata") or {}
    synthetic_score = metadata.get("synthetic_score") or {}
    return 1 if str(synthetic_score.get("evidence_match_grade") or "").strip().upper() == "C" else 0


def _safe_ratio(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        pos_in_tie = sum(1 for _, label in pairs[i:j] if label == 1)
        rank_sum_pos += pos_in_tie * avg_rank
        rank += j - i
        i = j

    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for label in labels if label == 1)
    if positives == 0:
        return None
    tp = 0
    fp = 0
    ap = 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        pos_in_tie = sum(1 for _, label in pairs[i:j] if label == 1)
        neg_in_tie = (j - i) - pos_in_tie
        tp += pos_in_tie
        fp += neg_in_tie
        if pos_in_tie:
            ap += (pos_in_tie / positives) * (tp / (tp + fp))
        i = j
    return ap


def _collect_items(path: Path) -> tuple[list[FailItem], dict[str, int]]:
    rows = _load_rows(path)
    items: list[FailItem] = []
    totals = {
        "tests": 0,
        "gt_fail": 0,
        "gt_pass": 0,
        "missing_gt": 0,
        "predicted_fail": 0,
        "scored_predicted_fail": 0,
    }
    for row_index, entry in enumerate(rows):
        dataset = str(entry.get("dataset") or "")
        method = str(entry.get("method") or "")
        method_model = str(entry.get("method_model") or "")
        for test_index, test in enumerate(entry.get("tests") or []):
            totals["tests"] += 1
            gt = _gt_label(entry, test)
            if gt is None:
                totals["missing_gt"] += 1
                continue
            if gt == 1:
                totals["gt_fail"] += 1
            else:
                totals["gt_pass"] += 1
            verdict = _predicted_verdict(test)
            if verdict != "FAIL":
                continue
            totals["predicted_fail"] += 1
            score = _verifier_score(test)
            if score is not None:
                totals["scored_predicted_fail"] += 1
            metadata = test.get("metadata") or {}
            items.append(
                FailItem(
                    file=str(path),
                    dataset=dataset,
                    method=method,
                    method_model=method_model,
                    row_index=row_index,
                    test_index=test_index,
                    property_id=str(metadata.get("property_id") or ""),
                    gt_label=gt,
                    evidence_match_label=_evidence_match_label(test),
                    predicted_verdict=verdict,
                    verifier_score=score,
                )
            )
    return items, totals


def _threshold_metrics(
    items: list[FailItem],
    *,
    total_positive: int,
    total_negative: int,
    threshold: float | None,
    missing_score: str,
    target: str,
) -> dict[str, Any]:
    tp = fp = 0
    for item in items:
        score = item.verifier_score
        if threshold is None:
            accept = True
        elif score is None:
            accept = missing_score == "accept"
        else:
            accept = score >= threshold
        if not accept:
            continue
        label = item.evidence_match_label if target == "evidence-match" else item.gt_label
        if label == 1:
            tp += 1
        else:
            fp += 1
    fn = total_positive - tp
    tn = total_negative - fp
    metrics = _confusion_metrics(tp, fp, fn, tn)
    metrics["threshold"] = threshold
    metrics["accepted_fail_predictions"] = tp + fp
    return metrics


def _candidate_thresholds(items: list[FailItem]) -> list[float]:
    scores = sorted({item.verifier_score for item in items if item.verifier_score is not None}, reverse=True)
    grid = {round(i / 100, 2) for i in range(0, 101)}
    grid.update(float(score) for score in scores)
    return sorted(grid, reverse=True)


def _summarize_collected(
    name: str,
    items: list[FailItem],
    totals: dict[str, int],
    *,
    default_threshold: float,
    missing_score: str,
    target: str,
) -> dict[str, Any]:
    if target == "evidence-match":
        total_positive = sum(1 for item in items if item.evidence_match_label == 1)
        total_negative = sum(1 for item in items if item.evidence_match_label == 0)
    else:
        total_positive = totals["gt_fail"]
        total_negative = totals["gt_pass"]

    baseline = _threshold_metrics(
        items,
        total_positive=total_positive,
        total_negative=total_negative,
        threshold=None,
        missing_score=missing_score,
        target=target,
    )
    at_default = _threshold_metrics(
        items,
        total_positive=total_positive,
        total_negative=total_negative,
        threshold=default_threshold,
        missing_score=missing_score,
        target=target,
    )

    best = None
    for threshold in _candidate_thresholds(items):
        metrics = _threshold_metrics(
            items,
            total_positive=total_positive,
            total_negative=total_negative,
            threshold=threshold,
            missing_score=missing_score,
            target=target,
        )
        if best is None:
            best = metrics
            continue
        f1 = metrics["f1"] if metrics["f1"] is not None else -1.0
        best_f1 = best["f1"] if best["f1"] is not None else -1.0
        if (f1, metrics["precision"] or -1.0, metrics["recall"] or -1.0) > (
            best_f1,
            best["precision"] or -1.0,
            best["recall"] or -1.0,
        ):
            best = metrics

    scored = [item for item in items if item.verifier_score is not None]
    labels = [
        item.evidence_match_label if target == "evidence-match" else item.gt_label
        for item in scored
    ]
    scores = [float(item.verifier_score) for item in scored if item.verifier_score is not None]
    auroc = _roc_auc(labels, scores) if scored else None
    ap = _average_precision(labels, scores) if scored else None

    return {
        "file": name,
        "target": target,
        **totals,
        "target_positive": total_positive,
        "target_negative": total_negative,
        "missing_verifier_scores": totals["predicted_fail"] - totals["scored_predicted_fail"],
        "baseline": baseline,
        "default_threshold": at_default,
        "best_threshold": best,
        "verifier_auroc": auroc,
        "verifier_average_precision": ap,
        "scored_positive_fail": sum(
            1
            for item in scored
            if (item.evidence_match_label if target == "evidence-match" else item.gt_label) == 1
        ),
        "scored_negative_fail": sum(
            1
            for item in scored
            if (item.evidence_match_label if target == "evidence-match" else item.gt_label) == 0
        ),
    }


def _summary_row(summary: dict[str, Any], metric_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    baseline = summary["baseline"]
    return {
        "file": summary["file"],
        "target": summary["target"],
        "metric": metric_name,
        "threshold": "" if metrics.get("threshold") is None else metrics.get("threshold"),
        "tests": summary["tests"],
        "gt_fail": summary["gt_fail"],
        "target_positive": summary["target_positive"],
        "target_negative": summary["target_negative"],
        "predicted_fail": summary["predicted_fail"],
        "scored_predicted_fail": summary["scored_predicted_fail"],
        "missing_verifier_scores": summary["missing_verifier_scores"],
        "accepted_fail_predictions": metrics["accepted_fail_predictions"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fp_reduction_vs_baseline": baseline["fp"] - metrics["fp"],
        "f1_delta_vs_baseline": None
        if baseline["f1"] is None or metrics["f1"] is None
        else metrics["f1"] - baseline["f1"],
        "verifier_auroc": summary["verifier_auroc"],
        "verifier_average_precision": summary["verifier_average_precision"],
    }


def _print_summary(summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    default = summary["default_threshold"]
    best = summary["best_threshold"]
    print(f"\n{summary['file']}")
    print(
        f"  target={summary['target']} tests={summary['tests']} gt_fail={summary['gt_fail']} "
        f"target_positive={summary['target_positive']} target_negative={summary['target_negative']} "
        f"predicted_fail={summary['predicted_fail']} scored_fail={summary['scored_predicted_fail']} "
        f"missing_scores={summary['missing_verifier_scores']}"
    )
    print(
        f"  verifier AUROC={_fmt(summary['verifier_auroc'])} "
        f"AP={_fmt(summary['verifier_average_precision'])} "
        f"(scored positives={summary['scored_positive_fail']}, scored negatives={summary['scored_negative_fail']})"
    )
    print(
        "  baseline: "
        f"P={_fmt(baseline['precision'])} R={_fmt(baseline['recall'])} F1={_fmt(baseline['f1'])} "
        f"TP={baseline['tp']} FP={baseline['fp']} FN={baseline['fn']}"
    )
    print(
        f"  threshold {default['threshold']:.3f}: "
        f"P={_fmt(default['precision'])} R={_fmt(default['recall'])} F1={_fmt(default['f1'])} "
        f"TP={default['tp']} FP={default['fp']} FN={default['fn']} "
        f"FP_reduction={baseline['fp'] - default['fp']} "
        f"F1_delta={_fmt(None if baseline['f1'] is None or default['f1'] is None else default['f1'] - baseline['f1'])}"
    )
    if best is not None:
        print(
            f"  best F1 threshold {best['threshold']:.3f}: "
            f"P={_fmt(best['precision'])} R={_fmt(best['recall'])} F1={_fmt(best['f1'])} "
            f"TP={best['tp']} FP={best['fp']} FN={best['fn']} "
            f"FP_reduction={baseline['fp'] - best['fp']} "
            f"F1_delta={_fmt(None if baseline['f1'] is None or best['f1'] is None else best['f1'] - baseline['f1'])}"
        )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="Synthetic result JSONL files with metadata.evidence_verifier scores.")
    parser.add_argument(
        "--target",
        choices=["evidence-match", "gt-label"],
        default="evidence-match",
        help="Verifier evaluation target. evidence-match treats only scorer grade C FAILs as positives; gt-label uses synthetic property labels.",
    )
    parser.add_argument("--threshold", type=float, default=0.7, help="Verifier score threshold to report.")
    parser.add_argument(
        "--missing-score",
        choices=["reject", "accept"],
        default="reject",
        help="How thresholded metrics handle predicted FAILs without verifier scores.",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV path for summary rows.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit("--threshold must be between 0 and 1.")
    summaries = []
    csv_rows = []
    all_items: list[FailItem] = []
    all_totals = {
        "tests": 0,
        "gt_fail": 0,
        "gt_pass": 0,
        "missing_gt": 0,
        "predicted_fail": 0,
        "scored_predicted_fail": 0,
    }
    for raw_path in args.results:
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"Result file not found: {path}")
        items, totals = _collect_items(path)
        summary = _summarize_collected(
            str(path),
            items,
            totals,
            default_threshold=args.threshold,
            missing_score=args.missing_score,
            target=args.target,
        )
        all_items.extend(items)
        for key, value in totals.items():
            all_totals[key] += value
        summaries.append(summary)
        _print_summary(summary)
        csv_rows.append(_summary_row(summary, "baseline", summary["baseline"]))
        csv_rows.append(_summary_row(summary, f"threshold_{args.threshold:g}", summary["default_threshold"]))
        if summary["best_threshold"] is not None:
            csv_rows.append(_summary_row(summary, "best_f1_threshold", summary["best_threshold"]))

    if len(args.results) > 1:
        combined = _summarize_collected(
            "ALL",
            all_items,
            all_totals,
            default_threshold=args.threshold,
            missing_score=args.missing_score,
            target=args.target,
        )
        _print_summary(combined)
        csv_rows.append(_summary_row(combined, "baseline", combined["baseline"]))
        csv_rows.append(_summary_row(combined, f"threshold_{args.threshold:g}", combined["default_threshold"]))
        if combined["best_threshold"] is not None:
            csv_rows.append(_summary_row(combined, "best_f1_threshold", combined["best_threshold"]))

    if args.csv:
        _write_csv(args.csv, csv_rows)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
