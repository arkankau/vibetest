"""Analyze evidence verifier scores against manual FAIL labels on real Kaggle results.

Manual labels use `C` for correct FAIL evidence and `I` for incorrect FAIL evidence.
Only predicted FAIL verdicts with a human label are evaluated.

The script reports:
- baseline precision among labeled predicted FAILs,
- precision after accepting only FAILs whose verifier score is at least a threshold,
- AUROC/AP of verifier score for separating correct from incorrect FAIL evidence,
- optional precision-recall curve points over verifier thresholds.
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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_property(text: str) -> str:
    return _normalize_whitespace(text).lower()


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


def _property_text(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    return _normalize_whitespace(
        str(metadata.get("property_text") or "")
        or str(metadata.get("test_description") or "")
        or str(test.get("property") or "")
    )


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


def _candidate_thresholds(scores: Iterable[float | None]) -> list[float]:
    observed = sorted({float(score) for score in scores if score is not None}, reverse=True)
    grid = {round(i / 100, 2) for i in range(0, 101)}
    grid.update(observed)
    return sorted(grid, reverse=True)


@dataclass(frozen=True)
class LabeledFailItem:
    file: str
    method_run_name: str
    repo_name: str
    property_text: str
    human_label: str
    human_correct: int
    row_index: int
    test_index: int
    verifier_score: float | None


def _load_human_labels(
    long_csv: Path,
    *,
    method_prefixes: tuple[str, ...],
) -> dict[tuple[str, str, str], str]:
    if not long_csv.exists():
        raise FileNotFoundError(f"Human annotation CSV not found: {long_csv}")

    out: dict[tuple[str, str, str], str] = {}
    with long_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method_run = _normalize_whitespace(str(row.get("method_run_name") or ""))
            if not method_run or not method_run.startswith(method_prefixes):
                continue
            label = _normalize_whitespace(str(row.get("human_label") or "")).upper()
            if label not in {"C", "I"}:
                continue
            repo_name = _normalize_whitespace(str(row.get("repo_name") or ""))
            prop = _normalize_property(str(row.get("property") or ""))
            if not repo_name or not prop:
                continue
            out[(method_run, repo_name, prop)] = label
    return out


def _reaudit_label_to_ci(
    *,
    label_column: str,
    row: dict[str, str],
    exclude_inconclusive: bool,
) -> str | None:
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
    if corrected == "pass":
        return "I"
    if corrected == "inconclusive":
        if exclude_inconclusive:
            return None
        return "I"
    return None


def _load_reaudit_labels(
    reaudit_csv: Path,
    *,
    method_suffix: str = "AT-gpt-5-mini",
    label_column: str = "corrected_label",
    exclude_inconclusive: bool = False,
) -> dict[tuple[str, str, str], str]:
    if not reaudit_csv.exists():
        raise FileNotFoundError(f"Re-audit CSV not found: {reaudit_csv}")
    if label_column not in {"corrected_label", "original_label"}:
        raise ValueError(f"Unsupported re-audit label column: {label_column}")

    out: dict[tuple[str, str, str], str] = {}
    with reaudit_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset = _normalize_whitespace(str(row.get("dataset") or ""))
            repo_name = _normalize_whitespace(str(row.get("repo") or ""))
            prop = _normalize_property(str(row.get("test_prompt") or ""))
            if not dataset or not repo_name or not prop:
                continue
            label = _reaudit_label_to_ci(
                label_column=label_column,
                row=row,
                exclude_inconclusive=exclude_inconclusive,
            )
            if label is None:
                continue
            method_run = f"kaggle_{dataset}_{method_suffix}"
            out[(method_run, repo_name, prop)] = label
    return out


def _collect_items(path: Path, human_labels: dict[tuple[str, str, str], str]) -> tuple[list[LabeledFailItem], dict[str, int]]:
    rows = _load_rows(path)
    method_run_name = path.stem.removesuffix("_verified")
    items: list[LabeledFailItem] = []
    totals = {
        "tests": 0,
        "predicted_fail": 0,
        "labeled_predicted_fail": 0,
        "correct_fail": 0,
        "incorrect_fail": 0,
        "scored_labeled_fail": 0,
        "missing_human_label": 0,
    }

    for row_index, entry in enumerate(rows):
        repo_name = _normalize_whitespace(str(entry.get("repo_name") or ""))
        for test_index, test in enumerate(entry.get("tests") or []):
            totals["tests"] += 1
            if _predicted_verdict(test) != "FAIL":
                continue
            totals["predicted_fail"] += 1

            prop_text = _property_text(test)
            prop_norm = _normalize_property(prop_text)
            human_label = human_labels.get((method_run_name, repo_name, prop_norm))
            if human_label is None:
                totals["missing_human_label"] += 1
                continue

            totals["labeled_predicted_fail"] += 1
            human_correct = 1 if human_label == "C" else 0
            if human_correct:
                totals["correct_fail"] += 1
            else:
                totals["incorrect_fail"] += 1

            score = _verifier_score(test)
            if score is not None:
                totals["scored_labeled_fail"] += 1

            items.append(
                LabeledFailItem(
                    file=str(path),
                    method_run_name=method_run_name,
                    repo_name=repo_name,
                    property_text=prop_text,
                    human_label=human_label,
                    human_correct=human_correct,
                    row_index=row_index,
                    test_index=test_index,
                    verifier_score=score,
                )
            )
    return items, totals


def _precision_metrics(items: list[LabeledFailItem], *, threshold: float | None, missing_score: str) -> dict[str, Any]:
    accepted = 0
    correct = 0
    for item in items:
        score = item.verifier_score
        if threshold is None:
            keep = True
        elif score is None:
            keep = missing_score == "accept"
        else:
            keep = score >= threshold
        if not keep:
            continue
        accepted += 1
        if item.human_correct == 1:
            correct += 1
    precision = _safe_ratio(correct, accepted)
    return {
        "threshold": threshold,
        "accepted_fail_predictions": accepted,
        "correct_fail_predictions": correct,
        "incorrect_fail_predictions": accepted - correct,
        "precision": precision,
        "recall_of_labeled_correct_fails": _safe_ratio(
            correct,
            sum(1 for item in items if item.human_correct == 1),
        ),
    }


def _precision_recall_curve(items: list[LabeledFailItem], *, missing_score: str) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for threshold in _candidate_thresholds(item.verifier_score for item in items):
        metrics = _precision_metrics(items, threshold=threshold, missing_score=missing_score)
        curve.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall_of_labeled_correct_fails"],
                "accepted_fail_predictions": metrics["accepted_fail_predictions"],
                "correct_fail_predictions": metrics["correct_fail_predictions"],
                "incorrect_fail_predictions": metrics["incorrect_fail_predictions"],
            }
        )
    return sorted(curve, key=lambda row: (row["recall"] or 0.0, row["precision"] or 0.0))


def _summarize(
    name: str,
    items: list[LabeledFailItem],
    totals: dict[str, int],
    *,
    default_threshold: float,
    missing_score: str,
) -> dict[str, Any]:
    baseline = _precision_metrics(items, threshold=None, missing_score=missing_score)
    at_default = _precision_metrics(items, threshold=default_threshold, missing_score=missing_score)
    curve = _precision_recall_curve(items, missing_score=missing_score)

    best = None
    for point in curve:
        precision = point.get("precision")
        recall = point.get("recall")
        f1 = _f1(precision, recall)
        candidate = {**point, "f1": f1}
        if best is None:
            best = candidate
            continue
        best_f1 = best.get("f1") if best.get("f1") is not None else -1.0
        candidate_f1 = f1 if f1 is not None else -1.0
        if (candidate_f1, precision or -1.0, recall or -1.0) > (
            best_f1,
            best.get("precision") or -1.0,
            best.get("recall") or -1.0,
        ):
            best = candidate

    scored = [item for item in items if item.verifier_score is not None]
    labels = [item.human_correct for item in scored]
    scores = [float(item.verifier_score) for item in scored if item.verifier_score is not None]
    auroc = _roc_auc(labels, scores) if scored else None
    ap = _average_precision(labels, scores) if scored else None

    return {
        "file": name,
        **totals,
        "missing_verifier_scores": totals["labeled_predicted_fail"] - totals["scored_labeled_fail"],
        "baseline": baseline,
        "default_threshold": at_default,
        "best_threshold": best,
        "precision_recall_curve": curve,
        "verifier_auroc": auroc,
        "verifier_average_precision": ap,
        "scored_correct_fail": sum(1 for item in scored if item.human_correct == 1),
        "scored_incorrect_fail": sum(1 for item in scored if item.human_correct == 0),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    default = summary["default_threshold"]
    best = summary["best_threshold"]
    print(f"\n{summary['file']}")
    print(
        f"  tests={summary['tests']} predicted_fail={summary['predicted_fail']} "
        f"labeled_fail={summary['labeled_predicted_fail']} "
        f"correct={summary['correct_fail']} incorrect={summary['incorrect_fail']} "
        f"scored={summary['scored_labeled_fail']} missing_scores={summary['missing_verifier_scores']}"
    )
    print(
        f"  verifier AUROC={_fmt(summary['verifier_auroc'])} "
        f"AP={_fmt(summary['verifier_average_precision'])} "
        f"(scored correct={summary['scored_correct_fail']}, scored incorrect={summary['scored_incorrect_fail']})"
    )
    print(
        "  baseline precision: "
        f"P={_fmt(baseline['precision'])} "
        f"accepted={baseline['accepted_fail_predictions']} "
        f"correct={baseline['correct_fail_predictions']} "
        f"incorrect={baseline['incorrect_fail_predictions']}"
    )
    print(
        f"  threshold {default['threshold']:.3f}: "
        f"P={_fmt(default['precision'])} "
        f"accepted={default['accepted_fail_predictions']} "
        f"correct={default['correct_fail_predictions']} "
        f"incorrect={default['incorrect_fail_predictions']} "
        f"recall_of_correct={_fmt(default['recall_of_labeled_correct_fails'])}"
    )
    if best is not None:
        print(
            f"  best F1 threshold {best['threshold']:.3f}: "
            f"P={_fmt(best['precision'])} R={_fmt(best['recall'])} F1={_fmt(best['f1'])} "
            f"accepted={best['accepted_fail_predictions']}"
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
    parser.add_argument("results", nargs="+", help="Verified VibeTest JSONL files for real Kaggle runs.")
    parser.add_argument(
        "--human-annotations-long",
        default="results/human-annotations/human_annotations_long.csv",
        help="Long-format human annotation CSV with method_run_name, repo_name, property, human_label.",
    )
    parser.add_argument(
        "--reaudit-csv",
        type=Path,
        help="Hand re-audit CSV with dataset, repo, test_prompt, corrected_label (Fail/Pass/Inconclusive).",
    )
    parser.add_argument(
        "--reaudit-method-suffix",
        default="AT-gpt-5-mini",
        help="Method suffix used to build method_run_name keys from re-audit dataset values.",
    )
    parser.add_argument(
        "--reaudit-label-column",
        choices=["corrected_label", "original_label"],
        default="corrected_label",
        help="Which re-audit column to map to C/I labels.",
    )
    parser.add_argument(
        "--exclude-inconclusive",
        action="store_true",
        help="When using --reaudit-csv with corrected_label, drop rows whose corrected_label is Inconclusive.",
    )
    parser.add_argument(
        "--method-prefix",
        default="kaggle_",
        help="Only use human labels whose method_run_name starts with this prefix.",
    )
    parser.add_argument("--threshold", type=float, default=0.7, help="Verifier score threshold to report.")
    parser.add_argument(
        "--missing-score",
        choices=["reject", "accept"],
        default="reject",
        help="How thresholded metrics handle labeled FAILs without verifier scores.",
    )
    parser.add_argument("--summary-csv", type=Path, help="Optional CSV path for summary rows.")
    parser.add_argument("--pr-curve-csv", type=Path, help="Optional CSV path for precision-recall curve points.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit("--threshold must be between 0 and 1.")

    human_labels = (
        _load_reaudit_labels(
            Path(args.reaudit_csv),
            method_suffix=args.reaudit_method_suffix,
            label_column=args.reaudit_label_column,
            exclude_inconclusive=bool(args.exclude_inconclusive),
        )
        if args.reaudit_csv
        else _load_human_labels(
            Path(args.human_annotations_long),
            method_prefixes=(args.method_prefix,),
        )
    )
    if not human_labels:
        label_source = args.reaudit_csv or args.human_annotations_long
        raise SystemExit(f"No C/I human labels found in {label_source}")

    summary_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    all_items: list[LabeledFailItem] = []
    all_totals = {
        "tests": 0,
        "predicted_fail": 0,
        "labeled_predicted_fail": 0,
        "correct_fail": 0,
        "incorrect_fail": 0,
        "scored_labeled_fail": 0,
        "missing_human_label": 0,
    }

    for raw_path in args.results:
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"Result file not found: {path}")
        items, totals = _collect_items(path, human_labels)
        summary = _summarize(
            str(path),
            items,
            totals,
            default_threshold=args.threshold,
            missing_score=args.missing_score,
        )
        all_items.extend(items)
        for key, value in totals.items():
            all_totals[key] += value
        _print_summary(summary)
        summary_rows.append(
            {
                "file": summary["file"],
                "metric": "baseline",
                "threshold": "",
                **summary["baseline"],
                "verifier_auroc": summary["verifier_auroc"],
                "verifier_average_precision": summary["verifier_average_precision"],
            }
        )
        summary_rows.append(
            {
                "file": summary["file"],
                "metric": f"threshold_{args.threshold:g}",
                "threshold": args.threshold,
                **summary["default_threshold"],
                "verifier_auroc": summary["verifier_auroc"],
                "verifier_average_precision": summary["verifier_average_precision"],
            }
        )
        if summary["best_threshold"] is not None:
            summary_rows.append(
                {
                    "file": summary["file"],
                    "metric": "best_f1_threshold",
                    "threshold": summary["best_threshold"]["threshold"],
                    "accepted_fail_predictions": summary["best_threshold"]["accepted_fail_predictions"],
                    "correct_fail_predictions": summary["best_threshold"]["correct_fail_predictions"],
                    "incorrect_fail_predictions": summary["best_threshold"]["incorrect_fail_predictions"],
                    "precision": summary["best_threshold"]["precision"],
                    "recall_of_labeled_correct_fails": summary["best_threshold"]["recall"],
                    "verifier_auroc": summary["verifier_auroc"],
                    "verifier_average_precision": summary["verifier_average_precision"],
                }
            )
        for point in summary["precision_recall_curve"]:
            pr_rows.append({"file": summary["file"], **point})

    if len(args.results) > 1:
        combined = _summarize(
            "ALL",
            all_items,
            all_totals,
            default_threshold=args.threshold,
            missing_score=args.missing_score,
        )
        _print_summary(combined)
        summary_rows.append(
            {
                "file": combined["file"],
                "metric": "baseline",
                "threshold": "",
                **combined["baseline"],
                "verifier_auroc": combined["verifier_auroc"],
                "verifier_average_precision": combined["verifier_average_precision"],
            }
        )
        summary_rows.append(
            {
                "file": combined["file"],
                "metric": f"threshold_{args.threshold:g}",
                "threshold": args.threshold,
                **combined["default_threshold"],
                "verifier_auroc": combined["verifier_auroc"],
                "verifier_average_precision": combined["verifier_average_precision"],
            }
        )
        for point in combined["precision_recall_curve"]:
            pr_rows.append({"file": combined["file"], **point})

    if args.summary_csv:
        _write_csv(args.summary_csv, summary_rows)
        print(f"\nWrote summary CSV: {args.summary_csv}")
    if args.pr_curve_csv:
        _write_csv(args.pr_curve_csv, pr_rows)
        print(f"Wrote PR curve CSV: {args.pr_curve_csv}")


if __name__ == "__main__":
    main()
