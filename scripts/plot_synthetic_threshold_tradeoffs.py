"""Plot threshold tradeoffs for synthetic VibeTest case scores.

At each threshold, predicted FAIL outputs below the selected score are converted
to INCONCLUSIVE. By default, original PASS and INCONCLUSIVE outputs are left unchanged.
With --threshold-pass-low-score, original PASS outputs below the threshold are
also converted to INCONCLUSIVE.
With --inconclusive-band-low T1, the plotted threshold is treated as T2 and
scored PASS/FAIL rows are remapped as PASS for score <= T1, INCONCLUSIVE for
T1 < score < T2, and FAIL for score >= T2.
This makes abstention part of the plotted metric instead of only filtering a
conditional FAIL set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL|INCONCLUSIVE|NOT\s+APPLICABLE)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TestItem:
    gt: int
    verdict: str
    score: float | None
    evidence_match: bool
    dataset: str
    file: str


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


def _case_score(test: dict[str, Any]) -> float | None:
    metadata = test.get("metadata") or {}
    raw = metadata.get("case_score", metadata.get("fail_support_score"))
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return max(0.0, min(1.0, score))


def _evidence_match(test: dict[str, Any]) -> bool:
    metadata = test.get("metadata") or {}
    synthetic_score = metadata.get("synthetic_score") or {}
    return str(synthetic_score.get("evidence_match_grade") or "").strip().upper() == "C"


def _run_mode_for_path(path: Path) -> str:
    return "dynamic" if "dynamic" in path.stem else "static"


def _label_for_path(path: Path) -> str:
    name = path.stem
    name = name.replace("synthetic_kaggle_", "")
    name = name.replace("_evidence_verified", "")
    name = name.replace("_scored-gpt-5.4-mini", "")
    name = name.replace("_AT-", " / ")
    return name


def _load_items(path: Path) -> list[TestItem]:
    items: list[TestItem] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            dataset = str(entry.get("dataset") or "")
            for test in entry.get("tests") or []:
                gt = _gt_label(entry, test)
                if gt is None:
                    continue
                items.append(
                    TestItem(
                        gt=gt,
                        verdict=_predicted_verdict(test),
                        score=_case_score(test),
                        evidence_match=_evidence_match(test),
                        dataset=dataset,
                        file=str(path),
                    )
                )
    return items


def _inconclusive_as_pass(items: list[TestItem]) -> list[TestItem]:
    return [
        TestItem(
            gt=item.gt,
            verdict="PASS" if item.verdict not in {"PASS", "FAIL"} else item.verdict,
            score=item.score,
            evidence_match=item.evidence_match,
            dataset=item.dataset,
            file=item.file,
        )
        for item in items
    ]


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _thresholds(
    items: list[TestItem],
    *,
    threshold_pass_low_score: bool,
    inconclusive_band_low: float | None,
) -> list[float]:
    score_verdicts = {"FAIL", "PASS"} if threshold_pass_low_score or inconclusive_band_low is not None else {"FAIL"}
    scores = {item.score for item in items if item.verdict in score_verdicts and item.score is not None}
    if inconclusive_band_low is not None:
        scores = {score for score in scores if score >= inconclusive_band_low}
        return sorted({inconclusive_band_low, 1.0, *scores})
    return sorted({0.0, 1.0, *scores})


def _thresholded_verdict(
    item: TestItem,
    threshold: float,
    *,
    threshold_pass_low_score: bool,
    inconclusive_band_low: float | None,
) -> str:
    if inconclusive_band_low is not None and item.verdict in {"PASS", "FAIL"} and item.score is not None:
        if item.score <= inconclusive_band_low:
            return "PASS"
        if item.score < threshold:
            return "INCONCLUSIVE"
        return "FAIL"
    if item.verdict == "FAIL":
        if item.score is not None and item.score >= threshold:
            return "FAIL"
        return "INCONCLUSIVE"
    if threshold_pass_low_score and item.verdict == "PASS":
        if item.score is not None and item.score < threshold:
            return "INCONCLUSIVE"
        return "PASS"
    return item.verdict


def _rows_for_group(
    items: list[TestItem],
    *,
    target: str,
    threshold_pass_low_score: bool,
    inconclusive_band_low: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(items)
    gt_fail = sum(1 for item in items if item.gt == 1)
    gt_pass = total - gt_fail
    for threshold in _thresholds(
        items,
        threshold_pass_low_score=threshold_pass_low_score,
        inconclusive_band_low=inconclusive_band_low,
    ):
        verdicts = [
            _thresholded_verdict(
                item,
                threshold,
                threshold_pass_low_score=threshold_pass_low_score,
                inconclusive_band_low=inconclusive_band_low,
            )
            for item in items
        ]
        fail_tp = 0
        fail_fp = 0
        fail_fn = 0
        pass_tp = 0
        pass_fp = 0
        pass_fn = 0
        covered_fail_tp = 0
        covered_fail_fp = 0
        covered_fail_fn = 0
        covered_pass_tp = 0
        covered_pass_fp = 0
        covered_pass_fn = 0
        not_fail_tp = 0
        not_fail_fp = 0
        not_fail_fn = 0
        accepted_fail = 0
        predicted_pass = 0
        inconclusive = 0
        for item, verdict in zip(items, verdicts, strict=True):
            is_true_fail = item.gt == 1 and (item.evidence_match if target == "evidence-match" else True)
            is_covered = verdict in {"PASS", "FAIL"}
            if verdict == "FAIL":
                accepted_fail += 1
                if is_true_fail:
                    fail_tp += 1
                    covered_fail_tp += 1
                else:
                    fail_fp += 1
                    covered_fail_fp += 1
            if item.gt == 1 and not (verdict == "FAIL" and is_true_fail):
                fail_fn += 1
                if is_covered:
                    covered_fail_fn += 1

            if verdict == "PASS":
                predicted_pass += 1
                if item.gt == 0:
                    pass_tp += 1
                    covered_pass_tp += 1
                else:
                    pass_fp += 1
                    covered_pass_fp += 1
            elif item.gt == 0:
                pass_fn += 1
                if is_covered:
                    covered_pass_fn += 1

            if verdict not in {"PASS", "FAIL"}:
                inconclusive += 1
            if verdict == "FAIL":
                if item.gt == 0:
                    not_fail_fn += 1
            else:
                if item.gt == 0:
                    not_fail_tp += 1
                else:
                    not_fail_fp += 1

        fail_precision = _safe_ratio(fail_tp, fail_tp + fail_fp)
        fail_recall = _safe_ratio(fail_tp, fail_tp + fail_fn)
        fail_f1 = _f1(fail_precision, fail_recall)
        not_fail_precision = _safe_ratio(not_fail_tp, not_fail_tp + not_fail_fp)
        not_fail_recall = _safe_ratio(not_fail_tp, not_fail_tp + not_fail_fn)
        not_fail_f1 = _f1(not_fail_precision, not_fail_recall)
        pass_precision = _safe_ratio(pass_tp, pass_tp + pass_fp)
        pass_recall = _safe_ratio(pass_tp, pass_tp + pass_fn)
        pass_f1 = _f1(pass_precision, pass_recall)
        macro_precision = (
            None
            if fail_precision is None or pass_precision is None
            else (fail_precision + pass_precision) / 2.0
        )
        macro_recall = (
            None
            if fail_recall is None or pass_recall is None
            else (fail_recall + pass_recall) / 2.0
        )
        macro_f1 = None if fail_f1 is None or pass_f1 is None else (fail_f1 + pass_f1) / 2.0
        balanced_accuracy = macro_recall
        covered_fail_precision = _safe_ratio(covered_fail_tp, covered_fail_tp + covered_fail_fp)
        covered_fail_recall = _safe_ratio(covered_fail_tp, covered_fail_tp + covered_fail_fn)
        covered_fail_f1 = _f1(covered_fail_precision, covered_fail_recall)
        covered_pass_precision = _safe_ratio(covered_pass_tp, covered_pass_tp + covered_pass_fp)
        covered_pass_recall = _safe_ratio(covered_pass_tp, covered_pass_tp + covered_pass_fn)
        covered_pass_f1 = _f1(covered_pass_precision, covered_pass_recall)
        covered_macro_precision = (
            None
            if covered_fail_precision is None or covered_pass_precision is None
            else (covered_fail_precision + covered_pass_precision) / 2.0
        )
        covered_macro_recall = (
            None
            if covered_fail_recall is None or covered_pass_recall is None
            else (covered_fail_recall + covered_pass_recall) / 2.0
        )
        covered_macro_f1 = (
            None
            if covered_fail_f1 is None or covered_pass_f1 is None
            else (covered_fail_f1 + covered_pass_f1) / 2.0
        )
        covered_balanced_accuracy = covered_macro_recall
        correct = fail_tp + pass_tp
        covered_correct = covered_fail_tp + covered_pass_tp
        coverage = _safe_ratio(predicted_pass + accepted_fail, total)
        covered_total = predicted_pass + accepted_fail
        combined_tp = correct
        combined_fp = covered_total - correct
        combined_fn = total - correct
        combined_precision = _safe_ratio(combined_tp, combined_tp + combined_fp)
        combined_recall = _safe_ratio(combined_tp, combined_tp + combined_fn)
        combined_f1 = _f1(combined_precision, combined_recall)
        rows.append(
            {
                "threshold": threshold,
                "threshold_low": inconclusive_band_low,
                "total": total,
                "gt_fail": gt_fail,
                "gt_pass": gt_pass,
                "accepted_fail": accepted_fail,
                "predicted_pass": predicted_pass,
                "inconclusive": inconclusive,
                "coverage": coverage,
                "inconclusive_rate": _safe_ratio(inconclusive, total),
                "accuracy_abstain_wrong": _safe_ratio(correct, total),
                "balanced_accuracy": balanced_accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "covered_accuracy": _safe_ratio(covered_correct, covered_total),
                "covered_balanced_accuracy": covered_balanced_accuracy,
                "covered_macro_precision": covered_macro_precision,
                "covered_macro_recall": covered_macro_recall,
                "covered_macro_f1": covered_macro_f1,
                "combined_precision": combined_precision,
                "combined_recall": combined_recall,
                "combined_f1": combined_f1,
                "fail_precision": fail_precision,
                "fail_recall": fail_recall,
                "fail_false_positive_rate": _safe_ratio(not_fail_fn, gt_pass),
                "fail_f1": fail_f1,
                "not_fail_precision": not_fail_precision,
                "not_fail_recall": not_fail_recall,
                "not_fail_f1": not_fail_f1,
                "pass_precision": pass_precision,
                "pass_recall": pass_recall,
                "pass_f1": pass_f1,
                "covered_fail_precision": covered_fail_precision,
                "covered_fail_recall": covered_fail_recall,
                "covered_fail_f1": covered_fail_f1,
                "covered_pass_precision": covered_pass_precision,
                "covered_pass_recall": covered_pass_recall,
                "covered_pass_f1": covered_pass_f1,
                "fail_tp": fail_tp,
                "fail_fp": fail_fp,
                "fail_fn": fail_fn,
                "not_fail_tp": not_fail_tp,
                "not_fail_fp": not_fail_fp,
                "not_fail_fn": not_fail_fn,
                "combined_tp": combined_tp,
                "combined_fp": combined_fp,
                "combined_fn": combined_fn,
                "pass_tp": pass_tp,
                "pass_fp": pass_fp,
                "pass_fn": pass_fn,
                "covered_fail_tp": covered_fail_tp,
                "covered_fail_fp": covered_fail_fp,
                "covered_fail_fn": covered_fail_fn,
                "covered_pass_tp": covered_pass_tp,
                "covered_pass_fp": covered_pass_fp,
                "covered_pass_fn": covered_pass_fn,
            }
        )
    return rows


def _write_csv(path: Path, curves: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series",
        "threshold",
        "threshold_low",
        "total",
        "gt_fail",
        "gt_pass",
        "accepted_fail",
        "predicted_pass",
        "inconclusive",
        "coverage",
        "inconclusive_rate",
        "accuracy_abstain_wrong",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "covered_accuracy",
        "covered_balanced_accuracy",
        "covered_macro_precision",
        "covered_macro_recall",
        "covered_macro_f1",
        "combined_precision",
        "combined_recall",
        "combined_f1",
        "fail_precision",
        "fail_recall",
        "fail_false_positive_rate",
        "fail_f1",
        "not_fail_precision",
        "not_fail_recall",
        "not_fail_f1",
        "pass_precision",
        "pass_recall",
        "pass_f1",
        "covered_fail_precision",
        "covered_fail_recall",
        "covered_fail_f1",
        "covered_pass_precision",
        "covered_pass_recall",
        "covered_pass_f1",
        "fail_tp",
        "fail_fp",
        "fail_fn",
        "not_fail_tp",
        "not_fail_fp",
        "not_fail_fn",
        "combined_tp",
        "combined_fp",
        "combined_fn",
        "pass_tp",
        "pass_fp",
        "pass_fn",
        "covered_fail_tp",
        "covered_fail_fp",
        "covered_fail_fn",
        "covered_pass_tp",
        "covered_pass_fp",
        "covered_pass_fn",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for series, rows in curves.items():
            for row in rows:
                writer.writerow({"series": series, **row})


# Light → dark blue for 0 / 10 / 20 calibration-example static VibeTest curves.
_VIBETEST_STATIC_BLUE = {
    0: "#B3D9F2",
    10: "#1F78B4",
    20: "#08306B",
}

_VIBETEST_MIXED_GREEN = {
    10: "#66C2A4",
    20: "#006D2C",
}


def _vibetest_static_example_count(series: str) -> int | None:
    if "VibeTest static + mixed ex" in series:
        return None
    if "VibeTest static + ex20" in series:
        return 20
    if "VibeTest static + ex10" in series:
        return 10
    if series.startswith("VibeTest static"):
        return 0
    return None


def _series_color(series: str) -> str | None:
    if "VibeTest static + mixed ex20" in series:
        return _VIBETEST_MIXED_GREEN[20]
    if "VibeTest static + mixed ex10" in series:
        return _VIBETEST_MIXED_GREEN[10]
    example_count = _vibetest_static_example_count(series)
    if example_count is not None:
        return _VIBETEST_STATIC_BLUE[example_count]
    if series.startswith("Reviewer mode 0"):
        return "#DDA0C6"
    if series.startswith("Reviewer mode 1"):
        return "#CC79A7"
    if series.startswith("Reviewer mode 2"):
        return "#8F4A73"
    if series.startswith("VibeTest dynamic"):
        return "#E69F00"
    if series.startswith("TrainCheck"):
        return "#666666"
    return None


def _plot_kwargs(series: str) -> dict[str, Any]:
    color = _series_color(series)
    return {"color": color} if color else {}


def _plot_metric_vs_coverage(curves: dict[str, list[dict[str, Any]]], output: Path, *, metric: str, ylabel: str, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    all_points: list[tuple[float, float]] = []
    for series, rows in curves.items():
        points = [(row["coverage"], row[metric]) for row in rows if row["coverage"] is not None and row[metric] is not None]
        points.sort()
        all_points.extend(points)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label=series, **_plot_kwargs(series))
    plt.xlabel("Coverage (non-INCONCLUSIVE rate)")
    plt.ylabel(ylabel)
    plt.title(title)
    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        x_pad = max((max(xs) - min(xs)) * 0.08, 0.02)
        y_pad = max((max(ys) - min(ys)) * 0.12, 0.03)
        plt.xlim(max(0.0, min(xs) - x_pad), min(1.0, max(xs) + x_pad))
        plt.ylim(max(0.0, min(ys) - y_pad), min(1.0, max(ys) + y_pad))
    else:
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def _plot_precision_recall(
    curves: dict[str, list[dict[str, Any]]],
    output: Path,
    *,
    precision_metric: str,
    recall_metric: str,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    for series, rows in curves.items():
        points = []
        for row in rows:
            recall = row[recall_metric]
            precision = row[precision_metric]
            if recall is None:
                continue
            if precision is None:
                if row.get("predicted_pass", 0) == 0 and row.get("accepted_fail", 0) == 0:
                    precision = 0.0
                else:
                    continue
            points.append((recall, precision))
        points.sort()
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label=series, **_plot_kwargs(series))
    plt.xlabel("Macro recall")
    plt.ylabel("Macro precision")
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def _plot_fail_precision_recall(curves: dict[str, list[dict[str, Any]]], output: Path, *, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    for series, rows in curves.items():
        points = [
            (row["fail_recall"], row["fail_precision"])
            for row in rows
            if row["fail_recall"] is not None and row["fail_precision"] is not None
        ]
        points.sort()
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label=series, **_plot_kwargs(series))
    plt.xlabel("FAIL recall")
    plt.ylabel("FAIL precision")
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def _plot_fail_roc(curves: dict[str, list[dict[str, Any]]], output: Path, *, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    for series, rows in curves.items():
        points = [
            (row["fail_false_positive_rate"], row["fail_recall"])
            for row in rows
            if row["fail_false_positive_rate"] is not None and row["fail_recall"] is not None
        ]
        points.extend([(0.0, 0.0), (1.0, 1.0)])
        points = sorted(set(points))
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label=series, **_plot_kwargs(series))
    plt.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1, alpha=0.45)
    plt.xlabel("FAIL false positive rate on GT PASS")
    plt.ylabel("FAIL recall")
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def _plot_class_precision_recall(curves: dict[str, list[dict[str, Any]]], output: Path, *, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    class_metrics = [
        ("FAIL class (accepted FAIL)", "fail_precision", "fail_recall", "-"),
        ("not-FAIL class (PASS or INCONCLUSIVE)", "not_fail_precision", "not_fail_recall", "--"),
    ]
    for series, rows in curves.items():
        for class_label, precision_metric, recall_metric, linestyle in class_metrics:
            points = [
                (row[recall_metric], row[precision_metric])
                for row in rows
                if row[recall_metric] is not None and row[precision_metric] is not None
            ]
            points.sort()
            plt.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                linestyle=linestyle,
                linewidth=2,
                label=f"{series}: {class_label}",
                **_plot_kwargs(series),
            )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def _plot_metrics_vs_threshold(curves: dict[str, list[dict[str, Any]]], output: Path, *, metrics: list[tuple[str, str]], title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.8, 5.4))
    linestyles = ["-", "--", ":", "-."]
    for series, rows in curves.items():
        for idx, (metric, label) in enumerate(metrics):
            points = [(row["threshold"], row[metric]) for row in rows if row[metric] is not None]
            points.sort()
            plt.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2,
                label=f"{series} {label}",
                **_plot_kwargs(series),
            )
    plt.xlabel("FAIL score threshold")
    plt.ylabel("Metric")
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def _plot_decision_rates(curves: dict[str, list[dict[str, Any]]], output: Path, *, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.8, 5.4))
    metrics = [
        ("accepted_fail", "accepted FAIL", "-"),
        ("predicted_pass", "PASS", "--"),
        ("inconclusive", "INCONCLUSIVE", ":"),
    ]
    for series, rows in curves.items():
        for metric, label, linestyle in metrics:
            points = [
                (row["threshold"], row[metric] / row["total"])
                for row in rows
                if row["total"]
            ]
            points.sort()
            plt.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                linestyle=linestyle,
                linewidth=2,
                label=f"{series} {label}",
                **_plot_kwargs(series),
            )
    plt.xlabel("FAIL score threshold")
    plt.ylabel("Fraction of tests")
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def _parse_series_spec(spec: str) -> tuple[str, list[Path]]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "--series must have the form LABEL=path1,path2,..."
        )
    label, paths_raw = spec.split("=", 1)
    label = label.strip()
    paths = [Path(p.strip()) for p in paths_raw.split(",") if p.strip()]
    if not label:
        raise argparse.ArgumentTypeError("--series label cannot be empty")
    if not paths:
        raise argparse.ArgumentTypeError("--series must include at least one path")
    return label, paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path, help="Synthetic result JSONL files.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Prefix for output files.")
    parser.add_argument(
        "--series",
        action="append",
        type=_parse_series_spec,
        default=[],
        metavar="LABEL=PATH[,PATH...]",
        help="Named curve series. When provided, positional results/group-by are ignored.",
    )
    parser.add_argument(
        "--target",
        choices=["evidence-match", "gt-label"],
        default="evidence-match",
        help="Whether FAIL correctness requires scorer evidence match or only the GT FAIL label.",
    )
    parser.add_argument(
        "--group-by",
        choices=["file", "run-mode"],
        default="run-mode",
        help="Curve grouping. run-mode combines files into static/dynamic curves based on filename.",
    )
    parser.add_argument(
        "--threshold-pass-low-score",
        action="store_true",
        help="Also convert original PASS outputs with case_score below the threshold to INCONCLUSIVE.",
    )
    parser.add_argument(
        "--inconclusive-band-low",
        type=float,
        help=(
            "Use two-sided score thresholds. The sweep threshold is T2: scored "
            "PASS/FAIL rows with score <= T1 become PASS, T1 < score < T2 "
            "become INCONCLUSIVE, and score >= T2 become FAIL."
        ),
    )
    parser.add_argument(
        "--inconclusive-as-pass-series",
        action="append",
        default=[],
        metavar="LABEL",
        help="For the named series only, treat INCONCLUSIVE/NOT APPLICABLE outputs as PASS before thresholding.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.inconclusive_band_low is not None and not (0.0 <= args.inconclusive_band_low < 1.0):
        raise SystemExit("--inconclusive-band-low must be >= 0 and < 1.")
    grouped: dict[str, list[TestItem]] = {}
    if args.series:
        for label, paths in args.series:
            for path in paths:
                if not path.exists():
                    raise SystemExit(f"Result file not found: {path}")
                grouped.setdefault(label, []).extend(_load_items(path))
    else:
        if not args.results:
            raise SystemExit("Provide result files or at least one --series.")
        for path in args.results:
            if not path.exists():
                raise SystemExit(f"Result file not found: {path}")
            label = _label_for_path(path) if args.group_by == "file" else _run_mode_for_path(path)
            grouped.setdefault(label, []).extend(_load_items(path))

    inconclusive_as_pass_series = set(args.inconclusive_as_pass_series or [])
    unknown_remap_labels = inconclusive_as_pass_series - set(grouped)
    if unknown_remap_labels:
        raise SystemExit(
            "--inconclusive-as-pass-series label(s) not found: "
            + ", ".join(sorted(unknown_remap_labels))
        )
    curves = {}
    for label, items in grouped.items():
        if label in inconclusive_as_pass_series:
            items = _inconclusive_as_pass(items)
        curves[label] = _rows_for_group(
            items,
            target=args.target,
            threshold_pass_low_score=args.threshold_pass_low_score,
            inconclusive_band_low=args.inconclusive_band_low,
        )
    csv_path = args.output_prefix.with_suffix(".csv")
    _write_csv(csv_path, curves)
    _plot_precision_recall(
        curves,
        args.output_prefix.with_name(args.output_prefix.name + "_macro_precision_recall.png"),
        precision_metric="macro_precision",
        recall_metric="macro_recall",
        title="Synthetic Kaggle Qwen3.6: Macro Precision-Recall",
    )
    _plot_fail_precision_recall(
        curves,
        args.output_prefix.with_name(args.output_prefix.name + "_fail_precision_recall.png"),
        title="Synthetic Kaggle Qwen3.6: FAIL Precision-Recall",
    )
    _plot_fail_roc(
        curves,
        args.output_prefix.with_name(args.output_prefix.name + "_fail_roc.png"),
        title="Synthetic Kaggle Qwen3.6: FAIL ROC",
    )
    _plot_metric_vs_coverage(
        curves,
        args.output_prefix.with_name(args.output_prefix.name + "_macro_f1_vs_coverage.png"),
        metric="macro_f1",
        ylabel="Abstention-penalized macro F1",
        title="Synthetic Kaggle Qwen3.6: End-to-End F1 vs Coverage",
    )
    _plot_metric_vs_coverage(
        curves,
        args.output_prefix.with_name(args.output_prefix.name + "_covered_macro_f1_vs_coverage.png"),
        metric="covered_macro_f1",
        ylabel="Macro F1",
        title="Synthetic Kaggle Qwen3.6: Selective F1 vs Coverage",
    )
    _plot_metric_vs_coverage(
        curves,
        args.output_prefix.with_name(args.output_prefix.name + "_selective_balanced_accuracy_vs_coverage.png"),
        metric="covered_balanced_accuracy",
        ylabel="Selective balanced accuracy",
        title="Synthetic Kaggle Qwen3.6: Selective Reliability vs Coverage",
    )

    print(f"Wrote CSV: {csv_path}")
    print(
        "Wrote plot: "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_macro_precision_recall.png')}"
    )
    print(
        "Wrote plot: "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_fail_precision_recall.png')}"
    )
    print(
        "Wrote plot: "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_fail_roc.png')}"
    )
    print(
        "Wrote plot: "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_macro_f1_vs_coverage.png')}"
    )
    print(
        "Wrote plot: "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_covered_macro_f1_vs_coverage.png')}"
    )
    print(
        "Wrote plot: "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_selective_balanced_accuracy_vs_coverage.png')}"
    )
    def _fmt(value: Any) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    for label, rows in curves.items():
        best = max(rows, key=lambda row: row["macro_f1"] if row["macro_f1"] is not None else -1.0)
        print(
            f"{label}: best macro-F1 threshold={best['threshold']:.4g} "
            f"macro_f1={_fmt(best['macro_f1'])} coverage={_fmt(best['coverage'])} "
            f"fail_f1={_fmt(best['fail_f1'])} pass_f1={_fmt(best['pass_f1'])} "
            f"inconclusive_rate={_fmt(best['inconclusive_rate'])}"
        )


if __name__ == "__main__":
    main()
