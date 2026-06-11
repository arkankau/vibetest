"""Plot dataset-level precision/recall curves from VibeTest or verifier scores.

The score threshold is applied only to predicted FAIL verdicts, because PASS and
INCONCLUSIVE rows do not represent accepted failure claims. Recall is computed
against all ground-truth FAIL labels in the full dataset, so unpredicted or
abstained GT failures remain false negatives.
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
class ScoredItem:
    label: int
    score: float
    dataset: str
    method_model: str
    file: str


@dataclass(frozen=True)
class LoadedItems:
    items: list[ScoredItem]
    missing_scores: int
    positive_total: int
    labeled_total: int


@dataclass
class CurveGroup:
    items: list[ScoredItem]
    positive_total: int = 0
    labeled_total: int = 0


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


def _evidence_match_label(test: dict[str, Any]) -> int:
    metadata = test.get("metadata") or {}
    synthetic_score = metadata.get("synthetic_score") or {}
    return 1 if str(synthetic_score.get("evidence_match_grade") or "").strip().upper() == "C" else 0


def _clamped_score(raw: Any) -> float | None:
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return max(0.0, min(1.0, score))


def _verifier_score(test: dict[str, Any]) -> float | None:
    metadata = test.get("metadata") or {}
    verifier = metadata.get("evidence_verifier") or {}
    return _clamped_score(verifier.get("score"))


def _case_score(test: dict[str, Any]) -> float | None:
    metadata = test.get("metadata") or {}
    return _clamped_score(
        metadata.get("case_score", metadata.get("fail_support_score"))
    )


def _load_scored_items(
    path: Path,
    *,
    target: str,
    score_source: str,
    missing_score: str,
) -> LoadedItems:
    out: list[ScoredItem] = []
    missing = 0
    positive_total = 0
    labeled_total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            dataset = str(entry.get("dataset") or "")
            method_model = str(entry.get("method_model") or "")
            for test in entry.get("tests") or []:
                gt = _gt_label(entry, test)
                if gt is None:
                    continue
                labeled_total += 1
                if gt == 1:
                    positive_total += 1

                predicted_verdict = _predicted_verdict(test)
                if predicted_verdict != "FAIL":
                    continue
                if target == "evidence-match":
                    label = 1 if gt == 1 and _evidence_match_label(test) else 0
                else:
                    label = gt
                score = _verifier_score(test) if score_source == "verifier" else _case_score(test)
                if score is None:
                    missing += 1
                    if missing_score == "reject":
                        score = -math.inf
                    else:
                        score = math.inf
                out.append(
                    ScoredItem(
                        label=label,
                        score=score,
                        dataset=dataset,
                        method_model=method_model,
                        file=str(path),
                    )
                )
    return LoadedItems(
        items=out,
        missing_scores=missing,
        positive_total=positive_total,
        labeled_total=labeled_total,
    )


def _curve(items: list[ScoredItem], *, positive_total: int, labeled_total: int) -> list[dict[str, Any]]:
    thresholds = sorted({item.score for item in items if math.isfinite(item.score)}, reverse=True)
    if not thresholds:
        return []
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [item for item in items if item.score >= threshold]
        tp = sum(1 for item in accepted if item.label == 1)
        fp = len(accepted) - tp
        fn = positive_total - tp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / positive_total if positive_total else None
        f1 = (
            None
            if precision is None or recall is None
            else 0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "accepted": len(accepted),
                "positives": positive_total,
                "total": labeled_total,
                "candidate_total": len(items),
            }
        )
    return rows


def _label_for_path(path: Path) -> str:
    name = path.stem
    name = name.replace("synthetic_kaggle_", "")
    name = name.replace("_evidence_verified", "")
    name = name.replace("_scored-gpt-5.4-mini", "")
    name = name.replace("_AT-", " / ")
    return name


def _run_mode_for_path(path: Path) -> str:
    return "dynamic" if "dynamic" in path.stem else "static"


def _write_curve_csv(path: Path, curves: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series",
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "accepted",
        "positives",
        "total",
        "candidate_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for series, rows in curves.items():
            for row in rows:
                writer.writerow({"series": series, **row})


def _plot_curves(
    curves: dict[str, list[dict[str, Any]]],
    *,
    title: str,
    score_source: str,
    output_png: Path,
    output_pdf: Path | None,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    for series, rows in curves.items():
        if not rows:
            continue
        recall = [float(row["recall"]) for row in rows if row["recall"] is not None]
        precision = [float(row["precision"]) for row in rows if row["precision"] is not None]
        plt.step(recall, precision, where="post", linewidth=2, label=series)
        plt.scatter(recall, precision, s=16)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    if output_pdf:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_pdf)
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Synthetic result JSONL files.")
    parser.add_argument(
        "--target",
        choices=["evidence-match", "gt-label"],
        default="evidence-match",
        help="Evaluation target. evidence-match counts accepted predicted FAILs as TP only when GT is FAIL and the scorer grade is C; gt-label uses GT FAIL directly.",
    )
    parser.add_argument(
        "--score-source",
        choices=["verifier", "case-score"],
        default="verifier",
        help="Score to threshold over predicted FAILs. verifier uses metadata.evidence_verifier.score; case-score uses VibeTest metadata.case_score/fail_support_score.",
    )
    parser.add_argument(
        "--missing-score",
        choices=["reject", "accept", "drop"],
        default="reject",
        help="How to handle predicted FAILs without the selected score.",
    )
    parser.add_argument("--output-png", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--output-pdf", type=Path, help="Optional output PDF path.")
    parser.add_argument("--output-csv", type=Path, help="Optional threshold metrics CSV path.")
    parser.add_argument("--title", default="Evidence Verifier Precision-Recall Curve", help="Plot title.")
    parser.add_argument(
        "--group-by",
        choices=["file", "run-mode"],
        default="file",
        help="Curve grouping. file plots each result file; run-mode combines files into static and dynamic curves based on filename.",
    )
    parser.add_argument("--no-combined", action="store_true", help="Do not add a combined curve when multiple files are provided.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    groups: dict[str, CurveGroup] = {}
    total_missing = 0
    for path in args.results:
        if not path.exists():
            raise SystemExit(f"Result file not found: {path}")
        loaded = _load_scored_items(
            path,
            target=args.target,
            score_source=args.score_source,
            missing_score=args.missing_score,
        )
        items = loaded.items
        if args.missing_score == "drop":
            items = [item for item in items if math.isfinite(item.score)]
        total_missing += loaded.missing_scores
        label = _label_for_path(path) if args.group_by == "file" else _run_mode_for_path(path)
        group = groups.setdefault(label, CurveGroup(items=[]))
        group.items.extend(items)
        group.positive_total += loaded.positive_total
        group.labeled_total += loaded.labeled_total

    curves = {
        label: _curve(
            group.items,
            positive_total=group.positive_total,
            labeled_total=group.labeled_total,
        )
        for label, group in groups.items()
    }

    if args.group_by == "file" and len(args.results) > 1 and not args.no_combined:
        combined_group = CurveGroup(items=[])
        for group in groups.values():
            combined_group.items.extend(group.items)
            combined_group.positive_total += group.positive_total
            combined_group.labeled_total += group.labeled_total
        curves = {
            "combined": _curve(
                combined_group.items,
                positive_total=combined_group.positive_total,
                labeled_total=combined_group.labeled_total,
            ),
            **curves,
        }

    _plot_curves(
        curves,
        title=args.title,
        score_source=args.score_source,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
    )
    if args.output_csv:
        _write_curve_csv(args.output_csv, curves)

    print(f"Wrote PNG: {args.output_png}")
    if args.output_pdf:
        print(f"Wrote PDF: {args.output_pdf}")
    if args.output_csv:
        print(f"Wrote CSV: {args.output_csv}")
    print(f"Predicted FAILs without selected scores: {total_missing}")


if __name__ == "__main__":
    main()
