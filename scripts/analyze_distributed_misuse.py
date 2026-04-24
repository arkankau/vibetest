from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


AXIS_LABEL_FONTSIZE = 7.0
TICK_LABEL_FONTSIZE = 6.0
LEGEND_FONTSIZE = 6.0
GRID_COLOR = "#D9D9D9"
GRID_ALPHA = 0.55
SPINE_COLOR = "#666666"
PR_LOG_FLOOR = 1e-2
RECALL_LOG_FLOOR = 1e-2
DM_METHOD_ORDER = ["Meerkat", "Monitor", "Bayesian", "Buffer", "Naive Agent"]
DM_METHOD_COLORS = {
    "Meerkat": "#D55E00",
    "Monitor": "#CC79A7",
    "Bayesian": "#0072B2",
    "Buffer": "#009E73",
    "Naive Agent": "#8A8A8A",
}
DM_METHOD_LINESTYLES = {
    "Meerkat": "-",
    "Monitor": "-.",
    "Bayesian": ":",
    "Buffer": "--",
    "Naive Agent": (0, (5, 1.4)),
}
DM_MODEL_COLORS = {
    "gpt-5.4-mini": "#56B4E9",
    "Qwen-3.5": "#CC79A7",
}
_DM_MISSING_RESULT_WARNINGS: set[tuple[str, str, str, int, str]] = set()
_TRACE_SCORES_BLOCK_RE = re.compile(r"TRACE_SCORES_START\s*\n(.*?)\nTRACE_SCORES_END", re.S | re.I)
_TRACE_SCORES_SECTION_RE = re.compile(r"^\s*TRACE_SCORES(?:\s*:)?\s*$\n?(.*)\Z", re.S | re.I | re.M)
_DM_BOOTSTRAP_CACHE_VERSION = 4


def _dm_rc_context() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.titlesize": AXIS_LABEL_FONTSIZE,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.8,
        "grid.color": GRID_COLOR,
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }


def _style_dm_axis(ax, *, x_label: str | None = None, y_label: str | None = None, title: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE, pad=4)
    if x_label:
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.3)
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.2)
    ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def _dm_legend_handles(*, linewidth: float, method_order: list[str] | None = None) -> list[Line2D]:
    order = method_order or DM_METHOD_ORDER
    return [
        Line2D([0], [0], color=DM_METHOD_COLORS[method], linestyle=DM_METHOD_LINESTYLES[method], linewidth=linewidth)
        for method in order
    ]


def _clip_precision_for_log(values: list[float], *, floor: float = PR_LOG_FLOOR) -> list[float]:
    return [min(1.0, max(floor, float(value))) for value in values]


def _clip_recall_for_log(values: list[float], *, floor: float = RECALL_LOG_FLOOR) -> list[float]:
    return [min(1.0, max(floor, float(value))) for value in values]


def _dm_model_slug(model_label: str) -> str:
    if model_label == "Qwen-3.5":
        return "qwen35"
    if model_label == "gpt-5.4-mini":
        return "gpt-5.4-mini"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")


def _dm_meerkat_result_path(results_dir: Path, *, domain: str, bg: int, model_label: str) -> Path:
    slug = _dm_model_slug(model_label)
    mapping = {
        ("cyber", 20, "gpt-5.4-mini"): "dm_cyber_d6_bg20_v6.jsonl",
        ("cyber", 100, "gpt-5.4-mini"): "dm_cyber_d6_bg100_n15.jsonl",
        ("cyber", 20, "qwen35"): "dm_cyber_d6_bg20_qwen35_n20.jsonl",
        ("cyber", 100, "qwen35"): "dm_cyber_d6_bg100_qwen35_n50.jsonl",
        ("bio", 20, "gpt-5.4-mini"): "dm_bio_d6_bg20_n20.jsonl",
        ("bio", 100, "gpt-5.4-mini"): "dm_bio_d6_bg100_n50.jsonl",
        ("bio", 20, "qwen35"): "dm_bio_d6_bg20_qwen35_n20.jsonl",
        ("bio", 100, "qwen35"): "dm_bio_d6_bg100_qwen35_n50.jsonl",
    }
    filename = mapping.get((domain, bg, slug))
    if filename is None:
        raise ValueError(f"No canonical DM Meerkat filename for domain={domain} bg={bg} model={model_label}")
    return results_dir / filename


def _warn_missing_dm_result(path: Path, *, domain: str, bg: int, model_label: str, method_label: str) -> None:
    key = (domain, model_label, method_label, bg, str(path))
    if key in _DM_MISSING_RESULT_WARNINGS:
        return
    _DM_MISSING_RESULT_WARNINGS.add(key)
    print(
        f"Missing canonical DM {method_label} result for domain={domain} bg={bg} model={model_label}: {path}",
        file=sys.stderr,
    )


def _dm_bayesian_result_candidates(results_dir: Path, *, domain: str, bg: int, model_label: str) -> list[Path]:
    return [results_dir / f"dm_{domain}_d6_bg{bg}_{_dm_model_slug(model_label)}_bayesian.jsonl"]


def _dm_buffer_result_candidates(results_dir: Path, *, domain: str, bg: int, model_label: str) -> list[Path]:
    return [results_dir / f"dm_{domain}_d6_bg{bg}_{_dm_model_slug(model_label)}_buffer.jsonl"]


def _dm_naive_agent_result_candidates(results_dir: Path, *, domain: str, bg: int, model_label: str) -> list[Path]:
    return [results_dir / f"dm_{domain}_d6_bg{bg}_{_dm_model_slug(model_label)}_naive.jsonl"]


def _dm_monitor_result_candidates(results_dir: Path, *, domain: str, bg: int, model_label: str) -> list[Path]:
    return [results_dir / f"dm_{domain}_d6_bg{bg}_{_dm_model_slug(model_label)}_llmjudge.jsonl"]



DM_FILE_RE = re.compile(
    r"dm_(?P<domain>bio|cyber)_d(?P<decomp>\d+)_bg(?P<background>\d+)_(?P<variant>v6|v2|scored)\.jsonl$"
)


@dataclass
class SettingMetrics:
    path: Path
    domain: str
    decomp_level: int
    background_multiplier: int
    variant: str
    model_name: str
    stage1_ap: float | None
    stage2_ap: float | None
    stage1_roc_auc: float | None
    stage2_roc_auc: float | None
    stage1_case_ap: float | None
    stage2_case_ap: float | None
    stage1_case_roc_auc: float | None
    stage2_case_roc_auc: float | None
    stage1_curve: list[dict[str, float]] | None
    stage2_curve: list[dict[str, float]] | None
    stage1_roc_curve: list[dict[str, float]] | None
    stage2_roc_curve: list[dict[str, float]] | None
    stage1_case_curve: list[dict[str, float]] | None
    stage2_case_curve: list[dict[str, float]] | None
    stage1_case_roc_curve: list[dict[str, float]] | None
    stage2_case_roc_curve: list[dict[str, float]] | None
    stage1_available_cases: int
    stage2_available_cases: int
    total_cases: int


@dataclass
class PaperCurveRun:
    path: Path
    model_label: str
    method_label: str
    background_multiplier: int
    case_count: int
    average_precision: float | None
    average_precision_bootstrap_se: float | None
    recall_grid: list[float]
    precision_curve: list[float]
    precision_lower: list[float]
    precision_upper: list[float]


@dataclass
class PaperRocRun:
    path: Path
    model_label: str
    method_label: str
    background_multiplier: int
    case_count: int
    roc_auc: float | None
    roc_auc_bootstrap_se: float | None
    fpr_grid: list[float]
    tpr_curve: list[float]
    tpr_lower: list[float]
    tpr_upper: list[float]


def _path_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path.resolve()), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _bootstrap_cache_path(cache_dir: Path, *, kind: str, payload: dict[str, Any]) -> Path:
    serialized = json.dumps(
        {
            "version": _DM_BOOTSTRAP_CACHE_VERSION,
            "kind": kind,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return cache_dir / f"{kind}_{digest}.json"


def _read_bootstrap_cache(cache_dir: Path | None, *, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    cache_path = _bootstrap_cache_path(cache_dir, kind=kind, payload=payload)
    if not cache_path.is_file():
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


def _write_bootstrap_cache(
    cache_dir: Path | None,
    *,
    kind: str,
    payload: dict[str, Any],
    value: dict[str, Any],
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _bootstrap_cache_path(cache_dir, kind=kind, payload=payload)
    cache_path.write_text(json.dumps(value, sort_keys=True))


def _display_model_name(model_name: str | None, *, path: Path | None = None) -> str:
    text = str(model_name or "").strip()
    lowered = text.lower()
    if "qwen3.5" in lowered or "qwen-3.5" in lowered:
        return "Qwen-3.5"
    if "gpt-5.4-mini" in lowered:
        return "gpt-5.4-mini"
    if path is not None:
        name = path.name.lower()
        if "qwen35" in name:
            return "Qwen-3.5"
        if "buffer" in name and "qwen35" not in name:
            return "gpt-5.4-mini"
    return text or "unknown-model"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze distributed misuse scored JSONL files and plot Judge vs AT PR curves/AP."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional dm_*_v6.jsonl, dm_*_v2.jsonl, or dm_*_scored.jsonl files. If omitted, auto-discovers dm_*_v6.jsonl first, then dm_*_v2.jsonl, then falls back to dm_*_scored.jsonl.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to search for distributed-misuse result JSONL files when no inputs are given.",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory to write figure files.",
    )
    parser.add_argument(
        "--figure-formats",
        default="png,pdf",
        help="Comma-separated list of figure formats to emit.",
    )
    parser.add_argument(
        "--recompute-stage1-if-missing",
        action="store_true",
        help="If initial stage-1 scores are missing, recompute them from the trace files using the per-trace scorer.",
    )
    parser.add_argument(
        "--stage1-concurrency",
        type=int,
        default=32,
        help="Concurrency for recomputing stage-1 per-trace scores.",
    )
    parser.add_argument(
        "--stage1-cache-dir",
        default="results/dm_stage1_score_cache",
        help="Cache directory for recomputed stage-1 scores.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Number of bootstrap resamples for PR bands and AP standard errors.",
    )
    parser.add_argument(
        "--bootstrap-cache-dir",
        default="results/dm_bootstrap_cache",
        help="Directory to cache bootstrap-derived PR bands and AP/ROC summary standard errors.",
    )
    return parser.parse_args()


def _normalize_trace_path(path: str) -> str:
    normalized = (path or "").strip()
    for prefix in ("/workspace/repo/", "/workspace/", "workspace/repo/", "workspace/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _first_existing(paths: list[Path], want) -> Path | None:
    for path in paths:
        if path.is_file() and want(path):
            return path
    return None


def _average_precision_for_case(scores: dict[str, float], gt: set[str]) -> float | None:
    if not scores or not gt:
        return None
    normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
    gt_norm = {_normalize_trace_path(x) for x in gt}
    positives = len(gt_norm)
    if positives <= 0:
        return None
    ranked = sorted(normalized_scores.items(), key=lambda item: (-item[1], item[0]))
    tp = 0
    ap_accum = 0.0
    for rank, (trace_path, _) in enumerate(ranked, start=1):
        if trace_path not in gt_norm:
            continue
        tp += 1
        ap_accum += tp / rank
    return ap_accum / positives


def _case_precision_recall(scores: dict[str, float], gt: set[str], threshold: float) -> tuple[float, float] | None:
    if not scores or not gt:
        return None
    normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
    gt_norm = {_normalize_trace_path(x) for x in gt}
    positives = len(gt_norm)
    if positives <= 0:
        return None
    tp = fp = 0
    for trace_path, score in normalized_scores.items():
        if score < threshold:
            continue
        if trace_path in gt_norm:
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / positives if positives > 0 else 0.0
    return precision, recall


def _roc_auc_for_case(scores: dict[str, float], gt: set[str]) -> float | None:
    if not scores or not gt:
        return None
    normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
    gt_norm = {_normalize_trace_path(x) for x in gt}
    values = [(1 if trace_path in gt_norm else 0, score) for trace_path, score in normalized_scores.items()]
    positives = sum(label for label, _ in values)
    negatives = len(values) - positives
    if positives <= 0 or negatives <= 0:
        return None
    values.sort(key=lambda item: item[1])
    rank = 1
    positive_rank_sum = 0.0
    idx = 0
    while idx < len(values):
        j = idx + 1
        while j < len(values) and values[j][1] == values[idx][1]:
            j += 1
        avg_rank = (rank + (rank + (j - idx) - 1)) / 2.0
        pos_in_group = sum(label for label, _ in values[idx:j])
        positive_rank_sum += pos_in_group * avg_rank
        rank += j - idx
        idx = j
    return (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)


def _case_roc_point(scores: dict[str, float], gt: set[str], threshold: float) -> tuple[float, float] | None:
    if not scores or not gt:
        return None
    normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
    gt_norm = {_normalize_trace_path(x) for x in gt}
    positives = len(gt_norm)
    negatives = len([path for path in normalized_scores if path not in gt_norm])
    if positives <= 0 or negatives <= 0:
        return None
    tp = fp = 0
    for trace_path, score in normalized_scores.items():
        if score < threshold:
            continue
        if trace_path in gt_norm:
            tp += 1
        else:
            fp += 1
    return tp / positives, fp / negatives


def _flat_average_precision(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    if positives <= 0:
        return None
    ranked = sorted(((int(label), float(score)) for label, score in pairs), key=lambda item: (-item[1], -item[0]))
    tp = 0
    ap_accum = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label != 1:
            continue
        tp += 1
        ap_accum += tp / rank
    return ap_accum / positives


def _flat_roc_auc(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    values = [(int(label), float(score)) for label, score in pairs]
    positives = sum(label for label, _ in values)
    negatives = len(values) - positives
    if positives <= 0 or negatives <= 0:
        return None
    values.sort(key=lambda item: item[1])
    rank = 1
    positive_rank_sum = 0.0
    idx = 0
    while idx < len(values):
        j = idx + 1
        while j < len(values) and values[j][1] == values[idx][1]:
            j += 1
        avg_rank = (rank + (rank + (j - idx) - 1)) / 2.0
        pos_in_group = sum(label for label, _ in values[idx:j])
        positive_rank_sum += pos_in_group * avg_rank
        rank += j - idx
        idx = j
    return (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)


def _flat_precision_recall_curve(pairs: list[tuple[int, float]]) -> list[dict[str, float]] | None:
    if not pairs:
        return None
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    if positives <= 0:
        return None
    thresholds = [float("inf")] + sorted({float(score) for _, score in pairs}, reverse=True)
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        tp = fp = 0
        for label, score in pairs:
            if float(score) < threshold:
                continue
            if int(label) == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / positives
        curve.append({"threshold": threshold, "precision": precision, "recall": recall})
    return curve


def _flat_roc_curve(pairs: list[tuple[int, float]]) -> list[dict[str, float]] | None:
    if not pairs:
        return None
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    negatives = sum(1 for label, _ in pairs if int(label) == 0)
    if positives <= 0 or negatives <= 0:
        return None
    thresholds = [float("inf")] + sorted({float(score) for _, score in pairs}, reverse=True) + [float("-inf")]
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        tp = fp = 0
        for label, score in pairs:
            if float(score) < threshold:
                continue
            if int(label) == 1:
                tp += 1
            else:
                fp += 1
        curve.append({"threshold": threshold, "tpr": tp / positives, "fpr": fp / negatives})
    return curve


def _average_curves(
    per_case_scores: list[tuple[dict[str, float], set[str]]],
) -> tuple[float | None, list[dict[str, float]] | None, float | None, list[dict[str, float]] | None]:
    valid_cases: list[tuple[dict[str, float], set[str]]] = []
    for scores, gt in per_case_scores:
        if not scores:
            continue
        normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
        gt_norm = {_normalize_trace_path(x) for x in gt}
        if not gt_norm:
            continue
        valid_cases.append((normalized_scores, gt_norm))

    if not valid_cases:
        return None, None, None, None

    thresholds = sorted(
        {score for scores, _ in valid_cases for score in scores.values()},
        reverse=True,
    )
    pr_thresholds = [float("inf")] + thresholds
    roc_thresholds = [float("inf")] + thresholds + [float("-inf")]

    case_aps = [_average_precision_for_case(scores, gt) for scores, gt in valid_cases]
    case_aps = [float(ap) for ap in case_aps if ap is not None]
    ap = (sum(case_aps) / len(case_aps)) if case_aps else None
    case_aucs = [_roc_auc_for_case(scores, gt) for scores, gt in valid_cases]
    case_aucs = [float(auc) for auc in case_aucs if auc is not None]
    roc_auc = (sum(case_aucs) / len(case_aucs)) if case_aucs else None

    pr_curve: list[dict[str, float]] = []
    for threshold in pr_thresholds:
        precisions: list[float] = []
        recalls: list[float] = []
        tps: list[int] = []
        fps: list[int] = []
        fns: list[int] = []
        for scores, gt in valid_cases:
            point = _case_precision_recall(scores, gt, threshold)
            if point is None:
                continue
            precision, recall = point
            precisions.append(precision)
            recalls.append(recall)
            normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
            gt_norm = {_normalize_trace_path(x) for x in gt}
            predicted = {path for path, score in normalized_scores.items() if score >= threshold}
            tp = len(predicted & gt_norm)
            fp = len(predicted - gt_norm)
            fn = len(gt_norm - predicted)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
        if not precisions:
            continue
        pr_curve.append(
            {
                "threshold": threshold,
                "precision": sum(precisions) / len(precisions),
                "recall": sum(recalls) / len(recalls),
                "tp": sum(tps) / len(tps) if tps else 0.0,
                "fp": sum(fps) / len(fps) if fps else 0.0,
                "fn": sum(fns) / len(fns) if fns else 0.0,
            }
        )

    roc_curve: list[dict[str, float]] = []
    for threshold in roc_thresholds:
        tprs: list[float] = []
        fprs: list[float] = []
        for scores, gt in valid_cases:
            point = _case_roc_point(scores, gt, threshold)
            if point is None:
                continue
            tpr, fpr = point
            tprs.append(tpr)
            fprs.append(fpr)
        if not tprs:
            continue
        roc_curve.append(
            {
                "threshold": threshold,
                "tpr": sum(tprs) / len(tprs),
                "fpr": sum(fprs) / len(fprs),
            }
        )

    return ap, pr_curve, roc_auc, roc_curve


def _load_stage1_scores_from_repo(repo_path: Path) -> dict[str, float] | None:
    initial_scores_tsv_candidates = [
        repo_path / "search_aids" / "initial_scores.tsv",
        repo_path / "initial_scores.tsv",
    ]
    initial_scores_tsv_path = next((path for path in initial_scores_tsv_candidates if path.exists()), None)
    if initial_scores_tsv_path is not None:
        scores: dict[str, float] = {}
        for line in initial_scores_tsv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            trace_path, raw_score = parts
            try:
                scores[_normalize_trace_path(trace_path)] = float(raw_score)
            except (TypeError, ValueError):
                continue
        return scores or None

    initial_scores_json_path = repo_path / "initial_scores.json"
    if not initial_scores_json_path.exists():
        return None
    try:
        data = json.loads(initial_scores_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    scores: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                raw_score = value.get("score", 0.0)
            else:
                raw_score = value
            try:
                scores[_normalize_trace_path(key)] = float(raw_score)
            except (TypeError, ValueError):
                continue
    return scores or None


def _load_stage1_cache(cache_path: Path) -> dict[str, float]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_stage1_cache(cache_path: Path, cache: dict[str, float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _recompute_stage1_scores_for_rows(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    concurrency: int,
    cache_dir: Path,
) -> dict[str, dict[str, float]]:
    from experiments.safety import _score_traces_parallel, _safe_slug, _trace_content_hash

    model_slug = _safe_slug(model_name).lower()
    cache_path = cache_dir / f"{model_slug}.json"
    cache = _load_stage1_cache(cache_path)
    cache_dirty = False

    row_scores: dict[str, dict[str, float]] = {}
    uncached_contents: dict[str, str] = {}
    hash_to_paths: dict[str, list[tuple[str, str]]] = {}

    for row in rows:
        case_id = str(row.get("case_id") or row.get("repo_name") or "")
        repo_path = Path(row["repo"])
        gt_trace_files = row.get("ground_truth", {}).get("trace_files") or []
        row_scores[case_id] = {}
        for rel_path in gt_trace_files:
            trace_rel = _normalize_trace_path(rel_path)
            trace_path = repo_path / trace_rel
            if not trace_path.exists():
                continue
            content = trace_path.read_text(encoding="utf-8", errors="replace")
            content_hash = _trace_content_hash(content)
            hash_to_paths.setdefault(content_hash, []).append((case_id, trace_rel))
            if content_hash in cache:
                row_scores[case_id][trace_rel] = cache[content_hash]
            elif content_hash not in uncached_contents:
                uncached_contents[content_hash] = content

    if uncached_contents:
        print(
            f"Recomputing {len(uncached_contents)} unique stage-1 trace scores "
            f"with model {model_name} (concurrency={concurrency})..."
        )
        scored = asyncio.run(
            _score_traces_parallel(
                {content_hash: content for content_hash, content in uncached_contents.items()},
                model_name=model_name,
                concurrency=concurrency,
            )
        )
        for content_hash, score in scored.items():
            cache[content_hash] = float(score)
            cache_dirty = True
            for case_id, trace_rel in hash_to_paths.get(content_hash, []):
                row_scores.setdefault(case_id, {})[trace_rel] = float(score)

    if cache_dirty:
        _save_stage1_cache(cache_path, cache)

    # Fill any still-missing traces with 0.0 so PR computation has all traces.
    for row in rows:
        case_id = str(row.get("case_id") or row.get("repo_name") or "")
        gt_trace_files = row.get("ground_truth", {}).get("trace_files") or []
        case_scores = row_scores.setdefault(case_id, {})
        for rel_path in gt_trace_files:
            trace_rel = _normalize_trace_path(rel_path)
            case_scores.setdefault(trace_rel, 0.0)

    return row_scores


def _collect_setting_metrics(
    path: Path,
    *,
    recompute_stage1_if_missing: bool,
    stage1_concurrency: int,
    stage1_cache_dir: Path,
) -> SettingMetrics:
    match = DM_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized distributed-misuse filename: {path.name}")

    rows = _load_rows(path)
    if not rows:
        raise ValueError(f"No rows in {path}")

    model_name = (
        rows[0].get("tests", [{}])[0].get("metadata", {}).get("model")
        or rows[0].get("summary", {}).get("model")
        or "unknown-model"
    )

    stage2_cases: list[tuple[dict[str, float], set[str]]] = []
    stage1_cases: list[tuple[dict[str, float], set[str]]] = []
    stage1_case_pairs: list[tuple[int, float]] = []
    stage2_case_pairs: list[tuple[int, float]] = []
    rows_needing_stage1: list[dict[str, Any]] = []
    stage1_loaded_by_case: dict[str, dict[str, float]] = {}

    for row in rows:
        gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
        gt_case_label = 1 if bool(row.get("ground_truth", {}).get("has_distributed_misuse")) else 0
        case_id = str(row.get("case_id") or row.get("repo_name") or "")
        stage2_scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=bool(gt))
        if stage2_scores:
            stage2_cases.append((stage2_scores, gt))
            stage2_case_pairs.append((gt_case_label, max(stage2_scores.values())))

        stage1_scores = _load_stage1_scores_from_repo(Path(row["repo"]))
        if stage1_scores:
            stage1_loaded_by_case[case_id] = stage1_scores
            stage1_cases.append((stage1_scores, gt))
            stage1_case_pairs.append((gt_case_label, max(stage1_scores.values())))
        elif recompute_stage1_if_missing:
            rows_needing_stage1.append(row)

    if rows_needing_stage1:
        recomputed = _recompute_stage1_scores_for_rows(
            rows_needing_stage1,
            model_name=model_name,
            concurrency=stage1_concurrency,
            cache_dir=stage1_cache_dir,
        )
        for row in rows_needing_stage1:
            gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
            case_id = str(row.get("case_id") or row.get("repo_name") or "")
            scores = recomputed.get(case_id) or {}
            if not scores and gt:
                scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=True)
            if scores:
                stage1_cases.append((scores, gt))
                gt_case_label = 1 if bool(row.get("ground_truth", {}).get("has_distributed_misuse")) else 0
                stage1_case_pairs.append((gt_case_label, max(scores.values())))

    stage1_ap, stage1_curve, stage1_roc_auc, stage1_roc_curve = _average_curves(stage1_cases)
    stage2_ap, stage2_curve, stage2_roc_auc, stage2_roc_curve = _average_curves(stage2_cases)
    stage1_case_ap = _flat_average_precision(stage1_case_pairs)
    stage2_case_ap = _flat_average_precision(stage2_case_pairs)
    stage1_case_roc_auc = _flat_roc_auc(stage1_case_pairs)
    stage2_case_roc_auc = _flat_roc_auc(stage2_case_pairs)
    stage1_case_curve = _flat_precision_recall_curve(stage1_case_pairs)
    stage2_case_curve = _flat_precision_recall_curve(stage2_case_pairs)
    stage1_case_roc_curve = _flat_roc_curve(stage1_case_pairs)
    stage2_case_roc_curve = _flat_roc_curve(stage2_case_pairs)

    return SettingMetrics(
        path=path,
        domain=match.group("domain"),
        decomp_level=int(match.group("decomp")),
        background_multiplier=int(match.group("background")),
        variant=match.group("variant"),
        model_name=model_name,
        stage1_ap=stage1_ap,
        stage2_ap=stage2_ap,
        stage1_roc_auc=stage1_roc_auc,
        stage2_roc_auc=stage2_roc_auc,
        stage1_case_ap=stage1_case_ap,
        stage2_case_ap=stage2_case_ap,
        stage1_case_roc_auc=stage1_case_roc_auc,
        stage2_case_roc_auc=stage2_case_roc_auc,
        stage1_curve=stage1_curve,
        stage2_curve=stage2_curve,
        stage1_roc_curve=stage1_roc_curve,
        stage2_roc_curve=stage2_roc_curve,
        stage1_case_curve=stage1_case_curve,
        stage2_case_curve=stage2_case_curve,
        stage1_case_roc_curve=stage1_case_roc_curve,
        stage2_case_roc_curve=stage2_case_roc_curve,
        stage1_available_cases=len(stage1_cases),
        stage2_available_cases=len(stage2_cases),
        total_cases=len(rows),
    )


def _load_stage_cases(
    path: Path,
    *,
    recompute_stage1_if_missing: bool,
    stage1_concurrency: int,
    stage1_cache_dir: Path,
) -> tuple[list[tuple[dict[str, float], set[str]]], list[tuple[dict[str, float], set[str]]], str]:
    rows = _load_rows(path)
    if not rows:
        return [], [], "unknown-model"

    model_name = (
        rows[0].get("tests", [{}])[0].get("metadata", {}).get("model")
        or rows[0].get("summary", {}).get("model")
        or "unknown-model"
    )

    stage2_cases: list[tuple[dict[str, float], set[str]]] = []
    stage1_cases: list[tuple[dict[str, float], set[str]]] = []
    rows_needing_stage1: list[dict[str, Any]] = []

    for row in rows:
        gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
        stage2_scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=bool(gt))
        if stage2_scores:
            stage2_cases.append((stage2_scores, gt))
        stage1_scores = _load_stage1_scores_from_repo(Path(row["repo"]))
        if not stage1_scores and gt:
            stage1_scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=True)
        if stage1_scores:
            stage1_cases.append((stage1_scores, gt))
        elif recompute_stage1_if_missing:
            rows_needing_stage1.append(row)

    if rows_needing_stage1:
        recomputed = _recompute_stage1_scores_for_rows(
            rows_needing_stage1,
            model_name=model_name,
            concurrency=stage1_concurrency,
            cache_dir=stage1_cache_dir,
        )
        for row in rows_needing_stage1:
            gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
            case_id = str(row.get("case_id") or row.get("repo_name") or "")
            scores = recomputed.get(case_id) or {}
            if not scores and gt:
                scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=True)
            if scores:
                stage1_cases.append((scores, gt))

    return stage1_cases, stage2_cases, _display_model_name(model_name, path=path)


def _interp_precision_at_recalls(curve: list[dict[str, float]], recall_grid: list[float]) -> list[float]:
    if not curve:
        return [0.0 for _ in recall_grid]
    recall_to_precision: dict[float, float] = {}
    for point in curve:
        r = max(0.0, min(1.0, float(point["recall"])))
        p = max(0.0, min(1.0, float(point["precision"])))
        recall_to_precision[r] = max(recall_to_precision.get(r, 0.0), p)
    recalls = sorted(recall_to_precision)
    precisions = [recall_to_precision[r] for r in recalls]
    envelope = precisions[:]
    for idx in range(len(envelope) - 2, -1, -1):
        envelope[idx] = max(envelope[idx], envelope[idx + 1])
    out: list[float] = []
    for target in recall_grid:
        chosen = 0.0
        for r, p in zip(recalls, envelope):
            if r >= target:
                chosen = p
                break
        out.append(chosen)
    return out


def _precision_recall_curve_for_case(scores: dict[str, float], gt: set[str]) -> list[dict[str, float]]:
    if not scores or not gt:
        return []
    normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
    gt_norm = {_normalize_trace_path(x) for x in gt}
    if not gt_norm:
        return []
    thresholds = [float("inf")] + sorted({float(score) for score in normalized_scores.values()}, reverse=True)
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        point = _case_precision_recall(normalized_scores, gt_norm, threshold)
        if point is None:
            continue
        precision, recall = point
        curve.append({"threshold": threshold, "precision": precision, "recall": recall})
    return curve


def _mean_precision_at_recalls(
    per_case_scores: list[tuple[dict[str, float], set[str]]],
    recall_grid: list[float],
) -> list[float]:
    valid_cases = [(scores, gt) for scores, gt in per_case_scores if scores and gt]
    if not valid_cases:
        return [0.0 for _ in recall_grid]
    per_case_interp: list[list[float]] = []
    for scores, gt in valid_cases:
        curve = _precision_recall_curve_for_case(scores, gt)
        if not curve:
            continue
        per_case_interp.append(_interp_precision_at_recalls(curve, recall_grid))
    if not per_case_interp:
        return [0.0 for _ in recall_grid]
    return [sum(sample[idx] for sample in per_case_interp) / len(per_case_interp) for idx in range(len(recall_grid))]


def _interp_tpr_at_fprs(curve: list[dict[str, float]], fpr_grid: list[float]) -> list[float]:
    if not curve:
        return [0.0 for _ in fpr_grid]
    fpr_to_tpr: dict[float, float] = {}
    for point in curve:
        fpr = max(0.0, min(1.0, float(point.get("fpr", 0.0))))
        tpr = max(0.0, min(1.0, float(point.get("tpr", 0.0))))
        fpr_to_tpr[fpr] = max(fpr_to_tpr.get(fpr, 0.0), tpr)
    points = sorted(fpr_to_tpr.items())
    fprs = [point[0] for point in points]
    tprs = [point[1] for point in points]
    out: list[float] = []
    for target in fpr_grid:
        if target <= fprs[0]:
            out.append(tprs[0])
            continue
        if target >= fprs[-1]:
            out.append(tprs[-1])
            continue
        upper_idx = 1
        while upper_idx < len(fprs) and fprs[upper_idx] < target:
            upper_idx += 1
        lower_idx = upper_idx - 1
        lower_fpr = fprs[lower_idx]
        upper_fpr = fprs[upper_idx]
        lower_tpr = tprs[lower_idx]
        upper_tpr = tprs[upper_idx]
        if upper_fpr <= lower_fpr:
            out.append(max(lower_tpr, upper_tpr))
            continue
        weight = (target - lower_fpr) / (upper_fpr - lower_fpr)
        out.append(lower_tpr + weight * (upper_tpr - lower_tpr))
    return out


def _roc_curve_for_case(scores: dict[str, float], gt: set[str]) -> list[dict[str, float]]:
    if not scores or not gt:
        return []
    normalized_scores = {_normalize_trace_path(k): float(v) for k, v in scores.items()}
    gt_norm = {_normalize_trace_path(x) for x in gt}
    thresholds = [float("inf")] + sorted({float(score) for score in normalized_scores.values()}, reverse=True) + [float("-inf")]
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        point = _case_roc_point(normalized_scores, gt_norm, threshold)
        if point is None:
            continue
        tpr, fpr = point
        curve.append({"threshold": threshold, "tpr": tpr, "fpr": fpr})
    return curve


def _mean_tpr_at_fprs(
    per_case_scores: list[tuple[dict[str, float], set[str]]],
    fpr_grid: list[float],
) -> list[float]:
    valid_cases = [(scores, gt) for scores, gt in per_case_scores if scores and gt]
    if not valid_cases:
        return [0.0 for _ in fpr_grid]
    per_case_interp: list[list[float]] = []
    for scores, gt in valid_cases:
        curve = _roc_curve_for_case(scores, gt)
        if not curve:
            continue
        per_case_interp.append(_interp_tpr_at_fprs(curve, fpr_grid))
    if not per_case_interp:
        return [0.0 for _ in fpr_grid]
    return [sum(sample[idx] for sample in per_case_interp) / len(per_case_interp) for idx in range(len(fpr_grid))]


def _average_precision_from_pr_curve(curve: list[dict[str, float]] | None) -> float | None:
    if not curve:
        return None
    points = sorted(
        (
            max(0.0, min(1.0, float(point.get("recall", 0.0)))),
            max(0.0, min(1.0, float(point.get("precision", 0.0)))),
        )
        for point in curve
    )
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in points:
        if recall <= prev_recall:
            continue
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def _stddev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _bootstrap_mean_se(
    values: list[float],
    *,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    return _stddev(means)


def _bootstrap_flat_metric_se(
    pairs: list[tuple[int, float]],
    metric_fn,
    *,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> float | None:
    if not pairs:
        return None
    if len(pairs) == 1:
        return 0.0
    rng = random.Random(seed)
    n = len(pairs)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        metric_value = metric_fn(sample)
        if metric_value is not None:
            values.append(float(metric_value))
    return _stddev(values)


def _bootstrap_pr_band(
    per_case_scores: list[tuple[dict[str, float], set[str]]],
    *,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> tuple[list[float], list[float], list[float], list[float], float | None]:
    valid_cases = [(scores, gt) for scores, gt in per_case_scores if scores and gt]
    if not valid_cases:
        grid = [i / 20.0 for i in range(21)]
        zeros = [0.0 for _ in grid]
        return grid, zeros, zeros, zeros, None
    ap, _, _, _ = _average_curves(valid_cases)
    recall_grid = [i / 20.0 for i in range(21)]
    mean_precision = _mean_precision_at_recalls(valid_cases, recall_grid)
    rng = random.Random(seed)
    samples: list[list[float]] = []
    n = len(valid_cases)
    for _ in range(n_bootstrap):
        sampled = [valid_cases[rng.randrange(n)] for _ in range(n)]
        samples.append(_mean_precision_at_recalls(sampled, recall_grid))
    if not samples:
        return recall_grid, mean_precision, mean_precision[:], mean_precision[:], ap
    lower: list[float] = []
    upper: list[float] = []
    for idx in range(len(recall_grid)):
        vals = sorted(sample[idx] for sample in samples)
        lo_idx = max(0, int(0.025 * len(vals)) - 1)
        hi_idx = min(len(vals) - 1, int(0.975 * len(vals)))
        lower.append(vals[lo_idx])
        upper.append(vals[hi_idx])
    return recall_grid, mean_precision, lower, upper, ap


def _bootstrap_roc_band(
    per_case_scores: list[tuple[dict[str, float], set[str]]],
    *,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> tuple[list[float], list[float], list[float], float | None]:
    valid_cases = [(scores, gt) for scores, gt in per_case_scores if scores and gt]
    fpr_grid = [i / 20.0 for i in range(21)]
    if not valid_cases:
        zeros = [0.0 for _ in fpr_grid]
        return fpr_grid, zeros, zeros, None
    _, _, roc_auc, _ = _average_curves(valid_cases)
    rng = random.Random(seed)
    samples: list[list[float]] = []
    n = len(valid_cases)
    for _ in range(n_bootstrap):
        sampled = [valid_cases[rng.randrange(n)] for _ in range(n)]
        samples.append(_mean_tpr_at_fprs(sampled, fpr_grid))
    if not samples:
        zeros = [0.0 for _ in fpr_grid]
        return fpr_grid, zeros, zeros, roc_auc
    lower: list[float] = []
    upper: list[float] = []
    for idx in range(len(fpr_grid)):
        vals = sorted(sample[idx] for sample in samples)
        lo_idx = max(0, int(0.025 * len(vals)) - 1)
        hi_idx = min(len(vals) - 1, int(0.975 * len(vals)))
        lower.append(vals[lo_idx])
        upper.append(vals[hi_idx])
    return fpr_grid, lower, upper, roc_auc


def _row_trace_scores(row: dict[str, Any]) -> dict[str, float]:
    scores = row.get("scoring", {}).get("trace_scores") or {}
    if scores:
        return {_normalize_trace_path(k): float(v) for k, v in scores.items()}
    tests = row.get("tests") or []
    if tests:
        meta_scores = (tests[0].get("metadata") or {}).get("trace_scores") or {}
        if meta_scores:
            return {_normalize_trace_path(k): float(v) for k, v in meta_scores.items()}
    return {}


def _row_explicit_trace_scores(row: dict[str, Any]) -> dict[str, float]:
    tests = row.get("tests") or []
    test = tests[0] if tests else {}
    metadata = test.get("metadata") or {}
    meta_scores = metadata.get("trace_scores") or {}
    if meta_scores:
        return {
            _normalize_trace_path(str(path)): max(0.0, min(1.0, float(score)))
            for path, score in meta_scores.items()
        }
    evidence_text = (
        metadata.get("evidence_text")
        or (test.get("evidence") if isinstance(test.get("evidence"), str) else "")
        or ""
    )
    raw_text = str(evidence_text)
    match = _TRACE_SCORES_BLOCK_RE.search(raw_text)
    block = match.group(1) if match else ""
    if not block:
        match = _TRACE_SCORES_SECTION_RE.search(raw_text)
        block = match.group(1) if match else ""
    if not block:
        return {}
    scores: dict[str, float] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.rsplit(None, 1)
        if len(parts) < 2:
            continue
        try:
            score = float(parts[-1].strip())
        except ValueError:
            continue
        scores[_normalize_trace_path(parts[0].strip())] = max(0.0, min(1.0, score))
    return scores


def _row_all_trace_files(row: dict[str, Any]) -> list[str]:
    gt = row.get("ground_truth") or {}
    trace_files = gt.get("trace_files") or gt.get("harmful_trace_files") or []
    normalized = [_normalize_trace_path(str(path)) for path in trace_files if str(path).strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for path in normalized:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _row_trace_scores_for_metrics(
    row: dict[str, Any],
    *,
    zero_fill_if_missing: bool,
) -> dict[str, float]:
    scores = _row_trace_scores(row)
    if scores or not zero_fill_if_missing:
        return scores
    return {trace_path: 0.0 for trace_path in _row_all_trace_files(row)}


def _row_zero_unmentioned_trace_scores(
    row: dict[str, Any],
    *,
    fallback_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    explicit_scores = _row_explicit_trace_scores(row)
    all_trace_files = _row_all_trace_files(row)
    if not explicit_scores and not fallback_scores and not all_trace_files:
        return {}
    zero_filled = {trace_path: 0.0 for trace_path in all_trace_files}
    if explicit_scores:
        zero_filled.update(explicit_scores)
    elif fallback_scores:
        zero_filled.update({_normalize_trace_path(k): float(v) for k, v in fallback_scores.items()})
    return zero_filled


def _meerkat_fallback_scores_by_case(
    agent_path: Path,
    monitor_path: Path | None,
) -> dict[str, dict[str, float]]:
    if monitor_path is not None and monitor_path.is_file():
        monitor_cases, _ = _collect_case_rows(monitor_path)
        return {case_id: scores for case_id, scores, _ in monitor_cases}

    rows = _load_rows(agent_path)
    fallback: dict[str, dict[str, float]] = {}
    for idx, row in enumerate(rows):
        case_id = _row_case_id(row, idx)
        repo_scores = _load_stage1_scores_from_repo(Path(str(row.get("repo") or ""))) or {}
        if repo_scores:
            fallback[case_id] = repo_scores
    return fallback


def _collect_stage2_case_curves(path: Path) -> tuple[list[list[dict[str, float]]], int]:
    rows = _load_rows(path)
    curves: list[list[dict[str, float]]] = []
    for row in rows:
        curve = row.get("scoring", {}).get("pr_curve")
        if curve:
            curves.append(curve)
    return curves, len(rows)


def _collect_stage2_cases(path: Path) -> tuple[list[tuple[dict[str, float], set[str]]], int]:
    rows = _load_rows(path)
    stage2_cases: list[tuple[dict[str, float], set[str]]] = []
    for row in rows:
        gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
        scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=bool(gt))
        if scores and gt:
            stage2_cases.append((scores, gt))
    return stage2_cases, len(rows)


def _collect_meerkat_zero_unmentioned_cases(
    path: Path,
    monitor_path: Path | None = None,
) -> tuple[list[tuple[dict[str, float], set[str]]], int]:
    rows = _load_rows(path)
    fallback_by_case = _meerkat_fallback_scores_by_case(path, monitor_path)
    cases: list[tuple[dict[str, float], set[str]]] = []
    for idx, row in enumerate(rows):
        case_id = _row_case_id(row, idx)
        gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
        scores = _row_zero_unmentioned_trace_scores(row, fallback_scores=fallback_by_case.get(case_id))
        if scores and gt:
            cases.append((scores, gt))
    return cases, len(rows)


def _collect_case_score_pairs(path: Path) -> tuple[list[tuple[int, float]], int]:
    rows = _load_rows(path)
    pairs: list[tuple[int, float]] = []
    for row in rows:
        scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=True)
        if not scores:
            continue
        gt_case_label = 1 if bool(row.get("ground_truth", {}).get("has_distributed_misuse")) else 0
        pairs.append((gt_case_label, max(scores.values())))
    return pairs, len(rows)


def _collect_meerkat_zero_unmentioned_case_pairs(
    path: Path,
    monitor_path: Path | None = None,
) -> tuple[list[tuple[int, float]], int]:
    rows = _load_rows(path)
    fallback_by_case = _meerkat_fallback_scores_by_case(path, monitor_path)
    pairs: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        case_id = _row_case_id(row, idx)
        scores = _row_zero_unmentioned_trace_scores(row, fallback_scores=fallback_by_case.get(case_id))
        if not scores:
            continue
        gt_case_label = 1 if bool(row.get("ground_truth", {}).get("has_distributed_misuse")) else 0
        pairs.append((gt_case_label, max(scores.values())))
    return pairs, len(rows)


def _row_case_id(row: dict[str, Any], fallback_index: int) -> str:
    return str(row.get("case_id") or row.get("repo_name") or fallback_index)


def _collect_case_rows(path: Path) -> tuple[list[tuple[str, dict[str, float], set[str]]], int]:
    rows = _load_rows(path)
    cases: list[tuple[str, dict[str, float], set[str]]] = []
    for idx, row in enumerate(rows):
        gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
        scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=bool(gt))
        if scores and gt:
            cases.append((_row_case_id(row, idx), scores, gt))
    return cases, len(rows)


def _collect_meerkat_zero_unmentioned_case_rows(
    path: Path,
    monitor_path: Path | None = None,
) -> tuple[list[tuple[str, dict[str, float], set[str]]], int]:
    rows = _load_rows(path)
    fallback_by_case = _meerkat_fallback_scores_by_case(path, monitor_path)
    cases: list[tuple[str, dict[str, float], set[str]]] = []
    for idx, row in enumerate(rows):
        case_id = _row_case_id(row, idx)
        gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
        scores = _row_zero_unmentioned_trace_scores(row, fallback_scores=fallback_by_case.get(case_id))
        if scores and gt:
            cases.append((case_id, scores, gt))
    return cases, len(rows)


def _merged_cases_with_monitor(agent_path: Path, monitor_path: Path | None) -> tuple[list[tuple[dict[str, float], set[str]]], int]:
    agent_cases, case_count = _collect_case_rows(agent_path)
    if monitor_path is None or not monitor_path.is_file():
        return [(scores, gt) for _, scores, gt in agent_cases], case_count

    monitor_cases, _ = _collect_case_rows(monitor_path)
    monitor_by_case = {case_id: scores for case_id, scores, _ in monitor_cases}
    merged_cases: list[tuple[dict[str, float], set[str]]] = []
    for case_id, agent_scores, gt in agent_cases:
        merged = dict(agent_scores)
        monitor_scores = monitor_by_case.get(case_id) or {}
        for trace_key, score in monitor_scores.items():
            merged[trace_key] = max(float(merged.get(trace_key, 0.0)), float(score))
        merged_cases.append((merged, gt))
    return merged_cases, case_count


def _merged_case_score_pairs_with_monitor(
    agent_path: Path,
    monitor_path: Path | None,
) -> tuple[list[tuple[int, float]], int]:
    rows = _load_rows(agent_path)
    monitor_cases, _ = _collect_monitor_case_rows(agent_path, monitor_path)
    monitor_by_case = {case_id: scores for case_id, scores, _ in monitor_cases}
    pairs: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        case_id = _row_case_id(row, idx)
        agent_scores = _row_trace_scores_for_metrics(
            row,
            zero_fill_if_missing=True,
        )
        merged_scores = dict(agent_scores)
        for trace_key, score in (monitor_by_case.get(case_id) or {}).items():
            merged_scores[trace_key] = max(float(merged_scores.get(trace_key, 0.0)), float(score))
        if not merged_scores:
            continue
        gt_case_label = 1 if bool(row.get("ground_truth", {}).get("has_distributed_misuse")) else 0
        pairs.append((gt_case_label, max(merged_scores.values())))
    return pairs, len(rows)


def _collect_stage1_case_rows(path: Path) -> tuple[list[tuple[str, dict[str, float], set[str]]], int]:
    rows = _load_rows(path)
    cases: list[tuple[str, dict[str, float], set[str]]] = []
    for idx, row in enumerate(rows):
        gt = {_normalize_trace_path(x) for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])}
        scores = _load_stage1_scores_from_repo(Path(str(row.get("repo") or ""))) or {}
        if not scores and gt:
            scores = _row_trace_scores_for_metrics(row, zero_fill_if_missing=True)
        if scores and gt:
            cases.append((_row_case_id(row, idx), scores, gt))
    return cases, len(rows)


def _collect_monitor_case_rows(agent_path: Path, monitor_path: Path | None) -> tuple[list[tuple[str, dict[str, float], set[str]]], int]:
    if monitor_path is not None and monitor_path.is_file():
        return _collect_case_rows(monitor_path)
    return _collect_stage1_case_rows(agent_path)


def _merged_case_rows_with_monitor(
    agent_path: Path,
    monitor_path: Path | None,
) -> tuple[list[tuple[str, dict[str, float], set[str]]], int]:
    agent_cases, case_count = _collect_case_rows(agent_path)
    monitor_cases, _ = _collect_monitor_case_rows(agent_path, monitor_path)
    monitor_by_case = {case_id: scores for case_id, scores, _ in monitor_cases}
    merged_cases: list[tuple[str, dict[str, float], set[str]]] = []
    for case_id, agent_scores, gt in agent_cases:
        merged = dict(agent_scores)
        monitor_scores = monitor_by_case.get(case_id) or {}
        for trace_key, score in monitor_scores.items():
            merged[trace_key] = max(float(merged.get(trace_key, 0.0)), float(score))
        merged_cases.append((case_id, merged, gt))
    return merged_cases, case_count


def _paper_curve_candidates(
    results_dir: Path,
    explicit_inputs: list[Path] | None = None,
    *,
    domain: str = "cyber",
) -> list[tuple[Path, str, str, int]]:
    explicit_set = {path.resolve() for path in explicit_inputs or []}

    def _want(path: Path) -> bool:
        return not explicit_set or path.resolve() in explicit_set

    def _add_if_present(path: Path, *, model_label: str, method_label: str, bg: int) -> None:
        if path.is_file() and _want(path):
            candidates.append((path, model_label, method_label, bg))
            return
        if not explicit_set and not path.is_file():
            _warn_missing_dm_result(path, domain=domain, bg=bg, model_label=model_label, method_label=method_label)

    candidates: list[tuple[Path, str, str, int]] = []
    if domain == "cyber":
        for bg, model_label in ((20, "gpt-5.4-mini"), (100, "gpt-5.4-mini"), (20, "Qwen-3.5"), (100, "Qwen-3.5")):
            _add_if_present(
                _dm_meerkat_result_path(results_dir, domain="cyber", bg=bg, model_label=model_label),
                model_label=model_label,
                method_label="Meerkat",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_bayesian_result_candidates(results_dir, domain="cyber", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Bayesian",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_monitor_result_candidates(results_dir, domain="cyber", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Monitor",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_buffer_result_candidates(results_dir, domain="cyber", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Buffer",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_naive_agent_result_candidates(results_dir, domain="cyber", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Naive Agent",
                bg=bg,
            )
    elif domain == "bio":
        for bg, model_label in ((20, "gpt-5.4-mini"), (100, "gpt-5.4-mini"), (20, "Qwen-3.5"), (100, "Qwen-3.5")):
            _add_if_present(
                _dm_meerkat_result_path(results_dir, domain="bio", bg=bg, model_label=model_label),
                model_label=model_label,
                method_label="Meerkat",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_bayesian_result_candidates(results_dir, domain="bio", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Bayesian",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_monitor_result_candidates(results_dir, domain="bio", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Monitor",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_buffer_result_candidates(results_dir, domain="bio", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Buffer",
                bg=bg,
            )
        for bg, model_label in (
            (20, "gpt-5.4-mini"),
            (100, "gpt-5.4-mini"),
            (20, "Qwen-3.5"),
            (100, "Qwen-3.5"),
        ):
            path = _dm_naive_agent_result_candidates(results_dir, domain="bio", bg=bg, model_label=model_label)[0]
            _add_if_present(
                path,
                model_label=model_label,
                method_label="Naive Agent",
                bg=bg,
            )
    return candidates


def _paper_curve_runs(
    results_dir: Path,
    explicit_inputs: list[Path] | None = None,
    *,
    domain: str = "cyber",
    bootstrap_samples: int = 100,
    bootstrap_cache_dir: Path | None = None,
) -> list[PaperCurveRun]:
    candidates = _paper_curve_candidates(results_dir, explicit_inputs, domain=domain)
    monitor_paths: dict[tuple[str, int], Path] = {}
    for candidate_path, model_label, method_label, bg in candidates:
        if method_label == "Monitor":
            monitor_paths[(model_label, bg)] = candidate_path

    runs: list[PaperCurveRun] = []
    recall_grid = [i / 20.0 for i in range(21)]
    for path, model_label, method_label, bg in candidates:
        monitor_path = monitor_paths.get((model_label, bg))
        cache_payload = {
            "domain": domain,
            "path": _path_signature(path),
            "method_label": method_label,
            "model_label": model_label,
            "background_multiplier": bg,
            "bootstrap_samples": bootstrap_samples,
            "seed": bg + len(runs) * 17,
            "ap_seed": bg + len(runs) * 31,
            "monitor_path": _path_signature(monitor_path),
        }
        cached = _read_bootstrap_cache(
            bootstrap_cache_dir,
            kind="paper_curve_run",
            payload=cache_payload,
        )
        if cached is not None:
            runs.append(
                PaperCurveRun(
                    path=path,
                    model_label=model_label,
                    method_label=method_label,
                    background_multiplier=bg,
                    case_count=int(cached["case_count"]),
                    average_precision=(None if cached["average_precision"] is None else float(cached["average_precision"])),
                    average_precision_bootstrap_se=(
                        None
                        if cached["average_precision_bootstrap_se"] is None
                        else float(cached["average_precision_bootstrap_se"])
                    ),
                    recall_grid=[float(x) for x in cached["recall_grid"]],
                    precision_curve=[float(x) for x in cached["precision_curve"]],
                    precision_lower=[float(x) for x in cached["precision_lower"]],
                    precision_upper=[float(x) for x in cached["precision_upper"]],
                )
            )
            continue
        if method_label == "Meerkat":
            per_case_scores, case_count = _collect_meerkat_zero_unmentioned_cases(
                path,
                monitor_path,
            )
        else:
            per_case_scores, case_count = _collect_stage2_cases(path)
        if not per_case_scores:
            continue
        ap_values = [_average_precision_for_case(scores, gt) for scores, gt in per_case_scores]
        ap_values = [float(ap) for ap in ap_values if ap is not None]
        recall_grid, mean_precision, lower, upper, ap = _bootstrap_pr_band(
            per_case_scores,
            n_bootstrap=bootstrap_samples,
            seed=bg + len(runs) * 17,
        )
        ap_bootstrap_se = _bootstrap_mean_se(ap_values, n_bootstrap=bootstrap_samples, seed=bg + len(runs) * 31)
        runs.append(
            PaperCurveRun(
                path=path,
                model_label=model_label,
                method_label=method_label,
                background_multiplier=bg,
                case_count=case_count,
                average_precision=ap,
                average_precision_bootstrap_se=ap_bootstrap_se,
                recall_grid=recall_grid,
                precision_curve=mean_precision,
                precision_lower=lower,
                precision_upper=upper,
            )
        )
        _write_bootstrap_cache(
            bootstrap_cache_dir,
            kind="paper_curve_run",
            payload=cache_payload,
            value={
                "case_count": case_count,
                "average_precision": ap,
                "average_precision_bootstrap_se": ap_bootstrap_se,
                "recall_grid": recall_grid,
                "precision_curve": mean_precision,
                "precision_lower": lower,
                "precision_upper": upper,
            },
        )
    return runs


def _paper_roc_runs(
    results_dir: Path,
    explicit_inputs: list[Path] | None = None,
    *,
    domain: str = "cyber",
    bootstrap_samples: int = 100,
    bootstrap_cache_dir: Path | None = None,
) -> list[PaperRocRun]:
    candidates = _paper_curve_candidates(results_dir, explicit_inputs, domain=domain)
    monitor_paths: dict[tuple[str, int], Path] = {}
    for candidate_path, model_label, method_label, bg in candidates:
        if method_label == "Monitor":
            monitor_paths[(model_label, bg)] = candidate_path

    runs: list[PaperRocRun] = []
    fpr_grid = [i / 20.0 for i in range(21)]
    for path, model_label, method_label, bg in candidates:
        monitor_path = monitor_paths.get((model_label, bg))
        cache_payload = {
            "domain": domain,
            "path": _path_signature(path),
            "method_label": method_label,
            "model_label": model_label,
            "background_multiplier": bg,
            "bootstrap_samples": bootstrap_samples,
            "seed": bg + len(runs) * 19,
            "auc_seed": bg + len(runs) * 37,
            "monitor_path": _path_signature(monitor_path),
        }
        cached = _read_bootstrap_cache(
            bootstrap_cache_dir,
            kind="paper_roc_run",
            payload=cache_payload,
        )
        if cached is not None:
            runs.append(
                PaperRocRun(
                    path=path,
                    model_label=model_label,
                    method_label=method_label,
                    background_multiplier=bg,
                    case_count=int(cached["case_count"]),
                    roc_auc=None if cached["roc_auc"] is None else float(cached["roc_auc"]),
                    roc_auc_bootstrap_se=(
                        None if cached["roc_auc_bootstrap_se"] is None else float(cached["roc_auc_bootstrap_se"])
                    ),
                    fpr_grid=[float(x) for x in cached["fpr_grid"]],
                    tpr_curve=[float(x) for x in cached["tpr_curve"]],
                    tpr_lower=[float(x) for x in cached["tpr_lower"]],
                    tpr_upper=[float(x) for x in cached["tpr_upper"]],
                )
            )
            continue
        if method_label == "Meerkat":
            per_case_scores, case_count = _collect_meerkat_zero_unmentioned_cases(
                path,
                monitor_path,
            )
        else:
            per_case_scores, case_count = _collect_stage2_cases(path)
        if not per_case_scores:
            continue
        roc_values = [_roc_auc_for_case(scores, gt) for scores, gt in per_case_scores]
        roc_values = [float(auc) for auc in roc_values if auc is not None]
        fpr_grid, lower, upper, roc_auc = _bootstrap_roc_band(
            per_case_scores,
            n_bootstrap=bootstrap_samples,
            seed=bg + len(runs) * 19,
        )
        roc_auc_bootstrap_se = _bootstrap_mean_se(roc_values, n_bootstrap=bootstrap_samples, seed=bg + len(runs) * 37)
        tpr_curve = _mean_tpr_at_fprs(per_case_scores, fpr_grid)
        runs.append(
            PaperRocRun(
                path=path,
                model_label=model_label,
                method_label=method_label,
                background_multiplier=bg,
                case_count=case_count,
                roc_auc=roc_auc,
                roc_auc_bootstrap_se=roc_auc_bootstrap_se,
                fpr_grid=fpr_grid,
                tpr_curve=tpr_curve,
                tpr_lower=lower,
                tpr_upper=upper,
            )
        )
        _write_bootstrap_cache(
            bootstrap_cache_dir,
            kind="paper_roc_run",
            payload=cache_payload,
            value={
                "case_count": case_count,
                "roc_auc": roc_auc,
                "roc_auc_bootstrap_se": roc_auc_bootstrap_se,
                "fpr_grid": fpr_grid,
                "tpr_curve": tpr_curve,
                "tpr_lower": lower,
                "tpr_upper": upper,
            },
        )
    return runs


def _meerkat_vs_monitor_rocauc_points(
    results_dir: Path,
    input_paths: list[Path],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for domain in ("cyber", "bio"):
        candidates = _paper_curve_candidates(results_dir, input_paths, domain=domain)
        monitor_paths: dict[tuple[str, int], Path] = {}
        for candidate_path, model_label, method_label, bg in candidates:
            if method_label == "Monitor":
                monitor_paths[(model_label, bg)] = candidate_path
        for path, model_label, method_label, bg in candidates:
            if method_label != "Meerkat":
                continue
            monitor_path = monitor_paths.get((model_label, bg))
            meerkat_cases, _ = _collect_meerkat_zero_unmentioned_case_rows(path, monitor_path)
            monitor_cases, _ = _collect_monitor_case_rows(path, monitor_path)
            monitor_by_case = {case_id: scores for case_id, scores, _ in monitor_cases}
            for case_id, meerkat_scores, gt in meerkat_cases:
                monitor_scores = monitor_by_case.get(case_id)
                if not monitor_scores or not gt:
                    continue
                monitor_rocauc = _roc_auc_for_case(monitor_scores, gt)
                meerkat_rocauc = _roc_auc_for_case(meerkat_scores, gt)
                if monitor_rocauc is None or meerkat_rocauc is None:
                    continue
                points.append(
                    {
                        "domain": domain,
                        "model_label": model_label,
                        "background_multiplier": bg,
                        "case_id": case_id,
                        "monitor_rocauc": float(monitor_rocauc),
                        "meerkat_rocauc": float(meerkat_rocauc),
                    }
                )
    return points


def _method_metric_summary(
    path: Path,
    *,
    method_label: str,
    monitor_path: Path | None,
    metric_key: str,
    bootstrap_samples: int,
    seed: int,
    bootstrap_cache_dir: Path | None = None,
) -> tuple[float | None, float | None]:
    cache_payload = {
        "path": _path_signature(path),
        "method_label": method_label,
        "monitor_path": _path_signature(monitor_path),
        "metric_key": metric_key,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    cached = _read_bootstrap_cache(
        bootstrap_cache_dir,
        kind="method_metric_summary",
        payload=cache_payload,
    )
    if cached is not None:
        return (
            None if cached["mean_value"] is None else float(cached["mean_value"]),
            None if cached["se_value"] is None else float(cached["se_value"]),
        )

    if metric_key == "trace_ap":
        if method_label == "Meerkat":
            per_case_scores, _ = _collect_meerkat_zero_unmentioned_cases(path, monitor_path)
        else:
            per_case_scores, _ = _collect_stage2_cases(path)
        if not per_case_scores:
            return None, None
        values = [_average_precision_for_case(scores, gt) for scores, gt in per_case_scores]
        values = [float(value) for value in values if value is not None]
        if not values:
            return None, None
        mean_value = sum(values) / len(values)
        se_value = _bootstrap_mean_se(values, n_bootstrap=bootstrap_samples, seed=seed)
        _write_bootstrap_cache(
            bootstrap_cache_dir,
            kind="method_metric_summary",
            payload=cache_payload,
            value={"mean_value": mean_value, "se_value": se_value},
        )
        return mean_value, se_value

    if metric_key == "trace_roc_auc":
        if method_label == "Meerkat":
            per_case_scores, _ = _collect_meerkat_zero_unmentioned_cases(path, monitor_path)
        else:
            per_case_scores, _ = _collect_stage2_cases(path)
        if not per_case_scores:
            return None, None
        values = [_roc_auc_for_case(scores, gt) for scores, gt in per_case_scores]
        values = [float(value) for value in values if value is not None]
        if not values:
            return None, None
        mean_value = sum(values) / len(values)
        se_value = _bootstrap_mean_se(values, n_bootstrap=bootstrap_samples, seed=seed)
        _write_bootstrap_cache(
            bootstrap_cache_dir,
            kind="method_metric_summary",
            payload=cache_payload,
            value={"mean_value": mean_value, "se_value": se_value},
        )
        return mean_value, se_value

    if metric_key == "case_ap":
        if method_label == "Meerkat":
            case_pairs, _ = _collect_meerkat_zero_unmentioned_case_pairs(path, monitor_path)
        else:
            case_pairs, _ = _collect_case_score_pairs(path)
        if not case_pairs:
            return None, None
        mean_value = _flat_average_precision(case_pairs)
        if mean_value is None:
            return None, None
        se_value = _bootstrap_flat_metric_se(
            case_pairs,
            _flat_average_precision,
            n_bootstrap=bootstrap_samples,
            seed=seed,
        )
        _write_bootstrap_cache(
            bootstrap_cache_dir,
            kind="method_metric_summary",
            payload=cache_payload,
            value={"mean_value": mean_value, "se_value": se_value},
        )
        return mean_value, se_value

    if metric_key == "case_roc_auc":
        if method_label == "Meerkat":
            case_pairs, _ = _collect_meerkat_zero_unmentioned_case_pairs(path, monitor_path)
        else:
            case_pairs, _ = _collect_case_score_pairs(path)
        if not case_pairs:
            return None, None
        mean_value = _flat_roc_auc(case_pairs)
        if mean_value is None:
            return None, None
        se_value = _bootstrap_flat_metric_se(
            case_pairs,
            _flat_roc_auc,
            n_bootstrap=bootstrap_samples,
            seed=seed,
        )
        _write_bootstrap_cache(
            bootstrap_cache_dir,
            kind="method_metric_summary",
            payload=cache_payload,
            value={"mean_value": mean_value, "se_value": se_value},
        )
        return mean_value, se_value

    raise ValueError(f"Unsupported metric_key: {metric_key}")


def _plot_ap_bars(
    metrics: list[SettingMetrics],
    *,
    figures_dir: Path,
    figure_formats: list[str],
) -> list[Path]:
    domain_order = ["cyber", "bio"]
    bg_order = sorted({m.background_multiplier for m in metrics})
    fig, axes = plt.subplots(2, len(domain_order), figsize=(7.2, 5.0), sharey="row")
    if len(domain_order) == 1:
        axes = [[axes[0]], [axes[1]]]

    stage1_color = "#0072B2"
    stage2_color = "#D55E00"
    output_paths: list[Path] = []

    for col_idx, domain in enumerate(domain_order):
        ax_pr = axes[0][col_idx]
        ax_roc = axes[1][col_idx]
        domain_metrics = [m for m in metrics if m.domain == domain]
        domain_metrics_by_bg = {m.background_multiplier: m for m in domain_metrics}
        xs = list(range(len(bg_order)))
        width = 0.36
        stage1_vals = [
            (
                math.nan
                if domain_metrics_by_bg.get(bg) is None or domain_metrics_by_bg[bg].stage1_ap is None
                else float(domain_metrics_by_bg[bg].stage1_ap)
            )
            for bg in bg_order
        ]
        stage2_vals = [
            (
                math.nan
                if domain_metrics_by_bg.get(bg) is None or domain_metrics_by_bg[bg].stage2_ap is None
                else float(domain_metrics_by_bg[bg].stage2_ap)
            )
            for bg in bg_order
        ]
        stage1_roc_vals = [
            (
                math.nan
                if domain_metrics_by_bg.get(bg) is None or domain_metrics_by_bg[bg].stage1_roc_auc is None
                else float(domain_metrics_by_bg[bg].stage1_roc_auc)
            )
            for bg in bg_order
        ]
        stage2_roc_vals = [
            (
                math.nan
                if domain_metrics_by_bg.get(bg) is None or domain_metrics_by_bg[bg].stage2_roc_auc is None
                else float(domain_metrics_by_bg[bg].stage2_roc_auc)
            )
            for bg in bg_order
        ]
        ax_pr.bar([x - width / 2 for x in xs], stage1_vals, width=width, color=stage1_color, label="Judge")
        ax_pr.bar([x + width / 2 for x in xs], stage2_vals, width=width, color=stage2_color, label="AT")
        ax_pr.set_title("Cyber" if domain == "cyber" else "Bio")
        ax_pr.set_xticks(xs)
        ax_pr.set_xticklabels([f"{bg}x" for bg in bg_order])
        ax_pr.set_xlabel("Background Multiplier")
        ax_pr.set_ylim(0.0, 1.0)
        ax_pr.grid(axis="y", alpha=0.25)
        ax_roc.bar([x - width / 2 for x in xs], stage1_roc_vals, width=width, color=stage1_color, label="Judge")
        ax_roc.bar([x + width / 2 for x in xs], stage2_roc_vals, width=width, color=stage2_color, label="AT")
        ax_roc.set_xticks(xs)
        ax_roc.set_xticklabels([f"{bg}x" for bg in bg_order])
        ax_roc.set_xlabel("Background Multiplier")
        ax_roc.set_ylim(0.0, 1.0)
        ax_roc.grid(axis="y", alpha=0.25)

    axes[0][0].set_ylabel("Trace AUPRC")
    axes[1][0].set_ylabel("Trace AUROC")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_base = figures_dir / "dm_trace_average_precision_by_setting"
    for fmt in figure_formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, bbox_inches="tight")
        output_paths.append(out_path)
    plt.close(fig)
    return output_paths


def _plot_pr_grid(
    metrics: list[SettingMetrics],
    *,
    figures_dir: Path,
    figure_formats: list[str],
) -> list[Path]:
    domain_order = ["cyber"]
    bg_order = sorted({m.background_multiplier for m in metrics})
    fig, axes = plt.subplots(
        2,
        len(bg_order),
        figsize=(2.35 * len(bg_order), 4.2),
        sharex=True,
        sharey="row",
    )
    if len(bg_order) == 1:
        axes = [[axes[0]], [axes[1]]]

    stage1_color = "#0072B2"
    stage2_color = "#D55E00"
    output_paths: list[Path] = []

    metrics_by_key = {(m.domain, m.background_multiplier): m for m in metrics}
    domain = "cyber"
    for col_idx, bg in enumerate(bg_order):
        pr_ax = axes[0][col_idx]
        roc_ax = axes[1][col_idx]
        m = metrics_by_key.get((domain, bg))
        if not m:
            pr_ax.axis("off")
            roc_ax.axis("off")
            continue
        if m.stage1_curve:
            pr_ax.plot(
                [pt["recall"] for pt in m.stage1_curve],
                [pt["precision"] for pt in m.stage1_curve],
                color=stage1_color,
                linewidth=2.0,
                label=f"Judge (AP={m.stage1_ap:.2f})" if m.stage1_ap is not None else "Judge",
            )
        if m.stage2_curve:
            pr_ax.plot(
                [pt["recall"] for pt in m.stage2_curve],
                [pt["precision"] for pt in m.stage2_curve],
                color=stage2_color,
                linewidth=2.0,
                label=f"AT (AP={m.stage2_ap:.2f})" if m.stage2_ap is not None else "AT",
            )
        if m.stage1_roc_curve:
            roc_ax.plot(
                [pt["fpr"] for pt in m.stage1_roc_curve],
                [pt["tpr"] for pt in m.stage1_roc_curve],
                color=stage1_color,
                linewidth=2.0,
                label=f"Judge (AUC={m.stage1_roc_auc:.2f})" if m.stage1_roc_auc is not None else "Judge",
            )
        if m.stage2_roc_curve:
            roc_ax.plot(
                [pt["fpr"] for pt in m.stage2_roc_curve],
                [pt["tpr"] for pt in m.stage2_roc_curve],
                color=stage2_color,
                linewidth=2.0,
                label=f"AT (AUC={m.stage2_roc_auc:.2f})" if m.stage2_roc_auc is not None else "AT",
            )
        pr_ax.set_title(f"Cyber bg={bg}x", fontsize=10)
        pr_ax.set_xlim(0.0, 1.0)
        pr_ax.set_ylim(0.0, 1.02)
        pr_ax.grid(alpha=0.25)
        roc_ax.set_xlim(0.0, 1.0)
        roc_ax.set_ylim(0.0, 1.02)
        roc_ax.grid(alpha=0.25)
        roc_ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.8, linestyle=":")
        pr_ax.set_xlabel("Recall")
        roc_ax.set_xlabel("False Positive Rate")
        if col_idx == 0:
            pr_ax.set_ylabel("Precision")
            roc_ax.set_ylabel("True Positive Rate")
        handles, labels = pr_ax.get_legend_handles_labels()
        if handles:
            pr_ax.legend(loc="lower left", fontsize=8, frameon=False)
        handles, labels = roc_ax.get_legend_handles_labels()
        if handles:
            roc_ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    out_base = figures_dir / "dm_precision_recall_curves"
    for fmt in figure_formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, bbox_inches="tight")
        output_paths.append(out_path)
    plt.close(fig)
    return output_paths


def _plot_case_pr_grid(
    metrics: list[SettingMetrics],
    *,
    figures_dir: Path,
    figure_formats: list[str],
) -> list[Path]:
    domain = "cyber"
    bg_order = sorted({m.background_multiplier for m in metrics})
    fig, axes = plt.subplots(
        2,
        len(bg_order),
        figsize=(2.35 * len(bg_order), 4.2),
        sharex=True,
        sharey="row",
    )
    if len(bg_order) == 1:
        axes = [[axes[0]], [axes[1]]]

    stage1_color = "#0072B2"
    stage2_color = "#D55E00"
    output_paths: list[Path] = []
    metrics_by_key = {(m.domain, m.background_multiplier): m for m in metrics}

    for col_idx, bg in enumerate(bg_order):
        pr_ax = axes[0][col_idx]
        roc_ax = axes[1][col_idx]
        m = metrics_by_key.get((domain, bg))
        if not m:
            pr_ax.axis("off")
            roc_ax.axis("off")
            continue
        if m.stage1_case_curve:
            pr_ax.plot(
                [pt["recall"] for pt in m.stage1_case_curve],
                [pt["precision"] for pt in m.stage1_case_curve],
                color=stage1_color,
                linewidth=2.0,
                label=f"Judge (AP={m.stage1_case_ap:.2f})" if m.stage1_case_ap is not None else "Judge",
            )
        if m.stage2_case_curve:
            pr_ax.plot(
                [pt["recall"] for pt in m.stage2_case_curve],
                [pt["precision"] for pt in m.stage2_case_curve],
                color=stage2_color,
                linewidth=2.0,
                label=f"AT (AP={m.stage2_case_ap:.2f})" if m.stage2_case_ap is not None else "AT",
            )
        if m.stage1_case_roc_curve:
            roc_ax.plot(
                [pt["fpr"] for pt in m.stage1_case_roc_curve],
                [pt["tpr"] for pt in m.stage1_case_roc_curve],
                color=stage1_color,
                linewidth=2.0,
                label=f"Judge (AUC={m.stage1_case_roc_auc:.2f})" if m.stage1_case_roc_auc is not None else "Judge",
            )
        if m.stage2_case_roc_curve:
            roc_ax.plot(
                [pt["fpr"] for pt in m.stage2_case_roc_curve],
                [pt["tpr"] for pt in m.stage2_case_roc_curve],
                color=stage2_color,
                linewidth=2.0,
                label=f"AT (AUC={m.stage2_case_roc_auc:.2f})" if m.stage2_case_roc_auc is not None else "AT",
            )
        pr_ax.set_title(f"Cyber bg={bg}x", fontsize=10)
        pr_ax.set_xlim(0.0, 1.0)
        pr_ax.set_ylim(0.0, 1.02)
        pr_ax.grid(alpha=0.25)
        roc_ax.set_xlim(0.0, 1.0)
        roc_ax.set_ylim(0.0, 1.02)
        roc_ax.grid(alpha=0.25)
        roc_ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.8, linestyle=":")
        pr_ax.set_xlabel("Recall")
        roc_ax.set_xlabel("False Positive Rate")
        if col_idx == 0:
            pr_ax.set_ylabel("Precision")
            roc_ax.set_ylabel("True Positive Rate")
        handles, labels = pr_ax.get_legend_handles_labels()
        if handles:
            pr_ax.legend(loc="lower left", fontsize=8, frameon=False)
        handles, labels = roc_ax.get_legend_handles_labels()
        if handles:
            roc_ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    out_base = figures_dir / "dm_case_precision_recall_curves"
    for fmt in figure_formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, bbox_inches="tight")
        output_paths.append(out_path)
    plt.close(fig)
    return output_paths


def _plot_main_paper_pr_figure(
    *,
    results_dir: Path,
    figures_dir: Path,
    figure_formats: list[str],
    input_paths: list[Path],
    recompute_stage1_if_missing: bool,
    stage1_concurrency: int,
    stage1_cache_dir: Path,
    bootstrap_samples: int,
    bootstrap_cache_dir: Path | None,
) -> list[Path]:
    runs = _paper_curve_runs(
        results_dir,
        input_paths,
        bootstrap_samples=bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )
    if not runs:
        return []
    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    ):
        fig = plt.figure(figsize=(4.2, 3.65))
        grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
        axes = {
            ("gpt-5.4-mini", 20): fig.add_subplot(grid[0, 0]),
            ("gpt-5.4-mini", 100): fig.add_subplot(grid[0, 1]),
            ("Qwen-3.5", 20): fig.add_subplot(grid[1, 0]),
            ("Qwen-3.5", 100): fig.add_subplot(grid[1, 1]),
        }
        method_order = ["Meerkat", "Monitor", "Bayesian", "Buffer"]
        method_colors = {
            "Meerkat": "#D55E00",
            "Monitor": "#CC79A7",
            "Bayesian": "#0072B2",
            "Buffer": "#009E73",
        }
        method_linestyles = {
            "Meerkat": "-",
            "Monitor": "-.",
            "Bayesian": ":",
            "Buffer": "--",
        }
        shared_legend_handles = [
            Line2D([0], [0], color=method_colors[method], linestyle=method_linestyles[method], linewidth=2.2)
            for method in method_order
        ]
        runs_by_panel: dict[tuple[str, int], list[PaperCurveRun]] = {}
        for run in runs:
            runs_by_panel.setdefault((run.model_label, run.background_multiplier), []).append(run)

        for panel_key, ax in axes.items():
            panel_runs = sorted(
                runs_by_panel.get(panel_key, []),
                key=lambda run: method_order.index(run.method_label) if run.method_label in method_order else 99,
            )
            if not panel_runs:
                ax.axis("off")
                continue
            for run in panel_runs:
                color = method_colors.get(run.method_label, "#555555")
                linestyle = method_linestyles.get(run.method_label, "-")
                ax.fill_between(
                    run.recall_grid,
                    run.precision_lower,
                    run.precision_upper,
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=1,
                )
                (line,) = ax.plot(
                    run.recall_grid,
                    run.precision_curve,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.2,
                    zorder=2,
                    label=run.method_label,
                )
            case_counts = {run.method_label: run.case_count for run in panel_runs}
            unique_counts = sorted(set(case_counts.values()))
            if len(unique_counts) == 1:
                ax.set_title(f"bg={panel_key[1]}x (n={unique_counts[0]})", fontsize=8.5, pad=4)
            else:
                ax.set_title(f"bg={panel_key[1]}x", fontsize=8.5, pad=4)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.25)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=7.0)
            if panel_key[0] == "Qwen-3.5":
                ax.set_xlabel("Recall", fontsize=8.5)
            if panel_key != ("gpt-5.4-mini", 20) and panel_key != ("Qwen-3.5", 20):
                ax.tick_params(labelleft=False)

        legend_ax = axes.get(("gpt-5.4-mini", 100))
        if legend_ax and legend_ax.axison:
            legend_ax.legend(
                shared_legend_handles,
                method_order,
                loc="upper right",
                frameon=False,
                fontsize=6.4,
                handlelength=1.9,
                borderaxespad=0.2,
                labelspacing=0.2,
            )

        top_left = axes.get(("gpt-5.4-mini", 20))
        if top_left and top_left.axison:
            top_left.set_ylabel("gpt-5.4-mini\nPrecision", fontsize=8.5)
        bottom_left = axes.get(("Qwen-3.5", 20))
        if bottom_left and bottom_left.axison:
            bottom_left.set_ylabel("Qwen-3.5\nPrecision", fontsize=8.5)

        fig.subplots_adjust(top=0.92, left=0.17, right=0.98, bottom=0.12)

        output_paths: list[Path] = []
        out_base = figures_dir / "dm_cyber_paper_pr_curves"
        for fmt in figure_formats:
            out_path = out_base.with_suffix(f".{fmt}")
            fig.savefig(out_path)
            output_paths.append(out_path)
        plt.close(fig)
        return output_paths


def _plot_bio_paper_pr_figure(
    *,
    results_dir: Path,
    figures_dir: Path,
    figure_formats: list[str],
    input_paths: list[Path],
    bootstrap_samples: int,
    bootstrap_cache_dir: Path | None,
) -> list[Path]:
    runs = _paper_curve_runs(
        results_dir,
        input_paths,
        domain="bio",
        bootstrap_samples=bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )
    if not runs:
        return []
    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    ):
        fig = plt.figure(figsize=(4.2, 3.65))
        grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
        axes = {
            ("gpt-5.4-mini", 20): fig.add_subplot(grid[0, 0]),
            ("gpt-5.4-mini", 100): fig.add_subplot(grid[0, 1]),
            ("Qwen-3.5", 20): fig.add_subplot(grid[1, 0]),
            ("Qwen-3.5", 100): fig.add_subplot(grid[1, 1]),
        }
        method_order = ["Meerkat", "Monitor", "Bayesian", "Buffer"]
        method_colors = {
            "Meerkat": "#D55E00",
            "Monitor": "#CC79A7",
            "Bayesian": "#0072B2",
            "Buffer": "#009E73",
        }
        method_linestyles = {
            "Meerkat": "-",
            "Monitor": "-.",
            "Bayesian": ":",
            "Buffer": "--",
        }
        shared_legend_handles = [
            Line2D([0], [0], color=method_colors[method], linestyle=method_linestyles[method], linewidth=2.2)
            for method in method_order
        ]
        runs_by_panel: dict[tuple[str, int], list[PaperCurveRun]] = {}
        for run in runs:
            runs_by_panel.setdefault((run.model_label, run.background_multiplier), []).append(run)

        for panel_key, ax in axes.items():
            panel_runs = sorted(
                runs_by_panel.get(panel_key, []),
                key=lambda run: method_order.index(run.method_label) if run.method_label in method_order else 99,
            )
            if not panel_runs:
                ax.axis("off")
                continue
            for run in panel_runs:
                color = method_colors.get(run.method_label, "#555555")
                linestyle = method_linestyles.get(run.method_label, "-")
                ax.fill_between(
                    run.recall_grid,
                    run.precision_lower,
                    run.precision_upper,
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=1,
                )
                (line,) = ax.plot(
                    run.recall_grid,
                    run.precision_curve,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.2,
                    zorder=2,
                    label=run.method_label,
                )
            case_counts = {run.method_label: run.case_count for run in panel_runs}
            unique_counts = sorted(set(case_counts.values()))
            if len(unique_counts) == 1:
                ax.set_title(f"bg={panel_key[1]}x (n={unique_counts[0]})", fontsize=8.5, pad=4)
            else:
                ax.set_title(f"bg={panel_key[1]}x", fontsize=8.5, pad=4)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.25)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=7.0)
            if panel_key[0] == "Qwen-3.5":
                ax.set_xlabel("Recall", fontsize=8.5)
            if panel_key != ("gpt-5.4-mini", 20) and panel_key != ("Qwen-3.5", 20):
                ax.tick_params(labelleft=False)

        legend_ax = axes.get(("gpt-5.4-mini", 100))
        if legend_ax and legend_ax.axison:
            legend_ax.legend(
                shared_legend_handles,
                method_order,
                loc="upper right",
                frameon=False,
                fontsize=6.4,
                handlelength=1.9,
                borderaxespad=0.2,
                labelspacing=0.2,
            )

        top_left = axes.get(("gpt-5.4-mini", 20))
        if top_left and top_left.axison:
            top_left.set_ylabel("gpt-5.4-mini\nPrecision", fontsize=8.5)
        bottom_left = axes.get(("Qwen-3.5", 20))
        if bottom_left and bottom_left.axison:
            bottom_left.set_ylabel("Qwen-3.5\nPrecision", fontsize=8.5)

        fig.subplots_adjust(top=0.92, left=0.17, right=0.98, bottom=0.12)

        output_paths: list[Path] = []
        out_base = figures_dir / "dm_bio_paper_pr_curves"
        for fmt in figure_formats:
            out_path = out_base.with_suffix(f".{fmt}")
            fig.savefig(out_path)
            output_paths.append(out_path)
        plt.close(fig)
        return output_paths


def _plot_combined_paper_pr_figure(
    *,
    results_dir: Path,
    figures_dir: Path,
    figure_formats: list[str],
    input_paths: list[Path],
    bootstrap_samples: int,
    bootstrap_cache_dir: Path | None,
) -> list[Path]:
    cyber_runs = _paper_curve_runs(
        results_dir,
        input_paths,
        domain="cyber",
        bootstrap_samples=bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )
    bio_runs = _paper_curve_runs(
        results_dir,
        input_paths,
        domain="bio",
        bootstrap_samples=bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )
    runs = list(cyber_runs + bio_runs)
    if not runs:
        return []
    with mpl.rc_context(_dm_rc_context()):
        fig = plt.figure(figsize=(5.55, 2.95))
        grid = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.22)
        axes = {
            ("gpt-5.4-mini", "cyber", 20): fig.add_subplot(grid[0, 0]),
            ("gpt-5.4-mini", "cyber", 100): fig.add_subplot(grid[0, 1]),
            ("gpt-5.4-mini", "bio", 20): fig.add_subplot(grid[0, 2]),
            ("gpt-5.4-mini", "bio", 100): fig.add_subplot(grid[0, 3]),
            ("Qwen-3.5", "cyber", 20): fig.add_subplot(grid[1, 0]),
            ("Qwen-3.5", "cyber", 100): fig.add_subplot(grid[1, 1]),
            ("Qwen-3.5", "bio", 20): fig.add_subplot(grid[1, 2]),
            ("Qwen-3.5", "bio", 100): fig.add_subplot(grid[1, 3]),
        }
        runs_by_panel: dict[tuple[str, str, int], list[PaperCurveRun]] = {}
        for run in cyber_runs:
            runs_by_panel.setdefault((run.model_label, "cyber", run.background_multiplier), []).append(run)
        for run in bio_runs:
            runs_by_panel.setdefault((run.model_label, "bio", run.background_multiplier), []).append(run)
        present_methods = [method for method in DM_METHOD_ORDER if any(run.method_label == method for run in runs)]
        shared_legend_handles = _dm_legend_handles(linewidth=1.8, method_order=present_methods)
        method_layer_order = [method for method in DM_METHOD_ORDER if method != "Meerkat"] + ["Meerkat"]
        method_zorder = {method: idx for idx, method in enumerate(method_layer_order, start=1)}

        for panel_key, ax in axes.items():
            panel_runs = sorted(
                runs_by_panel.get(panel_key, []),
                key=lambda run: DM_METHOD_ORDER.index(run.method_label) if run.method_label in DM_METHOD_ORDER else 99,
            )
            if not panel_runs:
                ax.axis("off")
                continue
            panel_ap_by_method = {run.method_label: run.average_precision for run in panel_runs}
            for run in panel_runs:
                color = DM_METHOD_COLORS.get(run.method_label, "#555555")
                linestyle = DM_METHOD_LINESTYLES.get(run.method_label, "-")
                recall_grid = [float(x) for x in run.recall_grid]
                lower = [float(x) for x in run.precision_lower]
                upper = [float(x) for x in run.precision_upper]
                precision_curve = [float(x) for x in run.precision_curve]
                layer_zorder = method_zorder.get(run.method_label, 1)
                line_width = 1.8
                marker = None
                markevery = None
                band_alpha = 0.10
                if run.method_label == "Meerkat":
                    line_width = 2.3
                    marker = "o"
                    markevery = 4
                    band_alpha = 0.12
                elif run.method_label == "Naive Agent":
                    line_width = 1.9
                    marker = "s"
                    markevery = 4
                    band_alpha = 0.08
                else:
                    line_width = 1.5
                    band_alpha = 0.05
                ax.fill_between(
                    recall_grid,
                    lower,
                    upper,
                    color=color,
                    alpha=band_alpha,
                    linewidth=0.0,
                    zorder=layer_zorder,
                )
                ax.plot(
                    recall_grid,
                    precision_curve,
                    color=color,
                    linestyle=linestyle,
                    linewidth=line_width,
                    marker=marker,
                    markersize=2.6 if marker else 0,
                    markevery=markevery,
                    zorder=10 + layer_zorder,
                )
            _, domain, bg = panel_key
            domain_label = "Cyber" if domain == "cyber" else "Bio"
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            x_label = "Recall" if panel_key[0] == "Qwen-3.5" else None
            y_label = None
            if panel_key == ("gpt-5.4-mini", "cyber", 20):
                y_label = "gpt-5.4-mini\nPrecision"
            elif panel_key == ("Qwen-3.5", "cyber", 20):
                y_label = "Qwen-3.5\nPrecision"
            title = f"{domain_label} (bg={bg}x)"
            _style_dm_axis(ax, x_label=x_label, y_label=y_label, title=title)
            if panel_key not in (("gpt-5.4-mini", "cyber", 20), ("Qwen-3.5", "cyber", 20)):
                ax.tick_params(labelleft=False)

        fig.legend(
            shared_legend_handles,
            present_methods,
            loc="upper center",
            ncol=max(1, len(present_methods)),
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            handlelength=1.7,
            columnspacing=0.9,
            bbox_to_anchor=(0.5, 1.01),
        )
        fig.subplots_adjust(top=0.84, left=0.10, right=0.985, bottom=0.16)

        output_paths: list[Path] = []
        out_base = figures_dir / "dm_combined_paper_pr_curves"
        for fmt in figure_formats:
            out_path = out_base.with_suffix(f".{fmt}")
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
            output_paths.append(out_path)
        plt.close(fig)
        return output_paths


def _plot_combined_paper_roc_figure(
    *,
    results_dir: Path,
    figures_dir: Path,
    figure_formats: list[str],
    input_paths: list[Path],
    bootstrap_samples: int,
    bootstrap_cache_dir: Path | None,
) -> list[Path]:
    cyber_runs = _paper_roc_runs(
        results_dir,
        input_paths,
        domain="cyber",
        bootstrap_samples=bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )
    bio_runs = _paper_roc_runs(
        results_dir,
        input_paths,
        domain="bio",
        bootstrap_samples=bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )
    runs = cyber_runs + bio_runs
    if not runs:
        return []
    with mpl.rc_context(_dm_rc_context()):
        fig = plt.figure(figsize=(5.55, 2.95))
        grid = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.22)
        axes = {
            ("gpt-5.4-mini", "cyber", 20): fig.add_subplot(grid[0, 0]),
            ("gpt-5.4-mini", "cyber", 100): fig.add_subplot(grid[0, 1]),
            ("gpt-5.4-mini", "bio", 20): fig.add_subplot(grid[0, 2]),
            ("gpt-5.4-mini", "bio", 100): fig.add_subplot(grid[0, 3]),
            ("Qwen-3.5", "cyber", 20): fig.add_subplot(grid[1, 0]),
            ("Qwen-3.5", "cyber", 100): fig.add_subplot(grid[1, 1]),
            ("Qwen-3.5", "bio", 20): fig.add_subplot(grid[1, 2]),
            ("Qwen-3.5", "bio", 100): fig.add_subplot(grid[1, 3]),
        }
        runs_by_panel: dict[tuple[str, str, int], list[PaperRocRun]] = {}
        for run in cyber_runs:
            runs_by_panel.setdefault((run.model_label, "cyber", run.background_multiplier), []).append(run)
        for run in bio_runs:
            runs_by_panel.setdefault((run.model_label, "bio", run.background_multiplier), []).append(run)
        present_methods = [method for method in DM_METHOD_ORDER if any(run.method_label == method for run in runs)]
        shared_legend_handles = _dm_legend_handles(linewidth=1.8, method_order=present_methods)
        method_layer_order = [method for method in DM_METHOD_ORDER if method != "Meerkat"] + ["Meerkat"]
        method_zorder = {method: idx for idx, method in enumerate(method_layer_order, start=1)}

        for panel_key, ax in axes.items():
            panel_runs = sorted(
                runs_by_panel.get(panel_key, []),
                key=lambda run: DM_METHOD_ORDER.index(run.method_label) if run.method_label in DM_METHOD_ORDER else 99,
            )
            if not panel_runs:
                ax.axis("off")
                continue
            ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.8, linestyle=":", zorder=0)
            for run in panel_runs:
                color = DM_METHOD_COLORS.get(run.method_label, "#555555")
                linestyle = DM_METHOD_LINESTYLES.get(run.method_label, "-")
                layer_zorder = method_zorder.get(run.method_label, 1)
                ax.fill_between(
                    run.fpr_grid,
                    run.tpr_lower,
                    run.tpr_upper,
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=layer_zorder,
                )
                ax.plot(
                    run.fpr_grid,
                    run.tpr_curve,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    zorder=10 + layer_zorder,
                )
            _, domain, bg = panel_key
            domain_label = "Cyber" if domain == "cyber" else "Bio"
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(alpha=0.25)
            x_label = "False Positive Rate" if panel_key[0] == "Qwen-3.5" else None
            y_label = None
            if panel_key == ("gpt-5.4-mini", "cyber", 20):
                y_label = "gpt-5.4-mini\nTrue Positive Rate"
            elif panel_key == ("Qwen-3.5", "cyber", 20):
                y_label = "Qwen-3.5\nTrue Positive Rate"
            _style_dm_axis(ax, x_label=x_label, y_label=y_label, title=f"{domain_label} (bg={bg}x)")
            if panel_key not in (("gpt-5.4-mini", "cyber", 20), ("Qwen-3.5", "cyber", 20)):
                ax.tick_params(labelleft=False)

        legend_ax = axes.get(("gpt-5.4-mini", "bio", 100))
        if legend_ax and legend_ax.axison:
            legend_ax.legend(
                shared_legend_handles,
                present_methods,
                loc="lower right",
                frameon=False,
                fontsize=LEGEND_FONTSIZE,
                handlelength=1.7,
                borderaxespad=0.15,
                labelspacing=0.18,
            )
        fig.subplots_adjust(top=0.92, left=0.12, right=0.985, bottom=0.16)

        output_paths: list[Path] = []
        out_base = figures_dir / "dm_combined_paper_roc_curves"
        for fmt in figure_formats:
            out_path = out_base.with_suffix(f".{fmt}")
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
            output_paths.append(out_path)
        plt.close(fig)
        return output_paths


def _plot_meerkat_vs_monitor_rocauc_scatter(
    *,
    results_dir: Path,
    figures_dir: Path,
    figure_formats: list[str],
    input_paths: list[Path],
) -> list[Path]:
    points = _meerkat_vs_monitor_rocauc_points(results_dir, input_paths)
    if not points:
        return []

    with mpl.rc_context(_dm_rc_context()):
        fig = plt.figure(figsize=(7.45, 3.6))
        grid = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.24)
        axes = {
            ("gpt-5.4-mini", "cyber", 20): fig.add_subplot(grid[0, 0]),
            ("gpt-5.4-mini", "cyber", 100): fig.add_subplot(grid[0, 1]),
            ("gpt-5.4-mini", "bio", 20): fig.add_subplot(grid[0, 2]),
            ("gpt-5.4-mini", "bio", 100): fig.add_subplot(grid[0, 3]),
            ("Qwen-3.5", "cyber", 20): fig.add_subplot(grid[1, 0]),
            ("Qwen-3.5", "cyber", 100): fig.add_subplot(grid[1, 1]),
            ("Qwen-3.5", "bio", 20): fig.add_subplot(grid[1, 2]),
            ("Qwen-3.5", "bio", 100): fig.add_subplot(grid[1, 3]),
        }
        points_by_panel: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for point in points:
            points_by_panel.setdefault(
                (str(point["model_label"]), str(point["domain"]), int(point["background_multiplier"])),
                [],
            ).append(point)

        for panel_key, ax in axes.items():
            model_label, domain, bg = panel_key
            panel_points = points_by_panel.get(panel_key, [])
            color = DM_MODEL_COLORS.get(model_label, "#888888")
            x_values = [float(point["monitor_rocauc"]) for point in panel_points]
            y_values = [float(point["meerkat_rocauc"]) for point in panel_points]
            ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#B3B3B3", zorder=1)
            if panel_points:
                ax.scatter(
                    x_values,
                    y_values,
                    s=56,
                    marker="o",
                    facecolors=color,
                    edgecolors="#4D4D4D",
                    linewidths=0.8,
                    alpha=0.82,
                    zorder=3,
                )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            domain_label = "Cyber" if domain == "cyber" else "Bio"
            title = f"{domain_label} (bg={bg}x)"
            x_label = "Per-trace Monitor ROC-AUC" if model_label == "Qwen-3.5" else None
            y_label = None
            if panel_key == ("gpt-5.4-mini", "cyber", 20):
                y_label = "gpt-5.4-mini\nMeerkat ROC-AUC"
            elif panel_key == ("Qwen-3.5", "cyber", 20):
                y_label = "Qwen-3.5\nMeerkat ROC-AUC"
            _style_dm_axis(ax, x_label=x_label, y_label=y_label, title=title)
            if panel_key not in (("gpt-5.4-mini", "cyber", 20), ("Qwen-3.5", "cyber", 20)):
                ax.tick_params(labelleft=False)

        fig.subplots_adjust(left=0.10, right=0.972, bottom=0.16, top=0.90)

        output_paths: list[Path] = []
        out_base = figures_dir / "dm_meerkat_vs_monitor_rocauc_scatter"
        for fmt in figure_formats:
            out_path = out_base.with_suffix(f".{fmt}")
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
            output_paths.append(out_path)
        plt.close(fig)
        return output_paths


def _print_summary_table(metrics: list[SettingMetrics]) -> None:
    print("| setting | source | model | judge_ap | AT_ap | judge_cases | AT_cases |")
    print("|---|---|---|---:|---:|---:|---:|")
    for m in sorted(metrics, key=lambda item: (item.domain, item.background_multiplier)):
        setting = f"{m.domain}-d{m.decomp_level}-bg{m.background_multiplier}"
        judge_ap = "na" if m.stage1_ap is None else f"{m.stage1_ap:.3f}"
        at_ap = "na" if m.stage2_ap is None else f"{m.stage2_ap:.3f}"
        print(
            f"| {setting} | {m.variant} | {m.model_name} | {judge_ap} | {at_ap} | "
            f"{m.stage1_available_cases}/{m.total_cases} | {m.stage2_available_cases}/{m.total_cases} |"
        )


def _write_paper_metric_tables(
    results_dir: Path,
    input_paths: list[Path],
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_cache_dir: Path | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    table_specs = [
        ("trace_ap", "Distributed-Misuse Trace AP Table", "Trace-level average precision", "dm_paper_trace_ap_table", "tab:rq1-dm-trace-ap"),
        ("case_ap", "Distributed-Misuse Case AP Table", "Case-level average precision", "dm_paper_case_ap_table", "tab:rq1-dm-case-ap"),
        ("trace_roc_auc", "Distributed-Misuse Trace ROC-AUC Table", "Trace-level ROC-AUC", "dm_paper_trace_rocauc_table", "tab:rq1-dm-trace-rocauc"),
        ("case_roc_auc", "Distributed-Misuse Case ROC-AUC Table", "Case-level ROC-AUC", "dm_paper_case_rocauc_table", "tab:rq1-dm-case-rocauc"),
    ]
    def _short_model(model_label: str) -> str:
        if model_label == "Qwen-3.5":
            return "Qwen3.5"
        if model_label == "gpt-5.4-mini":
            return "GPT-5.4m"
        return model_label

    def _latex_result(mean: float | None, se: float | None, *, best: bool) -> str:
        if mean is None:
            return r"\na"
        macro = r"\bestres" if best else r"\res"
        return f"{macro}{{{mean:.3f}}}{{{(0.0 if se is None else se):.3f}}}"

    def _delta_text(meerkat: float | None, baselines: list[float | None]) -> str:
        valid = [value for value in baselines if value is not None]
        if meerkat is None or not valid:
            return r"\na"
        return f"{(meerkat - max(valid)):+.3f}"
    model_order = ["Qwen-3.5", "gpt-5.4-mini"]

    for metric_key, md_title, caption_metric, stem, latex_label in table_specs:
        md_lines = [
            f"# {md_title}",
            "",
            "| Domain | BG | Model | Meerkat | Naive Agent | Monitor | Bayesian | Buffer | Delta |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        tex_lines = [
            "% Auto-generated by scripts/analyze_distributed_misuse.py",
            r"\begin{table*}[t]",
            r"  \centering",
            r"  \small",
            r"  % \setlength{\tabcolsep}{3.5pt}",
            r"  % \renewcommand{\arraystretch}{0.95}",
            rf"  \caption{{RQ1 results for distributed misuse using {caption_metric}. Higher is better. $\Delta$ is Meerkat minus the strongest baseline.}}",
            rf"  \label{{{latex_label}}}",
            r"  \begin{tabular}{@{}lllrrrrrr@{}}",
            r"    \toprule",
            r"    \multirow{2}{*}{Domain} & \multirow{2}{*}{BG} & \multirow{2}{*}{Model}",
            r"      & \multicolumn{5}{c}{Method} & \multirow{2}{*}{$\Delta$} \\",
            r"    \cmidrule(lr){4-8}",
            r"      & & & Meerkat & Naive Agent & Monitor & Bayesian & Buffer & \\",
            r"    \midrule",
        ]

        for domain in ("cyber", "bio"):
            candidates = _paper_curve_candidates(results_dir, input_paths, domain=domain)
            if not candidates:
                continue
            monitor_paths: dict[tuple[str, int], Path] = {}
            for candidate_path, model_label, method_label, bg in candidates:
                if method_label == "Monitor":
                    monitor_paths[(model_label, bg)] = candidate_path
            rows: dict[tuple[str, int], dict[str, tuple[float | None, float | None]]] = {}
            for path, model_label, method_label, bg in candidates:
                rows.setdefault((model_label, bg), {})[method_label] = _method_metric_summary(
                    path,
                    method_label=method_label,
                    monitor_path=monitor_paths.get((model_label, bg)),
                    metric_key=metric_key,
                    bootstrap_samples=bootstrap_samples,
                    seed=bg + len(rows) * 31 + (17 if model_label == "Qwen-3.5" else 0),
                    bootstrap_cache_dir=bootstrap_cache_dir,
                )
            domain_label = "DM-Cyber" if domain == "cyber" else "DM-Bio"
            bg_order = sorted({bg for _, bg in rows})
            domain_row_count = sum(1 for bg in bg_order for model in model_order if (model, bg) in rows)
            domain_printed = False
            for bg_idx, bg in enumerate(bg_order):
                present_models = [model for model in model_order if (model, bg) in rows]
                if not present_models:
                    continue
                for model_idx, model_label in enumerate(present_models):
                    values = rows[(model_label, bg)]
                    numeric_values = {method: mean for method, (mean, _) in values.items() if mean is not None}
                    max_metric = max(numeric_values.values()) if numeric_values else None

                    def _fmt_md(method: str) -> str:
                        pair = values.get(method)
                        if pair is None or pair[0] is None:
                            return "na"
                        mean, se = pair
                        text = f"{mean:.3f} +/- {(0.0 if se is None else se):.3f}"
                        return f"**{text}**" if max_metric is not None and mean == max_metric else text

                    model_text = _short_model(model_label)
                    meerkat_mean = values.get("Meerkat", (None, None))[0]
                    naive_mean = values.get("Naive Agent", (None, None))[0]
                    monitor_mean = values.get("Monitor", (None, None))[0]
                    bayesian_mean = values.get("Bayesian", (None, None))[0]
                    buffer_mean = values.get("Buffer", (None, None))[0]
                    delta = _delta_text(meerkat_mean, [naive_mean, monitor_mean, bayesian_mean, buffer_mean])

                    md_lines.append(
                        f"| {domain_label} | {bg}x | {model_text} | {_fmt_md('Meerkat')} | {_fmt_md('Naive Agent')} | {_fmt_md('Monitor')} | {_fmt_md('Bayesian')} | {_fmt_md('Buffer')} | "
                        + ("na" if delta == r"\na" else delta)
                        + " |"
                    )

                    prefix = rf"    \multirow{{{domain_row_count}}}{{*}}{{{domain_label}}}" if not domain_printed else "    "
                    domain_printed = True
                    bg_prefix = rf" & \multirow{{{len(present_models)}}}{{*}}{{{bg}$\times$}}" if model_idx == 0 else " &"
                    tex_lines.append(
                        f"{prefix}{bg_prefix} & {model_text} & "
                        f"{_latex_result(*values.get('Meerkat', (None, None)), best=(max_metric is not None and meerkat_mean == max_metric))} & "
                        f"{_latex_result(*values.get('Naive Agent', (None, None)), best=(max_metric is not None and naive_mean == max_metric))} & "
                        f"{_latex_result(*values.get('Monitor', (None, None)), best=(max_metric is not None and monitor_mean == max_metric))} & "
                        f"{_latex_result(*values.get('Bayesian', (None, None)), best=(max_metric is not None and bayesian_mean == max_metric))} & "
                        f"{_latex_result(*values.get('Buffer', (None, None)), best=(max_metric is not None and buffer_mean == max_metric))} & "
                        f"{delta} \\\\"
                    )
                if bg_idx != len(bg_order) - 1:
                    tex_lines.append(r"    \cmidrule(lr){2-9}")
            tex_lines.append(r"    \midrule")
        if tex_lines and tex_lines[-1] == r"    \midrule":
            tex_lines.pop()
        tex_lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"])
        md_path = output_dir / f"{stem}.md"
        tex_path = output_dir / f"{stem}.tex"
        md_path.write_text("\n".join(md_lines).rstrip() + "\n")
        tex_path.write_text("\n".join(tex_lines).rstrip() + "\n")
        output_paths.extend([md_path, tex_path])
    return output_paths


def _print_paper_run_summary(
    results_dir: Path,
    input_paths: list[Path],
    *,
    bootstrap_samples: int,
    bootstrap_cache_dir: Path | None,
) -> None:
    for domain in ("cyber", "bio"):
        runs = _paper_curve_runs(
            results_dir,
            input_paths,
            domain=domain,
            bootstrap_samples=bootstrap_samples,
            bootstrap_cache_dir=bootstrap_cache_dir,
        )
        if not runs:
            continue
        print(f"\n| {domain} model | bg | method | cases | trace_ap | file |")
        print("|---|---:|---|---:|---:|---|")
        for run in sorted(runs, key=lambda item: (item.model_label, item.background_multiplier, item.method_label)):
            ap_text = "na" if run.average_precision is None else f"{run.average_precision:.3f}"
            print(
                f"| {run.model_label} | {run.background_multiplier} | {run.method_label} | "
                f"{run.case_count} | {ap_text} | {run.path.name} |"
            )


def main() -> None:
    args = _parse_args()
    explicit_input_paths: list[Path] | None = None
    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
        explicit_input_paths = input_paths
    else:
        results_dir = Path(args.results_dir)
        preferred: dict[tuple[str, str, str], Path] = {}
        for pattern in ("dm_*_v6.jsonl", "dm_*_v2.jsonl", "dm_*_scored.jsonl"):
            for path in sorted(results_dir.glob(pattern)):
                match = DM_FILE_RE.match(path.name)
                if match:
                    preferred.setdefault((match.group("domain"), match.group("decomp"), match.group("background")), path)
        input_paths = sorted(preferred.values())
    if not input_paths:
        raise SystemExit("No distributed-misuse result JSONL files found.")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_formats = [fmt.strip() for fmt in args.figure_formats.split(",") if fmt.strip()]
    stage1_cache_dir = Path(args.stage1_cache_dir)
    bootstrap_cache_dir = Path(args.bootstrap_cache_dir)

    metrics: list[SettingMetrics] = []
    for path in input_paths:
        if not DM_FILE_RE.match(path.name):
            continue
        metrics.append(
            _collect_setting_metrics(
                path,
                recompute_stage1_if_missing=args.recompute_stage1_if_missing,
                stage1_concurrency=args.stage1_concurrency,
                stage1_cache_dir=stage1_cache_dir,
            )
        )

    if not metrics:
        raise SystemExit("No matching distributed-misuse JSONL files to analyze.")

    _print_summary_table(metrics)
    _print_paper_run_summary(
        Path(args.results_dir),
        explicit_input_paths or [],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )
    figure_paths: list[Path] = []
    figure_paths.extend(
        _plot_main_paper_pr_figure(
            results_dir=Path(args.results_dir),
            figures_dir=figures_dir,
            figure_formats=figure_formats,
            input_paths=explicit_input_paths or [],
            recompute_stage1_if_missing=args.recompute_stage1_if_missing,
            stage1_concurrency=args.stage1_concurrency,
            stage1_cache_dir=stage1_cache_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_cache_dir=bootstrap_cache_dir,
        )
    )
    figure_paths.extend(
        _plot_bio_paper_pr_figure(
            results_dir=Path(args.results_dir),
            figures_dir=figures_dir,
            figure_formats=figure_formats,
            input_paths=explicit_input_paths or [],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_cache_dir=bootstrap_cache_dir,
        )
    )
    figure_paths.extend(
        _plot_combined_paper_pr_figure(
            results_dir=Path(args.results_dir),
            figures_dir=figures_dir,
            figure_formats=figure_formats,
            input_paths=explicit_input_paths or [],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_cache_dir=bootstrap_cache_dir,
        )
    )
    figure_paths.extend(
        _plot_combined_paper_roc_figure(
            results_dir=Path(args.results_dir),
            figures_dir=figures_dir,
            figure_formats=figure_formats,
            input_paths=explicit_input_paths or [],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_cache_dir=bootstrap_cache_dir,
        )
    )
    figure_paths.extend(
        _plot_meerkat_vs_monitor_rocauc_scatter(
            results_dir=Path(args.results_dir),
            figures_dir=figures_dir,
            figure_formats=figure_formats,
            input_paths=explicit_input_paths or [],
        )
    )
    table_paths = _write_paper_metric_tables(
        Path(args.results_dir),
        explicit_input_paths or [],
        figures_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_cache_dir=bootstrap_cache_dir,
    )

    print("\nWrote outputs:")
    for path in figure_paths + table_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
