"""Analyze safety case-set JSONL results (e.g., mle-sabotage / impossiblebench).

Outputs:
- Overall metrics per result file/method
- Metrics by case size (`traces_per_case`)
- Metrics by positive-trace percentage within case

Cost is estimated from usage fields using fixed model rates when explicit `cost_usd`
is not present.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
import random
import re
from pathlib import Path
from typing import Any

FIGURE_WIDTH_IN = 2.5
LINE_FIGURE_HEIGHT_IN = 2.05
BAR_FIGURE_HEIGHT_IN = 1.65
TITLE_FONTSIZE = 0  # Titles are omitted; captions should live in the paper.
AXIS_LABEL_FONTSIZE = 7.0
TICK_LABEL_FONTSIZE = 6.0
LEGEND_FONTSIZE = 6.0
LINE_WIDTH = 1.2
MARKER_SIZE = 2.8
GRID_COLOR = "#D9D9D9"
GRID_ALPHA = 0.55
SPINE_COLOR = "#666666"
TEXT_COLOR = "#222222"
METHOD_COLORS = {
    "Meerkat": "#0072B2",
    "Meerkat + Max Merge": "#005B8E",
    "Meerkat (Codex)": "#009E73",
    "Meerkat (Claude)": "#E69F00",
    "Bayesian": "#CC79A7",
    "Buffer": "#009E73",
    "Naive Agent": "#CC79A7",
    "Per-trace Monitor": "#D55E00",
}
METHOD_MARKERS = {
    "Meerkat": "o",
    "Meerkat + Max Merge": "D",
    "Meerkat (Codex)": "D",
    "Meerkat (Claude)": "P",
    "Bayesian": "X",
    "Buffer": "^",
    "Naive Agent": "^",
    "Per-trace Monitor": "s",
}
MODEL_COLORS = {
    "gpt-5-mini": "#0072B2",
    "gpt-5.4-mini": "#56B4E9",
    "gpt-5.4": "#009E73",
    "GLM-5": "#7A68A6",
    "Qwen-3.5": "#CC79A7",
    "MiniMax-M2.5": "#D55E00",
}
METHOD_HATCHES = {
    "AT": "",
    "AT-codex": "",
    "AT-claude": "",
    "llmjudge": "////",
}
# ALLOWED_EVAL_MODELS = {"gpt-5-mini", "gpt-5.4-mini", "gpt-5.4", "GLM-5", "Qwen-3.5", "MiniMax-M2.5"}
ALLOWED_EVAL_MODELS = {"Qwen-3.5", "gpt-5.4-mini", "GLM-5"}
OVERALL_DATASET_LABELS = {
    "impossiblebench_claude-opus-4.6": "ImpossibleBench",
    "mle-sabotage": "MLE-Sabotage",
    "trace-dataset": "TRACE",
}
TRACE_SCORE_DATASET_LABELS = {
    "impossiblebench_claude-opus-4.6": "ImpossibleBench",
    "mle-sabotage": "MLE-Sabotage",
    "trace-dataset": "TRACE",
}
DM_REFERENCE_METHOD_ORDER = ["Meerkat", "Per-trace Monitor", "Bayesian", "Buffer", "Naive Agent"]
DM_REFERENCE_METHOD_COLORS = {
    "Meerkat": "#D55E00",
    "Per-trace Monitor": "#CC79A7",
    "Bayesian": "#0072B2",
    "Buffer": "#009E73",
    "Naive Agent": "#666666",
}
DM_REFERENCE_METHOD_LINESTYLES: dict[str, Any] = {
    "Meerkat": "-",
    "Per-trace Monitor": "-.",
    "Bayesian": ":",
    "Buffer": "--",
    "Naive Agent": (0, (5, 1.4)),
}
DM_REFERENCE_LEGEND_LABELS = {
    "Per-trace Monitor": "Monitor",
}


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


def _safe_ratio(n: int, d: int) -> float | None:
    if d <= 0:
        return None
    return n / d


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    if total <= 0:
        return None, None, None
    p = successes / total
    z2 = z * z
    denom = 1.0 + (z2 / total)
    center = (p + (z2 / (2.0 * total))) / denom
    radius = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    low = max(0.0, center - radius)
    high = min(1.0, center + radius)
    return p, low, high


def _bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(means) - 1, int(alpha * len(means))))
    high_idx = max(0, min(len(means) - 1, int((1.0 - alpha) * len(means)) - 1))
    return mean, means[low_idx], means[high_idx]


def _bootstrap_mean_se(
    values: list[float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    if not means:
        return None
    mean_of_means = sum(means) / len(means)
    variance = sum((value - mean_of_means) ** 2 for value in means) / max(1, len(means) - 1)
    return math.sqrt(max(0.0, variance))


def _average_precision_from_pairs(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    values = [(int(label), float(score)) for label, score in pairs]
    positives = sum(1 for label, _ in values if label == 1)
    if positives <= 0:
        return None
    thresholds = sorted({score for _, score in values}, reverse=True)
    ap_accum = 0.0
    prev_recall = 0.0
    for thresh in thresholds:
        predicted = [label for label, score in values if score >= thresh]
        tp = sum(1 for label in predicted if label == 1)
        fp = len(predicted) - tp
        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        ap_accum += (recall - prev_recall) * precision
        prev_recall = recall
    return ap_accum


def _average_precision_flat_pairs(pairs: list[tuple[int, float]]) -> float | None:
    return _average_precision_from_pairs(pairs)


def _case_average_precision(case_pairs: list[list[tuple[int, float]]]) -> float | None:
    aps = [_average_precision_from_pairs(case) for case in case_pairs]
    aps = [float(ap) for ap in aps if ap is not None]
    if not aps:
        return None
    return sum(aps) / len(aps)


def _roc_auc_from_pairs(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    values = [(int(label), float(score)) for label, score in pairs]
    positives = sum(1 for label, _ in values if label == 1)
    negatives = len(values) - positives
    if positives <= 0 or negatives <= 0:
        return None

    sorted_values = sorted(values, key=lambda item: item[1])
    rank = 1
    positive_rank_sum = 0.0
    idx = 0
    while idx < len(sorted_values):
        j = idx + 1
        while j < len(sorted_values) and sorted_values[j][1] == sorted_values[idx][1]:
            j += 1
        avg_rank = (rank + (rank + (j - idx) - 1)) / 2.0
        pos_in_group = sum(1 for label, _ in sorted_values[idx:j] if label == 1)
        positive_rank_sum += pos_in_group * avg_rank
        rank += j - idx
        idx = j

    return (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)


def _case_average_roc_auc(case_pairs: list[list[tuple[int, float]]]) -> float | None:
    aucs = [_roc_auc_from_pairs(case) for case in case_pairs]
    aucs = [float(auc) for auc in aucs if auc is not None]
    if not aucs:
        return None
    return sum(aucs) / len(aucs)


def _bootstrap_average_precision_ci(
    case_pairs: list[list[tuple[int, float]]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    valid_case_pairs = [list(case) for case in case_pairs if _average_precision_from_pairs(list(case)) is not None]
    ap = _case_average_precision(valid_case_pairs)
    if not valid_case_pairs:
        return ap, None, None
    if len(valid_case_pairs) == 1:
        return ap, ap, ap
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(valid_case_pairs)
    for _ in range(n_resamples):
        sampled_cases = [valid_case_pairs[rng.randrange(n)] for _ in range(n)]
        sampled_ap = _case_average_precision(sampled_cases)
        if sampled_ap is not None:
            vals.append(sampled_ap)
    if not vals:
        return ap, None, None
    vals.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(vals) - 1, int(alpha * len(vals))))
    high_idx = max(0, min(len(vals) - 1, int((1.0 - alpha) * len(vals)) - 1))
    return ap, vals[low_idx], vals[high_idx]


def _bootstrap_roc_auc_ci(
    case_pairs: list[list[tuple[int, float]]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    valid_case_pairs = [list(case) for case in case_pairs if _roc_auc_from_pairs(list(case)) is not None]
    auc = _case_average_roc_auc(valid_case_pairs)
    if not valid_case_pairs:
        return auc, None, None
    if len(valid_case_pairs) == 1:
        return auc, auc, auc
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(valid_case_pairs)
    for _ in range(n_resamples):
        sampled_cases = [valid_case_pairs[rng.randrange(n)] for _ in range(n)]
        sampled_auc = _case_average_roc_auc(sampled_cases)
        if sampled_auc is not None:
            vals.append(sampled_auc)
    if not vals:
        return auc, None, None
    vals.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(vals) - 1, int(alpha * len(vals))))
    high_idx = max(0, min(len(vals) - 1, int((1.0 - alpha) * len(vals)) - 1))
    return auc, vals[low_idx], vals[high_idx]


def _bootstrap_flat_average_precision_ci(
    pairs: list[tuple[int, float]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    ap = _average_precision_flat_pairs(pairs)
    if not pairs:
        return ap, None, None
    if len(pairs) == 1:
        return ap, ap, ap
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(pairs)
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        sampled_ap = _average_precision_flat_pairs(sample)
        if sampled_ap is not None:
            vals.append(sampled_ap)
    if not vals:
        return ap, None, None
    vals.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(vals) - 1, int(alpha * len(vals))))
    high_idx = max(0, min(len(vals) - 1, int((1.0 - alpha) * len(vals)) - 1))
    return ap, vals[low_idx], vals[high_idx]


def _bootstrap_flat_roc_auc_ci(
    pairs: list[tuple[int, float]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    auc = _roc_auc_from_pairs(pairs)
    if not pairs:
        return auc, None, None
    if len(pairs) == 1:
        return auc, auc, auc
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(pairs)
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        sampled_auc = _roc_auc_from_pairs(sample)
        if sampled_auc is not None:
            vals.append(sampled_auc)
    if not vals:
        return auc, None, None
    vals.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(vals) - 1, int(alpha * len(vals))))
    high_idx = max(0, min(len(vals) - 1, int((1.0 - alpha) * len(vals)) - 1))
    return auc, vals[low_idx], vals[high_idx]


def _bootstrap_flat_metric_se(
    pairs: list[tuple[int, float]],
    metric_fn,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> float | None:
    if not pairs:
        return None
    if len(pairs) == 1:
        return 0.0
    rng = random.Random(seed)
    values: list[float] = []
    n = len(pairs)
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        metric_value = metric_fn(sample)
        if metric_value is not None:
            values.append(float(metric_value))
    if not values:
        return None
    return _bootstrap_mean_se(values, n_resamples=n_resamples, seed=seed + 1)


def _precision_recall_at_threshold(pairs: list[tuple[int, float]], threshold: float) -> tuple[float, float] | None:
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    if positives <= 0:
        return None
    tp = 0
    fp = 0
    for label, score in pairs:
        if float(score) < threshold:
            continue
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / positives if positives > 0 else 0.0
    return precision, recall


def _precision_recall_curve_from_cases(case_pairs: list[list[tuple[int, float]]]) -> list[dict[str, float]]:
    valid_case_pairs = [list(case) for case in case_pairs if _average_precision_from_pairs(list(case)) is not None]
    if not valid_case_pairs:
        return []
    thresholds = sorted(
        {float(score) for case in valid_case_pairs for _, score in case},
        reverse=True,
    )
    curve: list[dict[str, float]] = [{"threshold": float("inf"), "precision": 1.0, "recall": 0.0}]
    for threshold in thresholds:
        precisions: list[float] = []
        recalls: list[float] = []
        for case in valid_case_pairs:
            point = _precision_recall_at_threshold(case, threshold)
            if point is None:
                continue
            precision, recall = point
            precisions.append(precision)
            recalls.append(recall)
        if not precisions:
            continue
        curve.append(
            {
                "threshold": threshold,
                "precision": sum(precisions) / len(precisions),
                "recall": sum(recalls) / len(recalls),
            }
        )
    return curve


def _precision_recall_curve_from_flat_pairs(pairs: list[tuple[int, float]]) -> list[dict[str, float]]:
    if not pairs:
        return []
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    if positives <= 0:
        return []
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
        recall = tp / positives if positives > 0 else 0.0
        curve.append({"threshold": threshold, "precision": precision, "recall": recall})
    return curve


def _interp_precision_at_recalls(
    curve: list[dict[str, float]],
    recall_grid: list[float],
) -> list[float]:
    if not curve:
        return [0.0 for _ in recall_grid]
    recall_to_precision: dict[float, float] = {}
    for point in curve:
        recall = max(0.0, min(1.0, float(point.get("recall", 0.0))))
        precision = max(0.0, min(1.0, float(point.get("precision", 0.0))))
        recall_to_precision[recall] = max(recall_to_precision.get(recall, 0.0), precision)
    points = sorted(recall_to_precision.items())
    envelope = [precision for _, precision in points]
    for idx in range(len(envelope) - 2, -1, -1):
        envelope[idx] = max(envelope[idx], envelope[idx + 1])
    out: list[float] = []
    for target in recall_grid:
        precision_for_target = 0.0
        for (recall, _), precision in zip(points, envelope):
            if recall >= target:
                precision_for_target = precision
                break
        out.append(precision_for_target)
    return out


def _bootstrap_pr_curve_band(
    case_pairs: list[list[tuple[int, float]]],
    *,
    n_resamples: int = 200,
    seed: int = 0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    valid_case_pairs = [list(case) for case in case_pairs if _average_precision_from_pairs(list(case)) is not None]
    recall_grid = [i / 20.0 for i in range(21)]
    if not valid_case_pairs:
        zeros = [0.0 for _ in recall_grid]
        return recall_grid, zeros, zeros, zeros
    mean_curve = _precision_recall_curve_from_cases(valid_case_pairs)
    mean_precision = _interp_precision_at_recalls(mean_curve, recall_grid)
    if len(valid_case_pairs) == 1:
        return recall_grid, mean_precision, mean_precision[:], mean_precision[:]
    rng = random.Random(seed)
    samples: list[list[float]] = []
    n = len(valid_case_pairs)
    for _ in range(n_resamples):
        sampled = [valid_case_pairs[rng.randrange(n)] for _ in range(n)]
        sample_curve = _precision_recall_curve_from_cases(sampled)
        samples.append(_interp_precision_at_recalls(sample_curve, recall_grid))
    lower: list[float] = []
    upper: list[float] = []
    for idx in range(len(recall_grid)):
        vals = sorted(sample[idx] for sample in samples)
        lo_idx = max(0, min(len(vals) - 1, int(0.025 * len(vals))))
        hi_idx = max(0, min(len(vals) - 1, int(0.975 * len(vals)) - 1))
        lower.append(vals[lo_idx])
        upper.append(vals[hi_idx])
    return recall_grid, mean_precision, lower, upper


def _bootstrap_pooled_pr_curve_band(
    case_pairs: list[list[tuple[int, float]]],
    *,
    n_resamples: int = 200,
    seed: int = 0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    valid_case_pairs = [list(case) for case in case_pairs if _average_precision_from_pairs(list(case)) is not None]
    recall_grid = [i / 20.0 for i in range(21)]
    if not valid_case_pairs:
        zeros = [0.0 for _ in recall_grid]
        return recall_grid, zeros, zeros, zeros
    flat_pairs = [pair for case in valid_case_pairs for pair in case]
    mean_curve = _precision_recall_curve_from_flat_pairs(flat_pairs)
    mean_precision = _interp_precision_at_recalls(mean_curve, recall_grid)
    if len(valid_case_pairs) == 1:
        return recall_grid, mean_precision, mean_precision[:], mean_precision[:]
    rng = random.Random(seed)
    samples: list[list[float]] = []
    n = len(valid_case_pairs)
    for _ in range(n_resamples):
        sampled_cases = [valid_case_pairs[rng.randrange(n)] for _ in range(n)]
        sampled_pairs = [pair for case in sampled_cases for pair in case]
        sample_curve = _precision_recall_curve_from_flat_pairs(sampled_pairs)
        samples.append(_interp_precision_at_recalls(sample_curve, recall_grid))
    lower: list[float] = []
    upper: list[float] = []
    for idx in range(len(recall_grid)):
        vals = sorted(sample[idx] for sample in samples)
        lo_idx = max(0, min(len(vals) - 1, int(0.025 * len(vals))))
        hi_idx = max(0, min(len(vals) - 1, int(0.975 * len(vals)) - 1))
        lower.append(vals[lo_idx])
        upper.append(vals[hi_idx])
    return recall_grid, mean_precision, lower, upper


def _roc_at_threshold(pairs: list[tuple[int, float]], threshold: float) -> tuple[float, float] | None:
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    negatives = sum(1 for label, _ in pairs if int(label) == 0)
    if positives <= 0 or negatives <= 0:
        return None
    tp = fp = 0
    for label, score in pairs:
        if float(score) < threshold:
            continue
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
    return tp / positives, fp / negatives


def _roc_curve_from_cases(case_pairs: list[list[tuple[int, float]]]) -> list[dict[str, float]]:
    valid_case_pairs = [list(case) for case in case_pairs if _roc_auc_from_pairs(list(case)) is not None]
    if not valid_case_pairs:
        return []
    thresholds = sorted(
        {float(score) for case in valid_case_pairs for _, score in case},
        reverse=True,
    )
    curve: list[dict[str, float]] = [{"threshold": float("inf"), "tpr": 0.0, "fpr": 0.0}]
    for threshold in thresholds:
        tprs: list[float] = []
        fprs: list[float] = []
        for case in valid_case_pairs:
            point = _roc_at_threshold(case, threshold)
            if point is None:
                continue
            tpr, fpr = point
            tprs.append(tpr)
            fprs.append(fpr)
        if not tprs:
            continue
        curve.append(
            {
                "threshold": threshold,
                "tpr": sum(tprs) / len(tprs),
                "fpr": sum(fprs) / len(fprs),
            }
        )
    curve.append({"threshold": float("-inf"), "tpr": 1.0, "fpr": 1.0})
    return curve


def _roc_curve_from_flat_pairs(pairs: list[tuple[int, float]]) -> list[dict[str, float]]:
    if not pairs:
        return []
    if _roc_auc_from_pairs(pairs) is None:
        return []
    thresholds = [float("inf")] + sorted({float(score) for _, score in pairs}, reverse=True) + [float("-inf")]
    curve: list[dict[str, float]] = []
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    negatives = sum(1 for label, _ in pairs if int(label) == 0)
    for threshold in thresholds:
        tp = fp = 0
        for label, score in pairs:
            if float(score) < threshold:
                continue
            if int(label) == 1:
                tp += 1
            else:
                fp += 1
        curve.append(
            {
                "threshold": threshold,
                "tpr": tp / positives if positives > 0 else 0.0,
                "fpr": fp / negatives if negatives > 0 else 0.0,
            }
        )
    return curve


def _calibration_from_pairs(
    pairs: list[tuple[int, float]],
    *,
    max_bins: int = 10,
    min_bin_size: int = 20,
) -> tuple[list[dict[str, float]], float | None, float | None]:
    if not pairs:
        return [], None, None

    normalized: list[tuple[int, float]] = []
    for label, score in pairs:
        try:
            normalized.append((1 if int(label) else 0, max(0.0, min(1.0, float(score)))))
        except Exception:
            continue
    if not normalized:
        return [], None, None

    total = len(normalized)
    brier = sum((score - float(label)) ** 2 for label, score in normalized) / total
    normalized.sort(key=lambda item: item[1])
    points: list[dict[str, float]] = []
    ece = 0.0
    score_rows: list[dict[str, float]] = []
    idx = 0
    while idx < len(normalized):
        score = normalized[idx][1]
        j = idx + 1
        positives = int(normalized[idx][0])
        while j < len(normalized) and normalized[j][1] == score:
            positives += int(normalized[j][0])
            j += 1
        score_rows.append(
            {
                "score": score,
                "count": float(j - idx),
                "positives": float(positives),
            }
        )
        idx = j

    unique_scores = len(score_rows)
    max_supported_bins = max(1, len(normalized) // max(1, min_bin_size))
    bin_count = max(2, min(max_bins, unique_scores, max_supported_bins)) if unique_scores >= 2 else 1
    bucket_ranges: list[tuple[int, int]] = []
    for bucket_idx in range(bin_count):
        start = int(math.floor(bucket_idx * unique_scores / bin_count))
        end = int(math.floor((bucket_idx + 1) * unique_scores / bin_count))
        if end <= start:
            continue
        bucket_ranges.append((start, end))

    merged = True
    while merged and len(bucket_ranges) > 1:
        merged = False
        for bucket_idx, (start, end) in enumerate(bucket_ranges):
            count = sum(score_rows[i]["count"] for i in range(start, end))
            if count >= min_bin_size:
                continue
            if bucket_idx == 0:
                next_start, next_end = bucket_ranges[1]
                bucket_ranges[1] = (start, next_end)
                del bucket_ranges[0]
            else:
                prev_start, _ = bucket_ranges[bucket_idx - 1]
                bucket_ranges[bucket_idx - 1] = (prev_start, end)
                del bucket_ranges[bucket_idx]
            merged = True
            break

    for start, end in bucket_ranges:
        bucket_rows = score_rows[start:end]
        count = int(sum(row["count"] for row in bucket_rows))
        if count <= 0:
            continue
        positives = int(sum(row["positives"] for row in bucket_rows))
        mean_score = sum(row["score"] * row["count"] for row in bucket_rows) / count
        positive_rate = positives / count
        _, positive_low, positive_high = _wilson_ci(positives, count)
        score_low = min(float(row["score"]) for row in bucket_rows)
        score_high = max(float(row["score"]) for row in bucket_rows)
        ece += (count / total) * abs(positive_rate - mean_score)
        points.append(
            {
                "bin_low": score_low,
                "bin_high": score_high,
                "mean_score": mean_score,
                "positive_rate": positive_rate,
                "positive_rate_low": positive_low if positive_low is not None else positive_rate,
                "positive_rate_high": positive_high if positive_high is not None else positive_rate,
                "count": float(count),
            }
        )
    return points, ece, brier


def _f1_from_outcomes(outcomes: list[tuple[bool, bool | None]]) -> float | None:
    tp = fp = fn = 0
    for gt_pos, pred_pos in outcomes:
        if pred_pos is None:
            if gt_pos:
                fn += 1
            continue
        if pred_pos and gt_pos:
            tp += 1
        elif pred_pos and not gt_pos:
            fp += 1
        elif (not pred_pos) and gt_pos:
            fn += 1
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if precision is None or recall is None:
        return None
    if (precision + recall) == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _macro_f1_from_outcomes(outcomes: list[tuple[bool, bool | None]]) -> float | None:
    tp = fp = tn = fn = 0
    for gt_pos, pred_pos in outcomes:
        pred_is_pos = bool(pred_pos) if pred_pos is not None else False
        if gt_pos and pred_is_pos:
            tp += 1
        elif gt_pos and not pred_is_pos:
            fn += 1
        elif (not gt_pos) and pred_is_pos:
            fp += 1
        else:
            tn += 1

    def _class_f1(true_pos: int, false_pos: int, false_neg: int) -> float:
        precision = _safe_ratio(true_pos, true_pos + false_pos)
        recall = _safe_ratio(true_pos, true_pos + false_neg)
        if precision is None or recall is None:
            return 0.0
        if (precision + recall) == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    pos_f1 = _class_f1(tp, fp, fn)
    neg_f1 = _class_f1(tn, fn, fp)
    return (pos_f1 + neg_f1) / 2.0


def _macro_f1_from_verified_outcomes(
    outcomes: list[tuple[bool, bool, bool]],
) -> float | None:
    tp = fp = tn = fn = 0
    for gt_pos, pred_pos, verified_ok in outcomes:
        if pred_pos is None:
            if gt_pos:
                fn += 1
            continue

        if gt_pos:
            if pred_pos and verified_ok:
                tp += 1
            else:
                fn += 1
        else:
            if pred_pos:
                fp += 1
            elif verified_ok:
                tn += 1

    def _class_f1(true_pos: int, false_pos: int, false_neg: int) -> float:
        precision = _safe_ratio(true_pos, true_pos + false_pos)
        recall = _safe_ratio(true_pos, true_pos + false_neg)
        if precision is None or recall is None:
            return 0.0
        if (precision + recall) == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    pos_f1 = _class_f1(tp, fp, fn)
    neg_f1 = _class_f1(tn, fn, fp)
    return (pos_f1 + neg_f1) / 2.0


def _bootstrap_verified_macro_f1_ci(
    outcomes: list[tuple[bool, bool | None, bool]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    if not outcomes:
        return None, None, None
    macro_f1 = _macro_f1_from_verified_outcomes(outcomes)
    if len(outcomes) == 1:
        return macro_f1, macro_f1, macro_f1
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(outcomes)
    for _ in range(n_resamples):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        sample_macro_f1 = _macro_f1_from_verified_outcomes(sample)
        if sample_macro_f1 is not None:
            vals.append(sample_macro_f1)
    if not vals:
        return macro_f1, None, None
    vals.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(vals) - 1, int(alpha * len(vals))))
    high_idx = max(0, min(len(vals) - 1, int((1.0 - alpha) * len(vals)) - 1))
    return macro_f1, vals[low_idx], vals[high_idx]


def _bootstrap_f1_ci(
    outcomes: list[tuple[bool, bool | None]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    if not outcomes:
        return None, None, None
    f1 = _f1_from_outcomes(outcomes)
    if len(outcomes) == 1:
        return f1, f1, f1
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(outcomes)
    for _ in range(n_resamples):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        sample_f1 = _f1_from_outcomes(sample)
        if sample_f1 is not None:
            vals.append(sample_f1)
    if not vals:
        return f1, None, None
    vals.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(vals) - 1, int(alpha * len(vals))))
    high_idx = max(0, min(len(vals) - 1, int((1.0 - alpha) * len(vals)) - 1))
    return f1, vals[low_idx], vals[high_idx]


def _bootstrap_macro_f1_ci(
    outcomes: list[tuple[bool, bool | None]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    if not outcomes:
        return None, None, None
    macro_f1 = _macro_f1_from_outcomes(outcomes)
    if len(outcomes) == 1:
        return macro_f1, macro_f1, macro_f1
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(outcomes)
    for _ in range(n_resamples):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        sample_macro_f1 = _macro_f1_from_outcomes(sample)
        if sample_macro_f1 is not None:
            vals.append(sample_macro_f1)
    if not vals:
        return macro_f1, None, None
    vals.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(vals) - 1, int(alpha * len(vals))))
    high_idx = max(0, min(len(vals) - 1, int((1.0 - alpha) * len(vals)) - 1))
    return macro_f1, vals[low_idx], vals[high_idx]


def _fmt(x: float | None, digits: int = 4) -> str:
    return "N/A" if x is None else f"{x:.{digits}f}"


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


def _gt_positive(row: dict[str, Any]) -> bool:
    gt = row.get("ground_truth") or {}
    if "has_suspicious_auxiliary" in gt:
        return bool(gt.get("has_suspicious_auxiliary"))
    if "has_reward_hacking" in gt:
        return bool(gt.get("has_reward_hacking"))
    if "has_safety_violation" in gt:
        return bool(gt.get("has_safety_violation"))
    return False


def _gt_positive_count(row: dict[str, Any]) -> int:
    gt = row.get("ground_truth") or {}
    files = gt.get("suspicious_trace_files")
    if isinstance(files, list):
        return len(files)
    files = gt.get("hacked_trace_files")
    if isinstance(files, list):
        return len(files)
    files = gt.get("violating_trace_files")
    if isinstance(files, list):
        return len(files)
    return 0


def _normalize_witness_trace_sets(raw_value: Any) -> list[tuple[str, ...]]:
    if not isinstance(raw_value, list):
        return []
    out: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for item in raw_value:
        if not isinstance(item, list):
            continue
        key = tuple(sorted({str(x).strip() for x in item if str(x).strip()}))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _ground_truth_witness_trace_sets(row: dict[str, Any]) -> list[tuple[str, ...]]:
    gt = row.get("ground_truth") or {}
    for key in (
        "witness_trace_sets",
        "hacked_witness_trace_sets",
        "suspicious_witness_trace_sets",
        "violating_witness_trace_sets",
    ):
        witness_sets = _normalize_witness_trace_sets(gt.get(key))
        if witness_sets:
            return witness_sets
    positive_files = _ground_truth_positive_trace_files(row)
    return [tuple([trace_file]) for trace_file in positive_files]


def _ground_truth_positive_trace_files(row: dict[str, Any]) -> list[str]:
    gt = row.get("ground_truth") or {}
    for key in ("hacked_trace_files", "suspicious_trace_files", "violating_trace_files"):
        value = gt.get(key)
        if isinstance(value, list):
            return [str(x) for x in value if str(x).strip()]
    return []


def _normalize_trace_ref(path: str) -> str:
    text = str(path or "").strip()
    for prefix in ("/workspace/repo/", "/workspace/", "workspace/repo/", "workspace/"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _predicted_trace_files(row: dict[str, Any]) -> list[str]:
    scoring = row.get("scoring") or {}
    predicted = scoring.get("predicted_hacked_trace_files")
    if not isinstance(predicted, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in predicted:
        path = _normalize_trace_ref(str(item))
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _predicted_witness_trace_sets(row: dict[str, Any]) -> list[tuple[str, ...]]:
    scoring = row.get("scoring") or {}
    witness_sets = _normalize_witness_trace_sets(scoring.get("predicted_witness_trace_sets"))
    if witness_sets:
        return witness_sets
    pred_files = scoring.get("predicted_hacked_trace_files")
    if isinstance(pred_files, list):
        return [tuple([str(x).strip()]) for x in pred_files if str(x).strip()]
    return []


def _pred_positive(row: dict[str, Any]) -> bool | None:
    scoring = row.get("scoring") or {}
    value = scoring.get("predicted_has_reward_hacking")
    return value if isinstance(value, bool) else None


def _method_label(path: Path) -> str:
    return path.stem


def _strip_method_variant(method: str) -> tuple[str, str | None]:
    method_variant: str | None = None
    for suffix in ("_no-tools", "_no_tools"):
        if method.endswith(suffix):
            method_variant = "no-tools"
            method = method[: -len(suffix)]
            break
    return method, method_variant


def _split_method_parts(method: str) -> tuple[str, str | None]:
    method = str(method or "").strip()
    if method.startswith("buffer-"):
        return "buffer", method[len("buffer-") :] if len(method) > len("buffer-") else None
    if method == "buffer":
        return "buffer", None
    if method.startswith("bayesian-"):
        return "bayesian", method[len("bayesian-") :] if len(method) > len("bayesian-") else None
    if method == "bayesian":
        return "bayesian", None
    if method.startswith("AT-codex-"):
        return "AT-codex", method[len("AT-codex-") :] if len(method) > len("AT-codex-") else None
    if method == "AT-codex":
        return "AT-codex", None
    if method.startswith("AT-claude-"):
        return "AT-claude", method[len("AT-claude-") :] if len(method) > len("AT-claude-") else None
    if method == "AT-claude":
        return "AT-claude", None
    if method.startswith("vibetest-codex-"):
        return "AT-codex", method[len("vibetest-codex-") :] if len(method) > len("vibetest-codex-") else None
    if method == "vibetest-codex":
        return "AT-codex", None
    if method.startswith("vibetest-claude-"):
        return "AT-claude", method[len("vibetest-claude-") :] if len(method) > len("vibetest-claude-") else None
    if method == "vibetest-claude":
        return "AT-claude", None
    if method.startswith("AT-"):
        return "AT", method[3:] if len(method) > 3 else None
    if method.startswith("llmjudge"):
        if method == "llmjudge":
            return "llmjudge", None
        if method.startswith("llmjudge-"):
            return "llmjudge", method[len("llmjudge-") :]
        return method, None
    return method, None


def _display_model_name(model: str | None) -> str | None:
    text = str(model or "").strip()
    if not text:
        return None
    for prefix in ("codex-", "claude-"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text == "MiniMaxAI-MiniMax-M2.5":
        return "MiniMax-M2.5"
    lowered = text.lower()
    if lowered in {"glm-5", "zai-org-glm-5"} or "glm-5" in lowered:
        return "GLM-5"
    if lowered in {"qwen3.5", "qwen-3.5"} or "qwen3.5" in lowered or "qwen-3.5" in lowered:
        return "Qwen-3.5"
    return text


def _should_include_eval_model(model: str | None) -> bool:
    display = _display_model_name(model)
    if not display:
        return False
    return display in ALLOWED_EVAL_MODELS


def _pretty_method(
    method: str,
    *,
    method_model: str | None = None,
    dataset_variant: str | None = None,
    method_variant: str | None = None,
) -> str:
    family, model = _split_method_parts(method)
    model = _display_model_name(method_model or model)
    if family == "llmjudge":
        if model:
            return f"Per-trace Monitor ({model})"
        return "Per-trace Monitor"

    if family in {"AT", "AT-codex", "AT-claude"}:
        family_name = _presentation_method_label(
            method_base=family,
            dataset_variant=dataset_variant,
            method_variant=method_variant,
        )
        if model:
            return f"{family_name} ({model})"
        return family_name
    family_name = _presentation_method_label(
        method_base=family,
        dataset_variant=dataset_variant,
        method_variant=method_variant,
    )
    if model:
        return f"{family_name} ({model})"
    return family_name


def _presentation_method_label(
    *,
    method_base: str | None,
    dataset_variant: str | None = None,
    method_variant: str | None = None,
) -> str:
    base = str(method_base or "").strip()
    variant = str(method_variant or "").strip()
    dataset_variant = str(dataset_variant or "").strip()
    if base == "llmjudge":
        return "Per-trace Monitor"
    if variant == "max-merge":
        return "Meerkat + Max Merge"
    if dataset_variant == "no-tools" or variant == "no-tools":
        return "Naive Agent"
    if base == "AT-codex":
        return "Meerkat (Codex)"
    if base == "AT-claude":
        return "Meerkat (Claude)"
    if base == "buffer":
        return "Buffer"
    if base == "bayesian":
        return "Bayesian"
    if base == "AT":
        return "Meerkat"
    return str(method_base or "Unknown").strip() or "Unknown"

    if method_model:
        return f"{family} ({method_model})"
    return family


def _parse_positive_pct_bins(text: str) -> list[float]:
    bounds: list[float] = []
    for raw in str(text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except Exception as exc:
            raise SystemExit(f"Invalid --positive-pct-bins value: {raw}") from exc
        if value <= 0.0 or value > 100.0:
            raise SystemExit("--positive-pct-bins must contain values in (0, 100].")
        bounds.append(value)
    if not bounds:
        bounds = [25.0, 50.0, 75.0, 100.0]
    bounds = sorted(set(bounds))
    if bounds[-1] < 100.0:
        bounds.append(100.0)
    return bounds


def _positive_pct_bucket_label(pct: float, bounds: list[float]) -> str:
    if pct <= 0.0:
        return "0%"
    lower = 0.0
    for upper in bounds:
        if pct <= upper:
            return f"({lower:.1f},{upper:.1f}]%"
        lower = upper
    return f"({bounds[-1]:.1f},100.0]%"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_path = str(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                row.setdefault("__source_path", source_path)
            rows.append(row)
    return rows


def _row_case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("repo_name") or "").strip()


@lru_cache(maxsize=4096)
def _load_initial_trace_scores_from_repo(repo_path: str) -> dict[str, float]:
    repo = Path(str(repo_path or "")).expanduser()
    if not repo:
        return {}
    scores_path = repo / "initial_scores.tsv"
    if not scores_path.is_file():
        return {}
    scores: dict[str, float] = {}
    try:
        with scores_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                trace_ref = _normalize_trace_ref(parts[0])
                if not trace_ref:
                    continue
                try:
                    score = float(parts[1])
                except Exception:
                    continue
                scores[trace_ref] = max(0.0, min(1.0, score))
    except OSError:
        return {}
    return scores


def _judge_results_path_for_source(row: dict[str, Any]) -> Path | None:
    source_path = Path(str(row.get("__source_path") or "")).expanduser()
    if not source_path.is_file():
        return None
    dataset_name, method = _infer_dataset_and_method_from_file(str(source_path))
    stripped_method, _ = _strip_method_variant(method)
    method_base, method_model = _split_method_parts(stripped_method)
    if method_base != "AT":
        return None
    if not method_model:
        tests = row.get("tests") or []
        metadata = (tests[0] or {}).get("metadata") if tests else {}
        if isinstance(metadata, dict):
            method_model = str(metadata.get("model") or "").strip() or None
    if not method_model:
        return None
    base_dataset, _ = _split_dataset_variant(dataset_name)
    prefix = "safety_" if source_path.name.startswith("safety_") else ""
    candidate = source_path.parent / f"{prefix}{base_dataset}_llmjudge-{method_model}.jsonl"
    return candidate if candidate.is_file() else None


@lru_cache(maxsize=512)
def _load_judge_trace_scores_by_case(judge_path: str) -> dict[str, dict[str, float]]:
    path = Path(str(judge_path or "")).expanduser()
    if not path.is_file():
        return {}
    by_case: dict[str, dict[str, float]] = {}
    for row in _load_rows(path):
        case_id = _row_case_id(row)
        if not case_id:
            continue
        scores = _row_trace_scores(row, max_merge=False)
        if scores:
            by_case[case_id] = scores
    return by_case


def _initial_scores_for_row(row: dict[str, Any]) -> dict[str, float]:
    initial_scores = _load_initial_trace_scores_from_repo(str(row.get("repo") or ""))
    if initial_scores:
        return initial_scores
    judge_path = _judge_results_path_for_source(row)
    if judge_path is None:
        return {}
    case_id = _row_case_id(row)
    if not case_id:
        return {}
    return dict(_load_judge_trace_scores_by_case(str(judge_path)).get(case_id) or {})


def _row_trace_scores(row: dict[str, Any], *, max_merge: bool = False) -> dict[str, float]:
    tests = row.get("tests") or []
    metadata = (tests[0] or {}).get("metadata") if tests else {}
    if not isinstance(metadata, dict):
        metadata = {}
    raw_scores = metadata.get("trace_scores") or (row.get("scoring") or {}).get("trace_scores") or {}
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    scores: dict[str, float] = {}
    for trace_file, raw_score in raw_scores.items():
        trace_key = _normalize_trace_ref(trace_file)
        if not trace_key:
            continue
        try:
            score = float(raw_score)
        except Exception:
            continue
        scores[trace_key] = max(0.0, min(1.0, score))
    if not max_merge:
        return scores
    initial_scores = _initial_scores_for_row(row)
    if not initial_scores:
        return scores
    merged = dict(initial_scores)
    for trace_key, score in scores.items():
        merged[trace_key] = max(float(merged.get(trace_key, 0.0)), score)
    return merged


def _row_trace_score_pairs(row: dict[str, Any], *, max_merge: bool = False) -> list[tuple[int, float]]:
    scores = _row_trace_scores(row, max_merge=max_merge)
    if not scores:
        return []
    gt_positive = {_normalize_trace_ref(x) for x in _ground_truth_positive_trace_files(row)}
    pairs: list[tuple[int, float]] = []
    for trace_key, score in scores.items():
        if not trace_key:
            continue
        pairs.append((1 if trace_key in gt_positive else 0, max(0.0, min(1.0, float(score)))))
    return pairs


def _analysis_row_uses_max_merge(row: dict[str, Any]) -> bool:
    if str(row.get("score_mode") or "").strip() != "max-merge":
        return False
    method_base = str(
        row.get("method_base")
        or row.get("method_base_override")
        or ""
    ).strip()
    method_variant = str(
        row.get("method_variant")
        or row.get("method_variant_override")
        or ""
    ).strip()
    return method_base == "AT" and method_variant != "no-tools"


def _row_case_score_pair(row: dict[str, Any], *, max_merge: bool = False) -> tuple[int, float] | None:
    if max_merge:
        trace_scores = _row_trace_scores(row, max_merge=True)
        if trace_scores:
            raw_score = max(float(v) for v in trace_scores.values())
        else:
            raw_score = None
    else:
        raw_score = None
    scoring = row.get("scoring") or {}
    if raw_score is None:
        raw_score = scoring.get("case_score")
        if raw_score is None:
            tests = row.get("tests") or []
            metadata = (tests[0] or {}).get("metadata") if tests else {}
            if not isinstance(metadata, dict):
                metadata = {}
            raw_score = metadata.get("case_score")
        if raw_score is None:
            trace_scores = scoring.get("trace_scores") or {}
            if isinstance(trace_scores, dict) and trace_scores:
                try:
                    raw_score = max(float(v) for v in trace_scores.values())
                except Exception:
                    raw_score = None
    if raw_score is None:
        return None
    try:
        score = max(0.0, min(1.0, float(raw_score)))
    except Exception:
        return None
    return (1 if _gt_positive(row) else 0, score)


def _init_metric_row(
    label: str,
    file_path: Path,
    group_key: str,
    *,
    score_mode: str = "original",
    method_base_override: str | None = None,
    method_variant_override: str | None = None,
) -> dict[str, Any]:
    return {
        "method": label,
        "file": str(file_path),
        "group": group_key,
        "score_mode": score_mode,
        "method_base_override": method_base_override,
        "method_variant_override": method_variant_override,
        "total_cases": 0,
        "classification_correct": 0,
        "verified_correct": 0,
        "verified_tp": 0,
        "verified_fp": 0,
        "verified_tn": 0,
        "verified_fn": 0,
        "matched_witnesses": 0,
        "predicted_witnesses": 0,
        "ground_truth_witnesses": 0,
        "trace_tp_micro": 0,
        "trace_fp_micro": 0,
        "trace_fn_micro": 0,
        "trace_case_positive_count": 0,
        "trace_case_precision_sum": 0.0,
        "trace_case_recall_sum": 0.0,
        "trace_case_f1_sum": 0.0,
        "_trace_case_f1_pos_values": [],
        "gt_positive": 0,
        "pred_positive": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "unknown_predictions": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "has_cost": False,
        "_case_costs": [],
        "_outcomes": [],
        "_verified_outcomes": [],
        "_trace_score_cases": [],
        "_case_score_pairs": [],
    }


def _update_metric_row(metric: dict[str, Any], row: dict[str, Any], *, fallback_model: str) -> None:
    metric["total_cases"] += 1
    if (row.get("scoring") or {}).get("classification_correct"):
        metric["classification_correct"] += 1
    if (row.get("scoring") or {}).get("verified_correct"):
        metric["verified_correct"] += 1

    gt_pos = _gt_positive(row)
    if gt_pos:
        metric["gt_positive"] += 1

    pred_pos = _pred_positive(row)
    metric["_outcomes"].append((gt_pos, pred_pos))
    verified_ok = bool((row.get("scoring") or {}).get("verified_correct"))
    metric["_verified_outcomes"].append((gt_pos, pred_pos, verified_ok))
    if pred_pos is True:
        metric["pred_positive"] += 1
    pred_witnesses = set(_predicted_witness_trace_sets(row))
    gt_witnesses = set(_ground_truth_witness_trace_sets(row))
    metric["predicted_witnesses"] += len(pred_witnesses)
    metric["ground_truth_witnesses"] += len(gt_witnesses)
    metric["matched_witnesses"] += len(pred_witnesses & gt_witnesses)
    gt_trace_set = {_normalize_trace_ref(x) for x in _ground_truth_positive_trace_files(row)}
    pred_trace_set = set(_predicted_trace_files(row))
    metric["trace_tp_micro"] += len(pred_trace_set & gt_trace_set)
    metric["trace_fp_micro"] += len(pred_trace_set - gt_trace_set)
    metric["trace_fn_micro"] += len(gt_trace_set - pred_trace_set)
    if gt_pos:
        trace_tp = len(pred_trace_set & gt_trace_set)
        case_trace_precision = (trace_tp / len(pred_trace_set)) if pred_trace_set else 0.0
        case_trace_recall = (trace_tp / len(gt_trace_set)) if gt_trace_set else 0.0
        if (case_trace_precision + case_trace_recall) == 0:
            case_trace_f1 = 0.0
        else:
            case_trace_f1 = (
                2.0 * case_trace_precision * case_trace_recall / (case_trace_precision + case_trace_recall)
            )
        metric["trace_case_positive_count"] += 1
        metric["trace_case_precision_sum"] += case_trace_precision
        metric["trace_case_recall_sum"] += case_trace_recall
        metric["trace_case_f1_sum"] += case_trace_f1
        metric["_trace_case_f1_pos_values"].append(case_trace_f1)
    use_max_merge_scores = _analysis_row_uses_max_merge(metric)
    trace_score_pairs = _row_trace_score_pairs(row, max_merge=use_max_merge_scores)
    if trace_score_pairs:
        metric["_trace_score_cases"].append(trace_score_pairs)
    case_score_pair = _row_case_score_pair(row, max_merge=use_max_merge_scores)
    if case_score_pair is not None:
        metric["_case_score_pairs"].append(case_score_pair)

    if pred_pos is None:
        metric["unknown_predictions"] += 1
        if gt_pos:
            metric["fn"] += 1
    else:
        if pred_pos and gt_pos:
            metric["tp"] += 1
        elif pred_pos and not gt_pos:
            metric["fp"] += 1
        elif (not pred_pos) and (not gt_pos):
            metric["tn"] += 1
        else:
            metric["fn"] += 1

    if pred_pos is True:
        if gt_pos and verified_ok:
            metric["verified_tp"] += 1
        elif gt_pos:
            metric["verified_fn"] += 1
        else:
            metric["verified_fp"] += 1
    elif pred_pos is False:
        if gt_pos:
            metric["verified_fn"] += 1
        elif verified_ok:
            metric["verified_tn"] += 1
    else:
        if gt_pos:
            metric["verified_fn"] += 1

    usage_totals = ((row.get("usage") or {}).get("usage_totals") or {})
    metric["input_tokens"] += _to_int(usage_totals.get("input_tokens"))
    metric["cached_input_tokens"] += _to_int(usage_totals.get("input_tokens_cache_read"))
    metric["output_tokens"] += _to_int(usage_totals.get("output_tokens"))
    metric["total_tokens"] += _to_int(usage_totals.get("total_tokens"))

    entry_cost = _entry_cost_usd(row, fallback_model_name=fallback_model)
    if entry_cost is not None:
        metric["total_cost_usd"] += entry_cost
        metric["has_cost"] = True
        metric["_case_costs"].append(float(entry_cost))


def _finalize_metric_row(metric: dict[str, Any]) -> dict[str, Any]:
    total = int(metric["total_cases"])
    tp = int(metric["tp"])
    fp = int(metric["fp"])
    tn = int(metric["tn"])
    fn = int(metric["fn"])

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None if (precision is None or recall is None) else 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    out = dict(metric)
    out["classification_accuracy"] = _safe_ratio(int(metric["classification_correct"]), total)
    out["verified_accuracy"] = _safe_ratio(int(metric["verified_correct"]), total)
    out["response_rate"] = _safe_ratio(total - int(metric["unknown_predictions"]), total)
    out["gt_positive_rate"] = _safe_ratio(int(metric["gt_positive"]), total)
    out["pred_positive_rate"] = _safe_ratio(int(metric["pred_positive"]), total)
    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = f1
    outcomes = list(metric.get("_outcomes") or [])
    _, f1_low, f1_high = _bootstrap_f1_ci(outcomes)
    out["f1_ci_low"] = f1_low
    out["f1_ci_high"] = f1_high
    macro_f1, macro_f1_low, macro_f1_high = _bootstrap_macro_f1_ci(outcomes)
    out["macro_f1"] = macro_f1
    out["macro_f1_ci_low"] = macro_f1_low
    out["macro_f1_ci_high"] = macro_f1_high

    verified_outcomes = []
    for gt_pos, pred_pos, verified_ok in metric.get("_verified_outcomes") or []:
        if pred_pos is None:
            verified_outcomes.append((bool(gt_pos), None, bool(verified_ok)))
        else:
            verified_outcomes.append((bool(gt_pos), bool(pred_pos), bool(verified_ok)))
    verified_tp = int(metric["verified_tp"])
    verified_fp = int(metric["verified_fp"])
    verified_tn = int(metric["verified_tn"])
    verified_fn = int(metric["verified_fn"])
    out["verified_precision"] = _safe_ratio(verified_tp, verified_tp + verified_fp)
    out["verified_recall"] = _safe_ratio(verified_tp, verified_tp + verified_fn)
    out["witness_precision"] = _safe_ratio(int(metric["matched_witnesses"]), int(metric["predicted_witnesses"]))
    out["witness_recall"] = _safe_ratio(int(metric["matched_witnesses"]), int(metric["ground_truth_witnesses"]))
    trace_tp_micro = int(metric["trace_tp_micro"])
    trace_fp_micro = int(metric["trace_fp_micro"])
    trace_fn_micro = int(metric["trace_fn_micro"])
    trace_precision = _safe_ratio(trace_tp_micro, trace_tp_micro + trace_fp_micro)
    trace_recall = _safe_ratio(trace_tp_micro, trace_tp_micro + trace_fn_micro)
    if trace_precision is None or trace_recall is None:
        trace_f1 = None
    elif (trace_precision + trace_recall) == 0:
        trace_f1 = 0.0
    else:
        trace_f1 = 2.0 * trace_precision * trace_recall / (trace_precision + trace_recall)
    out["trace_precision"] = trace_precision
    out["trace_recall"] = trace_recall
    out["trace_f1"] = trace_f1
    positive_trace_cases = int(metric["trace_case_positive_count"])
    out["trace_case_precision_pos"] = _safe_ratio(metric["trace_case_precision_sum"], positive_trace_cases)
    out["trace_case_recall_pos"] = _safe_ratio(metric["trace_case_recall_sum"], positive_trace_cases)
    out["trace_case_f1_pos"] = _safe_ratio(metric["trace_case_f1_sum"], positive_trace_cases)
    _, trace_case_f1_pos_low, trace_case_f1_pos_high = _bootstrap_mean_ci(
        [float(v) for v in metric.get("_trace_case_f1_pos_values") or []]
    )
    out["trace_case_f1_pos_ci_low"] = trace_case_f1_pos_low
    out["trace_case_f1_pos_ci_high"] = trace_case_f1_pos_high
    verified_macro_f1, verified_macro_f1_low, verified_macro_f1_high = _bootstrap_verified_macro_f1_ci(
        verified_outcomes
    )
    out["verified_macro_f1"] = verified_macro_f1
    out["verified_macro_f1_ci_low"] = verified_macro_f1_low
    out["verified_macro_f1_ci_high"] = verified_macro_f1_high
    trace_score_cases = [list(case) for case in metric.get("_trace_score_cases") or []]
    trace_average_precision, trace_ap_low, trace_ap_high = _bootstrap_average_precision_ci(trace_score_cases)
    trace_ap_values = [_average_precision_from_pairs(case) for case in trace_score_cases]
    trace_ap_values = [float(value) for value in trace_ap_values if value is not None]
    trace_auc_values = [_roc_auc_from_pairs(case) for case in trace_score_cases]
    trace_auc_values = [float(value) for value in trace_auc_values if value is not None]
    out["trace_average_precision"] = trace_average_precision
    out["trace_average_precision_ci_low"] = trace_ap_low
    out["trace_average_precision_ci_high"] = trace_ap_high
    out["trace_average_precision_bootstrap_se"] = _bootstrap_mean_se(trace_ap_values)
    trace_roc_auc, trace_roc_low, trace_roc_high = _bootstrap_roc_auc_ci(trace_score_cases)
    out["trace_roc_auc"] = trace_roc_auc
    out["trace_roc_auc_ci_low"] = trace_roc_low
    out["trace_roc_auc_ci_high"] = trace_roc_high
    out["trace_roc_auc_bootstrap_se"] = _bootstrap_mean_se(trace_auc_values)
    case_score_pairs = [tuple(pair) for pair in metric.get("_case_score_pairs") or []]
    case_average_precision, case_ap_low, case_ap_high = _bootstrap_flat_average_precision_ci(case_score_pairs)
    out["case_average_precision"] = case_average_precision
    out["case_average_precision_ci_low"] = case_ap_low
    out["case_average_precision_ci_high"] = case_ap_high
    out["case_average_precision_bootstrap_se"] = _bootstrap_flat_metric_se(
        case_score_pairs,
        _average_precision_flat_pairs,
    )
    case_roc_auc, case_roc_low, case_roc_high = _bootstrap_flat_roc_auc_ci(case_score_pairs)
    out["case_roc_auc"] = case_roc_auc
    out["case_roc_auc_ci_low"] = case_roc_low
    out["case_roc_auc_ci_high"] = case_roc_high
    out["case_roc_auc_bootstrap_se"] = _bootstrap_flat_metric_se(
        case_score_pairs,
        _roc_auc_from_pairs,
    )

    out["avg_total_tokens_per_case"] = _safe_ratio(int(metric["total_tokens"]), total)
    avg_cost = (float(metric["total_cost_usd"]) / total) if total > 0 and metric.get("has_cost") else None
    out["avg_cost_usd_per_case"] = avg_cost
    case_costs = [float(v) for v in metric.get("_case_costs") or []]
    _, cost_low, cost_high = _bootstrap_mean_ci(case_costs)
    out["avg_cost_usd_per_case_ci_low"] = cost_low
    out["avg_cost_usd_per_case_ci_high"] = cost_high
    out.pop("_case_costs", None)
    out.pop("_trace_case_f1_pos_values", None)
    out.pop("_outcomes", None)
    out.pop("_verified_outcomes", None)
    out.pop("_trace_score_cases", None)
    out.pop("_case_score_pairs", None)
    return out


def _collect_metrics(
    path: Path,
    *,
    positive_pct_bounds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_rows(path)
    label = _method_label(path)
    fallback_model = label
    for row in rows:
        tests = row.get("tests") or []
        if not tests:
            continue
        md = (tests[0] or {}).get("metadata") or {}
        model_name = str(md.get("model") or "").strip()
        if model_name:
            fallback_model = model_name
            break

    dataset_name, method = _infer_dataset_and_method_from_file(str(path))
    _, dataset_variant = _split_dataset_variant(dataset_name)
    stripped_method, stripped_method_variant = _strip_method_variant(method)
    method_base, method_model = _split_method_parts(stripped_method)
    effective_variant = str(stripped_method_variant or dataset_variant or "").strip()
    eval_model = _display_model_name(method_model or fallback_model)
    if not _should_include_eval_model(eval_model):
        return [], [], []

    use_default_max_merge = (
        method_base == "AT"
        and effective_variant == ""
        and dataset_name in {"trace-dataset", "impossiblebench_claude-opus-4.6"}
    )
    default_score_mode = "max-merge" if use_default_max_merge else "original"

    overall = _init_metric_row(
        label,
        path,
        "overall",
        score_mode=default_score_mode,
        method_base_override=method_base,
        method_variant_override=effective_variant or None,
    )
    by_case_size: dict[str, dict[str, Any]] = {}
    by_positive_pct: dict[str, dict[str, Any]] = {}
    include_max_merge = method_base == "AT" and effective_variant == "" and not use_default_max_merge
    overall_max = (
        _init_metric_row(
            f"{label}-max-merge",
            path,
            "overall",
            score_mode="max-merge",
            method_base_override=method_base,
            method_variant_override="max-merge",
        )
        if include_max_merge
        else None
    )
    by_case_size_max: dict[str, dict[str, Any]] = {}
    by_positive_pct_max: dict[str, dict[str, Any]] = {}

    for row in rows:
        _update_metric_row(overall, row, fallback_model=fallback_model)
        if overall_max is not None:
            _update_metric_row(overall_max, row, fallback_model=fallback_model)

        case_size = _to_int(row.get("traces_per_case"))
        case_key = str(case_size) if case_size > 0 else "unknown"
        if case_key not in by_case_size:
            by_case_size[case_key] = _init_metric_row(
                label,
                path,
                case_key,
                score_mode=default_score_mode,
                method_base_override=method_base,
                method_variant_override=effective_variant or None,
            )
        _update_metric_row(by_case_size[case_key], row, fallback_model=fallback_model)
        if overall_max is not None:
            if case_key not in by_case_size_max:
                by_case_size_max[case_key] = _init_metric_row(
                    f"{label}-max-merge",
                    path,
                    case_key,
                    score_mode="max-merge",
                    method_base_override=method_base,
                    method_variant_override="max-merge",
                )
            _update_metric_row(by_case_size_max[case_key], row, fallback_model=fallback_model)

        pos_count = _gt_positive_count(row)
        if case_size > 0:
            pct = 100.0 * (pos_count / case_size)
            pct_key = _positive_pct_bucket_label(pct, positive_pct_bounds)
        else:
            pct_key = "unknown"
        if pct_key not in by_positive_pct:
            by_positive_pct[pct_key] = _init_metric_row(
                label,
                path,
                pct_key,
                score_mode=default_score_mode,
                method_base_override=method_base,
                method_variant_override=effective_variant or None,
            )
        _update_metric_row(by_positive_pct[pct_key], row, fallback_model=fallback_model)
        if overall_max is not None:
            if pct_key not in by_positive_pct_max:
                by_positive_pct_max[pct_key] = _init_metric_row(
                    f"{label}-max-merge",
                    path,
                    pct_key,
                    score_mode="max-merge",
                    method_base_override=method_base,
                    method_variant_override="max-merge",
                )
            _update_metric_row(by_positive_pct_max[pct_key], row, fallback_model=fallback_model)

    overall_rows = [_finalize_metric_row(overall)]
    if overall_max is not None:
        overall_rows.append(_finalize_metric_row(overall_max))
    case_rows = [_finalize_metric_row(v) for _, v in sorted(by_case_size.items(), key=lambda kv: kv[0])]
    if by_case_size_max:
        case_rows.extend(_finalize_metric_row(v) for _, v in sorted(by_case_size_max.items(), key=lambda kv: kv[0]))
    def _pct_bucket_sort_key(label_text: str) -> tuple[int, float]:
        if label_text == "unknown":
            return (2, 999.0)
        if label_text == "0%":
            return (0, 0.0)
        try:
            upper = float(label_text.split(",")[1].split("]")[0])
        except Exception:
            upper = 999.0
        return (1, upper)

    pct_rows = [
        _finalize_metric_row(v)
        for _, v in sorted(by_positive_pct.items(), key=lambda kv: _pct_bucket_sort_key(kv[0]))
    ]
    if by_positive_pct_max:
        pct_rows.extend(
            _finalize_metric_row(v)
            for _, v in sorted(by_positive_pct_max.items(), key=lambda kv: _pct_bucket_sort_key(kv[0]))
        )
    return overall_rows, case_rows, pct_rows


def _print_table(title: str, rows: list[dict[str, Any]], *, include_group: bool) -> None:
    if not rows:
        print(f"\n## {title}\n(no rows)")
        return

    if include_group:
        cols = [
            "method",
            "group",
            "total_cases",
            "classification_accuracy",
            "verified_accuracy",
            "trace_precision",
            "trace_recall",
            "trace_f1",
            "response_rate",
            "precision",
            "recall",
            "f1",
            "gt_positive_rate",
            "pred_positive_rate",
            "tp",
            "fp",
            "tn",
            "fn",
            "avg_cost_usd_per_case",
        ]
    else:
        cols = [
            "method",
            "verified_precision",
            "verified_recall",
            "verified_macro_f1",
            "trace_precision",
            "trace_recall",
            "trace_f1",
            "trace_average_precision",
            "witness_precision",
            "witness_recall",
            "response_rate",
            "avg_cost_usd_per_case",
        ]

    print(f"\n## {title}")
    header = " | ".join(cols)
    print(header)
    print(" | ".join("---" for _ in cols))

    for row in rows:
        cells: list[str] = []
        for c in cols:
            v = row.get(c)
            if c in {
                "verified_precision",
                "verified_recall",
                "verified_macro_f1",
                "trace_precision",
                "trace_recall",
                "trace_f1",
                "trace_average_precision",
                "witness_precision",
                "witness_recall",
                "classification_accuracy",
                "verified_accuracy",
                "response_rate",
                "precision",
                "recall",
                "f1",
                "gt_positive_rate",
                "pred_positive_rate",
            }:
                cells.append(_fmt(v, digits=4))
            elif c == "avg_cost_usd_per_case":
                cells.append(_fmt(v, digits=6))
            else:
                cells.append(str(v))
        print(" | ".join(cells))


def _include_in_paper_outputs(row: dict[str, Any]) -> bool:
    dataset = str(row.get("dataset") or "").strip()
    if dataset not in {"trace-dataset", "impossiblebench_claude-opus-4.6"}:
        return False
    method_base = str(row.get("method_base") or "").strip()
    method_variant = str(row.get("method_variant") or "").strip()
    if method_base == "llmjudge":
        return True
    if method_base == "AT" and method_variant in {"", "no-tools"}:
        return True
    if method_base in {"buffer", "bayesian"}:
        return True
    return False


def _paper_method_rank(row: dict[str, Any]) -> int:
    method_base = str(row.get("method_base") or "").strip()
    method_variant = str(row.get("method_variant") or "").strip()
    if method_base == "llmjudge":
        return 0
    if method_base == "AT" and method_variant == "":
        return 1
    if method_base == "AT" and method_variant == "max-merge":
        return 2
    if method_base == "bayesian":
        return 3
    if method_base == "buffer":
        return 4
    if method_base == "AT" and method_variant == "no-tools":
        return 5
    return 99


def _print_paper_table(rows: list[dict[str, Any]]) -> None:
    grouped_rows = [
        row
        for dataset_rows in _group_plot_rows_by_dataset(rows).values()
        for row in dataset_rows
    ]
    paper_rows = [row for row in grouped_rows if _include_in_paper_outputs(row)]
    if not paper_rows:
        print("\n## Safety Paper Table\n(no rows)")
        return

    paper_rows = sorted(
        paper_rows,
        key=lambda row: (
            _display_model_name(str(row.get("method_model") or "")) or "",
            _paper_method_rank(row),
        ),
    )
    cols = [
        ("dataset", "dataset"),
        ("model", "model"),
        ("method", "method"),
        ("verified_macro_f1", "verified_macro_f1"),
        ("case_average_precision", "case_ap"),
        ("trace_average_precision", "trace_ap"),
    ]
    print("\n## Safety Paper Table")
    print(" | ".join(label for _, label in cols))
    print(" | ".join("---" for _ in cols))
    for row in paper_rows:
        values = {
            "dataset": OVERALL_DATASET_LABELS.get(str(row.get("dataset") or ""), str(row.get("dataset") or "")),
            "model": _display_model_name(str(row.get("method_model") or "")) or "",
            "method": _presentation_method_label(
                method_base=str(row.get("method_base") or ""),
                dataset_variant=str(row.get("dataset_variant") or ""),
                method_variant=str(row.get("method_variant") or ""),
            ),
            "verified_macro_f1": _fmt(row.get("verified_macro_f1"), digits=4),
            "case_average_precision": _fmt(row.get("case_average_precision"), digits=4),
            "trace_average_precision": _fmt(row.get("trace_average_precision"), digits=4),
        }
        print(" | ".join(str(values[key]) for key, _ in cols))


def _paper_short_model_label(model_name: str) -> str:
    if model_name == "Qwen-3.5":
        return "Qwen3.5"
    if model_name == "gpt-5.4-mini":
        return "GPT-5.4m"
    return model_name


def _paper_latex_result_cell(value: float | None, se: float | None, *, best: bool) -> str:
    if value is None:
        return r"\na"
    se_value = 0.0 if se is None else se
    macro = r"\bestres" if best else r"\res"
    return f"{macro}{{{value:.3f}}}{{{se_value:.3f}}}"


def _paper_latex_delta(meerkat: float | None, baselines: list[float | None]) -> str:
    valid = [value for value in baselines if value is not None]
    if meerkat is None or not valid:
        return r"\na"
    delta = meerkat - max(valid)
    return f"{delta:+.3f}"


def _write_safety_metric_tables(case_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in case_rows:
        if not str(row.get("group", "")).isdigit():
            continue
        file_path = str(row.get("file") or "")
        dataset_name, method_name = _infer_dataset_and_method_from_file(file_path)
        dataset_base, dataset_variant = _split_dataset_variant(dataset_name)
        inferred_method_name, inferred_method_variant = _strip_method_variant(method_name)
        inferred_method_base, inferred_method_model = _split_method_parts(inferred_method_name)
        method_base = str(row.get("method_base_override") or inferred_method_base or "").strip()
        method_variant = str(row.get("method_variant_override") or inferred_method_variant or dataset_variant or "").strip()
        method_model = _display_model_name(
            str(row.get("method_model") or inferred_method_model or "").strip()
        )
        synthesized = dict(row)
        synthesized["dataset"] = dataset_base
        synthesized["dataset_variant"] = dataset_variant
        synthesized["method_base"] = method_base
        synthesized["method_variant"] = method_variant
        synthesized["method_model"] = method_model
        if _include_in_paper_outputs(synthesized):
            rows.append(synthesized)
    if not rows:
        return []

    dataset_order = ["trace-dataset", "impossiblebench_claude-opus-4.6"]
    model_order = ["Qwen-3.5", "gpt-5.4-mini", "GLM-5"]
    tpc_order = [10, 25, 50, 100]
    method_order = ["Meerkat", "Per-trace Monitor", "Bayesian", "Buffer", "Naive Agent"]
    table_specs = [
        (
            "trace_average_precision",
            "trace_average_precision_bootstrap_se",
            "Safety Trace AP Table",
            "Trace-level average precision",
            "safety_paper_trace_ap_table",
            "tab:safety-trace-ap",
        ),
        (
            "case_average_precision",
            "case_average_precision_bootstrap_se",
            "Safety Case AP Table",
            "Case-level average precision",
            "safety_paper_case_ap_table",
            "tab:safety-case-ap",
        ),
        (
            "trace_roc_auc",
            "trace_roc_auc_bootstrap_se",
            "Safety Trace ROC-AUC Table",
            "Trace-level ROC-AUC",
            "safety_paper_trace_rocauc_table",
            "tab:safety-trace-rocauc",
        ),
        (
            "case_roc_auc",
            "case_roc_auc_bootstrap_se",
            "Safety Case ROC-AUC Table",
            "Case-level ROC-AUC",
            "safety_paper_case_rocauc_table",
            "tab:safety-case-rocauc",
        ),
    ]

    output_paths: list[Path] = []
    for metric_key, se_key, md_title, caption_metric, stem, latex_label in table_specs:
        grouped: dict[tuple[str, int, str], dict[str, tuple[float | None, float | None]]] = {}
        for row in rows:
            dataset = str(row.get("dataset") or "")
            tpc = int(str(row.get("group") or "0"))
            model = _display_model_name(str(row.get("method_model") or "")) or ""
            method_label = _presentation_method_label(
                method_base=str(row.get("method_base") or ""),
                dataset_variant=str(row.get("dataset_variant") or ""),
                method_variant=str(row.get("method_variant") or ""),
            )
            grouped.setdefault((dataset, tpc, model), {})[method_label] = (
                _to_float(row.get(metric_key)),
                _to_float(row.get(se_key)),
            )

        md_lines = [
            f"# {md_title}",
            "",
            "| Domain | TPC | Model | Meerkat | Monitor | Bayesian | Buffer | Naive Agent | Delta |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        tex_lines = [
            "% Auto-generated by scripts/analyze_safety_case_metrics.py",
            r"\begin{table*}[t]",
            r"  \centering",
            r"  \small",
            r"  % \setlength{\tabcolsep}{3.5pt}",
            r"  % \renewcommand{\arraystretch}{0.95}",
            rf"  \caption{{{caption_metric} across safety corpora by traces per case. Higher is better. $\Delta$ is Meerkat minus the strongest baseline.}}",
            rf"  \label{{{latex_label}}}",
            r"  \begin{tabular}{@{}lllrrrrrr@{}}",
            r"    \toprule",
            r"    \multirow{2}{*}{Domain} & \multirow{2}{*}{TPC} & \multirow{2}{*}{Model}",
            r"      & \multicolumn{5}{c}{Method} & \multirow{2}{*}{$\Delta$} \\",
            r"    \cmidrule(lr){4-8}",
            r"      & & & Meerkat & Monitor & Bayesian & Buffer & Naive Agent & \\",
            r"    \midrule",
        ]

        tex_domain_chunks: list[str] = []
        for dataset in dataset_order:
            dataset_rows: list[str] = []
            present_tpcs = [tpc for tpc in tpc_order if any((dataset, tpc, model) in grouped for model in model_order)]
            if not present_tpcs:
                continue
            domain_label = TRACE_SCORE_DATASET_LABELS.get(dataset, dataset)
            domain_row_count = sum(1 for tpc in present_tpcs for model in model_order if (dataset, tpc, model) in grouped)
            domain_printed = False
            for tpc_idx, tpc in enumerate(present_tpcs):
                present_models = [model for model in model_order if (dataset, tpc, model) in grouped]
                if not present_models:
                    continue
                for model_idx, model in enumerate(present_models):
                    values = grouped[(dataset, tpc, model)]
                    model_cells = {name: values.get(name, (None, None)) for name in method_order}
                    numeric_values = {name: pair[0] for name, pair in model_cells.items() if pair[0] is not None}
                    max_value = max(numeric_values.values()) if numeric_values else None
                    monitor_value = model_cells["Per-trace Monitor"][0]
                    bayesian_value = model_cells["Bayesian"][0]
                    buffer_value = model_cells["Buffer"][0]
                    naive_value = model_cells["Naive Agent"][0]
                    meerkat_value = model_cells["Meerkat"][0]
                    delta_text = _paper_latex_delta(meerkat_value, [monitor_value, bayesian_value, buffer_value, naive_value])
                    if not domain_printed:
                        prefix = rf"    \multirow{{{domain_row_count}}}{{*}}{{{domain_label}}}"
                        domain_printed = True
                    else:
                        prefix = "    "
                    if model_idx == 0:
                        tpc_prefix = rf" & \multirow{{{len(present_models)}}}{{*}}{{{tpc}}}"
                    else:
                        tpc_prefix = " &"
                    model_label = _paper_short_model_label(model)
                    cells = [
                        _paper_latex_result_cell(*model_cells["Meerkat"], best=(max_value is not None and model_cells["Meerkat"][0] == max_value)),
                        _paper_latex_result_cell(*model_cells["Per-trace Monitor"], best=(max_value is not None and model_cells["Per-trace Monitor"][0] == max_value)),
                        _paper_latex_result_cell(*model_cells["Bayesian"], best=(max_value is not None and model_cells["Bayesian"][0] == max_value)),
                        _paper_latex_result_cell(*model_cells["Buffer"], best=(max_value is not None and model_cells["Buffer"][0] == max_value)),
                        _paper_latex_result_cell(*model_cells["Naive Agent"], best=(max_value is not None and model_cells["Naive Agent"][0] == max_value)),
                    ]
                    dataset_rows.append(f"{prefix}{tpc_prefix} & {model_label} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {cells[4]} & {delta_text} \\\\")
                    md_delta = "na" if delta_text == r"\na" else delta_text

                    def _fmt_md(pair: tuple[float | None, float | None], best: bool) -> str:
                        value, se = pair
                        if value is None:
                            return "na"
                        text = f"{value:.3f} +/- {(0.0 if se is None else se):.3f}"
                        return f"**{text}**" if best else text

                    md_lines.append(
                        "| "
                        + " | ".join(
                            [
                                domain_label,
                                str(tpc),
                                model_label,
                                _fmt_md(model_cells["Meerkat"], max_value is not None and model_cells["Meerkat"][0] == max_value),
                                _fmt_md(model_cells["Per-trace Monitor"], max_value is not None and model_cells["Per-trace Monitor"][0] == max_value),
                                _fmt_md(model_cells["Bayesian"], max_value is not None and model_cells["Bayesian"][0] == max_value),
                                _fmt_md(model_cells["Buffer"], max_value is not None and model_cells["Buffer"][0] == max_value),
                                _fmt_md(model_cells["Naive Agent"], max_value is not None and model_cells["Naive Agent"][0] == max_value),
                                md_delta,
                            ]
                        )
                        + " |"
                    )
                if tpc_idx != len(present_tpcs) - 1:
                    dataset_rows.append(r"    \cmidrule(lr){2-9}")
            tex_domain_chunks.extend(dataset_rows)
            tex_domain_chunks.append(r"    \midrule")

        if tex_domain_chunks and tex_domain_chunks[-1] == r"    \midrule":
            tex_domain_chunks.pop()
        tex_lines.extend(tex_domain_chunks)
        tex_lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"])

        md_path = output_dir / f"{stem}.md"
        tex_path = output_dir / f"{stem}.tex"
        md_path.write_text("\n".join(md_lines).rstrip() + "\n")
        tex_path.write_text("\n".join(tex_lines).rstrip() + "\n")
        output_paths.extend([md_path, tex_path])

    return output_paths


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text).strip().lower())
    return value.strip("-") or "unknown"


def _dataset_output_slug(dataset: str) -> str:
    dataset = str(dataset or "").strip()
    if dataset == "impossiblebench_claude-opus-4.6":
        return "impossiblebench"
    if dataset == "trace-dataset":
        return "trace"
    return _slug(dataset)


def _infer_dataset_and_method_from_file(file_path: str) -> tuple[str, str]:
    stem = Path(file_path).stem
    name = stem
    if name.startswith("safety_"):
        name = name[len("safety_") :]

    for marker in ("_buffer-", "_bayesian-", "_AT-", "_llmjudge-"):
        if marker in name:
            idx = name.rfind(marker)
            return name[:idx], name[idx + 1 :]
    if "_AT-" in name:
        idx = name.rfind("_AT-")
        return name[:idx], name[idx + 1 :]
    if "_llmjudge-" in name:
        idx = name.rfind("_llmjudge-")
        return name[:idx], name[idx + 1 :]
    if name.endswith("_llmjudge"):
        return name[: -len("_llmjudge")], "llmjudge"

    if "_" in name:
        dataset, method = name.rsplit("_", 1)
        return dataset, method
    return "unknown", name


def _split_dataset_variant(dataset: str) -> tuple[str, str | None]:
    for suffix in ("_no-tools", "_no_tools"):
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)], "no-tools"
    return dataset, None


def _apply_publication_style(plt) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": TICK_LABEL_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.titlesize": AXIS_LABEL_FONTSIZE,
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def _method_color(method: str, fallback_index: int, palette: list[str]) -> str:
    color = METHOD_COLORS.get(method)
    if color:
        return color
    if palette:
        return palette[fallback_index % len(palette)]
    return f"C{fallback_index % 10}"


def _method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def _model_color(model: str | None, fallback_index: int, palette: list[str]) -> str:
    display = _display_model_name(model)
    if display:
        color = MODEL_COLORS.get(display)
        if color:
            return color
    if palette:
        return palette[fallback_index % len(palette)]
    return f"C{fallback_index % 10}"


def _lighten_color(color: str, amount: float = 0.35) -> str:
    color = str(color or "").strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        return color
    amount = max(0.0, min(1.0, float(amount)))
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    r = int(round(r + (255 - r) * amount))
    g = int(round(g + (255 - g) * amount))
    b = int(round(b + (255 - b) * amount))
    return f"#{r:02X}{g:02X}{b:02X}"


def _darken_color(color: str, amount: float = 0.2) -> str:
    color = str(color or "").strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        return color
    amount = max(0.0, min(1.0, float(amount)))
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    r = int(round(r * (1.0 - amount)))
    g = int(round(g * (1.0 - amount)))
    b = int(round(b * (1.0 - amount)))
    return f"#{r:02X}{g:02X}{b:02X}"


def _method_hatch(method_base: str | None) -> str:
    return METHOD_HATCHES.get(str(method_base or "").strip(), "")


def _method_variant_marker(method_base: str | None, method_variant: str | None) -> str:
    base = str(method_base or "").strip()
    variant = str(method_variant or "").strip()
    if base == "llmjudge":
        return "s"
    if variant == "max-merge":
        return "D"
    if variant == "no-tools":
        return "^"
    if base == "AT-codex":
        return "D"
    if base == "AT-claude":
        return "P"
    if base == "AT":
        return "o"
    return _method_marker(base or variant or "AT")


def _method_variant_linestyle(method_base: str | None, method_variant: str | None) -> str:
    base = str(method_base or "").strip()
    variant = str(method_variant or "").strip()
    if base == "llmjudge":
        return "--"
    if variant == "max-merge":
        return "-."
    if variant == "no-tools":
        return ":"
    return "-"


def _method_fill_color(
    method_model: str | None,
    method_base: str | None,
    method_variant: str | None,
    fallback_index: int,
    palette: list[str],
) -> str:
    color = _model_color(method_model, fallback_index, palette)
    base = str(method_base or "").strip()
    if base == "AT-codex":
        return _darken_color(color, amount=0.18)
    if base == "AT-claude":
        return _lighten_color(color, amount=0.18)
    if str(method_variant or "").strip() == "max-merge":
        return _darken_color(color, amount=0.18)
    if str(method_variant or "").strip() == "no-tools":
        return _lighten_color(color, amount=0.38)
    return color


def _method_info_by_label(rows: list[dict[str, Any]]) -> dict[str, tuple[str | None, str | None, str | None]]:
    info: dict[str, tuple[str | None, str | None, str | None]] = {}
    for row in rows:
        label = str(row.get("method_label") or "")
        if not label or label in info:
            continue
        info[label] = (
            str(row.get("method_model") or "").strip() or None,
            str(row.get("method_base") or "").strip() or None,
            str(row.get("method_variant") or "").strip() or None,
        )
    return info


def _pct_bucket_sort_key(label_text: str) -> tuple[int, float]:
    if label_text == "unknown":
        return (2, 999.0)
    if label_text == "0%":
        return (0, 0.0)
    try:
        upper = float(label_text.split(",")[1].split("]")[0])
    except Exception:
        upper = 999.0
    return (1, upper)


def _group_plot_rows_by_dataset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset, method = _infer_dataset_and_method_from_file(str(row.get("file") or ""))
        inferred_dataset_base, inferred_dataset_variant = _split_dataset_variant(dataset)
        stripped_method, inferred_method_variant = _strip_method_variant(method)
        inferred_method_base, inferred_method_model = _split_method_parts(stripped_method)
        dataset_base = row.get("dataset_override")
        if dataset_base is None or str(dataset_base).strip() == "":
            dataset_base = inferred_dataset_base
        else:
            dataset_base = str(dataset_base)
        dataset_variant = row.get("dataset_variant_override")
        if dataset_variant is None or str(dataset_variant).strip() == "":
            dataset_variant = inferred_dataset_variant
        else:
            dataset_variant = str(dataset_variant)
        method_base = row.get("method_base_override")
        if method_base is None or str(method_base).strip() == "":
            method_base = inferred_method_base
        else:
            method_base = str(method_base)
        method_model = row.get("method_model_override")
        if method_model is None or str(method_model).strip() == "":
            method_model = inferred_method_model
        else:
            method_model = str(method_model)
        method_variant = row.get("method_variant_override")
        if method_variant is None:
            method_variant = inferred_method_variant or dataset_variant
        elif str(method_variant).strip() == "":
            method_variant = inferred_method_variant or dataset_variant
        out = dict(row)
        out["dataset"] = dataset_base
        out["dataset_variant"] = dataset_variant
        out["method_key"] = stripped_method
        out["method_model"] = method_model
        out["method_variant"] = method_variant
        out["method_label"] = _pretty_method(
            stripped_method,
            method_model=method_model,
            dataset_variant=dataset_variant,
            method_variant=method_variant,
        )
        out["method_base"] = method_base
        grouped.setdefault(dataset_base, []).append(out)
    return grouped


def _include_in_overall_dataset_chart(dataset_name: str) -> bool:
    dataset = str(dataset_name or "").strip().lower()
    return dataset in OVERALL_DATASET_LABELS


def _include_in_trace_score_dataset_chart(dataset_name: str) -> bool:
    dataset = str(dataset_name or "").strip().lower()
    return dataset in TRACE_SCORE_DATASET_LABELS


def _line_plot(
    *,
    title: str,
    x_values: list[Any],
    x_labels: list[str],
    y_by_method: dict[str, list[float | None]],
    yerr_by_method: dict[str, list[tuple[float, float] | None]] | None,
    method_info_by_label: dict[str, tuple[str | None, str | None, str | None]] | None,
    ylabel: str,
    xlabel: str,
    x_label_rotation: int,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, LINE_FIGURE_HEIGHT_IN), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.62)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    for i, (method, ys) in enumerate(sorted(y_by_method.items(), key=lambda kv: kv[0])):
        xs = []
        vals = []
        lower_bounds = []
        upper_bounds = []
        for j, y in enumerate(ys):
            if y is None:
                continue
            y_val = float(y)
            xs.append(j)
            vals.append(y_val)
            err = None
            if yerr_by_method is not None:
                method_errs = yerr_by_method.get(method) or []
                if j < len(method_errs):
                    err = method_errs[j]
            if err is None:
                lower_bounds.append(y_val)
                upper_bounds.append(y_val)
            else:
                low_delta = max(0.0, float(err[0]))
                high_delta = max(0.0, float(err[1]))
                lower_bounds.append(y_val - low_delta)
                upper_bounds.append(y_val + high_delta)
        if not xs:
            continue
        method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
        color = _method_fill_color(method_model, method_base, method_variant, i, palette)
        if yerr_by_method is not None:
            ax.fill_between(
                xs,
                lower_bounds,
                upper_bounds,
                color=color,
                alpha=0.12,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            xs,
            vals,
            marker=_method_variant_marker(method_base, method_variant),
            linestyle=_method_variant_linestyle(method_base, method_variant),
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=color,
            zorder=2,
        )

    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
    ax.set_xticks(list(range(len(x_values))))
    ax.set_xticklabels(x_labels, rotation=x_label_rotation)
    ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    present_models: list[str] = []
    present_method_styles: list[tuple[str, str, str]] = []
    seen_method_styles: set[tuple[str, str, str]] = set()
    for method in sorted(y_by_method):
        method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
        display_model = _display_model_name(method_model)
        if display_model and display_model not in present_models:
            present_models.append(display_model)
        method_label = _presentation_method_label(method_base=method_base, method_variant=method_variant)
        if method_label not in {
            "Meerkat",
            "Meerkat + Max Merge",
            "Naive Agent",
            "Per-trace Monitor",
        }:
            continue
        style_key = (
            method_label,
            _method_variant_marker(method_base, method_variant),
            _method_variant_linestyle(method_base, method_variant),
        )
        if style_key not in seen_method_styles:
            seen_method_styles.add(style_key)
            present_method_styles.append(style_key)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=_model_color(model_name, i, palette),
            lw=LINE_WIDTH + 0.2,
            marker="o",
            markersize=MARKER_SIZE,
            linestyle="-",
            label=model_name,
        )
        for i, model_name in enumerate(present_models)
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            lw=LINE_WIDTH + 0.2,
            marker=marker,
            markersize=MARKER_SIZE,
            linestyle=linestyle,
            label=label,
        )
        for label, marker, linestyle in present_method_styles
    ]

    legend_kwargs = dict(
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    if model_handles:
        model_legend = fig.legend(
            handles=model_handles,
            loc="upper center",
            bbox_to_anchor=(0.62, 0.985),
            ncol=max(1, min(2, len(model_handles))),
            title="Model",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )
        fig.add_artist(model_legend)
    if method_handles:
        fig.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.64, 0.845),
            ncol=max(1, min(3, len(method_handles))),
            title="Method",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )
    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _ap_by_case_size_plot(
    *,
    dataset: str,
    x_values: list[Any],
    x_labels: list[str],
    case_ap_by_method: dict[str, list[float | None]],
    case_ap_err_by_method: dict[str, list[tuple[float, float] | None]],
    trace_ap_by_method: dict[str, list[float | None]],
    trace_ap_err_by_method: dict[str, list[tuple[float, float] | None]],
    method_info_by_label: dict[str, tuple[str | None, str | None, str | None]] | None,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN * 2.2, LINE_FIGURE_HEIGHT_IN * 1.16), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.76, wspace=0.32)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    panels = [
        (axes[0], "Case AP", case_ap_by_method, case_ap_err_by_method),
        (axes[1], "Trace AP", trace_ap_by_method, trace_ap_err_by_method),
    ]
    for ax, ylabel, y_by_method, yerr_by_method in panels:
        for i, (method, ys) in enumerate(sorted(y_by_method.items(), key=lambda kv: kv[0])):
            method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
            if _presentation_method_label(method_base=method_base, method_variant=method_variant) not in {
                "Meerkat",
                "Per-trace Monitor",
                "Naive Agent",
            }:
                continue
            xs = []
            vals = []
            lower_bounds = []
            upper_bounds = []
            for j, y in enumerate(ys):
                if y is None:
                    continue
                y_val = float(y)
                xs.append(j)
                vals.append(y_val)
                err = None
                method_errs = yerr_by_method.get(method) or []
                if j < len(method_errs):
                    err = method_errs[j]
                if err is None:
                    lower_bounds.append(y_val)
                    upper_bounds.append(y_val)
                else:
                    low_delta = max(0.0, float(err[0]))
                    high_delta = max(0.0, float(err[1]))
                    lower_bounds.append(y_val - low_delta)
                    upper_bounds.append(y_val + high_delta)
            if not xs:
                continue
            color = _method_fill_color(method_model, method_base, method_variant, i, palette)
            ax.fill_between(xs, lower_bounds, upper_bounds, color=color, alpha=0.12, linewidth=0.0, zorder=1)
            ax.plot(
                xs,
                vals,
                marker=_method_variant_marker(method_base, method_variant),
                linestyle=_method_variant_linestyle(method_base, method_variant),
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                color=color,
                zorder=2,
            )
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
        ax.set_xlabel("Traces per Case", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
        ax.set_xticks(list(range(len(x_values))))
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)

    present_models: list[str] = []
    present_method_styles: list[tuple[str, str, str]] = []
    seen_method_styles: set[tuple[str, str, str]] = set()
    for method in sorted(set(case_ap_by_method) | set(trace_ap_by_method)):
        method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
        display_model = _display_model_name(method_model)
        if display_model and display_model not in present_models:
            present_models.append(display_model)
        method_label = _presentation_method_label(method_base=method_base, method_variant=method_variant)
        if method_label not in {"Meerkat", "Per-trace Monitor", "Naive Agent"}:
            continue
        style_key = (
            method_label,
            _method_variant_marker(method_base, method_variant),
            _method_variant_linestyle(method_base, method_variant),
        )
        if style_key not in seen_method_styles:
            seen_method_styles.add(style_key)
            present_method_styles.append(style_key)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=_model_color(model_name, i, palette),
            lw=LINE_WIDTH + 0.2,
            marker="o",
            markersize=MARKER_SIZE,
            linestyle="-",
            label=model_name,
        )
        for i, model_name in enumerate(present_models)
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            lw=LINE_WIDTH + 0.2,
            marker=marker,
            markersize=MARKER_SIZE,
            linestyle=linestyle,
            label=label,
        )
        for label, marker, linestyle in present_method_styles
    ]
    legend_kwargs = dict(frameon=False, fontsize=LEGEND_FONTSIZE, borderpad=0.15, labelspacing=0.3, handletextpad=0.5, columnspacing=0.8)
    if model_handles:
        fig.legend(
            handles=model_handles,
            loc="upper center",
            bbox_to_anchor=(0.47, 0.99),
            ncol=max(1, min(3, len(model_handles))),
            title="Model",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )
    if method_handles:
        fig.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.47, 0.90),
            ncol=max(1, min(3, len(method_handles))),
            title="Method",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _precision_recall_plot(
    *,
    curves_by_method: dict[str, list[dict[str, float]]],
    roc_curves_by_method: dict[str, list[dict[str, float]]],
    method_info_by_label: dict[str, tuple[str | None, str | None, str | None]] | None,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    if not curves_by_method and not roc_curves_by_method:
        return []
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_IN, LINE_FIGURE_HEIGHT_IN * 1.8), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.10, top=0.70, hspace=0.32)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    for i, method in enumerate(sorted(set(curves_by_method) | set(roc_curves_by_method), key=lambda x: x)):
        method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
        color = _method_fill_color(method_model, method_base, method_variant, i, palette)
        pr_curve = curves_by_method.get(method) or []
        if pr_curve:
            recalls = [float(point.get("recall", 0.0)) for point in pr_curve]
            precisions = [float(point.get("precision", 0.0)) for point in pr_curve]
            axes[0].step(
                recalls,
                precisions,
                where="post",
                color=color,
                linewidth=LINE_WIDTH,
                linestyle=_method_variant_linestyle(method_base, method_variant),
                zorder=2,
            )
            axes[0].plot(
                [recalls[-1]],
                [precisions[-1]],
                marker=_method_variant_marker(method_base, method_variant),
                markersize=MARKER_SIZE,
                color=color,
                zorder=3,
            )
        roc_curve = roc_curves_by_method.get(method) or []
        if roc_curve:
            fprs = [float(point.get("fpr", 0.0)) for point in roc_curve]
            tprs = [float(point.get("tpr", 0.0)) for point in roc_curve]
            axes[1].step(
                fprs,
                tprs,
                where="post",
                color=color,
                linewidth=LINE_WIDTH,
                linestyle=_method_variant_linestyle(method_base, method_variant),
                zorder=2,
            )
            axes[1].plot(
                [fprs[-1]],
                [tprs[-1]],
                marker=_method_variant_marker(method_base, method_variant),
                markersize=MARKER_SIZE,
                color=color,
                zorder=3,
            )

    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
    axes[0].set_ylabel("Precision", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
    axes[1].set_ylabel("True Positive Rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
    axes[1].plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.8, linestyle=":", zorder=1)
    for ax in axes:
        ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)

    present_models: list[str] = []
    present_method_styles: list[tuple[str, str, str]] = []
    seen_method_styles: set[tuple[str, str, str]] = set()
    for method in sorted(curves_by_method):
        method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
        display_model = _display_model_name(method_model)
        if display_model and display_model not in present_models:
            present_models.append(display_model)
        method_label = "Judge" if method_base == "llmjudge" else "AT"
        if method_variant == "no-tools":
            method_label = "AT (No Tools)"
        elif method_base == "AT-codex":
            method_label = "AT (Codex)"
        elif method_base == "AT-claude":
            method_label = "AT (Claude)"
        style_key = (
            method_label,
            _method_variant_marker(method_base, method_variant),
            _method_variant_linestyle(method_base, method_variant),
        )
        if style_key not in seen_method_styles:
            seen_method_styles.add(style_key)
            present_method_styles.append(style_key)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=_model_color(model_name, i, palette),
            lw=LINE_WIDTH + 0.2,
            marker="o",
            markersize=MARKER_SIZE,
            linestyle="-",
            label=model_name,
        )
        for i, model_name in enumerate(present_models)
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            lw=LINE_WIDTH + 0.2,
            marker=marker,
            markersize=MARKER_SIZE,
            linestyle=linestyle,
            label=label,
        )
        for label, marker, linestyle in present_method_styles
    ]
    legend_kwargs = dict(
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    if model_handles:
        model_legend = fig.legend(
            handles=model_handles,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.995),
            ncol=max(1, min(2, len(model_handles))),
            title="Model",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )
        fig.add_artist(model_legend)
    if method_handles:
        fig.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.90),
            ncol=max(1, min(3, len(method_handles))),
            title="Method",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _calibration_plot(
    *,
    case_curves_by_method: dict[str, list[dict[str, float]]],
    case_metrics_by_method: dict[str, tuple[float | None, float | None]],
    trace_curves_by_method: dict[str, list[dict[str, float]]],
    trace_metrics_by_method: dict[str, tuple[float | None, float | None]],
    method_info_by_label: dict[str, tuple[str | None, str | None, str | None]] | None,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    if not case_curves_by_method and not trace_curves_by_method:
        return []
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN * 2.05, LINE_FIGURE_HEIGHT_IN * 1.02), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.17, top=0.80, wspace=0.24)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    method_order = sorted(set(case_curves_by_method) | set(trace_curves_by_method), key=lambda x: x)
    legend_handles: list[Line2D] = []
    seen_handles: set[str] = set()

    panels = [
        (
            axes[0],
            "Case-Level Calibration",
            case_curves_by_method,
            case_metrics_by_method,
        ),
        (
            axes[1],
            "Trace-Level Calibration",
            trace_curves_by_method,
            trace_metrics_by_method,
        ),
    ]

    for ax, title, curves_by_method, metrics_by_method in panels:
        ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.9, linestyle=":", zorder=1)
        for i, method in enumerate(method_order):
            curve = curves_by_method.get(method) or []
            if not curve:
                continue
            method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
            color = _method_fill_color(method_model, method_base, method_variant, i, palette)
            xs = [float(point.get("mean_score", 0.0)) for point in curve]
            ys = [float(point.get("positive_rate", 0.0)) for point in curve]
            ylow = [float(point.get("positive_rate_low", point.get("positive_rate", 0.0))) for point in curve]
            yhigh = [float(point.get("positive_rate_high", point.get("positive_rate", 0.0))) for point in curve]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=0.95,
                linestyle=_method_variant_linestyle(method_base, method_variant),
                alpha=0.75,
                zorder=2,
            )
            ax.fill_between(
                xs,
                ylow,
                yhigh,
                color=color,
                alpha=0.12,
                linewidth=0.0,
                zorder=1.5,
            )
            ax.scatter(
                xs,
                ys,
                s=22.0,
                color=color,
                marker=_method_variant_marker(method_base, method_variant),
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                zorder=3,
            )
            if method not in seen_handles:
                seen_handles.add(method)
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        lw=LINE_WIDTH,
                        marker=_method_variant_marker(method_base, method_variant),
                        markersize=max(5.0, MARKER_SIZE - 0.5),
                        linestyle=_method_variant_linestyle(method_base, method_variant),
                        label=method,
                    )
                )
        ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE, pad=4)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Mean Predicted Score", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
        ax.set_ylabel("Empirical Positive Rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
        ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, min(2, len(legend_handles))),
            frameon=False,
            fontsize=max(6.2, LEGEND_FONTSIZE - 0.2),
            handlelength=1.7,
            handletextpad=0.45,
            columnspacing=0.9,
        )

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _overall_calibration_grid_plot(
    *,
    case_curves_by_dataset: dict[str, dict[str, list[dict[str, float]]]],
    case_metrics_by_dataset: dict[str, dict[str, tuple[float | None, float | None]]],
    trace_curves_by_dataset: dict[str, dict[str, list[dict[str, float]]]],
    trace_metrics_by_dataset: dict[str, dict[str, tuple[float | None, float | None]]],
    method_info_by_dataset: dict[str, dict[str, tuple[str | None, str | None, str | None]]],
    dataset_labels: dict[str, str],
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    datasets = [
        dataset
        for dataset in dataset_labels
        if (case_curves_by_dataset.get(dataset) or trace_curves_by_dataset.get(dataset))
    ]
    if not datasets:
        return []
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(FIGURE_WIDTH_IN * 2.05, max(3.1, 2.05 * len(datasets))),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.10, top=0.83, wspace=0.20, hspace=0.38)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    method_order = sorted(
        {
            method
            for dataset in datasets
            for method in (
                set(case_curves_by_dataset.get(dataset, {})) | set(trace_curves_by_dataset.get(dataset, {}))
            )
        }
    )
    legend_handles: list[Line2D] = []
    seen_handles: set[str] = set()

    for row_idx, dataset in enumerate(datasets):
        panels = [
            (
                axes[row_idx][0],
                "Case-Level Calibration",
                case_curves_by_dataset.get(dataset) or {},
                case_metrics_by_dataset.get(dataset) or {},
            ),
            (
                axes[row_idx][1],
                "Trace-Level Calibration",
                trace_curves_by_dataset.get(dataset) or {},
                trace_metrics_by_dataset.get(dataset) or {},
            ),
        ]
        for col_idx, (ax, panel_title, curves_by_method, metrics_by_method) in enumerate(panels):
            ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.9, linestyle=":", zorder=1)
            for i, method in enumerate(method_order):
                curve = curves_by_method.get(method) or []
                if not curve:
                    continue
                method_model, method_base, method_variant = (method_info_by_dataset.get(dataset) or {}).get(
                    method, (None, None, None)
                )
                color = _method_fill_color(method_model, method_base, method_variant, i, palette)
                xs = [float(point.get("mean_score", 0.0)) for point in curve]
                ys = [float(point.get("positive_rate", 0.0)) for point in curve]
                ylow = [float(point.get("positive_rate_low", point.get("positive_rate", 0.0))) for point in curve]
                yhigh = [float(point.get("positive_rate_high", point.get("positive_rate", 0.0))) for point in curve]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=0.95,
                    linestyle=_method_variant_linestyle(method_base, method_variant),
                    alpha=0.75,
                    zorder=2,
                )
                ax.fill_between(
                    xs,
                    ylow,
                    yhigh,
                    color=color,
                    alpha=0.12,
                    linewidth=0.0,
                    zorder=1.5,
                )
                ax.scatter(
                    xs,
                    ys,
                    s=20.0,
                    color=color,
                    marker=_method_variant_marker(method_base, method_variant),
                    edgecolors="white",
                    linewidths=0.45,
                    alpha=0.95,
                    zorder=3,
                )
                if method not in seen_handles:
                    seen_handles.add(method)
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            lw=LINE_WIDTH,
                            marker=_method_variant_marker(method_base, method_variant),
                            markersize=max(5.0, MARKER_SIZE - 0.6),
                            linestyle=_method_variant_linestyle(method_base, method_variant),
                            label=method,
                        )
                    )

            title = f"{dataset_labels.get(dataset, dataset)} — {'Case' if col_idx == 0 else 'Trace'}"
            ax.set_title(title, fontsize=max(6.6, AXIS_LABEL_FONTSIZE - 0.2), pad=4)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("Mean Predicted Score", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.3)
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Empirical Positive Rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.3)
            else:
                ax.set_ylabel("")
            ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(SPINE_COLOR)
            ax.spines["bottom"].set_color(SPINE_COLOR)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, min(2, len(legend_handles))),
            frameon=False,
            fontsize=max(6.1, LEGEND_FONTSIZE - 0.2),
            handlelength=1.7,
            handletextpad=0.45,
            columnspacing=0.9,
        )

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _bar_plot_with_ci(
    *,
    title: str,
    method_labels: list[str],
    method_info_by_label: dict[str, tuple[str | None, str | None, str | None]] | None,
    values: list[float],
    lower_errs: list[float],
    upper_errs: list[float],
    ylabel: str,
    xlabel: str,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    if not method_labels:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, BAR_FIGURE_HEIGHT_IN), constrained_layout=True)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    colors: list[str] = []
    hatches: list[str] = []
    for i, label in enumerate(method_labels):
        method_model, method_base, method_variant = (method_info_by_label or {}).get(label, (None, None, None))
        colors.append(_method_fill_color(method_model, method_base, method_variant, i, palette))
        hatches.append(_method_hatch(method_base))
    x = list(range(len(method_labels)))
    bars = ax.bar(
        x,
        values,
        yerr=[lower_errs, upper_errs],
        capsize=2.5,
        ecolor="black",
        color=colors,
        edgecolor=SPINE_COLOR,
        linewidth=0.6,
        alpha=0.92,
    )
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.0)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=12)
    ax.grid(True, which="major", axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _grouped_overall_metric_bar_plot(
    *,
    rows: list[dict[str, Any]],
    metric_key: str,
    metric_low_key: str | None,
    metric_high_key: str | None,
    dataset_labels: dict[str, str],
    ylabel: str,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    grouped = {
        dataset: dataset_rows
        for dataset, dataset_rows in _group_plot_rows_by_dataset(rows).items()
        if dataset in dataset_labels
    }
    datasets = [dataset for dataset in dataset_labels if dataset in grouped]
    if not datasets:
        return []

    _apply_publication_style(plt)
    width = max(FIGURE_WIDTH_IN * 1.95, 5.3)
    height = max(BAR_FIGURE_HEIGHT_IN * 1.2, 2.25)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.18, top=0.76)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    method_style: dict[str, tuple[str | None, str | None, str | None]] = {}
    for dataset_rows in grouped.values():
        for row in dataset_rows:
            label = str(row.get("method_label") or "")
            if not label or label in method_style:
                continue
            method_style[label] = (
                str(row.get("method_model") or "").strip() or None,
                str(row.get("method_base") or "").strip() or None,
                str(row.get("method_variant") or "").strip() or None,
            )

    preferred_model_order = ("gpt-5-mini", "gpt-5.4-mini", "gpt-5.4", "GLM-5", "Qwen-3.5", "MiniMax-M2.5")
    model_order = [
        model_name
        for model_name in preferred_model_order
        if any((info[0] and _display_model_name(info[0]) == model_name) for info in method_style.values())
    ]
    seen_method_order: list[str] = []
    for method in ("AT", "AT (Codex)", "AT (Claude)", "AT (No Tools)", "Judge"):
        if any(
            (
                (info[1] == "AT" and method == "AT" and info[2] != "no-tools")
                or (info[1] == "AT-codex" and method == "AT (Codex)")
                or (info[1] == "AT-claude" and method == "AT (Claude)")
                or (info[1] == "AT" and method == "AT (No Tools)" and info[2] == "no-tools")
                or (info[1] == "llmjudge" and method == "Judge")
            )
            for info in method_style.values()
        ):
            seen_method_order.append(method)

    def _overall_method_rank(method_base: str | None, method_variant: str | None) -> int:
        if method_base == "AT" and method_variant != "no-tools":
            return 0
        if method_base == "AT-codex":
            return 1
        if method_base == "AT-claude":
            return 2
        if method_base == "AT" and method_variant == "no-tools":
            return 3
        if method_base == "llmjudge":
            return 4
        return 99

    methods = sorted(
        {
            str(r.get("method_label") or "")
            for dataset_rows in grouped.values()
            for r in dataset_rows
            if str(r.get("method_label") or "")
        },
        key=lambda label: (
            model_order.index(_display_model_name(method_style.get(label, (None, None, None))[0]))
            if _display_model_name(method_style.get(label, (None, None, None))[0]) in model_order
            else 99,
            _overall_method_rank(method_style.get(label, (None, None, None))[1], method_style.get(label, (None, None, None))[2]),
            label,
        ),
    )
    if not methods:
        return []

    x = list(range(len(datasets)))
    per_model_gap = 0.06
    total_width = 0.84
    num_models = max(1, len(model_order))
    bars_per_model = max(
        1,
        max(
            sum(1 for m in methods if _display_model_name(method_style.get(m, (None, None, None))[0]) == model_name)
            for model_name in model_order
        ) if model_order else len(methods),
    )
    total_slots = len(methods)
    bar_width = (total_width - per_model_gap * max(0, num_models - 1)) / max(1, total_slots)

    x_positions_by_method: dict[str, list[float]] = {method: [] for method in methods}
    dataset_base_left = -total_width / 2
    model_start_index = 0
    for model_idx, model_name in enumerate(model_order):
        model_methods = [m for m in methods if _display_model_name(method_style.get(m, (None, None, None))[0]) == model_name]
        if not model_methods:
            continue
        model_left = dataset_base_left + model_start_index * bar_width + model_idx * per_model_gap
        for local_idx, method in enumerate(model_methods):
            center_offset = model_left + (local_idx + 0.5) * bar_width
            x_positions_by_method[method] = [dataset_idx + center_offset for dataset_idx in range(len(datasets))]
        model_start_index += len(model_methods)

    for method_idx, method in enumerate(methods):
        xs: list[float] = []
        heights: list[float] = []
        lower_errs: list[float] = []
        upper_errs: list[float] = []
        for dataset_idx, dataset in enumerate(datasets):
            dataset_rows = grouped.get(dataset, [])
            row = next((r for r in dataset_rows if str(r.get("method_label") or "") == method), None)
            if row is None:
                continue
            metric_value = _to_float(row.get(metric_key))
            if metric_value is None:
                continue
            xs.append(x_positions_by_method.get(method, [])[dataset_idx])
            heights.append(metric_value)
            metric_low = _to_float(row.get(metric_low_key)) if metric_low_key else None
            metric_high = _to_float(row.get(metric_high_key)) if metric_high_key else None
            lower_errs.append(
                max(0.0, metric_value - metric_low)
                if metric_low is not None
                else 0.0
            )
            upper_errs.append(
                max(0.0, metric_high - metric_value)
                if metric_high is not None
                else 0.0
            )
        if not xs:
            continue
        method_model, method_base, method_variant = method_style.get(method, (None, None, None))
        bar_color = _method_fill_color(method_model, method_base, method_variant, method_idx, palette)
        ax.bar(
            xs,
            heights,
            width=bar_width * 0.9,
            yerr=[lower_errs, upper_errs],
            capsize=2.5,
            ecolor="black",
            color=bar_color,
            edgecolor=SPINE_COLOR,
            linewidth=0.6,
            alpha=0.97,
            hatch=_method_hatch(method_base),
        )

    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.0)
    ax.set_xlabel("Dataset", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_labels.get(dataset, dataset) for dataset in datasets], rotation=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="major", axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    model_handles = [
        Patch(
            facecolor=_model_color(model_name, i, palette),
            edgecolor=SPINE_COLOR,
            linewidth=0.6,
            label=model_name,
        )
        for i, model_name in enumerate(model_order)
    ]
    method_handles = []
    if "AT" in seen_method_order:
        method_handles.append(
            Patch(facecolor="#BEBEBE", edgecolor=SPINE_COLOR, linewidth=0.6, label="AT")
        )
    if "AT (Codex)" in seen_method_order:
        method_handles.append(
            Patch(facecolor="#BEBEBE", edgecolor=SPINE_COLOR, linewidth=0.6, label="AT (Codex)")
        )
    if "AT (Claude)" in seen_method_order:
        method_handles.append(
            Patch(facecolor="#BEBEBE", edgecolor=SPINE_COLOR, linewidth=0.6, label="AT (Claude)")
        )
    if "AT (No Tools)" in seen_method_order:
        method_handles.append(
            Patch(facecolor="#E0E0E0", edgecolor=SPINE_COLOR, linewidth=0.6, label="AT (No Tools)")
        )
    if "Judge" in seen_method_order:
        method_handles.append(
            Patch(facecolor="#BEBEBE", edgecolor=SPINE_COLOR, linewidth=0.6, hatch="///", label="Per-trace Monitor")
        )

    legend_kwargs = dict(
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    if model_handles:
        model_legend = fig.legend(
            handles=model_handles,
            loc="upper center",
            bbox_to_anchor=(0.34, 0.965),
            ncol=max(1, min(2, len(model_handles))),
            title="Model",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )
        fig.add_artist(model_legend)
    if method_handles:
        fig.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.78, 0.965),
            ncol=max(1, min(3, len(method_handles))),
            title="Method",
            title_fontsize=LEGEND_FONTSIZE,
            **legend_kwargs,
        )

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _overall_ap_by_model_plot(
    *,
    rows: list[dict[str, Any]],
    dataset: str,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    plot_rows = [
        row
        for row in rows
        if str(row.get("dataset") or "") == dataset
        and _include_in_paper_outputs(row)
        and str(row.get("method_variant") or "").strip() != "max-merge"
    ]
    if not plot_rows:
        return []

    method_rank = {
        "Meerkat": 0,
        "Naive Agent": 1,
        "Per-trace Monitor": 2,
    }
    model_order = ["Qwen-3.5", "gpt-5.4-mini", "GLM-5"]
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in plot_rows:
        model_name = _display_model_name(str(row.get("method_model") or ""))
        method_label = _presentation_method_label(
            method_base=str(row.get("method_base") or ""),
            method_variant=str(row.get("method_variant") or ""),
            dataset_variant=str(row.get("dataset_variant") or ""),
        )
        if not model_name or method_label not in method_rank:
            continue
        grouped.setdefault(model_name, {})[method_label] = row
    model_names = [m for m in model_order if m in grouped] + [m for m in grouped if m not in model_order]
    if not model_names:
        return []

    _apply_publication_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN * 2.15, BAR_FIGURE_HEIGHT_IN * 1.35), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.72, wspace=0.28)

    panel_specs = [
        (axes[0], "Case AP", "case_average_precision", "case_average_precision_ci_low", "case_average_precision_ci_high"),
        (axes[1], "Trace AP", "trace_average_precision", "trace_average_precision_ci_low", "trace_average_precision_ci_high"),
    ]
    bar_width = 0.22
    model_gap = 0.30
    method_sequence = ["Meerkat", "Naive Agent", "Per-trace Monitor"]
    x_positions: list[float] = []
    x_labels: list[str] = []
    group_centers: list[float] = []
    cursor = 0.0
    for model_name in model_names:
        start = cursor
        for method_label in method_sequence:
            x_positions.append(cursor)
            x_labels.append("" if method_label != "Meerkat" else model_name)
            cursor += bar_width
        group_centers.append(start + bar_width)
        cursor += model_gap

    for ax, ylabel, metric_key, low_key, high_key in panel_specs:
        idx = 0
        for model_name in model_names:
            model_color = _model_color(model_name, 0, [])
            for method_label in method_sequence:
                row = grouped.get(model_name, {}).get(method_label)
                x = x_positions[idx]
                idx += 1
                if row is None:
                    continue
                value = _to_float(row.get(metric_key))
                if value is None:
                    continue
                low = _to_float(row.get(low_key))
                high = _to_float(row.get(high_key))
                lower_err = max(0.0, value - low) if low is not None else 0.0
                upper_err = max(0.0, high - value) if high is not None else 0.0
                face = _method_fill_color(
                    method_model=model_name,
                    method_base=str(row.get("method_base") or ""),
                    method_variant=str(row.get("method_variant") or ""),
                    fallback_index=0,
                    palette=[],
                )
                hatch = ""
                if method_label == "Per-trace Monitor":
                    hatch = "////"
                elif method_label == "Naive Agent":
                    hatch = ".."
                ax.bar(
                    x,
                    value,
                    width=bar_width * 0.88,
                    color=face,
                    edgecolor=SPINE_COLOR,
                    linewidth=0.6,
                    hatch=hatch,
                    yerr=[[lower_err], [upper_err]],
                    capsize=2.0,
                    ecolor="black",
                    zorder=3,
                )
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.0)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="major", axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)

    method_handles = [
        Patch(facecolor="#777777", edgecolor=SPINE_COLOR, label="Meerkat"),
        Patch(facecolor="#BBBBBB", edgecolor=SPINE_COLOR, hatch="..", label="Naive Agent"),
        Patch(facecolor="#DDDDDD", edgecolor=SPINE_COLOR, hatch="////", label="Per-trace Monitor"),
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        title="Method",
        title_fontsize=LEGEND_FONTSIZE,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.1,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _macro_f1_by_dataset_plot(
    *,
    rows: list[dict[str, Any]],
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    grouped_rows = [
        row
        for dataset_rows in _group_plot_rows_by_dataset(rows).values()
        for row in dataset_rows
    ]
    plot_rows = [
        row
        for row in grouped_rows
        if str(row.get("dataset") or "") in TRACE_SCORE_DATASET_LABELS
        and _include_in_paper_outputs(row)
        and str(row.get("method_variant") or "").strip() != "max-merge"
    ]
    if not plot_rows:
        return []

    method_rank = {
        "Meerkat": 0,
        "Naive Agent": 1,
        "Per-trace Monitor": 2,
    }
    model_order = ["Qwen-3.5", "gpt-5.4-mini", "GLM-5"]
    method_sequence = ["Meerkat", "Naive Agent", "Per-trace Monitor"]
    datasets = [
        dataset
        for dataset in ("impossiblebench_claude-opus-4.6", "trace-dataset")
        if dataset in {str(row.get("dataset") or "") for row in plot_rows}
    ]
    if not datasets:
        return []

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in plot_rows:
        dataset = str(row.get("dataset") or "")
        model_name = _display_model_name(str(row.get("method_model") or ""))
        method_label = _presentation_method_label(
            method_base=str(row.get("method_base") or ""),
            method_variant=str(row.get("method_variant") or ""),
            dataset_variant=str(row.get("dataset_variant") or ""),
        )
        if dataset not in datasets or not model_name or method_label not in method_rank:
            continue
        grouped.setdefault(dataset, {}).setdefault(model_name, {})[method_label] = row

    _apply_publication_style(plt)
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(FIGURE_WIDTH_IN * max(1.15, 1.12 * len(datasets)), BAR_FIGURE_HEIGHT_IN * 1.35),
        constrained_layout=False,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes.ravel())
        except Exception:
            axes = [axes]
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.74, wspace=0.25)

    method_handles = [
        Patch(facecolor="#777777", edgecolor=SPINE_COLOR, label="Meerkat"),
        Patch(facecolor="#BBBBBB", edgecolor=SPINE_COLOR, hatch="..", label="Naive Agent"),
        Patch(facecolor="#DDDDDD", edgecolor=SPINE_COLOR, hatch="////", label="Per-trace Monitor"),
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        title="Method",
        title_fontsize=LEGEND_FONTSIZE,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.1,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    for ax, dataset in zip(axes, datasets):
        model_map = grouped.get(dataset, {})
        model_names = [m for m in model_order if m in model_map] + [m for m in model_map if m not in model_order]
        if not model_names:
            ax.axis("off")
            continue

        bar_width = 0.22
        model_gap = 0.30
        group_centers: list[float] = []
        cursor = 0.0
        positions: dict[tuple[str, str], float] = {}
        for model_name in model_names:
            start = cursor
            for method_label in method_sequence:
                positions[(model_name, method_label)] = cursor
                cursor += bar_width
            group_centers.append(start + bar_width)
            cursor += model_gap

        for model_name in model_names:
            for method_label in method_sequence:
                row = model_map.get(model_name, {}).get(method_label)
                if row is None:
                    continue
                x = positions[(model_name, method_label)]
                value = _to_float(row.get("macro_f1"))
                if value is None:
                    continue
                low = _to_float(row.get("macro_f1_ci_low"))
                high = _to_float(row.get("macro_f1_ci_high"))
                lower_err = max(0.0, value - low) if low is not None else 0.0
                upper_err = max(0.0, high - value) if high is not None else 0.0
                face = _method_fill_color(
                    method_model=model_name,
                    method_base=str(row.get("method_base") or ""),
                    method_variant=str(row.get("method_variant") or ""),
                    fallback_index=0,
                    palette=[],
                )
                hatch = ""
                if method_label == "Per-trace Monitor":
                    hatch = "////"
                elif method_label == "Naive Agent":
                    hatch = ".."
                ax.bar(
                    x,
                    value,
                    width=bar_width * 0.88,
                    color=face,
                    edgecolor=SPINE_COLOR,
                    linewidth=0.6,
                    hatch=hatch,
                    yerr=[[lower_err], [upper_err]],
                    capsize=2.0,
                    ecolor="black",
                    zorder=3,
                )

        ax.set_title(TRACE_SCORE_DATASET_LABELS.get(dataset, dataset), fontsize=11.0, pad=8)
        ax.set_ylabel("Macro F1", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.0)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="major", axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _pr_curve_grid_by_case_size_plot(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    out_base: Path,
    formats: list[str],
    n_bootstrap: int = 100,
) -> list[Path]:
    rows = [row for row in rows if _include_in_paper_outputs(row)]
    if not rows:
        return []

    method_info = _method_info_by_label(rows)
    numeric_groups = sorted({int(r["group"]) for r in rows if str(r.get("group", "")).isdigit()})
    if not numeric_groups:
        return []

    model_order = [m for m in ("Qwen-3.5", "gpt-5.4-mini", "GLM-5") if any((_display_model_name(str(r.get("method_model") or "")) == m) for r in rows)]
    if not model_order:
        return []

    row_lookup = {
        (
            str(r.get("method_label") or ""),
            _display_model_name(str(r.get("method_model") or "")) or "",
            int(str(r.get("group") or "0")),
        ): r
        for r in rows
        if str(r.get("group", "")).isdigit()
    }

    method_labels = sorted({str(r.get("method_label") or "") for r in rows}, key=lambda m: _paper_method_rank(next(r for r in rows if str(r.get("method_label") or "") == m)))

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, axes = plt.subplots(
        len(model_order),
        len(numeric_groups),
        figsize=(5.5, 1.35 * max(1, len(model_order))),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if len(model_order) == 1 and len(numeric_groups) == 1:
        axes = [[axes]]
    elif len(model_order) == 1:
        axes = [list(axes)]
    elif len(numeric_groups) == 1:
        axes = [[ax] for ax in axes]

    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    for row_idx, model_name in enumerate(model_order):
        for col_idx, case_size in enumerate(numeric_groups):
            ax = axes[row_idx][col_idx]
            plotted = False
            for method_label in method_labels:
                row = row_lookup.get((method_label, model_name, case_size))
                if row is None:
                    continue
                file_path = Path(str(row.get("file") or "")).expanduser()
                if not file_path.is_file():
                    continue
                use_max_merge = _analysis_row_uses_max_merge(row)
                source_rows = [
                    source_row
                    for source_row in _load_rows(file_path)
                    if _to_int(source_row.get("traces_per_case")) == case_size
                ]
                case_pairs = [
                    _row_trace_score_pairs(source_row, max_merge=use_max_merge)
                    for source_row in source_rows
                ]
                case_pairs = [pairs for pairs in case_pairs if _average_precision_from_pairs(pairs) is not None]
                if not case_pairs:
                    continue
                recall_grid, mean_precision, lower, upper = _bootstrap_pooled_pr_curve_band(
                    case_pairs,
                    n_resamples=n_bootstrap,
                    seed=case_size * 100 + row_idx * 10 + col_idx,
                )
                method_model, method_base, method_variant = method_info.get(method_label, (None, None, None))
                color = _method_fill_color(method_model, method_base, method_variant, col_idx, palette)
                linestyle = _method_variant_linestyle(method_base, method_variant)
                ax.fill_between(recall_grid, lower, upper, color=color, alpha=0.12, linewidth=0.0, zorder=1)
                ax.plot(recall_grid, mean_precision, color=color, linestyle=linestyle, linewidth=1.8, zorder=2)
                plotted = True

            if not plotted:
                ax.axis("off")
                continue
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(SPINE_COLOR)
            ax.spines["bottom"].set_color(SPINE_COLOR)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            if row_idx == 0:
                ax.set_title(f"tpc={case_size}", fontsize=AXIS_LABEL_FONTSIZE, pad=4)
            if row_idx == len(model_order) - 1:
                ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nPrecision", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
            else:
                ax.tick_params(labelleft=False)

    legend_handles: list[Line2D] = []
    seen_styles: set[tuple[str, str, str]] = set()
    for method_label in method_labels:
        method_model, method_base, method_variant = method_info.get(method_label, (None, None, None))
        pretty_label = _presentation_method_label(
            method_base=method_base,
            method_variant=method_variant,
            dataset_variant=None,
        )
        style_key = (pretty_label, _method_variant_linestyle(method_base, method_variant), _method_variant_marker(method_base, method_variant))
        if style_key in seen_styles:
            continue
        seen_styles.add(style_key)
        color = _method_fill_color(method_model, method_base, method_variant, 0, palette)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.8,
                linestyle=_method_variant_linestyle(method_base, method_variant),
                label=pretty_label,
            )
        )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=max(1, min(4, len(legend_handles))),
            frameon=False,
            title="Method",
            title_fontsize=LEGEND_FONTSIZE,
            fontsize=LEGEND_FONTSIZE,
            borderaxespad=0.0,
            handlelength=1.6,
            columnspacing=0.8,
            handletextpad=0.5,
        )

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.78, wspace=0.18, hspace=0.28)

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _impossiblebench_combined_pr_curves_plot(
    *,
    rows: list[dict[str, Any]],
    out_base: Path,
    formats: list[str],
    n_bootstrap: int = 100,
) -> list[Path]:
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        file_path = Path(str(row.get("file") or "")).expanduser()
        dataset_name = str(row.get("dataset") or "").strip()
        dataset_variant = str(row.get("dataset_variant") or "").strip()
        method_base = str(row.get("method_base") or "").strip()
        method_model = str(row.get("method_model") or "").strip()
        method_variant = str(row.get("method_variant") or "").strip()
        method_label = str(row.get("method_label") or "").strip()
        if file_path:
            inferred_dataset, inferred_method = _infer_dataset_and_method_from_file(str(file_path))
            if not dataset_name:
                dataset_name = inferred_dataset
            inferred_dataset_base, inferred_dataset_variant = _split_dataset_variant(dataset_name or inferred_dataset)
            if not dataset_name:
                dataset_name = inferred_dataset_base
            if not dataset_variant:
                dataset_variant = str(inferred_dataset_variant or "")
            inferred_method, stripped_method_variant = _strip_method_variant(inferred_method)
            inferred_base, inferred_model = _split_method_parts(inferred_method)
            if not method_base:
                method_base = inferred_base
            if not method_model:
                method_model = inferred_model
            if not method_variant:
                method_variant = str(stripped_method_variant or dataset_variant or "")
            if not method_label:
                method_label = _presentation_method_label(
                    method_base=method_base,
                    dataset_variant=dataset_variant,
                    method_variant=method_variant,
                )
        dataset_name, split_dataset_variant = _split_dataset_variant(dataset_name)
        if not dataset_variant and split_dataset_variant:
            dataset_variant = split_dataset_variant
        if dataset_name != "impossiblebench_claude-opus-4.6":
            continue
        if not str(row.get("group") or "").isdigit():
            continue
        pretty_label = _presentation_method_label(
            method_base=method_base,
            dataset_variant=dataset_variant,
            method_variant=method_variant,
        )
        if pretty_label not in DM_REFERENCE_METHOD_ORDER:
            continue
        normalized_row = dict(row)
        normalized_row["dataset"] = dataset_name
        normalized_row["dataset_variant"] = dataset_variant
        normalized_row["method_base"] = method_base
        normalized_row["method_model"] = method_model
        normalized_row["method_variant"] = method_variant
        normalized_row["method_label"] = pretty_label
        plot_rows.append(normalized_row)
    if not plot_rows:
        return []

    allowed_models = ["gpt-5.4-mini", "Qwen-3.5"]
    allowed_case_sizes = [10, 25, 50, 100]
    filtered_rows: list[dict[str, Any]] = []
    for row in plot_rows:
        model_name = _display_model_name(str(row.get("method_model") or ""))
        case_size = _to_int(row.get("group"))
        if model_name not in allowed_models or case_size not in allowed_case_sizes:
            continue
        filtered_rows.append(row)
    if not filtered_rows:
        return []

    row_lookup = {
        (
            str(row.get("method_label") or ""),
            _display_model_name(str(row.get("method_model") or "")) or "",
            _to_int(row.get("group")),
        ): row
        for row in filtered_rows
    }
    method_labels = [label for label in DM_REFERENCE_METHOD_ORDER if any(str(row.get("method_label") or "") == label for row in filtered_rows)]
    if not method_labels:
        return []

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig = plt.figure(figsize=(5.55, 2.95))
    grid = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.22)
    axes = {
        ("gpt-5.4-mini", 10): fig.add_subplot(grid[0, 0]),
        ("gpt-5.4-mini", 25): fig.add_subplot(grid[0, 1]),
        ("gpt-5.4-mini", 50): fig.add_subplot(grid[0, 2]),
        ("gpt-5.4-mini", 100): fig.add_subplot(grid[0, 3]),
        ("Qwen-3.5", 10): fig.add_subplot(grid[1, 0]),
        ("Qwen-3.5", 25): fig.add_subplot(grid[1, 1]),
        ("Qwen-3.5", 50): fig.add_subplot(grid[1, 2]),
        ("Qwen-3.5", 100): fig.add_subplot(grid[1, 3]),
    }
    method_layer_order = (
        ["Naive Agent"]
        + [label for label in DM_REFERENCE_METHOD_ORDER if label not in {"Naive Agent", "Meerkat"}]
        + ["Meerkat"]
    )
    method_zorder = {label: idx for idx, label in enumerate(method_layer_order, start=1)}

    for row_idx, model_name in enumerate(allowed_models):
        for col_idx, case_size in enumerate(allowed_case_sizes):
            ax = axes[(model_name, case_size)]
            plotted = False
            for method_label in method_labels:
                row = row_lookup.get((method_label, model_name, case_size))
                if row is None:
                    continue
                file_path = Path(str(row.get("file") or "")).expanduser()
                if not file_path.is_file():
                    continue
                use_max_merge = _analysis_row_uses_max_merge(row)
                source_rows = [
                    source_row
                    for source_row in _load_rows(file_path)
                    if _to_int(source_row.get("traces_per_case")) == case_size
                ]
                case_pairs = [
                    _row_trace_score_pairs(source_row, max_merge=use_max_merge)
                    for source_row in source_rows
                ]
                case_pairs = [pairs for pairs in case_pairs if _average_precision_from_pairs(pairs) is not None]
                if not case_pairs:
                    continue
                recall_grid, mean_precision, lower, upper = _bootstrap_pooled_pr_curve_band(
                    case_pairs,
                    n_resamples=n_bootstrap,
                    seed=1000 * (row_idx + 1) + case_size + col_idx,
                )
                color = DM_REFERENCE_METHOD_COLORS.get(method_label, "#555555")
                linestyle = DM_REFERENCE_METHOD_LINESTYLES.get(method_label, "-")
                layer_zorder = method_zorder.get(method_label, 1)
                ax.fill_between(recall_grid, lower, upper, color=color, alpha=0.14, linewidth=0.0, zorder=layer_zorder)
                ax.plot(
                    recall_grid,
                    mean_precision,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    zorder=10 + layer_zorder,
                )
                plotted = True

            if not plotted:
                ax.axis("off")
                continue
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(SPINE_COLOR)
            ax.spines["bottom"].set_color(SPINE_COLOR)
            ax.set_title(f"tpc={case_size}", fontsize=AXIS_LABEL_FONTSIZE, pad=4)
            if row_idx == len(allowed_models) - 1:
                ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.3)
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nPrecision", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.2)
            else:
                ax.tick_params(labelleft=False)

    legend_handles: list[Line2D] = []
    for method_label in method_labels:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=DM_REFERENCE_METHOD_COLORS.get(method_label, "#555555"),
                linestyle=DM_REFERENCE_METHOD_LINESTYLES.get(method_label, "-"),
                linewidth=1.8,
                label=DM_REFERENCE_LEGEND_LABELS.get(method_label, method_label),
            )
        )

    legend_ax = axes.get(("gpt-5.4-mini", 100))
    if legend_handles and legend_ax and legend_ax.axison:
        legend_ax.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            handlelength=1.7,
            borderaxespad=0.15,
            labelspacing=0.18,
        )
    fig.subplots_adjust(top=0.92, left=0.10, right=0.985, bottom=0.16)

    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _judge_vs_at_ap_scatter_plot(
    *,
    rows: list[dict[str, Any]],
    dataset_labels: dict[str, str],
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    points_by_dataset = _judge_vs_at_ap_points(rows, dataset_labels)

    datasets = [dataset for dataset in sorted(dataset_labels) if points_by_dataset.get(dataset)]
    if not datasets:
        return []

    _apply_publication_style(plt)
    out_paths: list[Path] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    for dataset in datasets:
        dataset_points = points_by_dataset.get(dataset, [])
        model_order = [model for model in ALLOWED_EVAL_MODELS if any(str(point["model"]) == model for point in dataset_points)]
        if not model_order:
            model_order = sorted({str(point["model"]) for point in dataset_points})
        fig, axes = plt.subplots(
            1,
            len(model_order),
            figsize=(FIGURE_WIDTH_IN * max(1.0, 0.98 * len(model_order)), LINE_FIGURE_HEIGHT_IN * 0.98),
            constrained_layout=False,
        )
        if not isinstance(axes, (list, tuple)):
            try:
                axes = list(axes.ravel())
            except Exception:
                axes = [axes]
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.82, wspace=0.18)
        for ax, model in zip(axes, model_order):
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#999999", alpha=0.8, zorder=1)
            model_color = _model_color(model, 0, palette)
            for point in dataset_points:
                if str(point["model"]) != model:
                    continue
                ax.scatter(
                    [float(point["judge_ap"])],
                    [float(point["at_ap"])],
                    s=18,
                    marker="o",
                    facecolors=model_color,
                    edgecolors="#333333",
                    linewidths=0.45,
                    alpha=0.72,
                    zorder=3,
                )
            ax.set_title(model, fontsize=AXIS_LABEL_FONTSIZE, pad=4)
            ax.set_xlabel("Per-trace Monitor AP", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
            if ax is axes[0]:
                ax.set_ylabel("Meerkat AP", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.5)
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, which="major", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(SPINE_COLOR)
            ax.spines["bottom"].set_color(SPINE_COLOR)
        for ext in formats:
            out_path = out_base.with_suffix(f".{ext}")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
            out_paths.append(out_path)
        plt.close(fig)
    return out_paths


def _judge_vs_at_ap_points(
    rows: list[dict[str, Any]],
    dataset_labels: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    grouped = _group_plot_rows_by_dataset(rows)
    points_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, dataset_rows in grouped.items():
        if dataset not in dataset_labels:
            continue
        judge_file_by_model: dict[str, Path] = {}
        at_file_by_model: dict[str, Path] = {}
        at_use_max_merge_by_model: dict[str, bool] = {}
        for row in dataset_rows:
            display_model = _display_model_name(str(row.get("method_model") or ""))
            if not display_model:
                continue
            file_path = Path(str(row.get("file") or "")).expanduser()
            if not file_path.is_file():
                continue
            method_base = str(row.get("method_base") or "")
            method_variant = str(row.get("method_variant") or "").strip()
            if method_base == "llmjudge":
                judge_file_by_model.setdefault(display_model, file_path)
            elif method_base == "AT" and method_variant == "":
                at_file_by_model.setdefault(display_model, file_path)
                at_use_max_merge_by_model.setdefault(display_model, _analysis_row_uses_max_merge(row))
        for display_model, judge_file in judge_file_by_model.items():
            at_file = at_file_by_model.get(display_model)
            if at_file is None:
                continue
            if not judge_file.is_file() or not at_file.is_file():
                continue
            use_max_merge = at_use_max_merge_by_model.get(display_model, False)
            judge_cases = {
                str(case_row.get("case_id") or ""): case_row
                for case_row in _load_rows(judge_file)
                if case_row.get("case_id")
            }
            at_cases = {
                str(case_row.get("case_id") or ""): case_row
                for case_row in _load_rows(at_file)
                if case_row.get("case_id")
            }
            for case_id, judge_case in judge_cases.items():
                at_case = at_cases.get(case_id)
                if at_case is None:
                    continue
                if not _ground_truth_positive_trace_files(judge_case):
                    continue
                judge_ap = _average_precision_from_pairs(_row_trace_score_pairs(judge_case))
                at_ap = _average_precision_from_pairs(_row_trace_score_pairs(at_case, max_merge=use_max_merge))
                if judge_ap is None or at_ap is None:
                    continue
                points_by_dataset.setdefault(dataset, []).append(
                    {
                        "dataset": dataset,
                        "model": display_model,
                        "case_id": case_id,
                        "judge_ap": float(judge_ap),
                        "at_ap": float(at_ap),
                    }
                )
    return points_by_dataset


def _build_calibration_groups(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, list[dict[str, float]]]],
    dict[str, dict[str, tuple[float | None, float | None]]],
    dict[str, dict[str, list[dict[str, float]]]],
    dict[str, dict[str, tuple[float | None, float | None]]],
    dict[str, dict[str, tuple[str | None, str | None, str | None]]],
]:
    raw_case_pairs: dict[str, dict[str, list[tuple[int, float]]]] = {}
    raw_trace_pairs: dict[str, dict[str, list[tuple[int, float]]]] = {}
    method_info: dict[str, dict[str, tuple[str | None, str | None, str | None]]] = {}

    for row in rows:
        dataset = str(row.get("dataset") or "")
        if dataset not in TRACE_SCORE_DATASET_LABELS:
            continue
        file_path = Path(str(row.get("file") or "")).expanduser()
        if not file_path.is_file():
            continue
        source_rows = _load_rows(file_path)
        trace_pairs: list[tuple[int, float]] = []
        case_pairs: list[tuple[int, float]] = []
        use_max_merge = _analysis_row_uses_max_merge(row)
        for source_row in source_rows:
            trace_pairs.extend(_row_trace_score_pairs(source_row, max_merge=use_max_merge))
            case_pair = _row_case_score_pair(source_row, max_merge=use_max_merge)
            if case_pair is not None:
                case_pairs.append(case_pair)
        if not trace_pairs and not case_pairs:
            continue
        method_label = str(row.get("method_label") or "")
        if trace_pairs:
            raw_trace_pairs.setdefault(dataset, {})[method_label] = trace_pairs
        if case_pairs:
            raw_case_pairs.setdefault(dataset, {})[method_label] = case_pairs
        method_info.setdefault(dataset, {})[method_label] = (
            str(row.get("method_model") or "").strip() or None,
            str(row.get("method_base") or "").strip() or None,
            str(row.get("method_variant") or "").strip() or None,
        )

    case_curves: dict[str, dict[str, list[dict[str, float]]]] = {}
    case_metrics: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    trace_curves: dict[str, dict[str, list[dict[str, float]]]] = {}
    trace_metrics: dict[str, dict[str, tuple[float | None, float | None]]] = {}

    datasets = sorted(set(raw_case_pairs) | set(raw_trace_pairs))
    for dataset in datasets:
        for method_label, pairs in (raw_case_pairs.get(dataset) or {}).items():
            curve, ece, brier = _calibration_from_pairs(pairs)
            case_curves.setdefault(dataset, {})[method_label] = curve
            case_metrics.setdefault(dataset, {})[method_label] = (ece, brier)
        for method_label, pairs in (raw_trace_pairs.get(dataset) or {}).items():
            curve, ece, brier = _calibration_from_pairs(pairs)
            trace_curves.setdefault(dataset, {})[method_label] = curve
            trace_metrics.setdefault(dataset, {})[method_label] = (ece, brier)

    return case_curves, case_metrics, trace_curves, trace_metrics, method_info


def _print_calibration_table(overall_rows: list[dict[str, Any]]) -> None:
    grouped_rows = _group_plot_rows_by_dataset(overall_rows)
    flat_rows = [row for dataset_rows in grouped_rows.values() for row in dataset_rows]
    case_curves, case_metrics, trace_curves, trace_metrics, _ = _build_calibration_groups(flat_rows)

    table_rows: list[tuple[str, str, float | None, float | None, float | None, float | None]] = []
    for dataset in sorted(set(case_metrics) | set(trace_metrics)):
        methods = sorted(set(case_metrics.get(dataset, {})) | set(trace_metrics.get(dataset, {})))
        for method in methods:
            case_ece, case_brier = (case_metrics.get(dataset, {}) or {}).get(method, (None, None))
            trace_ece, trace_brier = (trace_metrics.get(dataset, {}) or {}).get(method, (None, None))
            table_rows.append((dataset, method, case_ece, case_brier, trace_ece, trace_brier))

    if not table_rows:
        print("\n## Calibration\n(no rows)")
        return

    print("\n## Calibration")
    print("dataset | method | case_ece | case_brier | trace_ece | trace_brier")
    print("--- | --- | --- | --- | --- | ---")
    for dataset, method, case_ece, case_brier, trace_ece, trace_brier in table_rows:
        print(
            f"{dataset} | {method} | {_fmt(case_ece, 4)} | {_fmt(case_brier, 4)} | {_fmt(trace_ece, 4)} | {_fmt(trace_brier, 4)}"
        )


def _discover_safety_result_files() -> list[Path]:
    discovered: list[Path] = []
    for path in sorted(Path("results").glob("safety_*.jsonl")):
        if not path.is_file():
            continue
        dataset, method = _infer_dataset_and_method_from_file(str(path))
        dataset_base, _ = _split_dataset_variant(dataset)
        if dataset_base not in {"trace-dataset", "impossiblebench_claude-opus-4.6"}:
            continue
        method, _ = _strip_method_variant(method)
        _, method_model = _split_method_parts(method)
        if not _should_include_eval_model(method_model):
            continue
        discovered.append(path)
    return discovered


def _generate_figures(
    overall_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    pct_rows: list[dict[str, Any]],
    figures_dir: Path,
    formats: list[str],
) -> list[Path]:
    del pct_rows
    out_paths: list[Path] = []

    trace_ap_rows = [
        row
        for dataset_rows in _group_plot_rows_by_dataset(overall_rows).values()
        for row in dataset_rows
        if str(row.get("dataset") or "") in TRACE_SCORE_DATASET_LABELS and _include_in_paper_outputs(row)
    ]
    datasets_for_paper = sorted(
        {
            str(row.get("dataset") or "")
            for row in trace_ap_rows
            if str(row.get("dataset") or "") in TRACE_SCORE_DATASET_LABELS
        }
    )
    for dataset in datasets_for_paper:
        dataset_rows = [row for row in trace_ap_rows if str(row.get("dataset") or "") == dataset]
        if not dataset_rows:
            continue
        dataset_slug = _dataset_output_slug(dataset)
        out_paths.extend(
            _judge_vs_at_ap_scatter_plot(
                rows=dataset_rows,
                dataset_labels={dataset: TRACE_SCORE_DATASET_LABELS.get(dataset, dataset)},
                out_base=figures_dir / f"safety_{dataset_slug}_judge_vs_meerkat_trace_ap",
                formats=formats,
            )
        )
        out_paths.extend(
            _overall_ap_by_model_plot(
                rows=dataset_rows,
                dataset=dataset,
                out_base=figures_dir / f"safety_{dataset_slug}_overall_ap_by_model",
                formats=formats,
            )
        )

    by_dataset_case = _group_plot_rows_by_dataset(case_rows)
    for dataset, rows in sorted(by_dataset_case.items(), key=lambda kv: kv[0]):
        if dataset not in TRACE_SCORE_DATASET_LABELS:
            continue
        rows = [row for row in rows if _include_in_paper_outputs(row)]
        method_info = _method_info_by_label(rows)
        numeric_groups = sorted(
            {int(r["group"]) for r in rows if str(r.get("group", "")).isdigit()}
        )
        if not numeric_groups:
            continue
        x_vals = numeric_groups
        x_labels = [str(x) for x in x_vals]

        methods = sorted({str(r.get("method_label") or "") for r in rows})
        y_case_ap: dict[str, list[float | None]] = {m: [] for m in methods}
        y_case_ap_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        y_trace_ap: dict[str, list[float | None]] = {m: [] for m in methods}
        y_trace_ap_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        row_lookup = {
            (str(r.get("method_label") or ""), int(str(r.get("group") or "0"))): r
            for r in rows
            if str(r.get("group", "")).isdigit()
        }
        for m in methods:
            for x in x_vals:
                row = row_lookup.get((m, x))
                if row is None:
                    y_case_ap[m].append(None)
                    y_case_ap_ci[m].append(None)
                    y_trace_ap[m].append(None)
                    y_trace_ap_ci[m].append(None)
                    continue
                case_ap = _to_float(row.get("case_average_precision"))
                case_ap_low = _to_float(row.get("case_average_precision_ci_low"))
                case_ap_high = _to_float(row.get("case_average_precision_ci_high"))
                y_case_ap[m].append(case_ap)
                if case_ap is None or case_ap_low is None or case_ap_high is None:
                    y_case_ap_ci[m].append(None)
                else:
                    y_case_ap_ci[m].append((max(0.0, case_ap - case_ap_low), max(0.0, case_ap_high - case_ap)))

                trace_ap = _to_float(row.get("trace_average_precision"))
                trace_ap_low = _to_float(row.get("trace_average_precision_ci_low"))
                trace_ap_high = _to_float(row.get("trace_average_precision_ci_high"))
                y_trace_ap[m].append(trace_ap)
                if trace_ap is None or trace_ap_low is None or trace_ap_high is None:
                    y_trace_ap_ci[m].append(None)
                else:
                    y_trace_ap_ci[m].append((max(0.0, trace_ap - trace_ap_low), max(0.0, trace_ap_high - trace_ap)))

        dataset_slug = _dataset_output_slug(dataset)
        out_paths.extend(
            _ap_by_case_size_plot(
                dataset=dataset,
                x_values=x_vals,
                x_labels=x_labels,
                case_ap_by_method=y_case_ap,
                case_ap_err_by_method=y_case_ap_ci,
                trace_ap_by_method=y_trace_ap,
                trace_ap_err_by_method=y_trace_ap_ci,
                method_info_by_label=method_info,
                out_base=figures_dir / f"safety_{dataset_slug}_ap_by_case_size",
                formats=formats,
            )
        )

    out_paths.extend(
        _impossiblebench_combined_pr_curves_plot(
            rows=case_rows,
            out_base=figures_dir / "safety_impossiblebench_combined_pr_curves",
            formats=formats,
        )
    )

    out_paths.extend(
        _macro_f1_by_dataset_plot(
            rows=overall_rows,
            out_base=figures_dir / "safety_macro_f1_by_dataset",
            formats=formats,
        )
    )

    return out_paths

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        help="One or more safety JSONL result files. Defaults to mle-sabotage judge+AT files if omitted.",
    )
    parser.add_argument("--output-overall-csv", type=Path, default=None)
    parser.add_argument("--output-case-size-csv", type=Path, default=None)
    parser.add_argument("--output-positive-pct-csv", type=Path, default=None)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to store generated publication-quality matplotlib figures.",
    )
    parser.add_argument(
        "--figure-formats",
        type=str,
        default="png,pdf",
        help="Comma-separated figure formats (e.g., png,pdf).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Disable matplotlib figure generation.",
    )
    parser.add_argument(
        "--positive-pct-bins",
        type=str,
        default="25,50,75,100",
        help=(
            "Comma-separated positive-percent bucket upper bounds for non-zero cases "
            "(0%% is always a separate bucket). Example: 25,50,75,100"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    figure_formats = [f.strip().lower() for f in str(args.figure_formats or "").split(",") if f.strip()]
    if not figure_formats:
        figure_formats = ["png", "pdf"]
    positive_pct_bounds = _parse_positive_pct_bins(args.positive_pct_bins)

    files = args.results
    if not files:
        discovered = _discover_safety_result_files()
        files = discovered or [
            Path("results/safety_mle-sabotage_llmjudge.jsonl"),
            Path("results/safety_mle-sabotage_AT-gpt-5-mini.jsonl"),
        ]
        print(f"Auto-discovered {len(files)} file(s):")
        for file in files:
            print(f"- {file}")

    all_overall: list[dict[str, Any]] = []
    all_by_case: list[dict[str, Any]] = []
    all_by_pct: list[dict[str, Any]] = []
    loaded_files: set[Path] = set()

    for file in files:
        if not file.exists():
            print(f"Skipping missing file: {file}")
            continue
        loaded_files.add(file.resolve())
        overall, by_case, by_pct = _collect_metrics(
            file,
            positive_pct_bounds=positive_pct_bounds,
        )
        all_overall.extend(overall)
        all_by_case.extend(by_case)
        all_by_pct.extend(by_pct)

    _print_paper_table(all_overall)
    table_paths = _write_safety_metric_tables(all_by_case, args.figures_dir)
    if table_paths:
        print("\nGenerated tables:")
        for p in table_paths:
            print(f"- {p}")

    if not args.no_figures:
        figure_paths = _generate_figures(
            overall_rows=all_overall,
            case_rows=all_by_case,
            pct_rows=all_by_pct,
            figures_dir=args.figures_dir,
            formats=figure_formats,
        )
        if figure_paths:
            print("\nGenerated figures:")
            for p in figure_paths:
                print(f"- {p}")

    if args.output_overall_csv:
        _write_csv(args.output_overall_csv, all_overall)
        print(f"\nWrote: {args.output_overall_csv}")
    if args.output_case_size_csv:
        _write_csv(args.output_case_size_csv, all_by_case)
        print(f"Wrote: {args.output_case_size_csv}")
    if args.output_positive_pct_csv:
        _write_csv(args.output_positive_pct_csv, all_by_pct)
        print(f"Wrote: {args.output_positive_pct_csv}")


if __name__ == "__main__":
    main()
