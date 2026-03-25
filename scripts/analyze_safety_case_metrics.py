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
    "AT": "#0072B2",
    "AT (Codex)": "#009E73",
    "AT (No Tools)": "#CC79A7",
    "Judge": "#D55E00",
}
METHOD_MARKERS = {
    "AT": "o",
    "AT (Codex)": "D",
    "AT (No Tools)": "^",
    "Judge": "s",
}
MODEL_COLORS = {
    "gpt-5-mini": "#0072B2",
    "gpt-5.4-mini": "#56B4E9",
    "gpt-5.4": "#009E73",
    "Qwen-3.5": "#CC79A7",
    "MiniMax-M2.5": "#D55E00",
}
METHOD_HATCHES = {
    "AT": "",
    "AT-codex": "",
    "llmjudge": "////",
}
# ALLOWED_EVAL_MODELS = {"gpt-5-mini", "gpt-5.4-mini", "gpt-5.4", "Qwen-3.5", "MiniMax-M2.5"}
ALLOWED_EVAL_MODELS = {"Qwen-3.5"}
OVERALL_DATASET_LABELS = {
    # "impossiblebench_gpt-5-mini": "ImpossibleBench",
    "mle-sabotage": "MLE-Sabotage",
    "trace-dataset": "TRACE",
}
TRACE_SCORE_DATASET_LABELS = {
    "mle-sabotage": "MLE-Sabotage",
    "trace-dataset": "TRACE",
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


def _average_precision_from_pairs(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    if positives <= 0:
        return None
    ranked = sorted(
        ((int(label), float(score)) for label, score in pairs),
        key=lambda item: (-item[1], -item[0]),
    )
    tp = 0
    ap_accum = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label != 1:
            continue
        tp += 1
        ap_accum += tp / rank
    return ap_accum / positives


def _average_precision_flat_pairs(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    positives = sum(1 for label, _ in pairs if int(label) == 1)
    if positives <= 0:
        return None
    ranked = sorted(
        ((int(label), float(score)) for label, score in pairs),
        key=lambda item: (-item[1], -item[0]),
    )
    tp = 0
    ap_accum = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label != 1:
            continue
        tp += 1
        ap_accum += tp / rank
    return ap_accum / positives


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
    if method.endswith("_no-tools"):
        method_variant = "no-tools"
        method = method[: -len("_no-tools")]
    return method, method_variant


def _split_method_parts(method: str) -> tuple[str, str | None]:
    method = str(method or "").strip()
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
    if text == "MiniMaxAI-MiniMax-M2.5":
        return "MiniMax-M2.5"
    lowered = text.lower()
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
            return f"Judge ({model})"
        return "Judge"

    if family == "AT":
        method_variant_parts = []
        if dataset_variant == "no-tools" or method_variant == "no-tools":
            method_variant_parts.append("No Tools")

        if method.startswith("AT-codex-") or method.startswith("AT-codex"):
            family_name = "AT (Codex)"
            model = model[len("codex-") :] if model and model.startswith("codex-") else model
        else:
            family_name = "AT"

        if method_variant_parts:
            variant_text = ", ".join(method_variant_parts)
            if model:
                return f"{family_name} ({variant_text}, {model})"
            return f"{family_name} ({variant_text})"
        if model:
            return f"{family_name} ({model})"
        return family_name

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
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _row_trace_score_pairs(row: dict[str, Any]) -> list[tuple[int, float]]:
    tests = row.get("tests") or []
    metadata = (tests[0] or {}).get("metadata") if tests else {}
    if not isinstance(metadata, dict):
        metadata = {}
    raw_scores = metadata.get("trace_scores") or (row.get("scoring") or {}).get("trace_scores") or {}
    if not isinstance(raw_scores, dict):
        return []
    gt_positive = {str(x).strip() for x in _ground_truth_positive_trace_files(row) if str(x).strip()}
    pairs: list[tuple[int, float]] = []
    seen: set[str] = set()
    for trace_file, raw_score in raw_scores.items():
        trace_key = str(trace_file).strip()
        if not trace_key or trace_key in seen:
            continue
        try:
            score = float(raw_score)
        except Exception:
            continue
        seen.add(trace_key)
        pairs.append((1 if trace_key in gt_positive else 0, max(0.0, min(1.0, score))))
    return pairs


def _row_case_score_pair(row: dict[str, Any]) -> tuple[int, float] | None:
    scoring = row.get("scoring") or {}
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


def _init_metric_row(label: str, file_path: Path, group_key: str) -> dict[str, Any]:
    return {
        "method": label,
        "file": str(file_path),
        "group": group_key,
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
    trace_score_pairs = _row_trace_score_pairs(row)
    if trace_score_pairs:
        metric["_trace_score_cases"].append(trace_score_pairs)
    case_score_pair = _row_case_score_pair(row)
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
    out["trace_average_precision"] = trace_average_precision
    out["trace_average_precision_ci_low"] = trace_ap_low
    out["trace_average_precision_ci_high"] = trace_ap_high
    trace_roc_auc, trace_roc_low, trace_roc_high = _bootstrap_roc_auc_ci(trace_score_cases)
    out["trace_roc_auc"] = trace_roc_auc
    out["trace_roc_auc_ci_low"] = trace_roc_low
    out["trace_roc_auc_ci_high"] = trace_roc_high
    case_score_pairs = [tuple(pair) for pair in metric.get("_case_score_pairs") or []]
    case_average_precision, case_ap_low, case_ap_high = _bootstrap_flat_average_precision_ci(case_score_pairs)
    out["case_average_precision"] = case_average_precision
    out["case_average_precision_ci_low"] = case_ap_low
    out["case_average_precision_ci_high"] = case_ap_high
    case_roc_auc, case_roc_low, case_roc_high = _bootstrap_flat_roc_auc_ci(case_score_pairs)
    out["case_roc_auc"] = case_roc_auc
    out["case_roc_auc_ci_low"] = case_roc_low
    out["case_roc_auc_ci_high"] = case_roc_high

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

    _, method = _infer_dataset_and_method_from_file(str(path))
    _, method_model = _split_method_parts(method)
    eval_model = _display_model_name(method_model or fallback_model)
    if not _should_include_eval_model(eval_model):
        return [], [], []

    overall = _init_metric_row(label, path, "overall")
    by_case_size: dict[str, dict[str, Any]] = {}
    by_positive_pct: dict[str, dict[str, Any]] = {}

    for row in rows:
        _update_metric_row(overall, row, fallback_model=fallback_model)

        case_size = _to_int(row.get("traces_per_case"))
        case_key = str(case_size) if case_size > 0 else "unknown"
        if case_key not in by_case_size:
            by_case_size[case_key] = _init_metric_row(label, path, case_key)
        _update_metric_row(by_case_size[case_key], row, fallback_model=fallback_model)

        pos_count = _gt_positive_count(row)
        if case_size > 0:
            pct = 100.0 * (pos_count / case_size)
            pct_key = _positive_pct_bucket_label(pct, positive_pct_bounds)
        else:
            pct_key = "unknown"
        if pct_key not in by_positive_pct:
            by_positive_pct[pct_key] = _init_metric_row(label, path, pct_key)
        _update_metric_row(by_positive_pct[pct_key], row, fallback_model=fallback_model)

    overall_rows = [_finalize_metric_row(overall)]
    case_rows = [_finalize_metric_row(v) for _, v in sorted(by_case_size.items(), key=lambda kv: kv[0])]
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


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text).strip().lower())
    return value.strip("-") or "unknown"


def _infer_dataset_and_method_from_file(file_path: str) -> tuple[str, str]:
    stem = Path(file_path).stem
    name = stem
    if name.startswith("safety_"):
        name = name[len("safety_") :]

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
    if dataset.endswith("_no-tools"):
        return dataset[: -len("_no-tools")], "no-tools"
    return dataset, None


def _apply_publication_style(plt) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
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


def _method_hatch(method_base: str | None) -> str:
    return METHOD_HATCHES.get(str(method_base or "").strip(), "")


def _method_variant_marker(method_base: str | None, method_variant: str | None) -> str:
    base = str(method_base or "").strip()
    variant = str(method_variant or "").strip()
    if base == "llmjudge":
        return "s"
    if variant == "no-tools":
        return "^"
    if base == "AT-codex":
        return "D"
    if base == "AT":
        return "o"
    return _method_marker(base or variant or "AT")


def _method_variant_linestyle(method_base: str | None, method_variant: str | None) -> str:
    base = str(method_base or "").strip()
    if base == "llmjudge":
        return "--"
    return "-"


def _method_fill_color(
    method_model: str | None,
    method_variant: str | None,
    fallback_index: int,
    palette: list[str],
) -> str:
    color = _model_color(method_model, fallback_index, palette)
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
        dataset_base, dataset_variant = _split_dataset_variant(dataset)
        method, method_variant = _strip_method_variant(method)
        method_base, method_model = _split_method_parts(method)
        out = dict(row)
        out["dataset"] = dataset_base
        out["dataset_variant"] = dataset_variant
        out["method_key"] = method
        out["method_model"] = method_model
        out["method_variant"] = method_variant or dataset_variant
        out["method_label"] = _pretty_method(
            method,
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
        color = _method_fill_color(method_model, method_variant, i, palette)
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
        method_label = "Judge" if method_base == "llmjudge" else "AT"
        if method_variant == "no-tools":
            method_label = "AT (No Tools)"
        elif method_base == "AT-codex":
            method_label = "AT (Codex)"
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
        fig.savefig(out_path, dpi=300)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def _auprc_by_case_size_plot(
    *,
    dataset: str,
    x_values: list[Any],
    x_labels: list[str],
    case_ap_by_method: dict[str, list[float | None]],
    case_ap_err_by_method: dict[str, list[tuple[float, float] | None]],
    trace_ap_by_method: dict[str, list[float | None]],
    trace_ap_err_by_method: dict[str, list[tuple[float, float] | None]],
    case_roc_by_method: dict[str, list[float | None]],
    case_roc_err_by_method: dict[str, list[tuple[float, float] | None]],
    trace_roc_by_method: dict[str, list[float | None]],
    trace_roc_err_by_method: dict[str, list[tuple[float, float] | None]],
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
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN * 2.0, LINE_FIGURE_HEIGHT_IN * 1.8), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.72, wspace=0.28, hspace=0.32)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    panels = [
        (axes[0][0], "Case-Level AUPRC", case_ap_by_method, case_ap_err_by_method),
        (axes[0][1], "Trace-Level AUPRC", trace_ap_by_method, trace_ap_err_by_method),
        (axes[1][0], "Case-Level AUROC", case_roc_by_method, case_roc_err_by_method),
        (axes[1][1], "Trace-Level AUROC", trace_roc_by_method, trace_roc_err_by_method),
    ]
    for ax, ylabel, y_by_method, yerr_by_method in panels:
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
            color = _method_fill_color(method_model, method_variant, i, palette)
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
        method_label = "Judge" if method_base == "llmjudge" else "AT"
        if method_variant == "no-tools":
            method_label = "AT (No Tools)"
        elif method_base == "AT-codex":
            method_label = "AT (Codex)"
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
            bbox_to_anchor=(0.47, 0.98),
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
        fig.savefig(out_path, dpi=300)
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
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.70, hspace=0.32)
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    for i, method in enumerate(sorted(set(curves_by_method) | set(roc_curves_by_method), key=lambda x: x)):
        method_model, method_base, method_variant = (method_info_by_label or {}).get(method, (None, None, None))
        color = _method_fill_color(method_model, method_variant, i, palette)
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
        colors.append(_method_fill_color(method_model, method_variant, i, palette))
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

    preferred_model_order = ("gpt-5-mini", "gpt-5.4-mini", "gpt-5.4", "Qwen-3.5", "MiniMax-M2.5")
    model_order = [
        model_name
        for model_name in preferred_model_order
        if any((info[0] and _display_model_name(info[0]) == model_name) for info in method_style.values())
    ]
    seen_method_order: list[str] = []
    for method in ("AT", "AT (No Tools)", "Judge"):
        if any(
            (
                (info[1] == "AT" and method == "AT" and info[2] != "no-tools")
                or (info[1] == "AT" and method == "AT (No Tools)" and info[2] == "no-tools")
                or (info[1] == "llmjudge" and method == "Judge")
            )
            for info in method_style.values()
        ):
            seen_method_order.append(method)

    def _overall_method_rank(method_base: str | None, method_variant: str | None) -> int:
        if method_base == "AT" and method_variant != "no-tools":
            return 0
        if method_base == "AT" and method_variant == "no-tools":
            return 1
        if method_base == "llmjudge":
            return 2
        if method_base == "AT-codex":
            return 3
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
        bar_color = _method_fill_color(method_model, method_variant, method_idx, palette)
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
    if "AT (No Tools)" in seen_method_order:
        method_handles.append(
            Patch(facecolor="#E0E0E0", edgecolor=SPINE_COLOR, linewidth=0.6, label="AT (No Tools)")
        )
    if "Judge" in seen_method_order:
        method_handles.append(
            Patch(facecolor="#BEBEBE", edgecolor=SPINE_COLOR, linewidth=0.6, hatch="///", label="Judge")
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


def _discover_safety_result_files() -> list[Path]:
    return sorted(
        path
        for path in Path("results").glob("safety_*.jsonl")
        if path.is_file()
    )


def _generate_figures(
    *,
    overall_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    pct_rows: list[dict[str, Any]],
    figures_dir: Path,
    formats: list[str],
) -> list[Path]:
    out_paths: list[Path] = []
    out_paths.extend(
        _grouped_overall_metric_bar_plot(
            rows=overall_rows,
            metric_key="verified_macro_f1",
            metric_low_key="verified_macro_f1_ci_low",
            metric_high_key="verified_macro_f1_ci_high",
            dataset_labels=OVERALL_DATASET_LABELS,
            ylabel="Verified Macro F1",
            out_base=figures_dir / "safety_overall_verified_macro_f1_by_dataset",
            formats=formats,
        )
    )
    out_paths.extend(
        _grouped_overall_metric_bar_plot(
            rows=overall_rows,
            metric_key="witness_precision",
            metric_low_key=None,
            metric_high_key=None,
            dataset_labels=OVERALL_DATASET_LABELS,
            ylabel="Witness Precision",
            out_base=figures_dir / "safety_overall_witness_precision_by_dataset",
            formats=formats,
        )
    )
    trace_ap_rows = [
        row
        for dataset_rows in _group_plot_rows_by_dataset(overall_rows).values()
        for row in dataset_rows
        if str(row.get("dataset") or "") in TRACE_SCORE_DATASET_LABELS
    ]
    out_paths.extend(
        _grouped_overall_metric_bar_plot(
            rows=trace_ap_rows,
            metric_key="trace_average_precision",
            metric_low_key="trace_average_precision_ci_low",
            metric_high_key="trace_average_precision_ci_high",
            dataset_labels=TRACE_SCORE_DATASET_LABELS,
            ylabel="Trace-level AP",
            out_base=figures_dir / "safety_overall_trace_average_precision_by_dataset",
            formats=formats,
        )
    )
    by_dataset_case = _group_plot_rows_by_dataset(case_rows)
    by_dataset_pct = _group_plot_rows_by_dataset(pct_rows)

    pr_curve_groups: dict[str, dict[str, list[dict[str, float]]]] = {}
    roc_curve_groups: dict[str, dict[str, list[dict[str, float]]]] = {}
    case_pr_curve_groups: dict[str, dict[str, list[dict[str, float]]]] = {}
    case_roc_curve_groups: dict[str, dict[str, list[dict[str, float]]]] = {}
    pr_method_info: dict[str, dict[str, tuple[str | None, str | None, str | None]]] = {}
    for row in trace_ap_rows:
        dataset = str(row.get("dataset") or "")
        if dataset not in TRACE_SCORE_DATASET_LABELS:
            continue
        file_path = Path(str(row.get("file") or "")).expanduser()
        if not file_path.is_file():
            continue
        source_rows = _load_rows(file_path)
        case_pairs: list[list[tuple[int, float]]] = []
        case_score_pairs: list[tuple[int, float]] = []
        for source_row in source_rows:
            row_pairs = _row_trace_score_pairs(source_row)
            if row_pairs:
                case_pairs.append(row_pairs)
            case_pair = _row_case_score_pair(source_row)
            if case_pair is not None:
                case_score_pairs.append(case_pair)
        if not case_pairs:
            continue
        pr_curve_groups.setdefault(dataset, {})[str(row.get("method_label") or "")] = _precision_recall_curve_from_cases(case_pairs)
        roc_curve_groups.setdefault(dataset, {})[str(row.get("method_label") or "")] = _roc_curve_from_cases(case_pairs)
        case_pr_curve_groups.setdefault(dataset, {})[str(row.get("method_label") or "")] = _precision_recall_curve_from_flat_pairs(case_score_pairs)
        case_roc_curve_groups.setdefault(dataset, {})[str(row.get("method_label") or "")] = _roc_curve_from_flat_pairs(case_score_pairs)
        pr_method_info.setdefault(dataset, {})[str(row.get("method_label") or "")] = (
            str(row.get("method_model") or "").strip() or None,
            str(row.get("method_base") or "").strip() or None,
            str(row.get("method_variant") or "").strip() or None,
        )

    for dataset, curves_by_method in sorted(pr_curve_groups.items(), key=lambda kv: kv[0]):
        dataset_slug = _slug(dataset)
        out_paths.extend(
            _precision_recall_plot(
                curves_by_method=curves_by_method,
                roc_curves_by_method=roc_curve_groups.get(dataset) or {},
                method_info_by_label=pr_method_info.get(dataset) or {},
                out_base=figures_dir / f"safety_{dataset_slug}_trace_precision_recall_curve",
                formats=formats,
            )
        )
        out_paths.extend(
            _precision_recall_plot(
                curves_by_method=case_pr_curve_groups.get(dataset) or {},
                roc_curves_by_method=case_roc_curve_groups.get(dataset) or {},
                method_info_by_label=pr_method_info.get(dataset) or {},
                out_base=figures_dir / f"safety_{dataset_slug}_case_precision_recall_curve",
                formats=formats,
            )
        )

    for dataset, rows in sorted(by_dataset_case.items(), key=lambda kv: kv[0]):
        method_info = _method_info_by_label(rows)
        numeric_groups = sorted(
            {int(r["group"]) for r in rows if str(r.get("group", "")).isdigit()}
        )
        if not numeric_groups:
            continue
        x_vals = numeric_groups
        x_labels = [str(x) for x in x_vals]

        methods = sorted({str(r.get("method_label") or "") for r in rows})
        y_verified_macro_f1: dict[str, list[float | None]] = {m: [] for m in methods}
        y_verified_macro_f1_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        y_cost: dict[str, list[float | None]] = {m: [] for m in methods}
        y_cost_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        y_case_ap: dict[str, list[float | None]] = {m: [] for m in methods}
        y_case_ap_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        y_trace_ap: dict[str, list[float | None]] = {m: [] for m in methods}
        y_trace_ap_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        y_case_roc: dict[str, list[float | None]] = {m: [] for m in methods}
        y_case_roc_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        y_trace_roc: dict[str, list[float | None]] = {m: [] for m in methods}
        y_trace_roc_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        row_lookup = {
            (str(r.get("method_label") or ""), int(str(r.get("group") or "0"))): r for r in rows if str(r.get("group", "")).isdigit()
        }
        for m in methods:
            for x in x_vals:
                row = row_lookup.get((m, x))
                if row is None:
                    y_verified_macro_f1[m].append(None)
                    y_verified_macro_f1_ci[m].append(None)
                    y_cost[m].append(None)
                    y_cost_ci[m].append(None)
                    y_case_ap[m].append(None)
                    y_case_ap_ci[m].append(None)
                    y_trace_ap[m].append(None)
                    y_trace_ap_ci[m].append(None)
                    y_case_roc[m].append(None)
                    y_case_roc_ci[m].append(None)
                    y_trace_roc[m].append(None)
                    y_trace_roc_ci[m].append(None)
                    continue
                macro_f1 = _to_float(row.get("verified_macro_f1"))
                macro_f1_low = _to_float(row.get("verified_macro_f1_ci_low"))
                macro_f1_high = _to_float(row.get("verified_macro_f1_ci_high"))
                y_verified_macro_f1[m].append(macro_f1)
                if macro_f1 is None or macro_f1_low is None or macro_f1_high is None:
                    y_verified_macro_f1_ci[m].append(None)
                else:
                    y_verified_macro_f1_ci[m].append(
                        (max(0.0, macro_f1 - macro_f1_low), max(0.0, macro_f1_high - macro_f1))
                    )
                cost = _to_float(row.get("avg_cost_usd_per_case"))
                cost_low = _to_float(row.get("avg_cost_usd_per_case_ci_low"))
                cost_high = _to_float(row.get("avg_cost_usd_per_case_ci_high"))
                y_cost[m].append(cost)
                if cost is None or cost_low is None or cost_high is None:
                    y_cost_ci[m].append(None)
                else:
                    y_cost_ci[m].append((max(0.0, cost - cost_low), max(0.0, cost_high - cost)))
                case_ap = _to_float(row.get("case_average_precision"))
                case_ap_low = _to_float(row.get("case_average_precision_ci_low"))
                case_ap_high = _to_float(row.get("case_average_precision_ci_high"))
                y_case_ap[m].append(case_ap)
                if case_ap is None or case_ap_low is None or case_ap_high is None:
                    y_case_ap_ci[m].append(None)
                else:
                    y_case_ap_ci[m].append(
                        (max(0.0, case_ap - case_ap_low), max(0.0, case_ap_high - case_ap))
                    )
                trace_ap = _to_float(row.get("trace_average_precision"))
                trace_ap_low = _to_float(row.get("trace_average_precision_ci_low"))
                trace_ap_high = _to_float(row.get("trace_average_precision_ci_high"))
                y_trace_ap[m].append(trace_ap)
                if trace_ap is None or trace_ap_low is None or trace_ap_high is None:
                    y_trace_ap_ci[m].append(None)
                else:
                    y_trace_ap_ci[m].append(
                        (max(0.0, trace_ap - trace_ap_low), max(0.0, trace_ap_high - trace_ap))
                    )
                case_roc = _to_float(row.get("case_roc_auc"))
                case_roc_low = _to_float(row.get("case_roc_auc_ci_low"))
                case_roc_high = _to_float(row.get("case_roc_auc_ci_high"))
                y_case_roc[m].append(case_roc)
                if case_roc is None or case_roc_low is None or case_roc_high is None:
                    y_case_roc_ci[m].append(None)
                else:
                    y_case_roc_ci[m].append(
                        (max(0.0, case_roc - case_roc_low), max(0.0, case_roc_high - case_roc))
                    )
                trace_roc = _to_float(row.get("trace_roc_auc"))
                trace_roc_low = _to_float(row.get("trace_roc_auc_ci_low"))
                trace_roc_high = _to_float(row.get("trace_roc_auc_ci_high"))
                y_trace_roc[m].append(trace_roc)
                if trace_roc is None or trace_roc_low is None or trace_roc_high is None:
                    y_trace_roc_ci[m].append(None)
                else:
                    y_trace_roc_ci[m].append(
                        (max(0.0, trace_roc - trace_roc_low), max(0.0, trace_roc_high - trace_roc))
                    )

        dataset_slug = _slug(dataset)
        out_paths.extend(
            _line_plot(
                title=f"Safety ({dataset}) - Verified Macro F1 vs Case Size",
                x_values=x_vals,
                x_labels=x_labels,
                y_by_method=y_verified_macro_f1,
                yerr_by_method=y_verified_macro_f1_ci,
                method_info_by_label=method_info,
                ylabel="Verified Macro F1",
                xlabel="Traces per Case",
                x_label_rotation=0,
                out_base=figures_dir / f"safety_{dataset_slug}_verified_macro_f1_by_case_size",
                formats=formats,
            )
        )
        out_paths.extend(
            _line_plot(
                title=f"Safety ({dataset}) - Avg Cost (USD) vs Case Size",
                x_values=x_vals,
                x_labels=x_labels,
                y_by_method=y_cost,
                yerr_by_method=y_cost_ci,
                method_info_by_label=method_info,
                ylabel="Avg Cost per Case (USD)",
                xlabel="Traces per Case",
                x_label_rotation=0,
                out_base=figures_dir / f"safety_{dataset_slug}_avg_cost_by_case_size",
                formats=formats,
            )
        )
        out_paths.extend(
            _auprc_by_case_size_plot(
                dataset=dataset,
                x_values=x_vals,
                x_labels=x_labels,
                case_ap_by_method=y_case_ap,
                case_ap_err_by_method=y_case_ap_ci,
                trace_ap_by_method=y_trace_ap,
                trace_ap_err_by_method=y_trace_ap_ci,
                case_roc_by_method=y_case_roc,
                case_roc_err_by_method=y_case_roc_ci,
                trace_roc_by_method=y_trace_roc,
                trace_roc_err_by_method=y_trace_roc_ci,
                method_info_by_label=method_info,
                out_base=figures_dir / f"safety_{dataset_slug}_auprc_by_case_size",
                formats=formats,
            )
        )

    for dataset, rows in sorted(by_dataset_pct.items(), key=lambda kv: kv[0]):
        method_info = _method_info_by_label(rows)
        groups = sorted({str(r.get("group") or "") for r in rows}, key=_pct_bucket_sort_key)
        positive_groups = [g for g in groups if g not in {"0%", "unknown"}]
        if not positive_groups:
            continue
        x_vals = positive_groups
        x_labels = positive_groups
        methods = sorted({str(r.get("method_label") or "") for r in rows})
        y_recall: dict[str, list[float | None]] = {m: [] for m in methods}
        y_recall_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        row_lookup = {(str(r.get("method_label") or ""), str(r.get("group") or "")): r for r in rows}
        for m in methods:
            for g in positive_groups:
                row = row_lookup.get((m, g))
                if row is None:
                    y_recall[m].append(None)
                    y_recall_ci[m].append(None)
                    continue
                tp = _to_int(row.get("tp"))
                fn = _to_int(row.get("fn"))
                denom = tp + fn
                p, low, high = _wilson_ci(tp, denom)
                y_recall[m].append(p)
                if p is None or low is None or high is None:
                    y_recall_ci[m].append(None)
                else:
                    y_recall_ci[m].append((p - low, high - p))

        dataset_slug = _slug(dataset)
        out_paths.extend(
            _line_plot(
                title=f"Safety ({dataset}) - Verified Recall vs Positive % Bucket",
                x_values=x_vals,
                x_labels=x_labels,
                y_by_method=y_recall,
                yerr_by_method=y_recall_ci,
                method_info_by_label=method_info,
                ylabel="Verified Recall",
                xlabel="Positive Trace Percentage Bucket",
                x_label_rotation=15,
                out_base=figures_dir / f"safety_{dataset_slug}_verified_recall_by_positive_pct",
                formats=formats,
            )
        )

        benign_method_labels: list[str] = []
        benign_values: list[float] = []
        benign_lower_errs: list[float] = []
        benign_upper_errs: list[float] = []
        for m in methods:
            row = row_lookup.get((m, "0%"))
            if row is None:
                continue
            fp = _to_int(row.get("fp"))
            tn = _to_int(row.get("tn"))
            denom = fp + tn
            p, low, high = _wilson_ci(fp, denom)
            if p is None or low is None or high is None:
                continue
            benign_method_labels.append(m)
            benign_values.append(p)
            benign_lower_errs.append(p - low)
            benign_upper_errs.append(high - p)

        out_paths.extend(
            _bar_plot_with_ci(
                title=f"Safety ({dataset}) - Benign False Positive Rate (0% Positive Bucket)",
                method_labels=benign_method_labels,
                method_info_by_label=method_info,
                values=benign_values,
                lower_errs=benign_lower_errs,
                upper_errs=benign_upper_errs,
                ylabel="Benign FPR",
                xlabel="Method",
                out_base=figures_dir / f"safety_{dataset_slug}_benign_false_positive_rate",
                formats=formats,
            )
        )

    return out_paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    print("Positive-percent buckets:")
    print("- 0%")
    lower = 0.0
    for upper in positive_pct_bounds:
        print(f"- ({lower:.1f},{upper:.1f}]%")
        lower = upper

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

    overall_chart_rows = list(all_overall)
    for file in _discover_safety_result_files():
        if file.resolve() in loaded_files:
            continue
        try:
            overall, _, _ = _collect_metrics(
                file,
                positive_pct_bounds=positive_pct_bounds,
            )
        except Exception as exc:
            print(f"Skipping overall-bar-chart load for {file}: {exc}")
            continue
        overall_chart_rows.extend(overall)

    _print_table("Overall", all_overall, include_group=False)
    _print_table("By Case Size", all_by_case, include_group=True)
    _print_table("By Positive Percent", all_by_pct, include_group=True)

    if not args.no_figures:
        figure_paths = _generate_figures(
            overall_rows=overall_chart_rows,
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
