from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def _trace_content_hash(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_score_traces_parallel():
    from experiments.safety import _score_traces_parallel
    return _score_traces_parallel


DM_FILE_RE = re.compile(
    r"dm_(?P<domain>bio|cyber)_d(?P<decomp>\d+)_bg(?P<background>\d+)_(?P<variant>[A-Za-z0-9_]+)\.jsonl$"
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze distributed misuse scored JSONL files and plot Judge vs AT PR curves/AP."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional dm_*_v2.jsonl or dm_*_scored.jsonl files. If omitted, auto-discovers dm_*_v2.jsonl first and falls back to dm_*_scored.jsonl.",
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
    initial_scores_tsv_path = repo_path / "initial_scores.tsv"
    if initial_scores_tsv_path.exists():
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
            _get_score_traces_parallel()(
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
        stage2_scores = {
            _normalize_trace_path(k): float(v)
            for k, v in ((row.get("scoring", {}).get("trace_scores") or {}).items())
        }
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


def main() -> None:
    args = _parse_args()
    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        results_dir = Path(args.results_dir)
        preferred: dict[tuple[str, str, str], Path] = {}
        for path in sorted(results_dir.glob("dm_*_v2.jsonl")):
            match = DM_FILE_RE.match(path.name)
            if match:
                preferred[(match.group("domain"), match.group("decomp"), match.group("background"))] = path
        for path in sorted(results_dir.glob("dm_*_scored.jsonl")):
            match = DM_FILE_RE.match(path.name)
            if match:
                preferred.setdefault(
                    (match.group("domain"), match.group("decomp"), match.group("background")),
                    path,
                )
        input_paths = sorted(preferred.values())
    if not input_paths:
        raise SystemExit("No distributed-misuse result JSONL files found.")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_formats = [fmt.strip() for fmt in args.figure_formats.split(",") if fmt.strip()]
    stage1_cache_dir = Path(args.stage1_cache_dir)

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
    figure_paths: list[Path] = []
    figure_paths.extend(_plot_ap_bars(metrics, figures_dir=figures_dir, figure_formats=figure_formats))
    figure_paths.extend(_plot_pr_grid(metrics, figures_dir=figures_dir, figure_formats=figure_formats))
    figure_paths.extend(_plot_case_pr_grid(metrics, figures_dir=figures_dir, figure_formats=figure_formats))
    print("\nWrote figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
