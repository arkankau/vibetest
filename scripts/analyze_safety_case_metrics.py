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
import re
from pathlib import Path
from typing import Any

FIGURE_WIDTH_IN = 2.5
LINE_FIGURE_HEIGHT_IN = 1.85
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
    "AT": "#1F5AA6",
    "AT (Codex)": "#1B9E77",
    "Judge": "#C65D21",
}
METHOD_MARKERS = {
    "AT": "o",
    "AT (Codex)": "D",
    "Judge": "s",
}


PRICING_USD_PER_M: dict[str, dict[str, float]] = {
    "gpt-5-mini": {
        "input_tokens": 0.25,
        "input_tokens_cache_read": 0.025,
        "output_tokens": 2.00,
    },
    "gpt-5.2": {
        "input_tokens": 1.75,
        "input_tokens_cache_read": 0.175,
        "output_tokens": 14.00,
    },
    "gpt-5.3-codex": {
        "input_tokens": 1.75,
        "input_tokens_cache_read": 0.175,
        "output_tokens": 14.00,
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


def _fmt(x: float | None, digits: int = 4) -> str:
    return "N/A" if x is None else f"{x:.{digits}f}"


def _pricing_tier_for_model(model_name: str) -> str | None:
    name = (model_name or "").lower()
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
    output_tokens = _to_int(usage_totals.get("output_tokens"))
    total_tokens = _to_int(usage_totals.get("total_tokens"))
    if total_tokens <= 0 and input_tokens <= 0 and output_tokens <= 0:
        return None
    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    return (
        (uncached_input_tokens / 1_000_000.0) * pricing["input_tokens"]
        + (cached_input_tokens / 1_000_000.0) * pricing["input_tokens_cache_read"]
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
    return False


def _gt_positive_count(row: dict[str, Any]) -> int:
    gt = row.get("ground_truth") or {}
    files = gt.get("suspicious_trace_files")
    if isinstance(files, list):
        return len(files)
    files = gt.get("hacked_trace_files")
    if isinstance(files, list):
        return len(files)
    return 0


def _pred_positive(row: dict[str, Any]) -> bool | None:
    scoring = row.get("scoring") or {}
    value = scoring.get("predicted_has_reward_hacking")
    return value if isinstance(value, bool) else None


def _method_label(path: Path) -> str:
    return path.stem


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


def _init_metric_row(label: str, file_path: Path, group_key: str) -> dict[str, Any]:
    return {
        "method": label,
        "file": str(file_path),
        "group": group_key,
        "total_cases": 0,
        "classification_correct": 0,
        "verified_correct": 0,
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
    if pred_pos is True:
        metric["pred_positive"] += 1

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

    usage_totals = ((row.get("usage") or {}).get("usage_totals") or {})
    metric["input_tokens"] += _to_int(usage_totals.get("input_tokens"))
    metric["cached_input_tokens"] += _to_int(usage_totals.get("input_tokens_cache_read"))
    metric["output_tokens"] += _to_int(usage_totals.get("output_tokens"))
    metric["total_tokens"] += _to_int(usage_totals.get("total_tokens"))

    entry_cost = _entry_cost_usd(row, fallback_model_name=fallback_model)
    if entry_cost is not None:
        metric["total_cost_usd"] += entry_cost
        metric["has_cost"] = True


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
    out["avg_total_tokens_per_case"] = _safe_ratio(int(metric["total_tokens"]), total)
    out["avg_cost_usd_per_case"] = (
        (float(metric["total_cost_usd"]) / total) if total > 0 and metric.get("has_cost") else None
    )
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

    cols = [
        "method",
        "group",
        "total_cases",
        "classification_accuracy",
        "verified_accuracy",
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
    if not include_group:
        cols.remove("group")

    print(f"\n## {title}")
    header = " | ".join(cols)
    print(header)
    print(" | ".join("---" for _ in cols))

    for row in rows:
        cells: list[str] = []
        for c in cols:
            v = row.get(c)
            if c in {
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
    if name.endswith("_llmjudge"):
        return name[: -len("_llmjudge")], "llmjudge"

    if "_" in name:
        dataset, method = name.rsplit("_", 1)
        return dataset, method
    return "unknown", name


def _pretty_method(method: str) -> str:
    if method == "llmjudge":
        return "Judge"
    if method.startswith("AT-codex-"):
        return "AT (Codex)"
    if method.startswith("AT-"):
        return "AT"
    return method


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
        out = dict(row)
        out["dataset"] = dataset
        out["method_key"] = method
        out["method_label"] = _pretty_method(method)
        grouped.setdefault(dataset, []).append(out)
    return grouped


def _line_plot(
    *,
    title: str,
    x_values: list[Any],
    x_labels: list[str],
    y_by_method: dict[str, list[float | None]],
    yerr_by_method: dict[str, list[tuple[float, float] | None]] | None,
    ylabel: str,
    xlabel: str,
    x_label_rotation: int,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping figure generation: matplotlib is not installed.")
        return []

    _apply_publication_style(plt)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, LINE_FIGURE_HEIGHT_IN), constrained_layout=True)
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
        color = _method_color(method, i, palette)
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
            marker=_method_marker(method),
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=method,
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
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, min(2, len(y_by_method))),
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.0,
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
    colors = [_method_color(label, i, palette) for i, label in enumerate(method_labels)]
    x = list(range(len(method_labels)))
    ax.bar(
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


def _generate_figures(
    *,
    case_rows: list[dict[str, Any]],
    pct_rows: list[dict[str, Any]],
    figures_dir: Path,
    formats: list[str],
) -> list[Path]:
    out_paths: list[Path] = []
    by_dataset_case = _group_plot_rows_by_dataset(case_rows)
    by_dataset_pct = _group_plot_rows_by_dataset(pct_rows)

    for dataset, rows in sorted(by_dataset_case.items(), key=lambda kv: kv[0]):
        numeric_groups = sorted(
            {int(r["group"]) for r in rows if str(r.get("group", "")).isdigit()}
        )
        if not numeric_groups:
            continue
        x_vals = numeric_groups
        x_labels = [str(x) for x in x_vals]

        methods = sorted({str(r.get("method_label") or "") for r in rows})
        y_verified: dict[str, list[float | None]] = {m: [] for m in methods}
        y_verified_ci: dict[str, list[tuple[float, float] | None]] = {m: [] for m in methods}
        y_cost: dict[str, list[float | None]] = {m: [] for m in methods}
        row_lookup = {
            (str(r.get("method_label") or ""), int(str(r.get("group") or "0"))): r for r in rows if str(r.get("group", "")).isdigit()
        }
        for m in methods:
            for x in x_vals:
                row = row_lookup.get((m, x))
                if row is None:
                    y_verified[m].append(None)
                    y_verified_ci[m].append(None)
                    y_cost[m].append(None)
                    continue
                total_cases = _to_int(row.get("total_cases"))
                verified_correct = _to_int(row.get("verified_correct"))
                p, low, high = _wilson_ci(verified_correct, total_cases)
                y_verified[m].append(p)
                if p is None or low is None or high is None:
                    y_verified_ci[m].append(None)
                else:
                    y_verified_ci[m].append((p - low, high - p))
                y_cost[m].append(_to_float(row.get("avg_cost_usd_per_case")))

        dataset_slug = _slug(dataset)
        out_paths.extend(
            _line_plot(
                title=f"Safety ({dataset}) - Verified Accuracy vs Case Size",
                x_values=x_vals,
                x_labels=x_labels,
                y_by_method=y_verified,
                yerr_by_method=y_verified_ci,
                ylabel="Verified Accuracy",
                xlabel="Traces per Case",
                x_label_rotation=0,
                out_base=figures_dir / f"safety_{dataset_slug}_verified_accuracy_by_case_size",
                formats=formats,
            )
        )
        out_paths.extend(
            _line_plot(
                title=f"Safety ({dataset}) - Avg Cost (USD) vs Case Size",
                x_values=x_vals,
                x_labels=x_labels,
                y_by_method=y_cost,
                yerr_by_method=None,
                ylabel="Avg Cost per Case (USD)",
                xlabel="Traces per Case",
                x_label_rotation=0,
                out_base=figures_dir / f"safety_{dataset_slug}_avg_cost_by_case_size",
                formats=formats,
            )
        )

    for dataset, rows in sorted(by_dataset_pct.items(), key=lambda kv: kv[0]):
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
        discovered = sorted(Path("results").glob("safety_mle-sabotage_*.jsonl"))
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

    for file in files:
        if not file.exists():
            print(f"Skipping missing file: {file}")
            continue
        overall, by_case, by_pct = _collect_metrics(
            file,
            positive_pct_bounds=positive_pct_bounds,
        )
        all_overall.extend(overall)
        all_by_case.extend(by_case)
        all_by_pct.extend(by_pct)

    _print_table("Overall", all_overall, include_group=False)
    _print_table("By Case Size", all_by_case, include_group=True)
    _print_table("By Positive Percent", all_by_pct, include_group=True)

    if not args.no_figures:
        figure_paths = _generate_figures(
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
