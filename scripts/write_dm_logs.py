#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = REPO_ROOT / "scripts" / "analyze_distributed_misuse.py"
OUTPUT_PATH = REPO_ROOT / "DM_LOGS.md"
RESULTS_DIR = REPO_ROOT / "results"
BOOTSTRAP_SAMPLES = 1000
SEED = 42


@dataclass(frozen=True)
class RunMetric:
    bg: int
    result_path: Path
    naive_result_path: Path
    monitor_path: Path | None
    zero_unmentioned_ap: float | None
    zero_unmentioned_ap_se: float | None
    naive_ap: float | None
    naive_ap_se: float | None


def _load_analysis_module():
    spec = importlib.util.spec_from_file_location("dm_analysis", ANALYSIS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load analysis module from {ANALYSIS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


_TRACE_SCORES_RE = re.compile(r"TRACE_SCORES_START\s*\n(.*?)\nTRACE_SCORES_END", re.S)


def _extract_explicit_trace_scores(module, row: dict) -> dict[str, float]:
    test = (row.get("tests") or [{}])[0]
    evidence_text = (
        test.get("metadata", {}).get("evidence_text")
        or (test.get("evidence") if isinstance(test.get("evidence"), str) else "")
        or ""
    )
    match = _TRACE_SCORES_RE.search(evidence_text)
    if not match:
        return {}
    scores: dict[str, float] = {}
    for raw_line in match.group(1).splitlines():
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
        scores[module._normalize_trace_path(parts[0].strip())] = max(0.0, min(1.0, score))
    return scores


def _average(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _zero_unmentioned_trace_ap_metrics(module, *, result_path: Path) -> tuple[float | None, float | None]:
    rows = module._load_rows(result_path)
    zero_values: list[float] = []
    for row in rows:
        gt = {
            module._normalize_trace_path(x)
            for x in (row.get("ground_truth", {}).get("harmful_trace_files") or [])
        }
        if not gt:
            continue
        explicit_scores = _extract_explicit_trace_scores(module, row)
        zero_ap = module._average_precision_for_case(explicit_scores, gt)
        if zero_ap is not None:
            zero_values.append(float(zero_ap))
    zero_ap = _average(zero_values)
    zero_ap_se = module._bootstrap_mean_se(
        zero_values,
        n_bootstrap=BOOTSTRAP_SAMPLES,
        seed=SEED,
    ) if zero_values else None
    return zero_ap, zero_ap_se


def _compute_metric(module, *, bg: int) -> RunMetric:
    result_path = module._dm_meerkat_result_path(
        RESULTS_DIR,
        domain="cyber",
        bg=bg,
        model_label="gpt-5.4-mini",
    )
    naive_result_path = _existing_path(
        module._dm_naive_agent_result_candidates(
            RESULTS_DIR,
            domain="cyber",
            bg=bg,
            model_label="gpt-5.4-mini",
        )
    ) or module._dm_naive_agent_result_candidates(
        RESULTS_DIR,
        domain="cyber",
        bg=bg,
        model_label="gpt-5.4-mini",
    )[0]
    monitor_path = _existing_path(
        module._dm_monitor_result_candidates(
            RESULTS_DIR,
            domain="cyber",
            bg=bg,
            model_label="gpt-5.4-mini",
        )
    )
    if not result_path.is_file():
        return RunMetric(
            bg=bg,
            result_path=result_path,
            naive_result_path=naive_result_path,
            monitor_path=monitor_path,
            zero_unmentioned_ap=None,
            zero_unmentioned_ap_se=None,
            naive_ap=None,
            naive_ap_se=None,
        )
    zero_unmentioned_ap, zero_unmentioned_ap_se = _zero_unmentioned_trace_ap_metrics(
        module,
        result_path=result_path,
    )
    naive_ap = None
    naive_ap_se = None
    if naive_result_path.is_file():
        naive_ap, naive_ap_se = module._method_metric_summary(
            naive_result_path,
            method_label="Naive Agent",
            monitor_path=None,
            metric_key="trace_ap",
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            seed=SEED,
        )
    return RunMetric(
        bg=bg,
        result_path=result_path,
        naive_result_path=naive_result_path,
        monitor_path=monitor_path,
        zero_unmentioned_ap=zero_unmentioned_ap,
        zero_unmentioned_ap_se=zero_unmentioned_ap_se,
        naive_ap=naive_ap,
        naive_ap_se=naive_ap_se,
    )


def _format_value(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.6f}"


def _render_markdown(metrics: list[RunMetric]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# DM Logs",
        "",
        f"Generated: {generated_at}",
        "",
        "Current zero-unmentioned trace-level AP for `Meerkat` on distributed misuse with `gpt-5.4-mini` in the cyber domain, plus the `Naive Agent` baseline.",
        "",
        "| Setting | Meerkat Zero-Unmentioned AP | Meerkat Zero-Unmentioned SE | Naive Agent AP | Naive Agent SE | Meerkat Result | Naive Result | Monitor Result |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for metric in metrics:
        lines.append(
            "| "
            f"cyber bg={metric.bg}x | "
            f"{_format_value(metric.zero_unmentioned_ap)} | "
            f"{_format_value(metric.zero_unmentioned_ap_se)} | "
            f"{_format_value(metric.naive_ap)} | "
            f"{_format_value(metric.naive_ap_se)} | "
            f"`{metric.result_path.relative_to(REPO_ROOT)}` | "
            f"`{metric.naive_result_path.relative_to(REPO_ROOT)}` | "
            f"`{metric.monitor_path.relative_to(REPO_ROOT) if metric.monitor_path else 'missing'}` |"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log current DM Meerkat AP values for gpt-5.4-mini on cyber.")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the markdown to stdout without writing DM_LOGS.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    module = _load_analysis_module()
    metrics = [_compute_metric(module, bg=20), _compute_metric(module, bg=100)]
    markdown = _render_markdown(metrics)
    if args.print_only:
        print(markdown, end="")
        return
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
