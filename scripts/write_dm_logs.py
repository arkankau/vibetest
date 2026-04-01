#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
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
    monitor_path: Path | None
    trace_ap: float | None
    trace_ap_se: float | None


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


def _compute_metric(module, *, bg: int) -> RunMetric:
    result_path = module._dm_meerkat_result_path(
        RESULTS_DIR,
        domain="cyber",
        bg=bg,
        model_label="gpt-5.4-mini",
    )
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
            monitor_path=monitor_path,
            trace_ap=None,
            trace_ap_se=None,
        )
    trace_ap, trace_ap_se = module._method_metric_summary(
        result_path,
        method_label="Meerkat",
        monitor_path=monitor_path,
        metric_key="trace_ap",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=SEED,
    )
    return RunMetric(
        bg=bg,
        result_path=result_path,
        monitor_path=monitor_path,
        trace_ap=trace_ap,
        trace_ap_se=trace_ap_se,
    )


def _format_value(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.6f}"


def main() -> None:
    module = _load_analysis_module()
    metrics = [_compute_metric(module, bg=20), _compute_metric(module, bg=100)]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# DM Logs",
        "",
        f"Generated: {generated_at}",
        "",
        "Current trace-level AP for `Meerkat` on distributed misuse with `gpt-5.4-mini` in the cyber domain.",
        "",
        "| Setting | Trace AP | Bootstrap SE | Meerkat Result | Monitor Result |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for metric in metrics:
        lines.append(
            "| "
            f"cyber bg={metric.bg}x | "
            f"{_format_value(metric.trace_ap)} | "
            f"{_format_value(metric.trace_ap_se)} | "
            f"`{metric.result_path.relative_to(REPO_ROOT)}` | "
            f"`{metric.monitor_path.relative_to(REPO_ROOT) if metric.monitor_path else 'missing'}` |"
        )

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
