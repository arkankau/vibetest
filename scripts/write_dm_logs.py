#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = REPO_ROOT / "scripts" / "analyze_distributed_misuse.py"
OUTPUT_PATH = REPO_ROOT / "DM_LOGS.md"
RESULTS_DIR = REPO_ROOT / "results"
BOOTSTRAP_SAMPLES = 1000
SEED = 42


@dataclass(frozen=True)
class RunMetric:
    domain: str
    model_label: str
    bg: int
    result_path: Path
    naive_result_path: Path
    monitor_path: Path | None
    zero_unmentioned_ap: float | None
    zero_unmentioned_ap_se: float | None
    naive_ap: float | None
    naive_ap_se: float | None
    monitor_ap: float | None
    monitor_ap_se: float | None


def _canonical_meerkat_path(module, *, domain: str, model_label: str, bg: int) -> Path:
    if domain == "cyber" and model_label == "Qwen-3.5" and bg == 200:
        return RESULTS_DIR / "dm_cyber_d6_bg200_qwen35_n50.jsonl"
    if domain == "cyber" and model_label == "Qwen-3.5" and bg == 1000:
        return RESULTS_DIR / "dm_cyber_d6_bg1000_qwen35_n20.jsonl"
    try:
        return module._dm_meerkat_result_path(
            RESULTS_DIR,
            domain=domain,
            bg=bg,
            model_label=model_label,
        )
    except ValueError:
        return RESULTS_DIR / f"dm_{domain}_d6_bg{bg}_{module._dm_model_slug(model_label)}.jsonl"


def _canonical_naive_path(*, domain: str, model_label: str, bg: int) -> Path:
    slug = _model_slug(model_label)
    return RESULTS_DIR / f"dm_{domain}_d6_bg{bg}_{slug}_naive.jsonl"


def _canonical_monitor_path(*, domain: str, model_label: str, bg: int) -> Path:
    slug = _model_slug(model_label)
    return RESULTS_DIR / f"dm_{domain}_d6_bg{bg}_{slug}_llmjudge.jsonl"


def _model_slug(model_label: str) -> str:
    if model_label == "Qwen-3.5":
        return "qwen35"
    if model_label == "gpt-5.4-mini":
        return "gpt-5.4-mini"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")


def _resolved_naive_path(module, *, domain: str, model_label: str, bg: int) -> Path:
    candidates = module._dm_naive_agent_result_candidates(
        RESULTS_DIR,
        domain=domain,
        bg=bg,
        model_label=model_label,
    )
    return _existing_path(candidates) or _canonical_naive_path(domain=domain, model_label=model_label, bg=bg)


def _resolved_monitor_path(module, *, domain: str, model_label: str, bg: int) -> Path:
    candidates = module._dm_monitor_result_candidates(
        RESULTS_DIR,
        domain=domain,
        bg=bg,
        model_label=model_label,
    )
    return _existing_path(candidates) or _canonical_monitor_path(domain=domain, model_label=model_label, bg=bg)


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


def _compute_metric(module, *, domain: str, model_label: str, bg: int) -> RunMetric:
    result_path = _canonical_meerkat_path(module, domain=domain, model_label=model_label, bg=bg)
    naive_result_path = _resolved_naive_path(module, domain=domain, model_label=model_label, bg=bg)
    monitor_path = _resolved_monitor_path(module, domain=domain, model_label=model_label, bg=bg)
    zero_unmentioned_ap = None
    zero_unmentioned_ap_se = None
    if result_path.is_file():
        zero_unmentioned_ap, zero_unmentioned_ap_se = module._method_metric_summary(
            result_path,
            method_label="Meerkat",
            monitor_path=monitor_path,
            metric_key="trace_ap",
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            seed=SEED,
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
    monitor_ap = None
    monitor_ap_se = None
    if monitor_path.is_file():
        monitor_ap, monitor_ap_se = module._method_metric_summary(
            monitor_path,
            method_label="Per-trace Monitor",
            monitor_path=None,
            metric_key="trace_ap",
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            seed=SEED,
        )
    return RunMetric(
        domain=domain,
        model_label=model_label,
        bg=bg,
        result_path=result_path,
        naive_result_path=naive_result_path,
        monitor_path=monitor_path,
        zero_unmentioned_ap=zero_unmentioned_ap,
        zero_unmentioned_ap_se=zero_unmentioned_ap_se,
        naive_ap=naive_ap,
        naive_ap_se=naive_ap_se,
        monitor_ap=monitor_ap,
        monitor_ap_se=monitor_ap_se,
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
        "Current trace-level AP for `Meerkat` on distributed misuse across cyber and bio, using explicit submitted trace scores with omitted traces filled to `0.0`, plus the `Naive Agent` and `LLMJudge` baselines.",
        "",
        "| Domain | Model | Setting | Meerkat Trace AP | Meerkat Trace AP SE | Naive Agent AP | Naive Agent SE | LLMJudge AP | LLMJudge SE | Meerkat Result | Naive Result | Monitor Result |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for metric in metrics:
        lines.append(
            "| "
            f"{metric.domain.title()} | "
            f"{metric.model_label} | "
            f"bg={metric.bg}x | "
            f"{_format_value(metric.zero_unmentioned_ap)} | "
            f"{_format_value(metric.zero_unmentioned_ap_se)} | "
            f"{_format_value(metric.naive_ap)} | "
            f"{_format_value(metric.naive_ap_se)} | "
            f"{_format_value(metric.monitor_ap)} | "
            f"{_format_value(metric.monitor_ap_se)} | "
            f"`{metric.result_path.relative_to(REPO_ROOT)}` | "
            f"`{metric.naive_result_path.relative_to(REPO_ROOT)}` | "
            f"`{metric.monitor_path.relative_to(REPO_ROOT) if metric.monitor_path else 'missing'}` |"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log current DM Meerkat AP values, including cyber Claude Opus 4-6 rows."
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the markdown to stdout without writing DM_LOGS.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    module = _load_analysis_module()
    metrics = [
        _compute_metric(module, domain="cyber", model_label="gpt-5.4-mini", bg=20),
        _compute_metric(module, domain="cyber", model_label="gpt-5.4-mini", bg=100),
        _compute_metric(module, domain="cyber", model_label="Qwen-3.5", bg=20),
        _compute_metric(module, domain="cyber", model_label="Qwen-3.5", bg=100),
        _compute_metric(module, domain="cyber", model_label="Qwen-3.5", bg=200),
        _compute_metric(module, domain="cyber", model_label="Qwen-3.5", bg=1000),
        _compute_metric(module, domain="cyber", model_label="claude-opus-4-6", bg=20),
        _compute_metric(module, domain="cyber", model_label="claude-opus-4-6", bg=100),
        _compute_metric(module, domain="bio", model_label="gpt-5.4-mini", bg=20),
        _compute_metric(module, domain="bio", model_label="gpt-5.4-mini", bg=100),
        _compute_metric(module, domain="bio", model_label="Qwen-3.5", bg=20),
        _compute_metric(module, domain="bio", model_label="Qwen-3.5", bg=100),
    ]
    markdown = _render_markdown(metrics)
    if args.print_only:
        print(markdown, end="")
        return
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
