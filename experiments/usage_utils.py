"""Helpers for attaching and aggregating usage payloads in experiment outputs."""

from __future__ import annotations

from typing import Any

from vibetest.usage import (
    aggregate_usage_payloads,
    extract_cost_usd,
    normalize_model_usage,
    usage_totals_from_model_usage,
)


def _to_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _usage_payload_from_mapping(raw: Any) -> dict[str, Any]:
    payload = _as_mapping(raw)
    model_usage = normalize_model_usage(payload.get("model_usage"))
    usage_totals_raw = payload.get("usage_totals")
    if isinstance(usage_totals_raw, dict):
        usage_totals = {
            "input_tokens": _to_int(usage_totals_raw.get("input_tokens")),
            "output_tokens": _to_int(usage_totals_raw.get("output_tokens")),
            "total_tokens": _to_int(usage_totals_raw.get("total_tokens")),
            "input_tokens_cache_read": _to_int(usage_totals_raw.get("input_tokens_cache_read")),
            "input_tokens_cache_write": _to_int(usage_totals_raw.get("input_tokens_cache_write")),
            "reasoning_tokens": _to_int(usage_totals_raw.get("reasoning_tokens")),
        }
    else:
        usage_totals = usage_totals_from_model_usage(model_usage)

    cost_usd = payload.get("cost_usd")
    if cost_usd is None:
        cost_usd = extract_cost_usd(payload)

    return {
        "model_usage": model_usage,
        "usage_totals": usage_totals,
        "cost_usd": cost_usd,
    }


def usage_from_result_metadata(result: Any) -> dict[str, Any]:
    metadata = _as_mapping(getattr(result, "metadata", {}))
    return _usage_payload_from_mapping(metadata)


def aggregate_usage_from_results(results: list[Any]) -> dict[str, Any]:
    return aggregate_usage_payloads([usage_from_result_metadata(result) for result in results])


def aggregate_usage_from_tests(tests: list[dict[str, Any]]) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for test in tests:
        payload = _usage_payload_from_mapping(test)
        has_usage = bool(payload["model_usage"]) or any(payload["usage_totals"].values()) or payload["cost_usd"] is not None
        if has_usage:
            payloads.append(payload)
            continue

        metadata = _as_mapping(test.get("metadata"))
        review_usage = metadata.get("review_usage")
        if review_usage is not None:
            payloads.append(_usage_payload_from_mapping(review_usage))
            continue

        payloads.append(_usage_payload_from_mapping(metadata))
    return aggregate_usage_payloads(payloads)


def usage_from_eval_sample_data(sample_data: dict[str, Any]) -> dict[str, Any]:
    model_usage = normalize_model_usage(sample_data.get("model_usage"))
    if not model_usage:
        stats = _as_mapping(sample_data.get("stats"))
        model_usage = normalize_model_usage(stats.get("model_usage"))

    usage_totals = sample_data.get("usage_totals")
    if not isinstance(usage_totals, dict):
        usage_totals = usage_totals_from_model_usage(model_usage)
    else:
        usage_totals = {
            "input_tokens": _to_int(usage_totals.get("input_tokens")),
            "output_tokens": _to_int(usage_totals.get("output_tokens")),
            "total_tokens": _to_int(usage_totals.get("total_tokens")),
            "input_tokens_cache_read": _to_int(usage_totals.get("input_tokens_cache_read")),
            "input_tokens_cache_write": _to_int(usage_totals.get("input_tokens_cache_write")),
            "reasoning_tokens": _to_int(usage_totals.get("reasoning_tokens")),
        }

    return {
        "model_usage": model_usage,
        "usage_totals": usage_totals,
        "cost_usd": extract_cost_usd(sample_data),
    }
