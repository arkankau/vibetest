"""Utilities for recording token usage / cost in experiment outputs."""

from __future__ import annotations

from typing import Any, Mapping

from vibetest.usage import (
    aggregate_usage_payloads,
    usage_payload_from_metadata,
    usage_totals_from_model_usage,
    normalize_model_usage,
    usage_payload_from_sample_data,
)


def usage_from_result_metadata(result: Any) -> dict[str, Any]:
    """Extract usage payload from a TestResult-like object."""
    metadata = getattr(result, "metadata", None) or {}
    return usage_payload_from_metadata(metadata)


def aggregate_usage_from_results(results: list[Any]) -> dict[str, Any]:
    """Aggregate usage payload across TestResult-like objects."""
    payloads = [usage_from_result_metadata(result) for result in results]
    return aggregate_usage_payloads(payloads)


def aggregate_usage_from_tests(tests: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate usage payload across serialized test dicts."""
    payloads = []
    for test in tests:
        test = test or {}
        metadata_payload = usage_payload_from_metadata(test.get("metadata") or {})

        # Support usage fields serialized directly on test objects (e.g., from .eval parsing).
        direct_model_usage = normalize_model_usage(test.get("model_usage"))
        if direct_model_usage:
            direct_payload = {
                "model_usage": direct_model_usage,
                "usage_totals": test.get("usage_totals")
                if isinstance(test.get("usage_totals"), Mapping)
                else usage_totals_from_model_usage(direct_model_usage),
                "cost_usd": test.get("cost_usd"),
            }
            payloads.append(aggregate_usage_payloads([metadata_payload, direct_payload]))
        else:
            payloads.append(metadata_payload)
    return aggregate_usage_payloads(payloads)


def usage_from_eval_sample_data(sample_data: Mapping[str, Any]) -> dict[str, Any]:
    """Extract usage payload from raw sample JSON loaded from .eval."""
    return usage_payload_from_sample_data(sample_data)
