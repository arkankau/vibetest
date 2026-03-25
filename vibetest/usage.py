"""Helpers for normalizing token usage and optional cost metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Number
from typing import Any


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_tokens_cache_read",
    "input_tokens_cache_write",
    "reasoning_tokens",
)

_COST_FIELDS_PRIORITY = (
    "cost_usd",
    "total_cost_usd",
    "usd_cost",
    "total_cost",
    "cost",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _to_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, Number):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Number):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_model_usage(model_usage: Any) -> dict[str, dict[str, int]]:
    """Normalize model_usage into plain model->usage dicts."""
    normalized: dict[str, dict[str, int]] = {}
    usage_map = _as_mapping(model_usage)
    for model_name, usage_value in usage_map.items():
        usage_dict = _as_mapping(usage_value)
        model_key = str(model_name)
        normalized[model_key] = {field: _to_int(usage_dict.get(field, 0)) for field in _USAGE_FIELDS}
    return normalized


def usage_totals_from_model_usage(model_usage: Any) -> dict[str, int]:
    """Compute token totals across all models in a model_usage object."""
    normalized = normalize_model_usage(model_usage)
    totals = {field: 0 for field in _USAGE_FIELDS}
    for usage in normalized.values():
        for field in _USAGE_FIELDS:
            totals[field] += _to_int(usage.get(field, 0))
    return totals


def extract_cost_usd(payload: Any, *, max_depth: int = 8) -> float | None:
    """Best-effort extraction of USD cost from nested payloads.

    Returns None when no numeric cost-like field is present.
    """

    visited: set[int] = set()

    def walk(node: Any, depth: int) -> float | None:
        if depth > max_depth:
            return None
        node_id = id(node)
        if node_id in visited:
            return None
        visited.add(node_id)

        if isinstance(node, Mapping):
            for field in _COST_FIELDS_PRIORITY:
                cost_val = _to_float(node.get(field))
                if cost_val is not None:
                    return cost_val
            for value in node.values():
                found = walk(value, depth + 1)
                if found is not None:
                    return found
            return None
        if isinstance(node, list):
            for value in node:
                found = walk(value, depth + 1)
                if found is not None:
                    return found
            return None
        return None

    return walk(payload, 0)


def usage_payload_from_sample(sample: Any) -> dict[str, Any]:
    """Extract normalized usage payload from an Inspect sample object."""
    model_usage = normalize_model_usage(getattr(sample, "model_usage", None))
    usage_totals = usage_totals_from_model_usage(model_usage)
    sample_payload: dict[str, Any] = {}
    if hasattr(sample, "model_dump"):
        dumped = sample.model_dump()
        if isinstance(dumped, Mapping):
            sample_payload = dict(dumped)
    cost_usd = extract_cost_usd(sample_payload)
    return {
        "model_usage": model_usage,
        "usage_totals": usage_totals,
        "cost_usd": cost_usd,
    }


def usage_payload_from_sample_data(sample_data: Mapping[str, Any]) -> dict[str, Any]:
    """Extract usage payload from a raw sample JSON dict from a .eval archive."""
    model_usage = normalize_model_usage(sample_data.get("model_usage"))
    usage_totals = usage_totals_from_model_usage(model_usage)
    cost_usd = extract_cost_usd(sample_data)
    return {
        "model_usage": model_usage,
        "usage_totals": usage_totals,
        "cost_usd": cost_usd,
    }


def usage_payload_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract usage payload from result metadata."""
    metadata = metadata or {}
    model_usage = normalize_model_usage(metadata.get("model_usage"))
    usage_totals_value = metadata.get("usage_totals")
    usage_totals = (
        {field: _to_int(_as_mapping(usage_totals_value).get(field, 0)) for field in _USAGE_FIELDS}
        if isinstance(usage_totals_value, Mapping)
        else usage_totals_from_model_usage(model_usage)
    )
    cost_usd = _to_float(metadata.get("cost_usd"))
    if cost_usd is None:
        cost_usd = extract_cost_usd(metadata)
    return {
        "model_usage": model_usage,
        "usage_totals": usage_totals,
        "cost_usd": cost_usd,
    }


def aggregate_usage_payloads(payloads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate usage payloads into one combined usage object."""
    merged_model_usage: dict[str, dict[str, int]] = {}
    merged_totals = {field: 0 for field in _USAGE_FIELDS}
    total_cost = 0.0
    has_cost = False

    for payload in payloads:
        normalized = normalize_model_usage(payload.get("model_usage"))
        if not normalized and payload.get("usage_totals"):
            normalized = {}
        for model_name, usage in normalized.items():
            current = merged_model_usage.setdefault(
                model_name, {field: 0 for field in _USAGE_FIELDS}
            )
            for field in _USAGE_FIELDS:
                current[field] += _to_int(usage.get(field, 0))

        usage_totals = payload.get("usage_totals")
        if isinstance(usage_totals, Mapping):
            for field in _USAGE_FIELDS:
                merged_totals[field] += _to_int(usage_totals.get(field, 0))
        else:
            for usage in normalized.values():
                for field in _USAGE_FIELDS:
                    merged_totals[field] += _to_int(usage.get(field, 0))

        cost_usd = _to_float(payload.get("cost_usd"))
        if cost_usd is not None:
            total_cost += cost_usd
            has_cost = True

    return {
        "model_usage": merged_model_usage,
        "usage_totals": merged_totals,
        "cost_usd": round(total_cost, 8) if has_cost else None,
    }
