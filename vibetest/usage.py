"""Usage payload helpers shared across agents and experiment scripts."""

from __future__ import annotations

from typing import Any


_USAGE_INT_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_tokens_cache_read",
    "input_tokens_cache_write",
    "reasoning_tokens",
)


def _to_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def normalize_model_usage(raw: Any) -> dict[str, dict[str, int]]:
    """Normalize Inspect-style model_usage payloads into plain dictionaries."""

    payload = _as_mapping(raw)
    out: dict[str, dict[str, int]] = {}
    for model_name, usage_raw in payload.items():
        usage = _as_mapping(usage_raw)
        normalized = {field: _to_int(usage.get(field)) for field in _USAGE_INT_FIELDS}
        if any(normalized.values()):
            out[str(model_name)] = normalized
    return out


def usage_totals_from_model_usage(model_usage: Any) -> dict[str, int]:
    totals = {field: 0 for field in _USAGE_INT_FIELDS}
    for usage in normalize_model_usage(model_usage).values():
        for field in _USAGE_INT_FIELDS:
            totals[field] += _to_int(usage.get(field))
    return totals


def extract_cost_usd(payload: Any) -> float | None:
    """Extract an explicit USD cost from a nested payload when present."""

    stack: list[Any] = [payload]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        mapping = _as_mapping(current)
        if mapping:
            for key in ("cost_usd", "total_cost_usd"):
                value = _to_float(mapping.get(key))
                if value is not None:
                    return value
            stack.extend(mapping.values())
            continue

        if isinstance(current, list):
            stack.extend(current)

    return None


def aggregate_usage_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    combined_model_usage: dict[str, dict[str, int]] = {}
    combined_totals = {field: 0 for field in _USAGE_INT_FIELDS}
    cost_usd_total = 0.0
    saw_cost = False

    for payload in payloads:
        payload = _as_mapping(payload)
        model_usage = normalize_model_usage(payload.get("model_usage"))
        for model_name, usage in model_usage.items():
            bucket = combined_model_usage.setdefault(
                model_name,
                {field: 0 for field in _USAGE_INT_FIELDS},
            )
            for field in _USAGE_INT_FIELDS:
                bucket[field] += _to_int(usage.get(field))

        usage_totals_raw = payload.get("usage_totals")
        if isinstance(usage_totals_raw, dict):
            usage_totals = {field: _to_int(usage_totals_raw.get(field)) for field in _USAGE_INT_FIELDS}
        else:
            usage_totals = usage_totals_from_model_usage(model_usage)
        for field in _USAGE_INT_FIELDS:
            combined_totals[field] += usage_totals[field]

        cost_usd = _to_float(payload.get("cost_usd"))
        if cost_usd is None:
            cost_usd = extract_cost_usd(payload)
        if cost_usd is not None:
            cost_usd_total += cost_usd
            saw_cost = True

    return {
        "model_usage": combined_model_usage,
        "usage_totals": combined_totals,
        "cost_usd": round(cost_usd_total, 8) if saw_cost else None,
    }


def usage_payload_from_sample(sample: Any) -> dict[str, Any]:
    model_usage = normalize_model_usage(getattr(sample, "model_usage", {}))
    usage_totals = usage_totals_from_model_usage(model_usage)
    cost_usd = None
    if hasattr(sample, "model_dump"):
        cost_usd = extract_cost_usd(sample.model_dump())
    return {
        "model_usage": model_usage,
        "usage_totals": usage_totals,
        "cost_usd": cost_usd,
    }
