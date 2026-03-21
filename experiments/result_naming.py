"""Helpers for standardized experiment result filenames."""

from __future__ import annotations

import re
from pathlib import Path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip()).strip("-") or "results"


def _model_suffix(model_name: str | None) -> str | None:
    if not model_name:
        return None
    normalized = str(model_name).strip()
    if not normalized:
        return None
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    return _slug(normalized)


def standardized_results_path(
    dataset: str,
    method: str,
    *,
    model_name: str | None = None,
    recall_scored: bool = False,
) -> Path:
    dataset_slug = _slug(dataset)
    method_slug = _slug(method)
    model_suffix = _model_suffix(model_name)

    include_model = model_suffix is not None and (
        method_slug.upper() == "AT" or method_slug in {"codex"}
    )
    filename = f"{dataset_slug}_{method_slug}"
    if include_model and model_suffix:
        filename += f"-{model_suffix}"
    if recall_scored:
        filename += "_recall_scored"
    filename += ".jsonl"

    root = Path("results")
    if dataset_slug.startswith("synthetic_"):
        root = root / "synthetic"
    return root / filename
