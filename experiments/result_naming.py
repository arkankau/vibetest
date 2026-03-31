"""Utilities for standardized result file naming.

Pattern:
- <dataset-name>_<method-name>.jsonl
- <dataset-name>_<method-name>-<model-name>.jsonl  (for AT and codex only)
- <dataset-name>_<method-name>[_-<model-name>]_recall_scored.jsonl
"""

from __future__ import annotations

import re
from pathlib import Path


METHODS_WITH_MODEL = {"AT", "AT-codex", "codex", "llmjudge", "buffer", "bayesian"}


def normalize_model_name(model_name: str | None) -> str:
    raw = (model_name or "").strip()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
    return raw or "unknown-model"


def standardized_results_filename(
    dataset_name: str,
    method_name: str,
    *,
    model_name: str | None = None,
    recall_scored: bool = False,
) -> str:
    dataset = (dataset_name or "").strip()
    method = (method_name or "").strip()
    if not dataset or not method:
        raise ValueError("dataset_name and method_name are required")

    base = f"{dataset}_{method}"
    if method in METHODS_WITH_MODEL:
        base = f"{base}-{normalize_model_name(model_name)}"

    if recall_scored:
        return f"{base}_recall_scored.jsonl"
    return f"{base}.jsonl"


def standardized_results_path(
    dataset_name: str,
    method_name: str,
    *,
    model_name: str | None = None,
    recall_scored: bool = False,
) -> Path:
    return Path("results") / standardized_results_filename(
        dataset_name,
        method_name,
        model_name=model_name,
        recall_scored=recall_scored,
    )
