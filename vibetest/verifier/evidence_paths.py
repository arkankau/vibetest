"""Helpers for resolving VibeTest evidence bundle paths for verifier runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def model_suffix(model_name: str) -> str:
    return model_name.split("/")[-1] if model_name else "unknown-model"


def source_sample_id_candidates(entry: dict[str, Any], test: dict[str, Any], row_idx: int) -> list[str]:
    metadata = test.get("metadata") or {}
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    for obj in (metadata, test, entry):
        if not isinstance(obj, dict):
            continue
        for key in ("source_sample_id", "sample_id", "test_case_name", "name", "id"):
            _add(obj.get(key))

    synthetic_row = entry.get("synthetic_row_index", row_idx)
    property_id = str(metadata.get("property_id") or test.get("property_id") or "").strip()
    dataset = str(entry.get("dataset") or "").strip()
    if property_id:
        if dataset:
            _add(f"{dataset}_row{synthetic_row}_{property_id}")
        _add(f"row{synthetic_row}_{property_id}")
    return candidates


def source_sample_id(entry: dict[str, Any], test: dict[str, Any], row_idx: int) -> str | None:
    candidates = source_sample_id_candidates(entry, test, row_idx)
    return candidates[0] if candidates else None


def evidence_tar_candidates(
    *,
    entry: dict[str, Any],
    test: dict[str, Any],
    row_idx: int,
    evidence_root: Path,
    evidence_model: str | None,
) -> tuple[list[Path], str | None]:
    metadata = test.get("metadata") or {}
    explicit = (
        metadata.get("evidence_tar")
        or metadata.get("evidence_tar_path")
        or test.get("evidence_tar")
        or entry.get("evidence_tar")
        or entry.get("evidence_tar_path")
    )
    sample_id = source_sample_id(entry, test, row_idx)
    if str(explicit or "").strip():
        path = Path(str(explicit))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return [path], sample_id

    model_name = (
        evidence_model
        or str(metadata.get("model") or "")
        or str(entry.get("method_model") or "")
    )
    if not model_name:
        return [], sample_id

    model_dir = evidence_root / model_suffix(model_name)
    paths: list[Path] = []
    seen: set[str] = set()
    for candidate_id in source_sample_id_candidates(entry, test, row_idx):
        path = model_dir / f"evidence-{candidate_id}.tar.gz"
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths, sample_id


def resolve_evidence_tar(
    *,
    entry: dict[str, Any],
    test: dict[str, Any],
    row_idx: int,
    evidence_root: Path,
    evidence_model: str | None,
) -> tuple[Path | None, str | None]:
    candidates, sample_id = evidence_tar_candidates(
        entry=entry,
        test=test,
        row_idx=row_idx,
        evidence_root=evidence_root,
        evidence_model=evidence_model,
    )
    for path in candidates:
        if path.exists():
            return path, sample_id
    return (candidates[0] if candidates else None), sample_id
