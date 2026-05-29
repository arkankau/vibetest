"""Run a clean-context verifier over FAIL evidence in VibeTest result JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import jsonlines

from vibetest.agent import EvidenceVerifierAgent
from vibetest.testcases import TestCase, TestResult


_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL|INCONCLUSIVE|NOT\s+APPLICABLE)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.*?)(?:\nEVIDENCE:|\Z)", re.IGNORECASE | re.DOTALL)
_EVIDENCE_RE = re.compile(r"EVIDENCE:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _normalize_verdict(raw: Any, *, passed: Any = None) -> str:
    text = str(raw or "").strip().upper()
    if text in {"PASS", "FAIL", "INCONCLUSIVE", "NOT APPLICABLE"}:
        return text
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    if text.startswith("P"):
        return "PASS"
    if text.startswith("F"):
        return "FAIL"
    return "INCONCLUSIVE"


def _verdict_from_test(test: dict[str, Any]) -> str:
    desc = str(test.get("description") or "")
    match = _VERDICT_RE.search(desc)
    if match:
        return _normalize_verdict(match.group(1))

    metadata = test.get("metadata") or {}
    score = metadata.get("synthetic_score") or {}
    return _normalize_verdict(
        score.get("predicted_verdict") or metadata.get("verdict"),
        passed=test.get("passed"),
    )


def _section(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _evidence_text(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    if metadata.get("evidence_text"):
        return str(metadata.get("evidence_text"))

    desc = str(test.get("description") or "")
    from_desc = _section(_EVIDENCE_RE, desc)
    if from_desc:
        return from_desc

    evidence = test.get("evidence")
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, list):
        parts: list[str] = []
        for item in evidence:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("description", "text", "content", "value"):
                    if item.get(key):
                        parts.append(str(item[key]))
                        break
        return "\n".join(parts)
    return ""


def _reason_text(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    if metadata.get("reason_text"):
        return str(metadata.get("reason_text"))
    return _section(_REASON_RE, str(test.get("description") or ""))


def _property_text(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    return str(
        metadata.get("property_text")
        or metadata.get("test_description")
        or test.get("property_text")
        or ""
    ).strip()


def _repo_path(entry: dict[str, Any]) -> Path | None:
    raw = (
        entry.get("injected_repo_path")
        or entry.get("repo_path")
        or entry.get("repo")
        or ""
    )
    if not str(raw).strip():
        return None
    return Path(str(raw))


def _model_suffix(model_name: str) -> str:
    return model_name.split("/")[-1] if model_name else "unknown-model"


def _source_sample_id(entry: dict[str, Any], test: dict[str, Any], row_idx: int) -> str | None:
    candidates = _source_sample_id_candidates(entry, test, row_idx)
    return candidates[0] if candidates else None


def _source_sample_id_candidates(entry: dict[str, Any], test: dict[str, Any], row_idx: int) -> list[str]:
    metadata = test.get("metadata") or {}
    candidates: list[str] = []
    for obj in (metadata, test, entry):
        for key in ("source_sample_id", "sample_id", "test_case_name", "name", "id"):
            value = obj.get(key) if isinstance(obj, dict) else None
            if str(value or "").strip():
                sample_id = str(value).strip()
                if sample_id not in candidates:
                    candidates.append(sample_id)

    synthetic_row = entry.get("synthetic_row_index", row_idx)
    property_id = str(metadata.get("property_id") or test.get("property_id") or "").strip()
    if property_id:
        dataset = str(entry.get("dataset") or "").strip()
        if dataset:
            dataset_sample_id = f"{dataset}_row{synthetic_row}_{property_id}"
            if dataset_sample_id not in candidates:
                candidates.append(dataset_sample_id)
        legacy_sample_id = f"row{synthetic_row}_{property_id}"
        if legacy_sample_id not in candidates:
            candidates.append(legacy_sample_id)
    return candidates


def _evidence_tar_path(
    *,
    entry: dict[str, Any],
    test: dict[str, Any],
    row_idx: int,
    evidence_root: Path,
    evidence_model: str | None,
) -> tuple[Path | None, str | None]:
    metadata = test.get("metadata") or {}
    explicit = (
        metadata.get("evidence_tar")
        or metadata.get("evidence_tar_path")
        or test.get("evidence_tar")
        or entry.get("evidence_tar")
        or entry.get("evidence_tar_path")
    )
    if str(explicit or "").strip():
        path = Path(str(explicit))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path, _source_sample_id(entry, test, row_idx)

    sample_ids = _source_sample_id_candidates(entry, test, row_idx)
    if not sample_ids:
        return None, None

    model_name = (
        evidence_model
        or str(metadata.get("model") or "")
        or str(entry.get("method_model") or "")
    )
    if not model_name:
        return None, sample_ids[0]

    model_dir = evidence_root / _model_suffix(model_name)
    for sample_id in sample_ids:
        path = model_dir / f"evidence-{sample_id}.tar.gz"
        if path.exists():
            return path, sample_id
    return model_dir / f"evidence-{sample_ids[0]}.tar.gz", sample_ids[0]


def _safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_root = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = (dest_root / member.name).resolve()
            if not str(member_path).startswith(str(dest_root) + "/") and member_path != dest_root:
                raise ValueError(f"Unsafe path in evidence tar: {member.name}")
        tf.extractall(dest_root, filter="data")


def _prepare_evidence_data(
    evidence_tar: Path | None,
    temp_dirs: list[Path],
) -> tuple[dict[str, str], str | None]:
    if evidence_tar is None:
        return {}, None
    if not evidence_tar.exists():
        return {}, f"Evidence tar not found: {evidence_tar}"

    temp_dir = Path(tempfile.mkdtemp(prefix="vibetest-verifier-evidence-"))
    temp_dirs.append(temp_dir)
    try:
        _safe_extract_tar(evidence_tar, temp_dir)
    except Exception as exc:
        return {}, f"Failed to extract evidence tar {evidence_tar}: {exc}"

    evidence_dir = temp_dir / "evidence"
    if not evidence_dir.exists():
        return {}, f"Evidence tar has no evidence/ directory: {evidence_tar}"
    return {str(evidence_dir): "/"}, None


def _safe_sample_id(path: Path, row_idx: int, test_idx: int) -> str:
    stem = _SAFE_ID_RE.sub("_", path.stem)[:80]
    return f"{stem}_row{row_idx:04d}_test{test_idx:03d}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(str(path), mode="r") as reader:
        return [dict(row) for row in reader]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(path), mode="w") as writer:
        for row in rows:
            writer.write(row)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_evidence_verified{input_path.suffix}")


def _verifier_payload(result: TestResult) -> dict[str, Any]:
    metadata = result.metadata or {}
    return {
        "score": metadata.get("evidence_support_score"),
        "verifier_model": metadata.get("verifier_model"),
        "reason_text": metadata.get("reason_text") or "",
        "evidence_assessment": metadata.get("evidence_assessment") or "",
        "raw_output": result.message,
        "execution_log": result.execution_log,
        "usage": {
            key: metadata.get(key)
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "input_tokens_cache_read",
                "input_tokens_cache_write",
                "reasoning_tokens",
                "model_usage",
                "usage_totals",
            )
            if key in metadata
        },
        "total_time": metadata.get("total_time"),
        "working_time": metadata.get("working_time"),
        "error": metadata.get("error"),
    }


def _collect_verification_cases(
    *,
    result_path: Path,
    rows: list[dict[str, Any]],
    skip_existing: bool,
    limit: int,
    offset: int,
    fail_on_missing_repo: bool,
    evidence_root: Path,
    evidence_model: str | None,
    include_evidence_artifacts: bool,
    fail_on_missing_evidence: bool,
    temp_dirs: list[Path],
) -> tuple[list[TestCase], dict[str, tuple[int, int]], int, int]:
    test_cases: list[TestCase] = []
    sample_to_position: dict[str, tuple[int, int]] = {}
    skipped_for_offset = 0
    missing_repo = 0
    missing_evidence = 0

    for row_idx, entry in enumerate(rows):
        repo_path = _repo_path(entry)
        if repo_path is not None and not repo_path.is_absolute():
            repo_path = (Path.cwd() / repo_path).resolve()

        for test_idx, test in enumerate(entry.get("tests") or []):
            metadata = test.get("metadata") or {}
            if skip_existing and metadata.get("evidence_verifier"):
                continue
            if _verdict_from_test(test) != "FAIL":
                continue
            if skipped_for_offset < max(0, offset):
                skipped_for_offset += 1
                continue
            if limit > 0 and len(test_cases) >= limit:
                return test_cases, sample_to_position, missing_repo, missing_evidence

            if repo_path is None or not repo_path.exists():
                missing_repo += 1
                if fail_on_missing_repo:
                    raise SystemExit(f"Repo path not found for {result_path}:{row_idx}:{test_idx}: {repo_path}")
                metadata["evidence_verifier"] = {
                    "score": None,
                    "error": f"Repo path not found: {repo_path}",
                }
                test["metadata"] = metadata
                continue

            evidence_tar = None
            source_sample_id = None
            additional_data: dict[str, str] = {}
            evidence_error = None
            if include_evidence_artifacts:
                evidence_tar, source_sample_id = _evidence_tar_path(
                    entry=entry,
                    test=test,
                    row_idx=row_idx,
                    evidence_root=evidence_root,
                    evidence_model=evidence_model,
                )
                additional_data, evidence_error = _prepare_evidence_data(evidence_tar, temp_dirs)
                if evidence_error:
                    missing_evidence += 1
                    if fail_on_missing_evidence:
                        raise SystemExit(f"{result_path}:{row_idx}:{test_idx}: {evidence_error}")

            sample_id = _safe_sample_id(result_path, row_idx, test_idx)
            sample_to_position[sample_id] = (row_idx, test_idx)
            test_cases.append(
                TestCase(
                    name=sample_id,
                    description=_property_text(test),
                    repo_path=repo_path,
                    sandbox_path="/workspace",
                    additional_data=additional_data,
                    metadata={
                        "result_path": str(result_path),
                        "row_index": row_idx,
                        "test_index": test_idx,
                        "source_sample_id": source_sample_id,
                        "evidence_tar": str(evidence_tar) if evidence_tar else "",
                        "evidence_artifacts_available": bool(additional_data),
                        "evidence_artifacts_error": evidence_error or "",
                        "original_reason": _reason_text(test),
                        "original_evidence": _evidence_text(test),
                    },
                )
            )

    return test_cases, sample_to_position, missing_repo, missing_evidence


def _process_one_file(args: argparse.Namespace, result_path: Path) -> None:
    rows = _load_rows(result_path)
    temp_dirs: list[Path] = []
    try:
        test_cases, sample_to_position, missing_repo, missing_evidence = _collect_verification_cases(
            result_path=result_path,
            rows=rows,
            skip_existing=bool(args.skip_existing),
            limit=args.limit,
            offset=args.offset,
            fail_on_missing_repo=bool(args.fail_on_missing_repo),
            evidence_root=Path(args.evidence_root),
            evidence_model=args.evidence_model,
            include_evidence_artifacts=not bool(args.no_evidence_artifacts),
            fail_on_missing_evidence=bool(args.fail_on_missing_evidence),
            temp_dirs=temp_dirs,
        )

        print(
            f"{result_path}: {len(test_cases)} FAIL evidence item(s) queued "
            f"({missing_repo} skipped for missing repo path, "
            f"{missing_evidence} without evidence artifact bundle)."
        )
        if args.dry_run:
            return
        if not test_cases:
            output_path = result_path if args.in_place else Path(args.output_path or _default_output_path(result_path))
            _write_rows(output_path, rows)
            print(f"Wrote unchanged rows: {output_path}")
            return

        agent = EvidenceVerifierAgent(
            model=args.model,
            static=bool(args.static),
            log_dir=args.log_dir,
        )
        verifier_results = agent.execute_tests(test_cases, sandbox=args.sandbox)
        for result in verifier_results:
            position = sample_to_position.get(result.test_case.name)
            if position is None:
                continue
            row_idx, test_idx = position
            test = rows[row_idx]["tests"][test_idx]
            metadata = dict(test.get("metadata") or {})
            payload = _verifier_payload(result)
            payload["source_sample_id"] = result.test_case.metadata.get("source_sample_id")
            payload["evidence_tar"] = result.test_case.metadata.get("evidence_tar")
            payload["evidence_artifacts_available"] = result.test_case.metadata.get("evidence_artifacts_available")
            payload["evidence_artifacts_error"] = result.test_case.metadata.get("evidence_artifacts_error")
            metadata["evidence_verifier"] = payload
            test["metadata"] = metadata

        output_path = result_path if args.in_place else Path(args.output_path or _default_output_path(result_path))
        _write_rows(output_path, rows)
        print(f"Wrote verified results: {output_path}")
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="VibeTest result JSONL file(s) to annotate.")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="Verifier model.")
    parser.add_argument("--sandbox", default="docker", help="Inspect sandbox backend.")
    parser.add_argument("--static", action="store_true", help="Disable python tool in verifier agent.")
    parser.add_argument("--log-dir", default="./logs", help="Inspect log directory.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum FAIL items to verify per file.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many queued FAIL items per file.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tests that already have metadata.evidence_verifier.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite each input JSONL with verifier annotations.")
    parser.add_argument("--output-path", help="Output path for a single input file. Defaults to *_evidence_verified.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Only count queued FAIL evidence items.")
    parser.add_argument("--fail-on-missing-repo", action="store_true", help="Fail instead of annotating/skipping missing repos.")
    parser.add_argument("--evidence-root", default="evidence-dumps", help="Root directory containing base VibeTest evidence tarballs.")
    parser.add_argument("--evidence-model", help="Model folder/name used under --evidence-root. Defaults to each row's method_model/test metadata model.")
    parser.add_argument("--no-evidence-artifacts", action="store_true", help="Do not attach saved /evidence artifacts to verifier sandboxes.")
    parser.add_argument("--fail-on-missing-evidence", action="store_true", help="Fail when a matching evidence tarball is missing or invalid.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.output_path and len(args.results) != 1:
        raise SystemExit("--output-path can only be used with one input file.")

    for raw_path in args.results:
        result_path = Path(raw_path)
        if not result_path.exists():
            raise SystemExit(f"Result file not found: {result_path}")
        _process_one_file(args, result_path)


if __name__ == "__main__":
    main()
