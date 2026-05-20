"""Run a clean-context verifier over FAIL evidence in VibeTest result JSONL files."""

from __future__ import annotations

import argparse
import json
import re
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
) -> tuple[list[TestCase], dict[str, tuple[int, int]], int]:
    test_cases: list[TestCase] = []
    sample_to_position: dict[str, tuple[int, int]] = {}
    skipped_for_offset = 0
    missing_repo = 0

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
                return test_cases, sample_to_position, missing_repo

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

            sample_id = _safe_sample_id(result_path, row_idx, test_idx)
            sample_to_position[sample_id] = (row_idx, test_idx)
            test_cases.append(
                TestCase(
                    name=sample_id,
                    description=_property_text(test),
                    repo_path=repo_path,
                    sandbox_path="/workspace",
                    metadata={
                        "result_path": str(result_path),
                        "row_index": row_idx,
                        "test_index": test_idx,
                        "original_reason": _reason_text(test),
                        "original_evidence": _evidence_text(test),
                        "original_output": str(test.get("description") or ""),
                    },
                )
            )

    return test_cases, sample_to_position, missing_repo


def _process_one_file(args: argparse.Namespace, result_path: Path) -> None:
    rows = _load_rows(result_path)
    test_cases, sample_to_position, missing_repo = _collect_verification_cases(
        result_path=result_path,
        rows=rows,
        skip_existing=bool(args.skip_existing),
        limit=args.limit,
        offset=args.offset,
        fail_on_missing_repo=bool(args.fail_on_missing_repo),
    )

    print(
        f"{result_path}: {len(test_cases)} FAIL evidence item(s) queued "
        f"({missing_repo} skipped for missing repo path)."
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
        metadata["evidence_verifier"] = _verifier_payload(result)
        test["metadata"] = metadata

    output_path = result_path if args.in_place else Path(args.output_path or _default_output_path(result_path))
    _write_rows(output_path, rows)
    print(f"Wrote verified results: {output_path}")


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
