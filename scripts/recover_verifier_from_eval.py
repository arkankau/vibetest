"""Recover metadata.evidence_verifier from a partial Inspect eval log into result JSONL."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import jsonlines

from vibetest.agent.evidence_verifier_agent import parse_verifier_output
from vibetest.usage import usage_payload_from_sample_data


_POSITION_RE = re.compile(r"_row(\d+)_test(\d+)(?:_|$)")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(str(path), mode="r") as reader:
        return [dict(row) for row in reader]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(path), mode="w") as writer:
        for row in rows:
            writer.write(row)


def _extract_output(sample: dict[str, Any]) -> str:
    output = sample.get("output") or {}
    completion = str(output.get("completion") or "")
    if completion and "SCORE:" in completion.upper():
        return completion
    for msg in reversed(sample.get("messages") or []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and "SCORE:" in content.upper():
            return content
        if isinstance(content, list):
            parts = [
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = "\n".join(part for part in parts if part)
            if text and "SCORE:" in text.upper():
                return text
    return completion


def _execution_log(sample: dict[str, Any]) -> str:
    parts: list[str] = []
    for msg in sample.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content")
        text = content if isinstance(content, str) else str(content or "")
        if text:
            parts.append(f"[{role}] {text[:200]}...")
    return "\n".join(parts)


def _position_from_sample_id(sample_id: str) -> tuple[int, int] | None:
    match = _POSITION_RE.search(str(sample_id or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _payload_from_sample(sample: dict[str, Any], *, verifier_model: str) -> dict[str, Any]:
    output = _extract_output(sample)
    score, reason, assessment = parse_verifier_output(output)
    metadata = {
        "verifier_model": verifier_model,
        "evidence_support_score": score,
        "reason_text": reason,
        "evidence_assessment": assessment,
        "score": (sample.get("scores") or {}).get("includes"),
        "total_time": sample.get("total_time"),
        "working_time": sample.get("working_time"),
    }
    metadata.update(usage_payload_from_sample_data(sample))
    return {
        "score": score,
        "verifier_model": verifier_model,
        "reason_text": reason,
        "evidence_assessment": assessment,
        "raw_output": output,
        "execution_log": _execution_log(sample),
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
        "error": None if score is not None else "Execution or parsing failure",
        "recovered_from_eval": True,
    }


def recover_eval_into_jsonl(
    *,
    eval_path: Path,
    result_path: Path,
    verifier_model: str = "openai/gpt-5-mini",
    dry_run: bool = False,
) -> dict[str, int]:
    rows = _load_rows(result_path)
    recovered = 0
    skipped_existing = 0
    unmatched = 0

    with zipfile.ZipFile(eval_path) as zf:
        sample_names = [
            name for name in zf.namelist() if name.startswith("samples/") and name.endswith(".json")
        ]
        for name in sample_names:
            sample = json.loads(zf.read(name))
            sample_id = str(sample.get("id") or "")
            position = _position_from_sample_id(sample_id)
            if position is None:
                unmatched += 1
                continue
            row_idx, test_idx = position
            if row_idx >= len(rows):
                unmatched += 1
                continue
            tests = rows[row_idx].get("tests") or []
            if test_idx >= len(tests):
                unmatched += 1
                continue
            test = tests[test_idx]
            metadata = dict(test.get("metadata") or {})
            if metadata.get("evidence_verifier"):
                skipped_existing += 1
                continue
            payload = _payload_from_sample(sample, verifier_model=verifier_model)
            metadata["evidence_verifier"] = payload
            test["metadata"] = metadata
            recovered += 1

    if not dry_run and recovered:
        _write_rows(result_path, rows)
    return {
        "recovered": recovered,
        "skipped_existing": skipped_existing,
        "unmatched": unmatched,
        "samples_in_eval": len(sample_names),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_path", type=Path, help="Inspect .eval log to recover from.")
    parser.add_argument("result_path", type=Path, help="VibeTest JSONL to annotate in place.")
    parser.add_argument("--verifier-model", default="openai/gpt-5-mini")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.eval_path.exists():
        raise SystemExit(f"Eval log not found: {args.eval_path}")
    if not args.result_path.exists():
        raise SystemExit(f"Result file not found: {args.result_path}")
    stats = recover_eval_into_jsonl(
        eval_path=args.eval_path,
        result_path=args.result_path,
        verifier_model=args.verifier_model,
        dry_run=bool(args.dry_run),
    )
    print(
        f"{args.result_path}: recovered={stats['recovered']} "
        f"skipped_existing={stats['skipped_existing']} unmatched={stats['unmatched']} "
        f"samples_in_eval={stats['samples_in_eval']}"
    )


if __name__ == "__main__":
    main()
