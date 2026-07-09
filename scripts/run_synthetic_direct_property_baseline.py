"""Run a direct one-property LLM baseline on synthetic Kaggle labels.

This baseline is intentionally simpler than VibeTest:
- no ReAct loop
- no tool use
- no reviewer-to-property mapper
- one static prompt per repository-property pair

It asks the model to read a compact repository snapshot and return
PASS/FAIL/INCONCLUSIVE for exactly one property.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv


TEXT_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".r",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "wandb",
    "outputs",
    "results",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _property_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    props = row.get("ground_truth_by_property") or []
    out: list[dict[str, Any]] = []
    for idx, prop in enumerate(props):
        out.append(
            {
                "property_index": idx,
                "property_id": str(prop.get("property_id") or f"property_{idx}"),
                "property_text": str(prop.get("property_text") or ""),
            }
        )
    return out


def _decode_notebook(path: Path, text: str, max_chars: int) -> str:
    try:
        nb = json.loads(text)
    except Exception:
        return text[:max_chars]

    chunks: list[str] = []
    for idx, cell in enumerate(nb.get("cells") or []):
        source = cell.get("source") or []
        if isinstance(source, list):
            source_text = "".join(str(x) for x in source)
        else:
            source_text = str(source)
        if not source_text.strip():
            continue
        cell_type = cell.get("cell_type") or "cell"
        chunks.append(f"# {path.as_posix()} cell {idx} ({cell_type})\n{source_text}")
        if sum(len(c) for c in chunks) >= max_chars:
            break
    return "\n\n".join(chunks)[:max_chars]


def _repo_snapshot(repo_path: Path, *, max_total_chars: int, max_file_chars: int) -> str:
    files: list[Path] = []
    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(repo_path).parts):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        files.append(path)

    manifest = "\n".join(f"- {p.relative_to(repo_path).as_posix()}" for p in files[:200])
    parts = [f"Repository file list:\n{manifest}\n"]
    total = len(parts[0])

    for path in files:
        rel = path.relative_to(repo_path).as_posix()
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        body = (
            _decode_notebook(Path(rel), raw, max_file_chars)
            if path.suffix.lower() == ".ipynb"
            else raw[:max_file_chars]
        )
        block = f"\n\n===== FILE: {rel} =====\n{body}"
        if total + len(block) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 500:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _prompt(property_text: str, repo_context: str) -> str:
    return f"""/no_think
You are evaluating one machine-learning repository against one natural-language property.

Return only strict JSON. Do not include markdown.

Verdict rubric:
- PASS: the repository contains direct evidence satisfying the property.
- FAIL: the repository contains positive, independently checkable evidence violating the property.
- INCONCLUSIVE: the snapshot does not contain enough evidence for PASS or FAIL.

Evidence rules:
- Do not fail from speculation, weak evidence, missing optional best practices, or absence of a diagnostic unless the property explicitly requires it.
- Every FAIL must cite concrete code behavior from the repository snapshot.
- If the snapshot is insufficient, return INCONCLUSIVE.

Score rules:
- case_score is NOT confidence in your verdict.
- case_score is confidence that the repository VIOLATES the property.
- Use 0.0 to 0.2 when the verdict is PASS and the evidence is strong.
- Use 0.2 to 0.5 when the repository seems closer to PASS but evidence is incomplete, so the verdict is probably INCONCLUSIVE.
- Use 0.5 to 0.8 when the repository seems closer to FAIL but evidence is incomplete, so the verdict is probably INCONCLUSIVE.
- Use 0.8 to 1.0 when the verdict is FAIL and the evidence is strong.
- A confident PASS should have a low case_score, not a high one.

Property:
{property_text}

Repository snapshot:
{repo_context}

Return JSON with exactly these keys:
{{
  "verdict": "PASS or FAIL or INCONCLUSIVE",
  "case_score": 0.0,
  "evidence_strength": 0.0,
  "reason": "one to three concise sentences",
  "evidence": "specific supporting evidence or empty string"
}}
"""


def _call_model(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    parsed = _extract_json(content)
    usage = {}
    if response.usage is not None:
        usage = response.usage.model_dump()
    return parsed, content, usage


def _normalize_verdict(value: Any) -> str:
    verdict = str(value or "").strip().upper()
    if verdict in {"PASS", "FAIL", "INCONCLUSIVE"}:
        return verdict
    return "INCONCLUSIVE"


def _float_or_default(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, out))


def _entry_from_row(
    row: dict[str, Any],
    *,
    tests: list[dict[str, Any]],
    method_model: str,
    usage_totals: dict[str, int],
) -> dict[str, Any]:
    passed = sum(1 for t in tests if t.get("passed") is True)
    failed = sum(1 for t in tests if t.get("passed") is False)
    inconclusive = sum(1 for t in tests if t.get("passed") is None)
    return {
        "repo": row.get("output_repo_path") or row.get("repo_path") or "",
        "repo_name": row.get("repo_name") or row.get("repo_slug") or "",
        "domain": row.get("domain") or "",
        "dataset": row.get("dataset") or "",
        "synthetic_row_index": row.get("row_index"),
        "source_repo_path": row.get("source_repo_path") or "",
        "injected_sample_path": row.get("sample_name") or "",
        "injected_repo_path": row.get("output_repo_path") or "",
        "method": "direct-property-baseline",
        "method_model": method_model,
        "total_tests": len(tests),
        "passed_tests": passed,
        "failed_tests": failed,
        "inconclusive_tests": inconclusive,
        "tests": tests,
        "ground_truth_property_labels": row.get("ground_truth_property_labels") or {},
        "ground_truth_violation_descriptions": row.get("ground_truth_violation_descriptions") or {},
        "ground_truth_by_property": row.get("ground_truth_by_property") or [],
        "usage": {
            "model_usage": {},
            "usage_totals": usage_totals,
            "cost_usd": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--repo-limit", type=int, default=0)
    parser.add_argument("--repo-offset", type=int, default=0)
    parser.add_argument("--property-limit", type=int, default=0)
    parser.add_argument("--property-offset", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-file", help="Optional .env file to load before reading API settings.")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-file-chars", type=int, default=16000)
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=False)
        if not args.base_url:
            args.base_url = os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not args.api_key:
            args.api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not args.api_key:
        raise SystemExit("Missing API key. Set VLLM_API_KEY/OPENAI_API_KEY or pass --api-key.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    rows = _read_jsonl(Path(args.labels_path))
    if args.datasets:
        wanted = set(args.datasets)
        rows = [r for r in rows if r.get("dataset") in wanted]
    if args.repo_offset:
        rows = rows[args.repo_offset :]
    if args.repo_limit:
        rows = rows[: args.repo_limit]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for row_idx, row in enumerate(rows):
        repo_path = Path(row.get("output_repo_path") or "")
        if not repo_path.exists():
            raise SystemExit(f"Missing repo path for row {row_idx}: {repo_path}")
        repo_context = _repo_snapshot(
            repo_path,
            max_total_chars=args.max_context_chars,
            max_file_chars=args.max_file_chars,
        )
        props = _property_rows(row)
        if args.property_offset:
            props = props[args.property_offset :]
        if args.property_limit:
            props = props[: args.property_limit]

        tests: list[dict[str, Any]] = []
        for prop in props:
            started = time.time()
            prompt = _prompt(prop["property_text"], repo_context)
            parsed, raw, usage = _call_model(
                client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            for key in usage_totals:
                usage_totals[key] += int(usage.get(key) or 0)

            verdict = _normalize_verdict(parsed.get("verdict"))
            passed = True if verdict == "PASS" else False if verdict == "FAIL" else None
            case_score = _float_or_default(parsed.get("case_score"), 0.5)
            evidence_strength = _float_or_default(parsed.get("evidence_strength"), 0.0)
            reason = str(parsed.get("reason") or "Model did not return a valid reason.").strip()
            evidence_text = str(parsed.get("evidence") or "").strip()
            tests.append(
                {
                    "description": reason,
                    "passed": passed,
                    "evidence": [{"description": evidence_text}] if evidence_text else [],
                    "execution_log": raw[:12000],
                    "metadata": {
                        "reviewer": "direct-property-baseline",
                        "property_index": prop["property_index"],
                        "property_id": prop["property_id"],
                        "property_text": prop["property_text"],
                        "model": args.model,
                        "verdict": verdict,
                        "case_score": case_score,
                        "fail_support_score": case_score if verdict == "FAIL" else 0.0,
                        "evidence_strength": evidence_strength,
                        "reason_text": reason,
                        "evidence_text": evidence_text,
                        "total_time": time.time() - started,
                        "model_usage": usage,
                        "usage_totals": usage,
                        "cost_usd": None,
                    },
                }
            )
            print(
                f"row={row.get('row_index')} repo={row.get('repo_name')} "
                f"property={prop['property_id']} verdict={verdict} score={case_score:.2f}",
                flush=True,
            )

        entries.append(_entry_from_row(row, tests=tests, method_model=args.model, usage_totals=usage_totals))
        with output_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} rows to {output_path}")


if __name__ == "__main__":
    main()
