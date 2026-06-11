"""Annotate VibeTest FAIL results with a binary human-audit verifier via OpenRouter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL|INCONCLUSIVE|NOT\s+APPLICABLE)\b", re.I)
CELL_RE = re.compile(r"\bcell\s+(\d+)\b", re.I)
CITATION_RE = re.compile(r"\[(/kaggle/repo/[^\]:]+)(?::([^\]]+))?\]")


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_verdict(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    raw = str(metadata.get("verdict") or "").strip().upper()
    if raw:
        return raw
    match = VERDICT_RE.search(str(test.get("description") or ""))
    if match:
        return match.group(1).upper()
    if test.get("passed") is False:
        return "FAIL"
    if test.get("passed") is True:
        return "PASS"
    return "INCONCLUSIVE"


def get_reason(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    return str(metadata.get("reason_text") or "").strip()


def get_evidence(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    return str(metadata.get("evidence_text") or "").strip()


def get_property(test: dict[str, Any]) -> str:
    metadata = test.get("metadata") or {}
    return str(metadata.get("test_description") or metadata.get("property_text") or "").strip()


def notebook_cell_context(path: Path, evidence: str) -> str:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"[Could not read notebook {path}: {exc}]"

    cells = nb.get("cells") or []
    requested = sorted({int(m.group(1)) for m in CELL_RE.finditer(evidence)})
    if not requested:
        requested = list(range(min(8, len(cells))))

    parts: list[str] = []
    for idx in requested:
        zero_idx = idx
        if zero_idx < 0 or zero_idx >= len(cells):
            continue
        cell = cells[zero_idx]
        source = cell.get("source") or ""
        if isinstance(source, list):
            source = "".join(source)
        source = str(source).strip()
        if not source:
            continue
        if len(source) > 2500:
            source = source[:2500] + "\n...[truncated]"
        parts.append(f"--- Notebook cell {idx} ({cell.get('cell_type', 'unknown')}) ---\n{source}")
    return "\n\n".join(parts)


def line_context(path: Path, loc: str | None) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"[Could not read file {path}: {exc}]"

    start = 1
    end = min(len(lines), 80)
    if loc:
        match = re.search(r"(\d+)(?:-(\d+))?", loc)
        if match:
            start = max(1, int(match.group(1)) - 3)
            end = min(len(lines), int(match.group(2) or match.group(1)) + 3)

    excerpt = []
    for line_no in range(start, end + 1):
        excerpt.append(f"{line_no}: {lines[line_no - 1]}")
    return "\n".join(excerpt)


def build_source_context(repo_path: Path, evidence: str) -> str:
    contexts: list[str] = [f"Local repository path: {repo_path}"]
    seen: set[Path] = set()

    for citation_path, loc in CITATION_RE.findall(evidence):
        rel = citation_path.removeprefix("/kaggle/repo/").replace("/", os.sep)
        local_path = repo_path / rel
        if local_path in seen:
            continue
        seen.add(local_path)
        if not local_path.exists():
            contexts.append(f"--- Missing cited file ---\n{citation_path} -> {local_path}")
            continue
        if local_path.suffix.lower() == ".ipynb":
            snippet = notebook_cell_context(local_path, evidence)
        else:
            snippet = line_context(local_path, loc)
        contexts.append(f"--- Cited source: {citation_path}{':' + loc if loc else ''} ---\n{snippet}")

    if len("\n\n".join(contexts)) > 12000:
        return "\n\n".join(contexts)[:12000] + "\n...[source context truncated]"
    return "\n\n".join(contexts)


def collect_fail_items(paths: list[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for result_path in paths:
        with result_path.open(encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                if not line.strip():
                    continue
                entry = json.loads(line)
                repo_path = Path(str(entry.get("repo") or ""))
                if not repo_path.is_absolute():
                    repo_path = (Path.cwd() / repo_path).resolve()
                for test_index, test in enumerate(entry.get("tests") or []):
                    if normalize_verdict(test) != "FAIL":
                        continue
                    metadata = test.get("metadata") or {}
                    items.append(
                        {
                            "result_file": str(result_path),
                            "row_index": row_index,
                            "property_index": test_index,
                            "dataset": result_path.name.split("_AT-")[0].removeprefix("kaggle_"),
                            "examples": re.search(r"examples(\d+)", result_path.name).group(1)
                            if re.search(r"examples(\d+)", result_path.name)
                            else "",
                            "repo_name": entry.get("repo_name") or repo_path.name,
                            "repo_path": repo_path,
                            "test_prompt": get_property(test),
                            "verdict": "FAIL",
                            "case_score": metadata.get("case_score"),
                            "evidence_strength": metadata.get("evidence_strength"),
                            "reason": get_reason(test),
                            "evidence": get_evidence(test),
                        }
                    )
    return items


def collect_csv_fail_items(csv_path: Path, *, dataset: str | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if dataset and str(row.get("dataset") or "").strip().lower() != dataset.lower():
                continue
            if str(row.get("verdict") or "").strip().upper() != "FAIL":
                continue

            repo_name = str(row.get("repo_name") or "").strip()
            row_dataset = str(row.get("dataset") or "").strip()
            repo_path = Path("data") / "kaggle" / f"kaggle-{row_dataset}" / repo_name
            repo_path = (Path.cwd() / repo_path).resolve()
            items.append(
                {
                    "result_file": row.get("result_file") or str(csv_path),
                    "row_index": row.get("row_index") or "",
                    "property_index": row.get("property_index") or "",
                    "dataset": row_dataset,
                    "examples": row.get("examples") or "",
                    "repo_name": repo_name,
                    "repo_path": repo_path,
                    "test_prompt": row.get("test_prompt") or "",
                    "verdict": "FAIL",
                    "case_score": row.get("case_score") or "",
                    "evidence_strength": row.get("evidence_strength") or "",
                    "reason": row.get("reason") or "",
                    "evidence": row.get("evidence") or "",
                }
            )
    return items


def render_prompt(template: str, item: dict[str, Any]) -> str:
    source_context = build_source_context(Path(item["repo_path"]), item["evidence"])
    values = {k: "" if v is None else str(v) for k, v in item.items()}
    values["source_context"] = source_context
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--csv-input", type=Path, help="Existing fail annotation CSV to audit.")
    parser.add_argument("--dataset", help="Optional dataset filter when using --csv-input.")
    parser.add_argument("--prompt", type=Path, default=Path("docs/fail_verdict_verifier_prompt.md"))
    parser.add_argument("--output", type=Path, default=Path("results/openrouter_fail_human_audit.csv"))
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--extra-title", default="PhysDuels")
    parser.add_argument("--no-think", action="store_true", help="Prefix prompts with /no_think for Qwen reasoning models.")
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--json-response", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set.")

    template = load_prompt_template(args.prompt)
    if args.csv_input:
        items = collect_csv_fail_items(args.csv_input, dataset=args.dataset)
    else:
        if not args.results:
            raise SystemExit("Provide result JSONL files or --csv-input.")
        items = collect_fail_items(args.results)
    if args.offset:
        items = items[args.offset :]
    if args.limit:
        items = items[: args.limit]

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "audited_outcome",
        "confidence",
        "reason",
        "evidence_check",
        "kept_failure_reason",
        "kept_failure_evidence",
        "false_fail_explanation",
        "dataset",
        "examples",
        "repo_name",
        "test_prompt",
        "case_score",
        "evidence_strength",
        "original_reason",
        "original_evidence",
        "result_file",
        "row_index",
        "property_index",
        "model",
        "raw_response",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(items, start=1):
            prompt = render_prompt(template, item)
            if args.no_think:
                prompt = "/no_think\n" + prompt
            extra_headers = {"X-Title": args.extra_title} if args.extra_title else None
            request_kwargs = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "extra_headers": extra_headers,
            }
            if args.json_response:
                request_kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request_kwargs)
            raw = response.choices[0].message.content or ""
            try:
                audit = parse_json_response(raw)
            except Exception as exc:
                audit = {
                    "audited_outcome": "PARSE_ERROR",
                    "confidence": "",
                    "reason": f"Failed to parse verifier response: {exc}",
                    "evidence_check": "",
                    "kept_failure_reason": "",
                    "kept_failure_evidence": "",
                    "false_fail_explanation": "",
                }
            writer.writerow(
                {
                    **{key: audit.get(key, "") for key in fieldnames[:7]},
                    "dataset": item["dataset"],
                    "examples": item["examples"],
                    "repo_name": item["repo_name"],
                    "test_prompt": item["test_prompt"],
                    "case_score": item["case_score"],
                    "evidence_strength": item["evidence_strength"],
                    "original_reason": item["reason"],
                    "original_evidence": item["evidence"],
                    "result_file": item["result_file"],
                    "row_index": item["row_index"],
                    "property_index": item["property_index"],
                    "model": args.model,
                    "raw_response": raw,
                }
            )
            print(f"[{idx}/{len(items)}] {item['dataset']} ex{item['examples']} {item['repo_name']} -> {audit.get('audited_outcome')}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote {len(items)} audit row(s): {args.output}")


if __name__ == "__main__":
    main()
