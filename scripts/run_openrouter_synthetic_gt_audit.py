"""Audit synthetic Kaggle ground-truth labels for sampled high-confidence predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


CONTEXT_LIMIT = 80000


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def notebook_full_context(path: Path, *, limit: int) -> str:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return f"[Could not read notebook {path}: {exc}]"

    parts: list[str] = []
    for idx, cell in enumerate(nb.get("cells") or []):
        source = cell.get("source") or ""
        if isinstance(source, list):
            source = "".join(source)
        source = str(source).strip()
        if not source:
            continue
        parts.append(f"--- Notebook cell {idx} ({cell.get('cell_type', 'unknown')}) ---\n{source}")
        if len("\n\n".join(parts)) >= limit:
            break
    text = "\n\n".join(parts)
    return text[:limit] + ("\n...[notebook context truncated]" if len(text) > limit else "")


def file_full_context(path: Path, *, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[Could not read file {path}: {exc}]"
    return text[:limit] + ("\n...[file context truncated]" if len(text) > limit else "")


def build_full_repo_context(repo_path: Path, *, limit: int = CONTEXT_LIMIT) -> str:
    contexts: list[str] = [f"Local injected repository path: {repo_path}"]
    if not repo_path.exists():
        return f"Local injected repository path: {repo_path}\n[Repository path does not exist.]"

    files = [
        path
        for path in repo_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".ipynb", ".py", ".r", ".R", ".jl", ".md", ".txt", ".json", ".yaml", ".yml"}
        and ".git" not in path.parts
    ]
    files.sort(key=lambda path: (path.suffix.lower() != ".ipynb", str(path).lower()))

    for path in files:
        remaining = limit - len("\n\n".join(contexts))
        if remaining <= 1000:
            break
        rel = path.relative_to(repo_path)
        if path.suffix.lower() == ".ipynb":
            snippet = notebook_full_context(path, limit=remaining)
        else:
            snippet = file_full_context(path, limit=remaining)
        contexts.append(f"--- Injected repository source: {rel} ---\n{snippet}")

    text = "\n\n".join(contexts)
    return text[:limit] + ("\n...[source context truncated]" if len(text) > limit else "")


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def render_prompt(template: str, row: dict[str, str], source_context: str) -> str:
    values = {
        "dataset": row.get("dataset", ""),
        "repo_name": row.get("repo_name", ""),
        "examples": row.get("examples", ""),
        "source_repo_path": row.get("source_repo_path", ""),
        "injected_repo_path": row.get("injected_repo_path", ""),
        "property_id": row.get("property_id", ""),
        "test_property": row.get("property", ""),
        "ground_truth_label": row.get("ground_truth_label", ""),
        "ground_truth_violation_description": row.get("ground_truth_violation_description", ""),
        "prediction": row.get("prediction", ""),
        "error_type": row.get("error_type", ""),
        "qwen_correctness": row.get("qwen_correctness", ""),
        "case_score": row.get("case_score", ""),
        "reason": row.get("reason", ""),
        "evidence": row.get("evidence", ""),
        "source_context": source_context,
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def stratified_sample(rows: list[dict[str, str]], *, fraction: float, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.get("examples", ""),
                row.get("dataset", ""),
                row.get("qwen_correctness", ""),
                row.get("error_type", ""),
            )
        ].append(row)

    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        n = max(1, round(len(group) * fraction))
        selected.extend(group[:n])
    selected.sort(
        key=lambda row: (
            int(row.get("examples") or 0),
            row.get("dataset", ""),
            row.get("qwen_correctness", ""),
            row.get("error_type", ""),
            int(row.get("row_index") or 0),
            int(row.get("property_index") or 0),
        )
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/synthetic/synthetic_qwen_accepted_predictions_for_audit.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/synthetic/openrouter_synthetic_gt_audit_accepted_15pct.csv"))
    parser.add_argument("--prompt", type=Path, default=Path("docs/synthetic_ground_truth_audit_prompt.md"))
    parser.add_argument("--sample-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--json-response", action="store_true")
    parser.add_argument("--no-think", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set.")

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sampled = stratified_sample(rows, fraction=args.sample_fraction, seed=args.seed)
    if args.limit:
        sampled = sampled[: args.limit]

    template = load_prompt_template(args.prompt)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    audit_fields = [
        "audited_outcome",
        "confidence",
        "ground_truth_assessment",
        "qwen_prediction_assessment",
        "source_evidence_check",
        "correct_label",
        "correct_verdict",
        "clean_human_annotation",
    ]
    input_fields = list(sampled[0].keys()) if sampled else []
    fieldnames = audit_fields + input_fields + ["model", "raw_response"]

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(sampled, start=1):
            repo_path = Path(row.get("injected_repo_path") or "")
            if not repo_path.is_absolute():
                repo_path = (Path.cwd() / repo_path).resolve()
            source_context = build_full_repo_context(repo_path)
            prompt = render_prompt(template, row, source_context)
            if args.no_think:
                prompt = "/no_think\n" + prompt

            request_kwargs = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "extra_headers": {"X-Title": "PhysDuels"},
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
                    "ground_truth_assessment": f"Failed to parse verifier response: {exc}",
                    "qwen_prediction_assessment": "",
                    "source_evidence_check": "",
                    "correct_label": "",
                    "correct_verdict": "",
                    "clean_human_annotation": "",
                }

            writer.writerow(
                {
                    **{field: audit.get(field, "") for field in audit_fields},
                    **row,
                    "model": args.model,
                    "raw_response": raw,
                }
            )
            print(
                f"[{idx}/{len(sampled)}] ex{row.get('examples')} {row.get('dataset')} "
                f"{row.get('repo_name')} {row.get('property_id')} "
                f"{row.get('qwen_correctness')} {row.get('error_type')} "
                f"-> {audit.get('audited_outcome')}",
                flush=True,
            )
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote {len(sampled)} audit row(s): {args.output}")


if __name__ == "__main__":
    main()
