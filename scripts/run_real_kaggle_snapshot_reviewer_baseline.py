"""Run a repo-snapshot code-review baseline on real Kaggle repositories.

This is an OpenRouter-compatible replacement for the Codex CLI reviewer
baseline when the Codex/Inspect bridge cannot access the mounted repo or
cannot call the selected hosted model. It performs one general code review per
repository, then maps that review to the Kaggle properties.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_synthetic_direct_property_baseline import _repo_snapshot
from vibetest.baselines import load_kaggle_properties


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    for block in fenced:
        try:
            return json.loads(block)
        except Exception:
            continue
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def _call_chat(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature >= 0:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage is not None else {}
    return content, usage


def _review_prompt(repo_context: str) -> str:
    return f"""/no_think
You are a senior code reviewer auditing one machine-learning repository for correctness bugs and risky behavior.

Read the repository snapshot and report only concrete findings supported by code evidence. Focus on issues that affect correctness, data handling, train/evaluation behavior, reported results, reproducibility, or runtime behavior. Do not report speculative issues, missing optional best practices, or generic style concerns.

Return only strict JSON with this shape:
{{
  "summary": "one concise sentence",
  "findings": [
    {{
      "title": "short finding title",
      "body": "one to three sentences explaining the bug",
      "evidence": "specific file/cell/function evidence from the snapshot",
      "confidence": 0.0
    }}
  ]
}}

If you find no concrete bugs, return an empty findings list.

Repository snapshot:
{repo_context}
"""


def _mapping_prompt(review_text: str, properties: list[str]) -> str:
    props_payload = [
        {"property_index": idx, "text": prop} for idx, prop in enumerate(properties)
    ]
    return f"""/no_think
You are mapping a general code review to a fixed set of Kaggle ML test properties.

Use only the review text. Do not invent new repository evidence.

Rules:
- fail_support_score is a 0.0 to 1.0 score for how clearly the review supports the property failing.
- Use FAIL only when an explicit review finding clearly supports that exact property violation and fail_support_score is at least 0.7.
- Use PASS only when the review explicitly says the property is satisfied.
- Use INCONCLUSIVE when the property is not mentioned, ambiguous, or only weakly supported.
- A missing finding about a property is not evidence that the property passes.
- Evidence must quote or summarize the concrete review evidence that supports the mapped verdict.

Return only strict JSON:
[
  {{
    "property_index": 0,
    "verdict": "PASS or FAIL or INCONCLUSIVE",
    "fail_support_score": 0.0,
    "reason": "one concise sentence",
    "evidence": "specific mapped evidence or empty string"
  }}
]

Review JSON/text:
{review_text}

Properties:
{json.dumps(props_payload, ensure_ascii=False)}
"""


def _normal_verdict(value: Any) -> str:
    verdict = str(value or "").strip().upper()
    return verdict if verdict in {"PASS", "FAIL", "INCONCLUSIVE"} else "INCONCLUSIVE"


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _repo_paths(subset: str, *, repo_limit: int, repo_offset: int) -> list[Path]:
    root = Path(f"data/kaggle/kaggle-{subset}")
    repos = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if repo_offset > 0:
        repos = repos[repo_offset:]
    if repo_limit > 0:
        repos = repos[:repo_limit]
    return repos


def _usage_add(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += int(usage.get(key) or 0)


def _tests_from_mapping(
    mapping_text: str,
    *,
    properties: list[str],
    reviewer: str,
    mapper_model: str,
    review_text: str,
) -> list[dict[str, Any]]:
    parsed = _extract_json(mapping_text)
    if isinstance(parsed, dict):
        for key in ("results", "items", "output"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        parsed = []
    by_index: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("property_index"))
        except Exception:
            continue
        by_index[idx] = item

    tests: list[dict[str, Any]] = []
    for idx, prop in enumerate(properties):
        item = by_index.get(idx) or {}
        verdict = _normal_verdict(item.get("verdict"))
        fail_support_score = _score(item.get("fail_support_score"))
        reason = str(item.get("reason") or "Not mentioned in review.").strip()
        evidence = str(item.get("evidence") or "").strip()
        passed = True if verdict == "PASS" else False if verdict == "FAIL" else None
        tests.append(
            {
                "description": reason,
                "passed": passed,
                "evidence": [{"description": evidence}] if evidence else [],
                "execution_log": review_text[:12000],
                "metadata": {
                    "reviewer": reviewer,
                    "mapper_model": mapper_model,
                    "property_index": idx,
                    "property_text": prop,
                    "verdict": verdict,
                    "fail_support_score": fail_support_score,
                    "evidence_text": evidence,
                },
            }
        )
    return tests


def run_subset(
    *,
    subset: str,
    output_path: Path,
    client: OpenAI,
    model: str,
    mapper_model: str,
    max_review_tokens: int,
    max_map_tokens: int,
    max_context_chars: int,
    max_file_chars: int,
    temperature: float,
    repo_limit: int,
    repo_offset: int,
) -> None:
    properties = load_kaggle_properties()
    repos = _repo_paths(subset, repo_limit=repo_limit, repo_offset=repo_offset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Running snapshot reviewer baseline: subset={subset} repos={len(repos)} "
        f"model={model} mapper={mapper_model} output={output_path}",
        flush=True,
    )

    entries: list[dict[str, Any]] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for repo_idx, repo_path in enumerate(repos):
        started = time.time()
        repo_context = _repo_snapshot(
            repo_path,
            max_total_chars=max_context_chars,
            max_file_chars=max_file_chars,
        )
        review_text, review_usage = _call_chat(
            client,
            model=model,
            prompt=_review_prompt(repo_context),
            max_tokens=max_review_tokens,
            temperature=temperature,
        )
        _usage_add(usage_totals, review_usage)
        mapping_text, mapping_usage = _call_chat(
            client,
            model=mapper_model,
            prompt=_mapping_prompt(review_text, properties),
            max_tokens=max_map_tokens,
            temperature=temperature,
        )
        _usage_add(usage_totals, mapping_usage)
        tests = _tests_from_mapping(
            mapping_text,
            properties=properties,
            reviewer="snapshot-reviewer",
            mapper_model=mapper_model,
            review_text=review_text,
        )
        for test in tests:
            test["metadata"].update(
                {
                    "model": model,
                    "review_usage": review_usage,
                    "mapping_usage": mapping_usage,
                    "review_text": review_text[:12000],
                    "mapping_text": mapping_text[:12000],
                }
            )
        passed = sum(1 for t in tests if t["passed"] is True)
        failed = sum(1 for t in tests if t["passed"] is False)
        inconclusive = sum(1 for t in tests if t["passed"] is None)
        entries.append(
            {
                "repo": str(repo_path),
                "repo_name": repo_path.name,
                "dataset": f"kaggle_{subset}",
                "method": "snapshot-reviewer",
                "method_model": model,
                "total_tests": len(tests),
                "passed_tests": passed,
                "failed_tests": failed,
                "inconclusive_tests": inconclusive,
                "tests": tests,
                "usage": {
                    "review_usage": review_usage,
                    "mapping_usage": mapping_usage,
                    "usage_totals": dict(usage_totals),
                },
            }
        )
        with output_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(
            f"subset={subset} repo={repo_idx + 1}/{len(repos)} {repo_path.name} "
            f"PASS={passed} FAIL={failed} INCONCLUSIVE={inconclusive} "
            f"time={time.time() - started:.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", choices=["titanic", "nlp", "diabetic"], required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--mapper-model", default=None)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--env-file")
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--max-review-tokens", type=int, default=6000)
    parser.add_argument("--max-map-tokens", type=int, default=4000)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-file-chars", type=int, default=16000)
    parser.add_argument("--repo-limit", type=int, default=0)
    parser.add_argument("--repo-offset", type=int, default=0)
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=False)
        args.base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
        args.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    mapper_model = args.mapper_model or args.model
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    for subset in args.subsets:
        output_path = (
            Path(args.output_dir)
            / f"kaggle_{subset}_snapshot-reviewer-{args.model.replace('/', '-')}{args.output_suffix}.jsonl"
        )
        run_subset(
            subset=subset,
            output_path=output_path,
            client=client,
            model=args.model,
            mapper_model=mapper_model,
            max_review_tokens=args.max_review_tokens,
            max_map_tokens=args.max_map_tokens,
            max_context_chars=args.max_context_chars,
            max_file_chars=args.max_file_chars,
            temperature=args.temperature,
            repo_limit=args.repo_limit,
            repo_offset=args.repo_offset,
        )


if __name__ == "__main__":
    main()
