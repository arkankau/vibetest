"""Run a direct one-property LLM baseline on real Kaggle repositories.

This mirrors scripts/run_synthetic_direct_property_baseline.py, but enumerates
the real Kaggle repository folders under data/kaggle/kaggle-{subset}.
It produces VibeTest-shaped JSONL entries without requiring ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_synthetic_direct_property_baseline import (
    _call_model,
    _float_or_default,
    _normalize_verdict,
    _prompt,
    _repo_snapshot,
)
from vibetest.baselines import load_kaggle_properties


def _repo_paths(subset: str, *, repo_limit: int, repo_offset: int) -> list[Path]:
    root = Path(f"data/kaggle/kaggle-{subset}")
    if not root.exists():
        raise SystemExit(f"Missing Kaggle subset directory: {root}")
    repos = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if repo_offset > 0:
        repos = repos[repo_offset:]
    if repo_limit > 0:
        repos = repos[:repo_limit]
    return repos


def _entry(
    *,
    subset: str,
    repo_path: Path,
    tests: list[dict[str, Any]],
    method_model: str,
    usage_totals: dict[str, int],
    cost_total: float,
) -> dict[str, Any]:
    passed = sum(1 for t in tests if t.get("passed") is True)
    failed = sum(1 for t in tests if t.get("passed") is False)
    inconclusive = sum(1 for t in tests if t.get("passed") is None)
    return {
        "repo": str(repo_path),
        "repo_name": repo_path.name,
        "dataset": f"kaggle_{subset}",
        "method": "direct-property-baseline",
        "method_model": method_model,
        "total_tests": len(tests),
        "passed_tests": passed,
        "failed_tests": failed,
        "inconclusive_tests": inconclusive,
        "tests": tests,
        "usage": {
            "model_usage": {},
            "usage_totals": usage_totals,
            "cost_usd": cost_total,
        },
    }


def _run_subset(
    *,
    subset: str,
    output_path: Path,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    max_context_chars: int,
    max_file_chars: int,
    repo_limit: int,
    repo_offset: int,
) -> None:
    properties = load_kaggle_properties()
    repos = _repo_paths(subset, repo_limit=repo_limit, repo_offset=repo_offset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running real Kaggle direct-property baseline: subset={subset} "
        f"repos={len(repos)} properties={len(properties)} output={output_path}",
        flush=True,
    )

    entries: list[dict[str, Any]] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost_total = 0.0

    for repo_idx, repo_path in enumerate(repos):
        repo_context = _repo_snapshot(
            repo_path,
            max_total_chars=max_context_chars,
            max_file_chars=max_file_chars,
        )
        tests: list[dict[str, Any]] = []
        for prop_idx, property_text in enumerate(properties):
            started = time.time()
            prompt = _prompt(property_text, repo_context)
            parsed, raw, usage = _call_model(
                client,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for key in usage_totals:
                usage_totals[key] += int(usage.get(key) or 0)
            cost_total += float(usage.get("cost") or 0.0)

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
                        "property_index": prop_idx,
                        "property_text": property_text,
                        "model": model,
                        "verdict": verdict,
                        "case_score": case_score,
                        "fail_support_score": case_score if verdict == "FAIL" else 0.0,
                        "evidence_strength": evidence_strength,
                        "reason_text": reason,
                        "evidence_text": evidence_text,
                        "total_time": time.time() - started,
                        "model_usage": usage,
                        "usage_totals": usage,
                        "cost_usd": usage.get("cost"),
                    },
                }
            )
            print(
                f"subset={subset} repo={repo_idx}/{len(repos)} {repo_path.name} "
                f"property={prop_idx} verdict={verdict} score={case_score:.2f}",
                flush=True,
            )

        entries.append(
            _entry(
                subset=subset,
                repo_path=repo_path,
                tests=tests,
                method_model=model,
                usage_totals=usage_totals,
                cost_total=cost_total,
            )
        )
        with output_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} rows to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", choices=["titanic", "nlp", "diabetic"], required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-file", help="Optional .env file to load before reading API settings.")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-file-chars", type=int, default=16000)
    parser.add_argument("--repo-limit", type=int, default=0)
    parser.add_argument("--repo-offset", type=int, default=0)
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
    output_dir = Path(args.output_dir)
    for subset in args.subsets:
        output_path = output_dir / (
            f"kaggle_{subset}_direct-property-baseline-openrouter-qwen3.6-flash"
            f"{args.output_suffix}.jsonl"
        )
        _run_subset(
            subset=subset,
            output_path=output_path,
            client=client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_context_chars=args.max_context_chars,
            max_file_chars=args.max_file_chars,
            repo_limit=args.repo_limit,
            repo_offset=args.repo_offset,
        )


if __name__ == "__main__":
    main()
