"""Run hallucination experiments on ICLR 2026 arXiv LaTeX projects."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import jsonlines

from vibetest import TestCase, VibeTestAgent
from vibetest.agent import CodexReviewAgent
from vibetest.baselines import (
    load_hallucination_properties,
    map_review_to_hallucination_tests,
    run_refchecker,
)


def _safe_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _iter_paper_paths(limit: int, offset: int) -> list[Path]:
    root = Path("./data/hallucination/arxiv_downloads")
    if not root.exists():
        raise SystemExit(f"Missing data directory: {root}")
    paths: list[Path] = []
    for paper_path in sorted(root.iterdir()):
        if not paper_path.is_dir():
            continue
        if offset > 0:
            offset -= 1
            continue
        paths.append(paper_path)
        if limit and len(paths) >= limit:
            break
    return paths


def _augment_tests_with_review(
    tests: list[dict],
    review_text: str,
    extra_meta: dict,
) -> None:
    excerpt = review_text[:8000] if review_text else ""
    for t in tests:
        t["execution_log"] = excerpt
        t["metadata"].update(extra_meta)


def _all_pass_tests(
    properties: list[str],
    *,
    reviewer: str,
    mapper_model: str | None,
    reason: str,
) -> list[dict]:
    tests: list[dict] = []
    for idx, prop in enumerate(properties):
        tests.append(
            {
                "description": reason,
                "passed": True,
                "evidence": [],
                "execution_log": "",
                "metadata": {
                    "reviewer": reviewer,
                    "mapper_model": mapper_model,
                    "property_index": idx,
                    "property_text": prop,
                    "verdict": "PASS",
                    "evidence_text": "",
                },
            }
        )
    return tests


def _all_inconclusive_tests(
    properties: list[str],
    *,
    reviewer: str,
    mapper_model: str | None,
    reason: str,
) -> list[dict]:
    tests: list[dict] = []
    for idx, prop in enumerate(properties):
        tests.append(
            {
                "description": reason,
                "passed": False,
                "evidence": [],
                "execution_log": "",
                "metadata": {
                    "reviewer": reviewer,
                    "mapper_model": mapper_model,
                    "property_index": idx,
                    "property_text": prop,
                    "verdict": "INCONCLUSIVE",
                    "evidence_text": "",
                },
            }
        )
    return tests


def run_vibetest(
    *,
    model: str | None,
    dynamic: bool,
    paper_limit: int,
    paper_offset: int,
    output_path: str | None,
):
    print("=" * 80)
    print("Starting Hallucination Tests - VibeTest Method")
    print("=" * 80)

    properties = load_hallucination_properties()
    print(f"Loaded {len(properties)} properties from properties.md")

    paper_paths = _iter_paper_paths(paper_limit, paper_offset)
    tests_per_paper = len(properties)
    all_test_cases: list[TestCase] = []

    for paper_path in paper_paths:
        safe_name = _safe_id(paper_path.name)
        print(f"Queueing paper: {paper_path.name}")
        for idx, prop in enumerate(properties):
            all_test_cases.append(
                TestCase(
                    name=f"{safe_name}_prop{idx}",
                    description=prop,
                    repo_path=paper_path,
                    sandbox_path="/workdir",
                )
            )

    print(f"\nTotal papers: {len(paper_paths)}")
    print(f"Total test cases: {len(all_test_cases)} ({tests_per_paper} tests × {len(paper_paths)} papers)")
    print(f"\n{'=' * 80}")
    print("Executing all tests in parallel...")
    print(f"{'=' * 80}\n")

    agent = VibeTestAgent(model=model, static=not dynamic)
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")

    output_path = Path(output_path) if output_path else Path("results") / f"hallucination_results_AT-{agent.model_name.split('/')[1]}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(str(output_path), mode="w") as writer:
        for paper_idx, paper_path in enumerate(paper_paths):
            print(f"\n{'=' * 80}")
            print(f"Paper: {paper_path.name}")
            print(f"{'=' * 80}")

            start_idx = paper_idx * tests_per_paper
            end_idx = start_idx + tests_per_paper
            paper_results = all_results[start_idx:end_idx]

            test_results = []
            paper_passed = 0
            for idx, r in enumerate(paper_results):
                if r.passed:
                    paper_passed += 1
                meta = dict(r.metadata or {})
                meta["property_index"] = idx
                meta["property_text"] = properties[idx]
                test_results.append(
                    {
                        "description": r.message,
                        "passed": r.passed,
                        "evidence": [e.model_dump() for e in r.evidence],
                        "execution_log": r.execution_log,
                        "metadata": meta,
                    }
                )

            writer.write(
                {
                    "repo": str(paper_path),
                    "repo_name": paper_path.name,
                    "total_tests": len(test_results),
                    "passed_tests": paper_passed,
                    "failed_tests": len(test_results) - paper_passed,
                    "tests": test_results,
                }
            )

    print(f"\nResults saved to: {output_path}")


def run_codex_baseline(
    *,
    model: str | None,
    codex_cmd: str,
    codex_model: str | None,
    codex_prompt: str,
    codex_timeout_s: int,
    mapper_model: str | None,
    paper_limit: int,
    paper_offset: int,
    output_path: str | None,
):
    print("=" * 80)
    print("Starting Hallucination Tests - Codex Review Baseline")
    print("=" * 80)

    properties = load_hallucination_properties()
    paper_paths = _iter_paper_paths(paper_limit, paper_offset)
    results = []

    all_test_cases = [
        TestCase(
            name=_safe_id(paper_path.name),
            description="",
            repo_path=paper_path,
            sandbox_path="/workdir",
        )
        for paper_path in paper_paths
    ]

    agent = CodexReviewAgent(
        model=model or "openai/gpt-5-mini",
        codex_cmd=codex_cmd,
        codex_model=codex_model,
        codex_prompt=codex_prompt,
        timeout_s=codex_timeout_s,
    )
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    review_map = {
        paper_path.name: result.message for paper_path, result in zip(paper_paths, all_results)
    }

    for paper_path in paper_paths:
        print(f"\n{'=' * 80}")
        print(f"Paper: {paper_path.name}")
        print(f"{'=' * 80}")

        review_text = review_map.get(paper_path.name, "")
        meta = {
            "codex_ok": bool(review_text),
            "codex_error": None if review_text else "No review output captured",
        }
        tests = map_review_to_hallucination_tests(
            review_text,
            properties,
            reviewer="codex",
            mapper_model=mapper_model,
        )
        _augment_tests_with_review(tests, review_text, meta)

        passed = sum(1 for t in tests if t.get("passed"))
        total = len(tests)
        results.append(
            {
                "repo": str(paper_path),
                "repo_name": paper_path.name,
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "tests": tests,
            }
        )

    output_path = Path(output_path) if output_path else Path("results") / "hallucination_results_codex_baseline.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(output_path), mode="w") as writer:
        for entry in results:
            writer.write(entry)

    print(f"\nResults saved to: {output_path}")


def run_refchecker_baseline(
    *,
    refchecker_cmd: str,
    refchecker_timeout_s: int,
    mapper_model: str | None,
    paper_limit: int,
    paper_offset: int,
    output_path: str | None,
    refchecker_output_root: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    db_path: str | None,
    workdir: str | None,
):
    print("=" * 80)
    print("Starting Hallucination Tests - RefChecker Baseline")
    print("=" * 80)

    properties = load_hallucination_properties()
    paper_paths = _iter_paper_paths(paper_limit, paper_offset)
    results = []

    output_root = Path(refchecker_output_root) if refchecker_output_root else None
    db_path_obj = Path(db_path) if db_path else None
    workdir_obj = Path(workdir) if workdir else None

    for paper_path in paper_paths:
        print(f"\n{'=' * 80}")
        print(f"Paper: {paper_path.name}")
        print(f"{'=' * 80}")

        result = run_refchecker(
            paper_path,
            refchecker_cmd=refchecker_cmd,
            output_root=output_root,
            timeout_s=refchecker_timeout_s,
            llm_provider=llm_provider,
            llm_model=llm_model,
            db_path=db_path_obj,
            workdir=workdir_obj,
        )

        refchecker_ok = result.get("ok")
        review_text = result.get("review") or ""

        if refchecker_ok is False:
            tests = _all_inconclusive_tests(
                properties,
                reviewer="refchecker",
                mapper_model=mapper_model,
                reason="RefChecker failed to run.",
            )
        elif not review_text.strip():
            tests = _all_pass_tests(
                properties,
                reviewer="refchecker",
                mapper_model=mapper_model,
                reason="RefChecker produced no findings.",
            )
        else:
            tests = map_review_to_hallucination_tests(
                review_text,
                properties,
                reviewer="refchecker",
                mapper_model=mapper_model,
            )

        _augment_tests_with_review(
            tests,
            review_text,
            {
                "refchecker_ok": refchecker_ok,
                "refchecker_error": result.get("error"),
                "refchecker_report": result.get("report_path"),
                "refchecker_paper_file": result.get("paper_file"),
            },
        )

        passed = sum(1 for t in tests if t.get("passed"))
        total = len(tests)
        results.append(
            {
                "repo": str(paper_path),
                "repo_name": paper_path.name,
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "tests": tests,
            }
        )

    output_path = Path(output_path) if output_path else Path("results") / "hallucination_results_refchecker_baseline.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(output_path), mode="w") as writer:
        for entry in results:
            writer.write(entry)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["vibetest", "codex", "refchecker"],
        required=True,
        help="Which method to run.",
    )
    parser.add_argument("--model", type=str, help="Model name for vibetest or codex wrapper.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic (non-static) vibetest agent.")
    parser.add_argument("--paper-limit", type=int, default=100, help="Number of papers to run.")
    parser.add_argument("--paper-offset", type=int, default=0, help="Number of papers to skip before starting.")
    parser.add_argument("--output-path", type=str, help="Output JSONL path.")

    # Codex options
    parser.add_argument("--codex-cmd", type=str, default="codex", help="Codex CLI command.")
    parser.add_argument("--codex-model", type=str, default="inspect", help="Codex model name.")
    parser.add_argument("--codex-prompt", type=str, default="/review", help="Codex prompt.")
    parser.add_argument("--codex-timeout", type=int, default=1200, help="Codex timeout seconds.")

    # RefChecker options
    parser.add_argument("--refchecker-cmd", type=str, default="academic-refchecker", help="RefChecker command.")
    parser.add_argument("--refchecker-timeout", type=int, default=1200, help="RefChecker timeout seconds.")
    parser.add_argument("--refchecker-output-root", type=str, help="Directory to store RefChecker reports.")
    parser.add_argument("--refchecker-llm-provider", type=str, help="RefChecker LLM provider.")
    parser.add_argument("--refchecker-llm-model", type=str, help="RefChecker LLM model.")
    parser.add_argument("--refchecker-db-path", type=str, help="Path to RefChecker DB.")
    parser.add_argument("--refchecker-workdir", type=str, help="Working directory for RefChecker command.")

    parser.add_argument("--review-mapper-model", type=str, help="Model name for mapping reviews to test cases.")

    args = parser.parse_args()

    if args.method == "vibetest":
        run_vibetest(
            model=args.model,
            dynamic=args.dynamic,
            paper_limit=args.paper_limit,
            paper_offset=args.paper_offset,
            output_path=args.output_path,
        )
    elif args.method == "codex":
        run_codex_baseline(
            model=args.model,
            codex_cmd=args.codex_cmd,
            codex_model=args.codex_model,
            codex_prompt=args.codex_prompt,
            codex_timeout_s=args.codex_timeout,
            mapper_model=args.review_mapper_model,
            paper_limit=args.paper_limit,
            paper_offset=args.paper_offset,
            output_path=args.output_path,
        )
    else:
        run_refchecker_baseline(
            refchecker_cmd=args.refchecker_cmd,
            refchecker_timeout_s=args.refchecker_timeout,
            mapper_model=args.review_mapper_model,
            paper_limit=args.paper_limit,
            paper_offset=args.paper_offset,
            output_path=args.output_path,
            refchecker_output_root=args.refchecker_output_root,
            llm_provider=args.refchecker_llm_provider,
            llm_model=args.refchecker_llm_model,
            db_path=args.refchecker_db_path,
            workdir=args.refchecker_workdir,
        )


if __name__ == "__main__":
    main()
