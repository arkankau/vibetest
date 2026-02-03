"""Simple example of using vibetest programmatically."""

import argparse
from pathlib import Path
import jsonlines

from vibetest import TestCase, VibeTestAgent
from vibetest.agent import BaselineAgent, CodexReviewAgent
from vibetest.baselines import (
    load_kaggle_properties,
    map_review_to_kaggle_tests,
    prepare_reference_invariants,
    run_traincheck,
)


def get_tests():
    properties = load_kaggle_properties()
    print(f"Loaded {len(properties)} properties from properties.md")
    return properties


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


def run_traincheck_baseline(
    subset: str,
    *,
    script_path: str | None,
    model_var: str | None,
    timeout_s: int,
    output_root: str | None,
    infer_relations: list[str] | None,
    reference_script: str,
    reference_invariants: str | None,
    repo_limit: int | None,
    repo_offset: int | None,
    output_path: str | None,
    max_iters: int | None,
    mapper_model: str | None,
):
    print("=" * 80)
    print("Starting Kaggle Repository Tests - TrainCheck Baseline")
    print("=" * 80)

    properties = get_tests()
    repo_paths: list[Path] = []
    results = []

    total_repos = 0
    limit = repo_limit if repo_limit and repo_limit > 0 else 50
    offset = repo_offset if repo_offset and repo_offset > 0 else 0
    for repo_path in sorted(Path(f"./data/kaggle/kaggle-{subset}").iterdir()):
        if total_repos >= limit:
            break
        if repo_path.is_dir():
            if offset > 0:
                offset -= 1
                continue
            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            total_repos += 1

    output_root_path = Path(output_root) if output_root else None
    if reference_invariants:
        invariants_path = Path(reference_invariants)
        if not invariants_path.is_absolute():
            invariants_path = invariants_path.resolve()
        print(f"Using provided reference invariants: {invariants_path}")
    else:
        reference_path = Path(reference_script)
        if not reference_path.is_absolute():
            reference_path = reference_path.resolve()
        print(f"Preparing reference invariants from: {reference_path}")
        invariants_path = prepare_reference_invariants(
            reference_path,
            output_root=output_root_path,
            infer_relations=infer_relations,
            timeout_s=timeout_s,
        )
        print(f"Reference invariants: {invariants_path}")

    for repo_path in repo_paths:
        print(f"\n{'=' * 80}")
        print(f"Repository: {repo_path.name}")
        print(f"{'=' * 80}")

        result = run_traincheck(
            repo_path,
            script_path=script_path,
            model_var=model_var,
            output_root=output_root_path,
            invariants_path=invariants_path,
            infer_relations=infer_relations,
            max_iters=max_iters,
            timeout_s=timeout_s,
        )
        failed_invariants = result.get("failed_invariants") or []
        traincheck_ok = result.get("ok")
        if traincheck_ok is False:
            review_text = ""
            tests = _all_inconclusive_tests(
                properties,
                reviewer="traincheck",
                mapper_model=mapper_model,
                reason="TrainCheck failed to run.",
            )
        elif not failed_invariants:
            review_text = ""
            tests = _all_pass_tests(
                properties,
                reviewer="traincheck",
                mapper_model=mapper_model,
                reason="No failed TrainCheck invariants.",
            )
        else:
            review_text = result.get("review") or ""
            tests = map_review_to_kaggle_tests(
                review_text,
                properties,
                reviewer="traincheck",
                mapper_model=mapper_model,
            )
        _augment_tests_with_review(
            tests,
            review_text,
            {
                "traincheck_ok": traincheck_ok,
                "traincheck_error": result.get("error"),
                "trace_dir": result.get("trace_dir"),
                "report_files": result.get("report_files", []),
                "failed_invariants_count": result.get("failed_invariants_count", 0),
                "converted_notebook": result.get("converted_notebook"),
                "reference_invariants": str(invariants_path),
            },
        )

        passed = sum(1 for t in tests if t.get("passed"))
        total = len(tests)
        results.append(
            {
                "repo": str(repo_path),
                "repo_name": repo_path.name,
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "tests": tests,
            }
        )

    output_path = Path(output_path) if output_path else Path("results") / f"kaggle_{subset}_results_traincheck.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(output_path), mode="w") as writer:
        for entry in results:
            writer.write(entry)

    print(f"\nResults saved to: {output_path}")


def run_review_baseline(
    subset: str,
    *,
    reviewer: str,
    model: str | None,
    codex_cmd: str,
    codex_model: str | None,
    codex_prompt: str,
    codex_timeout_s: int,
    mapper_model: str | None,
    repo_limit: int | None = None,
    repo_offset: int | None = None,
    output_path: str | None = None,
):
    print("=" * 80)
    print(f"Starting Kaggle Repository Tests - {reviewer} Review Baseline")
    print("=" * 80)

    properties = get_tests()
    repo_paths: list[Path] = []
    results = []

    total_repos = 0
    limit = repo_limit if repo_limit and repo_limit > 0 else 50
    offset = repo_offset if repo_offset and repo_offset > 0 else 0
    for repo_path in Path(f"./data/kaggle/kaggle-{subset}").iterdir():
        if total_repos >= limit:
            break
        if repo_path.is_dir():
            if offset > 0:
                offset -= 1
                continue
            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            total_repos += 1

    all_test_cases = [
        TestCase(
            name=repo_path.name,
            description="",
            repo_path=repo_path,
            sandbox_path="/kaggle",
        )
        for repo_path in repo_paths
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
        repo_path.name: result.message for repo_path, result in zip(repo_paths, all_results)
    }

    for repo_path in repo_paths:
        print(f"\n{'=' * 80}")
        print(f"Repository: {repo_path.name}")
        print(f"{'=' * 80}")

        review_text = review_map.get(repo_path.name, "")
        meta = {
            "codex_ok": bool(review_text),
            "codex_error": None if review_text else "No review output captured",
        }
        tests = map_review_to_kaggle_tests(
            review_text,
            properties,
            reviewer=reviewer,
            mapper_model=mapper_model,
        )
        _augment_tests_with_review(tests, review_text, meta)

        passed = sum(1 for t in tests if t.get("passed"))
        total = len(tests)
        results.append(
            {
                "repo": str(repo_path),
                "repo_name": repo_path.name,
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "tests": tests,
            }
        )

    output_path = Path(output_path) if output_path else Path("results") / f"kaggle_{subset}_results_{reviewer}_baseline.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(output_path), mode="w") as writer:
        for entry in results:
            writer.write(entry)

    print(f"\nResults saved to: {output_path}")

def run_baseline(subset: str, model: str, static: bool):
    """Run baseline agent once per repository (no test cases).
    
    The baseline agent simply examines each repository for bugs without
    specific test criteria.
    """
    print("=" * 80)
    print("Starting Kaggle Repository Tests - Baseline Method")
    print("=" * 80)
    
    # Step 1: Collect all repositories and create one test case per repo
    all_test_cases = []
    repo_paths = []
    
    total_repos = 0
    for repo_path in Path(f"./data/kaggle/kaggle-{subset}").iterdir():
        if total_repos >= 50:
            break

        if repo_path.is_dir():
            # the code repo is the only directory inside each repo_path
            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            
            # Create ONE test case per repo (baseline doesn't use test descriptions)
            if "titanic" in subset:
                all_test_cases.append(TestCase(name=repo_path.name, description="", repo_path=repo_path, sandbox_path="/kaggle", additional_data={"./titanic-kaggle-data": "/kaggle/input/titanic"}))
            else:
                all_test_cases.append(TestCase(name=repo_path.name, description="", repo_path=repo_path, sandbox_path="/kaggle"))
            
            total_repos += 1
    
    print(f"\nTotal repositories: {len(repo_paths)}")
    print(f"Total test cases: {len(all_test_cases)} (1 per repo)")
    print(f"\n{'=' * 80}")
    print("Executing baseline agent on all repositories...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute baseline agent once per repository
    agent = BaselineAgent(model=model, static=static)
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Write results to file (one result per repo)
    with jsonlines.open(f"results/kaggle_{subset}_results_{agent.model_name.split('/')[1]}_baseline.jsonl", mode="w") as writer:
        for idx, (repo_path, result) in enumerate(zip(repo_paths, all_results)):
            print(f"\n{'=' * 80}")
            print(f"Repository: {repo_path.name}")
            print(f"{'=' * 80}")
            
            if result.passed:
                status = "✓ NO BUGS FOUND"
            else:
                status = "✗ BUGS FOUND"
            
            print(f"\n{status}")
            print(f"\nMessage: {result.message[:200]}...")
            
            # Write result for this repo
            writer.write({
                "repo": str(repo_path),
                "repo_name": repo_path.name,
                "bugs_found": not result.passed,
                "message": result.message,
                "execution_log": result.execution_log,
                "metadata": result.metadata,
            })
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: results/kaggle_{subset}_results_{agent.model_name.split('/')[1]}{'_static' if static else ''}_baseline.jsonl")
    print(f"{'=' * 80}")


def run_vibetest(subset: str, model: str, static: bool):
    """Run vibetest with specific test cases across all repositories."""
    print("=" * 80)
    print("Starting Kaggle Repository Tests - VibeTest Method")
    print("=" * 80)
    
    # Step 1: Collect all repositories and create test cases
    test_strs = get_tests()
    all_test_cases = []
    repo_paths = []
    tests_per_repo = len(test_strs)
    
    total_repos = 0
    for repo_path in Path(f"./data/kaggle/kaggle-{subset}").iterdir():
        if total_repos >= 50:
            break

        if repo_path.is_dir():
            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            
            # Create test cases for this repo
            for i, desc in enumerate(test_strs):
                if "titanic" in subset:
                    all_test_cases.append(TestCase(name=f"{repo_path.name}_prop{i}", description=desc, repo_path=repo_path, sandbox_path="/kaggle", additional_data={"./titanic-kaggle-data": "/kaggle/input"}))
                elif "nlp" in subset:
                    all_test_cases.append(TestCase(name=f"{repo_path.name}_prop{i}", description=desc, repo_path=repo_path, sandbox_path="/kaggle", additional_data={"./nlp-kaggle-data": "/kaggle/input"}))
                else:
                    all_test_cases.append(TestCase(name=f"{repo_path.name}_prop{i}", description=desc, repo_path=repo_path, sandbox_path="/kaggle"))
            
            total_repos += 1
    
    print(f"\nTotal repositories: {len(repo_paths)}")
    print(f"Total test cases: {len(all_test_cases)} ({tests_per_repo} tests × {len(repo_paths)} repos)")
    print(f"\n{'=' * 80}")
    print("Executing all tests in parallel...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute ALL tests in parallel across all repositories
    agent = VibeTestAgent(model=model, static=static)
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Group results by repository and write to file
    with jsonlines.open(f"results/kaggle_{subset}_results_{agent.model_name.split('/')[1]}{'_static' if static else ''}.jsonl", mode="w") as writer:
        # Group results by repository
        for repo_idx, repo_path in enumerate(repo_paths):
            print(f"\n{'=' * 80}")
            print(f"Repository: {repo_path.name}")
            print(f"{'=' * 80}")
            
            # Extract results for this repo (tests_per_repo consecutive results)
            start_idx = repo_idx * tests_per_repo
            end_idx = start_idx + tests_per_repo
            repo_results = all_results[start_idx:end_idx]
            
            # Collect test results for this repo
            test_results = []
            repo_passed = 0
            repo_total = len(repo_results)
            
            for r in repo_results:
                if "PASS" in r.message:
                    repo_passed += 1
                    status = "✓ PASSED"
                elif "INCONCLUSIVE" in r.message:
                    status = "⚠ INCONCLUSIVE"
                elif "NOT APPLICABLE" in r.message:
                    status = "ℹ NOT APPLICABLE"
                else:
                    status = "✗ FAILED"
                
                print(f"\n{status}: {r.test_case.description}")
                if r.evidence:
                    print(f"  Evidence items: {len(r.evidence)}")
                
                test_results.append({
                    "description": r.message,
                    "passed": r.passed,
                    "evidence": [e.model_dump() for e in r.evidence],
                    "execution_log": r.execution_log,
                    "metadata": r.metadata,
                })
            
            # Write all results for this repo as a single entry
            writer.write({
                "repo": str(repo_path),
                "repo_name": repo_path.name,
                "total_tests": repo_total,
                "passed_tests": repo_passed,
                "failed_tests": repo_total - repo_passed,
                "tests": test_results,
            })
            
            print(f"\nRepo Summary: {repo_passed}/{repo_total} tests passed")
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: results/kaggle_{subset}_results_{agent.model_name.split('/')[1]}.jsonl")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibetest on Kaggle repositories")
    parser.add_argument(
        "--method",
        type=str,
        choices=["vibetest", "baseline", "traincheck", "codex"],
        default="vibetest",
        help="Method to use: vibetest, baseline, traincheck, or codex"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use for the VibeTestAgent (e.g., 'openai/gpt-5')"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static (non-execution) mode"
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["titanic", "diabetic", "nlp"],
        default="titanic",
        help="Subset of Kaggle repositories to test"
    )
    parser.add_argument(
        "--traincheck-script",
        type=str,
        help="Optional training script path (relative to repo) for TrainCheck",
    )
    parser.add_argument(
        "--traincheck-model",
        type=str,
        default="model",
        help="Model variable name to track for TrainCheck",
    )
    parser.add_argument(
        "--traincheck-timeout",
        type=int,
        default=1200,
        help="Timeout seconds for each TrainCheck phase",
    )
    parser.add_argument(
        "--traincheck-output",
        type=str,
        help="Output root directory for TrainCheck traces",
    )
    parser.add_argument(
        "--traincheck-relations",
        type=str,
        help="Comma-separated list of TrainCheck relations to enable (e.g., APIContainRelation)",
    )
    parser.add_argument(
        "--traincheck-reference",
        type=str,
        default="data/traincheck/mnist.py",
        help="Reference script used to infer TrainCheck invariants",
    )
    parser.add_argument(
        "--traincheck-invariants",
        type=str,
        help="Path to a precomputed TrainCheck invariants file",
    )
    parser.add_argument(
        "--repo-limit",
        type=int,
        help="Optional limit on number of repositories to run",
    )
    parser.add_argument(
        "--repo-offset",
        type=int,
        help="Optional number of repositories to skip before starting",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Optional output path for TrainCheck results JSONL",
    )
    parser.add_argument(
        "--traincheck-max-iters",
        type=int,
        help="Optional cap for range() iterations inside training scripts",
    )
    parser.add_argument(
        "--codex-cmd",
        type=str,
        default="codex",
        help="Codex CLI command name available inside the sandbox",
    )
    parser.add_argument(
        "--codex-model",
        type=str,
        default="inspect",
        help="Model name for Codex CLI (e.g., inspect to route via agent bridge)",
    )
    parser.add_argument(
        "--codex-prompt",
        type=str,
        default="/review",
        help="Prompt to send to Codex CLI",
    )
    parser.add_argument(
        "--codex-timeout",
        type=int,
        default=1200,
        help="Timeout seconds for Codex review",
    )
    parser.add_argument(
        "--review-mapper-model",
        type=str,
        help="Model name for mapping reviews to test cases",
    )
    args = parser.parse_args()
    
    if args.method == "baseline":
        run_baseline(args.subset, args.model, args.static)
    elif args.method == "traincheck":
        relations = None
        if args.traincheck_relations:
            relations = [r.strip() for r in args.traincheck_relations.split(",") if r.strip()]
        run_traincheck_baseline(
            args.subset,
            script_path=args.traincheck_script,
            model_var=args.traincheck_model,
            timeout_s=args.traincheck_timeout,
            output_root=args.traincheck_output,
            infer_relations=relations,
            reference_script=args.traincheck_reference,
            reference_invariants=args.traincheck_invariants,
            repo_limit=args.repo_limit,
            repo_offset=args.repo_offset,
            output_path=args.output_path,
            max_iters=args.traincheck_max_iters,
            mapper_model=args.review_mapper_model,
        )
    elif args.method == "codex":
        run_review_baseline(
            args.subset,
            reviewer="codex",
            model=args.model,
            codex_cmd=args.codex_cmd,
            codex_model=args.codex_model,
            codex_prompt=args.codex_prompt,
            codex_timeout_s=args.codex_timeout,
            mapper_model=args.review_mapper_model,
            repo_limit=args.repo_limit,
            repo_offset=args.repo_offset,
        )
    else:
        run_vibetest(args.subset, args.model, args.static)
