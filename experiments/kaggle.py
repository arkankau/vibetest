"""Simple example of using vibetest programmatically."""

import argparse
from pathlib import Path
import jsonlines

from vibetest import TestCase, VibeTestAgent
from vibetest.agent import BaselineAgent


def get_tests():
    with open(f"./data/kaggle/properties.md", mode="r") as f:
        properties = f.read().split("- ")[1:]  # Split by headings
    properties = [p.strip() for p in properties if p.strip()]
    print(f"Loaded {len(properties)} properties from properties.md")
    return properties

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
        choices=["vibetest", "baseline"],
        default="vibetest",
        help="Method to use: 'vibetest' for VibeTestAgent or 'baseline' for BaselineAgent"
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
    args = parser.parse_args()
    
    if args.method == "baseline":
        run_baseline(args.subset, args.model, args.static)
    else:
        run_vibetest(args.subset, args.model, args.static)
