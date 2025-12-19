"""Simple example of using vibetest programmatically."""

import argparse
from pathlib import Path
import jsonlines
import csv
import sys

from vibetest import TestCase, VibeTestAgent
from vibetest.agent import BaselineAgent

csv.field_size_limit(sys.maxsize)

def run_baseline():
    """Run baseline agent once per repository (no test cases).
    
    The baseline agent simply examines each repository for bugs without
    specific test criteria.
    """
    print("=" * 80)
    print("Starting Vulnerability Tests - Baseline Method")
    print("=" * 80)
    
    # Step 1: Collect all repositories and create one test case per repo
    all_test_cases = []
    repo_paths = []
    
    for repo_path in Path("./data/vuln/repos/").iterdir():
        if repo_path.is_dir():
            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            
            all_test_cases.append(TestCase(description="", repo_path=repo_path))
    
    print(f"\nTotal repositories: {len(repo_paths)}")
    print(f"Total test cases: {len(all_test_cases)} (1 per repo)")
    print(f"\n{'=' * 80}")
    print("Executing baseline agent on all repositories...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute baseline agent once per repository
    agent = BaselineAgent(static=True, model="openai/gpt-5")
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Write results to file (one result per repo)
    with jsonlines.open(f"results/vuln_results_{agent.model_name.split('/')[1]}_baseline.jsonl", mode="w") as writer:
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
    print(f"\nResults saved to: results/vuln_results_{agent.model_name.split('/')[1]}_baseline.jsonl")
    print(f"{'=' * 80}")


def run_vibetest():
    """Run vibetest with specific test cases across all repositories."""
    print("=" * 80)
    print("Starting Vulnerability Tests - VibeTest Method")
    print("=" * 80)

    # load vuln metadata csv
    vuln_metadata = {}
    with open("./data/vuln/cwe-bench/vulnerability_patches.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_name = row["project_slug"]
            cwe_id = row["cwe_id"]
            cwe_name = row["cwe_name"]
            patch = row["patch"]
            vuln_metadata[repo_name] = (cwe_id, cwe_name, patch)

    # Step 1: Collect all repositories and create test cases
    all_test_cases = []
    repo_paths = []
    tests_per_repo = 1

    i = 0
    for repo_path in Path("./data/vuln/cwe-bench/repos/").iterdir():
        if i >= 10:
            break
        i += 1
        if repo_path.is_dir():
            if "_".join(str(repo_path.name).split("_")[1:]) not in vuln_metadata:
                continue

            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            
            # Create test cases for this repo
            test_description = vuln_metadata["_".join(str(repo_path.name).split("_")[1:])]
            desc = f"No violations of {test_description[0]}: {test_description[1]}."
            target = f"The prediction should describe the bug (as given in the reason and evidence) and should match the bug fixed by the following patch which removes the bug from the code:\n{test_description[2]}\n\nNote that the prediction corresponds to the buggy version of the code which the patch is fixing. The evidence in the prediction should correspond to the same bug as fixed by the patch. One way to determine if the evidence is correct is to see if any of the patched methods are mentioned in the evidence. If so, then the prediction should be treated as correct."
            all_test_cases.append(TestCase(description=desc, repo_path=repo_path, sandbox_path="/workdir", target=target))

    print(f"\nTotal repositories: {len(repo_paths)}")
    print(f"Total test cases: {len(all_test_cases)} ({tests_per_repo} tests × {len(repo_paths)} repos)")
    print(f"\n{'=' * 80}")
    print("Executing all tests in parallel...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute ALL tests in parallel across all repositories
    agent = VibeTestAgent(static=True, model="openai/gpt-5")
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Group results by repository and write to file
    with jsonlines.open(f"results/vuln_results_{agent.model_name.split('/')[1]}.jsonl", mode="w") as writer:
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
                if r.passed:
                    repo_passed += 1
                    status = "✓ PASSED"
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
    print(f"\nResults saved to: results/vuln_results_{agent.model_name.split('/')[1]}.jsonl")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibetest on Vulnerability repositories")
    parser.add_argument(
        "--method",
        type=str,
        choices=["vibetest", "baseline"],
        default="vibetest",
        help="Method to use: 'vibetest' for VibeTestAgent or 'baseline' for BaselineAgent"
    )
    args = parser.parse_args()
    
    if args.method == "baseline":
        run_baseline()
    else:
        run_vibetest()
