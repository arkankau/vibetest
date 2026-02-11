"""Simple example of using vibetest programmatically."""

import argparse
from pathlib import Path
import jsonlines

from vibetest import TestCase, VibeTestAgent
from vibetest.agent import BaselineAgent
try:
    from experiments.usage_utils import aggregate_usage_from_results, usage_from_result_metadata
except ImportError:
    from usage_utils import aggregate_usage_from_results, usage_from_result_metadata


def get_tests():
    tests = [
        # "For run_rbm.py, training loss generally decreases during training and plateaus within the number of epochs used (if no loss is logged, then add logging to check this).",
        # "For run_rbm.py, the model can overfit a single (or tiny) batch to near-zero loss.",
        # "For run_rbm.py, randomizing the labels results in accuracy dropping to be near a random guessing baseline (may not be 0.5 if the data is imbalanced) on a validation set.",
        # "For run_rbm.py, no leakage from test to train/val. If there is any model selection or hyperparameter tuning, then it uses val only.",
        # "For run_rbm.py, accuracy should be deterministic (same value) when running the model evaluation multiple times without retraining. If no seed is set, then add a seed to check this.",
        # "For run_rbm.py, all model parameters are updated during training (no frozen layers unless explicitly intended).",
        # "For run_rbm.py, no model parameters or gradients are NaN or Inf during training.",
        "For run_rbm.py using Python 3.6, the model after the full training procedure outperforms a simple baseline (e.g., random or majority class) on the evaluation set.",
        # "For run_rbm.py, visualizing the input to the model (i.e. the x in `y_hat = model(x)`) shows that the data is being loaded and preprocessed correctly (e.g., no all-black images, no text with weird or unexpected characters, tables look correct and feature values are reasonable).",
    ]
    return tests

def run_baseline():
    """Run baseline agent once per repository (no test cases).
    
    The baseline agent simply examines each repository for bugs without
    specific test criteria.
    """
    print("=" * 80)
    print("Starting Kaggle Repository Tests - Baseline Method")
    print("=" * 80)
    
    # Step 1: Collect all repositories and create one test case per repo
    all_test_cases = []
    repo_path = Path("./examples/example-repo")
    
    all_test_cases.append(TestCase(description="", repo_path=repo_path))
    
    print(f"Total test cases: {len(all_test_cases)} (1 per repo)")
    print(f"\n{'=' * 80}")
    print("Executing baseline agent on all repositories...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute baseline agent once per repository
    agent = BaselineAgent(max_attempts=20)
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    result = all_results[0]
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Write results to file (one result per repo)
    with jsonlines.open(f"results/kaggle_results_{agent.model_name.split('/')[1]}_baseline.jsonl", mode="w") as writer:
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
            "usage": usage_from_result_metadata(result),
        })
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: results/kaggle_results_{agent.model_name.split('/')[1]}_baseline.jsonl")
    print(f"{'=' * 80}")


def run_vibetest():
    """Run vibetest with specific test cases across all repositories."""
    print("=" * 80)
    print("Starting Kaggle Repository Tests - VibeTest Method")
    print("=" * 80)
    
    # Step 1: Collect all repositories and create test cases
    test_strs = get_tests()
    all_test_cases = []
    repo_paths = []
    tests_per_repo = len(test_strs)
    repo_path = Path("./examples/example-repo")
    
    for desc in test_strs:
        all_test_cases.append(TestCase(description=desc, repo_path=repo_path))
    
    print(f"\nTotal repositories: {len(repo_paths)}")
    print(f"Total test cases: {len(all_test_cases)} ({tests_per_repo} tests × {len(repo_paths)} repos)")
    print(f"\n{'=' * 80}")
    print("Executing all tests in parallel...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute ALL tests in parallel across all repositories
    agent = VibeTestAgent(max_attempts=20)
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Group results by repository and write to file
    with jsonlines.open(f"results/kaggle_results_{agent.model_name.split('/')[1]}.jsonl", mode="w") as writer:
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
                "usage": aggregate_usage_from_results(repo_results),
            })
            
            print(f"\nRepo Summary: {repo_passed}/{repo_total} tests passed")
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: results/kaggle_results_{agent.model_name.split('/')[1]}.jsonl")
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
    args = parser.parse_args()
    
    if args.method == "baseline":
        run_baseline()
    else:
        run_vibetest()
