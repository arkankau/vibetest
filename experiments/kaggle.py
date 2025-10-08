"""Simple example of using vibetest programmatically."""

from pathlib import Path
import jsonlines

from vibetest import TestCase, VibeTestAgent


def get_tests():
    tests = [
        "Training loss generally decreases during training (if no loss is logged, then add logging to check this).",
        "The model can overfit a single (or tiny) batch to near-zero loss.",
        "Randomizing the labels results in accuracy dropping to be near a random guessing baseline on a validation set.",
        "No leakage from test to train/val; model selection and hyperparameter tuning uses val only (if at all) and then testing happens once at the end.",
        "Test accuracy should be deterministic (same value) when running the test function multiple times without retraining.",
        "All model parameters are updated during training (no frozen layers unless explicitly intended).",
        "No model parameters or gradients are NaN or Inf.",
        "The trained model outperforms a simple baseline (e.g., random or majority class) on the test set.",
    ]
    return tests

def main():
    """Run a simple test example with parallel execution across all repositories."""
    # get all repos in data/kaggle
    print("=" * 80)
    print("Starting Kaggle Repository Tests (Parallel Execution)")
    print("=" * 80)
    
    # Step 1: Collect all repositories and create test cases
    test_strs = get_tests()
    all_test_cases = []
    repo_paths = []
    tests_per_repo = len(test_strs)
    
    total_repos = 0
    for repo_path in Path("./data/kaggle").iterdir():
        if total_repos >= 50:
            break

        if repo_path.is_dir():
            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            
            # Create test cases for this repo
            for desc in test_strs:
                all_test_cases.append(TestCase(description=desc, repo_path=repo_path))
            
            total_repos += 1
    
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
    with jsonlines.open("results/kaggle_results.jsonl", mode="w") as writer:
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
    print(f"\nResults saved to: results/kaggle_results.jsonl")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
