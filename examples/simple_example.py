"""Simple example of using vibetest programmatically."""

from pathlib import Path

from vibetest import TestCase, VibeTestAgent


def main():
    """Run a simple test example."""
    # Path to the repository you want to test
    repo_path = Path("./example_ml_repo")

    test1 = TestCase(
        description="Training loss is logged and generally decreases",
        repo_path=repo_path
    )
    test2 = TestCase(
        description="The model can overfit a single (or tiny) batch to near-zero loss.",
        repo_path=repo_path
    )
    test3 = TestCase(
        description="No leakage from test to train/val; model selection and hyperparameter tuning uses val only (if at all) and then testing happens once at the end.",
        repo_path=repo_path
    )
    test4 = TestCase(
        description="Test accuracy should be deterministic (same value) when running the test function multiple times without retraining.",
        repo_path=repo_path
    )
    tests = [test1, test2, test3, test4]
    # tests = [TestCase(description="Take a look at the repository and tell me if there are any bugs.", repo_path=repo_path)]

    # Create agent and run test
    print(f"Repository: {repo_path}\n")

    agent = VibeTestAgent(
        max_attempts=20,
    )

    # Execute with Docker sandbox
    # Note: execute_test() is synchronous - Inspect AI manages its own event loop
    results = agent.execute_tests(tests, sandbox="docker")
    # Print results in pytest style
    print("\n" + "=" * 80)
    print("VIBETEST RESULTS")
    print("=" * 80)

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    for i, result in enumerate(results, 1):
        status = "PASSED" if result.passed else "FAILED"
        status_symbol = "." if result.passed else "F"
        print(f"\n{tests[i-1].description} ... {status}")

        if not result.passed or result.evidence:
            print(f"  {result.message}")

            if result.evidence:
                print("  Evidence:")
                for evidence in result.evidence:
                    print(f"    - {evidence.description}")
                    print(f"      Type: {evidence.type}")
                    if evidence.data:
                        print(f"      Data: {evidence.data}")

    print("\n" + "=" * 80)
    print(f"{passed_count} passed, {failed_count} failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
