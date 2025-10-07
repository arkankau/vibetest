"""Command-line interface for vibetest."""

import json
import sys
from pathlib import Path

from vibetest import TestCase, VibeTestAgent
from vibetest.testcases import ml_tests


def run_test(
    test_case: TestCase, output_dir: Path, use_sandbox: bool = True
) -> None:
    """Run a single test case.

    Args:
        test_case: Test case to run
        output_dir: Directory for outputs
        use_sandbox: Whether to use Docker sandbox
    """
    print(f"\n{'='*80}")
    print(f"Running test: {test_case.description}")
    print(f"Repository: {test_case.repo_path}")
    print(f"{'='*80}\n")

    # Execute test (synchronous - Inspect AI manages its own event loop)
    agent = VibeTestAgent()
    sandbox = "docker" if use_sandbox else None
    tests = [test_case]
    results = agent.execute_tests(tests, sandbox=sandbox)

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

    # Save result
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "test_result.json"
    result_file.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    print(f"\nFull results saved to: {result_file}")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vibetest: AI agent for natural language test execution"
    )
    parser.add_argument(
        "repo_path", type=Path, help="Path to repository to test"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Test description or preset name (training_loss, hyperparams, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./vibetest_output"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable Docker sandbox",
    )
    parser.add_argument(
        "--logging-interval",
        type=int,
        default=100,
        help="Expected logging interval (for training_loss test)",
    )

    args = parser.parse_args()

    # Create test case based on preset or custom description
    if args.test in ["training_loss", "loss"]:
        test_case = ml_tests.training_loss_test(
            repo_path=args.repo_path,
            logging_interval=args.logging_interval,
        )
    elif args.test in ["hyperparams", "hyperparameters"]:
        test_case = ml_tests.hyperparameter_logging_test(repo_path=args.repo_path)
    elif args.test in ["checkpointing", "checkpoint"]:
        test_case = ml_tests.model_checkpointing_test(repo_path=args.repo_path)
    elif args.test in ["gradient_clipping", "grad_clip"]:
        test_case = ml_tests.gradient_clipping_test(repo_path=args.repo_path)
    elif args.test:
        # Treat as custom description
        test_case = TestCase(
            description=args.test,
            repo_path=args.repo_path,
        )
    else:
        print("Error: --test is required")
        print("\nPreset tests: training_loss, hyperparams, checkpointing, gradient_clipping")
        print("Or provide any custom test description in quotes")
        sys.exit(1)

    # Run test
    try:
        run_test(test_case, args.output_dir, not args.no_sandbox)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
