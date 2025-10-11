"""Command-line interface for vibetest."""

import importlib.util
import json
import sys
from pathlib import Path

from vibetest import TestCase, VibeTestAgent


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


def load_tests_from_file(file_path: Path) -> list[TestCase]:
    """Load test cases from a vibetest.py file.

    Args:
        file_path: Path to vibetest.py file

    Returns:
        List of TestCase objects

    The file should define a variable named 'tests' which is a list of TestCase objects.
    """
    spec = importlib.util.spec_from_file_location("vibetest_config", file_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load {file_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "tests"):
        print(f"Error: {file_path} must define a 'tests' variable containing a list of TestCase objects")
        sys.exit(1)

    tests = module.tests
    if not isinstance(tests, list):
        print(f"Error: 'tests' must be a list of TestCase objects")
        sys.exit(1)

    return tests


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vibetest: AI agent for natural language test execution",
        epilog="""
Examples:
  # Run a test with a natural language description
  vibetest --test "Training loss decreases during training"

  # Run tests defined in vibetest.py
  vibetest

  # Specify repo path (defaults to current directory)
  vibetest --repo /path/to/repo --test "Model saves checkpoints"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to repository to test (default: current directory)",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Natural language test description",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("vibetest.py"),
        help="Path to test configuration file (default: ./vibetest.py)",
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

    args = parser.parse_args()

    # Determine test cases to run
    test_cases = []

    if args.test:
        # Use test description from command line
        test_cases = [TestCase(
            description=args.test,
            repo_path=args.repo,
        )]
    elif args.config.exists():
        # Load tests from config file
        print(f"Loading tests from {args.config}")
        test_cases = load_tests_from_file(args.config)
    else:
        print(f"Error: No test specified and {args.config} not found")
        print("\nUsage:")
        print("  1. Provide --test with a natural language description")
        print("  2. Create a vibetest.py file with test definitions")
        print("\nExample vibetest.py:")
        print("  from pathlib import Path")
        print("  from vibetest import TestCase")
        print("")
        print("  tests = [")
        print('      TestCase(description="Training loss decreases", repo_path=Path(".")),')
        print('      TestCase(description="Model saves checkpoints", repo_path=Path(".")),')
        print("  ]")
        sys.exit(1)

    # Run tests
    try:
        for test_case in test_cases:
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
