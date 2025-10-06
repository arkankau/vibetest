"""Simple example of using vibetest programmatically."""

import asyncio
from pathlib import Path

from vibetest import TestCase, VibeTestAgent
from vibetest.testcases.ml_tests import training_loss_test


async def main():
    """Run a simple test example."""
    # Path to the repository you want to test
    repo_path = Path("./example_ml_repo")

    # Create a test case - just a string description!
    # Option 1: Use a helper function
    test = training_loss_test(
        repo_path=repo_path,
        logging_interval=100,
        expected_decrease=True,
    )

    # Option 2: Create directly with just a description
    # test = TestCase(
    #     description="Training loss is logged every 100 steps and generally decreases",
    #     repo_path=repo_path
    # )

    # Create agent and run test
    print(f"Testing: {test.description}")
    print(f"Repository: {repo_path}\n")

    agent = VibeTestAgent(
        max_attempts=20,
    )

    # Execute with Docker sandbox
    result = await agent.execute_test(test, sandbox="docker")

    # Print results
    print("\n" + "=" * 80)
    print(f"Test Result: {'PASSED' if result.passed else 'FAILED'}")
    print("=" * 80)
    print(f"\n{result.message}\n")

    if result.evidence:
        print("Evidence:")
        for evidence in result.evidence:
            print(f"  - {evidence.description}")
            print(f"    Type: {evidence.type}")
            print(f"    Data: {evidence.data}\n")


if __name__ == "__main__":
    asyncio.run(main())
