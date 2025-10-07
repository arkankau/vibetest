"""Example of creating a custom test case."""

from pathlib import Path

from vibetest import TestCase, VibeTestAgent


def main():
    """Run a custom test."""
    repo_path = Path("./example_ml_repo")

    # Creating a custom test is now just a string description!
    test = TestCase(
        description="Model uses gradient checkpointing for memory efficiency",
        repo_path=repo_path,
        metadata={"test_type": "custom", "focus": "memory_optimization"},
    )

    # Or test anything you want with natural language:
    # test = TestCase(
    #     description="The model architecture uses attention mechanisms",
    #     repo_path=repo_path
    # )
    #
    # test = TestCase(
    #     description="Training uses mixed precision (fp16 or bf16)",
    #     repo_path=repo_path
    # )
    #
    # test = TestCase(
    #     description="Data augmentation is applied during training",
    #     repo_path=repo_path
    # )

    agent = VibeTestAgent()

    print(f"Running custom test: {test.description}\n")

    # Note: execute_test() is synchronous - Inspect AI manages its own event loop
    result = agent.execute_test(test, sandbox="docker")

    print(f"\nResult: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Message: {result.message}")


if __name__ == "__main__":
    main()
