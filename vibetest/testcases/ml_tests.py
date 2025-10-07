"""ML-specific test case helpers.

These are convenience functions for creating common ML test cases.
"""

from pathlib import Path

from vibetest.testcases.base import TestCase


def training_loss_test(
    repo_path: Path,
    logging_interval: int = 100,
    expected_decrease: bool = True,
) -> TestCase:
    """Create a test case for training loss logging and decrease.

    Args:
        repo_path: Path to repository to test
        logging_interval: Expected logging interval (K steps)
        expected_decrease: Whether loss should decrease

    Returns:
        TestCase instance
    """
    description = (
        f"Training loss is logged every {logging_interval} steps and "
        f"{'generally decreases' if expected_decrease else 'is tracked properly'}"
    )

    return TestCase(
        description=description,
        repo_path=repo_path,
        metadata={
            "test_type": "training_loss",
            "logging_interval": logging_interval,
            "expected_decrease": expected_decrease,
        },
    )


def hyperparameter_logging_test(
    repo_path: Path,
    required_params: list[str] | None = None,
) -> TestCase:
    """Create a test case for hyperparameter logging.

    Args:
        repo_path: Path to repository to test
        required_params: List of required hyperparameter names

    Returns:
        TestCase instance
    """
    params = required_params or ["learning_rate", "batch_size", "num_epochs"]
    params_str = ", ".join(params)

    description = f"Hyperparameters ({params_str}) are logged at the start of training"

    return TestCase(
        description=description,
        repo_path=repo_path,
        metadata={
            "test_type": "hyperparameter_logging",
            "required_params": params,
        },
    )


def model_checkpointing_test(
    repo_path: Path,
    checkpoint_interval: int = 1000,
) -> TestCase:
    """Create a test case for model checkpointing.

    Args:
        repo_path: Path to repository to test
        checkpoint_interval: Expected checkpointing interval

    Returns:
        TestCase instance
    """
    description = f"Model checkpoints are saved every {checkpoint_interval} steps"

    return TestCase(
        description=description,
        repo_path=repo_path,
        metadata={
            "test_type": "model_checkpointing",
            "checkpoint_interval": checkpoint_interval,
        },
    )


def gradient_clipping_test(repo_path: Path) -> TestCase:
    """Create a test case for gradient clipping.

    Args:
        repo_path: Path to repository to test

    Returns:
        TestCase instance
    """
    description = "Gradient clipping is implemented to prevent exploding gradients"

    return TestCase(
        description=description,
        repo_path=repo_path,
        metadata={"test_type": "gradient_clipping"},
    )
