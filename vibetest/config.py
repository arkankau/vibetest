"""Configuration management for vibetest."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file if it exists
load_dotenv()


class VibeTestConfig(BaseModel):
    """Global configuration for vibetest."""

    # Model configuration
    default_model: str = Field(
        default_factory=lambda: os.getenv(
            "VIBETEST_MODEL", "anthropic/claude-3-5-sonnet-20241022"
        )
    )
    max_attempts: int = Field(default=20, description="Max ReAct iterations")

    # Execution configuration
    default_sandbox: str = Field(default="docker", description="Default sandbox type")
    execution_timeout: int = Field(default=600, description="Execution timeout in seconds")

    # Storage configuration
    evidence_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("VIBETEST_EVIDENCE_DIR", "./evidence")
        )
    )
    log_dir: Path = Field(default=Path("./logs"))

    # API configuration
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    class Config:
        arbitrary_types_allowed = True

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: VibeTestConfig | None = None


def get_config() -> VibeTestConfig:
    """Get the global configuration instance.

    Returns:
        Global VibeTestConfig instance
    """
    global _config
    if _config is None:
        _config = VibeTestConfig()
        _config.ensure_directories()
    return _config


def set_config(**kwargs: Any) -> None:
    """Update global configuration.

    Args:
        **kwargs: Configuration parameters to update
    """
    global _config
    if _config is None:
        _config = VibeTestConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(_config, key):
                setattr(_config, key, value)
    _config.ensure_directories()
