"""Configuration management for Meerkat."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from current working directory only.
load_dotenv(Path.cwd() / ".env")


def get_package_root() -> Path:
    """Return the root directory of the installed Meerkat package."""
    return Path(__file__).parent.parent


class MeerkatConfig(BaseModel):
    """Global configuration for Meerkat."""

    default_model: str = Field(default_factory=lambda: os.getenv("MEERKAT_MODEL", "no-model"))
    max_attempts: int = Field(default=20, description="Max ReAct iterations")
    default_sandbox: str = Field(default="docker", description="Default sandbox type")
    execution_timeout: int = Field(default=600, description="Execution timeout in seconds")
    evidence_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MEERKAT_EVIDENCE_DIR", "./evidence"))
    )
    log_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MEERKAT_LOG_DIR", "./logs"))
    )
    anthropic_api_key: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    class Config:
        arbitrary_types_allowed = True

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


_config: MeerkatConfig | None = None


def get_config() -> MeerkatConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = MeerkatConfig()
        _config.ensure_directories()
    return _config


def set_config(**kwargs: Any) -> None:
    """Update global configuration."""
    global _config
    if _config is None:
        _config = MeerkatConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(_config, key):
                setattr(_config, key, value)
    _config.ensure_directories()
