"""Static baselines (non-LLM) for vibetest."""

from .codeql import analyze_repo_with_codeql

__all__ = ["analyze_repo_with_codeql"]
