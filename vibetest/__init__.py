"""Public Meerkat trace-auditing package."""

from __future__ import annotations

__version__ = "0.1.0"

from vibetest.testcases.base import Evidence, TestCase, TestResult

__all__ = ["TestCase", "TestResult", "Evidence", "VibeTestAgent", "MeerkatAgent"]


def __getattr__(name: str):
    if name in {"VibeTestAgent", "MeerkatAgent"}:
        from vibetest.agent.react_agent import VibeTestAgent

        return VibeTestAgent
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
