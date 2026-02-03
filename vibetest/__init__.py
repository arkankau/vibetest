"""Vibetest: AI agent for executing natural language test cases over arbitrary codebases."""

from __future__ import annotations

__version__ = "0.1.0"

from vibetest.testcases.base import Evidence, TestCase, TestResult

__all__ = ["TestCase", "TestResult", "Evidence", "VibeTestAgent"]


def __getattr__(name: str):
    if name == "VibeTestAgent":
        from vibetest.agent.react_agent import VibeTestAgent

        return VibeTestAgent
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
