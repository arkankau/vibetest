"""Public Meerkat trace-auditing package."""

from __future__ import annotations

__version__ = "0.1.0"

from meerkat.testcases.base import Evidence, TestCase, TestResult

__all__ = ["Evidence", "MeerkatAgent", "TestCase", "TestResult", "prepare_search_aids"]


def __getattr__(name: str):
    if name == "MeerkatAgent":
        from meerkat.agent.react_agent import MeerkatAgent

        return MeerkatAgent
    if name == "prepare_search_aids":
        from meerkat.search_aids import prepare_search_aids

        return prepare_search_aids
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
