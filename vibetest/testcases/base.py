"""Base classes for test cases and results."""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Types of evidence that can be collected."""

    PLOT = "plot"
    LOG = "log"
    METRICS = "metrics"
    CODE_SNIPPET = "code_snippet"
    SCREENSHOT = "screenshot"
    CUSTOM = "custom"


class Evidence(BaseModel):
    """Evidence collected during test execution."""

    type: EvidenceType
    description: str
    data: Any = Field(..., description="Evidence data (file path, metrics dict, etc.)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class TestCase(BaseModel):
    """A test case specification.

    This is simply a container for test metadata - all execution logic
    is handled by the VibeTestAgent.
    """

    description: str = Field(..., description="Natural language description of what to test")
    repo_path: Path = Field(..., description="Path to the repository to test")
    sandbox_path: str = Field(
        default="/workspace", description="Path inside the sandbox where the repo will be placed"
    )
    target: Optional[str] = Field(
        default=None, description="Ground truth to compare model outputs against"
    )
    # key is the path to the additional data file/directory and the value is the path to be used inside the sandbox
    additional_data: Optional[dict[str, str]] = Field(
        default_factory=dict, description="Additional data files to include in the sandbox"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the test")

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"TestCase(description='{self.description}')"


class TestResult(BaseModel):
    """Result of executing a test case."""

    passed: bool
    test_case: TestCase
    message: str
    evidence: list[Evidence] = Field(default_factory=list)
    execution_log: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the result."""
        self.evidence.append(evidence)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return self.model_dump()