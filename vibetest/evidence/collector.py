"""Evidence collection utilities."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from vibetest.testcases.base import Evidence, EvidenceType


class EvidenceCollector:
    """Manages collection and storage of test evidence."""

    def __init__(self, output_dir: Path | str):
        """Initialize evidence collector.

        Args:
            output_dir: Directory to store evidence
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_items: list[Evidence] = []

    def add_plot(
        self, plot_path: Path | str, description: str, metadata: dict[str, Any] | None = None
    ) -> Evidence:
        """Add a plot as evidence.

        Args:
            plot_path: Path to the plot file
            description: Description of what the plot shows
            metadata: Additional metadata

        Returns:
            Evidence object
        """
        # Copy plot to evidence directory
        plot_path = Path(plot_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evidence_filename = f"plot_{timestamp}_{plot_path.name}"
        evidence_path = self.output_dir / evidence_filename

        shutil.copy(plot_path, evidence_path)

        evidence = Evidence(
            type=EvidenceType.PLOT,
            description=description,
            data=str(evidence_path),
            metadata=metadata or {},
        )
        self.evidence_items.append(evidence)
        return evidence

    def add_log(
        self, log_content: str, description: str, metadata: dict[str, Any] | None = None
    ) -> Evidence:
        """Add log content as evidence.

        Args:
            log_content: Log text
            description: Description of the log
            metadata: Additional metadata

        Returns:
            Evidence object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"log_{timestamp}.txt"
        log_path = self.output_dir / log_filename

        log_path.write_text(log_content)

        evidence = Evidence(
            type=EvidenceType.LOG,
            description=description,
            data=str(log_path),
            metadata=metadata or {},
        )
        self.evidence_items.append(evidence)
        return evidence

    def add_metrics(
        self,
        metrics: dict[str, Any],
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        """Add metrics as evidence.

        Args:
            metrics: Dictionary of metrics
            description: Description of metrics
            metadata: Additional metadata

        Returns:
            Evidence object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_filename = f"metrics_{timestamp}.json"
        metrics_path = self.output_dir / metrics_filename

        metrics_path.write_text(json.dumps(metrics, indent=2))

        evidence = Evidence(
            type=EvidenceType.METRICS,
            description=description,
            data=metrics,
            metadata=metadata or {},
        )
        self.evidence_items.append(evidence)
        return evidence

    def add_code_snippet(
        self,
        code: str,
        file_path: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        """Add code snippet as evidence.

        Args:
            code: Code snippet
            file_path: Original file path
            description: Description of why this code is relevant
            metadata: Additional metadata

        Returns:
            Evidence object
        """
        evidence = Evidence(
            type=EvidenceType.CODE_SNIPPET,
            description=description,
            data={"code": code, "file_path": file_path},
            metadata=metadata or {},
        )
        self.evidence_items.append(evidence)
        return evidence

    def get_all_evidence(self) -> list[Evidence]:
        """Get all collected evidence.

        Returns:
            List of all evidence items
        """
        return self.evidence_items

    def save_manifest(self, filename: str = "evidence_manifest.json") -> Path:
        """Save a manifest of all evidence.

        Args:
            filename: Name for the manifest file

        Returns:
            Path to the manifest file
        """
        manifest_path = self.output_dir / filename
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "evidence_count": len(self.evidence_items),
            "evidence": [e.model_dump() for e in self.evidence_items],
        }

        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        return manifest_path
