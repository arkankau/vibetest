"""
Build a lightweight JSON payload for the viewer from a .eval log archive.

Usage:
    python scripts/build_viewer_data.py \
        --eval logs/2025-11-10T19-02-58-05-00_task_oAYv8tQDiuxozNmQZWxK4j.eval \
        --output viewer/eval-results.json
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ATTACHMENT_PATTERN = re.compile(r"evidence-[\w.-]+\.tar\.gz", re.IGNORECASE)
VERDICT_PATTERN = re.compile(r"VERDICT\s*:\s*(.+)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^(VERDICT|REASON|EVIDENCE)\s*:\s*(.*)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        required=True,
        help="Path to the .eval zip archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the aggregated JSON payload.",
    )
    return parser.parse_args()


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        return "\n".join(flatten_message_content(item) for item in content)
    return ""


def parse_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {"verdict": [], "reason": [], "evidence": []}
    current: Optional[str] = None
    for line in text.splitlines():
        match = SECTION_PATTERN.match(line.strip())
        if match:
            current_key = match.group(1).upper()
            current = current_key.lower()
            remainder = match.group(2).strip()
            if remainder:
                sections[current].append(remainder)
            continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def extract_sections_from_messages(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    for message in reversed(messages):
        text = flatten_message_content(message.get("content"))
        if "VERDICT" not in text:
            continue
        sections = parse_sections(text)
        sections["fullText"] = text
        return sections
    return {"verdict": "", "reason": "", "evidence": "", "fullText": ""}


def extract_test_description(input_text: str) -> str:
    if not input_text:
        return ""
    match = re.search(r"Test:\s*([\s\S]*?)(?:\nRepository:|\Z)", input_text)
    return match.group(1).strip() if match else ""


def extract_repo_path(input_text: str) -> str:
    if not input_text:
        return ""
    match = re.search(r"Repository:\s*(.*)(?:\n|\Z)", input_text)
    return match.group(1).strip() if match else ""


def extract_evidence_bundles(sample: Dict[str, Any]) -> List[str]:
    bundles = set()

    def scan(value: Any) -> None:
        if isinstance(value, str):
            for found in ATTACHMENT_PATTERN.findall(value):
                bundles.add(found)
        elif isinstance(value, dict):
            for inner in value.values():
                scan(inner)
        elif isinstance(value, list):
            for inner in value:
                scan(inner)

    scan(sample.get("events", []))
    scan(sample.get("attachments", {}))
    return [f"evidence-dumps/gpt-5/{name}" for name in sorted(bundles)]


def verdict_to_bool(verdict_text: str) -> bool:
    return verdict_text.upper().startswith("PASS")


def load_samples(eval_path: Path) -> Iterable[Dict[str, Any]]:
    with zipfile.ZipFile(eval_path, "r") as archive:
        for name in archive.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            with archive.open(name) as handle:
                yield json.load(handle)


def aggregate_results(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    repos: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        sample_id = sample.get("id", "unknown-id")
        repo_slug = sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id
        repo_path = extract_repo_path(sample.get("input", ""))
        repo_key = repo_path or repo_slug
        repo_name = repo_slug or repo_path or "Unknown Repository"
        sections = extract_sections_from_messages(sample.get("messages", []))
        verdict_line = sections.get("verdict", "")
        verdict_match = VERDICT_PATTERN.search(verdict_line)
        verdict_value = verdict_match.group(1).strip() if verdict_match else verdict_line
        passed = verdict_to_bool(verdict_value)
        evidence_bundles = extract_evidence_bundles(sample)
        test_description = extract_test_description(sample.get("input", ""))

        if repo_key not in repos:
            repos[repo_key] = {
                "repo": repo_path or repo_slug or repo_key,
                "repo_name": repo_name,
                "repo_slug": repo_slug,
                "tests": [],
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
            }

        test_payload = {
            "description": sections.get("fullText", ""),
            "sections": {
                "verdict": sections.get("verdict", ""),
                "reason": sections.get("reason", ""),
                "evidence": sections.get("evidence", ""),
            },
            "passed": passed,
            "metadata": {
                "test_description": test_description,
                "sample_id": sample_id,
                "repo_path": repo_path,
                "model": sample.get("output", {}).get("model", "unknown"),
            },
            "evidenceBundles": evidence_bundles,
        }

        repo_entry = repos[repo_key]
        repo_entry["tests"].append(test_payload)
        repo_entry["total_tests"] += 1
        if passed:
            repo_entry["passed_tests"] += 1
        else:
            repo_entry["failed_tests"] += 1

    ordered = sorted(repos.values(), key=lambda item: item.get("repo_name") or item.get("repo"))
    return {"repos": ordered}


def main() -> None:
    args = parse_args()
    if not args.eval.exists():
        raise SystemExit(f"Eval archive not found: {args.eval}")
    data = aggregate_results(load_samples(args.eval))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.output} ({len(data['repos'])} repos)")


if __name__ == "__main__":
    main()
