"""
Build a lightweight JSON payload for the viewer from one or more result JSONL files.

Use this when the original `.eval` archives are unavailable but per-repo result
JSONLs (e.g. `results/kaggle_nlp_AT-gpt-5-mini.jsonl`) are. Each JSONL row is
expected to have the per-repo schema produced by
`experiments/kaggle.py::run_vibetest`:

    {
      "repo": "data/kaggle/kaggle-nlp/<repo_name>",
      "repo_name": "<repo_name>",
      "total_tests": <int>,
      "passed_tests": <int>,
      "failed_tests": <int>,
      "tests": [
        {
          "description": "VERDICT: ...\nREASON: ...\nEVIDENCE: ...",
          "passed": <bool>,
          "evidence": [...],
          "execution_log": "...",
          "metadata": {"test_description": "...", "model": "...", ...}
        },
        ...
      ]
    }

Usage:
    python scripts/build_viewer_data_from_jsonl.py \
        --jsonl results/kaggle_diabetic_AT-gpt-5-mini.jsonl \
        --jsonl results/kaggle_nlp_AT-gpt-5-mini.jsonl \
        --jsonl results/kaggle_titanic_AT-gpt-5-mini.jsonl \
        --output viewer/eval-results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ATTACHMENT_PATTERN = re.compile(r"evidence-[\w.-]+\.tar\.gz", re.IGNORECASE)
VERDICT_PATTERN = re.compile(r"VERDICT\s*:\s*(.+)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^(VERDICT|REASON|EVIDENCE)\s*:\s*(.*)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        dest="jsonl_files",
        required=True,
        help="Path to a result JSONL file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the aggregated JSON payload.",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default="/kaggle/repo",
        help="Sandbox repo path to record in test metadata (default: /kaggle/repo).",
    )
    return parser.parse_args()


def parse_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {"verdict": [], "reason": [], "evidence": []}
    current: Optional[str] = None
    for line in text.splitlines():
        match = SECTION_PATTERN.match(line.strip())
        if match:
            current = match.group(1).upper().lower()
            remainder = match.group(2).strip()
            if remainder:
                sections[current].append(remainder)
            continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def verdict_to_bool(verdict_text: str) -> bool:
    return verdict_text.upper().startswith("PASS")


def get_model_folder(model_name: str) -> str:
    """Convert model name to evidence folder name (mirrors build_viewer_data.py)."""
    name = model_name or ""
    if name.startswith("openai/"):
        name = name[len("openai/"):]
    if name.startswith("gpt-5-mini"):
        return "gpt-5-mini"
    if name.startswith("gpt-5-nano"):
        return "gpt-5-nano"
    if name.startswith("gpt-5.2"):
        return "gpt-5.2"
    if name.startswith("gpt-5"):
        return "gpt-5"
    return "gpt-5"


def extract_evidence_bundles_from_test(
    test: Dict[str, Any],
    *,
    fallback_sample_id: str,
    model_folder: str,
) -> List[str]:
    """Pull evidence-*.tar.gz attachments mentioned anywhere in the test record.

    Falls back to the canonical evidence-dumps path keyed off sample_id when the
    test record doesn't surface an explicit attachment reference.
    """
    bundles: set[str] = set()

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

    scan(test.get("description"))
    scan(test.get("execution_log"))
    scan(test.get("metadata"))
    scan(test.get("evidence"))

    if not bundles and fallback_sample_id:
        bundles.add(f"evidence-{fallback_sample_id}.tar.gz")

    return [f"evidence-dumps/{model_folder}/{name}" for name in sorted(bundles)]


def log_name_from_jsonl(path: Path) -> str:
    """`results/kaggle_nlp_AT-gpt-5-mini.jsonl` -> `kaggle-nlp-gpt-5-mini`."""
    stem = path.stem
    stem = stem.replace("_AT-", "-").replace("_AT_", "-")
    stem = stem.replace("_", "-")
    return stem


def normalize_repo_entry(
    repo_row: Dict[str, Any],
    *,
    eval_source: str,
    sandbox_repo_path: str,
) -> Dict[str, Any]:
    repo_name = (
        repo_row.get("repo_name")
        or Path(str(repo_row.get("repo") or "unknown-repo")).name
        or "unknown-repo"
    )

    out_tests: List[Dict[str, Any]] = []
    passed_count = 0
    failed_count = 0

    raw_tests = repo_row.get("tests") or []
    for idx, test in enumerate(raw_tests):
        description = test.get("description") or ""
        sections = parse_sections(description)
        verdict_line = sections.get("verdict", "")
        verdict_match = VERDICT_PATTERN.search(verdict_line) if verdict_line else None
        verdict_value = verdict_match.group(1).strip() if verdict_match else verdict_line
        passed = bool(test.get("passed")) if test.get("passed") is not None else verdict_to_bool(verdict_value)

        metadata = dict(test.get("metadata") or {})
        model_name = metadata.get("model") or "openai/gpt-5-mini"
        model_folder = get_model_folder(model_name)
        sample_id = metadata.get("sample_id") or f"{repo_name}_prop{idx}"

        evidence_bundles = extract_evidence_bundles_from_test(
            test,
            fallback_sample_id=sample_id,
            model_folder=model_folder,
        )

        test_payload = {
            "description": description,
            "sections": {
                "verdict": sections.get("verdict", ""),
                "reason": sections.get("reason", ""),
                "evidence": sections.get("evidence", ""),
            },
            "passed": passed,
            "metadata": {
                "test_description": metadata.get("test_description", ""),
                "sample_id": sample_id,
                "prop_index": idx,
                "repo_path": sandbox_repo_path,
                "model": model_name,
                "eval_source": eval_source,
            },
            "evidenceBundles": evidence_bundles,
        }
        out_tests.append(test_payload)

        if passed:
            passed_count += 1
        else:
            failed_count += 1

    return {
        "repo": repo_name,
        "repo_name": repo_name,
        "repo_slug": repo_name,
        "tests": out_tests,
        "total_tests": len(out_tests),
        "passed_tests": passed_count,
        "failed_tests": failed_count,
    }


def load_repo_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {jsonl_path}: {exc}") from exc
            if not isinstance(obj, dict) or "tests" not in obj:
                continue
            rows.append(obj)
    return rows


def build_log_entry(jsonl_path: Path, sandbox_repo_path: str) -> Dict[str, Any]:
    eval_name = log_name_from_jsonl(jsonl_path)
    repo_rows = load_repo_rows(jsonl_path)
    repos = [
        normalize_repo_entry(
            row,
            eval_source=eval_name,
            sandbox_repo_path=sandbox_repo_path,
        )
        for row in repo_rows
    ]
    repos.sort(key=lambda r: (r.get("repo_name") or r.get("repo") or "").lower())
    return {
        "name": eval_name,
        "path": f"logs/{eval_name}.eval",
        "repos": repos,
    }


def main() -> None:
    args = parse_args()

    jsonl_files = args.jsonl_files or []
    missing = [p for p in jsonl_files if not p.exists()]
    if missing:
        print(f"Warning: The following JSONL files were not found: {missing}")

    existing = [p for p in jsonl_files if p.exists()]
    if not existing:
        raise SystemExit("No valid JSONL files found.")

    logs = [build_log_entry(path, args.repo_path) for path in existing]
    payload = {"logs": logs}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    total_repos = sum(len(log["repos"]) for log in logs)
    total_tests = sum(
        len(repo["tests"]) for log in logs for repo in log["repos"]
    )
    print(
        f"Wrote {args.output} ({len(logs)} logs, {total_repos} total repos, "
        f"{total_tests} total tests)"
    )
    for log in logs:
        repo_counts = {len(r["tests"]) for r in log["repos"]}
        print(
            f"  - {log['name']}: {len(log['repos'])} repos, "
            f"tests-per-repo={sorted(repo_counts)}"
        )


if __name__ == "__main__":
    main()
