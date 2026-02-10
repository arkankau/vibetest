"""
Generate viewer/repo-paths.json so citation links resolve to local checkouts.

Usage:
    python scripts/build_repo_paths.py
    python scripts/build_repo_paths.py --path data/kaggle/kaggle-titanic:2 --path data/iclr-26/iclr2026_filter2:1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_SPECS = [
    ("data/kaggle/kaggle-titanic", 2),  # owner/repo
    ("data/kaggle/kaggle-diabetic", 2),  # owner/repo
    ("data/kaggle/kaggle-nlp", 2),  # owner/repo
    ("data/vuln/bibifi/repos", 1),  # repo
    ("data/vuln/cwe-bench/repos", 1),  # vuln_id_repo
    ("data/hallucination/arxiv_downloads", 1),  # paper folder
    ("data/iclr-26/iclr2026_filter2", 1),  # repo
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Root specification in the form <path>:<depth>. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("viewer/repo-paths.json"),
        help="Where to write the generated JSON.",
    )
    return parser.parse_args()


def parse_specs(raw_specs: Iterable[str]) -> List[Tuple[Path, int]]:
    specs: List[Tuple[Path, int]] = []
    if not raw_specs:
        specs = [(Path(path), depth) for path, depth in DEFAULT_SPECS]
    else:
        for spec in raw_specs:
            try:
                path_str, depth_str = spec.split(":", 1)
            except ValueError as exc:
                raise SystemExit(f"Invalid --path value '{spec}'. Expected <path>:<depth>.") from exc
            path = Path(path_str)
            try:
                depth = int(depth_str)
            except ValueError as exc:
                raise SystemExit(f"Invalid depth in '{spec}'. Must be an integer.") from exc
            specs.append((path, depth))
    return specs


def collect_repos(root: Path, depth: int) -> Iterable[Path]:
    if depth <= 0:
        return []
    if not root.exists():
        return []
    if depth == 1:
        return (child for child in root.iterdir() if child.is_dir())
    if depth == 2:
        repos = []
        for owner in root.iterdir():
            if not owner.is_dir():
                continue
            for repo_dir in owner.iterdir():
                if repo_dir.is_dir():
                    repos.append(repo_dir)
        return repos
    raise SystemExit(f"Unsupported depth {depth} for root {root}")


def main() -> None:
    args = parse_args()
    specs = parse_specs(args.path)
    repo_map: Dict[str, List[str]] = {}
    for root, depth in specs:
        for repo_dir in collect_repos(root, depth):
            repo_map.setdefault(repo_dir.name, []).append(str(repo_dir))

            # Alias cwe-bench-style "<id>_<project_slug>" entries by project_slug
            # so sample IDs that omit the numeric prefix can still resolve.
            if "_" in repo_dir.name:
                suffix = repo_dir.name.split("_", 1)[1]
                if suffix:
                    repo_map.setdefault(suffix, []).append(str(repo_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(repo_map, indent=2))
    print(f"Wrote {args.output} with {len(repo_map)} entries.")


if __name__ == "__main__":
    main()
