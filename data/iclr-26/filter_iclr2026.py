#!/usr/bin/env python3
"""
Filter ICLR 2026 submissions based on supplementary material contents.

Keeps submissions whose supplementary archive contains:
  * at least one README file (case-insensitive, any extension),
  * at least one Python file,
  * Python code that imports PyTorch, TensorFlow, JAX, or Keras.

Matching submissions are symlinked into the target directory.
"""
from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from pathlib import Path
from typing import Iterable
import warnings
import zipfile


FRAMEWORK_ROOT_MODULES = ("torch", "tensorflow", "jax", "keras")


def decode_bytes(data: bytes) -> str:
    """Decode raw bytes to text, falling back to latin-1 on errors."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore") or data.decode("latin-1", errors="ignore")


def python_file_uses_framework(source: str) -> bool:
    """Return True if the python source imports one of the target frameworks."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        lowered = source.lower()
        return any(f"import {name}" in lowered or f"from {name}" in lowered for name in FRAMEWORK_ROOT_MODULES)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FRAMEWORK_ROOT_MODULES:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".", 1)[0]
                if root in FRAMEWORK_ROOT_MODULES:
                    return True
    return False


def zip_has_required_content(zip_path: Path) -> bool:
    """Check whether the zip file satisfies README + python + framework usage requirements."""
    has_readme = False
    python_members: list[zipfile.ZipInfo] = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                filename = Path(info.filename)
                lowered_name = filename.name.lower()
                if lowered_name.startswith("readme"):
                    has_readme = True
                if lowered_name.endswith(".py"):
                    python_members.append(info)

            if not has_readme or not python_members:
                return False

            for info in python_members:
                try:
                    with zf.open(info, "r") as fh:
                        content = decode_bytes(fh.read())
                except (KeyError, RuntimeError, OSError):
                    continue

                if python_file_uses_framework(content):
                    return True
    except zipfile.BadZipFile:
        logging.warning("Failed to read zip file %s", zip_path)
        return False

    return False


def submission_has_qualifying_zip(submission_dir: Path) -> bool:
    """Return True if any zip file in the submission directory qualifies."""
    zip_paths = [p for p in submission_dir.iterdir() if p.suffix.lower() == ".zip"]
    if not zip_paths:
        logging.debug("No zip files found for %s", submission_dir)
        return False

    for archive in zip_paths:
        if zip_has_required_content(archive):
            return True

    return False


def iter_submission_dirs(download_root: Path) -> Iterable[Path]:
    for entry in sorted(download_root.iterdir()):
        if entry.is_dir():
            yield entry


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()


def create_symlinks(submission_dirs: Iterable[Path], output_dir: Path) -> int:
    count = 0
    for submission in submission_dirs:
        target = output_dir / submission.name
        if target.exists():
            target.unlink()
        target.symlink_to(submission.resolve())
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter ICLR submissions based on supplementary Python code quality signals."
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("iclr2026_downloads"),
        help="Directory containing downloaded submissions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iclr2026_filter1"),
        help="Directory where qualifying submission symlinks will be created.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar output.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    downloads_dir: Path = args.downloads_dir
    output_dir: Path = args.output_dir

    if not downloads_dir.exists():
        raise SystemExit(f"Downloads directory not found: {downloads_dir}")

    prepare_output_dir(output_dir)

    submission_dirs = list(iter_submission_dirs(downloads_dir))
    total = len(submission_dirs)
    show_progress = not args.no_progress and sys.stderr.isatty() and total > 0

    qualifying_dirs = []
    for idx, submission_dir in enumerate(submission_dirs, start=1):
        if show_progress:
            completed = idx / total
            bar_width = 30
            filled = int(bar_width * completed)
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stderr.write(
                f"\r[{bar}] {idx}/{total} {submission_dir.name[:60]}"
            )
            sys.stderr.flush()

        if submission_has_qualifying_zip(submission_dir):
            qualifying_dirs.append(submission_dir)
            logging.debug("Accepted: %s", submission_dir.name)
        else:
            logging.debug("Filtered out: %s", submission_dir.name)

    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    count = create_symlinks(qualifying_dirs, output_dir)
    logging.info("Created %d symlinks in %s", count, output_dir)


if __name__ == "__main__":
    main()
