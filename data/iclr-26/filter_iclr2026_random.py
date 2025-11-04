#!/usr/bin/env python3
"""
Randomly select ICLR submissions and mirror them via symbolic links.

The script samples a fixed number of submissions from an input directory
(default: iclr2026_filter1) and creates symbolic links to those submissions
inside the output directory (default: iclr2026_filter2).
"""
from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List


def iter_submission_dirs(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
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
        description="Randomly sample ICLR submissions and create symlinks for the selection."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("iclr2026_filter1"),
        help="Directory containing candidate submissions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iclr2026_filter2"),
        help="Directory where sampled submission symlinks will be created.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of submissions to sample (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--supplementary-dirname",
        default="supplementary",
        help="Directory name where supplementary files will be extracted inside each sample.",
    )
    return parser.parse_args()


def extract_supplementary(submission_dir: Path, dest_dirname: str) -> int:
    dest_root = submission_dir / dest_dirname
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(submission_dir.glob("*.zip"))
    if not zip_files:
        zip_files = sorted(submission_dir.rglob("*.zip"))
    extracted = 0
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(dest_root)
            extracted += 1
        except zipfile.BadZipFile:
            continue
    return extracted


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    submissions: List[Path] = list(iter_submission_dirs(args.input_dir))
    if not submissions:
        raise SystemExit(f"No submission directories found in {args.input_dir}")

    desired_count = args.count
    if desired_count <= 0:
        raise SystemExit("Count must be positive.")

    if args.seed is not None:
        random.seed(args.seed)

    if desired_count >= len(submissions):
        sampled = submissions
    else:
        sampled = random.sample(submissions, desired_count)

    prepare_output_dir(args.output_dir)
    created = create_symlinks(sampled, args.output_dir)
    print(f"Created {created} symlinks in {args.output_dir}")

    total_extracted = 0
    for submission in sampled:
        extracted = extract_supplementary(submission.resolve(), args.supplementary_dirname)
        total_extracted += extracted
    print(f"Extracted supplementary archives for {total_extracted} zip files across sampled submissions")


if __name__ == "__main__":
    main()
