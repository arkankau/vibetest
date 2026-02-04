"""RefChecker baseline runner for academic reference validation."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _iter_tex_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        base = os.path.basename(dirpath)
        if base in {".git", ".venv", "venv", "__pycache__", "node_modules"}:
            dirs[:] = []
            continue
        for name in files:
            if name.endswith(".tex"):
                candidates.append(Path(dirpath) / name)
    return candidates


def _select_main_tex(root: Path) -> Path | None:
    candidates = _iter_tex_files(root)
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return (0, path.stat().st_size if path.exists() else 0)
        has_docclass = 1 if re.search(r"\\documentclass", text) else 0
        has_begin = 1 if re.search(r"\\begin\\{document\\}", text) else 0
        return (has_docclass + has_begin, len(text))

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[0]


def run_refchecker(
    paper_path: Path,
    *,
    refchecker_cmd: str = "academic-refchecker",
    output_root: Path | None = None,
    timeout_s: int = 1200,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    db_path: Path | None = None,
    extra_args: list[str] | None = None,
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Run RefChecker on a LaTeX project or file."""
    if not paper_path.exists():
        return {
            "ok": False,
            "error": f"Paper path not found: {paper_path}",
            "report_path": None,
            "review": "",
            "paper_file": None,
        }

    paper_file = paper_path
    if paper_path.is_dir():
        main_tex = _select_main_tex(paper_path)
        if main_tex is None:
            return {
                "ok": False,
                "error": f"No .tex files found under: {paper_path}",
                "report_path": None,
                "review": "",
                "paper_file": None,
            }
        paper_file = main_tex

    output_root = output_root or Path("results") / "refchecker"
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"{paper_path.name}_refchecker.txt"

    args = shlex.split(refchecker_cmd)
    args.extend(["--paper", str(paper_file)])
    args.extend(["--output-file", str(output_file)])
    if llm_provider:
        args.extend(["--llm-provider", llm_provider])
    if llm_model:
        args.extend(["--llm-model", llm_model])
    if db_path:
        args.extend(["--db-path", str(db_path)])
    if extra_args:
        args.extend(extra_args)

    try:
        result = subprocess.run(
            args,
            cwd=str(workdir) if workdir else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"RefChecker timed out after {timeout_s}s",
            "report_path": str(output_file),
            "review": "",
            "paper_file": str(paper_file),
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": f"RefChecker failed to start: {exc}",
            "report_path": str(output_file),
            "review": "",
            "paper_file": str(paper_file),
        }

    review_text = ""
    if output_file.exists():
        review_text = output_file.read_text(encoding="utf-8", errors="replace")
    elif result.stdout:
        review_text = result.stdout

    ok = result.returncode == 0
    error = None
    if not ok:
        error = result.stderr.strip() or result.stdout.strip() or f"RefChecker exit code {result.returncode}"

    return {
        "ok": ok,
        "error": error,
        "report_path": str(output_file),
        "review": review_text,
        "paper_file": str(paper_file),
    }
