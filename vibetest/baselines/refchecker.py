"""RefChecker baseline runner for academic reference validation."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _read_env_file_value(var_name: str) -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return None
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != var_name:
                continue
            cleaned = value.strip()
            if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
                cleaned = cleaned[1:-1]
            return cleaned or None
    except OSError:
        return None
    return None


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


def _list_bib_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.bib"))


def _select_matching_bbl(root: Path, main_tex: Path) -> Path | None:
    direct_match = main_tex.with_suffix(".bbl")
    if direct_match.exists():
        return direct_match

    matching_bbls = sorted(root.rglob(f"{main_tex.stem}.bbl"))
    if matching_bbls:
        return matching_bbls[0]
    return None


def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "paper"


def _create_safe_view(paper_dir: Path, paper_file: Path, output_root: Path) -> tuple[Path, Path]:
    safe_dir = output_root / "safe_papers" / _safe_slug(paper_dir.name)
    safe_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        if safe_dir.exists() or safe_dir.is_symlink():
            if not safe_dir.exists():
                safe_dir.unlink()
        if not safe_dir.exists():
            os.symlink(paper_dir.resolve(), safe_dir, target_is_directory=True)
    except OSError:
        return paper_dir, paper_file

    try:
        relative = paper_file.relative_to(paper_dir)
    except ValueError:
        return paper_dir, paper_file

    safe_file = safe_dir / relative
    return safe_dir.resolve(), safe_file.resolve()


def run_refchecker(
    paper_path: Path,
    *,
    refchecker_cmd: str = "academic-refchecker",
    output_root: Path | None = None,
    timeout_s: int = 1200,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    semantic_scholar_api_key: str | None = None,
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
    generated_input: Path | None = None
    if paper_path.is_dir():
        bib_files = _list_bib_files(paper_path)
        if bib_files:
            output_root = output_root or Path("results") / "refchecker"
            output_root.mkdir(parents=True, exist_ok=True)
            inputs_dir = output_root / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            paper_inputs_dir = inputs_dir / _safe_slug(paper_path.name)
            paper_inputs_dir.mkdir(parents=True, exist_ok=True)
            generated_input = paper_inputs_dir / "combined.bib"
            try:
                combined_parts: list[str] = []
                for bib in bib_files:
                    bib_text = bib.read_text(encoding="utf-8", errors="replace")
                    combined_parts.append(f"% Source: {bib}\n{bib_text.strip()}\n")
                generated_input.write_text("\n\n".join(combined_parts), encoding="utf-8")
                paper_file = generated_input
            except OSError:
                paper_file = bib_files[0]
        else:
            main_tex = _select_main_tex(paper_path)
            if main_tex is None:
                return {
                    "ok": False,
                    "error": f"No .tex/.bib/.bbl files found under: {paper_path}",
                    "report_path": None,
                    "review": "",
                    "paper_file": None,
                }
            matching_bbl = _select_matching_bbl(paper_path, main_tex)
            if matching_bbl is not None:
                output_root = output_root or Path("results") / "refchecker"
                output_root.mkdir(parents=True, exist_ok=True)
                inputs_dir = output_root / "inputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                paper_inputs_dir = inputs_dir / _safe_slug(paper_path.name)
                paper_inputs_dir.mkdir(parents=True, exist_ok=True)
                generated_input = paper_inputs_dir / f"{main_tex.stem}.tex"
                try:
                    bbl_text = matching_bbl.read_text(encoding="utf-8", errors="replace")
                    generated_input.write_text(bbl_text, encoding="utf-8")
                    paper_file = generated_input
                except OSError:
                    paper_file = matching_bbl
            else:
                paper_file = main_tex

    if isinstance(paper_file, Path):
        try:
            paper_file = paper_file.resolve()
        except OSError:
            pass

    output_root = (output_root or Path("results") / "refchecker").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = (output_root / f"{_safe_slug(paper_path.name)}_refchecker.txt").resolve()
    if output_file.exists():
        output_file.unlink()

    paper_dir = paper_path if paper_path.is_dir() else paper_file.parent
    safe_workdir, safe_paper_file = _create_safe_view(paper_dir, paper_file, output_root)

    args = shlex.split(refchecker_cmd)
    repo_root = Path(__file__).resolve().parents[2]
    for idx, arg in enumerate(args):
        if arg.endswith(".py") and not os.path.isabs(arg):
            candidate = (repo_root / arg).resolve()
            if candidate.exists():
                args[idx] = str(candidate)
    args.extend(["--paper", str(safe_paper_file)])
    args.extend(["--output-file", str(output_file)])
    if llm_provider:
        args.extend(["--llm-provider", llm_provider])
    if llm_model:
        args.extend(["--llm-model", llm_model])
    if db_path:
        args.extend(["--db-path", str(db_path)])
    if extra_args:
        args.extend(extra_args)

    resolved_semantic_scholar_api_key = (
        semantic_scholar_api_key
        or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        or _read_env_file_value("SEMANTIC_SCHOLAR_API_KEY")
    )
    env = os.environ.copy()
    if resolved_semantic_scholar_api_key:
        env["SEMANTIC_SCHOLAR_API_KEY"] = resolved_semantic_scholar_api_key

    try:
        result = subprocess.run(
            args,
            cwd=str(workdir) if workdir else str(safe_workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
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
    else:
        review_text = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

    ok = result.returncode == 0
    error = None
    if not ok:
        error = result.stderr.strip() or result.stdout.strip() or f"RefChecker exit code {result.returncode}"

    return {
        "ok": ok,
        "error": error,
        "report_path": str(output_file),
        "review": review_text,
        "paper_file": str(safe_paper_file),
        "generated_input": str(generated_input) if generated_input else None,
    }
