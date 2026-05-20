"""Create synthetic bug experiment artifacts and injection plans.

This script has three subcommands:
- collect: build clean repo sets and bug exemplars into synth-data/
- plan: sample per-repo bug-injection assignments from collected artifacts
- inject: run LLM-driven bug injection over planned rows (requires --confirm)

Important:
- This script never writes into data/.
- It only reads existing results/ and data/ artifacts, then writes synth-data/.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE", "NOT APPLICABLE"}


@dataclass(frozen=True)
class DomainArtifacts:
    domain: str
    clean_repos: list[dict[str, Any]]
    bug_examples: list[dict[str, Any]]
    summary: dict[str, Any]


DOMAIN_CHOICES = ("security-vuln", "ml-bugs", "citation-hallucinations")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_property(text: str) -> str:
    return _normalize_whitespace(text).lower()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(text: str, *, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip())
    cleaned = cleaned.strip("-._")
    if not cleaned:
        cleaned = "item"
    return cleaned[:max_len]


def _model_suffix(model_name: str) -> str:
    raw = (model_name or "").strip()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return _slugify(raw, max_len=60)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    def _coerce(obj: Any) -> dict[str, Any] | None:
        return obj if isinstance(obj, dict) else None

    try:
        return _coerce(json.loads(text))
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            obj = json.loads(block)
            coerced = _coerce(obj)
            if coerced is not None:
                return coerced
        except Exception:
            continue

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            return _coerce(obj)
        except Exception:
            pass
    return None


def _build_bug_examples_text(examples: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append(f"- Property ID: {ex.get('property_id')}")
        lines.append(f"- Property Description: {ex.get('property_text')}")
        lines.append(f"- Property Failure Reason: {_normalize_whitespace(str(ex.get('reason_text') or ''))[:500]}")
        lines.append(f"- Evidence: {_normalize_whitespace(str(ex.get('evidence_text') or ''))[:1200]}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_property_catalog_text(properties: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        pid = str(prop.get("property_id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        text = _normalize_whitespace(str(prop.get("property_text") or ""))
        lines.append(f"- {pid}: {text}" if text else f"- {pid}")
    return "\n".join(lines).strip() or "(not provided)"


def _build_injection_prompt(plan_row: dict[str, Any], *, sandbox_repo_path: str = "/workspace/repo") -> str:
    selected_examples = plan_row.get("selected_bug_examples") or []
    selected_props = plan_row.get("selected_property_ids") or []
    available_properties = plan_row.get("available_properties") or []
    k = int(plan_row.get("k") or 0)
    examples_text = _build_bug_examples_text(selected_examples)
    property_catalog_text = _build_property_catalog_text(available_properties)
    domain = str(plan_row.get("domain") or "")
    return (
        "You are an expert software security/ML engineer. Modify the repository to inject synthetic bugs corresponding to failures of listed properties.\n\n"
        f"Domain: {domain}\n"
        f"Repository path: {sandbox_repo_path}\n"
        f"Number of target property failures to inject (k): {k}\n"
        f"Target property IDs which injected bugs should violate: {selected_props}\n\n"
        "Full property catalog for this dataset:\n"
        f"{property_catalog_text}\n\n"
        "Injection goals:\n"
        "1. Inject ONLY bugs corresponding to failures of the target properties.\n"
        "2. Edit existing repository files in place.\n"
        "3. Do NOT create backups or duplicate clean copies (for example: .bak/.orig files,\n"
        "   renamed originals, copy files, or parallel clean versions).\n"
        "4. Keep edits minimal and realistic.\n"
        "5. Do not add obvious markers like 'TODO bug' or 'synthetic bug'.\n"
        "6. Do not intentionally corrupt unrelated functionality.\n"
        "7. Preserve all non-target properties in the full property catalog as passing.\n"
        "   Before editing, review that catalog and choose an injection that is specific\n"
        "   to the target property IDs. Do not use an injection if it would also\n"
        "   make any non-target property fail.\n"
        "8. After editing, audit the modified repository against non-target properties. If\n"
        "   a non-target property would fail, revise the edit or report that the requested\n"
        "   isolated injection is not feasible.\n"
        "9. If k=0, keep code behavior unchanged and only produce the report/artifacts.\n\n"
        "Bug exemplars to imitate:\n"
        f"{examples_text}\n\n"
        "Required artifacts (must be created before submit):\n"
        "1. `/evidence/artifacts/injection_report.json` with keys:\n"
        "   - injected_property_ids (array)\n"
        "   - changed_files (array of relative paths)\n"
        "   - injection_summary (string)\n"
        "   - bug_locations (array of {file, hint})\n"
        f"2. `/evidence/artifacts/repo.tar.gz` containing the full modified contents of `{sandbox_repo_path}`.\n"
        "   You must archive from the repository root path itself (not from any nested subdirectory).\n"
        "   Example command:\n"
        f"   `mkdir -p /evidence/artifacts && tar -czf /evidence/artifacts/repo.tar.gz -C {sandbox_repo_path} .`\n"
        "   Do NOT use a subdirectory in `-C` (for example, do NOT use `/workspace/repo/dir`).\n\n"
        "Repository constraints:\n"
        "- Do not leave any backup artifacts in the repo (including files ending in .bak, .orig, .old,\n"
        "  ~, or names containing backup/copy of).\n"
        "- Do not rename clean originals and keep backup variants; modify target files directly.\n\n"
        "Final response format (submit this exact structure):\n"
        "VERDICT: PASS or FAIL\n"
        "REASON: one-paragraph summary\n"
        "EVIDENCE: list changed files and artifact paths\n"
    )


def _evidence_tar_path(evidence_root: Path, model_name: str, sample_name: str) -> Path:
    return evidence_root / _model_suffix(model_name) / f"evidence-{sample_name}.tar.gz"


def _extract_tar_compat(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract tar safely with Python 3.12+ filter support and older fallback."""
    try:
        tf.extractall(dest, filter="data")
    except TypeError:
        # Python <3.12 does not support the filter argument.
        tf.extractall(dest)


def _extract_injection_artifacts(
    evidence_tar: Path,
    *,
    output_repo_dir: Path,
) -> tuple[bool, dict[str, Any]]:
    info: dict[str, Any] = {
        "evidence_tar": str(evidence_tar),
        "repo_archive_found": False,
        "report_found": False,
        "output_repo_dir": str(output_repo_dir),
    }
    if not evidence_tar.exists():
        info["error"] = f"Evidence tar not found: {evidence_tar}"
        return False, info

    with tempfile.TemporaryDirectory(prefix="synth-inject-evidence-") as td:
        temp_dir = Path(td)
        try:
            with tarfile.open(evidence_tar, "r:gz") as tf:
                _extract_tar_compat(tf, temp_dir)
        except Exception as exc:
            info["error"] = f"Failed to extract evidence tar: {exc}"
            return False, info

        report_path = temp_dir / "evidence" / "artifacts" / "injection_report.json"
        repo_tgz = temp_dir / "evidence" / "artifacts" / "repo.tar.gz"
        info["report_found"] = report_path.exists()
        info["repo_archive_found"] = repo_tgz.exists()

        report_obj: dict[str, Any] | None = None
        if report_path.exists():
            try:
                report_obj = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                report_obj = None
        if report_obj is not None:
            info["injection_report"] = report_obj

        if not repo_tgz.exists():
            info["error"] = "Missing /evidence/artifacts/repo.tar.gz"
            return False, info

        extracted_dir = temp_dir / "repo_unpack"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(repo_tgz, "r:gz") as tf:
                members = tf.getmembers()
                member_names = [m.name for m in members]
                info["repo_archive_member_count"] = len(member_names)
                info["repo_archive_members_preview"] = member_names[:200]
                _extract_tar_compat(tf, extracted_dir)
        except Exception as exc:
            info["error"] = f"Failed to extract injected repo archive: {exc}"
            return False, info

        if output_repo_dir.exists():
            shutil.rmtree(output_repo_dir)
        output_repo_dir.mkdir(parents=True, exist_ok=True)
        # Preserve tar root layout exactly; do not flatten one-level directory trees.
        for child in extracted_dir.iterdir():
            shutil.move(str(child), str(output_repo_dir / child.name))

    return True, info


def _iter_repo_files(root: Path) -> set[str]:
    out: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        out.add(rel)
    return out


def _read_bytes(path: Path, max_bytes: int) -> tuple[bytes, bool]:
    data = path.read_bytes()
    truncated = False
    if max_bytes > 0 and len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True
    return data, truncated


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_notebook_path(rel_path: str) -> bool:
    return rel_path.lower().endswith(".ipynb")


def _truncate_text_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    if max_bytes <= 0:
        return text, False
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text, False
    return raw[:max_bytes].decode("utf-8", errors="replace"), True


def _nbdime_notebook_diff(
    original_notebook: Path,
    injected_notebook: Path,
    *,
    max_bytes: int,
) -> tuple[str | None, str | None]:
    nbdiff = shutil.which("nbdiff")
    if not nbdiff:
        return None, "nbdiff not found; install nbdime to get human-readable notebook diffs."

    try:
        proc = subprocess.run(
            [nbdiff, str(original_notebook), str(injected_notebook)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:
        return None, f"nbdiff execution failed: {exc}"

    # nbdiff commonly returns 1 when differences are present.
    stdout = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout or "")
    stderr = re.sub(r"\x1b\[[0-9;]*m", "", proc.stderr or "")
    if proc.returncode not in (0, 1):
        err = stderr.strip() or f"unexpected exit code {proc.returncode}"
        return None, f"nbdiff failed: {err}"

    if not stdout.strip():
        return None, "nbdiff returned no diff text."

    output = stdout.rstrip() + "\n"
    output, truncated = _truncate_text_bytes(output, max_bytes=max_bytes)
    if truncated:
        return output, "nbdiff output truncated to diff preview byte limit."

    if stderr.strip():
        return output, f"nbdiff warning: {stderr.strip()}"
    return output, None


def _compute_repo_diff(
    original_repo: Path,
    injected_repo: Path,
    *,
    diff_path: Path,
    diff_meta_path: Path,
    max_file_bytes: int,
) -> dict[str, Any]:
    a_files = _iter_repo_files(original_repo)
    b_files = _iter_repo_files(injected_repo)
    all_files = sorted(a_files | b_files)

    added: list[str] = []
    removed: list[str] = []
    modified: list[str] = []
    unchanged: list[str] = []
    patch_parts: list[str] = []
    notebooks_modified: list[str] = []
    nbdime_used: list[str] = []
    nbdime_fallback: list[str] = []
    nbdime_notes: list[dict[str, str]] = []

    for rel in all_files:
        a_path = original_repo / rel
        b_path = injected_repo / rel
        in_a = rel in a_files
        in_b = rel in b_files

        if in_a and not in_b:
            removed.append(rel)
            patch_parts.append(f"diff --git a/{rel} b/{rel}\n")
            patch_parts.append("deleted file mode 100644\n")
            patch_parts.append(f"--- a/{rel}\n")
            patch_parts.append(f"+++ /dev/null\n")
            continue
        if in_b and not in_a:
            added.append(rel)
            patch_parts.append(f"diff --git a/{rel} b/{rel}\n")
            patch_parts.append("new file mode 100644\n")
            patch_parts.append(f"--- /dev/null\n")
            patch_parts.append(f"+++ b/{rel}\n")
            b_data, b_truncated = _read_bytes(b_path, max_file_bytes)
            if _is_binary(b_data):
                patch_parts.append(f"Binary file added: b/{rel} sha256={_sha256_hex(b_path)}\n")
                if b_truncated:
                    patch_parts.append("NOTE: preview truncated while checking binary content.\n")
            else:
                b_text = b_data.decode("utf-8", errors="replace").splitlines()
                ud = difflib.unified_diff([], b_text, fromfile="/dev/null", tofile=f"b/{rel}", lineterm="")
                patch_parts.extend(line + "\n" for line in ud)
            continue

        a_data, a_truncated = _read_bytes(a_path, max_file_bytes)
        b_data, b_truncated = _read_bytes(b_path, max_file_bytes)

        if _is_binary(a_data) or _is_binary(b_data):
            if _sha256_hex(a_path) == _sha256_hex(b_path):
                unchanged.append(rel)
                continue
            modified.append(rel)
            patch_parts.append(f"diff --git a/{rel} b/{rel}\n")
            patch_parts.append(f"Binary files differ: a/{rel} ({_sha256_hex(a_path)}) vs b/{rel} ({_sha256_hex(b_path)})\n")
            if a_truncated or b_truncated:
                patch_parts.append("NOTE: preview truncated while checking binary content.\n")
            continue

        a_text_full = a_path.read_text(encoding="utf-8", errors="replace")
        b_text_full = b_path.read_text(encoding="utf-8", errors="replace")
        if a_text_full == b_text_full:
            unchanged.append(rel)
            continue

        modified.append(rel)
        if _is_notebook_path(rel):
            notebooks_modified.append(rel)
        patch_parts.append(f"diff --git a/{rel} b/{rel}\n")

        if _is_notebook_path(rel):
            nb_text, nb_note = _nbdime_notebook_diff(a_path, b_path, max_bytes=max_file_bytes)
            if nb_text:
                nbdime_used.append(rel)
                patch_parts.append("# Notebook diff generated by nbdime (nbdiff)\n")
                patch_parts.append(nb_text)
                if not nb_text.endswith("\n"):
                    patch_parts.append("\n")
                if nb_note:
                    nbdime_notes.append({"file": rel, "note": nb_note})
                    patch_parts.append(f"# NOTE: {nb_note}\n")
                continue

            nbdime_fallback.append(rel)
            note = nb_note or "nbdime unavailable; used unified text diff fallback."
            nbdime_notes.append({"file": rel, "note": note})
            patch_parts.append(f"# NOTE: {note}\n")

        a_lines = a_text_full.splitlines()
        b_lines = b_text_full.splitlines()
        ud = difflib.unified_diff(a_lines, b_lines, fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="")
        patch_parts.extend(line + "\n" for line in ud)

    diff_path.parent.mkdir(parents=True, exist_ok=True)
    patch_body = "".join(patch_parts)
    if not patch_body:
        patch_body = "# No differences detected between original and injected repo.\n"
    diff_path.write_text(patch_body, encoding="utf-8")

    meta = {
        "original_repo": str(original_repo),
        "injected_repo": str(injected_repo),
        "diff_patch_path": str(diff_path),
        "max_file_bytes": max_file_bytes,
        "added_files": added,
        "removed_files": removed,
        "modified_files": modified,
        "unchanged_file_count": len(unchanged),
        "added_count": len(added),
        "removed_count": len(removed),
        "modified_count": len(modified),
        "changed_count": len(added) + len(removed) + len(modified),
        "notebook_diffs": {
            "modified_ipynb_files": notebooks_modified,
            "nbdime_used_files": nbdime_used,
            "nbdime_fallback_files": nbdime_fallback,
            "notes": nbdime_notes,
        },
    }
    diff_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def _parse_verdict(raw: Any, passed: Any = None) -> str:
    text = _normalize_whitespace(str(raw) if raw is not None else "").upper()
    if text in VERDICTS:
        return text
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    m = re.search(r"\b(PASS|FAIL|INCONCLUSIVE|NOT\s+APPLICABLE)\b", text, re.IGNORECASE)
    if m:
        v = m.group(1).upper().replace("  ", " ")
        return "NOT APPLICABLE" if v == "NOT APPLICABLE" else v
    return ""


def _test_property_text(test: dict[str, Any]) -> str:
    meta = test.get("metadata") or {}
    return _normalize_whitespace(
        str(meta.get("property_text") or "")
        or str(meta.get("test_description") or "")
        or str(test.get("property") or "")
    )


def _test_verdict(test: dict[str, Any]) -> str:
    meta = test.get("metadata") or {}
    verdict = _parse_verdict(meta.get("verdict"), passed=test.get("passed"))
    if verdict:
        return verdict
    verdict = _parse_verdict(test.get("verdict"), passed=test.get("passed"))
    if verdict:
        return verdict
    return _parse_verdict(test.get("description"), passed=test.get("passed"))


def _test_evidence_text(test: dict[str, Any]) -> str:
    meta = test.get("metadata") or {}
    if meta.get("evidence_text"):
        return str(meta.get("evidence_text"))
    evidence = test.get("evidence")
    if isinstance(evidence, list):
        return "\n".join(str(item) for item in evidence if item is not None)
    if isinstance(evidence, str):
        return evidence
    return ""


def _property_catalog_from_plan_rows(plan_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
    catalog: dict[tuple[str, str], dict[str, str]] = {}
    for row in plan_rows:
        domain = str(row.get("domain") or "").strip()
        dataset = str(row.get("dataset") or "").strip()
        key = (domain, dataset)
        props = catalog.setdefault(key, {})

        for pid in row.get("selected_property_ids") or []:
            pid_s = str(pid or "").strip()
            if pid_s:
                props.setdefault(pid_s, "")

        for ex in row.get("selected_bug_examples") or []:
            if not isinstance(ex, dict):
                continue
            pid = str(ex.get("property_id") or "").strip()
            if not pid:
                continue
            ptxt = _normalize_whitespace(str(ex.get("property_text") or ""))
            if ptxt:
                props[pid] = ptxt
            else:
                props.setdefault(pid, "")
    return catalog


def _available_properties_for_plan_row(
    row: dict[str, Any],
    property_catalog: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    existing = row.get("available_properties") or []
    if existing:
        return existing

    domain = str(row.get("domain") or "").strip()
    dataset = str(row.get("dataset") or "").strip()
    props = property_catalog.get((domain, dataset)) or {}
    return [
        {"property_id": pid, "property_text": props.get(pid, "")}
        for pid in sorted(props)
    ]


def _property_description_from_bug_locations(
    property_id: str,
    bug_locations: list[dict[str, Any]],
    *,
    injection_summary: str,
    changed_files: list[str],
) -> str:
    pid = (property_id or "").strip()
    if not pid:
        return _normalize_whitespace(injection_summary)

    pid_lower = pid.lower()
    aliases = [pid_lower]
    suffix = pid_lower.split("_")[-1] if "_" in pid_lower else ""
    if suffix and suffix not in aliases:
        aliases.append(suffix)

    matched: list[str] = []
    for item in bug_locations:
        if not isinstance(item, dict):
            continue
        file_path = _normalize_whitespace(str(item.get("file") or ""))
        hint = _normalize_whitespace(str(item.get("hint") or ""))
        blob = f"{file_path} {hint}".lower()
        ok = False
        for tok in aliases:
            if not tok:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob):
                ok = True
                break
        if ok:
            if file_path and hint:
                matched.append(f"{file_path}: {hint}")
            elif hint:
                matched.append(hint)
            elif file_path:
                matched.append(file_path)

    if matched:
        return " | ".join(matched)

    summary = _normalize_whitespace(injection_summary)
    if summary:
        return f"Injected for {pid}. {summary}"
    if changed_files:
        return f"Injected for {pid}. Changed files: {', '.join(changed_files)}."
    return f"Injected for {pid}."


def _ground_truth_labels_for_row(
    row: dict[str, Any],
    *,
    extract_info: dict[str, Any] | None,
    property_catalog: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    domain = str(row.get("domain") or "").strip()
    dataset = str(row.get("dataset") or "").strip()
    key = (domain, dataset)

    catalog_for_key: dict[str, str] = dict(property_catalog.get(key, {}))
    # Ensure this row's selected bug examples are represented even if catalog was sparse.
    for ex in row.get("selected_bug_examples") or []:
        if not isinstance(ex, dict):
            continue
        pid = str(ex.get("property_id") or "").strip()
        if not pid:
            continue
        ptxt = _normalize_whitespace(str(ex.get("property_text") or ""))
        if ptxt:
            catalog_for_key[pid] = ptxt
        else:
            catalog_for_key.setdefault(pid, "")

    intended_failed = {str(pid).strip() for pid in (row.get("selected_property_ids") or []) if str(pid).strip()}

    report = ((extract_info or {}).get("injection_report") or {}) if isinstance(extract_info, dict) else {}
    injected_from_report = {
        str(pid).strip() for pid in (report.get("injected_property_ids") or []) if str(pid).strip()
    }
    failed_property_ids = sorted(intended_failed or injected_from_report)

    # Ensure all failed properties appear even if absent from catalog.
    for pid in failed_property_ids:
        catalog_for_key.setdefault(pid, "")

    bug_locations = report.get("bug_locations") or []
    if not isinstance(bug_locations, list):
        bug_locations = []
    changed_files = report.get("changed_files") or []
    if not isinstance(changed_files, list):
        changed_files = []
    injection_summary = _normalize_whitespace(str(report.get("injection_summary") or ""))

    ordered_pids = sorted(catalog_for_key.keys())
    property_rows: list[dict[str, Any]] = []
    property_labels: dict[str, int] = {}
    violation_descriptions: dict[str, str] = {}

    failed_set = set(failed_property_ids)
    for pid in ordered_pids:
        is_fail = pid in failed_set
        property_labels[pid] = 1 if is_fail else 0
        row_obj = {
            "property_id": pid,
            "property_text": catalog_for_key.get(pid) or "",
            "label": 1 if is_fail else 0,
            "verdict": "FAIL" if is_fail else "PASS",
            "gt_violation_description": "",
        }
        if is_fail:
            desc = _property_description_from_bug_locations(
                pid,
                bug_locations,
                injection_summary=injection_summary,
                changed_files=[str(x) for x in changed_files],
            )
            row_obj["gt_violation_description"] = desc
            violation_descriptions[pid] = desc
        property_rows.append(row_obj)

    return {
        "failed_property_ids": failed_property_ids,
        "property_labels": property_labels,
        "violation_descriptions": violation_descriptions,
        "by_property": property_rows,
    }


def _read_bullet_properties(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    return [_normalize_whitespace(chunk) for chunk in content.split("- ")[1:] if chunk.strip()]


def _read_cwe_properties(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    heading_re = re.compile(r"^##\s*(CWE-\d+[^\n]*)\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(content))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        section = _normalize_whitespace(content[start:end])
        cwe_match = re.search(r"CWE-(\d+)", match.group(1))
        if not cwe_match:
            continue
        cwe = cwe_match.group(1).lstrip("0") or "0"
        out[cwe] = section
    return out


def _limit_examples_per_property(
    examples: list[dict[str, Any]],
    *,
    max_per_property: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        grouped.setdefault(str(ex.get("property_id") or ""), []).append(ex)

    out: list[dict[str, Any]] = []
    for prop_id in sorted(grouped):
        rows = sorted(
            grouped[prop_id],
            key=lambda r: (
                str(r.get("dataset") or ""),
                str(r.get("repo_slug") or ""),
                str(r.get("repo_name") or ""),
            ),
        )
        out.extend(rows[:max_per_property])
    return out


def _balance_rows_by_subdataset(
    rows: list[dict[str, Any]],
    *,
    subdataset_key: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(subdataset_key) or "").strip()
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    if not grouped:
        return rows, {"balanced": False, "reason": f"missing key '{subdataset_key}'"}

    counts = {k: len(v) for k, v in grouped.items()}
    target = min(counts.values())
    if target <= 0:
        return [], {"balanced": True, "subdataset_counts_before": counts, "target_per_subdataset": 0}

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        selected = list(grouped[key])
        rng.shuffle(selected)
        out.extend(selected[:target])

    out.sort(key=lambda r: (str(r.get(subdataset_key) or ""), str(r.get("repo_slug") or ""), str(r.get("repo_name") or "")))
    meta = {
        "balanced": True,
        "subdataset_counts_before": counts,
        "target_per_subdataset": target,
        "subdataset_counts_after": {k: target for k in sorted(grouped)},
    }
    return out, meta


def _load_human_c_labels(
    long_csv: Path,
    *,
    method_prefixes: tuple[str, ...],
) -> dict[str, set[tuple[str, str]]]:
    """Return mapping: method_run_name -> set[(repo_name, normalized_property_text)] for label C."""
    if not long_csv.exists():
        return {}
    out: dict[str, set[tuple[str, str]]] = {}
    with long_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method_run = _normalize_whitespace(str(row.get("method_run_name") or ""))
            if not method_run or not method_run.startswith(method_prefixes):
                continue
            label = _normalize_whitespace(str(row.get("human_label") or "")).upper()
            if label != "C":
                continue
            repo_name = _normalize_whitespace(str(row.get("repo_name") or ""))
            prop = _normalize_property(str(row.get("property") or ""))
            if not repo_name or not prop:
                continue
            out.setdefault(method_run, set()).add((repo_name, prop))
    return out


def _resolve_existing(path_candidates: list[Path]) -> Path:
    for p in path_candidates:
        if p.exists():
            return p
    tried = ", ".join(str(p) for p in path_candidates)
    raise FileNotFoundError(f"No candidate file exists: {tried}")


def _resolve_cwe_bench_repo_path(cwe_repo_root: Path, repo_slug: str) -> Path:
    """Resolve CWE-Bench repo path supporting numeric-prefixed directory names."""
    slug = _normalize_whitespace(repo_slug)
    if not slug:
        raise FileNotFoundError("Empty CWE-Bench repo slug")

    exact = cwe_repo_root / slug
    if exact.exists() and exact.is_dir():
        return exact

    # Common layout is: <int>_<repo_slug>
    prefixed = sorted(p for p in cwe_repo_root.glob(f"[0-9]*_{slug}") if p.is_dir())
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise FileNotFoundError(
            f"Ambiguous CWE-Bench repo slug '{slug}' matched multiple directories: "
            + ", ".join(str(p.name) for p in prefixed[:10])
        )

    # Fallback: strip any leading numeric prefix and compare.
    fallback = sorted(
        p
        for p in cwe_repo_root.iterdir()
        if p.is_dir() and re.sub(r"^\d+_", "", p.name) == slug
    )
    if len(fallback) == 1:
        return fallback[0]
    if len(fallback) > 1:
        raise FileNotFoundError(
            f"Ambiguous CWE-Bench fallback resolution for slug '{slug}': "
            + ", ".join(str(p.name) for p in fallback[:10])
        )

    raise FileNotFoundError(
        f"CWE-Bench repo for slug '{slug}' not found under {cwe_repo_root}"
    )


def _collect_vulnerability_domain(
    results_dir: Path,
    data_dir: Path,
    max_per_property: int,
) -> DomainArtifacts:
    bibifi_recall = _resolve_existing(
        [
            results_dir / "bibifi_AT-gpt-5-mini_recall_scored.jsonl",
            results_dir / "bibifi_AT-gpt-5.2_recall_scored.jsonl",
        ]
    )
    cwe_recall = _resolve_existing(
        [
            results_dir / "cwe-bench_AT-gpt-5-mini_recall_scored.jsonl",
            results_dir / "cwe-bench_AT-gpt-5.2_recall_scored.jsonl",
        ]
    )

    bibifi_props = _read_bullet_properties(data_dir / "vuln" / "bibifi" / "properties.md")
    cwe_props = _read_cwe_properties(data_dir / "vuln" / "cwe-bench" / "properties.md")
    cwe_repo_root = data_dir / "vuln" / "cwe-bench" / "repos"

    clean_state: dict[tuple[str, str], dict[str, Any]] = {}
    bug_examples: list[dict[str, Any]] = []
    cwe_resolution_missing = 0

    for dataset, recall_path in (("bibifi", bibifi_recall), ("cwe-bench", cwe_recall)):
        for row in _load_jsonl(recall_path):
            if str(row.get("record_type") or "") != "sample":
                continue
            repo_slug = _normalize_whitespace(str(row.get("repo_slug") or ""))
            if not repo_slug:
                continue

            if dataset == "bibifi":
                repo_path = data_dir / "vuln" / "bibifi" / "repos" / repo_slug / "build"
                sample_id = _normalize_whitespace(str(row.get("sample_id") or ""))
                m = re.search(r"_vuln(\d+)$", sample_id)
                prop_idx = int(m.group(1)) if m else -1
                prop_id = f"bibifi_vuln{prop_idx}" if prop_idx >= 0 else "bibifi_unknown"
                prop_text = bibifi_props[prop_idx] if 0 <= prop_idx < len(bibifi_props) else ""
            else:
                try:
                    repo_path = _resolve_cwe_bench_repo_path(cwe_repo_root, repo_slug)
                except FileNotFoundError:
                    cwe_resolution_missing += 1
                    continue
                cwe_id = _normalize_whitespace(str(row.get("cwe_id") or "")).lstrip("0") or "0"
                prop_id = f"cwe-bench_CWE-{cwe_id}"
                prop_text = cwe_props.get(cwe_id, "")

            verification = row.get("verification") or {}
            verified_fail = bool(verification.get("verified_fail"))
            verdict = _normalize_whitespace(str(row.get("verdict") or "")).upper()
            key = (dataset, repo_slug)
            state = clean_state.setdefault(
                key,
                {
                    "dataset": dataset,
                    "repo_slug": repo_slug,
                    "repo_path": str(repo_path),
                    "source_recall_results": str(recall_path),
                    "has_verified_fail": False,
                    "has_inconclusive": False,
                },
            )
            if verdict == "INCONCLUSIVE":
                state["has_inconclusive"] = True
            if verified_fail:
                state["has_verified_fail"] = True
                bug_examples.append(
                    {
                        "domain": "security-vuln",
                        "dataset": dataset,
                        "property_id": prop_id,
                        "property_text": prop_text,
                        "repo_slug": repo_slug,
                        "repo_path": str(repo_path),
                        "sample_id": row.get("sample_id"),
                        "verdict": verdict,
                        "reason_text": row.get("reason_text"),
                        "evidence_text": row.get("evidence_text"),
                        "verification": verification,
                        "verified": True,
                        "verification_source": str(recall_path),
                    }
                )

    clean_repos = [
        {
            "domain": "security-vuln",
            "dataset": s["dataset"],
            "repo_slug": s["repo_slug"],
            "repo_path": s["repo_path"],
            "clean_criterion": "no_verified_fail_and_no_inconclusive",
            "source_recall_results": s["source_recall_results"],
        }
        for s in clean_state.values()
        if (not s["has_verified_fail"]) and (not s["has_inconclusive"])
    ]
    clean_repos.sort(key=lambda r: (r["dataset"], r["repo_slug"]))

    bug_examples = _limit_examples_per_property(bug_examples, max_per_property=max_per_property)
    summary = {
        "domain": "security-vuln",
        "sources": {
            "bibifi_recall": str(bibifi_recall),
            "cwe_bench_recall": str(cwe_recall),
        },
        "cwe_repo_root": str(cwe_repo_root),
        "cwe_repo_resolution_missing_count": cwe_resolution_missing,
        "clean_repo_count": len(clean_repos),
        "bug_example_count": len(bug_examples),
        "property_count_with_examples": len({e["property_id"] for e in bug_examples}),
        "clean_definition": "no verified FAIL and no INCONCLUSIVE in recall-scored vibetester output",
        "bug_example_definition": "verified FAIL from recall-scored vibetester output",
    }
    return DomainArtifacts("security-vuln", clean_repos, bug_examples, summary)


def _collect_ml_domain(
    results_dir: Path,
    human_long_csv: Path,
    max_per_property: int,
) -> DomainArtifacts:
    source_files = [
        _resolve_existing([results_dir / "kaggle_titanic_AT-gpt-5-mini.jsonl"]),
        _resolve_existing([results_dir / "kaggle_diabetic_AT-gpt-5-mini.jsonl"]),
        _resolve_existing([results_dir / "kaggle_nlp_AT-gpt-5-mini.jsonl"]),
    ]
    verified = _load_human_c_labels(human_long_csv, method_prefixes=("kaggle_",))
    kaggle_properties = _read_bullet_properties(Path("data/kaggle/properties.md"))
    prop_index = {_normalize_property(p): i for i, p in enumerate(kaggle_properties)}

    bug_examples: list[dict[str, Any]] = []
    clean_repos: list[dict[str, Any]] = []

    for results_file in source_files:
        method_run_name = results_file.stem
        rows = _load_jsonl(results_file)
        verified_keys = verified.get(method_run_name, set())

        for row in rows:
            repo_name = _normalize_whitespace(str(row.get("repo_name") or ""))
            repo_path = _normalize_whitespace(str(row.get("repo") or ""))
            tests = row.get("tests") or []
            has_verified_fail = False
            has_inconclusive = False

            for test in tests:
                if not isinstance(test, dict):
                    continue
                prop_text = _test_property_text(test)
                prop_norm = _normalize_property(prop_text)
                verdict = _test_verdict(test)
                if verdict == "INCONCLUSIVE":
                    has_inconclusive = True
                if (repo_name, prop_norm) in verified_keys:
                    has_verified_fail = True
                    pidx = (test.get("metadata") or {}).get("property_index")
                    if isinstance(pidx, int):
                        idx = pidx
                    else:
                        idx = prop_index.get(prop_norm, -1)
                    property_id = f"kaggle_p{idx}" if idx >= 0 else f"kaggle_{prop_norm[:40]}"
                    bug_examples.append(
                        {
                            "domain": "ml-bugs",
                            "dataset": method_run_name.replace("_AT-gpt-5-mini", ""),
                            "property_id": property_id,
                            "property_text": prop_text,
                            "repo_name": repo_name,
                            "repo_path": repo_path,
                            "verdict": verdict,
                            "reason_text": _normalize_whitespace(str(test.get("description") or "")),
                            "evidence_text": _test_evidence_text(test),
                            "execution_log_excerpt": str(test.get("execution_log") or "")[:4000],
                            "verified": True,
                            "verification_source": str(human_long_csv),
                            "method_run_name": method_run_name,
                        }
                    )

            if (not has_verified_fail) and (not has_inconclusive):
                clean_repos.append(
                    {
                        "domain": "ml-bugs",
                        "dataset": method_run_name.replace("_AT-gpt-5-mini", ""),
                        "repo_name": repo_name,
                        "repo_path": repo_path,
                        "clean_criterion": "no_verified_fail_and_no_inconclusive",
                        "verification_source": str(human_long_csv),
                        "method_run_name": method_run_name,
                    }
                )

    clean_repos.sort(key=lambda r: (r["dataset"], r["repo_name"]))
    bug_examples = _limit_examples_per_property(bug_examples, max_per_property=max_per_property)
    summary = {
        "domain": "ml-bugs",
        "sources": [str(p) for p in source_files],
        "human_annotation_source": str(human_long_csv),
        "clean_repo_count": len(clean_repos),
        "bug_example_count": len(bug_examples),
        "property_count_with_examples": len({e["property_id"] for e in bug_examples}),
        "clean_definition": "no verified FAIL (human C on vibetester FAILs) and no INCONCLUSIVE in vibetester output",
        "bug_example_definition": "verified FAIL from vibetester outputs via human C labels",
    }
    return DomainArtifacts("ml-bugs", clean_repos, bug_examples, summary)


def _collect_hallucination_domain(
    results_dir: Path,
    human_long_csv: Path,
    max_per_property: int,
    allow_unverified_fallback: bool,
) -> DomainArtifacts:
    source_file = _resolve_existing([results_dir / "hallucination_refchecker.jsonl"])
    method_run_name = source_file.stem
    rows = _load_jsonl(source_file)
    verified = _load_human_c_labels(human_long_csv, method_prefixes=("hallucination_refchecker",))
    verified_keys = verified.get(method_run_name, set())
    has_verified_labels = bool(verified_keys)

    bug_examples: list[dict[str, Any]] = []
    clean_repos: list[dict[str, Any]] = []

    for row in rows:
        repo_name = _normalize_whitespace(str(row.get("repo_name") or ""))
        repo_path = _normalize_whitespace(str(row.get("repo") or ""))
        tests = row.get("tests") or []
        has_verified_fail = False
        has_inconclusive = False
        any_fail = False

        for test in tests:
            if not isinstance(test, dict):
                continue
            prop_text = _test_property_text(test)
            prop_norm = _normalize_property(prop_text)
            verdict = _test_verdict(test)
            if verdict == "INCONCLUSIVE":
                has_inconclusive = True
            if verdict == "FAIL":
                any_fail = True

            if has_verified_labels and (repo_name, prop_norm) in verified_keys:
                has_verified_fail = True
                idx = (test.get("metadata") or {}).get("property_index")
                pid = f"hallucination_p{idx}" if isinstance(idx, int) else f"hallucination_{prop_norm[:40]}"
                bug_examples.append(
                    {
                        "domain": "citation-hallucinations",
                        "dataset": "hallucination",
                        "property_id": pid,
                        "property_text": prop_text,
                        "repo_name": repo_name,
                        "repo_path": repo_path,
                        "verdict": verdict,
                        "reason_text": _normalize_whitespace(str(test.get("description") or "")),
                        "evidence_text": _test_evidence_text(test),
                        "execution_log_excerpt": str(test.get("execution_log") or "")[:4000],
                        "verified": True,
                        "verification_source": str(human_long_csv),
                        "method_run_name": method_run_name,
                    }
                )

            if (not has_verified_labels) and allow_unverified_fallback and verdict == "FAIL":
                idx = (test.get("metadata") or {}).get("property_index")
                pid = f"hallucination_p{idx}" if isinstance(idx, int) else f"hallucination_{prop_norm[:40]}"
                bug_examples.append(
                    {
                        "domain": "citation-hallucinations",
                        "dataset": "hallucination",
                        "property_id": pid,
                        "property_text": prop_text,
                        "repo_name": repo_name,
                        "repo_path": repo_path,
                        "verdict": verdict,
                        "reason_text": _normalize_whitespace(str(test.get("description") or "")),
                        "evidence_text": _test_evidence_text(test),
                        "execution_log_excerpt": str(test.get("execution_log") or "")[:4000],
                        "verified": False,
                        "verification_source": "unverified_refchecker_fail_fallback",
                        "method_run_name": method_run_name,
                    }
                )

        if has_verified_labels:
            # Hallucination verification is sparse and provided on FAIL predictions only;
            # use "no verified FAIL" as the clean criterion for this domain.
            is_clean = not has_verified_fail
            clean_criterion = "no_verified_fail"
        else:
            is_clean = (not any_fail) and (not has_inconclusive)
            clean_criterion = "strict_all_pass_no_inconclusive"

        if is_clean:
            clean_repos.append(
                {
                    "domain": "citation-hallucinations",
                    "dataset": "hallucination",
                    "repo_name": repo_name,
                    "repo_path": repo_path,
                    "clean_criterion": clean_criterion,
                    "verification_source": str(human_long_csv) if has_verified_labels else "none",
                    "method_run_name": method_run_name,
                }
            )

    clean_repos.sort(key=lambda r: r["repo_name"])
    bug_examples = _limit_examples_per_property(bug_examples, max_per_property=max_per_property)
    summary = {
        "domain": "citation-hallucinations",
        "source": str(source_file),
        "human_annotation_source": str(human_long_csv),
        "has_verified_hallucination_refchecker_labels": has_verified_labels,
        "clean_repo_count": len(clean_repos),
        "bug_example_count": len(bug_examples),
        "property_count_with_examples": len({e["property_id"] for e in bug_examples}),
        "clean_definition": (
            "no verified FAIL in refchecker output"
            if has_verified_labels
            else "strict all-PASS and no INCONCLUSIVE in refchecker output"
        ),
        "bug_example_definition": (
            "verified FAIL from refchecker outputs via human C labels"
            if has_verified_labels
            else "unverified FAIL fallback from refchecker outputs (no refchecker human labels found)"
        ),
    }
    return DomainArtifacts("citation-hallucinations", clean_repos, bug_examples, summary)


def _write_domain(output_root: Path, artifacts: DomainArtifacts) -> None:
    domain_dir = output_root / artifacts.domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(domain_dir / "clean_repos.jsonl", artifacts.clean_repos)
    _write_jsonl(domain_dir / "bug_examples.jsonl", artifacts.bug_examples)
    (domain_dir / "summary.json").write_text(
        json.dumps(artifacts.summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _collect(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    human_long_csv = Path(args.human_annotations_long)
    output_root = Path(args.output_root)

    selected = list(args.domains) if args.domains else list(DOMAIN_CHOICES)
    artifacts_list: list[DomainArtifacts] = []
    if "security-vuln" in selected:
        artifacts_list.append(_collect_vulnerability_domain(results_dir, data_dir, args.max_examples_per_property))
    if "ml-bugs" in selected:
        artifacts_list.append(_collect_ml_domain(results_dir, human_long_csv, args.max_examples_per_property))
    if "citation-hallucinations" in selected:
        artifacts_list.append(
            _collect_hallucination_domain(
                results_dir,
                human_long_csv,
                args.max_examples_per_property,
                allow_unverified_fallback=bool(args.allow_unverified_hallucination_fallback),
            )
        )

    if bool(args.equalize_clean_repos_per_subdataset):
        balanced: list[DomainArtifacts] = []
        for i, artifacts in enumerate(artifacts_list):
            clean_repos = artifacts.clean_repos
            if artifacts.domain in {"security-vuln", "ml-bugs"}:
                balanced_rows, meta = _balance_rows_by_subdataset(
                    clean_repos,
                    subdataset_key="dataset",
                    seed=int(args.seed) + i,
                )
                summary = dict(artifacts.summary)
                summary["clean_repo_balancing"] = meta
                summary["clean_repo_count"] = len(balanced_rows)
                balanced.append(
                    DomainArtifacts(
                        domain=artifacts.domain,
                        clean_repos=balanced_rows,
                        bug_examples=artifacts.bug_examples,
                        summary=summary,
                    )
                )
            else:
                balanced.append(artifacts)
        artifacts_list = balanced

    for artifacts in artifacts_list:
        _write_domain(output_root, artifacts)

    manifest = {
        "generated_by": "scripts/create_synthetic_bug_dataset.py collect",
        "output_root": str(output_root),
        "max_examples_per_property": args.max_examples_per_property,
        "domains": {art.domain: art.summary for art in artifacts_list},
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote synthetic dataset artifacts to: {output_root}")
    for domain in artifacts_list:
        print(
            f"- {domain.domain}: clean_repos={len(domain.clean_repos)} "
            f"bug_examples={len(domain.bug_examples)} "
            f"properties_with_examples={domain.summary.get('property_count_with_examples', 0)}"
        )


def _plan(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    rng = random.Random(args.seed)
    plan_rows: list[dict[str, Any]] = []

    for domain_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        clean_path = domain_dir / "clean_repos.jsonl"
        bug_path = domain_dir / "bug_examples.jsonl"
        if not clean_path.exists() or not bug_path.exists():
            continue
        clean_repos = _load_jsonl(clean_path)
        bug_examples = _load_jsonl(bug_path)

        by_dataset_prop: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for ex in bug_examples:
            dataset_name = str(ex.get("dataset") or "").strip() or "__all__"
            prop_id = str(ex.get("property_id") or "")
            if not prop_id:
                continue
            by_dataset_prop.setdefault(dataset_name, {}).setdefault(prop_id, []).append(ex)

        # Fallback bucket for artifacts that do not carry dataset names.
        fallback_props = by_dataset_prop.get("__all__", {})
        if not by_dataset_prop:
            continue

        def _make_plan_row(repo: dict[str, Any]) -> dict[str, Any]:
            repo_dataset = str(repo.get("dataset") or "").strip()
            dataset_props = by_dataset_prop.get(repo_dataset) or fallback_props
            prop_ids = sorted([p for p in dataset_props.keys() if p])
            available_properties = []
            for pid in prop_ids:
                first_example = dataset_props[pid][0] if dataset_props.get(pid) else {}
                available_properties.append(
                    {
                        "property_id": pid,
                        "property_text": first_example.get("property_text"),
                    }
                )
            max_k = min(len(prop_ids), max(int(args.max_properties_per_repo), 0))
            k = rng.randint(0, max_k)
            chosen_props = rng.sample(prop_ids, k) if k > 0 else []
            chosen_examples: list[dict[str, Any]] = []
            for pid in chosen_props:
                chosen_examples.append(rng.choice(dataset_props[pid]))
            return {
                "domain": domain_dir.name,
                "dataset": repo.get("dataset"),
                "repo_name": repo.get("repo_name"),
                "repo_slug": repo.get("repo_slug"),
                "repo_path": repo.get("repo_path"),
                "k": k,
                "selected_property_ids": chosen_props,
                "selected_bug_examples": chosen_examples,
                "available_properties": available_properties,
            }

        samples_per_subdataset = int(args.samples_per_subdataset or 0)
        if samples_per_subdataset > 0:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for repo in clean_repos:
                key = str(repo.get("dataset") or "unknown")
                grouped.setdefault(key, []).append(repo)
            for dataset_name in sorted(grouped):
                repos = grouped[dataset_name]
                if not repos:
                    continue
                for _ in range(samples_per_subdataset):
                    plan_rows.append(_make_plan_row(rng.choice(repos)))
        else:
            for repo in clean_repos:
                plan_rows.append(_make_plan_row(repo))

    out_path = Path(args.plan_path)
    _write_jsonl(out_path, plan_rows)
    print(f"Wrote injection plan rows: {len(plan_rows)} -> {out_path}")


def _inject(args: argparse.Namespace) -> None:
    if not args.confirm:
        raise SystemExit(
            "Refusing to run injection without --confirm. "
            "This phase performs LLM-driven code edits and should be explicitly approved."
        )
    from vibetest import TestCase, VibeTestAgent

    plan_path = Path(args.plan_path)
    if not plan_path.exists():
        raise SystemExit(f"Plan file not found: {plan_path}")
    plan_rows = _load_jsonl(plan_path)
    if not plan_rows:
        raise SystemExit(f"No plan rows found in: {plan_path}")

    domain_filters = {str(x).strip() for x in (args.domains or []) if str(x).strip()}
    if domain_filters:
        plan_rows = [r for r in plan_rows if str(r.get("domain") or "").strip() in domain_filters]

    dataset_filters = {str(x).strip() for x in (args.datasets or []) if str(x).strip()}
    if dataset_filters:
        plan_rows = [r for r in plan_rows if str(r.get("dataset") or "").strip() in dataset_filters]

    if not plan_rows:
        raise SystemExit(
            "No plan rows remain after applying filters "
            f"(domains={sorted(domain_filters) or 'ALL'}, datasets={sorted(dataset_filters) or 'ALL'})."
        )
    property_catalog = _property_catalog_from_plan_rows(plan_rows)

    offset = max(int(args.offset), 0)
    if offset:
        plan_rows = plan_rows[offset:]
    if args.limit and args.limit > 0:
        plan_rows = plan_rows[: int(args.limit)]

    output_root = Path(args.output_root)
    repos_root = output_root / "repos"
    labels_path = Path(args.labels_path)
    run_id = _slugify(args.run_id or _utc_now_iso(), max_len=64)
    model_name = str(args.model or "openai/gpt-5-mini")
    sandbox_name = str(args.sandbox or "docker")
    evidence_root = Path(args.evidence_root)
    prompt_preview_path = output_root / "injection_prompt_template.txt"
    output_root.mkdir(parents=True, exist_ok=True)

    # Save one template preview for inspection/reproducibility.
    prompt_preview_path.write_text(
        _build_injection_prompt(
            {
                "domain": "<domain>",
                "k": 1,
                "selected_property_ids": ["<property_id>"],
                "selected_bug_examples": [
                    {
                        "property_id": "<property_id>",
                        "property_text": "<property_text>",
                        "reason_text": "<reason>",
                        "evidence_text": "<evidence>",
                    }
                ],
                "available_properties": [
                    {
                        "property_id": "<property_id>",
                        "property_text": "<property_text>",
                    },
                    {
                        "property_id": "<other_property_id>",
                        "property_text": "<other_property_text>",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    agent = VibeTestAgent(model=model_name, static=bool(args.static))
    label_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(plan_rows):
        row_index = idx + offset
        repo_path_raw = str(row.get("repo_path") or "").strip()
        if not repo_path_raw:
            label_rows.append(
                {
                    "run_id": run_id,
                    "row_index": row_index,
                    "status": "error",
                    "error": "missing repo_path in plan row",
                    "plan_row": row,
                }
            )
            continue
        source_repo_path = Path(repo_path_raw)
        if not source_repo_path.exists():
            print(source_repo_path)
            label_rows.append(
                {
                    "run_id": run_id,
                    "row_index": row_index,
                    "status": "error",
                    "error": f"repo_path does not exist: {source_repo_path}",
                    "plan_row": row,
                }
            )
            continue

        domain = _slugify(str(row.get("domain") or "unknown"), max_len=60)
        repo_name = str(row.get("repo_name") or row.get("repo_slug") or source_repo_path.name)
        repo_slug = _slugify(repo_name, max_len=100)
        sample_name = f"inject_{row_index:06d}_{repo_slug}"
        prompt_row = dict(row)
        prompt_row["available_properties"] = _available_properties_for_plan_row(row, property_catalog)
        prompt = _build_injection_prompt(prompt_row)
        test_case = TestCase(
            name=sample_name,
            description=prompt,
            repo_path=source_repo_path,
            sandbox_path="/workspace",
            metadata={
                "run_id": run_id,
                "plan_row_index": row_index,
            },
        )
        pending_rows.append(
            {
                "row_index": row_index,
                "row": row,
                "source_repo_path": source_repo_path,
                "domain": domain,
                "repo_name": repo_name,
                "repo_slug": repo_slug,
                "sample_name": sample_name,
                "test_case": test_case,
            }
        )

    all_error: str | None = None
    result_by_name: dict[str, Any] = {}
    if pending_rows:
        all_test_cases = [entry["test_case"] for entry in pending_rows]
        print(f"Running injection eval with {len(all_test_cases)} sample(s).")
        try:
            all_results = agent.execute_tests(all_test_cases, sandbox=sandbox_name)
            result_by_name = {res.test_case.name: res for res in all_results}
        except Exception as exc:
            all_error = str(exc)

    for entry in pending_rows:
        row_index = int(entry["row_index"])
        row = entry["row"]
        source_repo_path = entry["source_repo_path"]
        domain = str(entry["domain"])
        repo_name = str(entry["repo_name"])
        repo_slug = str(entry["repo_slug"])
        sample_name = str(entry["sample_name"])

        result = result_by_name.get(sample_name)
        if result is None and all_error:
            result_text = f"Injection execution error: {all_error}"
        else:
            result_text = result.message if result is not None else ""
        parsed_report = _extract_json_object(result_text)
        evidence_tar = _evidence_tar_path(evidence_root, model_name, sample_name)

        sample_dir = repos_root / domain / f"{row_index:06d}_{repo_slug}"
        repo_dir_name = _slugify(str(row.get("repo_slug") or repo_slug), max_len=100)
        injected_repo_dir = sample_dir / repo_dir_name
        ok_extract, extract_info = _extract_injection_artifacts(
            evidence_tar,
            output_repo_dir=injected_repo_dir,
        )
        ground_truth = _ground_truth_labels_for_row(
            row,
            extract_info=extract_info if ok_extract else None,
            property_catalog=property_catalog,
        )
        diff_info: dict[str, Any] | None = None
        if ok_extract:
            diff_patch_path = sample_dir / "injection.diff.patch"
            diff_meta_path = sample_dir / "injection.diff.json"
            try:
                diff_info = _compute_repo_diff(
                    source_repo_path,
                    injected_repo_dir,
                    diff_path=diff_patch_path,
                    diff_meta_path=diff_meta_path,
                    max_file_bytes=int(args.diff_max_file_bytes),
                )
            except Exception as exc:
                diff_info = {
                    "error": f"Failed to compute repo diff: {exc}",
                    "original_repo": str(source_repo_path),
                    "injected_repo": str(injected_repo_dir),
                    "diff_patch_path": str(diff_patch_path),
                    "diff_meta_path": str(diff_meta_path),
                }

        status = "ok" if ok_extract else "error"
        row_payload = {
            "run_id": run_id,
            "row_index": row_index,
            "status": status,
            "domain": row.get("domain"),
            "dataset": row.get("dataset"),
            "repo_name": repo_name,
            "repo_slug": row.get("repo_slug"),
            "source_repo_path": str(source_repo_path),
            "output_repo_path": str(injected_repo_dir),
            "output_sample_path": str(sample_dir),
            "k": row.get("k"),
            "selected_property_ids": row.get("selected_property_ids", []),
            "selected_bug_examples": row.get("selected_bug_examples", []),
            "agent_model": model_name,
            "agent_static": bool(args.static),
            "sandbox": sandbox_name,
            "sample_name": sample_name,
            "result_passed": bool(result.passed) if result is not None else False,
            "result_message": result_text,
            "result_report_json": parsed_report,
            "artifact_extraction": extract_info,
            "repo_diff": diff_info,
            "ground_truth": ground_truth,
            "ground_truth_by_property": ground_truth.get("by_property", []),
            "ground_truth_property_labels": ground_truth.get("property_labels", {}),
            "ground_truth_violation_descriptions": ground_truth.get("violation_descriptions", {}),
            "timestamp_utc": _utc_now_iso(),
        }
        if all_error and result is None:
            row_payload["execution_error"] = all_error
        label_rows.append(row_payload)

    _write_jsonl(labels_path, label_rows)
    summary = {
        "run_id": run_id,
        "plan_path": str(plan_path),
        "labels_path": str(labels_path),
        "output_root": str(output_root),
        "repos_root": str(repos_root),
        "model": model_name,
        "sandbox": sandbox_name,
        "domain_filters": sorted(domain_filters),
        "dataset_filters": sorted(dataset_filters),
        "static": bool(args.static),
        "row_count": len(label_rows),
        "ok_count": sum(1 for r in label_rows if r.get("status") == "ok"),
        "error_count": sum(1 for r in label_rows if r.get("status") != "ok"),
        "timestamp_utc": _utc_now_iso(),
    }
    (output_root / "injection_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote labels: {labels_path}")
    print(f"Wrote run summary: {output_root / 'injection_run_summary.json'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Build synth-data clean sets and bug exemplars.")
    collect.add_argument("--results-dir", default="results", help="Directory containing experiment JSONL results.")
    collect.add_argument("--data-dir", default="data", help="Read-only source data root.")
    collect.add_argument(
        "--human-annotations-long",
        default="results/human-annotations/human_annotations_long.csv",
        help="Long-format human annotation CSV.",
    )
    collect.add_argument("--output-root", default="synth-data", help="Output directory for synthetic artifacts.")
    collect.add_argument(
        "--max-examples-per-property",
        type=int,
        default=10,
        help="Maximum bug exemplars to keep per property.",
    )
    collect.add_argument(
        "--allow-unverified-hallucination-fallback",
        action="store_true",
        help="If hallucination AT verified labels are absent, use unverified AT FAIL examples as fallback.",
    )
    collect.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAIN_CHOICES,
        default=list(DOMAIN_CHOICES),
        help="Domains to collect (default: all).",
    )
    collect.add_argument(
        "--equalize-clean-repos-per-subdataset",
        action="store_true",
        help="Downsample clean repos to have equal counts per subdataset (uses 'dataset' field).",
    )
    collect.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for deterministic downsampling.",
    )
    collect.set_defaults(func=_collect)

    plan = sub.add_parser("plan", help="Sample bug-injection assignments for clean repos.")
    plan.add_argument("--output-root", default="synth-data", help="Synthetic artifacts root from collect.")
    plan.add_argument(
        "--plan-path",
        default="synth-data/injection_plan.jsonl",
        help="Output JSONL path for planned injections.",
    )
    plan.add_argument(
        "--max-properties-per-repo",
        type=int,
        default=9999,
        help="Upper bound for sampled k properties per clean repo.",
    )
    plan.add_argument(
        "--samples-per-subdataset",
        type=int,
        default=0,
        help="If >0, sample exactly this many plan rows per subdataset (dataset field), with replacement.",
    )
    plan.add_argument("--seed", type=int, default=1337, help="Random seed.")
    plan.set_defaults(func=_plan)

    inject = sub.add_parser("inject", help="Run LLM bug injection from an existing injection plan.")
    inject.add_argument("--plan-path", default="synth-data/injection_plan.jsonl")
    inject.add_argument("--output-root", default="synth-data/injected", help="Output directory for injected repos.")
    inject.add_argument(
        "--labels-path",
        default="synth-data/injected/labels.jsonl",
        help="Output JSONL path for injection labels/metadata.",
    )
    inject.add_argument("--model", default="openai/gpt-5-mini", help="Model for injection agent.")
    inject.add_argument(
        "--sandbox",
        default="docker",
        help="Sandbox backend passed to VibeTestAgent.execute_tests (default: docker).",
    )
    inject.add_argument(
        "--evidence-root",
        default="evidence-dumps",
        help="Root directory where agent evidence tars are written.",
    )
    inject.add_argument(
        "--diff-max-file-bytes",
        type=int,
        default=2_000_000,
        help="Max bytes per file preview while computing diffs (full text compare still used for UTF-8 text files).",
    )
    inject.add_argument("--run-id", default="", help="Optional run ID (default: UTC timestamp).")
    inject.add_argument(
        "--domains",
        nargs="+",
        default=[],
        help="Optional domain filters applied to plan rows before injection (e.g., ml-bugs security-vuln).",
    )
    inject.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Optional dataset filters applied to plan rows before injection (e.g., kaggle_titanic).",
    )
    inject.add_argument("--limit", type=int, default=0, help="Optional number of plan rows to execute.")
    inject.add_argument("--offset", type=int, default=0, help="Optional plan row offset.")
    inject.add_argument(
        "--static",
        action="store_true",
        help="Run agent in static mode (no command execution in sandbox).",
    )
    inject.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a row fails.",
    )
    inject.add_argument("--confirm", action="store_true", help="Required acknowledgement before injection.")
    inject.set_defaults(func=_inject)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
