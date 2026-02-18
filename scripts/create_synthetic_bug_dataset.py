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


def _build_injection_prompt(plan_row: dict[str, Any], *, sandbox_repo_path: str = "/workspace/repo") -> str:
    selected_examples = plan_row.get("selected_bug_examples") or []
    selected_props = plan_row.get("selected_property_ids") or []
    k = int(plan_row.get("k") or 0)
    examples_text = _build_bug_examples_text(selected_examples)
    domain = str(plan_row.get("domain") or "")
    return (
        "You are an expert software security/ML engineer. Modify the repository to inject synthetic bugs corresponding to failures of listed properties.\n\n"
        f"Domain: {domain}\n"
        f"Repository path: {sandbox_repo_path}\n"
        f"Number of target property failures to inject (k): {k}\n"
        f"Target property IDs which injected bugs should violate: {selected_props}\n\n"
        "Injection goals:\n"
        "1. Inject ONLY bugs corresponding to failures of the target properties.\n"
        "2. Edit existing repository files in place.\n"
        "3. Do NOT create backups or duplicate clean copies (for example: .bak/.orig files,\n"
        "   renamed originals, copy files, or parallel clean versions).\n"
        "4. Keep edits minimal and realistic.\n"
        "5. Do not add obvious markers like 'TODO bug' or 'synthetic bug'.\n"
        "6. Do not intentionally corrupt unrelated functionality.\n"
        "7. If k=0, keep code behavior unchanged and only produce the report/artifacts.\n\n"
        "Bug exemplars to imitate:\n"
        f"{examples_text}\n\n"
        "Required artifacts (must be created before submit):\n"
        "1. `/evidence/artifacts/injection_report.json` with keys:\n"
        "   - injected_property_ids (array)\n"
        "   - changed_files (array of relative paths)\n"
        "   - injection_summary (string)\n"
        "   - bug_locations (array of {file, hint})\n"
        "2. `/evidence/artifacts/repo.tar.gz` containing the full modified repo root contents.\n"
        "   Example command:\n"
        f"   `mkdir -p /evidence/artifacts && tar -czf /evidence/artifacts/repo.tar.gz -C {sandbox_repo_path} .`\n\n"
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
                tf.extractall(temp_dir)
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

        if output_repo_dir.exists():
            shutil.rmtree(output_repo_dir)
        output_repo_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(repo_tgz, "r:gz") as tf:
                tf.extractall(output_repo_dir)
        except Exception as exc:
            info["error"] = f"Failed to extract injected repo archive: {exc}"
            return False, info

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
                b_text = b_data.decode("utf-8", errors="replace").splitlines(keepends=True)
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
        patch_parts.append(f"diff --git a/{rel} b/{rel}\n")
        a_lines = a_text_full.splitlines(keepends=True)
        b_lines = b_text_full.splitlines(keepends=True)
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

    clean_state: dict[tuple[str, str], dict[str, Any]] = {}
    bug_examples: list[dict[str, Any]] = []

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
                repo_path = data_dir / "vuln" / "cwe-bench" / "repos" / repo_slug
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
    source_file = _resolve_existing([results_dir / "hallucination_AT-gpt-5-mini.jsonl"])
    method_run_name = source_file.stem
    rows = _load_jsonl(source_file)
    verified = _load_human_c_labels(human_long_csv, method_prefixes=("hallucination_",))
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
                        "verification_source": "unverified_vibetester_fail_fallback",
                        "method_run_name": method_run_name,
                    }
                )

        if has_verified_labels:
            is_clean = (not has_verified_fail) and (not has_inconclusive)
            clean_criterion = "no_verified_fail_and_no_inconclusive"
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
        "has_verified_hallucination_at_labels": has_verified_labels,
        "clean_repo_count": len(clean_repos),
        "bug_example_count": len(bug_examples),
        "property_count_with_examples": len({e["property_id"] for e in bug_examples}),
        "clean_definition": (
            "no verified FAIL and no INCONCLUSIVE in vibetester output"
            if has_verified_labels
            else "strict all-PASS and no INCONCLUSIVE in vibetester output"
        ),
        "bug_example_definition": (
            "verified FAIL from vibetester outputs via human C labels"
            if has_verified_labels
            else "unverified FAIL fallback from vibetester outputs (no AT human labels found)"
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

        by_prop: dict[str, list[dict[str, Any]]] = {}
        for ex in bug_examples:
            by_prop.setdefault(str(ex.get("property_id") or ""), []).append(ex)
        prop_ids = sorted([p for p in by_prop.keys() if p])
        if not prop_ids:
            continue

        for repo in clean_repos:
            max_k = min(len(prop_ids), max(int(args.max_properties_per_repo), 0))
            k = rng.randint(0, max_k)
            chosen_props = rng.sample(prop_ids, k) if k > 0 else []
            chosen_examples: list[dict[str, Any]] = []
            for pid in chosen_props:
                chosen_examples.append(rng.choice(by_prop[pid]))
            plan_rows.append(
                {
                    "domain": domain_dir.name,
                    "repo_name": repo.get("repo_name"),
                    "repo_slug": repo.get("repo_slug"),
                    "repo_path": repo.get("repo_path"),
                    "k": k,
                    "selected_property_ids": chosen_props,
                    "selected_bug_examples": chosen_examples,
                }
            )

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
            }
        )
        + "\n",
        encoding="utf-8",
    )

    agent = VibeTestAgent(model=model_name, static=bool(args.static))
    label_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(plan_rows):
        repo_path_raw = str(row.get("repo_path") or "").strip()
        if not repo_path_raw:
            label_rows.append(
                {
                    "run_id": run_id,
                    "row_index": idx + offset,
                    "status": "error",
                    "error": "missing repo_path in plan row",
                    "plan_row": row,
                }
            )
            continue
        source_repo_path = Path(repo_path_raw)
        if not source_repo_path.exists():
            label_rows.append(
                {
                    "run_id": run_id,
                    "row_index": idx + offset,
                    "status": "error",
                    "error": f"repo_path does not exist: {source_repo_path}",
                    "plan_row": row,
                }
            )
            continue

        domain = _slugify(str(row.get("domain") or "unknown"), max_len=60)
        repo_name = str(row.get("repo_name") or row.get("repo_slug") or source_repo_path.name)
        repo_slug = _slugify(repo_name, max_len=100)
        sample_name = f"inject_{idx + offset:06d}_{repo_slug}"
        prompt = _build_injection_prompt(row)
        test_case = TestCase(
            name=sample_name,
            description=prompt,
            repo_path=source_repo_path,
            sandbox_path="/workspace",
            metadata={
                "run_id": run_id,
                "plan_row_index": idx + offset,
            },
        )

        results = agent.execute_tests([test_case], sandbox=sandbox_name)
        result = results[0] if results else None
        result_text = result.message if result is not None else ""
        parsed_report = _extract_json_object(result_text)
        evidence_tar = _evidence_tar_path(evidence_root, model_name, sample_name)

        target_dir = repos_root / domain / f"{idx + offset:06d}_{repo_slug}"
        ok_extract, extract_info = _extract_injection_artifacts(evidence_tar, output_repo_dir=target_dir)
        diff_info: dict[str, Any] | None = None
        if ok_extract:
            diff_patch_path = target_dir / "injection.diff.patch"
            diff_meta_path = target_dir / "injection.diff.json"
            try:
                diff_info = _compute_repo_diff(
                    source_repo_path,
                    target_dir,
                    diff_path=diff_patch_path,
                    diff_meta_path=diff_meta_path,
                    max_file_bytes=int(args.diff_max_file_bytes),
                )
            except Exception as exc:
                diff_info = {
                    "error": f"Failed to compute repo diff: {exc}",
                    "original_repo": str(source_repo_path),
                    "injected_repo": str(target_dir),
                    "diff_patch_path": str(diff_patch_path),
                    "diff_meta_path": str(diff_meta_path),
                }

        status = "ok" if ok_extract else "error"
        label_rows.append(
            {
                "run_id": run_id,
                "row_index": idx + offset,
                "status": status,
                "domain": row.get("domain"),
                "repo_name": repo_name,
                "repo_slug": row.get("repo_slug"),
                "source_repo_path": str(source_repo_path),
                "output_repo_path": str(target_dir),
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
                "timestamp_utc": _utc_now_iso(),
            }
        )

        if bool(args.stop_on_error) and status != "ok":
            break

    _write_jsonl(labels_path, label_rows)
    summary = {
        "run_id": run_id,
        "plan_path": str(plan_path),
        "labels_path": str(labels_path),
        "output_root": str(output_root),
        "repos_root": str(repos_root),
        "model": model_name,
        "sandbox": sandbox_name,
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
