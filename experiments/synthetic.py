"""Run synthetic injected-bug experiments with explicit ground-truth scoring.

This script consumes injected synthetic labels from `synth-data/injected/*.jsonl`,
runs one method over the injected repos, and writes standardized JSONL results with
per-property predictions plus explicit correctness scoring against synthetic ground truth.

Scoring rules:
- PASS/FAIL prediction that disagrees with ground-truth binary label is incorrect.
- PASS on a ground-truth PASS is correct.
- FAIL on a ground-truth FAIL is correct only if an LLM verifier judges the
  predicted FAIL evidence as matching the ground-truth violation description.
- INCONCLUSIVE/NOT APPLICABLE are treated as abstentions (not correct).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonlines
from inspect_ai.model import GenerateConfig, get_model

from vibetest import TestCase, VibeTestAgent
from vibetest.agent import ClaudeCodeVibeTestAgent, CodexReviewAgent, CodexVibeTestAgent, EvidenceVerifierAgent
from vibetest.baselines import (
    analyze_repo_with_codeql,
    map_review_to_hallucination_tests,
    map_review_to_kaggle_tests,
    map_review_to_vuln_tests,
    prepare_reference_invariants,
    run_refchecker,
    run_traincheck,
)

try:
    from experiments.result_naming import standardized_results_path
    from experiments.usage_utils import (
        aggregate_usage_from_results,
        aggregate_usage_from_tests,
        usage_from_result_metadata,
    )
except ImportError:
    from result_naming import standardized_results_path
    from usage_utils import (
        aggregate_usage_from_results,
        aggregate_usage_from_tests,
        usage_from_result_metadata,
    )


VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE", "NOT APPLICABLE"}


@dataclass(frozen=True)
class PropertyGT:
    property_id: str
    property_text: str
    label: int  # 0=PASS, 1=FAIL in ground truth
    gt_violation_description: str


@dataclass(frozen=True)
class SyntheticCase:
    row_index: int
    domain: str
    dataset: str
    repo_name: str
    repo_slug: str
    source_repo_path: Path
    sample_path: Path
    repo_path: Path
    properties: list[PropertyGT]
    raw_row: dict[str, Any]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_verdict(raw: Any, *, passed: Any = None) -> str:
    text = _normalize_whitespace(str(raw) if raw is not None else "").upper()
    if text in VERDICTS:
        return text
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    if text.startswith("P"):
        return "PASS"
    if text.startswith("F"):
        return "FAIL"
    if text.startswith("I"):
        return "INCONCLUSIVE"
    return "INCONCLUSIVE"


def _model_suffix(model_name: str) -> str:
    return model_name.split("/")[-1] if model_name else "unknown-model"


def _slug_sample_component(value: str, *, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip()).strip("_")
    return (slug or "unknown")[:max_len]


def _synthetic_property_sample_id(case: "SyntheticCase", prop: "PropertyGT") -> str:
    dataset = _slug_sample_component(case.dataset, max_len=60)
    return f"{dataset}_row{case.row_index}_{prop.property_id}"


def _safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_root = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = (dest_root / member.name).resolve()
            if not str(member_path).startswith(str(dest_root) + "/") and member_path != dest_root:
                raise ValueError(f"Unsafe path in evidence tar: {member.name}")
        tf.extractall(dest_root, filter="data")


def _prepare_evidence_bundle(
    *,
    sample_id: str,
    model_name: str,
    evidence_root: Path,
    temp_dirs: list[Path],
) -> tuple[dict[str, str], str, str]:
    evidence_tar = evidence_root / _model_suffix(model_name) / f"evidence-{sample_id}.tar.gz"
    if not evidence_tar.exists():
        return {}, str(evidence_tar), f"Evidence tar not found: {evidence_tar}"

    temp_dir = Path(tempfile.mkdtemp(prefix="synthetic-verifier-evidence-"))
    temp_dirs.append(temp_dir)
    try:
        _safe_extract_tar(evidence_tar, temp_dir)
    except Exception as exc:
        return {}, str(evidence_tar), f"Failed to extract evidence tar {evidence_tar}: {exc}"

    evidence_dir = temp_dir / "evidence"
    if not evidence_dir.exists():
        return {}, str(evidence_tar), f"Evidence tar has no evidence/ directory: {evidence_tar}"
    return {str(evidence_dir): "/"}, str(evidence_tar), ""


def _property_sort_key(pid: str) -> tuple[int, int | str, str]:
    pid = (pid or "").strip()
    m = re.search(r"(?:_p|_vuln)(\d+)$", pid, re.IGNORECASE)
    if m:
        return (0, int(m.group(1)), pid)
    m = re.search(r"CWE-(\d+)", pid, re.IGNORECASE)
    if m:
        return (1, int(m.group(1)), pid)
    return (2, pid, pid)


def _extract_cwe_num(property_id: str, property_text: str) -> str | None:
    for text in (property_id or "", property_text or ""):
        m = re.search(r"CWE-(\d+)", text, re.IGNORECASE)
        if m:
            return str(int(m.group(1)))
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue

    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _resolve_injected_paths(row: dict[str, Any]) -> tuple[Path | None, Path | None, str | None]:
    """Return (repo_path, sample_path, error) for old and new synthetic labels."""
    raw_repo_path = str(row.get("output_repo_path") or "").strip()
    if not raw_repo_path:
        return None, None, "missing output_repo_path"

    repo_path = Path(raw_repo_path)
    if not repo_path.exists():
        return None, None, f"output_repo_path does not exist: {repo_path}"

    raw_sample_path = str(row.get("output_sample_path") or "").strip()
    sample_path = Path(raw_sample_path) if raw_sample_path else repo_path

    # Legacy Kaggle archives stored the sample wrapper in output_repo_path:
    #   sample/
    #     actual-kaggle-repo/
    #     injection.diff.patch
    #     injection.diff.json
    # Newer labels store output_repo_path as the actual injected repo and
    # output_sample_path as the wrapper. Resolve the legacy form here so the
    # experiment agent inspects the same repo root used during injection.
    has_injection_artifacts = (
        (repo_path / "injection.diff.patch").exists()
        or (repo_path / "injection.diff.json").exists()
    )
    if raw_sample_path:
        return repo_path, sample_path, None
    if has_injection_artifacts and repo_path.is_dir():
        child_dirs = [
            p
            for p in repo_path.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
        ]
        if len(child_dirs) == 1:
            return child_dirs[0], repo_path, None

    return repo_path, sample_path, None


def _test_evidence_text(test: dict[str, Any]) -> str:
    meta = test.get("metadata") or {}
    if meta.get("evidence_text"):
        return str(meta.get("evidence_text"))

    evidence = test.get("evidence")
    if isinstance(evidence, str):
        return evidence

    parts: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                for key in ("description", "text", "content", "value"):
                    value = item.get(key)
                    if value:
                        parts.append(str(value))
                        break
    return "\n".join(parts)


def _tests_from_props(
    props: list[PropertyGT],
    *,
    verdict: str,
    reason: str,
    reviewer: str,
    mapper_model: str | None,
) -> list[dict[str, Any]]:
    verdict = _normalize_verdict(verdict)
    passed = verdict == "PASS"
    out: list[dict[str, Any]] = []
    for idx, prop in enumerate(props):
        out.append(
            {
                "description": reason,
                "passed": passed,
                "evidence": [],
                "execution_log": "",
                "metadata": {
                    "reviewer": reviewer,
                    "mapper_model": mapper_model,
                    "property_index": idx,
                    "property_id": prop.property_id,
                    "property_text": prop.property_text,
                    "verdict": verdict,
                    "evidence_text": "",
                },
            }
        )
    return out


def _all_pass_tests(props: list[PropertyGT], *, reviewer: str, mapper_model: str | None, reason: str) -> list[dict[str, Any]]:
    return _tests_from_props(props, verdict="PASS", reason=reason, reviewer=reviewer, mapper_model=mapper_model)


def _all_inconclusive_tests(
    props: list[PropertyGT],
    *,
    reviewer: str,
    mapper_model: str | None,
    reason: str,
) -> list[dict[str, Any]]:
    return _tests_from_props(props, verdict="INCONCLUSIVE", reason=reason, reviewer=reviewer, mapper_model=mapper_model)


def _augment_tests_with_review(tests: list[dict[str, Any]], review_text: str, extra_meta: dict[str, Any]) -> None:
    excerpt = review_text[:12000] if review_text else ""
    for t in tests:
        t["execution_log"] = excerpt
        md = t.get("metadata") or {}
        md.update(extra_meta)
        t["metadata"] = md


def _load_injected_cases(
    labels_path: Path,
    *,
    domains: set[str],
    datasets: set[str],
    repo_limit: int,
    repo_offset: int,
) -> list[SyntheticCase]:
    if not labels_path.exists():
        raise SystemExit(f"Injected labels file not found: {labels_path}")

    rows: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    out: list[SyntheticCase] = []
    skipped = 0
    skipped_path_errors: list[str] = []
    for row in rows:
        if row.get("status") != "ok":
            continue

        domain = str(row.get("domain") or "").strip()
        dataset = str(row.get("dataset") or "").strip()
        if domains and domain not in domains:
            continue
        if datasets and dataset not in datasets:
            continue

        if skipped < max(repo_offset, 0):
            skipped += 1
            continue

        repo_path, sample_path, path_error = _resolve_injected_paths(row)
        if path_error or repo_path is None or sample_path is None:
            if len(skipped_path_errors) < 5:
                skipped_path_errors.append(
                    f"row_index={row.get('row_index', len(out))}: {path_error or 'unresolved repo path'}"
                )
            continue

        by_prop = row.get("ground_truth_by_property") or []
        gt_labels = row.get("ground_truth_property_labels") or {}
        gt_desc = row.get("ground_truth_violation_descriptions") or {}

        props: list[PropertyGT] = []
        if isinstance(by_prop, list) and by_prop:
            for item in by_prop:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("property_id") or "").strip()
                if not pid:
                    continue
                ptxt = str(item.get("property_text") or "")
                label = int(item.get("label", 1 if str(item.get("verdict", "")).upper() == "FAIL" else 0))
                vdesc = str(item.get("gt_violation_description") or "")
                props.append(
                    PropertyGT(
                        property_id=pid,
                        property_text=ptxt,
                        label=1 if label else 0,
                        gt_violation_description=vdesc,
                    )
                )
        else:
            # Fallback for older rows without `ground_truth_by_property`.
            prop_texts: dict[str, str] = {}
            for ex in row.get("selected_bug_examples") or []:
                if not isinstance(ex, dict):
                    continue
                pid = str(ex.get("property_id") or "").strip()
                if pid:
                    prop_texts[pid] = str(ex.get("property_text") or "")
            for pid, label_val in gt_labels.items():
                pid_s = str(pid).strip()
                if not pid_s:
                    continue
                props.append(
                    PropertyGT(
                        property_id=pid_s,
                        property_text=prop_texts.get(pid_s, ""),
                        label=1 if int(label_val) else 0,
                        gt_violation_description=str(gt_desc.get(pid_s) or ""),
                    )
                )

        if not props:
            continue
        props.sort(key=lambda p: _property_sort_key(p.property_id))

        out.append(
            SyntheticCase(
                row_index=int(row.get("row_index") or len(out)),
                domain=domain,
                dataset=dataset,
                repo_name=str(row.get("repo_name") or repo_path.name),
                repo_slug=str(row.get("repo_slug") or ""),
                source_repo_path=Path(str(row.get("source_repo_path") or "").strip()) if row.get("source_repo_path") else Path(),
                sample_path=sample_path,
                repo_path=repo_path,
                properties=props,
                raw_row=row,
            )
        )

        if repo_limit > 0 and len(out) >= repo_limit:
            break

    if not out:
        detail = ""
        if skipped_path_errors:
            detail = " Path resolution examples: " + "; ".join(skipped_path_errors)
        raise SystemExit(f"No injected rows matched filters (status=ok, domains/datasets, offset/limit).{detail}")
    return out


def _serialize_test_result(result, prop: PropertyGT, prop_idx: int) -> dict[str, Any]:
    metadata = dict(result.metadata or {})
    verdict = _normalize_verdict(metadata.get("verdict"), passed=result.passed)
    metadata.update(
        {
            "property_index": prop_idx,
            "property_id": prop.property_id,
            "property_text": prop.property_text,
            "verdict": verdict,
            "evidence_text": str(metadata.get("evidence_text") or ""),
        }
    )
    return {
        "description": result.message,
        "passed": verdict == "PASS",
        "evidence": [e.model_dump() for e in (result.evidence or [])],
        "execution_log": result.execution_log,
        "metadata": metadata,
    }


def _verdict_from_result(result) -> str:
    metadata = result.metadata or {}
    return _normalize_verdict(metadata.get("verdict"), passed=result.passed)


def _verifier_score(result) -> float | None:
    metadata = result.metadata or {}
    raw_score = metadata.get("evidence_support_score")
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return None


def _format_verifier_feedback(verifier_result, threshold: float) -> str:
    metadata = verifier_result.metadata or {}
    score = _verifier_score(verifier_result)
    score_text = "unparseable" if score is None else f"{score:.3f}"
    reason = str(metadata.get("reason_text") or "").strip()
    assessment = str(metadata.get("evidence_assessment") or "").strip()
    raw = str(verifier_result.message or "").strip()
    return f"""Previous VibeTest FAIL evidence was judged insufficient by an independent clean-context verifier.
Verifier support score: {score_text} (required >= {threshold:.3f}).
Verifier reason:
{reason or "(not provided)"}

Verifier evidence assessment:
{assessment or raw[:4000] or "(not provided)"}

When rerunning, do not simply repeat the prior unsupported FAIL evidence. Re-check the repository and either provide concrete, independently checkable evidence for FAIL with valid citations/artifacts, or return PASS/INCONCLUSIVE if the evidence does not support a high-confidence FAIL."""


def _make_vibetest_case(
    case: SyntheticCase,
    prop: PropertyGT,
    prop_idx: int,
    *,
    attempt: int,
    feedback: str = "",
) -> TestCase:
    base_name = _synthetic_property_sample_id(case, prop)
    name = base_name if attempt == 1 else f"{base_name}_iter{attempt}"
    return TestCase(
        name=name,
        description=prop.property_text,
        extra_instructions=feedback or None,
        repo_path=case.repo_path,
        sandbox_path="/workspace",
        metadata={
            "property_id": prop.property_id,
            "property_index": prop_idx,
            "source_sample_id": name,
            "iteration": attempt,
            "base_sample_id": base_name,
        },
    )


def _make_verifier_case(
    *,
    source_result,
    source_case: TestCase,
    verifier_sample_id: str,
    model_name: str,
    evidence_root: Path,
    temp_dirs: list[Path],
) -> TestCase:
    metadata = source_result.metadata or {}
    additional_data, evidence_tar, evidence_error = _prepare_evidence_bundle(
        sample_id=source_case.name,
        model_name=model_name,
        evidence_root=evidence_root,
        temp_dirs=temp_dirs,
    )
    return TestCase(
        name=verifier_sample_id,
        description=source_case.description,
        repo_path=source_case.repo_path,
        sandbox_path=source_case.sandbox_path,
        additional_data=additional_data,
        metadata={
            "source_sample_id": source_case.name,
            "evidence_tar": evidence_tar,
            "evidence_artifacts_available": bool(additional_data),
            "evidence_artifacts_error": evidence_error,
            "original_reason": str(metadata.get("reason_text") or "").strip(),
            "original_evidence": str(metadata.get("evidence_text") or "").strip(),
            "original_output": str(source_result.message or ""),
        },
    )


def _verifier_payload(result) -> dict[str, Any]:
    metadata = result.metadata or {}
    return {
        "score": metadata.get("evidence_support_score"),
        "verifier_model": metadata.get("verifier_model"),
        "reason_text": metadata.get("reason_text") or "",
        "evidence_assessment": metadata.get("evidence_assessment") or "",
        "raw_output": result.message,
        "execution_log": result.execution_log,
        "source_sample_id": result.test_case.metadata.get("source_sample_id"),
        "evidence_tar": result.test_case.metadata.get("evidence_tar"),
        "evidence_artifacts_available": result.test_case.metadata.get("evidence_artifacts_available"),
        "evidence_artifacts_error": result.test_case.metadata.get("evidence_artifacts_error"),
    }


def _ensure_property_alignment(tests: list[dict[str, Any]], props: list[PropertyGT], reviewer: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, prop in enumerate(props):
        if idx < len(tests):
            t = tests[idx]
            md = dict(t.get("metadata") or {})
            md.setdefault("reviewer", reviewer)
            md["property_index"] = idx
            md["property_id"] = prop.property_id
            md["property_text"] = prop.property_text
            md["verdict"] = _normalize_verdict(md.get("verdict"), passed=t.get("passed"))
            md["evidence_text"] = str(md.get("evidence_text") or "")
            t["metadata"] = md
            t["passed"] = md["verdict"] == "PASS"
            out.append(t)
        else:
            out.append(
                {
                    "description": "Property missing from mapped output.",
                    "passed": False,
                    "evidence": [],
                    "execution_log": "",
                    "metadata": {
                        "reviewer": reviewer,
                        "property_index": idx,
                        "property_id": prop.property_id,
                        "property_text": prop.property_text,
                        "verdict": "INCONCLUSIVE",
                        "evidence_text": "",
                    },
                }
            )
    return out


def _entry_from_tests(
    case: SyntheticCase,
    tests: list[dict[str, Any]],
    *,
    usage: dict[str, Any] | None,
    method: str,
    method_model: str | None,
) -> dict[str, Any]:
    aligned = _ensure_property_alignment(tests, case.properties, reviewer=method)

    passed = 0
    failed = 0
    inconclusive = 0
    for t in aligned:
        verdict = _normalize_verdict((t.get("metadata") or {}).get("verdict"), passed=t.get("passed"))
        if verdict == "PASS":
            passed += 1
        elif verdict == "FAIL":
            failed += 1
        else:
            inconclusive += 1

    gt_labels = {p.property_id: p.label for p in case.properties}
    gt_desc = {p.property_id: p.gt_violation_description for p in case.properties if p.label == 1}
    gt_by_property = [
        {
            "property_id": p.property_id,
            "property_text": p.property_text,
            "label": p.label,
            "verdict": "FAIL" if p.label == 1 else "PASS",
            "gt_violation_description": p.gt_violation_description if p.label == 1 else "",
        }
        for p in case.properties
    ]

    return {
        "repo": str(case.repo_path),
        "repo_name": case.repo_name,
        "domain": case.domain,
        "dataset": case.dataset,
        "synthetic_row_index": case.row_index,
        "source_repo_path": str(case.source_repo_path) if case.source_repo_path else "",
        "injected_sample_path": str(case.sample_path),
        "injected_repo_path": str(case.repo_path),
        "method": method,
        "method_model": method_model,
        "total_tests": len(aligned),
        "passed_tests": passed,
        "failed_tests": failed,
        "inconclusive_tests": inconclusive,
        "tests": aligned,
        "ground_truth_property_labels": gt_labels,
        "ground_truth_violation_descriptions": gt_desc,
        "ground_truth_by_property": gt_by_property,
        "usage": usage or aggregate_usage_from_tests(aligned),
    }


def _run_vibetest(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with VibeTest")
    print("=" * 80)

    all_test_cases: list[TestCase] = []
    case_ranges: list[tuple[int, int]] = []

    for case in cases:
        start = len(all_test_cases)
        for idx, prop in enumerate(case.properties):
            all_test_cases.append(
                TestCase(
                    name=_synthetic_property_sample_id(case, prop),
                    description=prop.property_text,
                    repo_path=case.repo_path,
                    sandbox_path="/workspace",
                    metadata={
                        "property_id": prop.property_id,
                        "property_index": idx,
                        "source_sample_id": _synthetic_property_sample_id(case, prop),
                    },
                )
            )
        case_ranges.append((start, len(all_test_cases)))

    agent = VibeTestAgent(model=args.model or "openai/gpt-5-mini", static=not bool(args.dynamic))
    all_results = agent.execute_tests(all_test_cases, sandbox=args.sandbox)

    entries: list[dict[str, Any]] = []
    for case, (start, end) in zip(cases, case_ranges):
        case_results = all_results[start:end]
        tests = [
            _serialize_test_result(r, case.properties[idx], idx)
            for idx, r in enumerate(case_results)
        ]
        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=aggregate_usage_from_results(case_results),
                method="AT",
                method_model=agent.model_name,
            )
        )
    return entries


def _run_vibetest_iterative(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with iterative VibeTest + evidence verifier")
    print("=" * 80)

    max_iterations = max(1, int(args.max_iterations or 3))
    threshold = float(args.verifier_threshold)
    vibetest_model = args.model or "openai/gpt-5-mini"
    verifier_model = args.verifier_model or vibetest_model
    evidence_root = Path(args.evidence_root)
    agent = VibeTestAgent(model=vibetest_model, static=not bool(args.dynamic))
    verifier = EvidenceVerifierAgent(
        model=verifier_model,
        static=bool(args.verifier_static),
        log_dir=args.verifier_log_dir,
    )

    slots: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        for prop_idx, prop in enumerate(case.properties):
            slots.append(
                {
                    "case_index": case_idx,
                    "prop_index": prop_idx,
                    "case": case,
                    "prop": prop,
                    "latest_result": None,
                    "latest_verifier": None,
                    "attempts": [],
                    "attempt_results": [],
                    "verifier_results": [],
                    "feedback": "",
                    "done": False,
                }
            )

    for attempt in range(1, max_iterations + 1):
        pending = [slot for slot in slots if not slot["done"]]
        if not pending:
            break

        test_cases = [
            _make_vibetest_case(
                slot["case"],
                slot["prop"],
                int(slot["prop_index"]),
                attempt=attempt,
                feedback=str(slot.get("feedback") or ""),
            )
            for slot in pending
        ]
        print(f"Iteration {attempt}/{max_iterations}: running VibeTest on {len(test_cases)} property case(s).")
        results = agent.execute_tests(test_cases, sandbox=args.sandbox)

        fail_slots: list[dict[str, Any]] = []
        verifier_cases: list[TestCase] = []
        temp_dirs: list[Path] = []
        try:
            for slot, test_case, result in zip(pending, test_cases, results):
                slot["latest_result"] = result
                slot["attempt_results"].append(result)
                verdict = _verdict_from_result(result)
                attempt_record: dict[str, Any] = {
                    "iteration": attempt,
                    "sample_id": test_case.name,
                    "verdict": verdict,
                }
                slot["attempts"].append(attempt_record)
                if verdict != "FAIL":
                    slot["done"] = True
                    continue

                verifier_sample_id = f"verify_{test_case.name}"
                verifier_case = _make_verifier_case(
                    source_result=result,
                    source_case=test_case,
                    verifier_sample_id=verifier_sample_id,
                    model_name=agent.model_name,
                    evidence_root=evidence_root,
                    temp_dirs=temp_dirs,
                )
                fail_slots.append(slot)
                verifier_cases.append(verifier_case)

            if verifier_cases:
                print(f"Iteration {attempt}/{max_iterations}: verifying {len(verifier_cases)} FAIL evidence item(s).")
                verifier_results = verifier.execute_tests(verifier_cases, sandbox=args.sandbox)
                for slot, verifier_result in zip(fail_slots, verifier_results):
                    slot["latest_verifier"] = verifier_result
                    slot["verifier_results"].append(verifier_result)
                    score = _verifier_score(verifier_result)
                    payload = _verifier_payload(verifier_result)
                    slot["attempts"][-1]["evidence_verifier"] = payload
                    if score is not None and score >= threshold:
                        slot["done"] = True
                    elif attempt >= max_iterations:
                        slot["done"] = True
                    else:
                        slot["feedback"] = _format_verifier_feedback(verifier_result, threshold)
        finally:
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)

    entries: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        case_slots = [slot for slot in slots if int(slot["case_index"]) == case_idx]
        case_slots.sort(key=lambda slot: int(slot["prop_index"]))
        tests: list[dict[str, Any]] = []
        results_for_usage = []
        for slot in case_slots:
            result = slot["latest_result"]
            prop_idx = int(slot["prop_index"])
            prop = slot["prop"]
            if result is None:
                tests.append(
                    {
                        "description": "Iterative VibeTest did not produce a result.",
                        "passed": False,
                        "evidence": [],
                        "execution_log": "",
                        "metadata": {
                            "reviewer": "AT-iterative",
                            "property_index": prop_idx,
                            "property_id": prop.property_id,
                            "property_text": prop.property_text,
                            "verdict": "INCONCLUSIVE",
                            "evidence_text": "",
                            "iterative_verifier": {"attempts": slot["attempts"]},
                        },
                    }
                )
                continue

            results_for_usage.extend(slot.get("attempt_results") or [result])
            results_for_usage.extend(slot.get("verifier_results") or [])
            test = _serialize_test_result(result, prop, prop_idx)
            metadata = dict(test.get("metadata") or {})
            if slot.get("latest_verifier") is not None:
                metadata["evidence_verifier"] = _verifier_payload(slot["latest_verifier"])
            metadata["iterative_verifier"] = {
                "enabled": True,
                "max_iterations": max_iterations,
                "threshold": threshold,
                "iterations_run": len(slot["attempts"]),
                "attempts": slot["attempts"],
            }
            test["metadata"] = metadata
            tests.append(test)

        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=aggregate_usage_from_results(results_for_usage),
                method="AT-iterative",
                method_model=agent.model_name,
            )
        )
    return entries


def _run_codex(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with Codex reviewer baseline")
    print("=" * 80)

    test_cases = [
        TestCase(
            name=f"{_slug_sample_component(case.dataset, max_len=60)}_row{case.row_index}",
            description="",
            repo_path=case.repo_path,
            sandbox_path="/workspace",
        )
        for case in cases
    ]

    agent = CodexReviewAgent(
        model=args.model or "openai/gpt-5-mini",
        codex_cmd=args.codex_cmd,
        codex_model=args.codex_model,
        codex_prompt=args.codex_prompt,
        timeout_s=args.codex_timeout,
        max_files=args.codex_max_files,
        max_total_bytes=args.codex_max_total_mb * 1024 * 1024 if args.codex_max_total_mb else None,
        log_dir=args.codex_log_dir,
    )
    all_results = agent.execute_tests(test_cases, sandbox=args.sandbox)

    entries: list[dict[str, Any]] = []
    for case, result in zip(cases, all_results):
        review_text = result.message or ""
        props = case.properties

        if not review_text.strip():
            tests = _all_inconclusive_tests(
                props,
                reviewer="codex",
                mapper_model=args.review_mapper_model,
                reason="No review output captured.",
            )
        else:
            if case.domain == "ml-bugs":
                mapped = map_review_to_kaggle_tests(
                    review_text,
                    [p.property_text for p in props],
                    reviewer="codex",
                    mapper_model=args.review_mapper_model,
                )
            elif case.domain == "security-vuln":
                if case.dataset == "bibifi":
                    vuln_props: list[Any] = [p.property_text for p in props]
                else:
                    vuln_props = []
                    for p in props:
                        cwe_num = _extract_cwe_num(p.property_id, p.property_text) or "0"
                        vuln_props.append((cwe_num, p.property_text))
                mapped = map_review_to_vuln_tests(
                    review_text,
                    case.dataset,
                    vuln_props,
                    repo_id_for_eval=case.repo_slug or case.repo_name,
                    reviewer="codex",
                    mapper_model=args.review_mapper_model,
                )
            else:
                mapped = _all_inconclusive_tests(
                    props,
                    reviewer="codex",
                    mapper_model=args.review_mapper_model,
                    reason="Unsupported domain for codex mapping.",
                )
            tests = mapped

        _augment_tests_with_review(
            tests,
            review_text,
            {
                "codex_ok": bool(review_text.strip()),
                "codex_error": None if review_text.strip() else "No review output captured",
            },
        )
        tests = _ensure_property_alignment(tests, props, reviewer="codex")

        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=usage_from_result_metadata(result),
                method="codex",
                method_model=agent.model_name,
            )
        )
    return entries


def _run_codex_vibetest(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with Codex VibeTest")
    print("=" * 80)

    all_test_cases: list[TestCase] = []
    case_ranges: list[tuple[int, int]] = []
    for case in cases:
        start = len(all_test_cases)
        for idx, prop in enumerate(case.properties):
            all_test_cases.append(
                TestCase(
                    name=_synthetic_property_sample_id(case, prop),
                    description=prop.property_text,
                    repo_path=case.repo_path,
                    sandbox_path="/workspace",
                    metadata={
                        "property_id": prop.property_id,
                        "property_index": idx,
                        "source_sample_id": _synthetic_property_sample_id(case, prop),
                    },
                )
            )
        case_ranges.append((start, len(all_test_cases)))

    agent = CodexVibeTestAgent(model=args.model or "openai/gpt-5-mini")
    all_results = agent.execute_tests(all_test_cases, sandbox=args.sandbox)

    entries: list[dict[str, Any]] = []
    for case, (start, end) in zip(cases, case_ranges):
        case_results = all_results[start:end]
        tests = [
            _serialize_test_result(r, case.properties[idx], idx)
            for idx, r in enumerate(case_results)
        ]
        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=aggregate_usage_from_results(case_results),
                method="AT-codex",
                method_model=agent.model_name,
            )
        )
    return entries


def _run_claude_vibetest(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with Claude Code VibeTest")
    print("=" * 80)

    all_test_cases: list[TestCase] = []
    case_ranges: list[tuple[int, int]] = []
    for case in cases:
        start = len(all_test_cases)
        for idx, prop in enumerate(case.properties):
            all_test_cases.append(
                TestCase(
                    name=_synthetic_property_sample_id(case, prop),
                    description=prop.property_text,
                    repo_path=case.repo_path,
                    sandbox_path="/workspace",
                    metadata={
                        "property_id": prop.property_id,
                        "property_index": idx,
                        "source_sample_id": _synthetic_property_sample_id(case, prop),
                    },
                )
            )
        case_ranges.append((start, len(all_test_cases)))

    agent = ClaudeCodeVibeTestAgent(model=args.model or "anthropic/claude-sonnet-4.5")
    all_results = agent.execute_tests(all_test_cases, sandbox=args.sandbox)

    entries: list[dict[str, Any]] = []
    for case, (start, end) in zip(cases, case_ranges):
        case_results = all_results[start:end]
        tests = [
            _serialize_test_result(r, case.properties[idx], idx)
            for idx, r in enumerate(case_results)
        ]
        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=aggregate_usage_from_results(case_results),
                method="AT-claude",
                method_model=agent.model_name,
            )
        )
    return entries


def _run_traincheck(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with TrainCheck baseline")
    print("=" * 80)

    for case in cases:
        if case.domain != "ml-bugs":
            raise SystemExit("TrainCheck synthetic run currently supports only ml-bugs domain labels.")

    output_root = Path(args.traincheck_output) if args.traincheck_output else None

    invariants_path: Path | None = None
    reference_error: str | None = None
    if args.traincheck_invariants:
        invariants_path = Path(args.traincheck_invariants)
        if not invariants_path.exists():
            reference_error = f"TrainCheck invariants file not found: {invariants_path}"
    else:
        reference_script = Path(args.traincheck_reference)
        try:
            invariants_path = prepare_reference_invariants(
                reference_script,
                output_root=output_root,
                infer_relations=[r.strip() for r in args.traincheck_relations.split(",") if r.strip()]
                if args.traincheck_relations
                else None,
                timeout_s=args.traincheck_timeout,
            )
        except Exception as exc:
            reference_error = f"TrainCheck reference invariant inference failed: {exc}"

    if reference_error:
        entries: list[dict[str, Any]] = []
        for case in cases:
            tests = _all_inconclusive_tests(
                case.properties,
                reviewer="traincheck",
                mapper_model=args.review_mapper_model,
                reason=reference_error,
            )
            _augment_tests_with_review(
                tests,
                "",
                {
                    "traincheck_ok": False,
                    "traincheck_error": reference_error,
                    "trace_dir": "",
                    "report_files": [],
                    "failed_invariants_count": 0,
                    "converted_notebook": None,
                    "reference_invariants": str(invariants_path) if invariants_path else "",
                },
            )
            tests = _ensure_property_alignment(tests, case.properties, reviewer="traincheck")
            entries.append(
                _entry_from_tests(
                    case,
                    tests,
                    usage=aggregate_usage_from_tests(tests),
                    method="traincheck",
                    method_model=None,
                )
            )
        return entries

    entries: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = run_traincheck(
                case.repo_path,
                script_path=args.traincheck_script,
                model_var=args.traincheck_model,
                output_root=output_root,
                invariants_path=invariants_path,
                infer_relations=[r.strip() for r in args.traincheck_relations.split(",") if r.strip()]
                if args.traincheck_relations
                else None,
                max_iters=args.traincheck_max_iters,
                timeout_s=args.traincheck_timeout,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "error": f"TrainCheck execution exception: {exc}",
                "trace_dir": "",
                "report_files": [],
                "failed_invariants_count": 0,
                "converted_notebook": None,
            }

        traincheck_ok = result.get("ok")
        failed_invariants = result.get("failed_invariants") or []
        if traincheck_ok is False:
            tests = _all_inconclusive_tests(
                case.properties,
                reviewer="traincheck",
                mapper_model=args.review_mapper_model,
                reason="TrainCheck failed to run.",
            )
            review_text = ""
        elif not failed_invariants:
            tests = _all_pass_tests(
                case.properties,
                reviewer="traincheck",
                mapper_model=args.review_mapper_model,
                reason="No failed TrainCheck invariants.",
            )
            review_text = ""
        else:
            review_text = result.get("review") or ""
            tests = map_review_to_kaggle_tests(
                review_text,
                [p.property_text for p in case.properties],
                reviewer="traincheck",
                mapper_model=args.review_mapper_model,
            )

        _augment_tests_with_review(
            tests,
            review_text,
            {
                "traincheck_ok": traincheck_ok,
                "traincheck_error": result.get("error"),
                "trace_dir": result.get("trace_dir"),
                "report_files": result.get("report_files", []),
                "failed_invariants_count": result.get("failed_invariants_count", 0),
                "converted_notebook": result.get("converted_notebook"),
                "reference_invariants": str(invariants_path) if invariants_path else "",
            },
        )
        tests = _ensure_property_alignment(tests, case.properties, reviewer="traincheck")

        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=aggregate_usage_from_tests(tests),
                method="traincheck",
                method_model=None,
            )
        )

    return entries


def _findings_by_cwe(findings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_cwe: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        cwes = set(str(c) for c in (finding.get("cwes") or []))
        for tag in finding.get("tags") or []:
            m = re.search(r"cwe-(\d+)", str(tag), flags=re.IGNORECASE)
            if m:
                cwes.add(str(int(m.group(1))))
        for cwe in cwes:
            cwe_norm = str(int(cwe)) if str(cwe).isdigit() else cwe
            by_cwe.setdefault(cwe_norm, []).append(finding)
    return by_cwe


def _run_codeql(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with CodeQL baseline")
    print("=" * 80)

    for case in cases:
        if case.domain != "security-vuln":
            raise SystemExit("CodeQL synthetic run currently supports only security-vuln domain labels.")

    entries: list[dict[str, Any]] = []
    total_cases = len(cases)
    for case_index, case in enumerate(cases, start=1):
        print(
            f"[CodeQL] {case_index}/{total_cases} "
            f"{case.dataset}/{case.repo_name} "
            f"(timeout={args.codeql_timeout}s, build_mode={args.codeql_build_mode})"
        )
        analysis = analyze_repo_with_codeql(
            case.repo_path,
            timeout_s=args.codeql_timeout,
            build_mode_override=args.codeql_build_mode,
        )
        ok = bool(analysis.get("ok"))
        property_failures = analysis.get("property_failures") or []
        findings = analysis.get("findings") or []
        by_cwe = _findings_by_cwe(findings)

        tests: list[dict[str, Any]] = []
        for idx, prop in enumerate(case.properties):
            matched_findings: list[dict[str, Any]] = []
            if ok:
                if case.dataset == "bibifi":
                    m = re.search(r"_vuln(\d+)$", prop.property_id, flags=re.IGNORECASE)
                    if m:
                        pid = int(m.group(1))
                        if 0 <= pid < len(property_failures):
                            matched_findings = property_failures[pid] or []
                else:
                    cwe = _extract_cwe_num(prop.property_id, prop.property_text)
                    if cwe is not None:
                        matched_findings = by_cwe.get(cwe, [])

            verdict = "PASS" if (not ok or not matched_findings) else "FAIL"
            reason = (
                "CodeQL analysis failed; defaulting to PASS."
                if not ok
                else (
                    "No matching CodeQL findings."
                    if verdict == "PASS"
                    else f"Matched {len(matched_findings)} finding(s)."
                )
            )
            tests.append(
                {
                    "description": reason,
                    "passed": verdict == "PASS",
                    "evidence": [],
                    "execution_log": "",
                    "metadata": {
                        "reviewer": "codeql",
                        "property_index": idx,
                        "property_id": prop.property_id,
                        "property_text": prop.property_text,
                        "verdict": verdict,
                        "evidence_text": json.dumps(matched_findings, ensure_ascii=False)[:12000]
                        if matched_findings
                        else "",
                        "codeql_ok": ok,
                        "codeql_error": analysis.get("error"),
                        "codeql_language": analysis.get("language"),
                        "codeql_suite": analysis.get("suite"),
                        "codeql_timeout_s": args.codeql_timeout,
                        "codeql_build_mode": args.codeql_build_mode,
                        "matched_findings": matched_findings,
                    },
                }
            )

        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=aggregate_usage_from_tests(tests),
                method="codeql",
                method_model=None,
            )
        )

    return entries


def _run_refchecker(cases: list[SyntheticCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    print("=" * 80)
    print("Running synthetic experiment with RefChecker baseline")
    print("=" * 80)

    for case in cases:
        if case.domain != "citation-hallucinations":
            raise SystemExit("RefChecker synthetic run currently supports only citation-hallucinations domain labels.")

    output_root = Path(args.refchecker_output_root) if args.refchecker_output_root else None
    db_path_obj = Path(args.refchecker_db_path) if args.refchecker_db_path else None
    workdir_obj = Path(args.refchecker_workdir) if args.refchecker_workdir else None

    entries: list[dict[str, Any]] = []
    for case in cases:
        result = run_refchecker(
            case.repo_path,
            refchecker_cmd=args.refchecker_cmd,
            output_root=output_root,
            timeout_s=args.refchecker_timeout,
            llm_provider=args.refchecker_llm_provider,
            llm_model=args.refchecker_llm_model,
            semantic_scholar_api_key=args.refchecker_semantic_scholar_api_key,
            db_path=db_path_obj,
            workdir=workdir_obj,
        )

        refchecker_ok = result.get("ok")
        review_text = str(result.get("review") or "")

        if refchecker_ok is False:
            tests = _all_inconclusive_tests(
                case.properties,
                reviewer="refchecker",
                mapper_model=args.review_mapper_model,
                reason="RefChecker failed to run.",
            )
        elif not review_text.strip():
            tests = _all_pass_tests(
                case.properties,
                reviewer="refchecker",
                mapper_model=args.review_mapper_model,
                reason="RefChecker produced no findings.",
            )
        else:
            tests = map_review_to_hallucination_tests(
                review_text,
                [p.property_text for p in case.properties],
                reviewer="refchecker",
                mapper_model=args.review_mapper_model,
            )

        _augment_tests_with_review(
            tests,
            review_text,
            {
                "refchecker_ok": refchecker_ok,
                "refchecker_error": result.get("error"),
                "refchecker_report": result.get("report_path"),
                "refchecker_paper_file": result.get("paper_file"),
                "resolved_refchecker_cmd": result.get("resolved_refchecker_cmd"),
                "resolved_refchecker_note": result.get("resolved_refchecker_note"),
            },
        )
        tests = _ensure_property_alignment(tests, case.properties, reviewer="refchecker")

        entries.append(
            _entry_from_tests(
                case,
                tests,
                usage=aggregate_usage_from_tests(tests),
                method="refchecker",
                method_model=None,
            )
        )

    return entries


async def _verify_fail_evidence_async(
    items: list[dict[str, Any]],
    *,
    model_name: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    if not items:
        return []

    model = get_model(
        model_name,
        config=GenerateConfig(max_tokens=500),
    )
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(item: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are grading whether a predicted FAIL matches the ground-truth violation for the same property in buggy code.\n"
            "Return strict JSON with keys: grade (C or I), explanation (string).\n"
            "Grade C only when the predicted failure clearly corresponds to the same underlying bug/violation as ground truth.\n"
            "Grade I if mismatched issue, too vague, missing evidence, or uncertain.\n\n"
            f"Property ID: {item['property_id']}\n"
            f"Property Text:\n{item['property_text']}\n\n"
            f"Ground Truth Violation Description:\n{item['gt_violation_description']}\n\n"
            f"Predicted Reason:\n{item['predicted_reason']}\n\n"
            f"Predicted Evidence:\n{item['predicted_evidence']}\n"
        )

        async with sem:
            try:
                resp = await asyncio.wait_for(model.generate(prompt), timeout=120)
                parsed = _extract_json_object(resp.completion or "") or {}
                grade = str(parsed.get("grade") or "I").strip().upper()
                if grade not in {"C", "I"}:
                    grade = "I"
                explanation = str(parsed.get("explanation") or "").strip()
                return {
                    "repo_index": item["repo_index"],
                    "test_index": item["test_index"],
                    "grade": grade,
                    "explanation": explanation,
                }
            except Exception as exc:
                return {
                    "repo_index": item["repo_index"],
                    "test_index": item["test_index"],
                    "grade": "I",
                    "explanation": f"Scorer error: {exc}",
                }

    return await asyncio.gather(*[_one(item) for item in items])


def _score_entries(entries: list[dict[str, Any]], *, scorer_model: str, scorer_concurrency: int) -> None:
    fail_verify_items: list[dict[str, Any]] = []

    for repo_idx, entry in enumerate(entries):
        gt_labels = entry.get("ground_truth_property_labels") or {}
        gt_desc = entry.get("ground_truth_violation_descriptions") or {}

        for test_idx, test in enumerate(entry.get("tests") or []):
            md = test.get("metadata") or {}
            pid = str(md.get("property_id") or "").strip()
            verdict = _normalize_verdict(
                md.get("verdict"),
                passed=test.get("passed"),
            )
            gt_label_raw = gt_labels.get(pid)
            gt_label = None if gt_label_raw is None else (1 if int(gt_label_raw) else 0)

            score_obj = {
                "property_id": pid,
                "predicted_verdict": verdict,
                "ground_truth_label": gt_label,
                "binary_match": False,
                "evidence_match_grade": None,
                "evidence_match_explanation": "",
                "correct": False,
                "decision_reason": "",
            }

            if gt_label is None:
                score_obj["decision_reason"] = "Missing ground-truth label for property id."
            elif verdict not in {"PASS", "FAIL"}:
                score_obj["decision_reason"] = "Abstention (INCONCLUSIVE/NOT APPLICABLE)."
            else:
                pred_label = 1 if verdict == "FAIL" else 0
                binary_match = pred_label == gt_label
                score_obj["binary_match"] = binary_match
                if not binary_match:
                    score_obj["decision_reason"] = "PASS/FAIL prediction mismatches ground-truth binary label."
                elif verdict == "PASS":
                    score_obj["correct"] = True
                    score_obj["decision_reason"] = "PASS prediction matches ground-truth PASS."
                else:
                    fail_verify_items.append(
                        {
                            "repo_index": repo_idx,
                            "test_index": test_idx,
                            "property_id": pid,
                            "property_text": str(md.get("property_text") or ""),
                            "gt_violation_description": str(gt_desc.get(pid) or ""),
                            "predicted_reason": str(test.get("description") or ""),
                            "predicted_evidence": _test_evidence_text(test),
                        }
                    )
                    score_obj["decision_reason"] = "Awaiting LLM verification of FAIL evidence match."

            md["synthetic_score"] = score_obj
            test["metadata"] = md

    if fail_verify_items:
        verdicts = asyncio.run(
            _verify_fail_evidence_async(
                fail_verify_items,
                model_name=scorer_model,
                concurrency=scorer_concurrency,
            )
        )
        by_key = {(v["repo_index"], v["test_index"]): v for v in verdicts}

        for repo_idx, entry in enumerate(entries):
            for test_idx, test in enumerate(entry.get("tests") or []):
                key = (repo_idx, test_idx)
                if key not in by_key:
                    continue
                v = by_key[key]
                md = test.get("metadata") or {}
                score_obj = dict(md.get("synthetic_score") or {})
                score_obj["evidence_match_grade"] = v["grade"]
                score_obj["evidence_match_explanation"] = v["explanation"]
                score_obj["correct"] = v["grade"] == "C"
                score_obj["decision_reason"] = (
                    "FAIL prediction matches ground-truth violation evidence."
                    if score_obj["correct"]
                    else "FAIL prediction does not sufficiently match ground-truth violation evidence."
                )
                md["synthetic_score"] = score_obj
                test["metadata"] = md

    for entry in entries:
        total = 0
        non_inconclusive = 0
        correct = 0
        mismatch = 0
        fail_checked = 0
        fail_checked_correct = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        fail_predictions = 0

        for test in entry.get("tests") or []:
            total += 1
            md = test.get("metadata") or {}
            s = md.get("synthetic_score") or {}
            verdict = _normalize_verdict(
                s.get("predicted_verdict") or md.get("verdict"),
                passed=test.get("passed"),
            )
            if verdict in {"PASS", "FAIL"}:
                non_inconclusive += 1
            if s.get("correct") is True:
                correct += 1
            if s.get("decision_reason", "").startswith("PASS/FAIL prediction mismatches"):
                mismatch += 1
            if verdict == "FAIL" and s.get("binary_match") is True:
                fail_checked += 1
                if s.get("correct") is True:
                    fail_checked_correct += 1

            # Positive class: GT FAIL.
            # Candidate positive prediction: any predicted FAIL.
            # Verifier-backed TP: predicted FAIL + evidence_match_grade == C on GT FAIL.
            gt_label = s.get("ground_truth_label")
            pred_fail = verdict == "FAIL"
            pred_pass = verdict == "PASS"
            grade = str(s.get("evidence_match_grade") or "").upper()
            if gt_label is None:
                continue
            gt_pos = bool(int(gt_label))
            if pred_fail:
                fail_predictions += 1
            tp_match = gt_pos and pred_fail and grade == "C"
            if tp_match:
                tp += 1
            # Any predicted FAIL not counted as TP is a false positive.
            if pred_fail and (not tp_match):
                fp += 1
            # Any GT FAIL not counted as TP is a false negative.
            if gt_pos and (not tp_match):
                fn += 1
            # Strict TN: only GT PASS with predicted PASS.
            if (not gt_pos) and pred_pass:
                tn += 1

        precision = (tp / (tp + fp)) if (tp + fp) else None
        recall = (tp / (tp + fn)) if (tp + fn) else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = 0.0 if (precision == 0.0 and recall == 0.0) else None
        else:
            f1 = 2 * precision * recall / (precision + recall)

        entry["scoring"] = {
            "total_tests": total,
            "non_inconclusive_predictions": non_inconclusive,
            "inconclusive_predictions": max(0, total - non_inconclusive),
            "non_inconclusive_rate": (non_inconclusive / total) if total else 0.0,
            "correct_predictions": correct,
            "accuracy_all": (correct / total) if total else 0.0,
            "accuracy_non_inconclusive": (correct / non_inconclusive) if non_inconclusive else 0.0,
            "binary_mismatch_count": mismatch,
            "fail_evidence_verified_count": fail_checked,
            "fail_evidence_verified_correct_count": fail_checked_correct,
            "fail_predictions": fail_predictions,
            "verified_fail_tp": tp,
            "verified_fail_fp": fp,
            "verified_fail_fn": fn,
            "verified_fail_tn": tn,
            "verified_fail_precision": precision,
            "verified_fail_recall": recall,
            "verified_fail_f1": f1,
            "scorer_model": scorer_model,
        }


def _write_results(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(path), mode="w") as writer:
        for entry in entries:
            writer.write(entry)


def _infer_output_path(
    cases: list[SyntheticCase],
    method: str,
    model: str | None,
    *,
    dynamic: bool = False,
    iterative_verifier: bool = False,
) -> Path:
    datasets = sorted({c.dataset for c in cases if c.dataset})
    if len(datasets) == 1:
        dataset_name = f"synthetic_{datasets[0]}"
    else:
        dataset_name = "synthetic_mixed"

    method_name = {
        "vibetest": "AT-iterative" if iterative_verifier else "AT",
        "codex-vibetest": "AT-codex",
        "vibetest-codex": "AT-codex",
        "vibetest-claude": "AT-claude",
        "codex": "codex",
        "traincheck": "traincheck",
        "codeql": "codeql",
        "refchecker": "refchecker",
    }[method]
    model_name = model
    if method == "vibetest" and dynamic:
        model_name = f"{model or 'unknown-model'}-dynamic"
    base_path = standardized_results_path(dataset_name, method_name, model_name=model_name)
    return base_path.parent / "synthetic" / base_path.name


def _run(args: argparse.Namespace) -> None:
    if args.method == "codex-vibetest":
        args.method = "vibetest-codex"
    if args.iterative_verifier and args.method != "vibetest":
        raise SystemExit("--iterative-verifier is currently supported only with --method vibetest.")
    if not (0.0 <= float(args.verifier_threshold) <= 1.0):
        raise SystemExit("--verifier-threshold must be between 0 and 1.")
    domain_filters = {d.strip() for d in (args.domains or []) if d.strip()}
    dataset_filters = {d.strip() for d in (args.datasets or []) if d.strip()}

    cases = _load_injected_cases(
        Path(args.labels_path),
        domains=domain_filters,
        datasets=dataset_filters,
        repo_limit=args.repo_limit,
        repo_offset=args.repo_offset,
    )

    print(f"Loaded synthetic cases: {len(cases)}")
    print(f"Domains: {sorted({c.domain for c in cases})}")
    print(f"Datasets: {sorted({c.dataset for c in cases})}")

    if args.method == "vibetest":
        if args.iterative_verifier:
            entries = _run_vibetest_iterative(cases, args)
        else:
            entries = _run_vibetest(cases, args)
        method_model = args.model or "openai/gpt-5-mini"
    elif args.method == "vibetest-codex":
        entries = _run_codex_vibetest(cases, args)
        method_model = args.model or "openai/gpt-5-mini"
    elif args.method == "vibetest-claude":
        entries = _run_claude_vibetest(cases, args)
        method_model = args.model or "anthropic/claude-sonnet-4.5"
    elif args.method == "codex":
        entries = _run_codex(cases, args)
        method_model = args.model or "openai/gpt-5-mini"
    elif args.method == "traincheck":
        entries = _run_traincheck(cases, args)
        method_model = None
    elif args.method == "codeql":
        entries = _run_codeql(cases, args)
        method_model = None
    elif args.method == "refchecker":
        entries = _run_refchecker(cases, args)
        method_model = None
    else:
        raise SystemExit(f"Unsupported method: {args.method}")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else _infer_output_path(
            cases,
            args.method,
            method_model,
            dynamic=bool(args.dynamic),
            iterative_verifier=bool(args.iterative_verifier),
        )
    )
    # Persist method outputs before scoring so long runs are recoverable even if scoring fails/interrupted.
    _write_results(output_path, entries)
    if not args.skip_scoring:
        print(f"Wrote unscored results: {output_path}")
        print("Scoring predictions against synthetic ground truth...")
        try:
            _score_entries(entries, scorer_model=args.scorer_model, scorer_concurrency=args.scorer_concurrency)
        except Exception:
            print("Scoring failed; unscored results were preserved.")
            raise
        _write_results(output_path, entries)

    total_tests = sum(int(e.get("total_tests") or 0) for e in entries)
    total_correct = sum(int((e.get("scoring") or {}).get("correct_predictions") or 0) for e in entries)
    total_non_inc = sum(int((e.get("scoring") or {}).get("non_inconclusive_predictions") or 0) for e in entries)
    total_tp = sum(int((e.get("scoring") or {}).get("verified_fail_tp") or 0) for e in entries)
    total_fp = sum(int((e.get("scoring") or {}).get("verified_fail_fp") or 0) for e in entries)
    total_fn = sum(int((e.get("scoring") or {}).get("verified_fail_fn") or 0) for e in entries)
    total_fail_predictions = sum(int((e.get("scoring") or {}).get("fail_predictions") or 0) for e in entries)
    response_rate = (total_non_inc / total_tests) if total_tests else None
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else None
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    def _fmt(x: float | None) -> str:
        return "N/A" if x is None else f"{x:.4f}"

    print("=" * 80)
    print(f"Wrote results: {output_path}")
    print(f"Rows: {len(entries)}")
    if not args.skip_scoring:
        print(f"Correct predictions: {total_correct}/{total_tests}")
        print(f"Response rate (non-inconclusive): {_fmt(response_rate)} ({total_non_inc}/{total_tests})")
        if total_non_inc:
            print(f"Accuracy (non-inconclusive): {total_correct/total_non_inc:.4f}")
        print(
            "Verified FAIL metrics "
            f"(positive class = GT FAIL; candidate positives = predicted FAIL; verifier determines TP/FP): "
            f"P={_fmt(precision)} R={_fmt(recall)} F1={_fmt(f1)} "
            f"[TP={total_tp}, FP={total_fp}, FN={total_fn}, FAIL predictions={total_fail_predictions}]"
        )
    print("=" * 80)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["vibetest", "vibetest-codex", "vibetest-claude", "codex-vibetest", "codex", "traincheck", "codeql", "refchecker"],
        required=True,
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        required=True,
        help="Path to synthetic injected labels JSONL (e.g., synth-data/injected/labels_kaggle_titanic.jsonl)",
    )
    parser.add_argument("--output-path", type=str, help="Optional explicit output JSONL path.")
    parser.add_argument("--domains", nargs="+", default=[], help="Optional domain filters (ml-bugs, security-vuln).")
    parser.add_argument("--datasets", nargs="+", default=[], help="Optional dataset filters (e.g., kaggle_titanic).")
    parser.add_argument("--repo-limit", type=int, default=0, help="Optional max repositories to run.")
    parser.add_argument("--repo-offset", type=int, default=0, help="Optional repositories to skip before running.")
    parser.add_argument("--sandbox", type=str, default="docker", help="Inspect sandbox backend.")

    parser.add_argument("--model", type=str, help="Model for vibetest/codex methods.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic VibeTest agent mode.")
    parser.add_argument("--review-mapper-model", type=str, help="Model for review->property mapping.")
    parser.add_argument(
        "--iterative-verifier",
        action="store_true",
        help="For --method vibetest, rerun weak FAIL evidence with clean-context verifier feedback.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum VibeTest+verifier attempts when --iterative-verifier is enabled.",
    )
    parser.add_argument(
        "--verifier-threshold",
        type=float,
        default=0.7,
        help="Minimum verifier evidence-support score needed to accept a FAIL in iterative mode.",
    )
    parser.add_argument("--verifier-model", type=str, help="Verifier model for iterative mode; defaults to --model.")
    parser.add_argument("--verifier-static", action="store_true", help="Disable python tool for the iterative verifier agent.")
    parser.add_argument("--verifier-log-dir", type=str, default="./logs", help="Inspect log directory for iterative verifier runs.")
    parser.add_argument("--evidence-root", type=str, default="evidence-dumps", help="Root for VibeTest /evidence tarballs.")

    parser.add_argument("--codex-cmd", type=str, default="codex", help="Codex CLI command.")
    parser.add_argument("--codex-model", type=str, default="inspect", help="Codex model argument passed to CLI.")
    parser.add_argument("--codex-prompt", type=str, default="/review", help="Codex prompt.")
    parser.add_argument("--codex-timeout", type=int, default=1200, help="Codex timeout seconds.")
    parser.add_argument("--codex-max-files", type=int, help="Optional cap on files passed to Codex sandbox.")
    parser.add_argument("--codex-max-total-mb", type=int, help="Optional cap on total MB passed to Codex sandbox.")
    parser.add_argument("--codex-log-dir", type=str, default="./logs", help="Inspect log directory for codex runs.")

    parser.add_argument("--traincheck-script", type=str, help="Optional explicit training script path inside each repo.")
    parser.add_argument("--traincheck-model", type=str, help="Optional model variable name to patch in script.")
    parser.add_argument("--traincheck-timeout", type=int, default=1200, help="TrainCheck timeout in seconds.")
    parser.add_argument("--traincheck-output", type=str, help="Optional TrainCheck output root.")
    parser.add_argument("--traincheck-relations", type=str, help="Comma-separated invariant relations for infer/check.")
    parser.add_argument(
        "--traincheck-reference",
        type=str,
        default="data/traincheck/mnist.py",
        help="Reference script for invariant inference when --traincheck-invariants is not provided.",
    )
    parser.add_argument(
        "--traincheck-invariants",
        type=str,
        help="Existing TrainCheck invariants JSON path (skip reference inference).",
    )
    parser.add_argument("--traincheck-max-iters", type=int, help="Optional TRAINCHECK_MAX_ITERS patch value.")

    parser.add_argument(
        "--codeql-timeout",
        type=int,
        default=600,
        help="CodeQL timeout in seconds for each command (database create/analyze).",
    )
    parser.add_argument(
        "--codeql-build-mode",
        type=str,
        choices=["none", "autobuild"],
        default="none",
        help="CodeQL database build mode for synthetic runs.",
    )

    parser.add_argument("--refchecker-cmd", type=str, default="academic-refchecker", help="RefChecker command.")
    parser.add_argument("--refchecker-timeout", type=int, default=1200, help="RefChecker timeout seconds.")
    parser.add_argument("--refchecker-output-root", type=str, help="Directory to store RefChecker reports.")
    parser.add_argument("--refchecker-llm-provider", type=str, help="RefChecker LLM provider.")
    parser.add_argument("--refchecker-llm-model", type=str, help="RefChecker LLM model.")
    parser.add_argument(
        "--refchecker-semantic-scholar-api-key",
        type=str,
        help="Semantic Scholar API key override (defaults to SEMANTIC_SCHOLAR_API_KEY env/.env).",
    )
    parser.add_argument("--refchecker-db-path", type=str, help="Path to RefChecker DB.")
    parser.add_argument("--refchecker-workdir", type=str, help="Working directory for RefChecker command.")

    parser.add_argument("--skip-scoring", action="store_true", help="Skip synthetic response scoring stage.")
    parser.add_argument(
        "--scorer-model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model used to verify FAIL evidence against ground truth.",
    )
    parser.add_argument("--scorer-concurrency", type=int, default=8, help="Concurrency for FAIL evidence verifier calls.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
