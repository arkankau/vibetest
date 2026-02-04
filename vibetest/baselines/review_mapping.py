"""LLM-based mapping from general code reviews to vibetest test cases."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, get_model


@dataclass(frozen=True)
class MappingItem:
    property_index: int
    verdict: str
    reason: str
    evidence: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _truncate(text: str, limit: int = 12000) -> str:
    if not text:
        return ""
    return text[:limit]


def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    if not text:
        return None
    def _coerce_result(parsed: Any) -> list[dict[str, Any]] | None:
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("results", "items", "output"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
        return None

    try:
        parsed = json.loads(text)
        coerced = _coerce_result(parsed)
        if coerced is not None:
            return coerced
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            parsed = json.loads(block)
            coerced = _coerce_result(parsed)
            if coerced is not None:
                return coerced
        except json.JSONDecodeError:
            continue

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        coerced = _coerce_result(parsed)
        if coerced is not None:
            return coerced
    except json.JSONDecodeError:
        pass
    try:
        import ast

        py_text = snippet.replace("null", "None").replace("true", "True").replace("false", "False")
        parsed = ast.literal_eval(py_text)
        coerced = _coerce_result(parsed)
        if coerced is not None:
            return coerced
    except Exception:
        return None
    return None


TRAINCHECK_INVARIANT_INSTRUCTIONS = (
    "The review text may consist of TrainCheck failed invariants in JSON form. "
    "Each failed invariant indicates behavior that deviated from the reference. "
    "Only failed invariants are evidence of violations; passed or not-triggered invariants "
    "are not evidence. Ignore event-order invariants such as FunctionCoverRelation and "
    "FunctionLeadRelation because they are often overfit/noisy. Focus on semantic/API "
    "relations (e.g., APIContainRelation) that imply missing or incorrect training behavior "
    "(e.g., optimizer step not updating parameters, zero_grad not resetting gradients). "
    "Map those failures to the most relevant properties. If the invariant does not clearly "
    "support a property violation, mark PASS or INCONCLUSIVE."
)


def _normalize_verdict(raw: str) -> str:
    if not raw:
        return "PASS"
    raw = raw.strip().upper()
    if raw in {"PASS", "FAIL", "INCONCLUSIVE"}:
        return raw
    if raw.startswith("P"):
        return "PASS"
    if raw.startswith("F"):
        return "FAIL"
    return "INCONCLUSIVE"


async def _map_review_async(
    *,
    review_text: str,
    properties: list[dict[str, Any]],
    mapper_model: str,
    extra_instructions: str | None = None,
) -> list[MappingItem]:
    mapper = get_model(
        mapper_model,
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=3000,
        ),
    )

    review_text = _truncate(review_text)
    properties_json = json.dumps(properties, ensure_ascii=False)

    rules = (
        "You are mapping a general code review to a set of test properties.\n"
        "For each property, determine whether the review provides evidence that the property is violated.\n"
        "Rules:\n"
        "- Use FAIL only if the review explicitly indicates a violation of the property.\n"
        "- Use PASS if the review does not mention the property or indicates it is satisfied.\n"
        "- Use INCONCLUSIVE only if the review provides partial/ambiguous evidence.\n"
        "- Never invent new issues. Only use the review text.\n\n"
    )
    if extra_instructions:
        rules += f"Additional instructions:\n{extra_instructions}\n\n"
    prompt = (
        f"{rules}"
        "Return a JSON array with one object per property, with keys:\n"
        "  property_index (int), verdict (PASS/FAIL/INCONCLUSIVE), reason (string), evidence (string)\n"
        "Constraints:\n"
        "- Output only JSON. Do not use markdown or code fences.\n"
        "- Keep reason <= 30 words and evidence <= 30 words.\n\n"
        "- Use compact JSON with no extra whitespace.\n\n"
        "Review:\n"
        f"{review_text}\n\n"
        "Properties (JSON list):\n"
        f"{properties_json}\n"
    )

    result = await mapper.generate(prompt)
    data = _extract_json_array(result.completion or "")
    if not data:
        return []

    items: list[MappingItem] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        idx_raw = entry.get("property_index")
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        items.append(
            MappingItem(
                property_index=idx,
                verdict=_normalize_verdict(str(entry.get("verdict") or "")),
                reason=str(entry.get("reason") or "").strip(),
                evidence=str(entry.get("evidence") or "").strip(),
            )
        )
    return items


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        raise RuntimeError(
            "Review mapping requires no running event loop in this process."
        ) from exc


def _default_mapper_model() -> str:
    return os.getenv("VIBETEST_REVIEW_MAPPER_MODEL", "openai/gpt-5-mini")


def load_kaggle_properties() -> list[str]:
    props_path = _repo_root() / "data" / "kaggle" / "properties.md"
    content = props_path.read_text(encoding="utf-8")
    properties = [p.strip() for p in content.split("- ")[1:] if p.strip()]
    return properties


def map_review_to_kaggle_tests(
    review_text: str,
    properties: list[str],
    *,
    reviewer: str,
    mapper_model: str | None = None,
) -> list[dict[str, Any]]:
    mapper_model = mapper_model or _default_mapper_model()
    props_payload = [
        {"property_index": idx, "text": prop} for idx, prop in enumerate(properties)
    ]
    extra_instructions = (
        TRAINCHECK_INVARIANT_INSTRUCTIONS if reviewer == "traincheck" else None
    )

    items = _run_async(
        _map_review_async(
            review_text=review_text,
            properties=props_payload,
            mapper_model=mapper_model,
            extra_instructions=extra_instructions,
        )
    )
    by_index = {item.property_index: item for item in items}

    tests: list[dict[str, Any]] = []
    for idx, prop in enumerate(properties):
        item = by_index.get(idx)
        verdict = item.verdict if item else "PASS"
        reason = item.reason if item else "Not mentioned in review."
        evidence = item.evidence if item else ""
        passed = verdict != "FAIL"
        tests.append(
            {
                "description": reason,
                "passed": passed,
                "evidence": [],
                "execution_log": "",
                "metadata": {
                    "reviewer": reviewer,
                    "mapper_model": mapper_model,
                    "property_index": idx,
                    "property_text": prop,
                    "verdict": verdict,
                    "evidence_text": evidence,
                },
            }
        )
    return tests


def load_vuln_properties(dataset: str) -> list[Any]:
    dataset = dataset.lower()
    props_path = _repo_root() / "data" / "vuln" / dataset / "properties.md"
    content = props_path.read_text(encoding="utf-8")

    if dataset == "bibifi":
        properties = [p.strip() for p in content.split("- ")[1:] if p.strip()]
        return properties

    # Prefer section-based properties (e.g., "## CWE-22 ..." headings)
    heading_re = re.compile(r"^##\s*(CWE-\d+[^\n]*)\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(content))
    if matches:
        cwe_properties: list[tuple[str, str]] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            section = content[start:end].strip()
            m = re.search(r"CWE-(\d+)", match.group(1))
            if not m:
                continue
            cwe = m.group(1).lstrip("0") or "0"
            cwe_properties.append((cwe, section))
        return cwe_properties

    # Fallback for bullet-list properties
    properties_raw = [p.strip() for p in content.split("- ")[1:] if p.strip()]
    cwe_properties: list[tuple[str, str]] = []
    for prop_text in properties_raw:
        m = re.search(r"CWE-(\d+)", prop_text)
        if not m:
            continue
        cwe = m.group(1).lstrip("0") or "0"
        cwe_properties.append((cwe, prop_text))
    return cwe_properties


def map_review_to_vuln_tests(
    review_text: str,
    dataset: str,
    properties: list[Any],
    *,
    repo_id_for_eval: str,
    reviewer: str,
    mapper_model: str | None = None,
) -> list[dict[str, Any]]:
    mapper_model = mapper_model or _default_mapper_model()
    props_payload: list[dict[str, Any]] = []

    if dataset == "bibifi":
        for idx, prop_text in enumerate(properties):
            props_payload.append(
                {
                    "property_index": idx,
                    "property_id": f"vuln{idx}",
                    "text": prop_text,
                }
            )
    else:
        for idx, (cwe, prop_text) in enumerate(properties):
            props_payload.append(
                {
                    "property_index": idx,
                    "property_id": f"CWE-{cwe}",
                    "text": prop_text,
                }
            )

    items = _run_async(
        _map_review_async(
            review_text=review_text,
            properties=props_payload,
            mapper_model=mapper_model,
        )
    )
    by_index = {item.property_index: item for item in items}

    tests: list[dict[str, Any]] = []
    if dataset == "bibifi":
        for idx, prop_text in enumerate(properties):
            item = by_index.get(idx)
            verdict = item.verdict if item else "PASS"
            reason = item.reason if item else "Not mentioned in review."
            evidence = item.evidence if item else ""
            passed = verdict != "FAIL"
            tests.append(
                {
                    "description": reason,
                    "passed": passed,
                    "evidence": [],
                    "execution_log": "",
                    "metadata": {
                        "reviewer": reviewer,
                        "mapper_model": mapper_model,
                        "test_id": f"repo{repo_id_for_eval}_vuln{idx}",
                        "property_index": idx,
                        "property_text": prop_text,
                        "verdict": verdict,
                        "evidence_text": evidence,
                    },
                }
            )
        return tests

    for idx, (cwe_num, prop_text) in enumerate(properties):
        item = by_index.get(idx)
        verdict = item.verdict if item else "PASS"
        reason = item.reason if item else "Not mentioned in review."
        evidence = item.evidence if item else ""
        passed = verdict != "FAIL"
        tests.append(
            {
                "description": reason,
                "passed": passed,
                "evidence": [],
                "execution_log": "",
                "metadata": {
                    "reviewer": reviewer,
                    "mapper_model": mapper_model,
                    "test_id": f"repo{repo_id_for_eval}_cwe{cwe_num}",
                    "property_index": idx,
                    "property_text": prop_text,
                    "verdict": verdict,
                    "evidence_text": evidence,
                },
            }
        )
    return tests
