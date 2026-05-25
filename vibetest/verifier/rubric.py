"""Evidence verifier rubrics and score mappings."""

from __future__ import annotations

import re
from enum import Enum


class EvidenceState(str, Enum):
    """Structured evidence verdict aligned with strict human re-audit."""

    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    PASS = "PASS"


_STATE_RE = re.compile(
    r"EVIDENCE_STATE:\s*(FAIL|INCONCLUSIVE|PASS)\b",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(r"SCORE:\s*(1(?:\.0+)?|0(?:\.\d+)?|\.\d+)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.*?)(?:\nEVIDENCE_ASSESSMENT:|\Z)", re.IGNORECASE | re.DOTALL)
_ASSESSMENT_RE = re.compile(r"EVIDENCE_ASSESSMENT:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)

# Map structured states to scalar scores for PR-curve analysis.
# Only FAIL should survive a high threshold (e.g. 0.7).
STRUCTURED_STATE_SCORES: dict[EvidenceState, float] = {
    EvidenceState.FAIL: 0.95,
    EvidenceState.INCONCLUSIVE: 0.15,
    EvidenceState.PASS: 0.05,
}


def structured_state_to_score(state: EvidenceState | None) -> float | None:
    if state is None:
        return None
    return STRUCTURED_STATE_SCORES[state]


def parse_evidence_state(text: str) -> EvidenceState | None:
    match = _STATE_RE.search(text or "")
    if not match:
        return None
    return EvidenceState(match.group(1).upper())


def parse_verifier_fields(output: str) -> tuple[float | None, str, str, EvidenceState | None]:
    """Parse score, reason, assessment, and optional structured evidence state."""
    text = output or ""
    state = parse_evidence_state(text)

    score: float | None = None
    score_match = _SCORE_RE.search(text)
    if score_match:
        try:
            score = max(0.0, min(1.0, float(score_match.group(1))))
        except ValueError:
            score = None
    elif state is not None:
        score = structured_state_to_score(state)

    reason_match = _REASON_RE.search(text)
    reason = reason_match.group(1).strip() if reason_match else ""
    assessment_match = _ASSESSMENT_RE.search(text)
    assessment = assessment_match.group(1).strip() if assessment_match else ""
    return score, reason, assessment, state


def accept_fail_at_threshold(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold
