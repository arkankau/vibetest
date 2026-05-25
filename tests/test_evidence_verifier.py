import shutil
import tempfile
from pathlib import Path

from vibetest.agent.evidence_verifier_agent import parse_verifier_output
from vibetest.verifier.evidence_paths import resolve_evidence_tar, source_sample_id_candidates
from vibetest.verifier.rubric import EvidenceState, parse_verifier_fields, structured_state_to_score


def test_parse_verifier_output_extracts_score_reason_and_assessment() -> None:
    output = """
Reviewed the cited notebook cells.

SCORE: 0.85
REASON: The cited train/test concat supports the claimed leakage.
EVIDENCE_ASSESSMENT: Checked train.ipynb lines 42-48; test.csv is concatenated before split.
"""
    score, reason, assessment = parse_verifier_output(output)

    assert score == 0.85
    assert "train/test concat" in reason
    assert "train.ipynb lines 42-48" in assessment


def test_parse_verifier_output_clamps_score_to_unit_interval() -> None:
    score, _, _ = parse_verifier_output("SCORE: 1.5\nREASON: too high\nEVIDENCE_ASSESSMENT: n/a")
    assert score == 1.0


def test_parse_structured_verifier_output_maps_state_to_score() -> None:
    output = """
EVIDENCE_STATE: INCONCLUSIVE
SCORE: 0.15
REASON: No randomized-label experiment is present in the repo.
EVIDENCE_ASSESSMENT: Checked train.ipynb; no label shuffle ablation.
"""
    score, reason, assessment, state = parse_verifier_fields(output)

    assert state == EvidenceState.INCONCLUSIVE
    assert score == 0.15
    assert "randomized-label" in reason
    assert structured_state_to_score(EvidenceState.FAIL) == 0.95


def test_parse_structured_verifier_output_infers_score_from_state() -> None:
    output = """
EVIDENCE_STATE: FAIL
REASON: Train and test are concatenated before splitting.
EVIDENCE_ASSESSMENT: train.ipynb lines 40-48.
"""
    score, _, _, state = parse_verifier_fields(output)

    assert state == EvidenceState.FAIL
    assert score == 0.95


def test_source_sample_id_candidates_prefers_dataset_prefixed_id() -> None:
    entry = {"dataset": "kaggle_titanic", "synthetic_row_index": 0}
    test = {"metadata": {"property_id": "kaggle_p3"}}

    candidates = source_sample_id_candidates(entry, test, row_idx=0)

    assert candidates[0] == "kaggle_titanic_row0_kaggle_p3"
    assert "row0_kaggle_p3" in candidates


def test_resolve_evidence_tar_uses_existing_dataset_prefixed_bundle() -> None:
    tmp_path = Path(tempfile.mkdtemp())
    try:
        evidence_root = tmp_path / "evidence-dumps"
        bundle_dir = evidence_root / "gpt-5-mini"
        bundle_dir.mkdir(parents=True)
        bundle = bundle_dir / "evidence-kaggle_titanic_row0_kaggle_p3.tar.gz"
        bundle.write_bytes(b"fake")

        entry = {
            "dataset": "kaggle_titanic",
            "synthetic_row_index": 0,
            "method_model": "openai/gpt-5-mini",
        }
        test = {"metadata": {"property_id": "kaggle_p3"}}

        resolved, sample_id = resolve_evidence_tar(
            entry=entry,
            test=test,
            row_idx=0,
            evidence_root=evidence_root,
            evidence_model=None,
        )

        assert sample_id == "kaggle_titanic_row0_kaggle_p3"
        assert resolved == bundle
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
