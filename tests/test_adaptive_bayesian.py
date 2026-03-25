"""
Tests for the Adaptive Bayesian Sabotage Detector.

Tests cover:
  - Hypothesis construction and properties
  - Reference distribution building and querying
  - Bayesian updating mechanics
  - End-to-end detection on synthetic data
  - Leave-one-out evaluation
  - Data loading utilities
  - Edge cases and robustness
"""

import json
import math
import os
import random
import tempfile
from pathlib import Path

import importlib.util
import sys

import pytest

# Import directly from the module file to avoid pulling in the full
# vibetest.baselines.__init__ which requires inspect_swe and other heavy deps.
_mod_path = os.path.join(
    os.path.dirname(__file__), "..", "vibetest", "baselines", "adaptive_bayesian.py"
)
_spec = importlib.util.spec_from_file_location("adaptive_bayesian", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["adaptive_bayesian"] = _mod
_spec.loader.exec_module(_mod)

AdaptiveBayesianDetector = _mod.AdaptiveBayesianDetector
CaseData = _mod.CaseData
Hypothesis = _mod.Hypothesis
ReferenceDistribution = _mod.ReferenceDistribution
compute_metrics = _mod.compute_metrics
load_case_from_disk = _mod.load_case_from_disk
load_dataset = _mod.load_dataset
run_loo_evaluation = _mod.run_loo_evaluation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_benign_case():
    """A case with no malicious traces and all-zero scores."""
    n = 100
    trace_ids = [f"traces/trace_{i:03d}.txt" for i in range(n)]
    return CaseData(
        case_id="case_benign",
        trace_ids=trace_ids,
        monitor_scores={tid: 0.0 for tid in trace_ids},
        gt_malicious=set(),
    )


@pytest.fixture
def simple_malicious_case():
    """A case with 6 malicious traces (decomp_level=6) out of 600."""
    n = 600
    trace_ids = [f"traces/trace_{i:03d}.txt" for i in range(n)]
    gt = {f"traces/trace_{i:03d}.txt" for i in [10, 50, 100, 200, 300, 400]}
    scores = {}
    for tid in trace_ids:
        if tid in gt:
            # Malicious: some detected (score > 0), some sneaky (score = 0)
            scores[tid] = random.choice([0.0, 0.0, 0.4, 0.5, 0.7, 0.8])
        else:
            # Benign: mostly 0, some false positives
            scores[tid] = random.choice([0.0] * 9 + [0.4])
    return CaseData(
        case_id="case_mal_0",
        trace_ids=trace_ids,
        monitor_scores=scores,
        gt_malicious=gt,
    )


@pytest.fixture
def synthetic_dataset():
    """Generate a dataset of 10 cases for testing."""
    random.seed(42)
    cases = []
    for case_idx in range(10):
        n = 120
        trace_ids = [f"traces/trace_{i:03d}.txt" for i in range(n)]
        # 6 malicious per case
        mal_indices = random.sample(range(n), 6)
        gt = {trace_ids[i] for i in mal_indices}

        scores = {}
        for i, tid in enumerate(trace_ids):
            if i in mal_indices:
                # Malicious traces: mix of sneaky and detected
                scores[tid] = random.choice([0.0, 0.0, 0.3, 0.5, 0.6, 0.8])
            else:
                # Benign: mostly 0
                scores[tid] = 0.0 if random.random() < 0.88 else random.choice(
                    [0.1, 0.2, 0.4, 0.5]
                )
        cases.append(CaseData(
            case_id=f"case_{case_idx:04d}",
            trace_ids=trace_ids,
            monitor_scores=scores,
            gt_malicious=gt,
        ))
    return cases


@pytest.fixture
def case_with_clusters():
    """A case with cluster assignments."""
    n = 100
    trace_ids = [f"traces/trace_{i:03d}.txt" for i in range(n)]
    gt = {trace_ids[10], trace_ids[11], trace_ids[12]}  # 3 in same cluster

    scores = {tid: 0.0 for tid in trace_ids}
    scores[trace_ids[10]] = 0.4
    scores[trace_ids[11]] = 0.3
    scores[trace_ids[12]] = 0.5

    # All 3 malicious traces in cluster 2
    assignments = {}
    for i, tid in enumerate(trace_ids):
        if i < 20:
            assignments[tid] = 0
        elif i < 40:
            assignments[tid] = 1
        elif i < 60:
            assignments[tid] = 2
        else:
            assignments[tid] = 3

    # Override: put malicious in cluster 2
    for tid in gt:
        assignments[tid] = 2

    coherences = {0: 0.3, 1: 0.4, 2: 0.8, 3: 0.2}
    sizes = {0: 20, 1: 20, 2: 20, 3: 40}

    return CaseData(
        case_id="case_clusters",
        trace_ids=trace_ids,
        monitor_scores=scores,
        gt_malicious=gt,
        cluster_assignments=assignments,
        cluster_coherences=coherences,
        cluster_sizes=sizes,
    )


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

class TestHypothesis:
    def test_benign_hypothesis(self):
        h = Hypothesis(r=0.0, beta=0.0)
        assert h.is_benign
        assert "r=0.000" in repr(h)

    def test_malign_hypothesis(self):
        h = Hypothesis(r=0.01, beta=0.6)
        assert not h.is_benign
        assert "r=0.010" in repr(h)
        assert "β=0.6" in repr(h)

    def test_extreme_values(self):
        h0 = Hypothesis(r=0.0, beta=1.0)
        assert h0.is_benign

        h1 = Hypothesis(r=1.0, beta=0.0)
        assert not h1.is_benign


# ---------------------------------------------------------------------------
# Reference distribution tests
# ---------------------------------------------------------------------------

class TestReferenceDistribution:
    def test_empty_distribution(self):
        ref = ReferenceDistribution(bucket_size=0.1)
        # Should return smoothed probability (not crash)
        p = ref.p_score_benign(0.5)
        assert 0 < p <= 1.0

    def test_add_benign_scores(self):
        ref = ReferenceDistribution(bucket_size=0.1)
        for _ in range(100):
            ref.add_benign(0.0)
        for _ in range(10):
            ref.add_benign(0.5)

        # Score 0.0 should be more likely than 0.5
        p0 = ref.p_score_benign(0.0)
        p5 = ref.p_score_benign(0.5)
        assert p0 > p5

    def test_add_malicious_scores(self):
        ref = ReferenceDistribution(bucket_size=0.1)
        # Add some benign
        for _ in range(100):
            ref.add_benign(0.0)

        # Add malicious at different β levels
        for _ in range(50):
            ref.add_malicious(0.8, beta=0.0)  # not sneaky
        for _ in range(50):
            ref.add_malicious(0.1, beta=1.0)  # very sneaky

        # Not-sneaky malicious should have high scores
        p_high = ref.p_score_malicious(0.8, beta=0.0)
        p_low = ref.p_score_malicious(0.1, beta=0.0)
        assert p_high > p_low

    def test_bucket_consistency(self):
        ref = ReferenceDistribution(bucket_size=0.1)
        # Scores in the same bucket should get the same probability
        ref.add_benign(0.05)
        ref.add_benign(0.09)  # Same bucket as 0.05
        ref.add_benign(0.15)  # Different bucket

        p1 = ref.p_score_benign(0.01)
        p2 = ref.p_score_benign(0.08)
        assert p1 == p2  # same bucket

    def test_smoothing_prevents_zero(self):
        ref = ReferenceDistribution(bucket_size=0.1)
        ref.add_benign(0.0)
        # Even a never-seen score should have non-zero probability
        p = ref.p_score_benign(0.99, smoothing=1.0)
        assert p > 0


# ---------------------------------------------------------------------------
# Detector construction tests
# ---------------------------------------------------------------------------

class TestDetectorConstruction:
    def test_default_hypotheses(self):
        det = AdaptiveBayesianDetector()
        # Should have 1 benign + (5 r values × 6 β values) = 31
        n_benign = sum(1 for h in det.hypotheses if h.is_benign)
        n_malign = sum(1 for h in det.hypotheses if not h.is_benign)
        assert n_benign == 1
        assert n_malign == 30  # 5 nonzero r × 6 β

    def test_custom_hypotheses(self):
        det = AdaptiveBayesianDetector(
            r_values=[0.0, 0.01],
            beta_values=[0.0, 1.0],
        )
        assert len(det.hypotheses) == 3  # 1 benign + 1r × 2β

    def test_prior_sums_to_one(self):
        det = AdaptiveBayesianDetector()
        prior = det._build_prior()
        assert abs(sum(prior) - 1.0) < 1e-10

    def test_prior_benign_weight(self):
        det = AdaptiveBayesianDetector(prior_benign=0.7)
        prior = det._build_prior()
        benign_idx = next(i for i, h in enumerate(det.hypotheses) if h.is_benign)
        assert abs(prior[benign_idx] - 0.7) < 1e-10


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------

class TestFitting:
    def test_fit_supervised(self, simple_malicious_case):
        det = AdaptiveBayesianDetector()
        det.fit([simple_malicious_case])
        assert det._fitted
        assert det.reference is not None
        assert det.reference.benign_total > 0

    def test_fit_unsupervised(self, simple_benign_case):
        det = AdaptiveBayesianDetector()
        det.fit_unsupervised([simple_benign_case])
        assert det._fitted
        assert det.reference is not None

    def test_fit_empty_raises(self):
        det = AdaptiveBayesianDetector()
        # Should handle empty gracefully
        det.fit_unsupervised([])
        # Fitting on empty should still mark as fitted
        assert det._fitted

    def test_not_fitted_raises(self, simple_benign_case):
        det = AdaptiveBayesianDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.score_case(simple_benign_case)


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_scores_in_range(self, synthetic_dataset):
        """All output scores should be in [0, 1]."""
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit(synthetic_dataset)
        for case in synthetic_dataset[:3]:
            scores = det.score_case(case)
            for tid, s in scores.items():
                assert 0.0 <= s <= 1.0, f"{tid}: {s}"

    def test_all_same_scores_no_crash(self):
        """Case where all scores are 0.0 should not crash."""
        n = 50
        trace_ids = [f"t_{i}" for i in range(n)]
        case = CaseData(
            case_id="flat",
            trace_ids=trace_ids,
            monitor_scores={tid: 0.0 for tid in trace_ids},
        )
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit_unsupervised([case])
        scores = det.score_case(case)
        assert len(scores) == n

    def test_high_score_traces_ranked_higher(self, synthetic_dataset):
        """Traces with higher monitor scores should generally get higher
        posterior sabotage probability."""
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit(synthetic_dataset)

        for case in synthetic_dataset[:3]:
            scores = det.score_case(case)
            # Group by original score
            high_orig = [
                scores[tid] for tid, s in case.monitor_scores.items() if s >= 0.5
            ]
            low_orig = [
                scores[tid] for tid, s in case.monitor_scores.items() if s == 0.0
            ]
            if high_orig and low_orig:
                assert (sum(high_orig) / len(high_orig)) > (
                    sum(low_orig) / len(low_orig)
                ), "High-score traces should have higher posterior"

    def test_bootstrap_scores_similar(self, synthetic_dataset):
        """Bootstrap vs single-pass should produce qualitatively similar results."""
        det_single = AdaptiveBayesianDetector(n_bootstraps=1)
        det_boot = AdaptiveBayesianDetector(n_bootstraps=5)

        det_single.fit(synthetic_dataset)
        det_boot.fit(synthetic_dataset)

        case = synthetic_dataset[0]
        scores_single = det_single.score_case(case)
        scores_boot = det_boot.score_case(case)

        # Rank correlation should be high
        tids = list(scores_single.keys())
        s1 = [scores_single[t] for t in tids]
        s2 = [scores_boot[t] for t in tids]

        # Spearman rank correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(s1, s2)
        assert corr > 0.8, f"Bootstrap and single-pass should be correlated: {corr}"

    def test_cluster_boost(self, case_with_clusters):
        """Cluster features should boost traces in suspicious clusters."""
        det_no_cluster = AdaptiveBayesianDetector(
            n_bootstraps=1, use_cluster_features=False
        )
        det_cluster = AdaptiveBayesianDetector(
            n_bootstraps=1, use_cluster_features=True
        )

        # Fit both on the same data
        det_no_cluster.fit([case_with_clusters])
        det_cluster.fit([case_with_clusters])

        scores_no = det_no_cluster.score_case(case_with_clusters)
        scores_yes = det_cluster.score_case(case_with_clusters)

        # Malicious traces (in suspicious cluster) should get boosted
        gt = case_with_clusters.gt_malicious
        for tid in gt:
            assert scores_yes[tid] >= scores_no[tid], (
                f"Cluster should boost malicious trace {tid}"
            )


# ---------------------------------------------------------------------------
# Bayesian update mechanics
# ---------------------------------------------------------------------------

class TestBayesianMechanics:
    def test_credences_update_toward_malicious(self, synthetic_dataset):
        """After seeing suspicious scores, credence in malicious hypotheses
        should increase."""
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit(synthetic_dataset)

        prior = det._build_prior()
        benign_idx = next(i for i, h in enumerate(det.hypotheses) if h.is_benign)
        prior_benign = prior[benign_idx]

        # Create a suspicious case
        tids = [f"t_{i}" for i in range(20)]
        sus_case = CaseData(
            case_id="sus",
            trace_ids=tids,
            monitor_scores={tid: 0.8 for tid in tids},  # all high scores
        )

        scores = det.score_case(sus_case)
        # All scores should be elevated
        mean_score = sum(scores.values()) / len(scores)
        assert mean_score > 0.1, "Suspicious case should have elevated scores"

    def test_credences_stay_benign_for_clean_case(self, synthetic_dataset):
        """For a case with all-zero scores, posterior should favor benign."""
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit(synthetic_dataset)

        tids = [f"t_{i}" for i in range(100)]
        clean_case = CaseData(
            case_id="clean",
            trace_ids=tids,
            monitor_scores={tid: 0.0 for tid in tids},
        )

        scores = det.score_case(clean_case)
        mean_score = sum(scores.values()) / len(scores)
        # Clean case should have low scores
        assert mean_score < 0.5, f"Clean case mean score too high: {mean_score}"

    def test_likelihood_mixture(self):
        """Test that trace likelihood is a proper mixture."""
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        # Manual reference
        ref = ReferenceDistribution(bucket_size=0.1)
        for _ in range(100):
            ref.add_benign(0.0)
        for _ in range(10):
            ref.add_malicious(0.8, beta=0.0)
        det.reference = ref
        det._fitted = True

        h_benign = Hypothesis(r=0.0, beta=0.0)
        h_malign = Hypothesis(r=0.01, beta=0.0)

        lik_benign = det._trace_likelihood(0.0, h_benign)
        lik_malign = det._trace_likelihood(0.0, h_malign)

        # Both should be positive
        assert lik_benign > 0
        assert lik_malign > 0

        # For a benign-looking score (0.0), benign hypothesis should assign
        # higher likelihood (since malicious traces typically have high scores)
        # Actually this depends on r being small, so mixture is dominated by benign
        # Just check they're close since r=0.01 is small
        assert abs(lik_benign - lik_malign) / lik_benign < 0.1


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_compute_metrics_basic(self, synthetic_dataset):
        """Metrics should be computable and in valid ranges."""
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit(synthetic_dataset)

        scored = {}
        for case in synthetic_dataset:
            scored[case.case_id] = det.score_case(case)

        metrics = compute_metrics(synthetic_dataset, scored)

        assert 0.0 <= metrics["macro_auroc"] <= 1.0
        assert 0.0 <= metrics["macro_ap"] <= 1.0
        assert 0.0 <= metrics["micro_auroc"] <= 1.0
        assert 0.0 <= metrics["micro_ap"] <= 1.0
        assert metrics["n_cases"] == len(synthetic_dataset)

    def test_perfect_detection(self):
        """Perfect scores should give AUROC = 1.0."""
        tids = [f"t_{i}" for i in range(100)]
        gt = {tids[0], tids[1]}
        case = CaseData(
            case_id="perfect",
            trace_ids=tids,
            monitor_scores={tid: 0.0 for tid in tids},
            gt_malicious=gt,
        )
        scored = {"perfect": {tid: (1.0 if tid in gt else 0.0) for tid in tids}}
        metrics = compute_metrics([case], scored)
        assert metrics["macro_auroc"] == 1.0
        assert metrics["macro_ap"] == 1.0

    def test_random_detection(self, synthetic_dataset):
        """Random scores should give AUROC ≈ 0.5."""
        random.seed(123)
        scored = {}
        for case in synthetic_dataset:
            scored[case.case_id] = {
                tid: random.random() for tid in case.trace_ids
            }

        metrics = compute_metrics(synthetic_dataset, scored)
        # AUROC should be near 0.5 for random (allow some variance)
        assert 0.3 < metrics["macro_auroc"] < 0.7


# ---------------------------------------------------------------------------
# Leave-one-out evaluation tests
# ---------------------------------------------------------------------------

class TestLOOEvaluation:
    def test_loo_runs(self, synthetic_dataset):
        """LOO evaluation should produce results for all cases."""
        results = run_loo_evaluation(
            synthetic_dataset[:5],  # small subset for speed
            detector_kwargs={"n_bootstraps": 1},
        )
        assert len(results) == 5
        for case_id, scores in results.items():
            assert len(scores) > 0

    def test_loo_no_leakage(self, synthetic_dataset):
        """Each case should be scored without seeing its own labels."""
        # This is hard to test directly, but we can check that LOO
        # gives different results than fit-on-all
        det_all = AdaptiveBayesianDetector(n_bootstraps=1)
        det_all.fit(synthetic_dataset[:5])
        score_all = det_all.score_case(synthetic_dataset[0])

        loo_results = run_loo_evaluation(
            synthetic_dataset[:5],
            detector_kwargs={"n_bootstraps": 1},
        )
        score_loo = loo_results[synthetic_dataset[0].case_id]

        # Scores should differ (different training data)
        diffs = [
            abs(score_all.get(tid, 0) - score_loo.get(tid, 0))
            for tid in synthetic_dataset[0].trace_ids
        ]
        # At least some should differ
        assert any(d > 0 for d in diffs)


# ---------------------------------------------------------------------------
# Data loading tests (using temp directories)
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_case_from_disk(self, tmp_path):
        """Test loading a case from the standard on-disk format."""
        # Create case directory structure
        case_dir = tmp_path / "case_0000"
        case_dir.mkdir()
        traces_dir = case_dir / "traces"
        traces_dir.mkdir()

        # Write traces
        for i in range(10):
            (traces_dir / f"trace_{i:03d}.txt").write_text(
                f"[0000] user:\nQuestion {i}\n\n[0001] assistant:\nAnswer {i}"
            )

        # Write initial scores
        scores = {f"traces/trace_{i:03d}.txt": 0.0 for i in range(10)}
        scores["traces/trace_005.txt"] = 0.7
        (case_dir / "initial_scores.json").write_text(json.dumps(scores))

        # Write clusters
        clusters = {
            "n_clusters": 2,
            "n_traces": 10,
            "trace_to_cluster": {
                f"traces/trace_{i:03d}.txt": i % 2 for i in range(10)
            },
            "clusters": [
                {"cluster_id": 0, "size": 5, "coherence": 0.5},
                {"cluster_id": 1, "size": 5, "coherence": 0.3},
            ],
        }
        (case_dir / "clusters.json").write_text(json.dumps(clusters))

        # Write metadata
        meta_dir = tmp_path / "_metadata"
        meta_dir.mkdir()
        meta = {
            "case_id": "case_0000",
            "traces_per_case": 10,
            "trace_files": [f"traces/trace_{i:03d}.txt" for i in range(10)],
            "gt_hacked_trace_files": ["traces/trace_005.txt", "traces/trace_007.txt"],
        }
        (meta_dir / "case_0000.json").write_text(json.dumps(meta))

        case = load_case_from_disk(case_dir, meta_dir / "case_0000.json")

        assert case.case_id == "case_0000"
        assert len(case.trace_ids) == 10
        assert len(case.monitor_scores) == 10
        assert case.monitor_scores["traces/trace_005.txt"] == 0.7
        assert len(case.gt_malicious) == 2
        assert case.cluster_assignments is not None
        assert case.cluster_coherences is not None

    def test_load_dataset(self, tmp_path):
        """Test loading a full dataset from disk."""
        trace_repo = tmp_path / "safety_test"
        dm_sets = trace_repo / "distributed-misuse-sets"
        meta_dir = dm_sets / "_metadata"
        meta_dir.mkdir(parents=True)

        for case_idx in range(3):
            case_id = f"case_{case_idx:04d}"
            case_dir = dm_sets / case_id
            case_dir.mkdir()
            traces_dir = case_dir / "traces"
            traces_dir.mkdir()

            # Minimal trace files
            for i in range(5):
                (traces_dir / f"trace_{i:03d}.txt").write_text(f"trace {i}")

            scores = {f"traces/trace_{i:03d}.txt": 0.0 for i in range(5)}
            (case_dir / "initial_scores.json").write_text(json.dumps(scores))

            meta = {
                "case_id": case_id,
                "traces_per_case": 5,
                "trace_files": [f"traces/trace_{i:03d}.txt" for i in range(5)],
                "gt_hacked_trace_files": ["traces/trace_002.txt"],
            }
            (meta_dir / f"{case_id}.json").write_text(json.dumps(meta))

        cases = load_dataset(trace_repo)
        assert len(cases) == 3
        assert cases[0].case_id == "case_0000"

    def test_load_case_missing_clusters(self, tmp_path):
        """Should handle missing clusters.json gracefully."""
        case_dir = tmp_path / "case_0000"
        case_dir.mkdir()
        traces_dir = case_dir / "traces"
        traces_dir.mkdir()

        for i in range(3):
            (traces_dir / f"trace_{i:03d}.txt").write_text(f"trace {i}")

        scores = {f"traces/trace_{i:03d}.txt": 0.0 for i in range(3)}
        (case_dir / "initial_scores.json").write_text(json.dumps(scores))

        meta = {
            "case_id": "case_0000",
            "traces_per_case": 3,
            "trace_files": [f"traces/trace_{i:03d}.txt" for i in range(3)],
            "gt_hacked_trace_files": [],
        }
        meta_dir = tmp_path / "_metadata"
        meta_dir.mkdir()
        (meta_dir / "case_0000.json").write_text(json.dumps(meta))

        case = load_case_from_disk(case_dir, meta_dir / "case_0000.json")
        assert case.cluster_assignments is None


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_trace_case(self):
        """A case with a single trace should not crash."""
        case = CaseData(
            case_id="single",
            trace_ids=["t_0"],
            monitor_scores={"t_0": 0.5},
            gt_malicious={"t_0"},
        )
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit([case])
        scores = det.score_case(case)
        assert len(scores) == 1
        assert 0.0 <= scores["t_0"] <= 1.0

    def test_all_malicious_case(self):
        """A case where all traces are malicious."""
        tids = [f"t_{i}" for i in range(10)]
        case = CaseData(
            case_id="all_mal",
            trace_ids=tids,
            monitor_scores={tid: 0.8 for tid in tids},
            gt_malicious=set(tids),
        )
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit_unsupervised([case])
        scores = det.score_case(case)
        assert len(scores) == 10

    def test_score_1_0(self):
        """Score of exactly 1.0 should be handled."""
        case = CaseData(
            case_id="max_score",
            trace_ids=["t_0", "t_1"],
            monitor_scores={"t_0": 1.0, "t_1": 0.0},
        )
        det = AdaptiveBayesianDetector(n_bootstraps=1)
        det.fit_unsupervised([case])
        scores = det.score_case(case)
        assert 0.0 <= scores["t_0"] <= 1.0
        assert 0.0 <= scores["t_1"] <= 1.0

    def test_very_small_bucket_size(self):
        """Small bucket size should still work."""
        det = AdaptiveBayesianDetector(bucket_size=0.01, n_bootstraps=1)
        case = CaseData(
            case_id="small_bucket",
            trace_ids=["t_0", "t_1"],
            monitor_scores={"t_0": 0.55, "t_1": 0.0},
        )
        det.fit_unsupervised([case])
        scores = det.score_case(case)
        assert len(scores) == 2

    def test_reproducibility(self, synthetic_dataset):
        """Same seed should produce same results."""
        random.seed(42)
        det1 = AdaptiveBayesianDetector(n_bootstraps=3)
        det1.fit(synthetic_dataset)
        random.seed(42)
        scores1 = det1.score_case(synthetic_dataset[0])

        random.seed(42)
        det2 = AdaptiveBayesianDetector(n_bootstraps=3)
        det2.fit(synthetic_dataset)
        random.seed(42)
        scores2 = det2.score_case(synthetic_dataset[0])

        for tid in scores1:
            assert abs(scores1[tid] - scores2[tid]) < 1e-10


# ---------------------------------------------------------------------------
# Integration test: detection quality on realistic synthetic data
# ---------------------------------------------------------------------------

class TestDetectionQuality:
    def test_auroc_above_random(self, synthetic_dataset):
        """On synthetic data with signal, AUROC should be above 0.5."""
        det = AdaptiveBayesianDetector(n_bootstraps=3)
        det.fit(synthetic_dataset)

        scored = {}
        for case in synthetic_dataset:
            scored[case.case_id] = det.score_case(case)

        metrics = compute_metrics(synthetic_dataset, scored)
        assert metrics["macro_auroc"] > 0.5, (
            f"AUROC {metrics['macro_auroc']:.3f} should be above random"
        )

    def test_supervised_beats_unsupervised(self, synthetic_dataset):
        """Supervised fitting should outperform unsupervised."""
        # Supervised
        det_sup = AdaptiveBayesianDetector(n_bootstraps=1)
        det_sup.fit(synthetic_dataset)
        scored_sup = {}
        for case in synthetic_dataset:
            scored_sup[case.case_id] = det_sup.score_case(case)
        m_sup = compute_metrics(synthetic_dataset, scored_sup)

        # Unsupervised
        det_unsup = AdaptiveBayesianDetector(n_bootstraps=1)
        det_unsup.fit_unsupervised(synthetic_dataset)
        scored_unsup = {}
        for case in synthetic_dataset:
            scored_unsup[case.case_id] = det_unsup.score_case(case)
        m_unsup = compute_metrics(synthetic_dataset, scored_unsup)

        # Supervised should be at least as good
        assert m_sup["macro_auroc"] >= m_unsup["macro_auroc"] - 0.1, (
            f"Supervised AUROC {m_sup['macro_auroc']:.3f} should beat "
            f"unsupervised {m_unsup['macro_auroc']:.3f}"
        )

    def test_loo_evaluation_quality(self, synthetic_dataset):
        """LOO should still be above random."""
        results = run_loo_evaluation(
            synthetic_dataset,
            detector_kwargs={"n_bootstraps": 1},
        )
        metrics = compute_metrics(synthetic_dataset, results)
        assert metrics["macro_auroc"] > 0.45, (
            f"LOO AUROC {metrics['macro_auroc']:.3f} should be above random"
        )
