"""
Adaptive Bayesian Sabotage Detector.

Adapts the macro-protocol from Wen et al. (arXiv:2411.17693) —
"Adaptive Deployment of Untrusted LLMs Reduces Distributed Threats" —
for sabotage/misuse detection in deployment trace corpora.

Key adaptations from the original paper:
  - Original: selects among micro-protocols to balance safety vs. performance.
  - Ours: we only care about detection. The Bayesian machinery is used to compute
    per-trace posterior sabotage probabilities.
  - Monitor scores come from the per-trace LLM scanner (Stage 1 initial_scores).
  - Hypotheses parameterize (misbehavior_rate r, sneakiness β) for the case.
  - Bootstrapping over reference subsets for robust credence estimation (§E.4).
  - Bucket-based likelihood estimation (§E.4, Algorithm 1).

Usage:
    detector = AdaptiveBayesianDetector()
    detector.fit(train_cases)           # build reference distributions
    scores = detector.score_case(case)  # per-trace sabotage probabilities
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A hypothesis about the attack profile of a case.

    Parameters
    ----------
    r : float
        Misbehavior rate — fraction of traces that are malicious (0 = benign).
    beta : float
        Sneakiness trade-off (0 = max harm/easy to detect, 1 = max evasion).
    """
    r: float
    beta: float

    @property
    def is_benign(self) -> bool:
        return self.r == 0.0

    def __repr__(self) -> str:
        return f"H(r={self.r:.3f}, β={self.beta:.1f})"


@dataclass
class CaseData:
    """All information needed to score a single case."""
    case_id: str
    trace_ids: list[str]
    monitor_scores: dict[str, float]         # trace_id -> initial score
    gt_malicious: set[str] | None = None     # ground-truth malicious trace IDs
    cluster_assignments: dict[str, int] | None = None  # trace_id -> cluster_id
    cluster_coherences: dict[int, float] | None = None  # cluster_id -> coherence
    cluster_sizes: dict[int, int] | None = None  # cluster_id -> size


@dataclass
class ReferenceDistribution:
    """Bucketed score distributions for likelihood computation.

    Following §E.4 of Wen et al., we bucket monitor scores and count
    matches to estimate P(s | hypothesis).
    """
    bucket_size: float
    benign_counts: dict[int, int] = field(default_factory=dict)  # bucket -> count
    benign_total: int = 0
    # malicious_counts[beta_bucket] -> {score_bucket -> count}
    malicious_counts: dict[int, dict[int, int]] = field(default_factory=dict)
    malicious_totals: dict[int, int] = field(default_factory=dict)

    def _bucket(self, score: float) -> int:
        """Map a score to its bucket index."""
        return int(math.floor(score / self.bucket_size))

    def _beta_bucket(self, beta: float) -> int:
        """Map sneakiness β to nearest available bucket."""
        available = sorted(self.malicious_counts.keys())
        if not available:
            return 0
        return min(available, key=lambda b: abs(b - round(beta / self.bucket_size)))

    def add_benign(self, score: float) -> None:
        b = self._bucket(score)
        self.benign_counts[b] = self.benign_counts.get(b, 0) + 1
        self.benign_total += 1

    def add_malicious(self, score: float, beta: float) -> None:
        bb = round(beta * 10)  # discretize β to 0..10
        if bb not in self.malicious_counts:
            self.malicious_counts[bb] = {}
            self.malicious_totals[bb] = 0
        sb = self._bucket(score)
        self.malicious_counts[bb][sb] = self.malicious_counts[bb].get(sb, 0) + 1
        self.malicious_totals[bb] += 1

    def p_score_benign(self, score: float, smoothing: float = 1.0) -> float:
        """P(score | benign) with Laplace smoothing."""
        b = self._bucket(score)
        count = self.benign_counts.get(b, 0)
        return (count + smoothing) / (self.benign_total + smoothing * self._n_buckets())

    def p_score_malicious(self, score: float, beta: float,
                          smoothing: float = 1.0) -> float:
        """P(score | malicious, β) with Laplace smoothing."""
        bb = self._beta_bucket_key(beta)
        if bb not in self.malicious_counts:
            # Fallback: use uniform
            return 1.0 / max(1, self._n_buckets())
        sb = self._bucket(score)
        counts = self.malicious_counts[bb]
        total = self.malicious_totals[bb]
        count = counts.get(sb, 0)
        return (count + smoothing) / (total + smoothing * self._n_buckets())

    def _beta_bucket_key(self, beta: float) -> int:
        bb = round(beta * 10)
        available = sorted(self.malicious_counts.keys())
        if not available:
            return bb
        return min(available, key=lambda b: abs(b - bb))

    def _n_buckets(self) -> int:
        """Estimate number of distinct buckets for smoothing."""
        all_buckets = set(self.benign_counts.keys())
        for mc in self.malicious_counts.values():
            all_buckets.update(mc.keys())
        return max(len(all_buckets), int(1.0 / self.bucket_size) + 1)


# ---------------------------------------------------------------------------
# Adaptive Bayesian Detector
# ---------------------------------------------------------------------------

class AdaptiveBayesianDetector:
    """
    Bayesian sabotage detector adapted from Wen et al. (2411.17693).

    The detector maintains a set of hypotheses H = {(r, β)} about the
    case's attack profile. For each trace, the monitor score is used as
    an observation to update credences via Bayes' rule. After processing
    all traces, per-trace posterior sabotage probabilities are computed.

    Parameters
    ----------
    r_values : list[float]
        Misbehavior rates to consider. Should include 0.0 (benign).
    beta_values : list[float]
        Sneakiness levels to consider.
    bucket_size : float
        Width of score buckets for likelihood estimation (paper §E.4).
    smoothing : float
        Laplace smoothing constant (paper's 'a' parameter).
    n_bootstraps : int
        Number of bootstrap resamples for robust credence estimation (paper §E.4).
    prior_benign : float
        Prior probability mass on the benign hypothesis (r=0).
    use_cluster_features : bool
        If True, incorporate cluster-level signals as additional features.
    """

    def __init__(
        self,
        r_values: list[float] | None = None,
        beta_values: list[float] | None = None,
        bucket_size: float = 0.1,
        smoothing: float = 1.0,
        n_bootstraps: int = 5,
        prior_benign: float = 0.5,
        use_cluster_features: bool = False,
    ):
        if r_values is None:
            r_values = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
        if beta_values is None:
            beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Build hypothesis grid
        self.hypotheses: list[Hypothesis] = []
        for r in r_values:
            if r == 0.0:
                # Single benign hypothesis (β irrelevant when r=0)
                self.hypotheses.append(Hypothesis(r=0.0, beta=0.0))
            else:
                for beta in beta_values:
                    self.hypotheses.append(Hypothesis(r=r, beta=beta))

        self.bucket_size = bucket_size
        self.smoothing = smoothing
        self.n_bootstraps = n_bootstraps
        self.prior_benign = prior_benign
        self.use_cluster_features = use_cluster_features

        self.reference: ReferenceDistribution | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting (building reference distributions)
    # ------------------------------------------------------------------

    def fit(self, cases: list[CaseData]) -> "AdaptiveBayesianDetector":
        """Build reference distributions from labeled training cases.

        Collects monitor scores for benign and malicious traces across
        all training cases to estimate P(s | benign) and P(s | malicious, β).

        For malicious traces, β is estimated from the score: traces with
        low scores despite being malicious are treated as high-β (sneaky),
        while traces with high scores are low-β (not sneaky).
        """
        ref = ReferenceDistribution(bucket_size=self.bucket_size)

        for case in cases:
            gt = case.gt_malicious or set()
            for tid, score in case.monitor_scores.items():
                if tid in gt:
                    # Estimate β from the score: lower score = higher sneakiness
                    # A malicious trace with score 0.0 is maximally sneaky (β≈1.0)
                    # A malicious trace with score 1.0 is not sneaky at all (β≈0.0)
                    beta_est = 1.0 - score
                    ref.add_malicious(score, beta_est)
                else:
                    ref.add_benign(score)

        self.reference = ref
        self._fitted = True
        return self

    def fit_unsupervised(self, cases: list[CaseData]) -> "AdaptiveBayesianDetector":
        """Build reference distributions without labels.

        Assumes the vast majority of traces are benign. Uses the empirical
        score distribution as the benign reference. Constructs synthetic
        malicious distributions by shifting scores upward.
        """
        ref = ReferenceDistribution(bucket_size=self.bucket_size)

        # Collect all scores as benign (>99% truly are)
        all_scores: list[float] = []
        for case in cases:
            for score in case.monitor_scores.values():
                ref.add_benign(score)
                all_scores.append(score)

        # If no data at all, use a default distribution
        if not all_scores:
            all_scores = [0.0]
            ref.add_benign(0.0)

        # Build synthetic malicious distributions for each β level
        # β=0 (not sneaky): malicious traces have high scores
        # β=1 (very sneaky): malicious traces look like benign
        for beta_int in range(0, 11, 2):  # 0, 2, 4, 6, 8, 10
            beta = beta_int / 10.0
            n_synthetic = max(100, len(all_scores) // 100)
            for _ in range(n_synthetic):
                if beta >= 1.0:
                    # Maximally sneaky: sample from benign distribution
                    synthetic_score = random.choice(all_scores)
                elif beta <= 0.0:
                    # Not sneaky: high scores
                    synthetic_score = min(1.0, 0.7 + random.random() * 0.3)
                else:
                    # Interpolate: with prob (1-β), draw from "obvious" (high score)
                    # with prob β, draw from benign
                    if random.random() < beta:
                        synthetic_score = random.choice(all_scores)
                    else:
                        synthetic_score = min(1.0, 0.5 + random.random() * 0.5)
                ref.add_malicious(synthetic_score, beta)

        self.reference = ref
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_case(self, case: CaseData) -> dict[str, float]:
        """Score all traces in a case using Bayesian updating.

        Returns dict of trace_id -> posterior sabotage probability.

        This implements Algorithm 1 from the paper (§E.4), adapted:
        - Step 1: Update credences via Bayes' rule over all traces
        - Step 2: Compute per-trace posterior sabotage probability
        - Bootstrapping: repeated with resampled reference for robustness
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() or fit_unsupervised().")

        scores = list(case.monitor_scores.items())  # [(trace_id, score), ...]

        if self.n_bootstraps > 1:
            return self._score_case_bootstrap(case, scores)
        else:
            return self._score_case_single(case, scores)

    def _score_case_single(
        self,
        case: CaseData,
        scores: list[tuple[str, float]],
    ) -> dict[str, float]:
        """Single-pass Bayesian scoring (no bootstrap)."""
        # Initialize prior
        prior = self._build_prior()
        credences = prior.copy()

        # Pre-compute per-trace likelihoods under each hypothesis
        # P(s_t | h_j) = (1-r_j)*P(s_t|benign) + r_j*P(s_t|malicious,β_j)
        trace_likelihoods: list[list[float]] = []  # [trace_idx][hyp_idx]

        for _, s in scores:
            liks = []
            for h in self.hypotheses:
                lik = self._trace_likelihood(s, h)
                liks.append(lik)
            trace_likelihoods.append(liks)

        # Step 1: Sequential Bayesian update (order-independent for final credences)
        for t, (_, s) in enumerate(scores):
            liks = trace_likelihoods[t]
            # Bayes update: c_j *= P(s_t | h_j)
            for j in range(len(self.hypotheses)):
                credences[j] *= liks[j]
            # Normalize
            total = sum(credences)
            if total > 0:
                credences = [c / total for c in credences]
            else:
                credences = [1.0 / len(credences)] * len(credences)

        # Pre-compute cluster campaign probabilities if using cluster features
        cluster_probs = None
        if self.use_cluster_features and case.cluster_assignments is not None:
            cluster_probs = self._compute_cluster_scores(case)

        # Step 2: Per-trace posterior sabotage probability
        result = {}
        for t, (tid, s) in enumerate(scores):
            p_malicious = self._trace_posterior(s, credences)

            # Optionally boost with cluster features
            if cluster_probs is not None:
                cluster_boost = self._cluster_boost(tid, case, cluster_probs)
                # Combine via noisy-or: P = 1 - (1-p_bayes)*(1-p_cluster)
                p_malicious = 1.0 - (1.0 - p_malicious) * (1.0 - cluster_boost)

            result[tid] = p_malicious

        return result

    def _score_case_bootstrap(
        self,
        case: CaseData,
        scores: list[tuple[str, float]],
    ) -> dict[str, float]:
        """Bootstrapped scoring following paper §E.4.

        Runs multiple passes with different reference subsets.
        Computes log-odds under normal approximation, then averages.
        """
        all_log_odds: dict[str, list[float]] = {tid: [] for tid, _ in scores}

        for boot_idx in range(self.n_bootstraps):
            # Score with a perturbed reference (add noise to smoothing)
            # Following the paper: resample reference distribution
            perturbed_smoothing = self.smoothing * (0.5 + random.random())
            orig_smoothing = self.smoothing
            self.smoothing = perturbed_smoothing

            trace_scores = self._score_case_single(case, scores)

            self.smoothing = orig_smoothing

            for tid, p in trace_scores.items():
                # Convert to log-odds (clamp to avoid inf)
                p_clamped = max(1e-10, min(1.0 - 1e-10, p))
                log_odds = math.log(p_clamped / (1.0 - p_clamped))
                all_log_odds[tid].append(log_odds)

        # Average log-odds and convert back to probability
        result = {}
        for tid in all_log_odds:
            mean_lo = sum(all_log_odds[tid]) / len(all_log_odds[tid])
            result[tid] = 1.0 / (1.0 + math.exp(-mean_lo))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prior(self) -> list[float]:
        """Construct prior distribution over hypotheses.

        Places prior_benign mass on H_0 (r=0), distributes rest uniformly.
        """
        n_malign = sum(1 for h in self.hypotheses if not h.is_benign)
        prior = []
        for h in self.hypotheses:
            if h.is_benign:
                prior.append(self.prior_benign)
            else:
                prior.append((1.0 - self.prior_benign) / max(1, n_malign))
        return prior

    def _trace_likelihood(self, score: float, hypothesis: Hypothesis) -> float:
        """P(score | hypothesis) using the reference distribution.

        For benign hypothesis (r=0): P(s|H_0) = P(s|benign)
        For malign hypothesis (r,β): mixture model
            P(s|H_j) = (1-r)*P(s|benign) + r*P(s|malicious,β)
        """
        assert self.reference is not None
        p_benign = self.reference.p_score_benign(score, self.smoothing)

        if hypothesis.is_benign:
            return p_benign

        p_malicious = self.reference.p_score_malicious(
            score, hypothesis.beta, self.smoothing
        )
        return (1.0 - hypothesis.r) * p_benign + hypothesis.r * p_malicious

    def _trace_posterior(
        self,
        score: float,
        credences: list[float],
    ) -> float:
        """Compute P(trace is malicious | score, credences).

        P(mal | s, data) = Σ_j c_j * [r_j * P(s|mal,β_j)] / P(s|H_j)
        """
        assert self.reference is not None
        p_mal = 0.0
        for j, h in enumerate(self.hypotheses):
            if h.is_benign:
                continue
            p_s_benign = self.reference.p_score_benign(score, self.smoothing)
            p_s_mal = self.reference.p_score_malicious(
                score, h.beta, self.smoothing
            )
            p_s_h = (1.0 - h.r) * p_s_benign + h.r * p_s_mal
            if p_s_h > 0:
                p_mal += credences[j] * (h.r * p_s_mal) / p_s_h
        return p_mal

    def _compute_cluster_scores(self, case: CaseData) -> dict[int, float]:
        """Compute per-cluster campaign probability using Bayesian reasoning.

        Key insight from data analysis: campaign traces form tight, small,
        coherent clusters where 5-6 out of 6 campaign steps land in the
        same cluster. This function computes P(cluster is campaign cluster)
        for each cluster.

        Features used:
        - Cluster size (campaign clusters are small, ~5-10 traces)
        - Coherence (campaign clusters are semantically tight)
        - Score concentration (fraction of traces with non-zero scores)
        - Max score in cluster
        - Whether the cluster matches the "campaign signature" pattern
        """
        if case.cluster_assignments is None:
            return {}

        # Build cluster -> traces mapping
        cluster_traces: dict[int, list[str]] = {}
        for tid, cid in case.cluster_assignments.items():
            if cid not in cluster_traces:
                cluster_traces[cid] = []
            cluster_traces[cid].append(tid)

        # Compute global statistics for normalization
        all_scores = list(case.monitor_scores.values())
        global_nonzero_rate = sum(1 for s in all_scores if s > 0) / max(1, len(all_scores))

        cluster_probs: dict[int, float] = {}

        for cid, traces in cluster_traces.items():
            size = len(traces)
            scores = [case.monitor_scores.get(t, 0.0) for t in traces]
            mean_score = sum(scores) / max(1, len(scores))
            max_score = max(scores) if scores else 0.0
            n_nonzero = sum(1 for s in scores if s > 0)
            nonzero_rate = n_nonzero / max(1, size)

            # Coherence
            coherence = 0.0
            if case.cluster_coherences and cid in case.cluster_coherences:
                coherence = case.cluster_coherences[cid]

            # --- Campaign signature scoring ---
            # 1. Size factor: campaign clusters are small (3-15 traces)
            #    Peak probability at size ~6 (typical decomp_level)
            if size <= 3:
                size_factor = 0.3
            elif size <= 8:
                size_factor = 1.0
            elif size <= 15:
                size_factor = 0.7
            elif size <= 25:
                size_factor = 0.3
            else:
                size_factor = 0.1

            # 2. Coherence factor: campaign clusters are coherent
            coherence_factor = coherence  # 0-1 range

            # 3. Score concentration: campaign clusters have elevated scores
            #    relative to the global rate
            if global_nonzero_rate > 0:
                concentration_ratio = nonzero_rate / global_nonzero_rate
            else:
                concentration_ratio = 1.0 if nonzero_rate > 0 else 0.0
            concentration_factor = min(1.0, concentration_ratio / 3.0)

            # 4. Max score: at least one trace with moderate score
            max_factor = min(1.0, max_score / 0.5) if max_score > 0 else 0.0

            # 5. Mean score elevation
            mean_factor = min(1.0, mean_score / 0.3)

            # Combine factors into campaign probability
            # Weighted geometric-ish mean emphasizing size + coherence + concentration
            if max_score > 0:
                prob = (
                    size_factor * 0.25
                    + coherence_factor * 0.25
                    + concentration_factor * 0.25
                    + max_factor * 0.15
                    + mean_factor * 0.10
                )
            else:
                # No suspicious traces at all → low probability
                # But size + coherence alone can still flag
                prob = (size_factor * 0.15 + coherence_factor * 0.15) * 0.3

            cluster_probs[cid] = min(1.0, max(0.0, prob))

        return cluster_probs

    def _cluster_boost(self, trace_id: str, case: CaseData,
                       cluster_probs: dict[int, float] | None = None) -> float:
        """Compute cluster-based suspicion boost for a trace.

        Uses the pre-computed cluster campaign probabilities to boost
        traces that belong to suspected campaign clusters.
        """
        if case.cluster_assignments is None:
            return 0.0

        cid = case.cluster_assignments.get(trace_id)
        if cid is None:
            return 0.0

        if cluster_probs is None:
            cluster_probs = self._compute_cluster_scores(case)

        return cluster_probs.get(cid, 0.0)


# ---------------------------------------------------------------------------
# Direct Bayesian Calibration (for small-case settings like MLE-sabotage)
# ---------------------------------------------------------------------------

class BayesianCalibrator:
    """Direct Bayesian score calibration without (r, β) decomposition.

    Better suited for settings with:
    - Few traces per case (4-10)
    - Higher positive rates (10-50%)
    - Moderate monitor discriminability

    Learns P(score | benign) and P(score | malicious) directly from labeled
    data, then computes calibrated posteriors:
        P(malicious | score, case) ∝ P(score | malicious) × P(malicious | case)

    Case-level Bayesian updating: processes all traces in a case to update
    the case-level prior P(case is positive), then recomputes per-trace posteriors.

    Parameters
    ----------
    bucket_size : float
        Width of score buckets for likelihood estimation.
    smoothing : float
        Laplace smoothing constant.
    n_bootstraps : int
        Number of bootstrap resamples.
    trace_prior_positive : float
        Prior probability that any individual trace is malicious.
    """

    def __init__(
        self,
        bucket_size: float = 0.1,
        smoothing: float = 1.0,
        n_bootstraps: int = 5,
        trace_prior_positive: float = 0.3,
    ):
        self.bucket_size = bucket_size
        self.smoothing = smoothing
        self.n_bootstraps = n_bootstraps
        self.trace_prior_positive = trace_prior_positive

        self._benign_counts: dict[int, int] = {}
        self._benign_total: int = 0
        self._malicious_counts: dict[int, int] = {}
        self._malicious_total: int = 0
        self._fitted = False

    def _bucket(self, score: float) -> int:
        return int(math.floor(score / self.bucket_size))

    def _n_buckets(self) -> int:
        all_b = set(self._benign_counts.keys()) | set(self._malicious_counts.keys())
        return max(len(all_b), int(1.0 / self.bucket_size) + 1)

    def fit(self, cases: list[CaseData]) -> "BayesianCalibrator":
        """Learn score distributions from labeled cases."""
        self._benign_counts = {}
        self._benign_total = 0
        self._malicious_counts = {}
        self._malicious_total = 0

        for case in cases:
            gt = case.gt_malicious or set()
            for tid, score in case.monitor_scores.items():
                b = self._bucket(score)
                if tid in gt:
                    self._malicious_counts[b] = self._malicious_counts.get(b, 0) + 1
                    self._malicious_total += 1
                else:
                    self._benign_counts[b] = self._benign_counts.get(b, 0) + 1
                    self._benign_total += 1

        self._fitted = True
        return self

    def _p_score(self, score: float, malicious: bool) -> float:
        """P(score | class) with Laplace smoothing."""
        b = self._bucket(score)
        n_b = self._n_buckets()
        if malicious:
            count = self._malicious_counts.get(b, 0)
            total = self._malicious_total
        else:
            count = self._benign_counts.get(b, 0)
            total = self._benign_total
        return (count + self.smoothing) / (total + self.smoothing * n_b)

    def score_case(self, case: CaseData) -> dict[str, float]:
        """Score all traces using Bayesian calibration with case-level update.

        1. Compute per-trace likelihood ratio
        2. Update case-level credence (is this a positive case?)
        3. Compute final per-trace posteriors informed by case context
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        scores = list(case.monitor_scores.items())

        # Step 1: Per-trace likelihood ratios
        lr = {}  # trace_id -> P(s|mal) / P(s|benign)
        for tid, s in scores:
            p_mal = self._p_score(s, malicious=True)
            p_ben = self._p_score(s, malicious=False)
            lr[tid] = p_mal / max(p_ben, 1e-15)

        # Step 2: Case-level Bayesian update
        # Prior: P(case positive) = based on positive_rate
        case_prior_pos = self.trace_prior_positive
        # Under H_pos: each trace is independently malicious with some rate
        # Under H_neg: all traces are benign
        # Simplified: use product of per-trace evidence
        log_lr_sum = sum(math.log(max(v, 1e-15)) for v in lr.values())
        case_log_odds = math.log(case_prior_pos / (1 - case_prior_pos)) + log_lr_sum * 0.1
        # Dampen the case-level update (0.1 factor) to prevent overconfidence
        case_posterior_pos = 1.0 / (1.0 + math.exp(-max(-50, min(50, case_log_odds))))

        # Step 3: Per-trace posteriors
        result = {}
        for tid, s in scores:
            # Direct Bayesian posterior for this trace
            prior_mal = self.trace_prior_positive
            p_s_mal = self._p_score(s, malicious=True)
            p_s_ben = self._p_score(s, malicious=False)
            p_s = prior_mal * p_s_mal + (1 - prior_mal) * p_s_ben
            if p_s > 0:
                p_mal_given_s = (prior_mal * p_s_mal) / p_s
            else:
                p_mal_given_s = prior_mal

            # Adjust by case-level evidence (mild boost/reduction)
            # If case looks positive, boost all trace posteriors slightly
            case_factor = case_posterior_pos / max(case_prior_pos, 1e-10)
            case_factor = min(2.0, max(0.5, case_factor))  # clamp
            adjusted = p_mal_given_s * case_factor
            result[tid] = max(0.0, min(1.0, adjusted))

        return result


def run_calibrated_loo(
    cases: list[CaseData],
    calibrator_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Leave-one-out evaluation using BayesianCalibrator."""
    if calibrator_kwargs is None:
        calibrator_kwargs = {}

    results: dict[str, dict[str, float]] = {}
    for i, test_case in enumerate(cases):
        train_cases = cases[:i] + cases[i + 1:]
        cal = BayesianCalibrator(**calibrator_kwargs)
        cal.fit(train_cases)
        results[test_case.case_id] = cal.score_case(test_case)

    return results


# ---------------------------------------------------------------------------
# Leave-one-out cross-validated evaluation
# ---------------------------------------------------------------------------

def run_loo_evaluation(
    cases: list[CaseData],
    detector_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Run leave-one-out cross-validated Bayesian scoring.

    For each case, fit on all other cases, then score the held-out case.
    Returns {case_id: {trace_id: score}}.
    """
    if detector_kwargs is None:
        detector_kwargs = {}

    results: dict[str, dict[str, float]] = {}

    for i, test_case in enumerate(cases):
        train_cases = cases[:i] + cases[i + 1:]
        detector = AdaptiveBayesianDetector(**detector_kwargs)
        if train_cases and any(c.gt_malicious for c in train_cases):
            detector.fit(train_cases)
        else:
            detector.fit_unsupervised(train_cases if train_cases else [test_case])
        results[test_case.case_id] = detector.score_case(test_case)

    return results


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_case_from_disk(
    case_dir: str | Path,
    metadata_path: str | Path,
    load_clusters: bool = True,
) -> CaseData:
    """Load a CaseData from the standard on-disk format.

    Parameters
    ----------
    case_dir : path
        Directory containing initial_scores.json, clusters.json, traces/
    metadata_path : path
        Path to _metadata/<case_id>.json
    load_clusters : bool
        Whether to load cluster assignments.
    """
    case_dir = Path(case_dir)
    metadata_path = Path(metadata_path)

    # Load metadata
    with open(metadata_path) as f:
        meta = json.load(f)

    case_id = meta["case_id"]
    trace_files = meta.get("trace_files", [])
    gt_hacked = set(meta.get("gt_hacked_trace_files", []))

    # Load initial scores
    scores_path = case_dir / "initial_scores.json"
    if scores_path.exists():
        with open(scores_path) as f:
            monitor_scores = json.load(f)
    else:
        monitor_scores = {tf: 0.0 for tf in trace_files}

    # Load clusters (optional)
    cluster_assignments = None
    cluster_coherences = None
    cluster_sizes = None
    if load_clusters:
        clusters_path = case_dir / "clusters.json"
        if clusters_path.exists():
            with open(clusters_path) as f:
                clusters_data = json.load(f)
            cluster_assignments = clusters_data.get("trace_to_cluster", {})
            cluster_coherences = {}
            cluster_sizes = {}
            for c in clusters_data.get("clusters", []):
                cid = c["cluster_id"]
                cluster_coherences[cid] = c.get("coherence", 0.0)
                cluster_sizes[cid] = c.get("size", 0)

    return CaseData(
        case_id=case_id,
        trace_ids=trace_files,
        monitor_scores=monitor_scores,
        gt_malicious=gt_hacked,
        cluster_assignments=cluster_assignments,
        cluster_coherences=cluster_coherences,
        cluster_sizes=cluster_sizes,
    )


def load_dataset(
    trace_repo_dir: str | Path,
    max_cases: int | None = None,
) -> list[CaseData]:
    """Load all cases from a distributed-misuse trace repo.

    Parameters
    ----------
    trace_repo_dir : path
        Directory like results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35/
    max_cases : int or None
        Maximum number of cases to load.
    """
    trace_repo_dir = Path(trace_repo_dir)
    dm_sets = trace_repo_dir / "distributed-misuse-sets"
    meta_dir = dm_sets / "_metadata"

    if not meta_dir.exists():
        raise FileNotFoundError(f"No _metadata directory at {meta_dir}")

    cases = []
    for meta_file in sorted(meta_dir.glob("case_*.json")):
        case_id = meta_file.stem
        case_dir = dm_sets / case_id

        if not case_dir.exists():
            continue
        if not (case_dir / "initial_scores.json").exists():
            continue

        case = load_case_from_disk(case_dir, meta_file)
        cases.append(case)

        if max_cases and len(cases) >= max_cases:
            break

    return cases


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    cases: list[CaseData],
    scored: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute AUROC and AP for scored cases.

    Returns dict with macro-averaged and per-case metrics.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    per_case_auroc = []
    per_case_ap = []
    all_y_true = []
    all_y_score = []

    per_case_details = {}

    for case in cases:
        if case.case_id not in scored:
            continue

        case_scores = scored[case.case_id]
        gt = case.gt_malicious or set()

        y_true = []
        y_score = []
        for tid in case.trace_ids:
            label = 1 if tid in gt else 0
            score = case_scores.get(tid, 0.0)
            y_true.append(label)
            y_score.append(score)

        all_y_true.extend(y_true)
        all_y_score.extend(y_score)

        # Per-case metrics (skip if no positive or all positive)
        if sum(y_true) > 0 and sum(y_true) < len(y_true):
            auroc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            per_case_auroc.append(auroc)
            per_case_ap.append(ap)
            per_case_details[case.case_id] = {
                "auroc": auroc,
                "ap": ap,
                "n_traces": len(y_true),
                "n_malicious": sum(y_true),
            }

    # Macro-averaged
    macro_auroc = sum(per_case_auroc) / len(per_case_auroc) if per_case_auroc else 0.0
    macro_ap = sum(per_case_ap) / len(per_case_ap) if per_case_ap else 0.0

    # Micro (pooled)
    micro_auroc = 0.0
    micro_ap = 0.0
    if sum(all_y_true) > 0 and sum(all_y_true) < len(all_y_true):
        micro_auroc = roc_auc_score(all_y_true, all_y_score)
        micro_ap = average_precision_score(all_y_true, all_y_score)

    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "micro_auroc": micro_auroc,
        "micro_ap": micro_ap,
        "n_cases": len(per_case_auroc),
        "per_case": per_case_details,
    }
