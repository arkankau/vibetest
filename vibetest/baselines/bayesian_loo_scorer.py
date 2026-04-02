"""
Bayesian LOO Scorer — bridge between disk-based cases and the Bayesian detector.

Runs leave-one-out Bayesian scoring on materialized case directories and writes
per-case bayesian_scores.json + bayesian_meta.json files for consumption by the
agent pipeline.

Usage:
    from vibetest.baselines.bayesian_loo_scorer import run_bayesian_loo_scoring
    run_bayesian_loo_scoring(trace_repo_dir)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from vibetest.baselines.adaptive_bayesian import (
    AdaptiveBayesianDetector,
    CaseData,
    load_dataset,
    run_loo_evaluation,
)


DEFAULT_DETECTOR_KWARGS = {
    "bucket_size": 0.1,
    "smoothing": 1.0,
    "n_bootstraps": 5,
    "prior_benign": 0.5,
    "use_cluster_features": True,
}


def _get_cluster_campaign_probs(
    case: CaseData,
    detector: AdaptiveBayesianDetector,
) -> dict[str, float]:
    """Get per-cluster campaign probabilities from a fitted detector.

    Returns {cluster_id_str: probability}.
    """
    cluster_probs = detector._compute_cluster_scores(case)
    return {str(cid): float(prob) for cid, prob in cluster_probs.items()}


def _get_case_level_posterior(
    case_scores: dict[str, float],
    cluster_probs: dict[str, float],
) -> float:
    """Compute a case-level posterior: max cluster campaign probability.

    This is more discriminative than a noisy-OR over trace scores (which
    saturates to 1.0 for cases with many traces). The max cluster campaign
    probability directly reflects whether any cluster looks like a campaign.
    """
    if cluster_probs:
        return max(cluster_probs.values())
    if case_scores:
        return max(case_scores.values())
    return 0.0


def run_bayesian_loo_scoring(
    trace_repo_dir: str | Path,
    *,
    detector_kwargs: dict[str, Any] | None = None,
    max_cases: int | None = None,
    force: bool = False,
    top_n: int = 20,
) -> dict[str, dict[str, float]]:
    """Run LOO Bayesian scoring and write bayesian_scores.json per case.

    For each case i:
      1. Load all cases' CaseData from disk
      2. Fit detector on cases[0..i-1] + cases[i+1..N]
      3. Score case i
      4. Write case_dir/bayesian_scores.json
      5. Write case_dir/bayesian_meta.json

    Parameters
    ----------
    trace_repo_dir : path
        Path to trace repo (e.g., results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35)
    detector_kwargs : dict
        Kwargs for AdaptiveBayesianDetector. Defaults to bayesian_cluster config.
    max_cases : int
        Limit on number of cases.
    force : bool
        Re-score even if bayesian_scores.json already exists.
    top_n : int
        Number of top traces to include in bayesian_meta.json.

    Returns
    -------
    dict[str, dict[str, float]]
        {case_id: {trace_id: bayesian_score}}
    """
    trace_repo_dir = Path(trace_repo_dir)
    if detector_kwargs is None:
        detector_kwargs = dict(DEFAULT_DETECTOR_KWARGS)

    cases = load_dataset(trace_repo_dir, max_cases=max_cases)
    if not cases:
        print(f"  No cases found in {trace_repo_dir}")
        return {}

    print(f"  Running Bayesian LOO scoring on {len(cases)} cases...")

    random.seed(42)
    np.random.seed(42)

    # LOO evaluation: for each case, fit on all others, score held-out
    all_scored: dict[str, dict[str, float]] = {}
    dm_sets = trace_repo_dir / "distributed-misuse-sets"

    for i, test_case in enumerate(cases):
        case_dir = dm_sets / test_case.case_id

        # Skip if already scored (unless force)
        scores_path = case_dir / "bayesian_scores.json"
        if not force and scores_path.exists():
            with open(scores_path) as f:
                all_scored[test_case.case_id] = json.load(f)
            continue

        # Fit on all other cases
        train_cases = cases[:i] + cases[i + 1:]
        detector = AdaptiveBayesianDetector(**detector_kwargs)
        if train_cases and any(c.gt_malicious for c in train_cases):
            detector.fit(train_cases)
        else:
            detector.fit_unsupervised(train_cases if train_cases else [test_case])

        # Score held-out case
        case_scores = detector.score_case(test_case)
        all_scored[test_case.case_id] = case_scores

        # Get cluster campaign probabilities
        cluster_probs = _get_cluster_campaign_probs(test_case, detector)

        # Compute case-level posterior
        case_posterior = _get_case_level_posterior(case_scores, cluster_probs)

        # Top-N most suspicious traces by Bayesian score
        sorted_traces = sorted(case_scores.items(), key=lambda x: x[1], reverse=True)
        high_suspicion = [tid for tid, s in sorted_traces[:top_n] if s > 0]

        # Write bayesian_scores.json
        with open(case_dir / "bayesian_scores.json", "w") as f:
            json.dump(case_scores, f, indent=2)

        # Write bayesian_meta.json
        meta = {
            "case_level_posterior": round(case_posterior, 6),
            "cluster_campaign_probs": {k: round(v, 4) for k, v in
                                       sorted(cluster_probs.items(),
                                              key=lambda x: x[1], reverse=True)},
            "high_suspicion_traces": high_suspicion,
            "n_traces_scored": len(case_scores),
            "n_nonzero_bayesian": sum(1 for s in case_scores.values() if s > 0),
            "detector_config": detector_kwargs,
        }
        with open(case_dir / "bayesian_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if (i + 1) % 10 == 0 or i == len(cases) - 1:
            print(f"    Scored {i + 1}/{len(cases)} cases")

    print(f"  Done. Wrote bayesian_scores.json + bayesian_meta.json to {len(all_scored)} cases.")
    return all_scored
