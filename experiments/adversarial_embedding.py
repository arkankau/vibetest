#!/usr/bin/env python3
"""Black-box adversarial prefix optimization against text-embedding-3-small.

Goal: Find per-trace adversarial prefixes that, when prepended to campaign
traces, push their embeddings apart so they no longer cluster together.

Approach: Coordinate-descent in text space using the embedding API as a
black-box oracle.  For each campaign trace we search for a prefix that:
  (a) minimizes cosine similarity with other campaign traces, and
  (b) maximizes cosine similarity with a designated benign cluster centroid.

This is inspired by GASLITE (Bhatt et al., 2024) and corpus-poisoning
(Zhong et al., EMNLP 2023), adapted for a black-box setting where only
embedding API queries are available.

Usage
-----
# Phase 1: Evaluate baseline clustering (no API calls beyond initial embed)
python experiments/adversarial_embedding.py --phase baseline --case-dir <path>

# Phase 2: Optimize adversarial prefixes
python experiments/adversarial_embedding.py --phase optimize \
    --case-dir <path> --budget 200

# Phase 3: Apply optimized prefixes and evaluate clustering
python experiments/adversarial_embedding.py --phase evaluate \
    --case-dir <path> --prefix-file <path>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
    max_chars: int = 2000,
) -> np.ndarray:
    """Embed a list of texts. Returns (n, 1536) array."""
    client = _get_client()
    truncated = [t[:max_chars] for t in texts]
    all_embs: list[list[float]] = []
    for start in range(0, len(truncated), batch_size):
        batch = truncated[start : start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_embs.extend([row.embedding for row in resp.data])
    return np.array(all_embs, dtype=np.float64)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_coherence(vecs: np.ndarray) -> float:
    """Mean pairwise cosine similarity (same as pipeline)."""
    if len(vecs) < 2:
        return 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = vecs / norms
    sim = normed @ normed.T
    n = len(vecs)
    n_pairs = n * (n - 1) / 2
    return float((sim.sum() - n) / (2 * n_pairs))


# ---------------------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------------------

def load_case(case_dir: Path) -> dict:
    """Load trace data and metadata for a single case."""
    meta_dir = case_dir.parent / "_metadata"
    case_id = case_dir.name
    meta_path = meta_dir / f"{case_id}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    traces_dir = case_dir / "traces"
    trace_files = sorted(traces_dir.glob("trace_*.txt"))

    trace_data = {}
    for f in trace_files:
        rel = f"traces/{f.name}"
        trace_data[rel] = f.read_text(encoding="utf-8")

    campaign_traces = meta.get("gt_hacked_trace_files", [])

    return {
        "case_id": case_id,
        "case_dir": case_dir,
        "trace_data": trace_data,  # {rel_path: content}
        "campaign_traces": campaign_traces,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_clustering(
    trace_data: dict[str, str],
    campaign_traces: list[str],
    prefixes: dict[str, str] | None = None,
    verbose: bool = True,
) -> dict:
    """Embed all traces, cluster, and report campaign clustering metrics.

    Args:
        trace_data: {rel_path: content} for all traces.
        campaign_traces: list of rel_paths for campaign traces.
        prefixes: optional {rel_path: prefix_text} for campaign traces.

    Returns:
        dict with coherence, cluster assignments, etc.
    """
    # Prepare texts (apply prefixes to campaign traces)
    rel_paths = sorted(trace_data.keys())
    texts = []
    for rel in rel_paths:
        content = trace_data[rel]
        if prefixes and rel in prefixes:
            content = prefixes[rel] + content
        texts.append(content)

    if verbose:
        print(f"Embedding {len(texts)} traces...")
    embeddings = embed_texts(texts)

    # Cluster (coherence strategy: k = n/8)
    from sklearn.cluster import KMeans

    n = len(texts)
    k = max(5, n // 8)
    k = min(k, n - 1, 100)
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(embeddings)

    # Find campaign trace indices and their cluster assignments
    campaign_indices = [rel_paths.index(rel) for rel in campaign_traces if rel in rel_paths]
    campaign_embeddings = embeddings[campaign_indices]
    campaign_labels = [labels[i] for i in campaign_indices]

    # Campaign coherence (pairwise similarity)
    coherence = pairwise_coherence(campaign_embeddings)

    # How many clusters do campaign traces span?
    unique_clusters = set(campaign_labels)
    max_in_one = max(campaign_labels.count(c) for c in unique_clusters) if campaign_labels else 0

    if verbose:
        print(f"\n=== Campaign Clustering Metrics ===")
        print(f"  Campaign traces: {len(campaign_traces)}")
        print(f"  Pairwise coherence: {coherence:.4f}")
        print(f"  Unique clusters: {len(unique_clusters)}")
        print(f"  Max in one cluster: {max_in_one}/{len(campaign_traces)}")
        print(f"  Cluster assignments: {dict(zip(campaign_traces, campaign_labels))}")

    return {
        "coherence": coherence,
        "unique_clusters": len(unique_clusters),
        "max_in_one_cluster": max_in_one,
        "campaign_labels": dict(zip(campaign_traces, campaign_labels)),
        "campaign_embeddings": campaign_embeddings,
        "all_embeddings": embeddings,
        "all_labels": labels,
        "rel_paths": rel_paths,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Adversarial prefix optimization
# ---------------------------------------------------------------------------

# Large pool of diverse topic sentences for prefix search
_PREFIX_POOL = [
    # Cloud/DevOps
    "Discuss Kubernetes pod scheduling and autoscaling with horizontal pod autoscaler configuration.",
    "Explain AWS Lambda cold start optimization and API Gateway throttling strategies.",
    "Describe Docker multi-stage build patterns for Go microservices deployment.",
    "Compare Terraform and Pulumi for infrastructure as code in multi-cloud environments.",
    "Analyze GitOps workflows using ArgoCD for continuous deployment to EKS clusters.",
    # Data Science / ML
    "Explain gradient boosting hyperparameter tuning with Optuna for tabular regression tasks.",
    "Describe transformer attention mechanism dimensionality and multi-head attention scaling.",
    "Compare PyTorch DataLoader parallelism strategies for distributed training on multiple GPUs.",
    "Analyze BERT fine-tuning approaches for named entity recognition in biomedical text.",
    "Discuss reinforcement learning reward shaping for robotic manipulation tasks.",
    # Web Development
    "Explain React Server Components streaming architecture and Suspense boundaries.",
    "Describe GraphQL schema stitching patterns for federated microservice APIs.",
    "Compare Next.js ISR and SSG caching strategies for e-commerce product pages.",
    "Analyze WebSocket connection pooling for real-time collaborative editing applications.",
    "Discuss CSS container queries and responsive design patterns for dashboard layouts.",
    # Database
    "Explain PostgreSQL MVCC vacuum tuning and autovacuum threshold configuration.",
    "Describe MongoDB sharding key selection strategies for time-series IoT data.",
    "Compare Redis Cluster slot migration with Sentinel failover for session storage.",
    "Analyze ClickHouse materialized view optimization for analytics query acceleration.",
    "Discuss graph database traversal optimization in Neo4j for social network analysis.",
    # Mobile
    "Explain SwiftUI state management with ObservableObject and Environment injection.",
    "Describe Kotlin Coroutines structured concurrency for Android background task management.",
    "Compare Flutter widget rebuild optimization strategies with const constructors.",
    "Analyze React Native bridge performance for native module communication on iOS.",
    "Discuss mobile app deep linking with Universal Links and Android App Links.",
    # Networking / Systems
    "Explain BGP route reflector topology design for multi-site WAN optimization.",
    "Describe Linux eBPF XDP programs for high-performance packet filtering.",
    "Compare QUIC protocol multiplexing with HTTP/2 stream prioritization.",
    "Analyze DNS-over-HTTPS resolver selection and privacy implications.",
    "Discuss NVMe-oF RDMA storage fabric latency optimization for HPC workloads.",
    # Security (benign)
    "Explain OAuth 2.0 PKCE flow implementation for single-page application authentication.",
    "Describe mTLS certificate rotation strategies in Istio service mesh.",
    "Compare SAST and DAST tool integration in CI/CD pipeline for vulnerability scanning.",
    "Analyze zero-trust network architecture with BeyondCorp principles for remote workforce.",
    "Discuss hardware security module key management for FIPS 140-2 compliance.",
    # Biology / Science (benign)
    "Explain CRISPR-Cas9 guide RNA design principles for off-target minimization.",
    "Describe protein crystallography data collection at synchrotron beamlines.",
    "Compare single-cell RNA sequencing library preparation protocols for droplet-based methods.",
    "Analyze phylogenetic tree construction methods using maximum likelihood estimation.",
    "Discuss enzyme kinetics Michaelis-Menten parameter estimation from progress curves.",
    # Math / Statistics
    "Explain Bayesian hierarchical model MCMC convergence diagnostics with R-hat statistics.",
    "Describe principal component analysis dimensionality reduction for high-dimensional genomic data.",
    "Compare bootstrap and permutation test power for non-parametric two-sample testing.",
    "Analyze time series ARIMA model selection using AIC and cross-validation.",
    "Discuss causal inference with propensity score matching for observational study design.",
    # Misc technical
    "Explain Rust ownership and borrowing rules for concurrent data structure implementation.",
    "Describe FPGA high-level synthesis workflow from C++ to RTL using Vitis HLS.",
    "Compare WebAssembly WASI preview-2 component model with traditional native extensions.",
    "Analyze compiler optimization passes for loop vectorization in LLVM IR.",
    "Discuss quantum computing error correction codes for superconducting qubit architectures.",
]


def _build_extended_pool(n_sentences: int = 4) -> list[str]:
    """Build a large pool of multi-sentence prefix candidates.

    Generates all k-combinations (k=1..n_sentences) from _PREFIX_POOL
    to create longer, more embedding-dominant prefixes.
    """
    pool = list(_PREFIX_POOL)  # 1-sentence
    base = list(_PREFIX_POOL)
    # 2-sentence combos (all pairs with stride to maximize diversity)
    for i in range(len(base)):
        for stride in [1, 7, 13, 23]:
            j = (i + stride) % len(base)
            if i != j:
                pool.append(base[i] + " " + base[j])
    # 3-sentence combos (selective)
    if n_sentences >= 3:
        for i in range(0, len(base), 2):
            j = (i + 5) % len(base)
            k = (i + 11) % len(base)
            pool.append(base[i] + " " + base[j] + " " + base[k])
    # 4-sentence combos (selective)
    if n_sentences >= 4:
        for i in range(0, len(base), 3):
            j = (i + 3) % len(base)
            k = (i + 7) % len(base)
            m = (i + 13) % len(base)
            pool.append(base[i] + " " + base[j] + " " + base[k] + " " + base[m])
    return pool


def optimize_prefixes(
    trace_data: dict[str, str],
    campaign_traces: list[str],
    budget: int = 500,
    candidates_per_step: int = 50,
    stale_restarts: int = 5,
    verbose: bool = True,
) -> dict[str, str]:
    """Black-box coordinate-descent optimization of adversarial prefixes.

    Improved version with:
    - Much larger prefix pool (multi-sentence combinations)
    - No early stopping until budget exhausted or stale_restarts exceeded
    - Mutation-based restarts when stuck (combine best prefix with new candidate)
    - Higher candidate count per step

    Args:
        trace_data: all traces.
        campaign_traces: campaign trace rel_paths.
        budget: max embedding API batches.
        candidates_per_step: prefixes to try per iteration per trace.
        stale_restarts: stop after this many consecutive no-improvement rounds.
        verbose: print progress.

    Returns:
        {rel_path: best_prefix} for each campaign trace.
    """
    # Phase 1: Get baseline campaign embeddings
    campaign_texts = [trace_data[rel] for rel in campaign_traces]
    campaign_embs = embed_texts(campaign_texts)
    queries_used = 1

    baseline_coh = pairwise_coherence(campaign_embs)
    if verbose:
        print(f"Baseline campaign coherence: {baseline_coh:.4f}")
        print(f"Budget: {budget} API batches, {candidates_per_step} candidates/step")

    # Initialize
    best_prefixes: dict[str, str] = {rel: "" for rel in campaign_traces}
    current_embs = campaign_embs.copy()
    best_coherence = baseline_coh

    # Build large multi-sentence pool
    extended_pool = _build_extended_pool(n_sentences=4)
    if verbose:
        print(f"Prefix pool size: {len(extended_pool)}")

    # Phase 2: Coordinate descent with restarts
    n_traces = len(campaign_traces)
    iteration = 0
    stale_count = 0

    while queries_used < budget and stale_count < stale_restarts:
        iteration += 1
        round_improved = False

        for trace_idx in range(n_traces):
            if queries_used >= budget:
                break

            rel = campaign_traces[trace_idx]
            base_text = trace_data[rel]

            # Build candidates: mix of fresh pool samples + mutations of current best
            rng = random.Random(iteration * 1000 + trace_idx * 7 + stale_count * 31)
            n_fresh = max(1, candidates_per_step - 10)
            n_mutated = candidates_per_step - n_fresh

            fresh = rng.sample(extended_pool, min(n_fresh, len(extended_pool)))

            # Mutations: combine current best prefix with a new pool sentence
            current_best = best_prefixes[rel]
            mutated = []
            if current_best:
                for _ in range(n_mutated):
                    extra = rng.choice(_PREFIX_POOL)
                    if rng.random() < 0.5:
                        mutated.append(extra + " " + current_best)
                    else:
                        mutated.append(current_best + " " + extra)
            else:
                mutated = rng.sample(extended_pool, min(n_mutated, len(extended_pool)))

            candidates = fresh + mutated

            # Embed all candidates
            candidate_texts = [f"{prefix}\n\n{base_text}" for prefix in candidates]
            # Handle batching if candidates > 64
            candidate_embs = embed_texts(candidate_texts)
            queries_used += math.ceil(len(candidate_texts) / 64)

            # Score: minimize max cosine sim with any OTHER campaign trace
            # (using max instead of mean is more aggressive at splitting)
            other_embs = np.delete(current_embs, trace_idx, axis=0)

            best_score = float("inf")
            best_candidate_idx = -1
            for ci, emb in enumerate(candidate_embs):
                # Use a blend of max and mean similarity for robustness
                sims = [cosine_sim(emb, other) for other in other_embs]
                score = 0.5 * max(sims) + 0.5 * np.mean(sims)
                if score < best_score:
                    best_score = score
                    best_candidate_idx = ci

            # Current score for comparison
            current_sims = [cosine_sim(current_embs[trace_idx], other) for other in other_embs]
            current_score = 0.5 * max(current_sims) + 0.5 * np.mean(current_sims)

            if best_score < current_score - 1e-6:
                current_embs[trace_idx] = candidate_embs[best_candidate_idx]
                best_prefixes[rel] = candidates[best_candidate_idx]
                round_improved = True

                new_coherence = pairwise_coherence(current_embs)
                if verbose and new_coherence < best_coherence - 0.001:
                    print(
                        f"  iter={iteration} trace={trace_idx}: "
                        f"coh {best_coherence:.4f} -> {new_coherence:.4f} "
                        f"(Δ={new_coherence - baseline_coh:+.4f}, q={queries_used})"
                    )
                best_coherence = min(best_coherence, new_coherence)

        if round_improved:
            stale_count = 0
        else:
            stale_count += 1
            if verbose and stale_count < stale_restarts:
                print(f"  iter={iteration}: no improvement (stale {stale_count}/{stale_restarts})")

    final_coherence = pairwise_coherence(current_embs)

    # Compute max pairwise sim (worst-case for clustering)
    n = len(current_embs)
    max_pair_sim = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_sim(current_embs[i], current_embs[j])
            if s > max_pair_sim:
                max_pair_sim = s

    if verbose:
        print(f"\n=== Optimization Complete ===")
        print(f"  Baseline coherence:   {baseline_coh:.4f}")
        print(f"  Final coherence:      {final_coherence:.4f}")
        print(f"  Max pairwise sim:     {max_pair_sim:.4f}")
        print(f"  Delta:                {final_coherence - baseline_coh:+.4f}")
        print(f"  API queries used:     {queries_used}/{budget}")
        print(f"  Iterations:           {iteration}")
        for rel, prefix in best_prefixes.items():
            print(f"  {rel}: \"{prefix[:80]}{'...' if len(prefix)>80 else ''}\"")

    return best_prefixes


# ---------------------------------------------------------------------------
# Full pipeline: optimize + evaluate with all traces
# ---------------------------------------------------------------------------

def run_full_attack(
    case_dir: Path,
    budget: int = 200,
    candidates_per_step: int = 20,
    verbose: bool = True,
) -> dict:
    """Run the full adversarial prefix attack pipeline."""
    case = load_case(case_dir)
    trace_data = case["trace_data"]
    campaign_traces = case["campaign_traces"]

    if verbose:
        print(f"Case: {case['case_id']}")
        print(f"Total traces: {len(trace_data)}")
        print(f"Campaign traces: {len(campaign_traces)}")
        print()

    # Step 1: Baseline clustering (all traces)
    if verbose:
        print("=== Phase 1: Baseline ===")
    baseline = evaluate_clustering(trace_data, campaign_traces, verbose=verbose)

    # Step 2: Optimize prefixes (only campaign traces)
    if verbose:
        print(f"\n=== Phase 2: Prefix Optimization ===")
    prefixes = optimize_prefixes(
        trace_data, campaign_traces,
        budget=budget,
        candidates_per_step=candidates_per_step,
        verbose=verbose,
    )

    # Step 3: Evaluate with optimized prefixes (all traces)
    if verbose:
        print(f"\n=== Phase 3: Evaluation with Optimized Prefixes ===")
    result = evaluate_clustering(trace_data, campaign_traces, prefixes=prefixes, verbose=verbose)

    return {
        "baseline_coherence": baseline["coherence"],
        "baseline_max_in_one": baseline["max_in_one_cluster"],
        "baseline_unique_clusters": baseline["unique_clusters"],
        "optimized_coherence": result["coherence"],
        "optimized_max_in_one": result["max_in_one_cluster"],
        "optimized_unique_clusters": result["unique_clusters"],
        "prefixes": prefixes,
        "campaign_labels_baseline": baseline["campaign_labels"],
        "campaign_labels_optimized": result["campaign_labels"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Adversarial embedding prefix optimization.")
    p.add_argument(
        "--case-dir", type=Path, required=True,
        help="Path to case directory (e.g., .../distributed-misuse-sets/case_0000).",
    )
    p.add_argument(
        "--phase", choices=["baseline", "optimize", "full"], default="full",
        help="Phase to run (default: full).",
    )
    p.add_argument("--budget", type=int, default=200, help="Max embedding API batches.")
    p.add_argument("--candidates", type=int, default=20, help="Candidates per optimization step.")
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON file for optimized prefixes.",
    )
    args = p.parse_args()

    case = load_case(args.case_dir)

    if args.phase == "baseline":
        evaluate_clustering(case["trace_data"], case["campaign_traces"])

    elif args.phase == "optimize":
        prefixes = optimize_prefixes(
            case["trace_data"], case["campaign_traces"],
            budget=args.budget,
            candidates_per_step=args.candidates,
        )
        out = args.output or Path(f"optimized_prefixes_{case['case_id']}.json")
        out.write_text(json.dumps(prefixes, indent=2))
        print(f"\nSaved prefixes to {out}")

    elif args.phase == "full":
        result = run_full_attack(
            args.case_dir,
            budget=args.budget,
            candidates_per_step=args.candidates,
        )
        out = args.output or Path(f"attack_result_{case['case_id']}.json")
        # Convert non-serializable data
        def _json_safe(obj):
            if isinstance(obj, np.ndarray):
                return None
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            return obj

        serializable = {k: _json_safe(v) for k, v in result.items()
                       if not isinstance(v, np.ndarray)}
        out.write_text(json.dumps(serializable, indent=2))
        print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
