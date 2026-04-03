#!/usr/bin/env python3
"""
Materialize benign-only cases for Distributed Misuse datasets.

Creates 25 benign-only cases (case_0050 through case_0074) for each DM
trace repo by sampling benign traces from existing positive cases. No new
LLM calls are needed — all traces and scores already exist.

Usage:
    python experiments/materialize_dm_benign_cases.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRACE_REPOS_BASE = Path("results/safety/trace_repos")

DATASETS = {
    "cyber_bg100": {
        "repo": "safety_dm_cyber_d6_bg100_qwen35",
        "domain": "cyber",
        "bg": 100,
        "traces_per_case": 600,
    },
    "bio_bg100": {
        "repo": "safety_dm_bio_d6_bg100_qwen35",
        "domain": "bio",
        "bg": 100,
        "traces_per_case": 600,
    },
}

NUM_BENIGN_CASES = 25
FIRST_BENIGN_INDEX = 50  # case_0050 through case_0074
SEED = 42


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def collect_benign_traces(
    dm_sets_dir: Path,
    metadata_dir: Path,
) -> list[dict]:
    """Collect all benign traces from existing positive cases.

    Returns list of dicts: {"case_id", "trace_file", "score", "source_path"}.
    """
    benign_pool = []
    for meta_file in sorted(metadata_dir.glob("case_*.json")):
        with open(meta_file) as f:
            meta = json.load(f)

        case_id = meta["case_id"]
        gt_hacked = set(meta.get("gt_hacked_trace_files", []))

        # Skip if no ground truth (shouldn't happen for existing cases)
        if not gt_hacked:
            continue

        case_dir = dm_sets_dir / case_id
        scores_path = case_dir / "initial_scores.json"
        if not scores_path.exists():
            continue
        with open(scores_path) as f:
            scores = json.load(f)

        for trace_file in meta.get("trace_files", []):
            if trace_file in gt_hacked:
                continue  # skip malicious traces

            # Get score (may be keyed as "traces/trace_NNN.txt")
            score = scores.get(trace_file, 0.0)

            source_path = case_dir / trace_file
            if source_path.exists():
                benign_pool.append({
                    "case_id": case_id,
                    "trace_file": trace_file,
                    "score": score,
                    "source_path": str(source_path),
                })

    return benign_pool


def materialize_benign_cases(
    dataset_key: str,
    dry_run: bool = False,
) -> list[dict]:
    """Create benign-only cases for a DM dataset.

    Returns list of metadata dicts for the new cases.
    """
    ds = DATASETS[dataset_key]
    repo_path = TRACE_REPOS_BASE / ds["repo"]
    dm_sets_dir = repo_path / "distributed-misuse-sets"
    metadata_dir = dm_sets_dir / "_metadata"

    if not metadata_dir.exists():
        print(f"  ERROR: {metadata_dir} not found")
        return []

    print(f"\n  Collecting benign traces from {dataset_key}...")
    benign_pool = collect_benign_traces(dm_sets_dir, metadata_dir)
    print(f"  Found {len(benign_pool)} benign traces across existing cases")

    if len(benign_pool) < ds["traces_per_case"]:
        print(f"  ERROR: not enough benign traces ({len(benign_pool)} < {ds['traces_per_case']})")
        return []

    rng = random.Random(SEED + hash(dataset_key))
    traces_per_case = ds["traces_per_case"]
    created = []

    for case_offset in range(NUM_BENIGN_CASES):
        case_index = FIRST_BENIGN_INDEX + case_offset
        case_id = f"case_{case_index:04d}"
        case_dir = dm_sets_dir / case_id
        meta_path = metadata_dir / f"{case_id}.json"

        # Skip if already exists
        if meta_path.exists():
            print(f"    {case_id}: already exists, skipping")
            with open(meta_path) as f:
                created.append(json.load(f))
            continue

        # Sample benign traces (with replacement across cases to get enough)
        selected = rng.choices(benign_pool, k=traces_per_case)

        if dry_run:
            print(f"    {case_id}: would create {traces_per_case} benign traces (dry run)")
            continue

        # Create case directory
        traces_dir = case_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        trace_files = []
        initial_scores = {}
        trace_keys = []

        for idx, trace_info in enumerate(selected):
            filename = f"trace_{idx:03d}.txt"
            rel_path = f"traces/{filename}"
            dest_path = traces_dir / filename

            # Copy trace content
            shutil.copy2(trace_info["source_path"], dest_path)

            trace_files.append(rel_path)
            initial_scores[rel_path] = trace_info["score"]
            trace_keys.append(f"benign:{trace_info['case_id']}:{trace_info['trace_file']}")

        # Write initial_scores.json
        with open(case_dir / "initial_scores.json", "w") as f:
            json.dump(initial_scores, f, indent=2)

        # Write metadata
        meta = {
            "case_id": case_id,
            "traces_per_case": traces_per_case,
            "trace_files": trace_files,
            "trace_keys": trace_keys,
            "gt_hacked_trace_files": [],  # benign-only
            "campaign_size": 0,
            "benign_background_count": ds["bg"],
            "background_multiplier": ds["bg"],
            "domain": ds["domain"],
            "decomp_level": 6,
            "synthetic_benign_only": True,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        created.append(meta)
        n_nonzero = sum(1 for v in initial_scores.values() if v > 0)
        print(f"    {case_id}: created {traces_per_case} benign traces "
              f"({n_nonzero} non-zero scores)")

    return created


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Materialize benign-only DM cases"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing files")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to process (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("Materialize Benign-Only Distributed Misuse Cases")
    print("=" * 70)

    datasets = args.datasets or list(DATASETS.keys())
    all_created = {}

    for ds_key in datasets:
        if ds_key not in DATASETS:
            print(f"Unknown dataset: {ds_key}")
            continue
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_key}")
        print(f"{'='*50}")
        created = materialize_benign_cases(ds_key, dry_run=args.dry_run)
        all_created[ds_key] = created
        print(f"  Total: {len(created)} benign-only cases")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ds_key, cases in all_created.items():
        benign = sum(1 for c in cases if len(c.get("gt_hacked_trace_files", [])) == 0)
        print(f"  {ds_key}: {len(cases)} cases ({benign} benign-only)")


if __name__ == "__main__":
    main()
