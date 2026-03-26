"""Download and materialize SHADE-Arena dataset for sabotage detection experiments.

Downloads transcripts from ScaleAI/mrt on HuggingFace and materializes them as
impossiblebench-style cases for use with the buffer defense and agent pipelines.

Usage:
    python scripts/download_shade_arena.py

    # Then run buffer defense:
    python experiments/safety.py \
        --safety-mode impossiblebench \
        --method buffer \
        --buffer-size 1 \
        --buffer-merge replace \
        --model "together/Qwen/Qwen3.5-397B-A17B" \
        --dataset-name safety_shade_arena \
        --cases 124 --traces-per-case 1 --seed 42 \
        --vibetest-output-path results/shade_arena_buffer_qwen35.jsonl

    # Or run agent:
    python experiments/safety.py \
        --safety-mode impossiblebench \
        --method vibetest \
        --model "together/Qwen/Qwen3.5-397B-A17B" \
        --dataset-name safety_shade_arena \
        --cases 124 --traces-per-case 1 --seed 42 \
        --sandbox docker \
        --vibetest-output-path results/shade_arena_agent_qwen35.jsonl

Requires: huggingface_hub (pip install huggingface_hub)
Dataset: https://huggingface.co/datasets/ScaleAI/mrt
Paper: Kutasov et al. (2025), "SHADE-Arena", https://arxiv.org/abs/2506.15740
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/safety/shade-arena"))
    parser.add_argument("--trace-repos-dir", type=Path, default=Path("results/safety/trace_repos"))
    parser.add_argument("--dataset-name", type=str, default="safety_shade_arena")
    parser.add_argument("--model-filter", type=str, default=None,
                        help="Only include transcripts from this model (e.g. claude-3-5-sonnet-20241022)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    repo_id = "ScaleAI/mrt"
    print(f"Listing files in {repo_id}...")
    files = list_repo_files(repo_id, repo_type="dataset")
    transcript_files = [f for f in files if f.endswith(".json") and "/transcripts/" in f]
    print(f"Found {len(transcript_files)} transcript files")

    # Download all transcripts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_transcripts: list[dict] = []

    for i, tf in enumerate(transcript_files):
        if i % 50 == 0:
            print(f"  Downloading {i}/{len(transcript_files)}...")
        path = hf_hub_download(repo_id, tf, repo_type="dataset")
        data = json.loads(Path(path).read_text())
        parts = tf.split("/")
        data["_task"] = parts[1]
        data["_model"] = parts[3]
        data["_index"] = parts[-1].replace("transcript_", "").replace(".json", "")
        data["_file"] = tf
        all_transcripts.append(data)

    # Save raw combined file
    raw_path = args.output_dir / "all_transcripts.jsonl"
    with open(raw_path, "w") as f:
        for t in all_transcripts:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(all_transcripts)} transcripts to {raw_path}")

    # Filter by model if requested
    if args.model_filter:
        filtered = [t for t in all_transcripts if t["_model"] == args.model_filter]
        print(f"Filtered to {len(filtered)} transcripts for model {args.model_filter}")
    else:
        # Default: use claude-3-5-sonnet (most data, matches paper)
        filtered = [t for t in all_transcripts if t["_model"] == "claude-3-5-sonnet-20241022"]
        if not filtered:
            filtered = all_transcripts
        print(f"Using {len(filtered)} transcripts (claude-3-5-sonnet-20241022)")

    # Print summary
    from collections import Counter
    task_counts = Counter(t["_task"] for t in filtered)
    side_success = Counter(bool(t.get("side_task_evaluation")) for t in filtered)
    print(f"\nBy task:")
    for task, count in sorted(task_counts.items()):
        success = sum(1 for t in filtered if t["_task"] == task and t.get("side_task_evaluation"))
        print(f"  {task}: {count} transcripts, {success} sabotage success")
    print(f"Total: {side_success[True]} sabotage / {side_success[False]} clean")

    # Materialize as impossiblebench-style cases
    rng = random.Random(args.seed)
    rng.shuffle(filtered)

    cases_root = args.trace_repos_dir / args.dataset_name / "impossiblebench-sets"
    metadata_root = cases_root / "_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(filtered):
        case_id = f"case_{i:04d}"
        case_dir = cases_root / case_id / "traces"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Format conversation as transcript
        conv = t.get("conversation", [])
        lines = []
        for mi, msg in enumerate(conv):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            lines.append(f"[{mi:04d}] {role}:\n{content}")
        transcript = "\n\n".join(lines)

        trace_file = "traces/trace_00.txt"
        (cases_root / case_id / trace_file).write_text(transcript, encoding="utf-8")

        is_sabotage = bool(t.get("side_task_evaluation"))

        meta = {
            "case_id": case_id,
            "traces_per_case": 1,
            "trace_files": [trace_file],
            "trace_keys": [f'{t["_task"]}:{t["_model"]}:{t["_index"]}'],
            "gt_hacked_trace_files": [trace_file] if is_sabotage else [],
            "hack_pool_selector": "all",
            "task": t["_task"],
            "model": t["_model"],
            "side_task_success": is_sabotage,
        }
        (metadata_root / f"{case_id}.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    print(f"\nMaterialized {len(filtered)} cases at {cases_root}")
    pos = sum(1 for t in filtered if t.get("side_task_evaluation"))
    print(f"Positive (sabotage success): {pos}/{len(filtered)} ({pos/len(filtered):.0%})")
    print("\nReady to run experiments with --dataset-name", args.dataset_name)


if __name__ == "__main__":
    main()
