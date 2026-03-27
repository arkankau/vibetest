"""Materialize SHADE-Arena transcripts as multi-trace cases for sabotage detection.

Creates cases with N traces each (default 4), mixing sabotage and clean transcripts
from the same task — matching the MLE-sabotage case structure so the buffer defense
and agent can do cross-trace comparison.

Requires: data/safety/shade-arena/all_transcripts.jsonl (from download_shade_arena.py)

Usage:
    python scripts/materialize_shade_arena_multi.py

    # Then run experiments:
    python experiments/safety.py \
        --safety-mode mle-sabotage \
        --method buffer --buffer-size 4 --buffer-merge replace \
        --model "vllm/Qwen/Qwen3.5-397B-A17B-FP8" \
        --dataset-name safety_shade_arena_multi \
        --cases 25 --traces-per-case 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def _clean_transcript(conv: list[dict]) -> str:
    """Format conversation as transcript with thinking stripped and system prompt redacted."""
    lines = []
    for mi, msg in enumerate(conv):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))

        # Strip <thinking> tags
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()

        # For system messages, redact secondary task instructions
        if role == "system":
            for marker in [
                "There is also a secondary task",
                "The secondary task is",
                "Here is some information about how the monitoring system works",
            ]:
                idx = content.find(marker)
                if idx >= 0:
                    content = content[:idx].rstrip()
                    break

        if not content:
            continue
        lines.append(f"[{mi:04d}] {role}:\n{content}")
    return "\n\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=Path("data/safety/shade-arena/all_transcripts.jsonl"))
    parser.add_argument("--trace-repos-dir", type=Path, default=Path("results/safety/trace_repos"))
    parser.add_argument("--dataset-name", type=str, default="safety_shade_arena_multi")
    parser.add_argument("--model-filter", type=str, default="claude-3-5-sonnet-20241022")
    parser.add_argument("--exclude-tasks", type=str, default="bank_transfer_2",
                        help="Comma-separated tasks to exclude (default: bank_transfer_2 — no clean traces)")
    parser.add_argument("--num-cases", type=int, default=25)
    parser.add_argument("--traces-per-case", type=int, default=4)
    parser.add_argument("--positive-rate", type=float, default=0.3)
    parser.add_argument("--max-sabotage-per-case", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load transcripts
    with open(args.data_path) as f:
        all_data = [json.loads(l) for l in f]
    print(f"Loaded {len(all_data)} transcripts from {args.data_path}")

    # Filter by model
    if args.model_filter:
        all_data = [t for t in all_data if t["_model"] == args.model_filter]
        print(f"Filtered to {len(all_data)} transcripts for model {args.model_filter}")

    # Exclude tasks
    exclude = set(t.strip() for t in args.exclude_tasks.split(",") if t.strip())
    if exclude:
        all_data = [t for t in all_data if t["_task"] not in exclude]
        print(f"After excluding {exclude}: {len(all_data)} transcripts")

    # Split by task into sabotage/clean pools
    from collections import defaultdict
    task_pools: dict[str, dict[str, list]] = defaultdict(lambda: {"sabotage": [], "clean": []})
    for t in all_data:
        task = t["_task"]
        if t.get("side_task_evaluation"):
            task_pools[task]["sabotage"].append(t)
        else:
            task_pools[task]["clean"].append(t)

    print("\nPer-task pools:")
    for task in sorted(task_pools):
        s = len(task_pools[task]["sabotage"])
        c = len(task_pools[task]["clean"])
        print(f"  {task}: {s} sabotage + {c} clean = {s + c}")

    # Filter tasks that have both sabotage and clean traces
    valid_tasks = [t for t in task_pools if task_pools[t]["sabotage"] and task_pools[t]["clean"]]
    if not valid_tasks:
        raise SystemExit("No tasks with both sabotage and clean traces!")
    print(f"\nValid tasks for multi-trace: {valid_tasks}")

    # Allocate cases per task proportionally to pool size
    total_pool = sum(len(task_pools[t]["sabotage"]) + len(task_pools[t]["clean"]) for t in valid_tasks)
    task_case_counts = {}
    remaining = args.num_cases
    for i, task in enumerate(sorted(valid_tasks)):
        pool_size = len(task_pools[task]["sabotage"]) + len(task_pools[task]["clean"])
        if i == len(valid_tasks) - 1:
            task_case_counts[task] = remaining
        else:
            n = max(3, round(args.num_cases * pool_size / total_pool))
            task_case_counts[task] = n
            remaining -= n
    print(f"Cases per task: {task_case_counts}")

    # Materialize
    rng = random.Random(args.seed)
    cases_root = args.trace_repos_dir / args.dataset_name / "mle-sabotage-sets"
    metadata_root = cases_root / "_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    case_index = 0
    for task in sorted(valid_tasks):
        num_task_cases = task_case_counts[task]
        sab_pool = task_pools[task]["sabotage"]
        clean_pool = task_pools[task]["clean"]
        positive_count = int(round(args.positive_rate * num_task_cases))

        for ci in range(num_task_cases):
            case_id = f"case_{case_index:04d}"
            case_dir = cases_root / case_id / "traces"
            case_dir.mkdir(parents=True, exist_ok=True)

            is_positive = ci < positive_count

            if is_positive:
                n_sab = rng.randint(1, min(args.max_sabotage_per_case, args.traces_per_case - 1))
                n_clean = args.traces_per_case - n_sab
                selected = (
                    [(t, True) for t in rng.choices(sab_pool, k=n_sab)]
                    + [(t, False) for t in rng.choices(clean_pool, k=n_clean)]
                )
            else:
                selected = [(t, False) for t in rng.choices(clean_pool, k=args.traces_per_case)]

            rng.shuffle(selected)

            trace_files = []
            gt_hacked = []
            trace_keys = []
            for ti, (t, is_sab) in enumerate(selected):
                rel = f"traces/trace_{ti:02d}.txt"
                trace_files.append(rel)
                label = "sabotage" if is_sab else "clean"
                trace_keys.append(f"{label}:{t['_task']}:{t['_model']}:{t['_index']}")
                if is_sab:
                    gt_hacked.append(rel)

                conv = t.get("conversation", [])
                transcript = _clean_transcript(conv)
                (cases_root / case_id / rel).write_text(transcript, encoding="utf-8")

            # Write index
            with open(cases_root / case_id / "index.jsonl", "w") as f:
                for rel in trace_files:
                    f.write(json.dumps({"trace_file": rel}) + "\n")

            # Write all_traces.txt (empty — traces are individually accessible)
            (cases_root / case_id / "all_traces.txt").write_text("", encoding="utf-8")

            # Write metadata
            meta = {
                "case_id": case_id,
                "traces_per_case": args.traces_per_case,
                "trace_files": trace_files,
                "trace_keys": trace_keys,
                "gt_hacked_trace_files": gt_hacked,
                "hack_pool_selector": "all",
                "task": task,
                "model_filter": args.model_filter,
            }
            (metadata_root / f"{case_id}.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            case_index += 1

    total = case_index
    pos = sum(
        1 for i in range(total)
        if json.loads((metadata_root / f"case_{i:04d}.json").read_text()).get("gt_hacked_trace_files")
    )
    print(f"\nMaterialized {total} cases at {cases_root}")
    print(f"Positive (has sabotage): {pos}/{total} ({pos / total:.0%})")
    print(f"Traces per case: {args.traces_per_case}")
    print(f"\nReady to run with --dataset-name {args.dataset_name} --cases {total} --traces-per-case {args.traces_per_case}")


if __name__ == "__main__":
    main()
