"""Materialize SHADE-Arena transcripts as variable-size cases for sabotage detection.

Creates cases with N traces each, mixing sabotage and clean transcripts.
Supports variable case size and sabotage proportion for sweep experiments.

Requires: data/safety/shade-arena/all_transcripts.jsonl (from download_shade_arena.py)

Usage:
    # Single-trace baseline (matches paper)
    python scripts/materialize_shade_arena_multi.py --traces-per-case 1

    # Case size sweep
    for tpc in 10 25 50; do
      python scripts/materialize_shade_arena_multi.py \\
        --traces-per-case $tpc --sabotage-per-case $((tpc/5)) \\
        --mix-tasks --dataset-name safety_shade_arena_tpc${tpc}
    done

    # All-in-one case
    python scripts/materialize_shade_arena_multi.py \\
      --traces-per-case 423 --sabotage-per-case 186 \\
      --mix-tasks --num-cases 1 --positive-rate 1.0 \\
      --dataset-name safety_shade_arena_all
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def _clean_transcript(conv: list[dict]) -> str:
    """Format conversation with thinking stripped and system prompt redacted."""
    lines = []
    for mi, msg in enumerate(conv):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
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


def _trace_key(t: dict, is_sab: bool) -> str:
    label = "sabotage" if is_sab else "clean"
    return f'{label}:{t["_task"]}:{t["_model"]}:{t["_index"]}'


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path,
                        default=Path("data/safety/shade-arena/all_transcripts.jsonl"))
    parser.add_argument("--trace-repos-dir", type=Path,
                        default=Path("results/safety/trace_repos"))
    parser.add_argument("--dataset-name", type=str, default="safety_shade_arena_multi")
    parser.add_argument("--model-filter", type=str, default="claude-3-5-sonnet-20241022")
    parser.add_argument("--exclude-tasks", type=str, default="bank_transfer_2")
    parser.add_argument("--num-cases", type=int, default=None,
                        help="Number of cases (default: auto based on pool size)")
    parser.add_argument("--traces-per-case", type=int, default=4)
    parser.add_argument("--max-sabotage-per-case", type=int, default=None,
                        help="Max sabotage traces per positive case (default: max(1, tpc//4)). "
                             "Actual count is uniformly distributed from 1..max across positive cases.")
    parser.add_argument("--positive-rate", type=float, default=0.5,
                        help="Fraction of cases that are positive (have sabotage traces)")
    parser.add_argument("--mix-tasks", action="store_true",
                        help="Mix tasks freely within cases (default: task-homogeneous)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load and filter transcripts
    with open(args.data_path) as f:
        all_data = [json.loads(l) for l in f]
    print(f"Loaded {len(all_data)} transcripts")

    if args.model_filter:
        all_data = [t for t in all_data if t["_model"] == args.model_filter]
        print(f"Filtered to {len(all_data)} for model {args.model_filter}")

    exclude = set(t.strip() for t in args.exclude_tasks.split(",") if t.strip())
    if exclude:
        all_data = [t for t in all_data if t["_task"] not in exclude]
        print(f"After excluding {exclude}: {len(all_data)}")

    # Ensure _is_sabotage field exists
    for t in all_data:
        if "_is_sabotage" not in t:
            t["_is_sabotage"] = bool(t.get("side_task_evaluation"))

    rng = random.Random(args.seed)

    if args.mix_tasks:
        # Combined pools across all tasks
        sab_pool = [t for t in all_data if t["_is_sabotage"]]
        clean_pool = [t for t in all_data if not t["_is_sabotage"]]
        rng.shuffle(sab_pool)
        rng.shuffle(clean_pool)
        print(f"\nMixed-task pools: {len(sab_pool)} sab + {len(clean_pool)} clean")

        # Auto-compute num_cases if not specified
        if args.num_cases is None:
            args.num_cases = (len(sab_pool) + len(clean_pool)) // args.traces_per_case
        n_positive = int(round(args.positive_rate * args.num_cases))
        n_negative = args.num_cases - n_positive

        task_configs = [("mixed", sab_pool, clean_pool, n_positive, n_negative)]
    else:
        # Task-homogeneous: per-task pools
        task_pools: dict[str, dict[str, list]] = defaultdict(lambda: {"sabotage": [], "clean": []})
        for t in all_data:
            if t["_is_sabotage"]:
                task_pools[t["_task"]]["sabotage"].append(t)
            else:
                task_pools[t["_task"]]["clean"].append(t)

        valid_tasks = [t for t in task_pools if task_pools[t]["sabotage"] and task_pools[t]["clean"]]
        print(f"\nTask-homogeneous: {len(valid_tasks)} valid tasks")

        # Allocate cases per task
        total_pool = sum(
            len(task_pools[t]["sabotage"]) + len(task_pools[t]["clean"]) for t in valid_tasks
        )
        if args.num_cases is None:
            args.num_cases = total_pool // args.traces_per_case

        task_configs = []
        remaining = args.num_cases
        for i, task in enumerate(sorted(valid_tasks)):
            sp = task_pools[task]["sabotage"]
            cp = task_pools[task]["clean"]
            pool = len(sp) + len(cp)
            if i == len(valid_tasks) - 1:
                n = remaining
            else:
                n = max(2, round(args.num_cases * pool / total_pool))
                remaining -= n
            n_pos = int(round(args.positive_rate * n))
            n_neg = n - n_pos
            rng.shuffle(sp)
            rng.shuffle(cp)
            task_configs.append((task, sp, cp, n_pos, n_neg))
            print(f"  {task}: {n} cases ({n_pos} pos, {n_neg} neg)")

    # Materialize
    cases_root = args.trace_repos_dir / args.dataset_name / "mle-sabotage-sets"
    metadata_root = cases_root / "_metadata"
    if cases_root.exists():
        shutil.rmtree(cases_root)
    metadata_root.mkdir(parents=True, exist_ok=True)

    # Also create impossiblebench-sets symlink
    ib_link = args.trace_repos_dir / args.dataset_name / "impossiblebench-sets"
    if ib_link.exists() or ib_link.is_symlink():
        ib_link.unlink()
    ib_link.symlink_to("mle-sabotage-sets")

    # Compute max sabotage per case
    effective_max_sab = args.max_sabotage_per_case
    if effective_max_sab is None:
        effective_max_sab = max(1, args.traces_per_case // 4)
    effective_max_sab = max(1, min(effective_max_sab, args.traces_per_case))

    case_index = 0
    for task_name, sab_pool, clean_pool, n_positive, n_negative in task_configs:
        total_task_cases = n_positive + n_negative

        # Randomly assign which case indices are positive (interleaved ordering)
        positive_indices = set(rng.sample(range(total_task_cases), n_positive)) if n_positive else set()

        # Stratified sabotage counts: uniformly distribute 1..max across positive cases
        possible_counts = list(range(1, effective_max_sab + 1))
        repeats, rem = divmod(n_positive, len(possible_counts))
        positive_hacked_counts = possible_counts * repeats + possible_counts[:rem]
        rng.shuffle(positive_hacked_counts)
        positive_cursor = 0

        for ci in range(total_task_cases):
            case_id = f"case_{case_index:04d}"
            (cases_root / case_id / "traces").mkdir(parents=True, exist_ok=True)

            is_positive = ci in positive_indices

            if is_positive:
                n_sab = positive_hacked_counts[positive_cursor]
                positive_cursor += 1
                n_clean = args.traces_per_case - n_sab
                selected = (
                    [(t, True) for t in rng.choices(sab_pool, k=n_sab)]
                    + [(t, False) for t in rng.choices(clean_pool, k=n_clean)]
                )
            else:
                n_sab = 0
                selected = [(t, False) for t in rng.choices(clean_pool, k=args.traces_per_case)]

            rng.shuffle(selected)

            trace_files, gt_hacked, trace_keys = [], [], []
            for ti, (t, is_sab) in enumerate(selected):
                rel = f"traces/trace_{ti:02d}.txt"
                trace_files.append(rel)
                trace_keys.append(_trace_key(t, is_sab))
                if is_sab:
                    gt_hacked.append(rel)
                transcript = _clean_transcript(t.get("conversation", []))
                (cases_root / case_id / rel).write_text(transcript, encoding="utf-8")

            with open(cases_root / case_id / "index.jsonl", "w") as f:
                for rel in trace_files:
                    f.write(json.dumps({"trace_file": rel}) + "\n")
            (cases_root / case_id / "all_traces.txt").write_text("", encoding="utf-8")

            meta = {
                "case_id": case_id,
                "traces_per_case": args.traces_per_case,
                "trace_files": trace_files,
                "trace_keys": trace_keys,
                "gt_hacked_trace_files": gt_hacked,
                "hack_pool_selector": "all",
                "task": task_name,
                "model_filter": args.model_filter,
                "sabotage_per_case": n_sab if is_positive else 0,
            }
            (metadata_root / f"{case_id}.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8",
            )
            case_index += 1

    total = case_index
    pos = sum(
        1 for i in range(total)
        if json.loads((metadata_root / f"case_{i:04d}.json").read_text()).get("gt_hacked_trace_files")
    )
    print(f"\nMaterialized {total} cases at {cases_root}")
    print(f"  traces_per_case={args.traces_per_case}, max_sabotage_per_case={effective_max_sab}")
    print(f"  Positive: {pos}/{total} ({pos/total:.0%})")
    print(f"\nRun with: --dataset-name {args.dataset_name} --cases {total} --traces-per-case {args.traces_per_case}")


if __name__ == "__main__":
    main()
