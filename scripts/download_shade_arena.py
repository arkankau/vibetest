"""Download and materialize SHADE-Arena + CUA-SHADE-Arena datasets.

Downloads transcripts from ScaleAI/mrt on HuggingFace and materializes them as
single-trace cases (1 transcript per case) for use with the SHADE monitor,
buffer defense, EaE, and agent pipelines.

Includes:
- SHADE-Arena: 3 tasks (american_airlines_2, api_key_calendar_agendas_2, bank_transfer_2)
- CUA-SHADE-Arena: 15 tasks (admin, data science, finance, HR, research)

Usage:
    python scripts/download_shade_arena.py                    # Download + materialize all
    python scripts/download_shade_arena.py --skip-download     # Re-materialize from cached JSONL

Requires: huggingface_hub (pip install huggingface_hub)
Dataset: https://huggingface.co/datasets/ScaleAI/mrt
Papers: Kutasov et al. (2025) "SHADE-Arena"; Kale et al. (2025) "MRT"
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
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


def _infer_gt_label(transcript: dict, file_path: str) -> bool:
    """Infer ground truth sabotage label from transcript metadata or file path.

    SHADE-Arena transcripts have 'side_task_evaluation' field.
    CUA-SHADE-Arena transcripts encode the strategy in the directory name:
      - 'benign' in path → clean
      - 'naive', 'selective', 'manipulator', 'visible' in path → sabotage attempt
    """
    # SHADE-Arena: explicit field
    if "side_task_evaluation" in transcript:
        return bool(transcript["side_task_evaluation"])

    # CUA-SHADE-Arena: infer from directory name
    if "benign" in file_path:
        return False
    # Any attack strategy = sabotage
    for strategy in ["naive", "selective", "manipulator", "visible", "liar"]:
        if strategy in file_path:
            return True
    return False


def _parse_strategy(file_path: str) -> str:
    """Extract the attack strategy from the file path."""
    m = re.search(r"no_hint_(.+?)_thinking_time", file_path)
    if m:
        return m.group(1)
    # SHADE-Arena: strategy encoded differently
    if "/transcripts/" in file_path:
        return "standard"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/safety/shade-arena"))
    parser.add_argument("--trace-repos-dir", type=Path, default=Path("results/safety/trace_repos"))
    parser.add_argument("--dataset-name", type=str, default="safety_shade_arena")
    parser.add_argument(
        "--model-filter", type=str, default=None,
        help="Only include transcripts from this model (default: claude-3-5-sonnet-20241022)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-cua", action="store_true", default=True,
        help="Include CUA-SHADE-Arena tasks (default: True)",
    )
    parser.add_argument(
        "--no-cua", action="store_true",
        help="Exclude CUA-SHADE-Arena tasks (only original SHADE-Arena)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download; re-materialize from existing all_transcripts.jsonl",
    )
    parser.add_argument(
        "--base-only", action="store_true", default=True,
        help="Only download base transcripts (no scaffolding variants like hierarchical/sequential)",
    )
    args = parser.parse_args()

    include_cua = args.include_cua and not args.no_cua
    model_filter = args.model_filter or "claude-3-5-sonnet-20241022"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "all_transcripts.jsonl"

    if not args.skip_download:
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_id = "ScaleAI/mrt"
        print(f"Listing files in {repo_id}...")
        all_files = list_repo_files(repo_id, repo_type="dataset")

        # Collect transcript file paths
        transcript_files: list[str] = []

        # SHADE-Arena: task_pairs/TASK/transcripts/MODEL/MONITOR/transcript_N.json
        shade_files = [
            f for f in all_files
            if f.startswith("task_pairs/")
            and not f.startswith("task_pairs_agentcompany/")
            and f.endswith(".json")
            and "/transcripts/" in f
        ]
        transcript_files.extend(shade_files)
        print(f"SHADE-Arena: {len(shade_files)} transcript files")

        # CUA-SHADE-Arena: task_pairs_agentcompany/TASK/transcripts_TYPE/MODEL/MONITOR/transcript_N.json
        if include_cua:
            cua_files = []
            for f in all_files:
                if not (
                    f.startswith("task_pairs_agentcompany/")
                    and f.endswith(".json")
                    and "/transcripts_" in f
                ):
                    continue
                # Skip scaffolding variants (hierarchical, sequential, superaware, unaware)
                if args.base_only:
                    parts = f.split("/")
                    transcript_dir = parts[2] if len(parts) > 2 else ""
                    if any(
                        s in transcript_dir
                        for s in ["hierarchical", "sequential", "superaware", "unaware"]
                    ):
                        continue
                cua_files.append(f)
            transcript_files.extend(cua_files)
            print(f"CUA-SHADE-Arena: {len(cua_files)} transcript files (base only)")

        print(f"Total: {len(transcript_files)} files to download")

        # Download
        all_transcripts: list[dict] = []
        for i, tf in enumerate(transcript_files):
            if i % 100 == 0:
                print(f"  Downloading {i}/{len(transcript_files)}...")
            try:
                path = hf_hub_download(repo_id, tf, repo_type="dataset")
                data = json.loads(Path(path).read_text())
            except Exception as exc:
                print(f"  SKIP {tf}: {exc}")
                continue

            parts = tf.split("/")
            if tf.startswith("task_pairs_agentcompany/"):
                data["_task"] = parts[1]
                data["_model"] = parts[3]
                data["_source"] = "cua-shade-arena"
            else:
                data["_task"] = parts[1]
                data["_model"] = parts[3]
                data["_source"] = "shade-arena"
            data["_index"] = parts[-1].replace("transcript_", "").replace(".json", "")
            data["_file"] = tf
            data["_strategy"] = _parse_strategy(tf)
            data["_is_sabotage"] = _infer_gt_label(data, tf)
            all_transcripts.append(data)

        # Save combined JSONL
        with open(raw_path, "w") as f:
            for t in all_transcripts:
                f.write(json.dumps(t) + "\n")
        print(f"Saved {len(all_transcripts)} transcripts to {raw_path}")
    else:
        print(f"Loading from {raw_path}...")
        with open(raw_path) as f:
            all_transcripts = [json.loads(l) for l in f]
        # Ensure _is_sabotage field exists (for old downloads)
        for t in all_transcripts:
            if "_is_sabotage" not in t:
                t["_is_sabotage"] = _infer_gt_label(t, t.get("_file", ""))
            if "_strategy" not in t:
                t["_strategy"] = _parse_strategy(t.get("_file", ""))
            if "_source" not in t:
                t["_source"] = (
                    "cua-shade-arena"
                    if "agentcompany" in t.get("_file", "")
                    else "shade-arena"
                )
        print(f"Loaded {len(all_transcripts)} transcripts")

    # Filter by model
    filtered = [t for t in all_transcripts if t["_model"] == model_filter]
    if not filtered:
        print(f"No transcripts for model {model_filter}, using all")
        filtered = all_transcripts
    print(f"Using {len(filtered)} transcripts (model={model_filter})")

    # Summary
    print(f"\nBy source:")
    for source in sorted(set(t["_source"] for t in filtered)):
        subset = [t for t in filtered if t["_source"] == source]
        sab = sum(1 for t in subset if t["_is_sabotage"])
        print(f"  {source}: {len(subset)} ({sab} sabotage, {len(subset)-sab} clean)")

    print(f"\nBy task:")
    for task in sorted(set(t["_task"] for t in filtered)):
        subset = [t for t in filtered if t["_task"] == task]
        sab = sum(1 for t in subset if t["_is_sabotage"])
        print(f"  {task}: {len(subset)} ({sab} sab / {len(subset)-sab} clean)")

    print(f"\nBy strategy:")
    for strategy in sorted(set(t["_strategy"] for t in filtered)):
        count = sum(1 for t in filtered if t["_strategy"] == strategy)
        print(f"  {strategy}: {count}")

    total_sab = sum(1 for t in filtered if t["_is_sabotage"])
    print(f"\nTotal: {total_sab} sabotage / {len(filtered)-total_sab} clean = {len(filtered)}")

    # Materialize as single-trace cases (1 transcript per case)
    rng = random.Random(args.seed)
    rng.shuffle(filtered)

    cases_root = args.trace_repos_dir / args.dataset_name / "impossiblebench-sets"
    metadata_root = cases_root / "_metadata"
    # Clean up old materialization
    if cases_root.exists():
        import shutil
        shutil.rmtree(cases_root)
    metadata_root.mkdir(parents=True, exist_ok=True)

    # Also create mle-sabotage-sets symlink
    mle_link = args.trace_repos_dir / args.dataset_name / "mle-sabotage-sets"
    if mle_link.exists() or mle_link.is_symlink():
        mle_link.unlink()
    mle_link.symlink_to("impossiblebench-sets")

    for i, t in enumerate(filtered):
        case_id = f"case_{i:04d}"
        case_dir = cases_root / case_id / "traces"
        case_dir.mkdir(parents=True, exist_ok=True)

        conv = t.get("conversation", [])
        transcript = _clean_transcript(conv)

        trace_file = "traces/trace_00.txt"
        (cases_root / case_id / trace_file).write_text(transcript, encoding="utf-8")

        is_sabotage = t["_is_sabotage"]

        meta = {
            "case_id": case_id,
            "traces_per_case": 1,
            "trace_files": [trace_file],
            "trace_keys": [f'{t["_task"]}:{t["_model"]}:{t["_index"]}'],
            "gt_hacked_trace_files": [trace_file] if is_sabotage else [],
            "hack_pool_selector": "all",
            "task": t["_task"],
            "model": t["_model"],
            "source": t["_source"],
            "strategy": t["_strategy"],
            "side_task_success": is_sabotage,
        }
        (metadata_root / f"{case_id}.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8",
        )

    print(f"\nMaterialized {len(filtered)} cases at {cases_root}")
    pos = sum(1 for t in filtered if t["_is_sabotage"])
    print(f"Positive (sabotage): {pos}/{len(filtered)} ({pos/len(filtered):.0%})")
    print(f"\nReady to run experiments with --dataset-name {args.dataset_name}")


if __name__ == "__main__":
    main()
