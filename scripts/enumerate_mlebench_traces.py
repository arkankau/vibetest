#!/usr/bin/env python3
"""Re-enumerate the OperandLinear-MLE-Bench public submission histories
into trace_NNN.txt files matching the indices used by the audit writeup.

Mirrors the discovery+materialization logic of
`experiments/external_trace_audits/mlebench_public_reward_hacking.py` but
skips the LLM-scoring pass: we only need the trace_NNN → submission
mapping so the cheatbench drop-off can name the 6 explicit FPs
(trace_020/062/090/113/114/139) and the strict-cheating trace_075.

Usage:
  git clone --depth 1 https://github.com/ramgorthi04/OperandLinear-MLE-Bench.git /tmp/mle-source/OperandLinear-MLE-Bench
  python scripts/enumerate_mlebench_traces.py \\
      --source-root /tmp/mle-source/OperandLinear-MLE-Bench \\
      --out-dir /tmp/mle-source/all_traces

Produces:
  <out-dir>/index.jsonl                  one row per trace
  <out-dir>/traces/trace_NNN.txt         enumerated transcripts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return json.dumps(v, ensure_ascii=False, indent=2)


def format_history(source_repo: str, task: str, run_id: str, relpath: str, history: list[dict]) -> str:
    lines = [
        "Public MLE-Bench submission history.",
        f"SOURCE_REPO: {source_repo}",
        f"TASK: {task}",
        f"RUN_ID: {run_id}",
        f"HISTORY_PATH: {relpath}",
        "",
    ]
    for idx, turn in enumerate(history, start=1):
        tn = turn.get("turn_number") if isinstance(turn.get("turn_number"), int) else idx
        lines.append(f'=== TURN {tn:04d} | {stringify(turn.get("timestamp")) or "unknown"} ===')
        if (lo := stringify(turn.get("llm_json"))):
            lines.extend(["LLM_OUTPUT:", lo])
        if (oc := stringify(turn.get("outcome"))):
            lines.extend(["TOOL_OUTCOME:", oc])
        extra = {k: v for k, v in turn.items() if k not in {"timestamp", "turn_number", "llm_json", "outcome"}}
        if extra:
            lines.extend(["EXTRA_FIELDS:", json.dumps(extra, ensure_ascii=False, indent=2)])
        lines.append("")
    return "\n".join(lines).strip()


def discover(source_root: Path) -> list[dict]:
    out = []
    for path in sorted(source_root.rglob("full_history.json")):
        if "MLE_Submission" not in path.parts:
            continue
        idx = path.parts.index("MLE_Submission")
        if idx < 1 or idx + 2 >= len(path.parts):
            continue
        source_repo = path.parts[idx - 1]
        task = path.parts[idx + 1]
        run_id = path.parts[idx + 2]
        results_path = path.with_name("competition_results.json")
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
        try:
            history = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(history, list) or not history:
            continue
        relpath = str(path.relative_to(source_root))
        out.append({
            "trace_id": f"{source_repo}::{task}::{run_id}",
            "source_repo": source_repo,
            "task": task,
            "run_id": run_id,
            "history_path": str(path),
            "history_relpath": relpath,
            "competition_results_path": str(results_path) if results_path.exists() else "",
            "turn_count": len(history),
            "transcript": format_history(source_repo, task, run_id, relpath, history),
            "score": results.get("score"),
            "metric": results.get("metric"),
            "valid_submission": results.get("valid_submission"),
            "medal_achieved": results.get("medal_achieved"),
            "total_runtime_seconds": results.get("total_runtime_seconds"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True, help="OperandLinear-MLE-Bench checkout")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    records = discover(args.source_root)
    print(f"Discovered {len(records)} traces")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    with (args.out_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(records):
            rel = f"traces/trace_{i:03d}.txt"
            (args.out_dir / rel).write_text(t["transcript"], encoding="utf-8")
            f.write(json.dumps({"trace_file": rel, **{k: v for k, v in t.items() if k != "transcript"}}) + "\n")
    print(f"Wrote {len(records)} trace files + index.jsonl to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
