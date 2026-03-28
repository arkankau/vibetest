#!/usr/bin/env bash
# Download and convert CyBench traces from BoxPwnr-Traces
# Usage: bash experiments/download_cybench_traces.sh
set -euo pipefail

REPO_DIR="data/safety/cybench/BoxPwnr-Traces"
OUT_FILE="data/safety/cybench/cybench_successful_traces.jsonl"

echo "Cloning BoxPwnr-Traces..."
git clone --depth 1 https://github.com/0ca/BoxPwnr-Traces.git "$REPO_DIR"

echo "Converting successful traces to JSONL..."
python3 << 'PYEOF'
import json, jsonlines
from pathlib import Path

traces_root = Path("data/safety/cybench/BoxPwnr-Traces/cybench")
out_path = Path("data/safety/cybench/cybench_successful_traces.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

all_traces = []
for task_dir in sorted(traces_root.iterdir()):
    if not task_dir.is_dir() or task_dir.name == "README.md":
        continue
    traces_dir = task_dir / "traces"
    if not traces_dir.exists():
        continue
    for trace_dir in sorted(traces_dir.iterdir()):
        if not trace_dir.is_dir():
            continue
        stats_path = trace_dir / "stats.json"
        config_path = trace_dir / "config.json"
        report_path = trace_dir / "report.md"
        if not stats_path.exists() or not config_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        if stats.get("status") != "success":
            continue
        config = json.loads(config_path.read_text())
        content = report_path.read_text() if report_path.exists() else ""
        if not content.strip():
            continue
        all_traces.append({
            "task": task_dir.name,
            "model": config.get("model", "unknown"),
            "agent": "BoxPwnr",
            "result": "1.0",
            "trial_name": f"{task_dir.name}__{trace_dir.name}",
            "conversations": [
                {"role": "system", "content": f"CyBench CTF Task: {task_dir.name}"},
                {"role": "assistant", "content": content},
            ],
            "verifier_output": "",
        })

with jsonlines.open(str(out_path), mode="w") as w:
    for t in all_traces:
        w.write(t)
print(f"Converted {len(all_traces)} successful CyBench traces to {out_path}")
PYEOF

echo "Cleaning up clone..."
rm -rf "$REPO_DIR"
echo "Done. Traces at $OUT_FILE"
