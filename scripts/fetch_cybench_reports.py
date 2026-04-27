#!/usr/bin/env python3
"""Fetch all successful BoxPwnr-Traces cybench reports from S3.

Walks /tmp/cb-fetch/BoxPwnr-Traces/cybench/ for trace folders whose
stats.json has status=="success". For each, downloads report.md from
S3 and emits one JSON line into /tmp/cb-fetch/cybench_full.jsonl with:

    {trace_id, task, model, timestamp, status, stats, config, report_md, source_url}

Deletes nothing locally (the BoxPwnr-Traces clone is small).
"""
from __future__ import annotations
import json
import sys
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/tmp/cb-fetch/BoxPwnr-Traces/cybench")
OUT = Path("/tmp/cb-fetch/cybench_full.jsonl")
S3_BASE = "https://boxpwnr-traces.s3.us-east-1.amazonaws.com/cybench"


def list_successful() -> list[dict]:
    rows = []
    for stats_path in ROOT.rglob("stats.json"):
        try:
            s = json.loads(stats_path.read_text())
        except Exception:
            continue
        if s.get("status") != "success":
            continue
        trace_dir = stats_path.parent
        timestamp = trace_dir.name
        task = trace_dir.parent.parent.name
        config_path = trace_dir / "config.json"
        try:
            cfg = json.loads(config_path.read_text())
        except Exception:
            cfg = {}
        rows.append({
            "task": task,
            "timestamp": timestamp,
            "model": cfg.get("model", "unknown"),
            "stats": s,
            "config": cfg,
            "url": f"{S3_BASE}/{urllib.parse.quote(task)}/traces/{urllib.parse.quote(timestamp)}/report.md",
        })
    return rows


def fetch(row: dict) -> dict | None:
    try:
        req = urllib.request.Request(row["url"], headers={"User-Agent": "cheatbench-extras/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            row["report_md"] = r.read().decode("utf-8", errors="replace")
        return row
    except urllib.error.HTTPError as e:
        if e.code == 404:
            row["report_md"] = None
            row["error"] = "404"
            return row
        row["error"] = f"http_{e.code}"
        return row
    except Exception as e:
        row["error"] = str(e)
        return row


def main():
    rows = list_successful()
    print(f"successful traces: {len(rows)}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    errors = 0
    with OUT.open("w") as out, ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(fetch, r) for r in rows]
        for f in as_completed(futs):
            row = f.result()
            if row is None:
                continue
            if row.get("error"):
                errors += 1
            out.write(json.dumps(row) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)} ({errors} errors)", file=sys.stderr)
    print(f"done: {done}/{len(rows)} ({errors} errors)  ->  {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
