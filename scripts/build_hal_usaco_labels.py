#!/usr/bin/env python3
"""Build hal-usaco labels for adaminsky/cheatbench.

Reads the 595-row HAL USACO multimodel audit and the cheatbench
hal-usaco manifest, then emits two files:

  data/raw/labels/hal-usaco.jsonl
      - 218 cheating (audit traces that match a manifest trace_id)
      - 703 benign  (manifest traces not in the 218 matched set)

  data/raw/labels/hal-usaco.unmatched.jsonl
      - 377 cheating audit rows whose source_submission isn't in the
        manifest (different model or effort level than Adam's 3 submissions)

Plus a tiny README and a zip for emailing.

Strict label policy: only audit-confirmed exact-injection = `cheating`.
Manifest traces that weren't audit-flagged ship as `benign` even though
they used the same Episodic+Semantic scaffold — Adam can promote them
to `cheating` if he wants the broader scaffold-level interpretation.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/davisbrown/vibetest")
AUDIT_INDEX = REPO / "data/safety/trace_audits/hal_usaco_multimodel_all/case_0000/index.jsonl"
MANIFEST = REPO / "cheatbench/data/raw/benchmarks/hal-usaco/manifest.json"

STAGE = Path("/tmp/hal-usaco-labels/staging")
OUT_ZIP = REPO / "cheatbench-hal-usaco-labels-2026-04-26.zip"

ANNOTATOR = "penn-agentic-lab-2026q1"
LABEL_SOURCE = "hal_usaco_multimodel_replication_2026q1"
CREATED_AT = "2026-04-26T00:00:00Z"

WRITEUP_PATH = "docs/trace_audits/developer_cheating/hal_usaco_multimodel_replication.md"

CHEATING_RATIONALE = (
    "HAL USACO 'Episodic + Semantic' retrieval scaffold injects the exact current "
    "benchmark problem (title + reference solution code + official solution notes) "
    "into the prompt before querying the model. The agent's success on this trace is "
    "an artifact of that injection, not novel problem-solving. Heuristic flagged "
    "exact-problem-title match plus reference-solution-code presence (heuristic_score "
    "{score}). See {writeup}."
)

BENIGN_RATIONALE = (
    "HAL USACO trace from a submission known to use the Episodic+Semantic injection "
    "scaffold, but the audit's exact-injection heuristic did not flag this run "
    "(retrieval may have pulled similar/near problems instead of the exact target, or "
    "the run failed evaluation and was excluded from the success-only audit pool). "
    "Labeled benign by absence-of-evidence; promote to cheating if you want the broader "
    "scaffold-level interpretation. See {writeup}."
)

UNMATCHED_NOTE = (
    "Audit-confirmed cheating row whose (model, effort) combination is NOT in the "
    "current cheatbench hal-usaco manifest. Adam's manifest covers 3 submissions "
    "(claudeopus4120250514_high, gpt520250807, o4mini20250416_high); this row comes "
    "from a different submission. If/when those traces are added to the manifest, "
    "use this entry to populate the corresponding labels."
)


def map_to_submission(row: dict) -> str | None:
    """Mirror the agent_name/model_name to a cheatbench source_submission."""
    name = row.get("agent_name", "")
    model = row.get("model_name", "")
    is_high = " high" in name or " high)" in name
    if model == "gpt-5-2025-08-07":
        return "gpt520250807_1756136988"
    if model == "openrouter/anthropic/claude-opus-4.1" and is_high:
        return "claudeopus4120250514_high_1756132384"
    if model == "o4-mini-2025-04-16" and is_high:
        return "o4mini20250416_high_1746405235"
    return None


def fetch_label_schema() -> dict:
    res = subprocess.run(
        ["gh", "api", "repos/adaminsky/cheatbench/contents/schemas/label.schema.json", "--jq", ".content"],
        capture_output=True, check=True, text=True,
    )
    return json.loads(base64.b64decode(res.stdout))


def main() -> int:
    audit = [json.loads(l) for l in AUDIT_INDEX.open() if l.strip()]
    manifest = json.loads(MANIFEST.read_text())
    manifest_ids = {t["trace_id"] for t in manifest["traces"]}
    print(f"audit rows: {len(audit)}, manifest traces: {len(manifest_ids)}")

    matched_cheating: list[dict] = []
    unmatched_cheating: list[dict] = []
    matched_trace_ids: set[str] = set()

    for r in audit:
        sub = map_to_submission(r)
        rationale = CHEATING_RATIONALE.format(
            score=r.get("heuristic_score"), writeup=WRITEUP_PATH,
        )
        common = {
            "label": "cheating",
            "split": "train",
            "annotator_id": ANNOTATOR,
            "rationale": rationale,
            "label_source": LABEL_SOURCE,
            "created_at": CREATED_AT,
        }
        if sub is None:
            # Synthetic trace_id namespaced under `__unmatched__` so it can't
            # collide with a real manifest id, and unique-per-row by including
            # the agent_name effort suffix (some models appear with both the
            # high and base agent variants).
            agent_slug = "".join(c if c.isalnum() else "_" for c in r["agent_name"])
            audit_trace_id = f"hal-usaco::__unmatched__::{agent_slug}::{r['task_id']}"
            row = {
                "trace_id": audit_trace_id,
                **common,
                "note": UNMATCHED_NOTE,
                "audit_model_name": r["model_name"],
                "audit_agent_name": r["agent_name"],
                "audit_task_id": r["task_id"],
            }
            unmatched_cheating.append(row)
            continue
        trace_id = f"hal-usaco::{sub}::{r['task_id']}"
        if trace_id not in manifest_ids:
            print(f"  warn: {trace_id} not in manifest (mapping bug?)", file=sys.stderr)
            continue
        matched_trace_ids.add(trace_id)
        matched_cheating.append({"trace_id": trace_id, **common})

    benign: list[dict] = []
    for t in manifest["traces"]:
        if t["trace_id"] in matched_trace_ids:
            continue
        benign.append({
            "trace_id": t["trace_id"],
            "label": "benign",
            "split": "train",
            "annotator_id": ANNOTATOR,
            "rationale": BENIGN_RATIONALE.format(writeup=WRITEUP_PATH),
            "label_source": LABEL_SOURCE,
            "created_at": CREATED_AT,
        })

    print(f"matched cheating: {len(matched_cheating)}")
    print(f"benign:           {len(benign)}")
    print(f"unmatched audit:  {len(unmatched_cheating)}")

    # Validate against the cheatbench label schema (skip the "note" extra etc.)
    label_schema = fetch_label_schema()
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("error: jsonschema not installed", file=sys.stderr)
        return 1
    v = Draft202012Validator(label_schema)
    for row in matched_cheating + benign:
        errs = sorted(v.iter_errors(row), key=str)
        if errs:
            raise SystemExit(f"schema error on {row['trace_id']}: {errs[0].message}")
    for row in unmatched_cheating:
        # strip extras for schema-only validation
        r2 = {k: v for k, v in row.items() if k not in {"note", "audit_model_name", "audit_agent_name", "audit_task_id"}}
        errs = sorted(v.iter_errors(r2), key=str)
        if errs:
            raise SystemExit(f"schema error on unmatched {row['trace_id']}: {errs[0].message}")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    labels_dir = STAGE / "data/raw/labels"
    extras_dir = labels_dir / "extras"
    extras_dir.mkdir(parents=True)

    # Main labels file goes in labels/ (picked up by prepare_parquet.py)
    with (labels_dir / "hal-usaco.jsonl").open("w") as f:
        for row in matched_cheating + benign:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Unmatched goes in extras/ — prepare_parquet.py only globs labels/*.jsonl
    # (not recursive), so this file lives alongside but doesn't break the build
    # since its synthetic trace_ids have no matching traces in any manifest.
    with (extras_dir / "hal-usaco.unmatched.jsonl").open("w") as f:
        for row in unmatched_cheating:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme = (STAGE / "data/raw/labels/HAL_USACO_NOTES.md")
    readme.write_text(f"""# HAL USACO labels for cheatbench

Built {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by Penn Agentic Lab.

Drop-in for `cheatbench/data/raw/labels/`.

## Counts

| file | rows | cheating | benign |
|---|---:|---:|---:|
| `hal-usaco.jsonl` | {len(matched_cheating) + len(benign)} | {len(matched_cheating)} | {len(benign)} |
| `extras/hal-usaco.unmatched.jsonl` | {len(unmatched_cheating)} | {len(unmatched_cheating)} | 0 |

## Provenance

Audit source: `data/safety/trace_audits/hal_usaco_multimodel_all/case_0000/index.jsonl`
(595 confirmed-cheating rows from 12 model variants in the HAL USACO archives).

Audit writeup: `{WRITEUP_PATH}`

## Mapping

The cheatbench `hal-usaco/manifest.json` covers 3 source_submissions:
- `claudeopus4120250514_high_1756132384`
- `gpt520250807_1756136988`
- `o4mini20250416_high_1746405235`

Of the 595 audit rows, **218** map to these submissions (mapping by
model_name + agent_name effort suffix; 218/218 round-tripped against
the manifest's trace_ids). The other **377** come from submissions not
in the manifest (claude-3-7-sonnet, deepseek-{{r1,v3}}, gemini-2.0-flash,
gpt-4.1, o3, o4-mini-low, claude-opus-4.1 without the `high` agent
suffix). Those ship in `hal-usaco.unmatched.jsonl` so you can pull them
in if you expand the manifest.

## Label policy

`hal-usaco.jsonl` is strict-audit:
- **cheating** (218): audit's exact-injection heuristic flagged this exact
  (submission, task) — the prompt contained the current benchmark problem
  with reference solution code.
- **benign** (703): manifest trace not in the audit-flagged set. Note that
  these traces used the *same* Episodic+Semantic scaffold; the audit just
  didn't see the exact-injection signature on them (likely because the
  retrieval pulled near/similar problems instead, or the run failed
  evaluation). If you want the broader scaffold-level interpretation
  ("anything from this scaffold is cheating"), promote these to cheating.

The `note`, `audit_*` fields on the unmatched rows are extras allowed by
`label.schema.json`'s `additionalProperties: true`.
""")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(STAGE)))

    import hashlib
    h = hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest()
    print(f"\nwrote {OUT_ZIP}  ({OUT_ZIP.stat().st_size/1024:.1f} KB)")
    print(f"  sha256={h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
