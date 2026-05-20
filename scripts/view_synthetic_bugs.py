"""Inspect synthetic bug injections from labels JSONL files.

Examples:
  # List all injected samples across labels files
  uv run --active python scripts/view_synthetic_bugs.py --list

  # Show details for one sample in the filtered list
  uv run --active python scripts/view_synthetic_bugs.py --dataset cwe-bench --show 3

  # Export selected samples to markdown for slides/notes
  uv run --active python scripts/view_synthetic_bugs.py \
    --dataset bibifi \
    --status ok \
    --output-md /tmp/synthetic-bug-examples.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BugRecord:
    source_labels: Path
    row: dict[str, Any]
    output_sample_path: Path
    output_repo_path: Path
    diff_patch_path: Path | None
    changed_count: int
    fail_properties: list[dict[str, str]]
    injected_summary: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _discover_labels(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.glob("labels_*.jsonl") if p.is_file())


def _fail_properties(row: dict[str, Any]) -> list[dict[str, str]]:
    by_prop = row.get("ground_truth_by_property") or []
    out: list[dict[str, str]] = []
    for item in by_prop:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").strip().upper()
        label = int(item.get("label") or 0)
        if verdict != "FAIL" and label != 1:
            continue
        out.append(
            {
                "property_id": str(item.get("property_id") or "").strip(),
                "property_text": str(item.get("property_text") or "").strip(),
                "gt_violation_description": str(item.get("gt_violation_description") or "").strip(),
            }
        )
    return out


def _resolve_paths(row: dict[str, Any]) -> tuple[Path, Path, Path | None]:
    repo_path = Path(str(row.get("output_repo_path") or "").strip())
    sample_path_raw = str(row.get("output_sample_path") or "").strip()
    sample_path = Path(sample_path_raw) if sample_path_raw else repo_path.parent

    diff_path: Path | None = None
    repo_diff = row.get("repo_diff") or {}
    diff_hint = str(repo_diff.get("diff_patch_path") or "").strip()
    if diff_hint:
        hint_path = Path(diff_hint)
        if hint_path.exists():
            diff_path = hint_path
    if diff_path is None:
        candidate = sample_path / "injection.diff.patch"
        if candidate.exists():
            diff_path = candidate
    return sample_path, repo_path, diff_path


def _build_records(labels_files: list[Path]) -> list[BugRecord]:
    records: list[BugRecord] = []
    for labels_file in labels_files:
        for row in _load_jsonl(labels_file):
            sample_path, repo_path, diff_path = _resolve_paths(row)
            repo_diff = row.get("repo_diff") or {}
            changed_count = int(repo_diff.get("changed_count") or 0)
            report = ((row.get("artifact_extraction") or {}).get("injection_report") or {})
            summary = str(report.get("injection_summary") or "").strip()

            records.append(
                BugRecord(
                    source_labels=labels_file,
                    row=row,
                    output_sample_path=sample_path,
                    output_repo_path=repo_path,
                    diff_patch_path=diff_path,
                    changed_count=changed_count,
                    fail_properties=_fail_properties(row),
                    injected_summary=summary,
                )
            )
    return records


def _matches(record: BugRecord, args: argparse.Namespace) -> bool:
    row = record.row
    if args.status and str(row.get("status") or "").strip() != args.status:
        return False
    if args.domain and str(row.get("domain") or "").strip() != args.domain:
        return False
    if args.dataset and str(row.get("dataset") or "").strip() != args.dataset:
        return False
    if args.repo and args.repo not in str(row.get("repo_name") or "") and args.repo not in str(row.get("repo_slug") or ""):
        return False
    if args.row_index is not None and int(row.get("row_index") or -1) != args.row_index:
        return False
    if args.only_failed and not record.fail_properties:
        return False
    return True


def _truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: max(0, n - 3)] + "..."


def _print_list(records: list[BugRecord], max_summary: int) -> None:
    headers = ["#", "domain", "dataset", "row", "repo", "status", "fail_props", "changed", "summary"]
    rows: list[list[str]] = []
    for idx, rec in enumerate(records):
        row = rec.row
        rows.append(
            [
                str(idx),
                str(row.get("domain") or ""),
                str(row.get("dataset") or ""),
                str(row.get("row_index") or ""),
                str(row.get("repo_name") or row.get("repo_slug") or ""),
                str(row.get("status") or ""),
                str(len(rec.fail_properties)),
                str(rec.changed_count),
                _truncate(rec.injected_summary, max_summary),
            ]
        )

    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt(r: list[str]) -> str:
        return "  ".join(r[i].ljust(widths[i]) for i in range(len(headers)))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def _read_diff(path: Path | None, max_lines: int) -> tuple[str, bool]:
    if path is None or not path.exists():
        return "(diff file not found)", False
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    truncated = max_lines > 0 and len(lines) > max_lines
    if truncated:
        lines = lines[:max_lines]
    return "\n".join(lines), truncated


def _print_one(rec: BugRecord, show_diff: bool, max_diff_lines: int) -> None:
    row = rec.row
    print(f"labels_file: {rec.source_labels}")
    print(f"domain/dataset: {row.get('domain')}/{row.get('dataset')}")
    print(f"row_index: {row.get('row_index')}  status: {row.get('status')}")
    print(f"repo_name: {row.get('repo_name')}  repo_slug: {row.get('repo_slug')}")
    print(f"source_repo_path: {row.get('source_repo_path')}")
    print(f"output_repo_path: {rec.output_repo_path}")
    print(f"output_sample_path: {rec.output_sample_path}")
    print(f"diff_patch_path: {rec.diff_patch_path}")
    print(f"selected_property_ids: {row.get('selected_property_ids')}")
    print("")

    report = ((row.get("artifact_extraction") or {}).get("injection_report") or {})
    summary = str(report.get("injection_summary") or "").strip()
    print("Injected bug summary:")
    print(summary or "(none)")
    print("")

    bug_locations = report.get("bug_locations") or []
    if bug_locations:
        print("Bug locations:")
        for loc in bug_locations:
            if not isinstance(loc, dict):
                continue
            print(f"- {loc.get('file')}: {loc.get('hint')}")
        print("")

    print(f"Fail properties ({len(rec.fail_properties)}):")
    if not rec.fail_properties:
        print("- (none)")
    for fp in rec.fail_properties:
        print(f"- {fp['property_id']}: {fp['property_text']}")
        if fp["gt_violation_description"]:
            print(f"  gt_violation_description: {fp['gt_violation_description']}")
    print("")

    if show_diff:
        diff_text, truncated = _read_diff(rec.diff_patch_path, max_lines=max_diff_lines)
        print("Diff:")
        print(diff_text)
        if truncated:
            print(f"\n[diff truncated to first {max_diff_lines} lines]")


def _record_to_markdown(rec: BugRecord, *, max_diff_lines: int) -> str:
    row = rec.row
    report = ((row.get("artifact_extraction") or {}).get("injection_report") or {})
    lines: list[str] = []
    lines.append(f"## {row.get('domain')}/{row.get('dataset')} :: row {row.get('row_index')} :: {row.get('repo_name')}")
    lines.append("")
    lines.append(f"- `status`: `{row.get('status')}`")
    lines.append(f"- `source_repo_path`: `{row.get('source_repo_path')}`")
    lines.append(f"- `output_repo_path`: `{rec.output_repo_path}`")
    lines.append(f"- `diff_patch_path`: `{rec.diff_patch_path}`")
    lines.append(f"- `selected_property_ids`: `{row.get('selected_property_ids')}`")
    lines.append("")
    lines.append("### Injected Bug Summary")
    lines.append(report.get("injection_summary") or "(none)")
    lines.append("")
    lines.append("### Fail Properties")
    if rec.fail_properties:
        for fp in rec.fail_properties:
            lines.append(f"- `{fp['property_id']}`: {fp['property_text']}")
            if fp["gt_violation_description"]:
                lines.append(f"  - GT description: {fp['gt_violation_description']}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### Diff")
    diff_text, truncated = _read_diff(rec.diff_patch_path, max_lines=max_diff_lines)
    lines.append("```diff")
    lines.append(diff_text)
    lines.append("```")
    if truncated:
        lines.append(f"_Diff truncated to first {max_diff_lines} lines._")
    lines.append("")
    return "\n".join(lines)


def _write_markdown(path: Path, records: list[BugRecord], *, max_diff_lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        "# Synthetic Injection Inspection",
        "",
        f"Total samples: {len(records)}",
        "",
    ]
    for rec in records:
        chunks.append(_record_to_markdown(rec, max_diff_lines=max_diff_lines))
    path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        action="append",
        default=[],
        help="Path to labels_*.jsonl. Can be repeated. If omitted, auto-discovers in synth-data/injected.",
    )
    parser.add_argument("--labels-root", default="synth-data/injected", help="Root for auto-discovery.")
    parser.add_argument("--status", default="ok", help="Filter by status (default: ok). Use empty string for all.")
    parser.add_argument("--domain", help="Filter by domain.")
    parser.add_argument("--dataset", help="Filter by dataset.")
    parser.add_argument("--repo", help="Substring filter on repo_name/repo_slug.")
    parser.add_argument("--row-index", type=int, help="Filter to a specific row index.")
    parser.add_argument("--only-failed", action="store_true", help="Show only samples with at least one FAIL property.")
    parser.add_argument("--list", action="store_true", help="Print compact list table.")
    parser.add_argument("--show", type=int, help="Show detailed record by index in the filtered list.")
    parser.add_argument("--no-diff", action="store_true", help="Do not print diff in detailed view.")
    parser.add_argument("--max-diff-lines", type=int, default=250, help="Max diff lines for detailed/markdown output.")
    parser.add_argument("--max-summary-chars", type=int, default=80, help="Max chars for list summary cell.")
    parser.add_argument("--output-md", help="Optional markdown output path for filtered records.")
    args = parser.parse_args()

    explicit = [Path(p) for p in args.labels if str(p).strip()]
    if explicit:
        labels_files = [p for p in explicit if p.exists()]
    else:
        labels_files = _discover_labels(Path(args.labels_root))

    if not labels_files:
        raise SystemExit("No labels files found.")

    records = _build_records(labels_files)
    if args.status == "":
        status_filter = None
    else:
        status_filter = args.status
    args.status = status_filter
    filtered = [r for r in records if _matches(r, args)]

    if not filtered:
        raise SystemExit("No records matched filters.")

    if args.list or args.show is None:
        _print_list(filtered, max_summary=args.max_summary_chars)

    if args.show is not None:
        idx = args.show
        if idx < 0 or idx >= len(filtered):
            raise SystemExit(f"--show index out of range: {idx} (0..{len(filtered)-1})")
        print("")
        _print_one(filtered[idx], show_diff=not args.no_diff, max_diff_lines=args.max_diff_lines)

    if args.output_md:
        out = Path(args.output_md)
        _write_markdown(out, filtered, max_diff_lines=args.max_diff_lines)
        print(f"\nWrote markdown: {out}")


if __name__ == "__main__":
    main()
