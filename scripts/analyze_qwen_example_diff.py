from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


RESULT_DIR = Path("results/synthetic")
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
SUFFIX = "_case_score_bands"


def load_tests(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for repo_idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            for test_idx, test in enumerate(row.get("tests") or []):
                meta = test.get("metadata") or {}
                out[(repo_idx, test_idx)] = {
                    "verdict": str(meta.get("verdict") or "UNKNOWN").upper(),
                    "case_score": meta.get("case_score"),
                    "evidence_strength": meta.get("evidence_strength"),
                    "reason_text": meta.get("reason_text") or "",
                }
    return out


def summarize(items: dict[tuple[int, int], dict]) -> dict:
    counts = Counter(item["verdict"] for item in items.values())
    scores = [float(item["case_score"]) for item in items.values() if isinstance(item.get("case_score"), (int, float))]
    evs = [
        float(item["evidence_strength"])
        for item in items.values()
        if isinstance(item.get("evidence_strength"), (int, float))
    ]
    total = len(items)
    by_verdict = defaultdict(list)
    for item in items.values():
        if isinstance(item.get("case_score"), (int, float)):
            by_verdict[item["verdict"]].append(float(item["case_score"]))
    return {
        "total": total,
        "counts": counts,
        "coverage": (total - counts["INCONCLUSIVE"]) / total if total else 0.0,
        "avg_score": statistics.mean(scores) if scores else None,
        "avg_evidence_strength": statistics.mean(evs) if evs else None,
        "missing_score": total - len(scores),
        "empty_reason": sum(1 for item in items.values() if not item["reason_text"].strip()),
        "by_verdict_avg": {
            verdict: statistics.mean(vals) for verdict, vals in by_verdict.items() if vals
        },
    }


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def main() -> None:
    for dataset in ("titanic", "diabetic", "nlp"):
        p10 = RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples10{SUFFIX}.jsonl"
        p20 = RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples20{SUFFIX}.jsonl"
        if not p10.exists() or not p20.exists():
            print(f"\n{dataset}: missing files")
            print(f"  ex10 exists={p10.exists()} ex20 exists={p20.exists()}")
            continue

        t10 = load_tests(p10)
        t20 = load_tests(p20)
        s10 = summarize(t10)
        s20 = summarize(t20)
        common = sorted(set(t10) & set(t20))
        transitions = Counter((t10[key]["verdict"], t20[key]["verdict"]) for key in common)
        changed = sum(count for (a, b), count in transitions.items() if a != b)
        score_deltas = [
            float(t20[key]["case_score"]) - float(t10[key]["case_score"])
            for key in common
            if isinstance(t10[key].get("case_score"), (int, float))
            and isinstance(t20[key].get("case_score"), (int, float))
        ]

        print(f"\n{dataset}")
        for label, summary in (("ex10", s10), ("ex20", s20)):
            counts = summary["counts"]
            print(
                f"  {label}: coverage={summary['coverage']:.3f} "
                f"PASS={counts['PASS']} FAIL={counts['FAIL']} INC={counts['INCONCLUSIVE']} "
                f"avg_score={fmt(summary['avg_score'])} "
                f"avg_ev={fmt(summary['avg_evidence_strength'])} "
                f"missing_score={summary['missing_score']} empty_reason={summary['empty_reason']}"
            )
        print(
            "  delta: "
            f"coverage={s20['coverage'] - s10['coverage']:+.3f} "
            f"PASS={s20['counts']['PASS'] - s10['counts']['PASS']:+d} "
            f"FAIL={s20['counts']['FAIL'] - s10['counts']['FAIL']:+d} "
            f"INC={s20['counts']['INCONCLUSIVE'] - s10['counts']['INCONCLUSIVE']:+d} "
            f"avg_score={((s20['avg_score'] or 0) - (s10['avg_score'] or 0)):+.3f}"
        )
        print(f"  paired_changed={changed}/{len(common)} ({changed / len(common):.1%})")
        if score_deltas:
            print(
                f"  score_delta mean={statistics.mean(score_deltas):+.3f} "
                f"median={statistics.median(score_deltas):+.3f} "
                f"min={min(score_deltas):+.2f} max={max(score_deltas):+.2f}"
            )
        print("  top transitions:")
        for (src, dst), count in transitions.most_common(8):
            if src != dst:
                print(f"    {src}->{dst}: {count}")


if __name__ == "__main__":
    main()
