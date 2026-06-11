from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import jsonlines


RESULTS = [
    Path("results/kaggle_titanic_AT-gpt-5-mini-static_examples0.jsonl"),
    Path("results/kaggle_diabetic_AT-gpt-5-mini-static_examples0.jsonl"),
    Path("results/kaggle_nlp_AT-gpt-5-mini-static_examples0.jsonl"),
    Path("results/kaggle_titanic_AT-gpt-5-mini-static_examples10.jsonl"),
    Path("results/kaggle_diabetic_AT-gpt-5-mini-static_examples10.jsonl"),
]


def verdict(test: dict) -> str:
    meta = test.get("metadata") or {}
    value = str(meta.get("verdict") or "").strip().upper()
    if value:
        return value
    desc = str(test.get("description") or "")
    if test.get("passed") is True:
        return "PASS"
    if "INCONCLUSIVE" in desc.upper():
        return "INCONCLUSIVE"
    if "NOT APPLICABLE" in desc.upper():
        return "NOT APPLICABLE"
    return "FAIL"


def load_verdicts(path: Path) -> dict[tuple[str, int], str]:
    out = {}
    with jsonlines.open(str(path)) as reader:
        for row in reader:
            repo = str(row.get("repo_name") or row.get("repo") or "")
            for idx, test in enumerate(row.get("tests") or []):
                out[(repo, idx)] = verdict(test)
    return out


def summarize(path: Path) -> dict:
    rows = list(jsonlines.open(str(path)))
    counts: Counter[str] = Counter()
    scores = []
    evidence_strengths = []
    empty = 0
    prompt_echo = 0
    total = 0

    for row in rows:
        for test in row.get("tests") or []:
            total += 1
            counts[verdict(test)] += 1
            desc = str(test.get("description") or "")
            meta = test.get("metadata") or {}
            if not desc.strip():
                empty += 1
            if "Calibration notes from a manual audit" in desc:
                prompt_echo += 1
            if isinstance(meta.get("case_score"), (int, float)):
                scores.append(float(meta["case_score"]))
            if isinstance(meta.get("evidence_strength"), (int, float)):
                evidence_strengths.append(float(meta["evidence_strength"]))

    coverage = (total - counts["INCONCLUSIVE"]) / total if total else 0.0
    avg_score = sum(scores) / len(scores) if scores else None
    avg_evidence = sum(evidence_strengths) / len(evidence_strengths) if evidence_strengths else None
    return {
        "repos": len(rows),
        "tests": total,
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "inconclusive": counts["INCONCLUSIVE"],
        "not_applicable": counts["NOT APPLICABLE"],
        "coverage": coverage,
        "avg_case_score": avg_score,
        "avg_evidence_strength": avg_evidence,
        "empty": empty,
        "prompt_echo": prompt_echo,
    }


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def main() -> None:
    pattern = re.compile(r"kaggle_(.*?)_AT-gpt-5-mini-static_examples(\d+)")
    print("Completed full result files")
    for path in RESULTS:
        match = pattern.search(path.name)
        if not match or not path.exists():
            continue
        subset, examples = match.group(1), int(match.group(2))
        s = summarize(path)
        print(
            f"{subset:8s} examples={examples:<2d} repos={s['repos']:<2d} tests={s['tests']:<4d} "
            f"PASS={s['pass']:<4d} FAIL={s['fail']:<4d} INCONCLUSIVE={s['inconclusive']:<4d} "
            f"coverage={s['coverage']:.3f} avg_case_score={fmt(s['avg_case_score'])} "
            f"avg_evidence_strength={fmt(s['avg_evidence_strength'])} empty={s['empty']} echo={s['prompt_echo']}"
        )

    print("\nPaired shifts from examples=0 to examples=10")
    for subset in ("titanic", "diabetic"):
        base = Path(f"results/kaggle_{subset}_AT-gpt-5-mini-static_examples0.jsonl")
        ex10 = Path(f"results/kaggle_{subset}_AT-gpt-5-mini-static_examples10.jsonl")
        if not base.exists() or not ex10.exists():
            continue
        a = load_verdicts(base)
        b = load_verdicts(ex10)
        keys = sorted(set(a) & set(b))
        transitions = Counter((a[k], b[k]) for k in keys)
        changed = sum(1 for k in keys if a[k] != b[k])
        print(f"{subset}: paired={len(keys)} changed={changed} ({changed / len(keys):.1%})")
        for (before, after), count in transitions.most_common():
            if before != after:
                print(f"  {before:14s} -> {after:14s} {count}")


if __name__ == "__main__":
    main()
