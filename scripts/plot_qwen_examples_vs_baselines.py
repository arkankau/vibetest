from __future__ import annotations

from pathlib import Path

from plot_synthetic_threshold_tradeoffs import (
    _inconclusive_as_pass,
    _load_items,
    _plot_metric_vs_coverage,
    _rows_for_group,
    _write_csv,
)


RESULT_DIR = Path("results/synthetic")
FIG_DIR = RESULT_DIR / "figures"
MODEL = "Qwen-Qwen3.6-35B-A3B-FP8"
SUFFIX = "_case_score_bands"
DATASETS = ("titanic", "diabetic", "nlp")


def example_paths(examples: int) -> list[Path]:
    return [
        RESULT_DIR / f"synthetic_kaggle_{dataset}_AT-{MODEL}-static_examples{examples}{SUFFIX}.jsonl"
        for dataset in DATASETS
    ]


def reviewer_paths(mode: int) -> list[Path]:
    return [
        RESULT_DIR
        / f"synthetic_kaggle_{dataset}_baseline-reviewer-mode{mode}-static-{MODEL}_mapper-gpt-5.4-mini.jsonl"
        for dataset in DATASETS
    ]


def traincheck_paths() -> list[Path]:
    return [
        RESULT_DIR / f"synthetic_kaggle_{dataset}_traincheck_mapper-{MODEL}.jsonl"
        for dataset in DATASETS
    ]


def output_tokens_per_property(paths: list[Path]) -> float | None:
    import json

    output_tokens = 0
    total_tests = 0
    saw_usage = False
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_tests += int(row.get("total_tests") or len(row.get("tests") or []))
                usage = row.get("usage") or {}
                totals = usage.get("usage_totals") or {}
                if totals.get("output_tokens") is not None:
                    saw_usage = True
                    output_tokens += int(totals.get("output_tokens") or 0)
                else:
                    for test in row.get("tests") or []:
                        md = test.get("metadata") or {}
                        tt = (md.get("usage_totals") or {}).get("output_tokens")
                        if tt is not None:
                            saw_usage = True
                            output_tokens += int(tt or 0)
    if not saw_usage or total_tests <= 0:
        return None
    return output_tokens / total_tests


def label_with_tokens(name: str, paths: list[Path]) -> str:
    out_per_prop = output_tokens_per_property(paths)
    if out_per_prop is None:
        return name
    return f"{name} ({out_per_prop / 1000:.2f}K out/prop)"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, list[Path], bool]] = []
    for examples in (0, 10, 20):
        paths = example_paths(examples)
        if examples == 0:
            name = "VibeTest static"
        else:
            name = f"VibeTest static + ex{examples}"
        specs.append((label_with_tokens(name, paths), paths, False))
    paths = traincheck_paths()
    specs.append((label_with_tokens("TrainCheck", paths), paths, True))
    for mode in (0, 1, 2):
        paths = reviewer_paths(mode)
        specs.append((label_with_tokens(f"Reviewer mode {mode}", paths), paths, True))

    missing = [str(path) for _, paths, _ in specs for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    curves = {}
    for label, paths, inconclusive_as_pass in specs:
        items = []
        for path in paths:
            items.extend(_load_items(path))
        if inconclusive_as_pass:
            items = _inconclusive_as_pass(items)
        curves[label] = _rows_for_group(
            items,
            target="gt-label",
            threshold_pass_low_score=False,
            inconclusive_band_low=None,
        )

    prefix = FIG_DIR / "synthetic_kaggle_qwen36_examples_vs_reviewers_traincheck_gt_label"
    _write_csv(prefix.with_suffix(".csv"), curves)
    _plot_metric_vs_coverage(
        curves,
        prefix.with_name(prefix.name + "_covered_macro_f1_vs_coverage.png"),
        metric="covered_macro_f1",
        ylabel="Macro F1",
        title="Synthetic Kaggle Qwen3.6: Selective F1 vs Coverage",
    )
    _plot_metric_vs_coverage(
        curves,
        prefix.with_name(prefix.name + "_macro_f1_vs_coverage.png"),
        metric="macro_f1",
        ylabel="Abstention-penalized macro F1",
        title="Synthetic Kaggle Qwen3.6: End-to-End F1 vs Coverage",
    )

    print(f"Wrote {prefix.with_suffix('.csv')}")
    print(f"Wrote {prefix.with_name(prefix.name + '_covered_macro_f1_vs_coverage.png')}")
    print(f"Wrote {prefix.with_name(prefix.name + '_macro_f1_vs_coverage.png')}")
    for label, rows in curves.items():
        valid = [row for row in rows if row["covered_macro_f1"] is not None]
        if not valid:
            print(f"{label}: no valid selective-F1 points")
            continue
        best = max(valid, key=lambda row: row["covered_macro_f1"])
        print(
            f"{label}: best selective_f1={best['covered_macro_f1']:.3f} "
            f"coverage={best['coverage']:.3f} threshold={best['threshold']:.3g}"
        )


if __name__ == "__main__":
    main()
