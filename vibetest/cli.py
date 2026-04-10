"""Command-line interface for the public Meerkat release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vibetest import TestCase, VibeTestAgent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meerkat: audit a repository of traces against a natural-language safety property.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the trace repository to audit.",
    )
    parser.add_argument(
        "--property",
        dest="property_text",
        type=str,
        help="Safety property to audit for.",
    )
    parser.add_argument(
        "--property-file",
        type=Path,
        help="Path to a text file containing the safety property.",
    )
    parser.add_argument(
        "--extra-instructions",
        type=str,
        default=None,
        help="Optional extra instructions appended to the audit prompt.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional audit name used in logs and result files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Inspect model identifier. Falls back to VIBETEST_MODEL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("meerkat_output/result.json"),
        help="Path to write the JSON result.",
    )
    parser.add_argument(
        "--sandbox",
        choices=["docker", "none"],
        default="docker",
        help="Sandbox mode for the audit run.",
    )
    parser.add_argument(
        "--analysis-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the embedding, clustering, and parallel scanner tools.",
    )
    parser.add_argument(
        "--search-aids",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to tell Meerkat to expect search_aids/ artifacts.",
    )
    return parser.parse_args()


def _load_property_text(args: argparse.Namespace) -> str:
    if bool(args.property_text) == bool(args.property_file):
        raise SystemExit("Specify exactly one of --property or --property-file.")
    if args.property_text:
        return args.property_text.strip()
    return args.property_file.read_text(encoding="utf-8").strip()


def _resolve_model(args: argparse.Namespace) -> str:
    model = args.model or os.getenv("VIBETEST_MODEL")
    if not model:
        raise SystemExit("Provide --model or set VIBETEST_MODEL.")
    return model


def _detect_search_aids(repo_path: Path) -> bool:
    search_aids_dir = repo_path / "search_aids"
    return any(
        path.exists()
        for path in (
            search_aids_dir / "initial_scores.tsv",
            search_aids_dir / "clusters.json",
        )
    )


def _resolve_search_aids_mode(args: argparse.Namespace) -> bool:
    if args.search_aids == "on":
        return True
    if args.search_aids == "off":
        return False
    return _detect_search_aids(args.repo)


def _build_test_case(args: argparse.Namespace, property_text: str) -> TestCase:
    return TestCase(
        name=args.name or f"audit-{args.repo.name}",
        description=property_text,
        extra_instructions=args.extra_instructions,
        repo_path=args.repo.resolve(),
    )


def _result_payload(result) -> dict:
    return result.to_dict()


def _print_summary(result, output_path: Path) -> None:
    metadata = result.metadata or {}
    verdict = metadata.get("verdict", "INCONCLUSIVE")
    print(f"Verdict: {verdict}")
    case_score = metadata.get("case_score")
    if case_score is not None:
        print(f"Case score: {case_score:.3f}")
    reason_text = (metadata.get("reason_text") or "").strip()
    if reason_text:
        print(f"Reason: {reason_text}")
    usage_totals = metadata.get("usage_totals") or {}
    total_tokens = usage_totals.get("total_tokens")
    if total_tokens:
        print(f"Total tokens: {total_tokens}")
    print(f"Result JSON: {output_path}")


def main() -> None:
    args = _parse_args()
    if not args.repo.is_dir():
        raise SystemExit(f"Repository not found: {args.repo}")

    property_text = _load_property_text(args)
    if not property_text:
        raise SystemExit("The safety property is empty.")

    model = _resolve_model(args)
    search_aids_enabled = _resolve_search_aids_mode(args)
    test_case = _build_test_case(args, property_text)

    agent = VibeTestAgent(
        model=model,
        safety_agent=True,
        safety_analysis_tools=args.analysis_tools,
        safety_repo_artifacts=search_aids_enabled,
    )
    sandbox = None if args.sandbox == "none" else args.sandbox

    try:
        results = agent.execute_tests([test_case], sandbox=sandbox)
    except KeyboardInterrupt:
        raise SystemExit(130) from None

    if not results:
        raise SystemExit("Meerkat did not return any results.")

    result = results[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_result_payload(result), indent=2, default=str),
        encoding="utf-8",
    )
    _print_summary(result, args.output)


if __name__ == "__main__":
    main()
