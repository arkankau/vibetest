"""Sample a fixed fraction of fail-annotation rows into a new CSV."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    n = math.ceil(len(rows) * args.fraction)
    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(rows)), n))
    sampled = [rows[i] for i in sampled_indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Input rows: {len(rows)}")
    print(f"Sampled rows: {len(sampled)} ({args.fraction:.1%}), seed={args.seed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
