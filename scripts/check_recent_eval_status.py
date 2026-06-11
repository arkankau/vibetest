from __future__ import annotations

import argparse
import time
from pathlib import Path

from inspect_ai.log import read_eval_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    logs = sorted(Path("logs").glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)[: args.limit]
    for path in logs:
        log = read_eval_log(str(path))
        samples = getattr(log, "samples", None) or []
        completed = sum(1 for sample in samples if getattr(sample, "output", None) is not None)
        errors = sum(1 for sample in samples if getattr(sample, "error", None) is not None)
        last_ids = [getattr(sample, "id", None) for sample in samples[-3:]]
        mtime = time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime))
        print(
            f"{path.name} mtime={mtime} status={getattr(log, 'status', None)} "
            f"completed={completed} errors={errors} last={last_ids}"
        )


if __name__ == "__main__":
    main()
