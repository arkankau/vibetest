from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _load_old_owners(subset: str, limit: int | None) -> list[str]:
    path = Path(f"results/kaggle_{subset}_AT-gpt-5-mini.jsonl")
    owners: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            repo = str(row.get("repo") or "")
            owner = Path(repo).name
            if owner and owner not in owners:
                owners.append(owner)
            if limit is not None and len(owners) >= limit:
                break
    return owners


def _load_repo_paths() -> list[Path]:
    data = json.loads(Path("viewer/repo-paths.json").read_text(encoding="utf-8"))
    paths: list[Path] = []
    for values in data.values():
        for value in values:
            paths.append(Path(value))
    return paths


def _refs_for_subset(subset: str, owners: list[str]) -> list[tuple[str, Path]]:
    owner_set = set(owners)
    refs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    prefix = Path("data") / "kaggle" / f"kaggle-{subset}"
    for path in _load_repo_paths():
        try:
            rel = path.relative_to(prefix)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2:
            continue
        owner, slug = parts[0], parts[1]
        if owner not in owner_set:
            continue
        ref = f"{owner}/{slug}"
        if ref in seen:
            continue
        seen.add(ref)
        refs.append((ref, prefix / owner / slug))
    return refs


def _pull(ref: str, dest: Path, *, force: bool) -> bool:
    if force and dest.exists():
        shutil.rmtree(dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {ref} -> {dest}")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[pull] {ref} -> {dest}")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "pull", ref, "-p", str(dest), "-m"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        print(f"[fail] {ref}\n{result.stdout}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["titanic", "diabetic", "nlp"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    owners = _load_old_owners(args.subset, args.limit)
    refs = _refs_for_subset(args.subset, owners)
    if not refs:
        raise SystemExit(f"No Kaggle refs found for subset={args.subset}, owners={owners[:5]}")
    print(f"Subset {args.subset}: {len(owners)} old owners, {len(refs)} kernel refs")

    ok = 0
    failed = 0
    for ref, dest in refs:
        if _pull(ref, dest, force=args.force):
            ok += 1
        else:
            failed += 1
        time.sleep(args.sleep)
    print(f"Done subset={args.subset}: ok={ok}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
