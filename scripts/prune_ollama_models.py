#!/usr/bin/env python3
"""
Remove Ollama models that are not in a strict keep-list.

Default keep-list:
  - llama3.2
  - llama3
  - deepseek-r1

Dry-run (default): print what would be deleted.
Apply:  python scripts/prune_ollama_models.py --apply

Requires: ollama CLI on PATH.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# Keep only these base families (any tag).
BASE_MODEL_STEMS = frozenset({"llama3", "deepseek-r1", "llama3.2"})


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ollama_list_model_names() -> list[str]:
    r = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    # First line is header like NAME ID SIZE MODIFIED
    names: list[str] = []
    for ln in lines[1:]:
        parts = ln.split()
        if parts:
            names.append(parts[0])
    return names


def stem(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run `ollama delete`. Without this, only print planned removals.",
    )
    args = parser.parse_args()

    keep_stems = set(BASE_MODEL_STEMS)
    installed = ollama_list_model_names()
    to_remove = [m for m in installed if stem(m) not in keep_stems]

    if not to_remove:
        print("Nothing to remove; all installed models are kept.")
        return

    print(f"Keeping stems: {sorted(keep_stems)}")
    print(f"Will remove {len(to_remove)} model(s):")
    for m in sorted(to_remove):
        print(f"  - {m}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to delete.")

        return

    for m in to_remove:
        print(f"Deleting {m} ...")
        subprocess.run(["ollama", "rm", m], check=False)


if __name__ == "__main__":
    main()
