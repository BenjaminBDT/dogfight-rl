#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune dense checkpoint directories conservatively.")
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--keep-step",
        dest="keep_steps",
        action="append",
        type=int,
        default=[],
        help="Exact numeric step to keep. Can be passed multiple times.",
    )
    parser.add_argument(
        "--keep-name",
        dest="keep_names",
        action="append",
        default=[],
        help="Exact filename to keep. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print files that would be deleted.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"missing checkpoint dir: {checkpoint_dir}")

    keep_names = set(args.keep_names)
    keep_names.add("latest.pt")
    keep_steps = set(args.keep_steps)

    deleted = 0
    kept = 0
    for path in sorted(checkpoint_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name in keep_names:
            kept += 1
            continue
        match = _STEP_RE.match(name)
        if match is None:
            kept += 1
            continue
        if int(match.group(1)) in keep_steps:
            kept += 1
            continue
        print(f"delete {path}")
        if not args.dry_run:
            path.unlink()
        deleted += 1

    print(
        f"checkpoint_dir={checkpoint_dir} kept={kept} deleted={deleted} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
