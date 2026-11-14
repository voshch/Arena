"""The command line every study shares: a manifest, `--out`, `--limit` and `--write`, the rejections and a summary, then the index."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import Item, Report, write_index


def parser(prog: str, description: str | None, limit: str) -> argparse.ArgumentParser:
    """The study parser, `limit` saying what `--limit N` keeps."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="directory to write the worlds and the index into, default the writable world root")
    parser.add_argument("--limit", type=int, default=0, help=limit)
    parser.add_argument("--write", action="store_true", help="write the worlds, a dry run without it")
    return parser


def summarize[I: Item](report: Report[I], noun: str) -> None:
    """Print every rejected item with its reason, then how many `noun` were placed in how many worlds."""
    for outcome in report.outcomes:
        if outcome.reason is not None:
            print(f"rejected {outcome.item.world_key}/{outcome.item.scenario_key}: {outcome.reason}")
    print(f"{report.placed} of {len(report.outcomes)} {noun} placed in {len(report.placed_worlds())} worlds")


def finish[I: Item](report: Report[I], name: str, out: Path | None, write: bool) -> int:
    """Write the index with `write`, and exit 1 when nothing was placed."""
    if write:
        index = write_index(report, name, out)
        if index is not None:
            print(f"index written to {index}")
    else:
        print("dry run, pass --write to save")
    return 0 if report.placed else 1
