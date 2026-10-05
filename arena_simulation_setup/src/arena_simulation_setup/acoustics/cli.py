"""CLI for the fixed acoustics scenario generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .scenario_generator import plan, write_scenarios


def _default_worlds_root() -> Path:
    source_root = Path(__file__).resolve().parents[3] / "acoustics" / "worlds"
    if source_root.is_dir():
        return source_root
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("arena_simulation_setup")) / "worlds"
    except (ImportError, LookupError):
        return source_root


DEFAULT_WORLDS_ROOT = _default_worlds_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_acoustics_scenarios",
        description=("Generate exactly 12 scenarios per world: idle/moving robot, 1/2/3 pedestrians, and both robot/pedestrian endpoint assignments."),
    )
    parser.add_argument(
        "--worlds-root",
        type=Path,
        default=DEFAULT_WORLDS_ROOT,
        help=f"acoustics worlds directory (default: {DEFAULT_WORLDS_ROOT})",
    )
    parser.add_argument(
        "--world",
        action="append",
        default=[],
        metavar="NAME_OR_GLOB",
        help="generate only matching worlds; repeatable (does not add a scenario variant)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write scenario.yaml files; without this flag the command is a dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = plan(args.worlds_root, args.world)
        written, unchanged = write_scenarios(scenarios) if args.write else (0, 0)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    worlds = sorted({scenario.world_dir.name for scenario in scenarios})
    print(
        json.dumps(
            {
                "mode": "write" if args.write else "dry-run",
                "world_count": len(worlds),
                "scenario_count": len(scenarios),
                "scenarios_per_world": 12,
                "worlds": worlds,
                "written": written,
                "unchanged": unchanged,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
