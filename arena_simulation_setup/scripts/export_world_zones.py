#!/usr/bin/env python3
"""Export zone metadata from Arena world.yaml files to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterator

import yaml


DEFAULT_WORLDS_DIR = Path(__file__).resolve().parents[1] / "worlds"
DEFAULT_OUTPUT = DEFAULT_WORLDS_DIR / "zones.csv"
FIELDNAMES = (
    "world",
    "level",
    "zone_name",
    "zone_description",
    "zone_entity_names",
)


def entity_names(entities: Any) -> list[str]:
    """Return entity names from every entity group in a zone."""
    if not isinstance(entities, dict):
        return []

    names: list[str] = []
    for group in entities.values():
        if not isinstance(group, list):
            continue
        for entity in group:
            if isinstance(entity, dict) and entity.get("name") is not None:
                names.append(str(entity["name"]))
    return names


def rows_from_world(world_file: Path, worlds_dir: Path) -> Iterator[dict[str, str]]:
    """Yield CSV rows from one world.yaml file."""
    with world_file.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}

    if not isinstance(document, dict):
        raise ValueError("the YAML root must be a mapping")

    zones = document.get("zones", [])
    if not isinstance(zones, list):
        raise ValueError("'zones' must be a list")

    relative = world_file.relative_to(worlds_dir)
    world = relative.parts[0] if len(relative.parts) >= 2 else ""
    level = relative.parts[-2] if len(relative.parts) >= 2 else ""

    for zone in zones:
        if not isinstance(zone, dict):
            continue
        yield {
            "world": world,
            "level": level,
            "zone_name": str(zone.get("name") or ""),
            "zone_description": str(zone.get("description") or ""),
            "zone_entity_names": ";".join(entity_names(zone.get("entities"))),
        }


def export_zones(worlds_dir: Path, output: Path) -> tuple[int, int]:
    """Write all discovered zones and return (world file count, zone count)."""
    world_files = sorted(worlds_dir.glob("*/*/world.yaml"))
    output.parent.mkdir(parents=True, exist_ok=True)

    zone_count = 0
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        for world_file in world_files:
            try:
                rows = rows_from_world(world_file, worlds_dir)
                for row in rows:
                    writer.writerow(row)
                    zone_count += 1
            except (OSError, yaml.YAMLError, ValueError) as error:
                raise RuntimeError(f"Could not parse {world_file}: {error}") from error

    return len(world_files), zone_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export names, descriptions, and entity names for every zone."
    )
    parser.add_argument(
        "--worlds-dir",
        type=Path,
        default=DEFAULT_WORLDS_DIR,
        help=f"worlds directory (default: {DEFAULT_WORLDS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output CSV path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worlds_dir = args.worlds_dir.resolve()
    output = args.output.resolve()

    if not worlds_dir.is_dir():
        raise SystemExit(f"Worlds directory does not exist: {worlds_dir}")

    world_count, zone_count = export_zones(worlds_dir, output)
    print(f"Exported {zone_count} zones from {world_count} files to {output}")


if __name__ == "__main__":
    main()
