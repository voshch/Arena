#!/usr/bin/env python3
"""Convert zone entity identifiers in a zones CSV to readable names."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_WORLDS_DIR = Path(__file__).resolve().parents[1] / "worlds"
DEFAULT_INPUT = DEFAULT_WORLDS_DIR / "zones.csv"
DEFAULT_OUTPUT = DEFAULT_WORLDS_DIR / "zones_readable.csv"
ENTITY_COLUMNS = ("zone_entity_names", "zone_entities_names")


def readable_name(identifier: str) -> str:
    """Turn an identifier such as ``waitingChair_01`` into ``waiting Chair``."""
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", identifier)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    value = re.sub(r"\d+", " ", value)
    value = re.sub(r"[_\-]+", " ", value)
    return " ".join(value.split())


def readable_entities(value: str) -> str:
    """Convert each entity in a semicolon-separated CSV field."""
    return ";".join(
        name
        for identifier in value.split(";")
        if (name := readable_name(identifier))
    )


def transform_csv(input_path: Path, output_path: Path) -> tuple[int, str]:
    """Transform the entity-name column and return the row count and column name."""
    with input_path.open(encoding="utf-8", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        if reader.fieldnames is None:
            raise ValueError("the input CSV has no header")

        entity_column = next(
            (column for column in ENTITY_COLUMNS if column in reader.fieldnames), None
        )
        if entity_column is None:
            expected = " or ".join(repr(column) for column in ENTITY_COLUMNS)
            raise ValueError(f"the input CSV must contain {expected}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with output_path.open("w", encoding="utf-8", newline="") as output_stream:
            writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                row[entity_column] = readable_entities(row.get(entity_column, ""))
                writer.writerow(row)
                row_count += 1

    return row_count, entity_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make zone entity identifiers readable and remove digits."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input CSV does not exist: {input_path}")
    if input_path == output_path:
        raise SystemExit("Input and output paths must be different")

    try:
        row_count, entity_column = transform_csv(input_path, output_path)
    except (OSError, csv.Error, ValueError) as error:
        raise SystemExit(f"Could not transform {input_path}: {error}") from error

    print(
        f"Converted {row_count} rows in '{entity_column}' and wrote {output_path}"
    )


if __name__ == "__main__":
    main()
