#!/usr/bin/env python3
"""Generate spoken navigation requests from zones_readable.csv with Piper."""

from __future__ import annotations

import argparse
import csv
import random
import re
import wave
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "worlds" / "zones_readable.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "task_generator" / "sounds" / "tts_generated_requests"
REQUESTS = ("Move", "Go", "Navigate")
VOICES = ("en_US-hfc_female-medium", "en_US-bryce-medium")
ENTITY_COLUMNS = ("zone_entity_names", "zone_entities_names")


def safe_filename(value: str) -> str:
    """Return a portable lowercase filename component."""
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value or "unnamed"


def spoken_name(value: str) -> str:
    """Make a CSV identifier natural to pronounce."""
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    value = re.sub(r"\d+", " ", value)
    value = re.sub(r"[_\-]+", " ", value)
    return " ".join(value.split())


def load_requests(
    csv_path: Path, rng: random.Random, voice_name: str
) -> list[tuple[str, str]]:
    """Return ``(filename, spoken request)`` pairs for zones with entities."""
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("the input CSV has no header")

        entity_column = next(
            (column for column in ENTITY_COLUMNS if column in reader.fieldnames), None
        )
        if entity_column is None:
            expected = " or ".join(repr(column) for column in ENTITY_COLUMNS)
            raise ValueError(f"the input CSV must contain {expected}")
        if "zone_name" not in reader.fieldnames:
            raise ValueError("the input CSV must contain 'zone_name'")

        requests: list[tuple[str, str]] = []
        used_filenames: dict[str, int] = {}
        for row in reader:
            zone_identifier = (row.get("zone_name") or "").strip()
            zone_name = spoken_name(zone_identifier)
            entity_name = (row.get(entity_column) or "").split(";", 1)[0].strip()
            if not zone_name or not entity_name:
                continue

            spoken_text = f"Hey, assistant. {rng.choice(REQUESTS)} to {entity_name} in {zone_name}"
            base = "_".join(
                (
                    safe_filename(zone_identifier),
                    safe_filename(entity_name),
                    voice_name,
                )
            )
            occurrence = used_filenames.get(base, 0) + 1
            used_filenames[base] = occurrence
            filename = f"{base}.wav" if occurrence == 1 else f"{base}_{occurrence}.wav"
            requests.append((filename, spoken_text))

    return requests


def load_piper_voice(model_path: Path, use_cuda: bool) -> Any:
    """Load Piper lazily so CSV errors do not require Piper to be installed."""
    try:
        from piper import PiperVoice
    except ImportError as error:
        raise RuntimeError(
            "piper-tts is not installed; install it with 'python3 -m pip install piper-tts'"
        ) from error

    return PiperVoice.load(str(model_path), use_cuda=use_cuda)


def synthesize_requests(
    voice: Any, requests: list[tuple[str, str]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, spoken_text in requests:
        output_path = output_dir / filename
        with wave.open(str(output_path), "wb") as wav_file:
            voice.synthesize_wav(spoken_text, wav_file)
        print(f"{output_path.name}: {spoken_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Piper TTS navigation request per zone with an entity."
    )
    parser.add_argument(
        "--voice",
        required=True,
        choices=VOICES,
        help="Piper voice to use",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="directory containing the downloaded Piper .onnx model and JSON config",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"readable zones CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"WAV output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="optional random seed for repeatable request choices",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="use Piper's CUDA execution provider",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    model_path = (args.data_dir / f"{args.voice}.onnx").resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input CSV does not exist: {input_path}")
    if not model_path.is_file():
        raise SystemExit(f"Piper model does not exist: {model_path}")

    try:
        requests = load_requests(input_path, random.Random(args.seed), args.voice)
        voice = load_piper_voice(model_path, args.use_cuda)
        synthesize_requests(voice, requests, output_dir)
    except (OSError, csv.Error, ValueError, RuntimeError) as error:
        raise SystemExit(f"Could not generate TTS requests: {error}") from error

    print(f"Generated {len(requests)} WAV files in {output_dir}")


if __name__ == "__main__":
    main()
