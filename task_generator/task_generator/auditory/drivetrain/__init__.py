"""Embedded source-only drivetrain synthesizer used by Arena playback."""

from .spec import DrivetrainSpec, JACKAL
from .voice import (
    OUTPUT_GAIN,
    DrivetrainVoice,
    TransferFilter,
    cache_bytes,
    clear_cache,
    design_transfer_fir,
    prewarm,
)

__all__ = [
    "DrivetrainSpec",
    "JACKAL",
    "OUTPUT_GAIN",
    "DrivetrainVoice",
    "TransferFilter",
    "cache_bytes",
    "clear_cache",
    "design_transfer_fir",
    "prewarm",
]
