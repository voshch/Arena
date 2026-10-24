"""Digital sound levels: the dBFS scale and the active level a manifest's normalize_dbfs refers to."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FULL_SCALE_SINE_RMS = 1.0 / math.sqrt(2.0)
ACTIVE_LEVEL_WINDOW_S = 0.01
ACTIVE_LEVEL_GATE_DB = -20.0


def dbfs_from_rms(value: float, *, floor_db: float = -120.0) -> float:
    """Level in dBFS where a full-scale sine reads 0 dBFS."""
    return max(20.0 * math.log10(max(float(value), 1e-12) / FULL_SCALE_SINE_RMS), floor_db)


def rms_from_dbfs(level_dbfs: float) -> float:
    """RMS of a sine at level_dbfs."""
    return FULL_SCALE_SINE_RMS * 10.0 ** (float(level_dbfs) / 20.0)


def active_rms(samples: NDArray[np.floating], sample_rate: int) -> float:
    """RMS of the 10 ms windows within 20 dB of the loudest, framed from the first non-silent frame."""
    audio = np.asarray(samples, dtype=np.float64)
    power = audio * audio if audio.ndim == 1 else np.mean(audio * audio, axis=1)
    sounding = np.flatnonzero(power)
    if sounding.size == 0:
        return 0.0
    power = power[sounding[0] : sounding[-1] + 1]
    window = max(round(sample_rate * ACTIVE_LEVEL_WINDOW_S), 1)
    starts = np.arange(0, power.size, window)
    sums = np.add.reduceat(power, starts)
    lengths = np.diff(np.append(starts, power.size))
    gated = sums / lengths >= np.max(sums / lengths) * 10.0 ** (ACTIVE_LEVEL_GATE_DB / 10.0)
    return float(np.sqrt(np.sum(sums[gated]) / np.sum(lengths[gated])))
