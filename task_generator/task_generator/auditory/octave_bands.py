from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

DEFAULT_OCTAVE_CENTERS_HZ = (63, 125, 250, 500, 1000, 2000,4000, 8000)


def calculate_octave_band_levels_db(samples: np.ndarray, sample_rate: int, *, centers_hz: tuple[int, ...] = DEFAULT_OCTAVE_CENTERS_HZ, floor_db: float = -120.0) -> dict[int, float]:
    """Return relative octave-band powers whose linear powers sum to one."""

    if samples.size == 0:
        return {}

    if samples.ndim == 2:
        mono = np.mean(samples, axis=1, dtype=np.float64)
    else:
        mono = samples.astype(np.float64, copy=False)

    mono = mono - np.mean(mono)

    if not np.any(mono):
        return {center: floor_db for center in centers_hz}

    nperseg = min(4096, len(mono))
    if nperseg < 32:
        return {}

    frequencies, psd = welch(mono,fs=sample_rate, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant", scaling="density")
    nyquist = sample_rate / 2.0
    band_powers: dict[int, float] = {}

    for center in centers_hz:
        lower = center / math.sqrt(2.0)
        upper = center * math.sqrt(2.0)

        if lower >= nyquist:
            continue

        upper = min(upper, nyquist)
        mask = (frequencies >= lower) & (frequencies < upper)

        if np.count_nonzero(mask) < 2:
            band_powers[center] = 0.0
            continue

        band_powers[center] = float(trapezoid(psd[mask], frequencies[mask]))

    total_power = sum(band_powers.values())
    if total_power <= 0.0:
        return {center: floor_db for center in band_powers}

    minimum_ratio = 10.0 ** (floor_db / 10.0)

    return {
        center: float(10.0 * math.log10(max(power / total_power, minimum_ratio)))
        for center, power in band_powers.items()
    }
