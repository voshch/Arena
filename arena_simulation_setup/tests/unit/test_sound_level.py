from __future__ import annotations

import numpy as np
import pytest

from arena_simulation_setup.tree.assets.sound_level import ACTIVE_LEVEL_GATE_DB, FULL_SCALE_SINE_RMS, active_rms, dbfs_from_rms, rms_from_dbfs

RATE = 16_000


def _tone(amplitude: float, seconds: float = 0.5) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * 440.0 * np.arange(round(seconds * RATE)) / RATE)


def test_full_scale_sine_reads_zero_dbfs_and_the_scale_inverts() -> None:
    assert dbfs_from_rms(float(np.sqrt(np.mean(_tone(1.0) ** 2)))) == pytest.approx(0.0, abs=1e-3)
    assert rms_from_dbfs(0.0) == FULL_SCALE_SINE_RMS
    for level in (-6.0, -26.5, -60.0):
        assert dbfs_from_rms(rms_from_dbfs(level)) == pytest.approx(level)
    assert dbfs_from_rms(0.0) == -120.0
    assert dbfs_from_rms(0.0, floor_db=-200.0) == -200.0


def test_active_level_ignores_surrounding_silence_and_quiet_windows() -> None:
    tone = _tone(0.5)
    padded = np.concatenate([np.zeros(RATE), tone, np.zeros(2 * RATE)])
    assert active_rms(padded, RATE) == active_rms(tone, RATE)
    assert active_rms(tone, RATE) == pytest.approx(0.5 * FULL_SCALE_SINE_RMS, rel=1e-3)
    below_gate = 0.5 * 10.0 ** ((ACTIVE_LEVEL_GATE_DB - 6.0) / 20.0)
    assert active_rms(np.concatenate([tone, _tone(below_gate)]), RATE) == pytest.approx(active_rms(tone, RATE), rel=1e-3)
    above_gate = 0.5 * 10.0 ** ((ACTIVE_LEVEL_GATE_DB + 6.0) / 20.0)
    assert active_rms(np.concatenate([tone, _tone(above_gate)]), RATE) < 0.9 * active_rms(tone, RATE)


def test_active_level_of_silence_is_zero_and_stereo_averages_the_channels() -> None:
    assert active_rms(np.zeros(RATE), RATE) == 0.0
    assert active_rms(np.zeros(0), RATE) == 0.0
    tone = _tone(0.5)
    assert active_rms(np.column_stack([tone, tone]), RATE) == pytest.approx(active_rms(tone, RATE))
