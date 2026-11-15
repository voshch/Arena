"""Device tracks fitted for humansim's agent types (arena_animation_retarget.devices): harmonics of the warped gait phase."""

from __future__ import annotations

import math

import numpy as np
import pytest

devices = pytest.importorskip("arena_simulation_setup.adapters.arena_animation_retarget.devices")


def _phase(samples: int = 360) -> np.ndarray:
    return np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)


def test_a_track_is_fitted_with_the_fewest_harmonics_that_reproduce_it() -> None:
    w = _phase()
    v = 0.3 + 0.05 * np.sin(w + 0.2) + 0.02 * np.sin(3 * w - 1.0)
    mean, harmonics, err = devices.fit_track(w, v)
    assert mean == pytest.approx(0.3, abs=1e-4)
    assert len(harmonics) == 3
    assert harmonics[0] == pytest.approx([0.05, 0.2], abs=1e-4)
    assert harmonics[1][0] == pytest.approx(0.0, abs=1e-4)
    assert harmonics[2] == pytest.approx([0.02, -1.0], abs=1e-4)
    assert err < devices.TOLERANCE
    assert float(np.max(np.abs(devices.evaluate(w, mean, harmonics) - v))) == pytest.approx(err)


def test_a_harmonic_within_the_tolerance_is_left_out() -> None:
    w = _phase()
    mean, harmonics, _ = devices.fit_track(w, 0.1 + 0.03 * np.sin(w) + 0.0004 * np.sin(5 * w))
    assert mean == pytest.approx(0.1, abs=1e-4)
    assert len(harmonics) == 1  # the 0.4 mm fifth harmonic is within the tolerance


def test_a_constant_track_is_a_plain_number() -> None:
    w = _phase(90)
    block, err = devices.track_block(w, np.stack([np.full_like(w, 0.36), 0.1 * np.sin(w), np.zeros_like(w)], axis=1))
    assert block == {"x": pytest.approx(0.36), "y": {"mean": 0.0, "harmonics": [[0.1, 0.0]]}, "z": 0.0}
    assert err < devices.TOLERANCE
