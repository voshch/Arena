from __future__ import annotations

import math

import numpy as np
import pytest
from task_generator.auditory.spatial_audio import (
    CHANNEL_NAMES,
    apply_monitor_controls,
    calibrate_mems,
    fractional_delay,
    gcc_phat,
    geometric_delays_seconds,
    headphone_stereo,
    hearing_waveform,
    monitor_amplify,
    rectangular_array,
    streaming_fractional_delays,
    transform_array,
)


def _delays(source: tuple[float, float, float]) -> np.ndarray:
    microphones = rectangular_array()
    return geometric_delays_seconds(
        source,
        tuple(mic.position_m for mic in microphones),
    )


def test_four_mic_geometry_matches_jackal_chassis_inset() -> None:
    microphones = rectangular_array()
    assert tuple(mic.name for mic in microphones) == CHANNEL_NAMES
    assert microphones[0].position_m == (0.19, 0.135, 0.22)
    assert microphones[1].position_m == (0.19, -0.135, 0.22)
    assert microphones[2].position_m == (-0.19, 0.135, 0.22)
    assert microphones[3].position_m == (-0.19, -0.135, 0.22)
    assert [round(math.degrees(mic.yaw_rad)) for mic in microphones] == [45, -45, 135, -135]


def test_array_moves_rigidly_with_robot() -> None:
    microphones = rectangular_array()
    world = transform_array(
        microphones,
        robot_position_m=(2.0, -1.0, 0.1),
        robot_yaw_rad=math.pi / 2.0,
    )
    assert np.allclose(world[0], (2.0 - 0.135, -1.0 + 0.19, 0.32))
    original_distances = np.linalg.norm(
        np.asarray([mic.position_m for mic in microphones])[:, None, :]
        - np.asarray([mic.position_m for mic in microphones])[None, :, :],
        axis=2,
    )
    moved_distances = np.linalg.norm(np.asarray(world)[:, None, :] - np.asarray(world)[None, :, :], axis=2)
    assert np.allclose(original_distances, moved_distances)


def test_left_right_and_front_rear_geometric_arrivals() -> None:
    left = _delays((0.0, 5.0, 0.22))
    right = _delays((0.0, -5.0, 0.22))
    front = _delays((5.0, 0.0, 0.22))
    rear = _delays((-5.0, 0.0, 0.22))
    assert max(left[0], left[2]) < min(left[1], left[3])
    assert max(right[1], right[3]) < min(right[0], right[2])
    assert max(front[0], front[1]) < min(front[2], front[3])
    assert max(rear[2], rear[3]) < min(rear[0], rear[1])
    assert (left[1] - left[0]) == pytest.approx(0.000787, abs=2e-6)
    assert (front[2] - front[0]) == pytest.approx(0.001107, abs=2e-6)


def test_independent_channels_keep_tdoa_and_stereo_asymmetry() -> None:
    rate = 16000
    delays = _delays((0.0, 5.0, 0.22))
    pulse = np.hanning(64).astype(np.float32)
    audio = np.zeros((4, 512), dtype=np.float32)
    base = 80
    for channel, delay in enumerate(delays - delays.min()):
        start = base + round(float(delay) * rate)
        audio[channel, start:start + len(pulse)] = pulse * (1.0 - 0.08 * channel)
    assert not np.array_equal(audio[0], audio[1])
    estimate, confidence = gcc_phat(audio[1], audio[0], sample_rate_hz=rate, max_tau_seconds=0.002)
    assert estimate > 0.0
    assert abs(estimate - (delays[1] - delays[0])) < 1.0 / rate
    assert confidence > 0.0
    stereo = headphone_stereo(audio)
    assert not np.array_equal(stereo[0], stereo[1])


def test_detection_fusion_selects_strongest_without_phase_averaging() -> None:
    audio = np.zeros((4, 32), dtype=np.float32)
    audio[0] = 0.1
    audio[2] = -0.8
    assert np.array_equal(hearing_waveform(audio), audio[2])


def test_mems_calibration_and_silence_are_stable() -> None:
    tone = np.sin(np.linspace(0.0, 4.0 * math.pi, 1600, endpoint=False)).astype(np.float32)
    calibrated = calibrate_mems(tone, 94.0)
    measured_dbfs = 20.0 * math.log10(float(np.sqrt(np.mean(calibrated**2))))
    assert measured_dbfs == pytest.approx(-26.0, abs=0.02)
    assert np.array_equal(calibrate_mems(np.zeros(64, dtype=np.float32), 80.0), np.zeros(64, dtype=np.float32))


def test_fractional_delay_is_not_rounded_away() -> None:
    integer, delayed = fractional_delay(np.asarray([1.0, 0.0], dtype=np.float32), 3.25)
    assert integer == 3
    assert delayed[0] == pytest.approx(0.75)
    assert delayed[1] == pytest.approx(0.25)


def test_disable_mute_solo_and_reenable_controls() -> None:
    audio = np.arange(32, dtype=np.float32).reshape(4, 8)
    assert not np.any(apply_monitor_controls(audio, enabled=False))
    assert not np.any(apply_monitor_controls(audio, muted=True))
    assert np.array_equal(apply_monitor_controls(audio, enabled=True), audio)
    solo = apply_monitor_controls(audio, solo_channel="rear_left")
    assert np.array_equal(solo[2], audio[2])
    assert not np.any(solo[[0, 1, 3]])


def test_monitor_gain_does_not_change_calibrated_raw_audio() -> None:
    raw = np.asarray([0.001, -0.001, 0.02], dtype=np.float32)
    untouched = raw.copy()
    monitored = monitor_amplify(raw, gain_db=40.0, limit=0.98)
    np.testing.assert_allclose(monitored[:2], [0.1, -0.1], atol=1e-6)
    assert monitored[2] == pytest.approx(0.98)
    np.testing.assert_array_equal(raw, untouched)


def test_streaming_fractional_delays_retain_history_between_blocks() -> None:
    first, history = streaming_fractional_delays(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.asarray([0.0, 1.5]),
    )
    np.testing.assert_allclose(first[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(first[1], [0.0, 0.5, 1.5, 2.5])

    second, _ = streaming_fractional_delays(
        np.asarray([5.0, 6.0], dtype=np.float32),
        np.asarray([0.0, 1.5]),
        history,
    )
    np.testing.assert_allclose(second[0], [5.0, 6.0])
    np.testing.assert_allclose(second[1], [3.5, 4.5])
