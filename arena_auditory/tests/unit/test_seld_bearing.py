"""The bearing fit recovers every azimuth on the sim array; the front-end's gcc path does too, with the real checkpoint."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from arena_auditory.hearing.doa import ArrayBearing
from arena_auditory.spatial_audio import fractional_delay, geometric_delays_seconds, rectangular_array

FS = 16000
BEARINGS_DEG = (0, 45, 90, 180, 225, 270)


def _place(clip: np.ndarray, az_deg: float, *, repeats: int, period_s: float, lead_s: float, total_s: float) -> np.ndarray:
    """4-channel (samples, ch) rendering of ``clip`` from ``az_deg`` at 3 m, sim delays, no reverb."""
    az = math.radians(az_deg)
    delays = geometric_delays_seconds((3.0 * math.cos(az), 3.0 * math.sin(az), 0.0), tuple(m.position_m for m in rectangular_array()))
    delays = delays - delays.min()
    out = np.zeros((int(total_s * FS), 4), dtype=np.float32)
    for k in range(repeats):
        t0 = int((lead_s + period_s * k) * FS)
        for ch in range(4):
            i, s = fractional_delay(clip, delays[ch] * FS)
            out[t0 + i : t0 + i + len(s), ch] += s
    return out


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("az_deg", BEARINGS_DEG)
def test_array_bearing_fits_noise_burst(az_deg: int) -> None:
    rng = np.random.default_rng(az_deg)
    burst = (rng.standard_normal(int(0.05 * FS)) * 0.3).astype(np.float32)
    frame = _place(burst, az_deg, repeats=1, period_s=1.0, lead_s=0.02, total_s=0.2)
    theta, residual, valid = ArrayBearing(FS).bearing(frame)
    assert valid
    assert abs(_wrap_deg(math.degrees(theta) - az_deg)) <= 3.0
    assert residual < 1e-4


def _ego_noise(seed: int, amp: float, *, total_s: float = 0.2) -> np.ndarray:
    """Identical 4-channel noise, standing in for drivetrain noise rendered from the array centre."""
    rng = np.random.default_rng(seed)
    mono = (rng.standard_normal(int(total_s * FS)) * amp).astype(np.float32)
    return np.tile(mono[:, None], (1, 4))


def test_array_bearing_gates_ego_noise() -> None:
    _theta, _residual, valid = ArrayBearing(FS).bearing(_ego_noise(1, 0.3))
    assert not valid


def test_array_bearing_valid_when_footstep_above_ego_noise() -> None:
    rng = np.random.default_rng(90)
    burst = (rng.standard_normal(int(0.05 * FS)) * 0.3).astype(np.float32)
    foot = _place(burst, 90, repeats=1, period_s=1.0, lead_s=0.02, total_s=0.2)
    theta, _residual, valid = ArrayBearing(FS).bearing(_ego_noise(2, 0.05) + foot)
    assert valid
    assert abs(_wrap_deg(math.degrees(theta) - 90)) <= 5.0


def test_array_bearing_gated_when_footstep_drowned_by_ego_noise() -> None:
    rng = np.random.default_rng(91)
    burst = (rng.standard_normal(int(0.05 * FS)) * 0.02).astype(np.float32)
    foot = _place(burst, 90, repeats=1, period_s=1.0, lead_s=0.02, total_s=0.2)
    _theta, _residual, valid = ArrayBearing(FS).bearing(_ego_noise(3, 0.3) + foot)
    assert not valid


def _weights() -> dict[str, str] | None:
    from arena_auditory.hearing import weights

    files = {entry["role"]: Path(weights.data_dir()) / entry["dest"] for entry in weights.manifest()}
    return {role: str(path) for role, path in files.items()} if all(p.is_file() for p in files.values()) else None


def _footstep() -> np.ndarray:
    librosa = pytest.importorskip("librosa")
    soundfile = pytest.importorskip("soundfile")
    from ament_index_python.packages import get_package_share_directory

    audio, fs = soundfile.read(os.path.join(get_package_share_directory("arena_auditory"), "sounds", "footstep_default.wav"), dtype="float32", always_2d=True)
    return librosa.resample(audio[:, 0], orig_sr=fs, target_sr=FS).astype(np.float32) * 0.3


@pytest.mark.parametrize("az_deg", BEARINGS_DEG)
def test_frontend_gcc_bearing_with_checkpoint(az_deg: int) -> None:
    pytest.importorskip("torch")
    files = _weights()
    if files is None:
        pytest.skip("SELD weights not fetched (ros2 run arena_auditory hearing_setup)")
    from arena_auditory.hearing.seld import SeldFrontend, SeldStream

    fe = SeldFrontend(files["checkpoint"], files["scaler"], device="cpu")
    stream = SeldStream(fe, lookahead=5)
    doa = ArrayBearing(fe.fs)
    audio = _place(_footstep(), az_deg, repeats=12, period_s=0.5, lead_s=0.4, total_s=7.0)
    fitted: list[float] = []
    hop = fe.label_hop_len
    for start in range(0, audio.shape[0] - hop + 1, hop):
        stream.push(audio[start : start + hop])
        dets, _end, seg = stream.step()
        if dets:
            fitted.append(math.degrees(doa.bearing(seg)[0]))
    assert len(fitted) >= 10
    errors = np.abs([_wrap_deg(f - az_deg) for f in fitted])
    assert float(np.median(errors)) <= 5.0
