"""Pure spatial-audio primitives used by the four-microphone Jackal receiver.

This module deliberately has no ROS dependencies so geometry, synchronization,
fusion and TDoA behavior can be unit tested without a running simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CHANNEL_NAMES = (
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)


@dataclass(frozen=True, slots=True)
class Microphone:
    name: str
    position_m: tuple[float, float, float]
    yaw_rad: float


def rectangular_array(
    *,
    width_m: float = 0.310,
    length_m: float = 0.420,
    height_m: float = 0.220,
    corner_inset_m: float = 0.020,
) -> tuple[Microphone, ...]:
    """Return the REP-103 four-microphone array in canonical channel order."""
    values = (width_m, length_m, height_m, corner_inset_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("microphone geometry must be finite")
    if width_m <= 0.0 or length_m <= 0.0 or height_m < 0.0:
        raise ValueError("width/length must be positive and height non-negative")
    if corner_inset_m < 0.0 or 2.0 * corner_inset_m >= min(width_m, length_m):
        raise ValueError("corner inset must leave a positive rectangular aperture")
    x = length_m / 2.0 - corner_inset_m
    y = width_m / 2.0 - corner_inset_m
    return (
        Microphone("front_left", (x, y, height_m), math.radians(45.0)),
        Microphone("front_right", (x, -y, height_m), math.radians(-45.0)),
        Microphone("rear_left", (-x, y, height_m), math.radians(135.0)),
        Microphone("rear_right", (-x, -y, height_m), math.radians(-135.0)),
    )


def transform_array(
    microphones: tuple[Microphone, ...],
    *,
    robot_position_m: tuple[float, float, float],
    robot_yaw_rad: float,
) -> tuple[tuple[float, float, float], ...]:
    """Rigidly transform base_link microphone coordinates into a world frame."""
    cosine, sine = math.cos(robot_yaw_rad), math.sin(robot_yaw_rad)
    rx, ry, rz = robot_position_m
    return tuple(
        (
            rx + cosine * mic.position_m[0] - sine * mic.position_m[1],
            ry + sine * mic.position_m[0] + cosine * mic.position_m[1],
            rz + mic.position_m[2],
        )
        for mic in microphones
    )


def geometric_delays_seconds(
    source_position_m: tuple[float, float, float],
    microphone_positions_m: tuple[tuple[float, float, float], ...],
    *,
    speed_of_sound_mps: float = 343.0,
) -> NDArray[np.float64]:
    if speed_of_sound_mps <= 0.0 or not math.isfinite(speed_of_sound_mps):
        raise ValueError("speed of sound must be finite and positive")
    source = np.asarray(source_position_m, dtype=np.float64)
    positions = np.asarray(microphone_positions_m, dtype=np.float64)
    return np.linalg.norm(positions - source[None, :], axis=1) / speed_of_sound_mps


def rms(samples: NDArray[np.floating], *, axis: int | None = None) -> NDArray[np.float64] | float:
    values = np.asarray(samples, dtype=np.float64)
    result = np.sqrt(np.mean(values * values, axis=axis)) if values.size else 0.0
    return result


def dbfs_from_rms(value: float, *, floor_db: float = -120.0) -> float:
    return max(20.0 * math.log10(max(float(value), 1e-12)), floor_db)


def calibrate_mems(
    samples: NDArray[np.floating],
    received_spl_db: float,
    *,
    sensitivity_dbfs_at_94_dbspl: float = -26.0,
) -> NDArray[np.float32]:
    """Apply one fixed MEMS sensitivity; this is calibration, never AGC.

    At the reference defaults 94 dB SPL maps to -26 dBFS and 120 dB SPL to
    0 dBFS, matching the cited XVF3800 microphone and overload figures.
    """
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    source_rms = float(rms(mono))
    if source_rms <= 1e-12:
        return np.zeros_like(mono)
    target_dbfs = float(received_spl_db) - 94.0 + sensitivity_dbfs_at_94_dbspl
    target_rms = 10.0 ** (target_dbfs / 20.0)
    return np.ascontiguousarray(mono * (target_rms / source_rms), dtype=np.float32)


def fractional_delay(
    samples: NDArray[np.floating],
    delay_samples: float,
) -> tuple[int, NDArray[np.float32]]:
    """Split a non-negative delay into integer scheduling and fractional FIR.

    Linear interpolation is intentionally small and causal.  It retains the
    sub-sample timing/phase encoded by propagation instead of rounding every
    microphone arrival to the nearest 62.5 us at 16 kHz.
    """
    if delay_samples < 0.0 or not math.isfinite(delay_samples):
        raise ValueError("delay_samples must be finite and non-negative")
    integer = int(math.floor(delay_samples))
    fraction = delay_samples - integer
    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source.size == 0:
        return integer, source
    if fraction <= 1e-9:
        return integer, np.ascontiguousarray(source)
    shifted = np.empty(source.size + 1, dtype=np.float32)
    shifted[0] = (1.0 - fraction) * source[0]
    shifted[1:-1] = (1.0 - fraction) * source[1:] + fraction * source[:-1]
    shifted[-1] = fraction * source[-1]
    return integer, np.ascontiguousarray(shifted)


def hearing_waveform(channels: NDArray[np.floating]) -> NDArray[np.float32]:
    """Detection-safe mono: return the highest-energy raw channel per block.

    Selecting (rather than phase-averaging) prevents cancellation and guarantees
    block RMS is at least that of any single alternative selected for listening.
    """
    audio = np.asarray(channels, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError("channels must have shape (4, frames)")
    energies = np.asarray(rms(audio, axis=1))
    return np.ascontiguousarray(audio[int(np.argmax(energies))])


def apply_monitor_controls(
    channels: NDArray[np.floating],
    *,
    enabled: bool = True,
    muted: bool = False,
    solo_channel: str = "",
) -> NDArray[np.float32]:
    """Apply reversible runtime gating without changing channel alignment."""
    audio = np.asarray(channels, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError("channels must have shape (4, frames)")
    output = np.ascontiguousarray(audio.copy())
    if not enabled or muted:
        output.fill(0.0)
        return output
    if solo_channel:
        if solo_channel not in CHANNEL_NAMES:
            raise ValueError("solo_channel must be empty or a canonical name")
        keep = CHANNEL_NAMES.index(solo_channel)
        output[np.arange(4) != keep] = 0.0
    return output


def headphone_stereo(
    channels: NDArray[np.floating],
    *,
    front_gain: float = 1.0,
    rear_gain: float = 0.75,
    output_gain: float = 0.8,
) -> NDArray[np.float32]:
    """Map FL/RL and FR/RR without time alignment or independent processing."""
    audio = np.asarray(channels, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError("channels must have shape (4, frames)")
    denominator = max(abs(front_gain) + abs(rear_gain), 1e-12)
    left = (front_gain * audio[0] + rear_gain * audio[2]) / denominator
    right = (front_gain * audio[1] + rear_gain * audio[3]) / denominator
    stereo = np.stack((left, right), axis=0) * float(output_gain)
    return np.ascontiguousarray(np.clip(stereo, -1.0, 1.0), dtype=np.float32)


def monitor_amplify(
    samples: NDArray[np.floating],
    *,
    gain_db: float = 36.0,
    limit: float = 0.98,
) -> NDArray[np.float32]:
    """Amplify calibrated microphone PCM for listening, with peak limiting.

    Raw array products remain physically calibrated.  This function is only
    for workstation monitoring, where a real microphone preamplifier and
    headphone level control would otherwise be missing from the simulation.
    """
    if not math.isfinite(gain_db):
        raise ValueError("monitor gain must be finite")
    if not math.isfinite(limit) or not 0.0 < limit <= 1.0:
        raise ValueError("monitor limit must be finite and in (0, 1]")
    audio = np.asarray(samples, dtype=np.float32)
    gain = 10.0 ** (gain_db / 20.0)
    return np.ascontiguousarray(
        np.clip(audio * gain, -limit, limit),
        dtype=np.float32,
    )


def streaming_fractional_delays(
    samples: NDArray[np.floating],
    delay_samples: NDArray[np.floating],
    history: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply independent causal delays to one streaming mono block.

    The returned channels share one source waveform and retain their history
    across calls.  Growing a delay pads unavailable older samples with silence;
    normal microphone motion changes delay gradually and keeps valid history.
    """
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    delays = np.asarray(delay_samples, dtype=np.float64).reshape(-1)
    if delays.size == 0:
        raise ValueError("at least one channel delay is required")
    if not np.all(np.isfinite(delays)) or np.any(delays < 0.0):
        raise ValueError("channel delays must be finite and non-negative")
    required = max(int(math.ceil(float(np.max(delays)))) + 1, 1)
    previous = np.zeros(required, dtype=np.float32) if history is None else np.asarray(history, dtype=np.float32).reshape(-1)
    if previous.size < required:
        previous = np.pad(previous, (required - previous.size, 0))
    combined = np.concatenate((previous, mono))
    origin = previous.size
    output = np.zeros((delays.size, mono.size), dtype=np.float32)
    frames = np.arange(mono.size, dtype=np.float64)
    for channel, delay in enumerate(delays):
        positions = origin + frames - delay
        lower = np.floor(positions).astype(np.int64)
        fraction = positions - lower
        lower_valid = (lower >= 0) & (lower < combined.size)
        output[channel, lower_valid] = combined[lower[lower_valid]] * (1.0 - fraction[lower_valid])
        upper = lower + 1
        upper_valid = (fraction > 1e-12) & (upper >= 0) & (upper < combined.size)
        output[channel, upper_valid] += combined[upper[upper_valid]] * fraction[upper_valid]
    return output, np.ascontiguousarray(combined[-required:], dtype=np.float32)


def gcc_phat(
    signal: NDArray[np.floating],
    reference: NDArray[np.floating],
    *,
    sample_rate_hz: int,
    max_tau_seconds: float | None = None,
    interpolation: int = 8,
) -> tuple[float, float]:
    """Estimate signal-minus-reference delay and normalized peak confidence."""
    sig = np.asarray(signal, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if sig.size == 0 or ref.size == 0 or sample_rate_hz <= 0:
        return 0.0, 0.0
    n = sig.size + ref.size
    spectrum = np.fft.rfft(sig, n=n) * np.conj(np.fft.rfft(ref, n=n))
    magnitude = np.abs(spectrum)
    spectrum /= np.maximum(magnitude, 1e-15)
    correlation = np.fft.irfft(spectrum, n=interpolation * n)
    maximum_shift = interpolation * n // 2
    if max_tau_seconds is not None:
        maximum_shift = min(
            maximum_shift,
            int(interpolation * sample_rate_hz * max_tau_seconds),
        )
    correlation = np.concatenate((correlation[-maximum_shift:], correlation[: maximum_shift + 1]))
    peak_index = int(np.argmax(np.abs(correlation)))
    shift = peak_index - maximum_shift
    confidence = float(np.abs(correlation[peak_index]))
    return shift / float(interpolation * sample_rate_hz), confidence


def interleave(channels: NDArray[np.floating]) -> list[float]:
    audio = np.asarray(channels, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("audio must be channels-first")
    return audio.T.reshape(-1).tolist()
