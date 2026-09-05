from __future__ import annotations

import threading
from collections.abc import Hashable

import numpy as np

from task_generator.auditory.asset_lib import CachedSample
from task_generator.auditory.drivetrain import (
    JACKAL,
    DrivetrainVoice,
    clear_cache,
    prewarm,
)


def clear_drivetrain_audio_cache() -> None:
    clear_cache()


class PartitionedConvolver:
    """Uniform partitioned mono FIR convolution for fixed audio blocks."""

    def __init__(self, impulse: np.ndarray, block_size: int) -> None:
        self.block_size = int(block_size)
        impulse = np.asarray(impulse, dtype=np.float32).reshape(-1)
        if self.block_size <= 0 or impulse.size == 0:
            raise ValueError("block_size and impulse must be non-empty")
        count = (len(impulse) + self.block_size - 1) // self.block_size
        padded = np.pad(
            impulse,
            (0, count * self.block_size - len(impulse)),
        ).reshape(count, self.block_size)
        self._filters = np.fft.rfft(
            np.pad(padded, ((0, 0), (0, self.block_size))),
            axis=1,
        )
        self._history = np.zeros_like(self._filters)
        self._position = 0
        self._overlap = np.zeros(self.block_size, dtype=np.float64)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if len(block) != self.block_size:
            raise ValueError(f"convolver requires {self.block_size} frames, got {len(block)}")
        spectrum = np.fft.rfft(np.pad(block, (0, self.block_size)))
        self._history[self._position] = spectrum
        first_count = self._position + 1
        output_spectrum = np.sum(
            self._filters[:first_count] * self._history[self._position :: -1],
            axis=0,
        )
        if first_count < len(self._filters):
            output_spectrum += np.sum(
                self._filters[first_count:] * self._history[: self._position : -1],
                axis=0,
            )
        rendered = np.fft.irfft(output_spectrum)
        output = rendered[: self.block_size] + self._overlap
        self._overlap = rendered[self.block_size :].copy()
        self._position = (self._position + 1) % len(self._history)
        return output.astype(np.float32)


class LoopingSampleRenderSource:
    """Loop a decoded WAV through one independently updated RIR."""

    def __init__(
        self,
        sample: CachedSample,
        *,
        block_size: int,
        loop: bool,
        start_frame: int = 0,
        rir_crossfade_seconds: float = 0.1,
    ) -> None:
        self.block_size = int(block_size)
        self.channels = int(sample.channels)
        self._samples = sample.samples
        self._sample_rate = int(sample.sample_rate)
        self._loop = bool(loop)
        self._position = max(int(start_frame), 0)
        if self._loop:
            self._position %= len(self._samples)
        self._lock = threading.Lock()
        self._active = True
        self._program_finished = not self._loop and self._position >= len(self._samples)
        self._current_gain = 0.0
        self._target_gain = 0.0
        self._inactive_frames = 0
        self._rir_tail_frames = self.block_size
        self._convolvers: list[PartitionedConvolver] | None = None
        self._old_convolvers: list[PartitionedConvolver] | None = None
        self._crossfade_total = max(
            int(self._sample_rate * rir_crossfade_seconds),
            1,
        )
        self._crossfade_remaining = 0
        self._rir_signature: tuple[Hashable, ...] | None = None

    def update(
        self,
        *,
        gain_db: float,
        active: bool,
        impulse: np.ndarray | None,
        rir_signature: tuple[Hashable, ...] | None,
    ) -> None:
        with self._lock:
            self._target_gain = 10.0 ** (float(gain_db) / 20.0) if active else 0.0
            self._active = bool(active)
            if active and not self._program_finished:
                self._inactive_frames = 0
            if impulse is not None and rir_signature != self._rir_signature:
                next_convolvers = [PartitionedConvolver(impulse, self.block_size) for _ in range(self.channels)]
                self._old_convolvers = self._convolvers
                self._convolvers = next_convolvers
                self._rir_signature = rir_signature
                self._rir_tail_frames = max(len(impulse), self.block_size)
                self._crossfade_remaining = self._crossfade_total if self._old_convolvers is not None else 0

    def render(self, frames: int) -> np.ndarray:
        if int(frames) != self.block_size:
            raise ValueError(f"looping source configured for {self.block_size} frames, audio callback requested {frames}")
        with self._lock:
            program_finished = self._program_finished
            position = self._position
            active = self._active and not program_finished
            target_gain = self._target_gain
            convolvers = self._convolvers
            old_convolvers = self._old_convolvers
            crossfade_remaining = self._crossfade_remaining

        dry = np.zeros((frames, self.channels), dtype=np.float32)
        if not program_finished:
            written = 0
            while written < frames and not program_finished:
                remaining = len(self._samples) - position
                count = min(frames - written, remaining)
                dry[written : written + count] = self._samples[position : position + count]
                position += count
                written += count
                if position >= len(self._samples):
                    if self._loop:
                        position = 0
                    else:
                        program_finished = True
            with self._lock:
                self._position = position
                self._program_finished = program_finished

        gain = np.linspace(
            self._current_gain,
            target_gain if active else 0.0,
            frames,
            dtype=np.float32,
        )
        self._current_gain = float(gain[-1])
        dry *= gain[:, None]

        if convolvers is not None:
            wet = np.stack(
                [convolver.process(dry[:, channel]) for channel, convolver in enumerate(convolvers)],
                axis=1,
            )
            if old_convolvers is not None and crossfade_remaining > 0:
                old_wet = np.stack(
                    [convolver.process(dry[:, channel]) for channel, convolver in enumerate(old_convolvers)],
                    axis=1,
                )
                elapsed = self._crossfade_total - crossfade_remaining
                alpha = np.clip(
                    (elapsed + np.arange(frames)) / self._crossfade_total,
                    0.0,
                    1.0,
                ).astype(np.float32)
                wet = old_wet * np.sqrt(1.0 - alpha)[:, None] + wet * np.sqrt(alpha)[:, None]
                remaining = max(crossfade_remaining - frames, 0)
                with self._lock:
                    self._crossfade_remaining = remaining
                    if remaining == 0:
                        self._old_convolvers = None
            dry = wet

        with self._lock:
            if not active:
                self._inactive_frames += frames
        return np.asarray(dry, dtype=np.float32)

    @property
    def finished(self) -> bool:
        with self._lock:
            return (not self._active or self._program_finished) and self._inactive_frames >= self._rir_tail_frames + self._crossfade_total


class DrivetrainRenderSource:
    """Persistent Jackal source with one source-only room treatment stage."""

    def __init__(
        self,
        *,
        field_seed: int,
        phase_index: int,
        block_size: int,
        channels: int,
        volume_db: float,
        frequency_scale: float,
        tonal_gain_db: float,
        broadband_gain_db: float,
        speed_exponent: float,
        velocity_smoothing_seconds: float,
        rir_crossfade_seconds: float = 0.1,
    ) -> None:
        self.block_size = int(block_size)
        self.channels = int(channels)
        self._seed = int(field_seed) & 0xFFFFFFFF
        phase_index = int(phase_index) & 0x0FFFFFFF
        prewarm(JACKAL, seed=self._seed)
        # The bundled transfer contains recording-room and microphone colour.
        # Disable it so pyroomacoustics is the only simulated RIR.
        self._left = DrivetrainVoice(
            JACKAL,
            index=phase_index * 2,
            count=2,
            seed=self._seed,
            transfer=False,
        )
        self._right = DrivetrainVoice(
            JACKAL,
            index=phase_index * 2 + 1,
            count=2,
            seed=self._seed,
            transfer=False,
        )
        self._lock = threading.Lock()
        self._target_left = 0.0
        self._target_right = 0.0
        self._current_left = 0.0
        self._current_right = 0.0
        self._target_gain = 0.0
        self._current_gain = 0.0
        self._target_volume_gain = 10.0 ** (float(volume_db) / 20.0)
        self._current_volume_gain = self._target_volume_gain
        self._frequency_scale = float(frequency_scale)
        self._tonal_gain_db = float(tonal_gain_db)
        self._broadband_gain_db = float(broadband_gain_db)
        self._speed_exponent = float(speed_exponent)
        self._velocity_smoothing_seconds = float(velocity_smoothing_seconds)
        self._active = False
        self._inactive_frames = 0
        self._rir_tail_frames = self.block_size
        self._convolver: PartitionedConvolver | None = None
        self._old_convolver: PartitionedConvolver | None = None
        self._crossfade_total = max(int(JACKAL.sample_rate * rir_crossfade_seconds), 1)
        self._crossfade_remaining = 0
        self._rir_signature: tuple[Hashable, ...] | None = None

    def update(
        self,
        *,
        left_velocity: float,
        right_velocity: float,
        gain_db: float,
        active: bool,
        impulse: np.ndarray | None,
        rir_signature: tuple[Hashable, ...] | None,
    ) -> None:
        with self._lock:
            self._target_left = float(left_velocity) if active else 0.0
            self._target_right = float(right_velocity) if active else 0.0
            self._target_gain = 10.0 ** (float(gain_db) / 20.0) if active else 0.0
            self._active = bool(active)
            if active:
                self._inactive_frames = 0
            if impulse is not None and rir_signature != self._rir_signature:
                next_convolver = PartitionedConvolver(impulse, self.block_size)
                self._old_convolver = self._convolver
                self._convolver = next_convolver
                self._rir_signature = rir_signature
                self._rir_tail_frames = max(len(impulse), self.block_size)
                self._crossfade_remaining = self._crossfade_total if self._old_convolver is not None else 0

    def tune(
        self,
        *,
        volume_db: float,
        frequency_scale: float,
        tonal_gain_db: float,
        broadband_gain_db: float,
        speed_exponent: float,
        velocity_smoothing_seconds: float,
    ) -> None:
        with self._lock:
            self._target_volume_gain = 10.0 ** (float(volume_db) / 20.0)
            self._frequency_scale = float(frequency_scale)
            self._tonal_gain_db = float(tonal_gain_db)
            self._broadband_gain_db = float(broadband_gain_db)
            self._speed_exponent = float(speed_exponent)
            self._velocity_smoothing_seconds = float(velocity_smoothing_seconds)

    def render(self, frames: int) -> np.ndarray:
        if int(frames) != self.block_size:
            raise ValueError(f"drivetrain source configured for {self.block_size} frames, audio callback requested {frames}")
        with self._lock:
            target_left = self._target_left
            target_right = self._target_right
            target_gain = self._target_gain
            convolver = self._convolver
            old_convolver = self._old_convolver
            crossfade_remaining = self._crossfade_remaining
            target_volume_gain = self._target_volume_gain
            frequency_scale = self._frequency_scale
            tonal_gain_db = self._tonal_gain_db
            broadband_gain_db = self._broadband_gain_db
            speed_exponent = self._speed_exponent
            velocity_smoothing_seconds = self._velocity_smoothing_seconds

        if velocity_smoothing_seconds <= 0.0:
            left_speed = np.full(frames, target_left, dtype=np.float64)
            right_speed = np.full(frames, target_right, dtype=np.float64)
        else:
            decay = np.exp(-np.arange(1, frames + 1, dtype=np.float64) / (JACKAL.sample_rate * velocity_smoothing_seconds))
            left_speed = target_left + (self._current_left - target_left) * decay
            right_speed = target_right + (self._current_right - target_right) * decay
        self._current_left = float(left_speed[-1])
        self._current_right = float(right_speed[-1])
        render_options = {
            "frequency_scale": frequency_scale,
            "tonal_gain_db": tonal_gain_db,
            "broadband_gain_db": broadband_gain_db,
            "speed_exponent": speed_exponent,
        }
        dry = self._left.render(
            left_speed,
            **render_options,
        ) + self._right.render(
            right_speed,
            **render_options,
        )

        gain = np.linspace(self._current_gain, target_gain, frames, dtype=np.float32)
        self._current_gain = target_gain
        dry = np.asarray(dry, dtype=np.float32) * gain
        volume_gain = np.linspace(
            self._current_volume_gain,
            target_volume_gain,
            frames,
            dtype=np.float32,
        )
        self._current_volume_gain = target_volume_gain
        dry *= volume_gain

        if convolver is not None:
            wet = convolver.process(dry)
            if old_convolver is not None and crossfade_remaining > 0:
                old_wet = old_convolver.process(dry)
                elapsed = self._crossfade_total - crossfade_remaining
                alpha = np.clip(
                    (elapsed + np.arange(frames)) / self._crossfade_total,
                    0.0,
                    1.0,
                ).astype(np.float32)
                wet = old_wet * np.sqrt(1.0 - alpha) + wet * np.sqrt(alpha)
                remaining = max(crossfade_remaining - frames, 0)
                with self._lock:
                    self._crossfade_remaining = remaining
                    if remaining == 0:
                        self._old_convolver = None
            dry = wet
        mono = np.asarray(dry, dtype=np.float32)
        with self._lock:
            if not self._active:
                self._inactive_frames += frames
        return np.repeat(mono[:, None], self.channels, axis=1)

    @property
    def finished(self) -> bool:
        with self._lock:
            velocity_tail_frames = int(JACKAL.sample_rate * self._velocity_smoothing_seconds * 5.0)
            return (
                not self._active
                and self._inactive_frames
                >= max(
                    self._rir_tail_frames,
                    velocity_tail_frames,
                )
                + self._crossfade_total
            )
