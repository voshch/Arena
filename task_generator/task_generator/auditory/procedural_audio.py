from __future__ import annotations

import threading

import numpy as np

from task_generator.auditory.drivetrain import (
    DrivetrainVoice,
    JACKAL,
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
            raise ValueError(
                f"convolver requires {self.block_size} frames, got {len(block)}"
            )
        spectrum = np.fft.rfft(np.pad(block, (0, self.block_size)))
        self._history[self._position] = spectrum
        first_count = self._position + 1
        output_spectrum = np.sum(
            self._filters[:first_count]
            * self._history[self._position :: -1],
            axis=0,
        )
        if first_count < len(self._filters):
            output_spectrum += np.sum(
                self._filters[first_count:]
                * self._history[: self._position : -1],
                axis=0,
            )
        rendered = np.fft.irfft(output_spectrum)
        output = rendered[: self.block_size] + self._overlap
        self._overlap = rendered[self.block_size :].copy()
        self._position = (self._position + 1) % len(self._history)
        return output.astype(np.float32)


class DrivetrainRenderSource:
    """Persistent Jackal source with one source-only room treatment stage."""

    def __init__(
        self,
        *,
        field_seed: int,
        phase_index: int,
        block_size: int,
        channels: int,
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
        self._active = False
        self._inactive_frames = 0
        self._tail_frames = self.block_size
        self._convolver: PartitionedConvolver | None = None
        self._old_convolver: PartitionedConvolver | None = None
        self._crossfade_total = max(
            int(JACKAL.sample_rate * rir_crossfade_seconds), 1
        )
        self._crossfade_remaining = 0
        self._rir_signature: tuple[object, ...] | None = None

    def update(
        self,
        *,
        left_velocity: float,
        right_velocity: float,
        gain_db: float,
        active: bool,
        impulse: np.ndarray | None,
        rir_signature: tuple[object, ...] | None,
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
                self._tail_frames = max(len(impulse), self.block_size)
                self._crossfade_remaining = (
                    self._crossfade_total
                    if self._old_convolver is not None
                    else 0
                )

    def render(self, frames: int) -> np.ndarray:
        if int(frames) != self.block_size:
            raise ValueError(
                f"drivetrain source configured for {self.block_size} frames, "
                f"audio callback requested {frames}"
            )
        with self._lock:
            target_left = self._target_left
            target_right = self._target_right
            target_gain = self._target_gain
            convolver = self._convolver
            old_convolver = self._old_convolver
            crossfade_remaining = self._crossfade_remaining

        left_speed = np.linspace(
            self._current_left, target_left, frames, dtype=np.float64
        )
        right_speed = np.linspace(
            self._current_right, target_right, frames, dtype=np.float64
        )
        dry = self._left.render(left_speed) + self._right.render(right_speed)
        self._current_left = target_left
        self._current_right = target_right

        gain = np.linspace(
            self._current_gain, target_gain, frames, dtype=np.float32
        )
        self._current_gain = target_gain
        dry = np.asarray(dry, dtype=np.float32) * gain

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
            return (
                not self._active
                and self._inactive_frames
                >= self._tail_frames + self._crossfade_total
            )
