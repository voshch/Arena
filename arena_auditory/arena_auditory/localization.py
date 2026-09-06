from __future__ import annotations

from typing import Protocol, runtime_checkable

import attrs
import numpy as np
from numpy.typing import NDArray

Position3D = tuple[float, float, float]


@attrs.frozen(slots=True)
class SourceHypothesis:
    """One candidate physical source location."""

    frame_id: str
    position_m: Position3D
    weight: float = 1.0
    covariance_m2: tuple[float, ...] | None = None
    nlos_probability: float | None = None
    path_type: str | None = None

    def __attrs_post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame_id must be non-empty")
        if not np.isfinite(self.weight):
            raise ValueError("weight must be finite")
        if self.nlos_probability is not None and not (0.0 <= self.nlos_probability <= 1.0):
            raise ValueError("nlos_probability must be between 0 and 1")


@attrs.frozen(slots=True)
class AudioLocalizationInput:
    """Backend-agnostic multichannel audio input for localization."""

    timestamp_ns: int
    sample_rate_hz: int
    audio: NDArray[np.floating]
    microphone_positions_m: tuple[Position3D, ...]
    microphone_frame_ids: tuple[str, ...] = ()
    robot_frame_id: str = ""
    world_frame_id: str = ""
    robot_position_world_m: Position3D | None = None
    robot_yaw_world_rad: float | None = None

    def __attrs_post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.audio.ndim != 2:
            raise ValueError("audio must be a 2D array with channels first")
        if len(self.microphone_positions_m) != self.audio.shape[0]:
            raise ValueError("microphone_positions_m must match audio channels")
        if self.microphone_frame_ids and len(self.microphone_frame_ids) != self.audio.shape[0]:
            raise ValueError("microphone_frame_ids must match audio channels")
        if self.robot_frame_id and not self.robot_frame_id.strip():
            raise ValueError("robot_frame_id must be non-empty when provided")
        if self.world_frame_id and not self.world_frame_id.strip():
            raise ValueError("world_frame_id must be non-empty when provided")
        if self.robot_yaw_world_rad is not None and not np.isfinite(self.robot_yaw_world_rad):
            raise ValueError("robot_yaw_world_rad must be finite")

    @property
    def channel_count(self) -> int:
        return int(self.audio.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.audio.shape[1])


@attrs.frozen(slots=True)
class LocalizationObservation:
    """Primary output for a localization backend."""

    timestamp_ns: int
    source_present: bool
    azimuth_rad: float | None
    confidence: float
    elevation_rad: float | None = None
    generic_sound_probability: float | None = None
    event_class: str | None = None
    robot_frame_id: str = ""
    world_frame_id: str = ""
    source_position_robot_m: Position3D | None = None
    source_position_world_m: Position3D | None = None
    position_covariance_m2: tuple[float, ...] | None = None
    tracking_status: str = "unknown"
    nlos_probability: float | None = None
    path_type: str | None = None
    source_hypotheses: tuple[SourceHypothesis, ...] = ()

    def __attrs_post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not np.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        if self.generic_sound_probability is not None and not (0.0 <= self.generic_sound_probability <= 1.0):
            raise ValueError("generic_sound_probability must be between 0 and 1")
        if self.nlos_probability is not None and not (0.0 <= self.nlos_probability <= 1.0):
            raise ValueError("nlos_probability must be between 0 and 1")


@runtime_checkable
class LocalizationBackend(Protocol):
    """Callable backend contract for acoustic localization modules."""

    def localize(self, audio_input: AudioLocalizationInput) -> LocalizationObservation:
        """Return one localization observation for one audio frame."""
