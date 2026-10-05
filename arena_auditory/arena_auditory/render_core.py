"""Block renderer for the four-microphone array, free of ROS dependencies.

Interpolation and smoothing of incoming propagation state happens in the node,
which hands this module fully resolved source descriptions.  What lives here is
the carried DSP state (read cursors, delay history, drivetrain phase) and the
accumulation that turns one set of descriptions into one block of PCM.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from arena_auditory.procedural_audio import DrivetrainRenderSource
from arena_auditory.spatial_audio import (
    calibrate_mems,
    fractional_delay,
    ramped_read,
    streaming_fractional_delays,
)

CHANNELS = 4

_LOG = logging.getLogger(__name__)


def _log_error(message: str) -> None:
    _LOG.error(message)


@dataclass(slots=True)
class ScheduledClip:
    start: int
    samples: np.ndarray


@dataclass(slots=True)
class ContinuousVoice:
    samples: np.ndarray
    program_start: int
    delay_samples: float
    delay_target: float
    gain: float
    loop: bool
    active: bool


@dataclass(slots=True)
class ProceduralArrayVoice:
    source: DrivetrainRenderSource
    deterministic_seed: int
    gains: np.ndarray
    delay_samples: np.ndarray
    active_channels: np.ndarray
    source_id: str
    source_agent_id: int
    source_agent_name: str
    sound_type: str
    asset_id: str
    history: np.ndarray | None = None


@dataclass(frozen=True)
class DrivetrainTuning:
    volume_db: float
    frequency_scale: float
    tonal_gain_db: float
    broadband_gain_db: float
    speed_exponent: float
    velocity_smoothing_seconds: float


@dataclass(frozen=True)
class ClipInput:
    """One finite arrival, already calibrated and fractionally delayed."""

    channel: int
    start: int
    asset_key: str
    samples: np.ndarray
    anchor: int
    received_volume_db: float
    sensitivity_dbfs_at_94_dbspl: float
    delay_samples: float


@dataclass(frozen=True)
class ContinuousInput:
    source_id: str
    channel: int
    asset_key: str
    gain: float
    delay_target: float
    loop: bool
    program_start: int


@dataclass(frozen=True)
class DrivetrainInput:
    source_id: str
    deterministic_seed: int
    gains: tuple[float, float, float, float]
    delay_samples: tuple[float, float, float, float]
    active_channels: tuple[bool, bool, bool, bool]
    left_velocity: float
    right_velocity: float
    active: bool
    tuning: DrivetrainTuning
    source_agent_id: int
    source_agent_name: str
    sound_type: str
    asset_id: str


@dataclass(frozen=True)
class RenderInputs:
    """Everything one block needs: clips starting now, live sources in full."""

    block_index: int
    clips: tuple[ClipInput, ...]
    continuous: tuple[ContinuousInput, ...]
    drivetrain: tuple[DrivetrainInput, ...]


@dataclass(frozen=True)
class RenderParams:
    block_size: int
    sample_rate: int
    resolve: Callable[[str], np.ndarray]
    log_error: Callable[[str], None] = _log_error


@dataclass
class RenderState:
    cursor: int = 0
    clips: list[list[ScheduledClip]] = field(default_factory=lambda: [[] for _ in range(CHANNELS)])
    continuous: dict[tuple[str, int], ContinuousVoice] = field(default_factory=dict)
    procedural: dict[str, ProceduralArrayVoice] = field(default_factory=dict)


@dataclass
class RenderResult:
    raw: np.ndarray
    ped: np.ndarray
    motor: np.ndarray
    clipped_samples: int


def prepare_clip(
    mono: np.ndarray,
    received_volume_db: float,
    sensitivity_dbfs_at_94_dbspl: float,
    delay_samples: float,
) -> tuple[int, np.ndarray]:
    """Calibrate one arrival and split its delay, shared by the live and offline drivers."""
    calibrated = calibrate_mems(
        mono,
        received_volume_db,
        sensitivity_dbfs_at_94_dbspl=sensitivity_dbfs_at_94_dbspl,
    )
    return fractional_delay(calibrated, delay_samples)


def render_inputs_to_json(inputs: RenderInputs) -> str:
    """Serialize one block's inputs. Samples stay out: a clip is its scheduling parameters."""
    return json.dumps(
        {
            "block_index": inputs.block_index,
            "clips": [
                {
                    "channel": clip.channel,
                    "start": clip.start,
                    "asset_key": clip.asset_key,
                    "anchor": clip.anchor,
                    "received_volume_db": clip.received_volume_db,
                    "sensitivity_dbfs_at_94_dbspl": clip.sensitivity_dbfs_at_94_dbspl,
                    "delay_samples": clip.delay_samples,
                }
                for clip in inputs.clips
            ],
            "continuous": [asdict(source) for source in inputs.continuous],
            "drivetrain": [asdict(source) for source in inputs.drivetrain],
        },
        separators=(",", ":"),
    )


def render_inputs_from_json(text: str, resolve: Callable[[str], np.ndarray]) -> RenderInputs:
    """Rebuild one block's inputs, reconstructing every clip through ``prepare_clip``."""
    payload = json.loads(text)
    clips: list[ClipInput] = []
    for entry in payload["clips"]:
        delay, samples = prepare_clip(
            resolve(entry["asset_key"]),
            entry["received_volume_db"],
            entry["sensitivity_dbfs_at_94_dbspl"],
            entry["delay_samples"],
        )
        start = entry["anchor"] + delay
        if start != entry["start"]:
            raise ValueError(f"clip {entry['asset_key']!r} replays at sample {start}, trace recorded {entry['start']}")
        clips.append(
            ClipInput(
                channel=entry["channel"],
                start=start,
                asset_key=entry["asset_key"],
                samples=samples,
                anchor=entry["anchor"],
                received_volume_db=entry["received_volume_db"],
                sensitivity_dbfs_at_94_dbspl=entry["sensitivity_dbfs_at_94_dbspl"],
                delay_samples=entry["delay_samples"],
            )
        )
    drivetrain: list[DrivetrainInput] = []
    for entry in payload["drivetrain"]:
        drivetrain.append(
            DrivetrainInput(
                source_id=entry["source_id"],
                deterministic_seed=entry["deterministic_seed"],
                gains=tuple(entry["gains"]),
                delay_samples=tuple(entry["delay_samples"]),
                active_channels=tuple(entry["active_channels"]),
                left_velocity=entry["left_velocity"],
                right_velocity=entry["right_velocity"],
                active=entry["active"],
                tuning=DrivetrainTuning(**entry["tuning"]),
                source_agent_id=entry["source_agent_id"],
                source_agent_name=entry["source_agent_name"],
                sound_type=entry["sound_type"],
                asset_id=entry["asset_id"],
            )
        )
    return RenderInputs(
        block_index=payload["block_index"],
        clips=tuple(clips),
        continuous=tuple(ContinuousInput(**entry) for entry in payload["continuous"]),
        drivetrain=tuple(drivetrain),
    )


def _ingest_clips(state: RenderState, inputs: RenderInputs) -> None:
    for clip in inputs.clips:
        state.clips[clip.channel].append(ScheduledClip(clip.start, clip.samples))


def _reconcile_continuous(state: RenderState, inputs: RenderInputs, params: RenderParams) -> None:
    live: set[tuple[str, int]] = set()
    for source in inputs.continuous:
        key = (source.source_id, source.channel)
        live.add(key)
        voice = state.continuous.get(key)
        if voice is None:
            state.continuous[key] = ContinuousVoice(
                samples=params.resolve(source.asset_key),
                program_start=source.program_start,
                delay_samples=source.delay_target,
                delay_target=source.delay_target,
                gain=source.gain,
                loop=source.loop,
                active=True,
            )
            continue
        voice.gain = source.gain
        voice.delay_target = source.delay_target
        voice.active = True
    for key in tuple(state.continuous):
        if key not in live:
            state.continuous.pop(key)


def _reconcile_drivetrain(state: RenderState, inputs: RenderInputs, params: RenderParams) -> None:
    live: set[str] = set()
    for source in inputs.drivetrain:
        live.add(source.source_id)
        tuning = asdict(source.tuning)
        voice = state.procedural.get(source.source_id)
        if voice is None or voice.deterministic_seed != source.deterministic_seed:
            voice = ProceduralArrayVoice(
                source=DrivetrainRenderSource(
                    field_seed=source.deterministic_seed,
                    phase_index=source.deterministic_seed,
                    block_size=params.block_size,
                    channels=1,
                    sample_rate=params.sample_rate,
                    **tuning,
                ),
                deterministic_seed=source.deterministic_seed,
                gains=np.zeros(CHANNELS, dtype=np.float32),
                delay_samples=np.zeros(CHANNELS, dtype=np.float64),
                active_channels=np.zeros(CHANNELS, dtype=np.bool_),
                source_id=source.source_id,
                source_agent_id=source.source_agent_id,
                source_agent_name=source.source_agent_name,
                sound_type=source.sound_type,
                asset_id=source.asset_id,
            )
            state.procedural[source.source_id] = voice
        voice.gains = np.asarray(source.gains, dtype=np.float32)
        voice.delay_samples = np.asarray(source.delay_samples, dtype=np.float64)
        voice.active_channels = np.asarray(source.active_channels, dtype=np.bool_)
        voice.source.tune(**tuning)
        voice.source.update(
            left_velocity=source.left_velocity,
            right_velocity=source.right_velocity,
            gain_db=0.0,
            active=source.active,
            impulse=None,
            rir_signature=None,
        )
    for source_id in tuple(state.procedural):
        if source_id not in live:
            state.procedural.pop(source_id)


def render_block(state: RenderState, inputs: RenderInputs, params: RenderParams) -> RenderResult:
    """Render one block and advance the carried state by ``block_size`` samples."""
    _ingest_clips(state, inputs)
    _reconcile_continuous(state, inputs, params)
    _reconcile_drivetrain(state, inputs, params)

    output = np.zeros((CHANNELS, params.block_size), dtype=np.float32)
    motor = np.zeros((CHANNELS, params.block_size), dtype=np.float32)
    block_start, block_end = state.cursor, state.cursor + params.block_size
    for channel, clips in enumerate(state.clips):
        active: list[ScheduledClip] = []
        for clip in clips:
            clip_end = clip.start + len(clip.samples)
            overlap_start = max(block_start, clip.start)
            overlap_end = min(block_end, clip_end)
            if overlap_start < overlap_end:
                dst = slice(overlap_start - block_start, overlap_end - block_start)
                src = slice(overlap_start - clip.start, overlap_end - clip.start)
                output[channel, dst] += clip.samples[src]
            if clip_end > block_end:
                active.append(clip)
        state.clips[channel] = active

    for (_, channel), voice in tuple(state.continuous.items()):
        if not voice.active:
            continue
        output[channel] += (
            ramped_read(
                voice.samples,
                block_start - voice.program_start,
                voice.delay_samples,
                voice.delay_target,
                params.block_size,
                loop=voice.loop,
            )
            * voice.gain
        )
        voice.delay_samples = voice.delay_target
    for source_id, voice in tuple(state.procedural.items()):
        try:
            mono = voice.source.render(params.block_size)[:, 0]
        except Exception as exc:
            params.log_error(f"procedural drivetrain render failed for {source_id!r}: {exc}")
            state.procedural.pop(source_id, None)
            continue
        delayed, voice.history = streaming_fractional_delays(
            mono,
            voice.delay_samples,
            voice.history,
        )
        term = delayed * voice.gains[:, None]
        output += term
        motor += term
        if voice.source.finished:
            state.procedural.pop(source_id, None)
    state.cursor = block_end
    clipped = int(np.count_nonzero((output < -1.0) | (output > 1.0)))
    return RenderResult(
        raw=np.ascontiguousarray(np.clip(output, -1.0, 1.0)),
        ped=output - motor,
        motor=motor,
        clipped_samples=clipped,
    )
