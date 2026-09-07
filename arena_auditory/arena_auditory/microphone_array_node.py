"""Synchronized four-microphone Jackal PCM receiver and monitoring products.

The node is a waveform view of the existing propagation node.  It does not
calculate geometry or wall loss itself: every scheduled channel comes from a
listener-specific HeardSoundEvent / ContinuousHeardSoundState produced at the
physical microphone position by SoundPropagationNode.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rcl_interfaces.msg import SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RosTime
from rosgraph_msgs.msg import Clock as ClockMsg
from std_msgs.msg import ColorRGBA, Float32MultiArray, String
from task_generator_msgs.msg import (
    AudioFrame,
    ContinuousHeardSoundState,
    HeardSoundEvent,
    RenderedSoundActivity,
    RobotFleet,
)
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.asset_lib import (
    AcousticAssetCatalog,
    CachedSample,
    footstep_material_tags,
)
from arena_auditory.lockstep import register_hard_channel
from arena_auditory.procedural_audio import (
    DEFAULT_MOTOR_VOLUME_DB,
    DrivetrainRenderSource,
)
from arena_auditory.qos_profiles import (
    acoustic_metadata_qos,
    continuous_audio_qos,
    transient_event_qos,
)
from arena_auditory.render_clock import RenderCursor
from arena_auditory.spatial_audio import (
    CHANNEL_NAMES,
    apply_monitor_controls,
    calibrate_mems,
    dbfs_from_rms,
    fractional_delay,
    gcc_phat,
    headphone_stereo,
    hearing_waveform,
    interleave,
    monitor_amplify,
    ramped_read,
    rectangular_array,
    rms,
    streaming_fractional_delays,
)


@dataclass(slots=True)
class ScheduledClip:
    start: int
    samples: np.ndarray


@dataclass(slots=True)
class EventLoad:
    future: Future[CachedSample]
    messages: dict[int, HeardSoundEvent]
    scheduled_channels: set[int]
    anchor: int
    updated_at: float


@dataclass(slots=True)
class SemanticEventGroup:
    messages: dict[int, HeardSoundEvent]
    updated_at: float


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


@dataclass(slots=True)
class ProceduralLoad:
    future: Future[DrivetrainRenderSource]
    deterministic_seed: int
    messages: dict[int, ContinuousHeardSoundState]


class MicrophoneArrayNode(Node):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("microphone_array_node", **kwargs)
        share = Path(get_package_share_directory("arena_auditory"))
        self.declare_parameter("robot_name", "")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("block_size", 320)
        self.declare_parameter("max_catchup_blocks", 10)
        self.declare_parameter("mic_array_width", 0.310)
        self.declare_parameter("mic_array_length", 0.420)
        self.declare_parameter("mic_height", 0.220)
        self.declare_parameter("mic_corner_inset", 0.020)
        self.declare_parameter("speed_of_sound", 343.0)
        self.declare_parameter("sensitivity_dbfs_at_94_dbspl", -26.0)
        self.declare_parameter("asset_catalog", str(share / "config" / "acoustic_assets.yaml"))
        self.declare_parameter("sound_dir", str(share / "sounds"))
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
        self.declare_parameter(
            "fused_heard_sound_events_topic",
            "four_mic_heard_sound_events",
        )
        self.declare_parameter("semantic_group_timeout_sec", 1.0)
        self.declare_parameter("continuous_heard_sounds_topic", "continuous_heard_sounds")
        self.declare_parameter("robot_fleet_topic", "state/robots")
        self.declare_parameter("microphone_marker_topic", "microphone_markers")
        self.declare_parameter("enabled", True)
        self.declare_parameter("headphones_enabled", True)
        self.declare_parameter("mute_all", False)
        self.declare_parameter("master_gain", 0.8)
        self.declare_parameter("monitor_gain_db", 36.0)
        self.declare_parameter("monitor_limit", 0.98)
        self.declare_parameter("headphone_front_gain", 1.0)
        self.declare_parameter("headphone_rear_gain", 0.75)
        self.declare_parameter("solo_channel", "")
        self.declare_parameter("monitor_mode", "headphones")
        self.declare_parameter("visualization_enabled", True)
        self.declare_parameter("tdoa_enabled", True)
        self.declare_parameter("max_tdoa_seconds", 0.002)
        self.declare_parameter("audio_device", "none")
        self.declare_parameter("audio_retry_period_sec", 2.0)
        self.declare_parameter("audio_diagnostics_period_sec", 5.0)
        self.declare_parameter("motor_volume_db", DEFAULT_MOTOR_VOLUME_DB)
        self.declare_parameter("motor_enabled", True)
        self.declare_parameter("motor_mems_calibration_db", -40.0)
        self.declare_parameter("motor_frequency_scale", 1.0)
        self.declare_parameter("motor_tonal_gain_db", 0.0)
        self.declare_parameter("motor_broadband_gain_db", -12.0)
        self.declare_parameter("motor_speed_exponent", 1.5)
        self.declare_parameter("motor_velocity_smoothing_sec", 0.015)

        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.block_size = int(self.get_parameter("block_size").value)
        if self.sample_rate <= 0 or self.block_size <= 0:
            raise ValueError("sample_rate and block_size must be positive")
        self.microphones = rectangular_array(
            width_m=float(self.get_parameter("mic_array_width").value),
            length_m=float(self.get_parameter("mic_array_length").value),
            height_m=float(self.get_parameter("mic_height").value),
            corner_inset_m=float(self.get_parameter("mic_corner_inset").value),
        )
        self._catalog = AcousticAssetCatalog(
            self.get_parameter("asset_catalog").value,
            self.get_parameter("sound_dir").value,
            output_sample_rate=self.sample_rate,
            output_channels=1,
        )
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="array_audio_loader")
        self._procedural_loader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="array_drivetrain_loader",
        )
        self._robot_name = str(self.get_parameter("robot_name").value).strip()
        self._robot_frame_prefix = ""
        self._publishers_ready = False
        self._cursor = 0
        self._stream_start_ns: int | None = None
        self._render = RenderCursor(
            block_ns=round(self.block_size * 1_000_000_000 / self.sample_rate),
            max_catchup=max(int(self.get_parameter("max_catchup_blocks").value), 1),
        )
        self._reported_skipped = 0
        self._render_behind = False
        self._clips: list[list[ScheduledClip]] = [[] for _ in CHANNEL_NAMES]
        self._event_loads: dict[str, EventLoad] = {}
        self._semantic_event_groups: dict[str, SemanticEventGroup] = {}
        self._fused_events = 0
        self._incomplete_semantic_events = 0
        self._continuous: dict[tuple[str, int], ContinuousVoice] = {}
        self._procedural: dict[str, ProceduralArrayVoice] = {}
        self._procedural_pending: dict[str, ProceduralLoad] = {}
        self._reported_procedural_activity: dict[tuple[str, int], bool] = {}
        self._continuous_pending: dict[tuple[str, int], tuple[Future[CachedSample], ContinuousHeardSoundState]] = {}
        self._last_levels = np.zeros(7, dtype=np.float32)
        self._output_lock = threading.Lock()
        self._output_blocks: deque[np.ndarray] = deque(maxlen=8)
        self._output_current: np.ndarray | None = None
        self._output_current_offset = 0
        self._stream = None
        self._audio_callbacks = 0
        self._audio_underflows = 0
        self._audio_overflows = 0
        self._reported_underflows = 0
        self._reported_overflows = 0
        self._playback_degraded = False
        self._audio_peak = 0.0
        self._audio_status = ""
        self._stream_error = ""
        self._heard_events = 0
        self._accepted_events = 0
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

        self.create_subscription(
            RobotFleet,
            self.get_parameter("robot_fleet_topic").value,
            self._on_fleet,
            acoustic_metadata_qos(),
        )
        self.create_subscription(
            HeardSoundEvent,
            self.get_parameter("heard_sound_events_topic").value,
            self._on_heard_event,
            transient_event_qos(),
        )
        self._fused_heard_pub = self.create_publisher(
            HeardSoundEvent,
            self.get_parameter("fused_heard_sound_events_topic").value,
            transient_event_qos(),
        )
        self.create_subscription(
            ContinuousHeardSoundState,
            self.get_parameter("continuous_heard_sounds_topic").value,
            self._on_continuous,
            continuous_audio_qos(),
        )
        self._marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("microphone_marker_topic").value,
            acoustic_metadata_qos(),
        )
        self.add_on_set_parameters_callback(self._on_parameters)
        if bool(self.get_parameter("use_sim_time").value):
            # Render off /clock, not an rcl timer: a lockstep gate holding the
            # clock for this node's block would never fire the timer.
            self.create_subscription(ClockMsg, "/clock", self._on_clock, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        else:
            self.create_timer(self.block_size / self.sample_rate, self._publish_block, clock=self._steady_clock)
        self.create_timer(0.25, self._publish_markers)
        self.create_timer(
            max(float(self.get_parameter("audio_retry_period_sec").value), 0.25),
            self._retry_audio_device,
            clock=self._steady_clock,
        )
        self.create_timer(
            max(float(self.get_parameter("audio_diagnostics_period_sec").value), 1.0),
            self._publish_audio_diagnostics,
            clock=self._steady_clock,
        )
        self.get_logger().info(f"four-microphone raw array ready: {self.sample_rate} Hz, {self.block_size} frames; channel order FL,FR,RL,RR")

    def _on_fleet(self, msg: RobotFleet) -> None:
        if not self._robot_name:
            selected = next(
                (state.descriptor for state in msg.robots if str(state.descriptor.model).lower() == "jackal"),
                None,
            )
            if selected is None:
                return
            self._robot_name = str(selected.name).strip()
            self._robot_frame_prefix = str(selected.frame).strip("/")
        else:
            selected = next(
                (state.descriptor for state in msg.robots if str(state.descriptor.name) == self._robot_name),
                None,
            )
            if selected is not None:
                self._robot_frame_prefix = str(selected.frame).strip("/")
        if self._robot_name and not self._publishers_ready:
            self._create_output_publishers()

    def _create_output_publishers(self) -> None:
        prefix = f"{self._robot_name}/audio"
        self._raw_pub = self.create_publisher(AudioFrame, f"{prefix}/raw_array", 10)
        self._channel_pubs = [self.create_publisher(AudioFrame, f"{prefix}/mic_{name}", 10) for name in CHANNEL_NAMES]
        self._hearing_pub = self.create_publisher(AudioFrame, f"{prefix}/hearing/mono", 10)
        self._energy_pub = self.create_publisher(Float32MultiArray, f"{prefix}/hearing/energy", 10)
        self._headphone_left_pub = self.create_publisher(AudioFrame, f"{prefix}/headphones/left", 10)
        self._headphone_right_pub = self.create_publisher(AudioFrame, f"{prefix}/headphones/right", 10)
        self._headphone_pub = self.create_publisher(AudioFrame, f"{prefix}/headphones/stereo", 10)
        self._tdoa_pub = self.create_publisher(String, f"{prefix}/diagnostics/tdoa", 10)
        self._activity_pub = self.create_publisher(
            RenderedSoundActivity,
            f"{prefix}/rendered_sound_activity",
            transient_event_qos(),
        )
        self._publishers_ready = True
        # Audio sample zero is anchored once, on the first /clock under sim
        # time. Every later block stamp is derived from the sample cursor, so
        # timer jitter cannot desynchronize audio from robot/pedestrian poses.
        if not bool(self.get_parameter("use_sim_time").value):
            self._stream_start_ns = self._steady_clock.now().nanoseconds
        if str(self.get_parameter("audio_device").value).strip() not in {"", "none"}:
            with self._output_lock:
                for _ in range(2):
                    self._output_blocks.append(np.zeros((self.block_size, 2), dtype=np.float32))
        self._open_audio_device()
        if bool(self.get_parameter("use_sim_time").value):
            register_hard_channel(
                self,
                name=f"audio/{self._robot_name}",
                topic=self._raw_pub.topic_name,
                msg_type="task_generator_msgs/msg/AudioFrame",
                period_s=self.block_size / self.sample_rate,
                env=self.get_namespace(),
            )

    def _on_clock(self, msg: ClockMsg) -> None:
        if not self._publishers_ready:
            return
        now_ns = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec
        if self._stream_start_ns is None:
            self._stream_start_ns = now_ns
            self._render.start_ns = now_ns
        render, skip = self._render.owed(now_ns)
        for _ in range(render):
            self._publish_block()
        if skip:
            self._cursor += skip * self.block_size

    def _listener_channel(self, listener_id: str) -> int | None:
        if not self._robot_name:
            for index, name in enumerate(CHANNEL_NAMES):
                suffix = f"_mic_{name}"
                if listener_id.endswith(suffix):
                    self._robot_name = listener_id[: -len(suffix)]
                    self._create_output_publishers()
                    return index
            return None
        for index, name in enumerate(CHANNEL_NAMES):
            if listener_id == f"{self._robot_name}_mic_{name}":
                return index
        return None

    def _on_heard_event(self, msg: HeardSoundEvent) -> None:
        self._heard_events += 1
        channel = self._listener_channel(str(msg.listener_id))
        if channel is None:
            return
        event_id = self._event_key(msg)
        self._collect_semantic_event(event_id, channel, msg)
        if not msg.audible:
            return
        self._accepted_events += 1
        asset_id = str(msg.asset_id).strip() or str(msg.sound_type).strip()
        required_tags = footstep_material_tags(msg.semantic_tags) if asset_id == "footstep" else frozenset()
        selected = self._catalog.select(
            asset_id,
            episode_seed=0,
            agent_id=int(msg.source_agent_id),
            occurrence=self._stable_occurrence(str(msg.event_id)),
            required_tags=required_tags,
        )
        if selected is None:
            self.get_logger().warning(f"no raw-array acoustic asset {asset_id!r}")
            return
        load = self._event_loads.get(event_id)
        if load is None:
            _, spec = selected
            load = EventLoad(
                future=self._loader.submit(self._catalog.load, spec),
                messages={},
                scheduled_channels=set(),
                anchor=self._cursor + 4 * self.block_size,
                updated_at=time.monotonic(),
            )
            self._event_loads[event_id] = load
        load.messages[channel] = msg
        load.updated_at = time.monotonic()

    @staticmethod
    def _event_key(msg: HeardSoundEvent) -> str:
        return str(msg.event_id).strip() or (f"{msg.source_agent_id}:{msg.asset_id or msg.sound_type}:{msg.header.stamp.sec}:{msg.header.stamp.nanosec}")

    def _collect_semantic_event(
        self,
        event_id: str,
        channel: int,
        msg: HeardSoundEvent,
    ) -> None:
        group = self._semantic_event_groups.get(event_id)
        if group is None:
            group = SemanticEventGroup(messages={}, updated_at=time.monotonic())
            self._semantic_event_groups[event_id] = group
        group.messages[channel] = msg
        group.updated_at = time.monotonic()
        if len(group.messages) == len(CHANNEL_NAMES):
            self._publish_fused_semantic_event(event_id, group)

    def _publish_fused_semantic_event(
        self,
        event_id: str,
        group: SemanticEventGroup,
    ) -> None:
        self._semantic_event_groups.pop(event_id, None)
        if len(group.messages) != len(CHANNEL_NAMES) or not self._robot_name:
            self._incomplete_semantic_events += 1
            return

        messages = tuple(group.messages[index] for index in range(len(CHANNEL_NAMES)))
        audible = tuple(message for message in messages if message.audible)
        candidates = audible or messages
        best = max(candidates, key=lambda message: float(message.received_volume_db))
        fused = copy.deepcopy(best)
        fused.listener_id = f"robot:{self._robot_name}"
        fused.listener_position.x = sum(float(message.listener_position.x) for message in messages) / len(messages)
        fused.listener_position.y = sum(float(message.listener_position.y) for message in messages) / len(messages)
        fused.listener_position.z = sum(float(message.listener_position.z) for message in messages) / len(messages)
        dx = float(fused.source_position.x - fused.listener_position.x)
        dy = float(fused.source_position.y - fused.listener_position.y)
        dz = float(fused.source_position.z - fused.listener_position.z)
        fused.distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        fused.bearing_rad = math.atan2(dy, dx)
        fused.received_volume_db = float(best.received_volume_db)
        fused.hearing_threshold_db = float(best.hearing_threshold_db)
        fused.direct_delay_sec = min(float(message.direct_delay_sec) for message in candidates)
        fused.audible = bool(audible) and bool(self.get_parameter("enabled").value) and not bool(self.get_parameter("mute_all").value)
        fused.occluded = bool(audible) and all(message.occluded for message in audible)
        self._fused_heard_pub.publish(fused)
        self._fused_events += 1

    def _on_continuous(self, msg: ContinuousHeardSoundState) -> None:
        channel = self._listener_channel(str(msg.listener_id))
        if channel is None:
            return
        if msg.source_backend == "drivetrain":
            self._on_drivetrain(msg, channel)
            return
        if msg.source_backend != "wav_loop":
            return
        key = (str(msg.source_id), channel)
        if not msg.active or not msg.audible:
            self._continuous.pop(key, None)
            pending = self._continuous_pending.pop(key, None)
            if pending is not None:
                pending[0].cancel()
            return
        voice = self._continuous.get(key)
        if voice is not None:
            voice.gain = self._samples_spl_gain(
                float(msg.received_volume_db),
                voice.samples,
            )
            voice.delay_target = self._delay_samples(msg)
            voice.active = True
            return
        if key in self._continuous_pending:
            future, _ = self._continuous_pending[key]
            self._continuous_pending[key] = (future, msg)
            return
        asset_id = str(msg.asset_id).strip() or str(msg.sound_type).strip()
        selected = self._catalog.select(asset_id, episode_seed=0, agent_id=int(msg.source_agent_id), occurrence=0)
        if selected is None:
            return
        _, spec = selected
        self._continuous_pending[key] = (self._loader.submit(self._catalog.load, spec), msg)

    def _on_drivetrain(
        self,
        msg: ContinuousHeardSoundState,
        channel: int,
    ) -> None:
        source_id = str(msg.source_id)
        seed = int(msg.deterministic_seed)
        voice = self._procedural.get(source_id)
        if voice is not None and voice.deterministic_seed == seed:
            self._apply_drivetrain_message(voice, msg, channel)
            return
        if voice is not None:
            self._procedural.pop(source_id, None)

        pending = self._procedural_pending.get(source_id)
        if pending is not None and pending.deterministic_seed != seed:
            pending.future.cancel()
            self._procedural_pending.pop(source_id, None)
            pending = None
        if pending is None:
            if not msg.active:
                return
            pending = ProceduralLoad(
                future=self._procedural_loader.submit(
                    self._make_drivetrain_source,
                    seed,
                    self._motor_tuning(),
                ),
                deterministic_seed=seed,
                messages={},
            )
            self._procedural_pending[source_id] = pending
        pending.messages[channel] = msg

    def _make_drivetrain_source(
        self,
        seed: int,
        tuning: dict[str, float],
    ) -> DrivetrainRenderSource:
        return DrivetrainRenderSource(
            field_seed=seed,
            phase_index=seed,
            block_size=self.block_size,
            channels=1,
            sample_rate=self.sample_rate,
            **tuning,
        )

    def _apply_drivetrain_message(
        self,
        voice: ProceduralArrayVoice,
        msg: ContinuousHeardSoundState,
        channel: int,
    ) -> None:
        active = bool(msg.active and msg.audible and self.get_parameter("motor_enabled").value)
        voice.active_channels[channel] = active
        propagation_gain_db = float(msg.received_volume_db) - float(msg.source_volume_db) if active else 0.0
        voice.gains[channel] = 10.0 ** (propagation_gain_db / 20.0) if active else 0.0
        direct_delay = float(msg.direct_delay_sec) * self.sample_rate
        voice.delay_samples[channel] = direct_delay if math.isfinite(direct_delay) and direct_delay >= 0.0 else 0.0
        voice.source.update(
            left_velocity=float(msg.left_velocity_mps),
            right_velocity=float(msg.right_velocity_mps),
            gain_db=0.0,
            active=bool(np.any(voice.active_channels)),
            impulse=None,
            rir_signature=None,
        )

    def _motor_tuning(self) -> dict[str, float]:
        return {
            "volume_db": (float(self.get_parameter("motor_volume_db").value) + float(self.get_parameter("motor_mems_calibration_db").value)),
            "frequency_scale": float(self.get_parameter("motor_frequency_scale").value),
            "tonal_gain_db": float(self.get_parameter("motor_tonal_gain_db").value),
            "broadband_gain_db": float(self.get_parameter("motor_broadband_gain_db").value),
            "speed_exponent": float(self.get_parameter("motor_speed_exponent").value),
            "velocity_smoothing_seconds": float(self.get_parameter("motor_velocity_smoothing_sec").value),
        }

    def _poll_loads(self) -> None:
        now = time.monotonic()
        semantic_timeout = max(
            float(self.get_parameter("semantic_group_timeout_sec").value),
            0.1,
        )
        for event_id, group in tuple(self._semantic_event_groups.items()):
            if now - group.updated_at <= semantic_timeout:
                continue
            self._semantic_event_groups.pop(event_id, None)
            self._incomplete_semantic_events += 1
            self.get_logger().warning(f"discarding incomplete four-mic semantic event {event_id!r}: received {len(group.messages)}/{len(CHANNEL_NAMES)} channels")
        for event_id, load in tuple(self._event_loads.items()):
            if not load.future.done():
                continue
            try:
                sample = load.future.result()
            except Exception as exc:
                self.get_logger().error(f"raw-array asset decode failed: {exc}")
                self._event_loads.pop(event_id, None)
                continue
            if not load.scheduled_channels:
                # A cold decode must not consume the common pre-roll. Move the
                # unscheduled event as a unit, retaining all physical deltas.
                load.anchor = max(
                    load.anchor,
                    self._cursor + 2 * self.block_size,
                )
            for channel, msg in load.messages.items():
                if channel in load.scheduled_channels:
                    continue
                mono = sample.samples[:, 0]
                calibrated = calibrate_mems(
                    mono,
                    float(msg.received_volume_db),
                    sensitivity_dbfs_at_94_dbspl=float(self.get_parameter("sensitivity_dbfs_at_94_dbspl").value),
                )
                delay, delayed = fractional_delay(
                    calibrated,
                    float(msg.direct_delay_sec) * self.sample_rate,
                )
                self._clips[channel].append(ScheduledClip(load.anchor + delay, delayed))
                self._publish_discrete_activity(
                    event_id,
                    channel,
                    msg,
                    load.anchor + delay,
                    load.anchor + delay + len(delayed),
                )
                load.scheduled_channels.add(channel)
            if len(load.scheduled_channels) == 4 or now - load.updated_at > 1.0:
                self._event_loads.pop(event_id, None)

        for key, pending in tuple(self._continuous_pending.items()):
            future, msg = pending
            if not future.done():
                continue
            self._continuous_pending.pop(key, None)
            if future.cancelled() or not msg.active:
                continue
            try:
                sample = future.result()
            except Exception as exc:
                self.get_logger().error(f"continuous raw-array asset decode failed: {exc}")
                continue
            delay_samples = self._delay_samples(msg)
            mono = np.asarray(sample.samples[:, 0], dtype=np.float32)
            elapsed = max(
                self.get_clock().now().nanoseconds - (int(msg.program_start_time.sec) * 1_000_000_000 + int(msg.program_start_time.nanosec)),
                0,
            )
            self._continuous[key] = ContinuousVoice(
                samples=np.ascontiguousarray(mono),
                program_start=self._cursor - round(elapsed * self.sample_rate / 1e9),
                delay_samples=delay_samples,
                delay_target=delay_samples,
                gain=self._spl_gain(float(msg.received_volume_db), sample),
                loop=bool(msg.loop),
                active=True,
            )

        for source_id, pending in tuple(self._procedural_pending.items()):
            if not pending.future.done():
                continue
            self._procedural_pending.pop(source_id, None)
            if pending.future.cancelled():
                continue
            try:
                source = pending.future.result()
            except Exception as exc:
                self.get_logger().error(f"procedural drivetrain initialization failed: {exc}")
                continue
            source.tune(**self._motor_tuning())
            voice = ProceduralArrayVoice(
                source=source,
                deterministic_seed=pending.deterministic_seed,
                gains=np.zeros(4, dtype=np.float32),
                delay_samples=np.zeros(4, dtype=np.float64),
                active_channels=np.zeros(4, dtype=np.bool_),
                source_id=source_id,
                source_agent_id=int(next(iter(pending.messages.values())).source_agent_id),
                source_agent_name=str(next(iter(pending.messages.values())).source_agent_name),
                sound_type=str(next(iter(pending.messages.values())).sound_type),
                asset_id=str(next(iter(pending.messages.values())).asset_id),
            )
            self._procedural[source_id] = voice
            for channel, msg in pending.messages.items():
                self._apply_drivetrain_message(voice, msg, channel)

    def _delay_samples(self, msg: ContinuousHeardSoundState) -> float:
        delay = float(msg.direct_delay_sec) * self.sample_rate
        return delay if math.isfinite(delay) and delay > 0.0 else 0.0

    def _spl_gain(self, received_spl_db: float, sample: CachedSample) -> float:
        return self._samples_spl_gain(received_spl_db, sample.samples[:, 0])

    def _samples_spl_gain(
        self,
        received_spl_db: float,
        samples: np.ndarray,
    ) -> float:
        mono_rms = float(rms(samples))
        if mono_rms <= 1e-12:
            return 0.0
        target_dbfs = received_spl_db - 94.0 + float(self.get_parameter("sensitivity_dbfs_at_94_dbspl").value)
        return (10.0 ** (target_dbfs / 20.0)) / mono_rms

    def _render_raw(self) -> np.ndarray:
        output = np.zeros((4, self.block_size), dtype=np.float32)
        block_start, block_end = self._cursor, self._cursor + self.block_size
        for channel, clips in enumerate(self._clips):
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
            self._clips[channel] = active

        for (_, channel), voice in tuple(self._continuous.items()):
            if not voice.active:
                continue
            output[channel] += (
                ramped_read(
                    voice.samples,
                    block_start - voice.program_start,
                    voice.delay_samples,
                    voice.delay_target,
                    self.block_size,
                    loop=voice.loop,
                )
                * voice.gain
            )
            voice.delay_samples = voice.delay_target
        for source_id, voice in tuple(self._procedural.items()):
            try:
                mono = voice.source.render(self.block_size)[:, 0]
            except Exception as exc:
                self.get_logger().error(f"procedural drivetrain render failed for {source_id!r}: {exc}")
                self._procedural.pop(source_id, None)
                continue
            delayed, voice.history = streaming_fractional_delays(
                mono,
                voice.delay_samples,
                voice.history,
            )
            output += delayed * voice.gains[:, None]
            if voice.source.finished:
                self._procedural.pop(source_id, None)
        self._cursor = block_end
        return np.ascontiguousarray(np.clip(output, -1.0, 1.0))

    def _publish_block(self) -> None:
        if not self._publishers_ready:
            return
        block_start = self._cursor
        self._poll_loads()
        self._publish_procedural_activity_transitions(block_start)
        raw = self._render_raw()
        enabled = bool(self.get_parameter("enabled").value)
        muted = bool(self.get_parameter("mute_all").value)
        solo = str(self.get_parameter("solo_channel").value).strip().lower()
        raw = apply_monitor_controls(raw, enabled=enabled, muted=muted)
        monitor = apply_monitor_controls(raw, solo_channel=solo)
        hearing = hearing_waveform(monitor)
        master_gain = float(self.get_parameter("master_gain").value)
        stereo = headphone_stereo(
            monitor,
            front_gain=float(self.get_parameter("headphone_front_gain").value),
            rear_gain=float(self.get_parameter("headphone_rear_gain").value),
            output_gain=master_gain,
        )
        monitor_gain_db = float(self.get_parameter("monitor_gain_db").value)
        monitor_limit = float(self.get_parameter("monitor_limit").value)
        stereo = monitor_amplify(
            stereo,
            gain_db=monitor_gain_db,
            limit=monitor_limit,
        )
        if not bool(self.get_parameter("headphones_enabled").value):
            stereo.fill(0.0)
        # AudioFrame.header.stamp is the simulation time of this block's first
        # sample. Sample i is therefore stamp + i / sample_rate.
        stamp_ns = self._stream_start_ns + round(block_start * 1_000_000_000 / self.sample_rate)
        stamp = RosTime(nanoseconds=stamp_ns).to_msg()
        self._raw_pub.publish(self._audio_frame(raw, stamp, CHANNEL_NAMES))
        for index, publisher in enumerate(self._channel_pubs):
            publisher.publish(self._audio_frame(raw[index : index + 1], stamp, (CHANNEL_NAMES[index],)))
        self._hearing_pub.publish(
            self._audio_frame(
                hearing[None, :],
                stamp,
                ("hearing",),
                spatial=False,
            )
        )
        self._headphone_left_pub.publish(self._audio_frame(stereo[0:1], stamp, ("left",), spatial=False))
        self._headphone_right_pub.publish(self._audio_frame(stereo[1:2], stamp, ("right",), spatial=False))
        self._headphone_pub.publish(self._audio_frame(stereo, stamp, ("left", "right"), spatial=False))
        levels = np.concatenate((np.asarray(rms(raw, axis=1)), [rms(hearing), rms(stereo[0]), rms(stereo[1])]))
        self._last_levels = levels.astype(np.float32)
        self._energy_pub.publish(Float32MultiArray(data=self._last_levels.tolist()))
        if bool(self.get_parameter("tdoa_enabled").value) and self._tdoa_pub.get_subscription_count() > 0:
            self._publish_tdoa(raw, stamp)
        playback = stereo.T
        if str(self.get_parameter("monitor_mode").value) == "hearing":
            playback = np.repeat(
                monitor_amplify(
                    hearing * master_gain,
                    gain_db=monitor_gain_db,
                    limit=monitor_limit,
                )[:, None],
                2,
                axis=1,
            )
            if not bool(self.get_parameter("headphones_enabled").value):
                playback.fill(0.0)
        if str(self.get_parameter("audio_device").value).strip() not in {"", "none"}:
            with self._output_lock:
                if len(self._output_blocks) == self._output_blocks.maxlen:
                    self._audio_overflows += 1
                self._output_blocks.append(playback.copy())

    def _sample_time(self, sample_index: int) -> object:
        if self._stream_start_ns is None:
            raise RuntimeError("audio sample clock is not initialized")
        return RosTime(nanoseconds=self._stream_start_ns + round(sample_index * 1_000_000_000 / self.sample_rate)).to_msg()

    def _publish_discrete_activity(
        self,
        event_id: str,
        channel: int,
        source: HeardSoundEvent,
        start_sample: int,
        end_sample: int,
    ) -> None:
        msg = RenderedSoundActivity()
        msg.header.stamp = self._sample_time(start_sample)
        msg.header.frame_id = self._base_frame()
        msg.stream_id = f"{self._robot_name}/audio/raw_array"
        msg.event_id = event_id
        msg.source_id = str(source.source_agent_name) or str(source.source_agent_id)
        msg.source_agent_id = int(source.source_agent_id)
        msg.source_agent_name = str(source.source_agent_name)
        msg.source_type = "robot" if str(source.sound_type).strip().lower() == "motor" else "pedestrian" if int(source.source_agent_id) >= 0 else "unknown"
        msg.sound_type = str(source.sound_type)
        msg.asset_id = str(source.asset_id)
        msg.channel_name = CHANNEL_NAMES[channel]
        msg.continuous = False
        msg.active = True
        msg.start_sample_index = start_sample
        msg.end_sample_index = end_sample
        msg.start_time = self._sample_time(start_sample)
        msg.end_time = self._sample_time(end_sample)
        self._activity_pub.publish(msg)

    def _publish_procedural_activity_transitions(self, sample_index: int) -> None:
        current: dict[tuple[str, int], bool] = {}
        voices = dict(self._procedural)
        for source_id, voice in voices.items():
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                key = (source_id, channel)
                active = bool(voice.active_channels[channel])
                current[key] = active
                if self._reported_procedural_activity.get(key, False) == active:
                    continue
                msg = RenderedSoundActivity()
                msg.header.stamp = self._sample_time(sample_index)
                msg.header.frame_id = self._base_frame()
                msg.stream_id = f"{self._robot_name}/audio/raw_array"
                msg.event_id = f"continuous:{source_id}"
                msg.source_id = source_id
                msg.source_agent_id = voice.source_agent_id
                msg.source_agent_name = voice.source_agent_name
                msg.source_type = "robot"
                msg.sound_type = voice.sound_type or "motor"
                msg.asset_id = voice.asset_id
                msg.channel_name = channel_name
                msg.continuous = True
                msg.active = active
                msg.start_sample_index = sample_index
                msg.end_sample_index = sample_index
                msg.start_time = self._sample_time(sample_index)
                msg.end_time = msg.start_time
                self._activity_pub.publish(msg)
        for key, was_active in tuple(self._reported_procedural_activity.items()):
            if was_active and key not in current:
                source_id, channel = key
                # A finished voice still needs an explicit closing transition.
                voice = voices.get(source_id)
                msg = RenderedSoundActivity()
                msg.header.stamp = self._sample_time(sample_index)
                msg.header.frame_id = self._base_frame()
                msg.stream_id = f"{self._robot_name}/audio/raw_array"
                msg.event_id = f"continuous:{source_id}"
                msg.source_id = source_id
                msg.source_agent_id = voice.source_agent_id if voice else -1
                msg.source_agent_name = voice.source_agent_name if voice else ""
                msg.source_type = "robot"
                msg.sound_type = voice.sound_type if voice else "motor"
                msg.asset_id = voice.asset_id if voice else ""
                msg.channel_name = CHANNEL_NAMES[channel]
                msg.continuous = True
                msg.active = False
                msg.start_sample_index = sample_index
                msg.end_sample_index = sample_index
                msg.start_time = self._sample_time(sample_index)
                msg.end_time = msg.start_time
                self._activity_pub.publish(msg)
                current[key] = False
        self._reported_procedural_activity = current

    def _audio_frame(self, audio: np.ndarray, stamp: object, names: tuple[str, ...], *, spatial: bool = True) -> AudioFrame:
        msg = AudioFrame()
        msg.header.stamp = stamp
        msg.header.frame_id = self._base_frame()
        msg.sample_rate = self.sample_rate
        msg.channel_count = audio.shape[0]
        msg.frame_count = audio.shape[1]
        msg.encoding = "32FC1"
        msg.interleaved = True
        msg.channel_names = list(names)
        if spatial:
            selected = [self.microphones[CHANNEL_NAMES.index(name)] for name in names]
            msg.frame_ids = [self._frame_id(mic.name) for mic in selected]
            msg.microphone_positions = [Point(x=mic.position_m[0], y=mic.position_m[1], z=mic.position_m[2]) for mic in selected]
            msg.microphone_yaw_rad = [mic.yaw_rad for mic in selected]
        msg.data = interleave(audio)
        return msg

    def _publish_tdoa(self, raw: np.ndarray, stamp: object) -> None:
        pairs = ((0, 1, "FL-FR"), (2, 3, "RL-RR"), (0, 2, "FL-RL"), (1, 3, "FR-RR"))
        estimates = {}
        for first, second, label in pairs:
            delay, confidence = gcc_phat(
                raw[first],
                raw[second],
                sample_rate_hz=self.sample_rate,
                max_tau_seconds=float(self.get_parameter("max_tdoa_seconds").value),
            )
            estimates[label] = {"delay_us": delay * 1e6, "confidence": confidence}
        estimates["stamp"] = {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}
        self._tdoa_pub.publish(String(data=json.dumps(estimates, separators=(",", ":"))))

    def _publish_markers(self) -> None:
        if not self._publishers_ready or not bool(self.get_parameter("visualization_enabled").value):
            return
        stamp = self.get_clock().now().to_msg()
        markers: list[Marker] = []
        for index, mic in enumerate(self.microphones):
            level_dbfs = dbfs_from_rms(float(self._last_levels[index]))
            active = bool(self.get_parameter("enabled").value) and not bool(self.get_parameter("mute_all").value)
            color = ColorRGBA(r=0.1, g=0.9 if active else 0.25, b=0.3, a=0.95)
            body = Marker()
            body.header.frame_id = self._base_frame()
            body.header.stamp = stamp
            body.ns = "jackal_four_mic_array"
            body.id = index * 3
            body.type = Marker.SPHERE
            body.action = Marker.ADD
            body.pose.position = Point(x=mic.position_m[0], y=mic.position_m[1], z=mic.position_m[2])
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = body.scale.z = 0.045
            body.color = color
            body.lifetime.sec = 1
            arrow = Marker()
            arrow.header = body.header
            arrow.ns = "jackal_microphone_inlet_normals"
            arrow.id = index * 3 + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points = [
                body.pose.position,
                Point(
                    x=mic.position_m[0] + 0.18 * math.cos(mic.yaw_rad),
                    y=mic.position_m[1] + 0.18 * math.sin(mic.yaw_rad),
                    z=mic.position_m[2],
                ),
            ]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.018, 0.035, 0.05
            arrow.color = color
            arrow.lifetime = body.lifetime
            label = Marker()
            label.header = body.header
            label.ns = "jackal_microphone_levels"
            label.id = index * 3 + 2
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = Point(x=mic.position_m[0], y=mic.position_m[1], z=mic.position_m[2] + 0.10)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.075
            label.color = ColorRGBA(r=0.95, g=0.95, b=0.95, a=1.0)
            label.text = f"{mic.name}\n{level_dbfs:.1f} dBFS | {'ON' if active else 'OFF'}"
            label.lifetime = body.lifetime
            markers.extend((body, arrow, label))
        self._marker_pub.publish(MarkerArray(markers=markers))

    def _on_parameters(self, parameters: list[Parameter]) -> SetParametersResult:
        tuning_names = {
            "motor_volume_db",
            "motor_mems_calibration_db",
            "motor_frequency_scale",
            "motor_tonal_gain_db",
            "motor_broadband_gain_db",
            "motor_speed_exponent",
            "motor_velocity_smoothing_sec",
        }
        for parameter in parameters:
            if parameter.name in {"enabled", "headphones_enabled", "mute_all", "visualization_enabled", "tdoa_enabled", "motor_enabled"}:
                if parameter.type_ != Parameter.Type.BOOL:
                    return SetParametersResult(successful=False, reason=f"{parameter.name} must be boolean")
            elif parameter.name in {"master_gain", "headphone_front_gain", "headphone_rear_gain"}:
                if parameter.type_ not in {Parameter.Type.DOUBLE, Parameter.Type.INTEGER} or not 0.0 <= float(parameter.value) <= 4.0:
                    return SetParametersResult(successful=False, reason=f"{parameter.name} must be in [0,4]")
            elif parameter.name == "monitor_gain_db":
                if parameter.type_ not in {Parameter.Type.DOUBLE, Parameter.Type.INTEGER} or not 0.0 <= float(parameter.value) <= 60.0:
                    return SetParametersResult(successful=False, reason="monitor_gain_db must be in [0,60]")
            elif parameter.name == "monitor_limit":
                if parameter.type_ not in {Parameter.Type.DOUBLE, Parameter.Type.INTEGER} or not 0.0 < float(parameter.value) <= 1.0:
                    return SetParametersResult(successful=False, reason="monitor_limit must be in (0,1]")
            elif parameter.name == "semantic_group_timeout_sec":
                if parameter.type_ not in {Parameter.Type.DOUBLE, Parameter.Type.INTEGER} or not math.isfinite(float(parameter.value)) or float(parameter.value) <= 0.0:
                    return SetParametersResult(successful=False, reason="semantic_group_timeout_sec must be finite and positive")
            elif parameter.name in tuning_names:
                if parameter.type_ not in {Parameter.Type.DOUBLE, Parameter.Type.INTEGER} or not math.isfinite(float(parameter.value)):
                    return SetParametersResult(successful=False, reason=f"{parameter.name} must be finite")
            elif parameter.name == "solo_channel" and str(parameter.value) not in {"", *CHANNEL_NAMES}:
                return SetParametersResult(successful=False, reason="solo_channel must be empty or a canonical channel name")
            elif parameter.name == "monitor_mode" and str(parameter.value) not in {"headphones", "hearing"}:
                return SetParametersResult(successful=False, reason="monitor_mode must be headphones or hearing")
        if any(parameter.name in tuning_names for parameter in parameters):
            tuning = self._motor_tuning()
            parameter_keys = {
                "motor_volume_db": "volume_db",
                "motor_frequency_scale": "frequency_scale",
                "motor_tonal_gain_db": "tonal_gain_db",
                "motor_broadband_gain_db": "broadband_gain_db",
                "motor_speed_exponent": "speed_exponent",
                "motor_velocity_smoothing_sec": "velocity_smoothing_seconds",
            }
            overrides = {parameter.name: float(parameter.value) for parameter in parameters}
            if "motor_volume_db" in overrides or "motor_mems_calibration_db" in overrides:
                tuning["volume_db"] = overrides.get(
                    "motor_volume_db",
                    float(self.get_parameter("motor_volume_db").value),
                ) + overrides.get(
                    "motor_mems_calibration_db",
                    float(self.get_parameter("motor_mems_calibration_db").value),
                )
            for parameter_name, tuning_name in parameter_keys.items():
                if parameter_name in overrides and parameter_name != "motor_volume_db":
                    tuning[tuning_name] = overrides[parameter_name]
            for voice in self._procedural.values():
                voice.source.tune(**tuning)
        return SetParametersResult(successful=True)

    def _open_audio_device(self) -> None:
        requested = str(self.get_parameter("audio_device").value).strip()
        if requested in {"", "none"}:
            self._stream_error = "playback disabled"
            return
        if self._stream is not None:
            try:
                if self._stream.active:
                    return
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        try:
            import sounddevice as sd

            selected: str | int | None = None if requested == "auto" else requested
            if requested == "auto" and os.environ.get("PULSE_SERVER"):
                pulse_outputs = [index for index, description in enumerate(sd.query_devices()) if "pulse" in str(description["name"]).lower() and int(description["max_output_channels"]) >= 2]
                if pulse_outputs:
                    selected = pulse_outputs[0]
            sd.query_devices(selected, "output")
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                blocksize=self.block_size,
                device=selected,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._stream_error = ""
            self.get_logger().info(f"stereo headphone output active on device {self._stream.device}")
        except Exception as exc:
            error = str(exc)
            if error != self._stream_error:
                self.get_logger().warning(f"cannot open stereo headphone output {requested!r}: {error}; retrying")
            self._stream_error = error
            self._stream = None

    def _retry_audio_device(self) -> None:
        requested = str(self.get_parameter("audio_device").value).strip()
        if self._publishers_ready and requested not in {"", "none"} and self._stream is None:
            self._open_audio_device()

    def _audio_callback(self, outdata: np.ndarray, frames: int, _time: object, status: object) -> None:
        outdata.fill(0.0)
        self._audio_callbacks += 1
        status_text = str(status).strip()
        if status_text:
            self._audio_status = status_text
        written = 0
        while written < frames:
            if self._output_current is None or self._output_current_offset >= len(self._output_current):
                with self._output_lock:
                    self._output_current = self._output_blocks.popleft() if self._output_blocks else None
                self._output_current_offset = 0
                if self._output_current is None:
                    self._audio_underflows += 1
                    break
            available = len(self._output_current) - self._output_current_offset
            count = min(frames - written, available)
            outdata[written : written + count] = self._output_current[self._output_current_offset : self._output_current_offset + count]
            written += count
            self._output_current_offset += count
        self._audio_peak = float(np.max(np.abs(outdata))) if outdata.size else 0.0

    def _publish_audio_diagnostics(self) -> None:
        active = False
        device = None
        if self._stream is not None:
            try:
                active = bool(self._stream.active)
                device = self._stream.device
            except Exception as exc:
                self._stream_error = str(exc)
                self._stream = None
        with self._output_lock:
            queued = len(self._output_blocks)
        new_underflows = self._audio_underflows - self._reported_underflows
        new_overflows = self._audio_overflows - self._reported_overflows
        new_skipped = self._render.skipped - self._reported_skipped
        self._reported_underflows = self._audio_underflows
        self._reported_overflows = self._audio_overflows
        self._reported_skipped = self._render.skipped
        message = (
            "four-mic audio diagnostics: "
            f"robot={self._robot_name or None}, "
            f"publishers_ready={self._publishers_ready}, "
            f"heard={self._heard_events}, accepted={self._accepted_events}, "
            f"fused={self._fused_events}, "
            f"semantic_pending={len(self._semantic_event_groups)}, "
            f"semantic_incomplete={self._incomplete_semantic_events}, "
            f"finite_pending={len(self._event_loads)}, "
            f"wav_voices={len(self._continuous)}, "
            f"drivetrain_voices={len(self._procedural)}, "
            f"drivetrain_pending={len(self._procedural_pending)}, "
            f"stream_active={active}, device={device}, queued={queued}, "
            f"callbacks={self._audio_callbacks}, "
            f"underflows={self._audio_underflows}, "
            f"overflows={self._audio_overflows}, "
            f"output_peak={self._audio_peak:.4f}, "
            f"rendered={self._render.rendered}, skipped={self._render.skipped}, "
            f"status={self._audio_status!r}, error={self._stream_error!r}"
        )
        if new_skipped > 0 and not self._render_behind:
            self.get_logger().warning(f"four-mic render fell behind /clock, skipped {new_skipped} block(s) ({new_skipped * self.block_size / self.sample_rate:.2f} s of audio) to stay current")
        elif self._render_behind and new_skipped == 0:
            self.get_logger().info("four-mic render caught up with /clock")
        self._render_behind = new_skipped > 0
        playback = str(self.get_parameter("audio_device").value).strip() not in {"", "none"}
        degraded = playback and (not active or new_underflows > 0 or new_overflows > 0)
        if degraded and not self._playback_degraded:
            self.get_logger().warning(f"four-mic playback degraded (stream_active={active}, underflows={self._audio_underflows}, overflows={self._audio_overflows}), monitoring only, hearing is unaffected")
        elif self._playback_degraded and not degraded:
            self.get_logger().info("four-mic playback recovered")
        self._playback_degraded = degraded
        self.get_logger().debug(message)

    @staticmethod
    def _stable_occurrence(event_id: str) -> int:
        return sum((index + 1) * ord(character) for index, character in enumerate(event_id)) & 0x7FFFFFFF

    def _base_frame(self) -> str:
        return "/".join(part for part in (self._robot_frame_prefix, "base_link") if part)

    def _frame_id(self, name: str) -> str:
        return "/".join(part for part in (self._robot_frame_prefix, f"mic_{name}") if part)

    def destroy_node(self) -> bool:
        self._loader.shutdown(wait=False, cancel_futures=True)
        self._procedural_loader.shutdown(wait=False, cancel_futures=True)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = MicrophoneArrayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
