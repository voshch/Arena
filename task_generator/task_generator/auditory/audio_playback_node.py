from __future__ import annotations

import json
import math
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import attrs
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_simulation_setup.tree.World import WorldIdentifier
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import (
    FloatingPointRange,
    ParameterDescriptor,
    SetParametersResult,
)
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.signal import fftconvolve
from shapely.geometry import Polygon
from std_msgs.msg import String
from task_generator_msgs.msg import (
    ContinuousHeardSoundState,
    EpisodeRecord,
    HeardSoundEvent,
    SoundEvent,
)

from task_generator.auditory.acoustic_frame import (
    realize_rooms_and_graph,
    runtime_acoustic_offset,
)
from task_generator.auditory.acoustic_room_spec import (
    AcousticRoomSpec,
    AcousticRoomSpecBuilder,
    AcousticRoomSpecConfig,
)
from task_generator.auditory.acoustic_world_graph import AcousticWorldGraph
from task_generator.auditory.asset_lib import (
    AcousticAsset,
    AcousticAssetCatalog,
    CachedSample,
)
from task_generator.auditory.audio_mixer import AudioMixer
from task_generator.auditory.material_catalog import AcousticMaterialCatalog
from task_generator.auditory.portal_coupling import (
    MultiPortalRirCoupler,
    PortalCouplingConfig,
)
from task_generator.auditory.procedural_audio import (
    DrivetrainRenderSource,
    clear_drivetrain_audio_cache,
)
from task_generator.auditory.pyroomacoustics_adapter import (
    PyroomacousticsAdapter,
    PyroomacousticsConfig,
)
from task_generator.auditory.qos_profiles import (
    acoustic_metadata_qos,
    continuous_audio_qos,
    transient_event_qos,
)

#: Discrete events the playback path renders: dry or propagated.
PlaybackEventMsg = HeardSoundEvent | SoundEvent
#: Propagation output, carrying listener geometry, zones and backend metadata.
PropagatedEventMsg = ContinuousHeardSoundState | HeardSoundEvent
#: Any of the three event flavours the node handles.
SoundEventMsg = ContinuousHeardSoundState | HeardSoundEvent | SoundEvent

FOOTSTEP_VARIANT_TAGS = frozenset({"default", "walnut_planks", "oak_planks", "marble_tile", "smooth_concrete", "ceramic_tile"})

MOTOR_TUNING_PARAMETERS = {
    "motor_volume_db": (
        -9.0,
        -40.0,
        6.0,
        "Motor output trim in dB. -6 dB is half amplitude.",
    ),
    "motor_frequency_scale": (
        1.0,
        0.25,
        4.0,
        "Motor pitch multiplier while preserving velocity-driven level.",
    ),
    "motor_tonal_gain_db": (
        0.0,
        -24.0,
        12.0,
        "Gear-mesh tone trim in dB.",
    ),
    "motor_broadband_gain_db": (
        -12.0,
        -40.0,
        6.0,
        "Broadband mechanical-noise trim in dB.",
    ),
    "motor_speed_exponent": (
        1.5,
        0.25,
        3.0,
        "Exponent controlling how strongly volume follows wheel speed.",
    ),
    "motor_velocity_smoothing_sec": (
        0.015,
        0.0,
        0.5,
        "Wheel-velocity response smoothing in seconds.",
    ),
}


class SoundPlaybackNode(Node):
    def __init__(
        self,
        node_name: str,
        source_kind: str,
        **kwargs: object,
    ) -> None:
        super().__init__(node_name, **kwargs)
        if source_kind not in {"environment", "human", "robot"}:
            raise ValueError(f"unsupported playback source kind {source_kind!r}")
        self._source_kind = source_kind

        share_dir = Path(get_package_share_directory("task_generator"))
        self.declare_parameter("sound_events_topic", "human_sound_events")
        self.declare_parameter("sound_dir", str(share_dir / "sounds"))
        self.declare_parameter(
            "asset_catalog",
            str(share_dir / "config" / "auditory" / "acoustic_assets.yaml"),
        )
        self.declare_parameter("output_sample_rate", 44100)
        self.declare_parameter("output_channels", 2)
        self.declare_parameter("block_size", 1024)
        self.declare_parameter("audio_device", "")
        self.declare_parameter("master_gain_db", 0.0)
        self.declare_parameter("episode_topic", "state/episode")
        self.declare_parameter("use_rir", False)
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
        if self._source_kind in {"environment", "robot"}:
            self.declare_parameter(
                "continuous_heard_sounds_topic",
                "continuous_heard_sounds",
            )
        self.declare_parameter("listener_id", "")
        self.declare_parameter(
            "microphone_listeners_topic",
            "microphone_listeners",
        )
        self.declare_parameter("world_topic", "state/world")
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("rir_sample_rate_hz", 44100)
        self.declare_parameter("rir_max_order", 3)
        self.declare_parameter("rir_temperature_c", 20.0)
        self.declare_parameter("rir_relative_humidity_percent", 50.0)
        self.declare_parameter("rir_ceiling_height_m", 3.0)
        self.declare_parameter(
            "rir_cache_position_quantization_m",
            0.10,
        )
        self.declare_parameter("rir_cache_size", 512)
        self.declare_parameter("rir_dry_fallback", True)
        self.declare_parameter("play_inaudible_events", True)
        self.declare_parameter("minimum_playback_gain_db", -60.0)
        if self._source_kind == "robot":
            self.declare_parameter("motor_playback_mode", "sequence")
            self.declare_parameter("motor_audio_mode", "wav")
            self.declare_parameter("enable_motor_playback", True)
            self.declare_parameter("motor_rir_crossfade_sec", 0.1)
            self.declare_parameter("motor_single_asset_id", "motor")
            for name, (default, minimum, maximum, description) in (
                MOTOR_TUNING_PARAMETERS.items()
            ):
                self.declare_parameter(
                    name,
                    default,
                    ParameterDescriptor(
                        description=description,
                        floating_point_range=[
                            FloatingPointRange(
                                from_value=minimum,
                                to_value=maximum,
                                step=0.0,
                            )
                        ],
                    ),
                )
        self.declare_parameter("portal_adjacency_tolerance_m", 0.08)
        self.declare_parameter("portal_inset_m", 0.03)
        self.declare_parameter("portal_loss_db", 3.0)
        self.declare_parameter("opening_portal_loss_db", 0.5)
        self.declare_parameter("derive_opening_portals", True)
        self.declare_parameter("minimum_opening_width_m", 0.30)
        self.declare_parameter("enable_multi_portal_rir", True)
        self.declare_parameter("max_portal_hops", 4)
        self.declare_parameter("route_distance_loss_db_per_m", 0.05)
        self.declare_parameter("portal_source_early_window_sec", 0.08)
        self.declare_parameter("portal_max_rir_duration_sec", 2.0)
        self.declare_parameter("portal_position_quantization_m", 0.10)
        self.declare_parameter("portal_rir_cache_size", 256)
        self.declare_parameter("rir_event_buffer_size", 128)

        sample_rate = int(self.get_parameter("output_sample_rate").value)
        channels = int(self.get_parameter("output_channels").value)
        device = str(self.get_parameter("audio_device").value).strip()

        self._catalog = AcousticAssetCatalog(
            config_path=Path(str(self.get_parameter("asset_catalog").value)),
            sound_dir=Path(str(self.get_parameter("sound_dir").value)),
            output_sample_rate=sample_rate,
            output_channels=channels,
        )
        self._material_catalog = AcousticMaterialCatalog(
            share_dir / "config" / "auditory" / "acoustic_materials.yaml"
        )
        if self._source_kind == "human":
            footstep_asset = self._catalog.require("footstep")
            greeting_asset = self._catalog.require("greeting")
            self.get_logger().info(
                "registered lazy human audio assets: "
                f"footstep_variants={len(footstep_asset.variants)}, "
                f"greeting_variants={len(greeting_asset.variants)}, "
                f"sound_dir={self.get_parameter('sound_dir').value!r}"
            )
        self._asset_loader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="audio_asset_loader",
        )
        self._pending_asset_loads: deque[
            tuple[Future, PlaybackEventMsg, AcousticAsset | tuple[AcousticAsset, ...], str | None]
        ] = deque()
        self._cancelled_motor_starts: set[str] = set()
        self._asset_poll_timer = self.create_timer(
            0.01,
            self._poll_asset_loads,
        )

        master_gain_db = float(self.get_parameter("master_gain_db").value)
        if device == "none":
            self._mixer = AudioMixer(channels=channels, master_gain_db=master_gain_db)
        else:
            try:
                self._mixer = AudioMixer.open(
                    sample_rate=sample_rate,
                    channels=channels,
                    block_size=int(self.get_parameter("block_size").value),
                    device=None if device in ("", "auto") else device,
                    master_gain_db=master_gain_db,
                )
            except RuntimeError as exc:
                self.get_logger().warning(f"{exc}; continuing without audio output")
                self._mixer = AudioMixer(channels=channels, master_gain_db=master_gain_db)
        if self._source_kind == "robot":
            self._mixer.set_bus_enabled(
                "motor",
                bool(self.get_parameter("enable_motor_playback").value),
            )

        self._use_rir = bool(self.get_parameter("use_rir").value)
        self._rir_dry_fallback = bool(
            self.get_parameter("rir_dry_fallback").value
        )
        self._play_inaudible_events = bool(
            self.get_parameter("play_inaudible_events").value
        )
        self._minimum_playback_gain_db = float(
            self.get_parameter("minimum_playback_gain_db").value
        )
        self._motor_playback_mode = ""
        if self._source_kind == "robot":
            self._motor_playback_mode = str(
                self.get_parameter("motor_playback_mode").value
            ).strip()
            if self._motor_playback_mode not in {
                "sequence",
                "single_loop",
            }:
                raise ValueError(
                    "motor_playback_mode must be 'sequence' or 'single_loop', "
                    f"got {self._motor_playback_mode!r}"
                )
            self.get_logger().info(
                f"motor playback mode={self._motor_playback_mode!r}"
            )
        self._room_specs: tuple[AcousticRoomSpec, ...] = ()
        self._authored_room_specs: tuple[AcousticRoomSpec, ...] = ()
        self._continuous_sources: dict[
            tuple[str, str], DrivetrainRenderSource
        ] = {}
        self._continuous_rir_signatures: dict[
            tuple[str, str], tuple[Hashable, ...]
        ] = {}
        self._microphone_listener_ids: set[str] = set()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._world_graph: AcousticWorldGraph | None = None
        self._authored_world_graph: AcousticWorldGraph | None = None
        self._authored_map_origin: tuple[float, float] | None = None
        self._map: OccupancyGrid | None = None
        self._acoustic_alignment_signature: tuple[Hashable, ...] | None = None
        self._portal_coupler: MultiPortalRirCoupler | None = None
        self._world_name = ""
        self._world_loader = ThreadPoolExecutor(max_workers=1)
        self._world_load_future = None
        self._pra_adapter = None
        self._heard_received = 0
        self._heard_robot_matched = 0
        self._heard_filtered = 0
        self._played_events = 0
        self._warned_no_heard_events = False
        self._pending_heard_events: deque[HeardSoundEvent] = deque()
        self._rir_warning_times: dict[tuple[str, str], float] = {}
        self._last_diagnostic_time = time.monotonic()
        self._diagnostic_timer = self.create_timer(
            5.0,
            self._publish_diagnostics,
        )

        if self._use_rir:
            materials = AcousticMaterialCatalog(
                share_dir / "config" / "auditory" / "acoustic_materials.yaml"
            )
            self._pra_adapter = PyroomacousticsAdapter(
                materials,
                PyroomacousticsConfig(
                    sample_rate_hz=int(
                        self.get_parameter("rir_sample_rate_hz").value
                    ),
                    max_order=int(
                        self.get_parameter("rir_max_order").value
                    ),
                    temperature_c=float(
                        self.get_parameter("rir_temperature_c").value
                    ),
                    relative_humidity_percent=float(
                        self.get_parameter(
                            "rir_relative_humidity_percent"
                            ).value
                    ),
                    cache_position_quantization_m=float(
                        self.get_parameter(
                            "rir_cache_position_quantization_m"
                        ).value
                    ),
                    cache_size=int(
                        self.get_parameter("rir_cache_size").value
                    ),
                ),
            )

            world_topic = str(self.get_parameter("world_topic").value)
            self._world_subscription = self.create_subscription(
                String,
                world_topic,
                self._cb_world,
                acoustic_metadata_qos(),
            )
            self._map_subscription = self.create_subscription(
                OccupancyGrid,
                str(self.get_parameter("map_topic").value),
                self._cb_map,
                acoustic_metadata_qos(),
            )
            heard_topic = str(
                self.get_parameter("heard_sound_events_topic").value
            )
            self.create_subscription(
                HeardSoundEvent,
                heard_topic,
                self._cb_heard_sound,
                transient_event_qos(),
            )
            self.create_subscription(
                String,
                str(
                    self.get_parameter(
                        "microphone_listeners_topic"
                    ).value
                ),
                self._cb_microphone_registry,
                acoustic_metadata_qos(),
            )
            if self._source_kind in {"environment", "robot"}:
                self.create_subscription(
                    ContinuousHeardSoundState,
                    str(
                        self.get_parameter(
                            "continuous_heard_sounds_topic"
                        ).value
                    ),
                    self._cb_continuous_heard_sound,
                    continuous_audio_qos(),
                )
            self.get_logger().info(
                "RIR audio rendering enabled; listening for "
                f"{self._listener_description()}"
            )
        else:
            topic = str(self.get_parameter("sound_events_topic").value)
            self.create_subscription(
                SoundEvent,
                topic,
                self._cb_sound_event,
                transient_event_qos(),
            )
            self.get_logger().info(f"playing dry sound events from {topic}")

        self._episode_seed = 0
        self._episode_id = -1
        self._occurrences: dict[tuple[int, str], int] = defaultdict(int)
        self._event_occurrences: dict[tuple[str, str], int] = {}

        episode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            EpisodeRecord,
            str(self.get_parameter("episode_topic").value),
            self._cb_episode,
            episode_qos,
        )
        self._world_poll_timer = self.create_timer(
            0.1,
            self._poll_room_load,
        )

    def _cb_world(self, msg: String) -> None:
        world_name = str(msg.data).strip()
        if not world_name or world_name == self._world_name:
            return
        if self._world_load_future is not None and not self._world_load_future.done():
            return
        ceiling = float(
            self.get_parameter("rir_ceiling_height_m").value
        )
        self._world_load_future = self._world_loader.submit(
            self._load_room_specs,
            world_name,
            ceiling,
            float(self.get_parameter("portal_adjacency_tolerance_m").value),
            bool(self.get_parameter("derive_opening_portals").value),
            float(self.get_parameter("minimum_opening_width_m").value),
            float(self.get_parameter("portal_loss_db").value),
            float(self.get_parameter("opening_portal_loss_db").value),
        )

    def _cb_map(self, msg: OccupancyGrid) -> None:
        self._map = msg
        self._realize_acoustic_geometry()

    def _on_set_parameters(self, parameters: list[Parameter]) -> SetParametersResult:
        selection_changed = False
        for parameter in parameters:
            if parameter.name == "listener_id":
                if parameter.type_ != Parameter.Type.STRING:
                    return SetParametersResult(
                        successful=False,
                        reason="listener_id must be a string",
                    )
                selection_changed = True

        if selection_changed:
            self._mixer.stop_all()
            self._pending_heard_events.clear()
            while self._pending_asset_loads:
                future, _, _, _ = self._pending_asset_loads.popleft()
                future.cancel()
            if self._source_kind == "robot":
                self._cancelled_motor_starts.clear()
                self._continuous_sources.clear()
                self._continuous_rir_signatures.clear()

        if self._source_kind != "robot":
            return SetParametersResult(successful=True)

        tuning = self._motor_tuning()
        for parameter in parameters:
            if parameter.name == "enable_motor_playback":
                if parameter.type_ != Parameter.Type.BOOL:
                    return SetParametersResult(
                        successful=False,
                        reason="enable_motor_playback must be a boolean",
                    )
                continue
            if parameter.name not in MOTOR_TUNING_PARAMETERS:
                continue
            if parameter.type_ != Parameter.Type.DOUBLE:
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be a double",
                )
            value = float(parameter.value)
            _, minimum, maximum, _ = MOTOR_TUNING_PARAMETERS[parameter.name]
            if not math.isfinite(value) or not minimum <= value <= maximum:
                return SetParametersResult(
                    successful=False,
                    reason=(
                        f"{parameter.name} must be in "
                        f"[{minimum}, {maximum}]"
                    ),
                )
            tuning[self._motor_tuning_key(parameter.name)] = value

        for parameter in parameters:
            if parameter.name == "enable_motor_playback":
                self._mixer.set_bus_enabled("motor", bool(parameter.value))
        for source in self._continuous_sources.values():
            source.tune(**tuning)
        return SetParametersResult(successful=True)

    @staticmethod
    def _motor_tuning_key(parameter_name: str) -> str:
        return {
            "motor_volume_db": "volume_db",
            "motor_frequency_scale": "frequency_scale",
            "motor_tonal_gain_db": "tonal_gain_db",
            "motor_broadband_gain_db": "broadband_gain_db",
            "motor_speed_exponent": "speed_exponent",
            "motor_velocity_smoothing_sec": "velocity_smoothing_seconds",
        }[parameter_name]

    def _motor_tuning(self) -> dict[str, float]:
        return {
            self._motor_tuning_key(name): float(
                self.get_parameter(name).value
            )
            for name in MOTOR_TUNING_PARAMETERS
        }

    @staticmethod
    def _load_room_specs(
        world_name: str,
        ceiling_height_m: float,
        adjacency_tolerance_m: float,
        derive_opening_portals: bool,
        minimum_opening_width_m: float,
        door_portal_loss_db: float,
        opening_portal_loss_db: float,
    ) -> tuple[
        str,
        tuple[AcousticRoomSpec, ...],
        AcousticWorldGraph,
        tuple[float, float] | None,
    ]:
        world_view = WorldIdentifier(world_name).resolve_sync()
        world = world_view.load()
        authored_map_origin = None
        level_origins = world_view.level_origins()
        if level_origins is not None:
            world = world.compact_world(level_origins)
            _, authored_map_origin = world.render_grid()
        else:
            for level_id in sorted(world.levels):
                map_yaml = Path(world_view.path) / str(level_id) / "map.yaml"
                if not map_yaml.exists():
                    continue
                map_config = yaml.safe_load(
                    map_yaml.read_text(encoding="utf-8")
                )
                origin = map_config.get("origin", (0.0, 0.0, 0.0))
                authored_map_origin = (float(origin[0]), float(origin[1]))
                break
        specs = AcousticRoomSpecBuilder(
            AcousticRoomSpecConfig(ceiling_height_m=ceiling_height_m)
        ).from_world(world)
        graph = AcousticWorldGraph.from_world(
            world,
            specs,
            adjacency_tolerance_m=adjacency_tolerance_m,
            derive_opening_portals=derive_opening_portals,
            minimum_opening_width_m=minimum_opening_width_m,
            door_portal_loss_db=door_portal_loss_db,
            opening_portal_loss_db=opening_portal_loss_db,
        )
        return world_name, specs, graph, authored_map_origin

    def _poll_room_load(self) -> None:
        if self._world_load_future is None or not self._world_load_future.done():
            return
        future = self._world_load_future
        self._world_load_future = None
        try:
            (
                self._world_name,
                self._authored_room_specs,
                self._authored_world_graph,
                self._authored_map_origin,
            ) = future.result()
            if self._authored_map_origin is None:
                self.get_logger().error(
                    f"cannot realize acoustic rooms for {self._world_name!r}: "
                    "no level map.yaml origin is available"
                )
            self._room_specs = ()
            self._world_graph = None
            self._portal_coupler = None
            self._acoustic_alignment_signature = None
            self._realize_acoustic_geometry()
        except Exception as exc:
            self.get_logger().error(f"failed to load acoustic rooms: {exc!r}")

    def _realize_acoustic_geometry(self) -> None:
        if (
            self._map is None
            or self._authored_map_origin is None
            or not self._authored_room_specs
            or self._authored_world_graph is None
        ):
            return
        offset = runtime_acoustic_offset(
            self._map,
            self._authored_map_origin,
        )
        signature = (self._world_name, *offset)
        if signature == self._acoustic_alignment_signature:
            return
        self._acoustic_alignment_signature = signature
        self._room_specs, self._world_graph = realize_rooms_and_graph(
            self._authored_room_specs,
            self._authored_world_graph,
            offset,
        )
        self._portal_coupler = (
            MultiPortalRirCoupler(
                self._pra_adapter,
                self._world_graph,
                world_name=self._world_name,
                config=self._portal_coupling_config(),
            )
            if self._pra_adapter is not None else None
        )
        self._continuous_rir_signatures.clear()
        warmup_seconds, warmup_error = self._warm_up_realized_room()
        self.get_logger().info(
            f"realized {len(self._room_specs)} acoustic rooms for "
            f"{self._world_name!r} in runtime map frame "
            f"{self._map.header.frame_id!r} with "
            f"offset=({offset[0]:.2f},{offset[1]:.2f}), "
            f"door_portals="
            f"{sum(p.portal_kind == 'door' for p in self._world_graph.portals)}, "
            f"opening_portals="
            f"{sum(p.portal_kind == 'opening' for p in self._world_graph.portals)}, "
            f"components={len(self._world_graph.connected_components())}, "
            f"unpaired_doors={len(self._world_graph.unpaired_doors)}, "
            f"pyroom_warmup={warmup_seconds:.3f}s"
        )
        if warmup_error:
            self.get_logger().warning(
                "pyroomacoustics warmup failed, but room loading "
                f"continues: {warmup_error}"
            )
        while self._pending_heard_events:
            self._process_heard_sound(self._pending_heard_events.popleft())

    def _warm_up_realized_room(self) -> tuple[float, str]:
        if self._pra_adapter is None or not self._room_specs:
            return 0.0, ""
        try:
            polygon = Polygon(self._room_specs[0].corners_xy)
            point = polygon.representative_point()
            started = time.perf_counter()
            self._pra_adapter.compute_rir(
                self._room_specs[0],
                source_position_m=(point.x, point.y, 0.5),
                listener_position_m=(point.x, point.y, 1.5),
            )
            return time.perf_counter() - started, ""
        except Exception as exc:
            return 0.0, f"{type(exc).__name__}: {exc}"

    def _cb_heard_sound(self, msg: HeardSoundEvent) -> None:
        if not self._matches_source_kind(msg):
            return
        self._heard_received += 1
        if not self._matches_listener(msg.listener_id):
            self._heard_filtered += 1
            return
        if self._use_rir and not self._room_specs:
            maximum = max(
                int(self.get_parameter("rir_event_buffer_size").value),
                1,
            )
            if len(self._pending_heard_events) >= maximum:
                dropped = self._pending_heard_events.popleft()
                self.get_logger().warning(
                    f"playback RIR event buffer full; dropping "
                    f"{dropped.event_id!r}"
                )
            self._pending_heard_events.append(msg)
            return
        self._process_heard_sound(msg)

    def _configured_listener_id(self) -> str:
        return str(self.get_parameter("listener_id").value).strip()

    def _matches_listener(self, listener_id: str) -> bool:
        selected = self._configured_listener_id()
        return bool(selected) and listener_id == selected

    def _listener_description(self) -> str:
        return self._configured_listener_id() or "no microphone selected"

    def _cb_microphone_registry(self, msg: String) -> None:
        try:
            listener_ids = json.loads(msg.data)
            if not isinstance(listener_ids, list) or not all(
                isinstance(listener_id, str)
                and listener_id.strip()
                for listener_id in listener_ids
            ):
                raise ValueError("expected a list of microphone listener IDs")
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().warning(
                f"ignoring invalid microphone registry: {exc}"
            )
            return
        self._microphone_listener_ids = set(listener_ids)

    def _process_heard_sound(self, msg: HeardSoundEvent) -> None:
        # A stop event is also a control transition for an already-playing
        # keyed motor voice, so it must be handled even if the source is no
        # longer acoustically audible at the listener.
        if (
            msg.asset_id != "motor_stop"
            and not msg.audible
            and not self._play_inaudible_events
        ):
            self._heard_filtered += 1
            return
        self._heard_robot_matched += 1
        try:
            self._play_event(msg)
        except Exception:
            self.get_logger().error(
                "unhandled exception while processing HeardSoundEvent "
                f"{msg.event_id!r}:\n{traceback.format_exc()}"
            )

    def _cb_episode(self, msg: EpisodeRecord) -> None:
        if self._use_rir and str(msg.world).strip():
            self._cb_world(String(data=str(msg.world).strip()))
        if msg.episode_id == self._episode_id:
            return

        self._episode_id = int(msg.episode_id)
        self._episode_seed = int(msg.seed)
        self._occurrences.clear()
        self._event_occurrences.clear()
        self._mixer.stop_all()
        while self._pending_asset_loads:
            future, _, _, _ = self._pending_asset_loads.popleft()
            future.cancel()
        if self._source_kind == "robot":
            self._cancelled_motor_starts.clear()
            self._continuous_sources.clear()
            self._continuous_rir_signatures.clear()
            clear_drivetrain_audio_cache()

    def _cb_sound_event(self, msg: SoundEvent) -> None:
        if not self._matches_source_kind(msg):
            return
        try:
            self._play_event(msg)
        except Exception:
            self.get_logger().error(
                "unhandled exception while processing SoundEvent "
                f"{msg.event_id!r}:\n{traceback.format_exc()}"
            )

    def _matches_source_kind(
        self,
        msg: SoundEvent | HeardSoundEvent,
    ) -> bool:
        event_kind = str(msg.event_id).partition(":")[0]
        if event_kind in {"human", "robot"}:
            return event_kind == self._source_kind
        fallback_kind = "robot" if str(msg.sound_type) == "motor" else "human"
        return fallback_kind == self._source_kind

    def _cb_continuous_heard_sound(
        self,
        msg: ContinuousHeardSoundState,
    ) -> None:
        if not self._matches_listener(msg.listener_id):
            return
        if str(self.get_parameter("motor_audio_mode").value).strip() != (
            "procedural"
        ):
            return
        if msg.source_backend != "drivetrain":
            return
        if self._use_rir and not self._room_specs:
            return

        source_key = (msg.listener_id, msg.source_id)
        voice_id = f"{msg.listener_id}|{msg.source_id}"
        source = self._continuous_sources.get(source_key)
        if source is not None and source.finished:
            self._continuous_sources.pop(source_key, None)
            self._continuous_rir_signatures.pop(source_key, None)
            source = None
        if source is None:
            if not msg.active:
                return
            source = DrivetrainRenderSource(
                field_seed=self._episode_seed,
                phase_index=int(msg.deterministic_seed),
                block_size=int(self.get_parameter("block_size").value),
                channels=int(self.get_parameter("output_channels").value),
                rir_crossfade_seconds=float(
                    self.get_parameter("motor_rir_crossfade_sec").value
                ),
                **self._motor_tuning(),
            )
            self._continuous_sources[source_key] = source
            self._mixer.add_render_source(
                source,
                voice_id=voice_id,
                bus="motor",
            )

        signature = self._continuous_rir_signature(msg)
        impulse = None
        if (
            msg.active
            and msg.audible
            and signature != self._continuous_rir_signatures.get(source_key)
        ):
            try:
                impulse, _ = self._compute_normalized_rir(msg)
                self._continuous_rir_signatures[source_key] = signature
            except Exception as exc:
                self.get_logger().warning(
                    f"continuous RIR unavailable for {msg.source_id!r}: "
                    f"{exc}; retaining the previous RIR"
                )

        gain_db = (
            float(msg.received_volume_db)
            - float(msg.source_volume_db)
        )
        has_rir = (
            impulse is not None
            or source_key in self._continuous_rir_signatures
        )
        source.update(
            left_velocity=float(msg.left_velocity_mps),
            right_velocity=float(msg.right_velocity_mps),
            gain_db=gain_db,
            active=bool(
                msg.active
                and msg.audible
                and (
                    has_rir
                    or not self._use_rir
                    or self._rir_dry_fallback
                )
            ),
            impulse=impulse,
            rir_signature=signature if impulse is not None else None,
        )

    def _continuous_rir_signature(
        self,
        msg: ContinuousHeardSoundState,
    ) -> tuple[Hashable, ...]:
        quantum = max(
            float(
                self.get_parameter(
                    "rir_cache_position_quantization_m"
                ).value
            ),
            1e-6,
        )

        def quantize(value: float) -> int:
            return round(float(value) / quantum)

        return (
            msg.propagation_backend,
            msg.source_zone,
            msg.listener_zone,
            tuple(msg.portal_ids),
            quantize(msg.source_position.x),
            quantize(msg.source_position.y),
            quantize(msg.listener_position.x),
            quantize(msg.listener_position.y),
            quantize(self._listener_height(msg)),
        )

    def _publish_diagnostics(self) -> None:
        portal_cache = (
            f", portal_cache_entries={self._portal_coupler.cache_entries}, "
            f"portal_cache_hits={self._portal_coupler.cache_hits}, "
            f"portal_cache_misses={self._portal_coupler.cache_misses}, "
            f"route_cache_entries={self._portal_coupler.route_cache_entries}, "
            f"route_cache_hits={self._portal_coupler.route_cache_hits}, "
            f"route_cache_misses={self._portal_coupler.route_cache_misses}"
            if self._portal_coupler is not None else ""
        )
        rir_cache = (
            f", rir_cache_entries={self._pra_adapter.cache_entries}, "
            f"rir_cache_hits={self._pra_adapter.cache_hits}, "
            f"rir_cache_misses={self._pra_adapter.cache_misses}"
            if self._pra_adapter is not None else ""
        )
        message = (
            "audio diagnostics: "
            f"heard={self._heard_received}, "
            f"robot_matched={self._heard_robot_matched}, "
            f"filtered={self._heard_filtered}, "
            f"played={self._played_events}, "
            f"rooms={len(self._room_specs)}, "
            f"rir={self._use_rir}{rir_cache}{portal_cache}, "
            f"stream_active={self._mixer.active}, "
            f"device={self._mixer.device}, "
            f"voices={self._mixer.voice_count}, "
            f"callbacks={self._mixer.callback_count}, "
            f"output_peak={self._mixer.last_output_peak:.4f}, "
            f"stream_status={self._mixer.last_status!r}, "
            f"stream_status_count={self._mixer.status_count}, "
            f"asset_cache_entries={self._catalog.cached_samples}, "
            f"asset_cache_hits={self._catalog.cache_hits}, "
            f"asset_cache_misses={self._catalog.cache_misses}, "
            f"asset_loads_pending={len(self._pending_asset_loads)}"
        )
        if self._heard_received > 0 and self._played_events == 0:
            self.get_logger().warning(message)
        elif (
            self._room_specs
            and self._heard_received == 0
            and not self._warned_no_heard_events
        ):
            self._warned_no_heard_events = True
            self.get_logger().warning(
                "audio stream is ready but no HeardSoundEvent has reached "
                f"playback; {message}"
            )
        else:
            self.get_logger().info(message)

    def _portal_coupling_config(self) -> PortalCouplingConfig:
        return PortalCouplingConfig(
            portal_inset_m=float(self.get_parameter("portal_inset_m").value),
            portal_loss_db=float(self.get_parameter("portal_loss_db").value),
            source_room_early_window_sec=float(
                self.get_parameter("portal_source_early_window_sec").value
            ),
            maximum_rir_duration_sec=float(
                self.get_parameter("portal_max_rir_duration_sec").value
            ),
            position_quantization_m=float(
                self.get_parameter("portal_position_quantization_m").value
            ),
            cache_size=int(self.get_parameter("portal_rir_cache_size").value),
            opening_portal_loss_db=float(
                self.get_parameter("opening_portal_loss_db").value
            ),
            max_portal_hops=(
                int(self.get_parameter("max_portal_hops").value)
                if bool(self.get_parameter("enable_multi_portal_rir").value)
                else 1
            ),
            route_distance_loss_db_per_m=float(
                self.get_parameter("route_distance_loss_db_per_m").value
            ),
        )

    def _play_event(self, msg: PlaybackEventMsg) -> None:
        asset_id = msg.asset_id.strip() or msg.sound_type.strip()

        listener_id = "dry" if isinstance(msg, SoundEvent) else str(msg.listener_id)
        motor_voice_id = (
            f"{listener_id}|motor:{int(msg.source_agent_id)}"
        )
        if asset_id == "motor_stop":
            if self._mixer.stop(motor_voice_id):
                self.get_logger().info(
                    f"stopping motor loop for {msg.source_agent_name!r}"
                )
            else:
                # The intro may still be decoding on the worker. Prevent it
                # from starting after a stop event has already arrived.
                self._cancelled_motor_starts.add(motor_voice_id)
            return

        if asset_id == "motor_start":
            if self._motor_playback_mode == "single_loop":
                self._schedule_single_motor_loop(msg, motor_voice_id)
            else:
                self._schedule_motor_sequence(msg, motor_voice_id)
            return

        occurrence = self._event_occurrence(msg, asset_id)

        required_tags = frozenset()

        if asset_id == "footstep":
            semantic_tags = (
                tuple(str(tag) for tag in msg.semantic_tags)
                if isinstance(msg, SoundEvent)
                else ()
            )
            if not semantic_tags:
                room = self._room_for_event(msg)
                floor_id = (
                    room.floor_material_id.lower()
                    if room is not None else ""
                )
                semantic_tags = (
                    ("walnut_planks",) if "walnut" in floor_id
                    else ("oak_planks",) if "oak" in floor_id
                    else ("marble_tile",) if "marble" in floor_id
                    else ("smooth_concrete",) if "smooth" in floor_id
                    else ("ceramic_tile",) if "ceramic" in floor_id
                    else ("default",)
                )
            required_tags = frozenset(
                FOOTSTEP_VARIANT_TAGS.intersection(
                    semantic_tags
                )
            ) or frozenset({"default"})

        selected = self._catalog.select(
            asset_id,
            episode_seed=self._episode_seed,
            agent_id=int(msg.source_agent_id),
            occurrence=occurrence,
            required_tags=required_tags,
        )

        if selected is None:
            self.get_logger().warning(f"no acoustic asset for asset_id={asset_id!r}")
            return

        asset, sample = selected
        future = self._asset_loader.submit(self._catalog.load, sample)
        self._pending_asset_loads.append((future, msg, asset, None))

    def _poll_asset_loads(self) -> None:
        """Complete worker decodes in event order on the ROS executor thread."""
        while (
            self._pending_asset_loads
            and self._pending_asset_loads[0][0].done()
        ):
            future, msg, context, motor_voice_id = (
                self._pending_asset_loads.popleft()
            )
            if future.cancelled():
                continue
            try:
                loaded = future.result()
                if motor_voice_id is None:
                    self._play_loaded_sample(msg, context, loaded)
                elif motor_voice_id in self._cancelled_motor_starts:
                    self._cancelled_motor_starts.discard(motor_voice_id)
                elif isinstance(context, tuple):
                    self._play_loaded_motor_sequence(
                        msg,
                        motor_voice_id,
                        context,
                        loaded,
                    )
                else:
                    self._play_loaded_single_motor_loop(
                        msg,
                        motor_voice_id,
                        context,
                        loaded,
                    )
            except Exception:
                self.get_logger().error(
                    "lazy acoustic asset load failed:\n"
                    f"{traceback.format_exc()}"
                )

    def _play_loaded_sample(self, msg: PlaybackEventMsg, asset: AcousticAsset, sample: CachedSample) -> None:
        asset_id = asset.asset_id
        rendered = sample
        playback_backend = "dry"
        playback_fallback_reason = ""
        if self._use_rir:
            rendered, playback_backend, playback_fallback_reason = (
                self._render_with_rir(msg, sample)
            )

        if isinstance(msg, SoundEvent):
            playback_gain_db = (
                float(msg.source_volume_db)
                - asset.reference_level_db
            )
        else:
            # HeardSoundEvent already contains distance, room, and occlusion
            # effects. Use it for amplitude instead of the source level, so
            # distant humans are quieter than nearby humans.
            playback_gain_db = (
                float(msg.received_volume_db)
                - asset.reference_level_db
            )

        playback_gain_db += asset.playback_gain_db
        playback_gain_db += self._material_gain_db(msg, asset_id)
        if playback_gain_db < self._minimum_playback_gain_db:
            self._heard_filtered += 1
            return

        event_loop = isinstance(msg, SoundEvent) and bool(msg.loop)
        self._mixer.play(rendered,loop=event_loop or asset.loop,gain_db=playback_gain_db)
        self._played_events += 1
        self.get_logger().info(
            f"playing {asset_id} / {sample.sample_id}: "
            f"{rendered.duration_sec:.3f}s, "
            f"propagation_backend={self._propagation_backend(msg)!r}, "
            f"playback_backend={playback_backend!r}, "
            f"playback_fallback={playback_fallback_reason!r}, "
            f"gain={playback_gain_db:.1f} dB"
        )

    def _schedule_motor_sequence(self, msg: PlaybackEventMsg, voice_id: str) -> None:
        self._cancelled_motor_starts.discard(voice_id)
        selected_segments = []
        for asset_id in ("motor_start", "motor_loop", "motor_stop"):
            occurrence = self._event_occurrence(msg, asset_id)
            selected = self._catalog.select(
                asset_id,
                episode_seed=self._episode_seed,
                agent_id=int(msg.source_agent_id),
                occurrence=occurrence,
            )
            if selected is None:
                self.get_logger().error(
                    f"motor sequence is missing asset {asset_id!r}"
                )
                return
            selected_segments.append(selected)

        assets = tuple(selected[0] for selected in selected_segments)
        sample_specs = tuple(selected[1] for selected in selected_segments)
        future = self._asset_loader.submit(
            self._catalog.load_many,
            sample_specs,
        )
        self._pending_asset_loads.append((future, msg, assets, voice_id))

    def _schedule_single_motor_loop(self, msg: PlaybackEventMsg, voice_id: str) -> None:
        self._cancelled_motor_starts.discard(voice_id)
        asset_id = str(
            self.get_parameter("motor_single_asset_id").value
        ).strip()
        occurrence = self._event_occurrence(msg, asset_id)
        selected = self._catalog.select(
            asset_id,
            episode_seed=self._episode_seed,
            agent_id=int(msg.source_agent_id),
            occurrence=occurrence,
        )
        if selected is None:
            self.get_logger().error(
                f"single motor loop is missing asset {asset_id!r}"
            )
            return
        asset, sample_spec = selected
        future = self._asset_loader.submit(
            self._catalog.load,
            sample_spec,
        )
        self._pending_asset_loads.append(
            (future, msg, asset, voice_id)
        )

    def _play_loaded_single_motor_loop(
        self,
        msg: PlaybackEventMsg,
        voice_id: str,
        asset: AcousticAsset,
        sample: CachedSample,
    ) -> None:
        playback_gain_db = self._event_playback_gain_db(msg, asset)
        playback_gain_db += self._material_gain_db(msg, asset.asset_id)
        if playback_gain_db < self._minimum_playback_gain_db:
            self._heard_filtered += 1
            return

        if self._use_rir:
            rendered, backend, fallback_reason = self._render_with_rir(
                msg,
                sample,
            )
        else:
            rendered, backend, fallback_reason = sample, "dry", ""

        self._mixer.play(
            rendered,
            loop=True,
            gain_db=playback_gain_db,
            voice_id=voice_id,
            bus="motor",
        )
        self._played_events += 1
        self.get_logger().info(
            f"playing single motor loop for "
            f"{msg.source_agent_name!r}: "
            f"asset={asset.asset_id!r}, "
            f"sample={sample.sample_id!r}, "
            f"playback_backend={backend!r}, "
            f"playback_fallback={fallback_reason!r}, "
            f"gain={playback_gain_db:.1f} dB"
        )

    def _play_loaded_motor_sequence(
        self,
        msg: PlaybackEventMsg,
        voice_id: str,
        assets: tuple[AcousticAsset, ...],
        samples: tuple[CachedSample, ...],
    ) -> None:
        start_asset = assets[0]
        playback_gain_db = self._event_playback_gain_db(msg, start_asset)
        playback_gain_db += self._material_gain_db(msg, start_asset.asset_id)
        if playback_gain_db < self._minimum_playback_gain_db:
            self._heard_filtered += 1
            return

        rendered_segments = []
        playback_backends = []
        playback_fallback_reasons = []
        for sample in samples:
            if self._use_rir:
                rendered, backend, reason = self._render_with_rir(msg, sample)
            else:
                rendered, backend, reason = sample, "dry", ""
            rendered_segments.append(rendered)
            playback_backends.append(backend)
            playback_fallback_reasons.append(reason)

        self._mixer.play_looping_sequence(
            rendered_segments[0],
            rendered_segments[1],
            rendered_segments[2],
            voice_id=voice_id,
            gain_db=playback_gain_db,
            bus="motor",
        )
        self._played_events += 1
        self.get_logger().info(
            f"playing motor sequence for {msg.source_agent_name!r}: "
            "intro -> loop -> outro; "
            f"propagation_backend={self._propagation_backend(msg)!r}, "
            f"playback_backends={sorted(set(playback_backends))}, "
            f"playback_fallbacks="
            f"{sorted(set(reason for reason in playback_fallback_reasons if reason))}"
        )

    @staticmethod
    def _propagation_backend(msg: PlaybackEventMsg) -> str:
        """Backend that produced the event; dry events never went through one."""
        if isinstance(msg, SoundEvent):
            return "direct_event"
        return str(msg.propagation_backend)

    @staticmethod
    def _event_playback_gain_db(msg: PlaybackEventMsg, asset: AcousticAsset) -> float:
        if isinstance(msg, SoundEvent):
            level_db = float(msg.source_volume_db)
        else:
            level_db = float(msg.received_volume_db)
        return level_db - asset.reference_level_db + asset.playback_gain_db

    def _event_occurrence(self, msg: PlaybackEventMsg, asset_id: str) -> int:
        event_id = str(msg.event_id)
        event_key = (event_id, asset_id)
        existing = self._event_occurrences.get(event_key)
        if event_id and existing is not None:
            return existing
        source_key = (int(msg.source_agent_id), asset_id)
        occurrence = self._occurrences[source_key]
        self._occurrences[source_key] += 1
        if event_id:
            self._event_occurrences[event_key] = occurrence
        return occurrence

    def _material_gain_db(self, msg: PlaybackEventMsg, asset_id: str) -> float:
        """Apply the centralized material damping model to direct events."""
        material_gain_db = 0.0
        if asset_id == "footstep" and not isinstance(msg, SoundEvent):
            source_zone = str(msg.source_zone).strip()
            listener_zone = str(msg.listener_zone).strip()
            if source_zone and listener_zone and source_zone == listener_zone:
                room = next(
                    (
                        spec
                        for spec in self._room_specs
                        if spec.zone_name == source_zone
                    ),
                    None,
                )
                if room is not None:
                    material_gain_db += self._material_catalog.surface_damping_db(
                        room.floor_material_id,
                        "floor",
                    )
        return material_gain_db

    @staticmethod
    def _source_height(msg: PropagatedEventMsg) -> float:
        if (
            isinstance(msg, ContinuousHeardSoundState)
            and msg.source_model == "static_audio_source"
        ):
            return float(msg.source_position.z)
        sound = f"{msg.sound_type} {msg.label}".lower()
        if "foot" in sound or "step" in sound:
            return 0.05
        if "motor" in sound or "robot" in sound:
            return 0.25
        return 1.60

    def _listener_height(self, msg: PropagatedEventMsg) -> float:
        if str(msg.listener_id) in self._microphone_listener_ids:
            return float(msg.listener_position.z)
        return 0.35

    def _room_for_event(self, msg: SoundEventMsg) -> AcousticRoomSpec | None:
        if not self._room_specs or isinstance(msg, SoundEvent):
            return None
        source_zone = str(msg.source_zone).strip()
        listener_zone = str(msg.listener_zone).strip()
        # A missing listener zone indicates that propagation could not place
        # the microphone in the source room (normally a cross-zone fallback).
        if not source_zone or not listener_zone:
            return None
        if source_zone != listener_zone:
            return None
        return next(
            (spec for spec in self._room_specs
             if spec.zone_name == source_zone),
            None,
        )

    def _render_with_rir(self, msg: PropagatedEventMsg, sample: CachedSample) -> tuple[CachedSample, str, str]:
        try:
            impulse, playback_backend = self._compute_normalized_rir(msg)
            channels = [
                fftconvolve(sample.samples[:, channel], impulse, mode="full")
                for channel in range(sample.samples.shape[1])
            ]
            rendered = np.stack(channels, axis=1).astype(np.float32)
            return (
                attrs.evolve(
                    sample,
                    samples=np.ascontiguousarray(rendered),
                    duration_sec=len(rendered) / sample.sample_rate,
                ),
                playback_backend,
                "",
            )
        except Exception as exc:
            unavailable = isinstance(exc, LookupError)
            reason = (
                f"rir_unavailable:{exc}"
                if unavailable
                else f"rir_render_error:{type(exc).__name__}:{exc}"
            )
            warning_key = (str(msg.asset_id), reason)
            now = time.monotonic()
            last_warning = self._rir_warning_times.get(
                warning_key,
                -float("inf"),
            )
            if now - last_warning >= 5.0:
                self._rir_warning_times[warning_key] = now
                self.get_logger().warning(
                    f"RIR {'unavailable' if unavailable else 'rendering failed'} "
                    f"for {msg.asset_id!r}: {exc}; using "
                    f"{'dry sample' if self._rir_dry_fallback else 'silence'}"
                )
            fallback = (
                sample
                if self._rir_dry_fallback
                else attrs.evolve(sample, samples=np.zeros_like(sample.samples))
            )
            return (
                fallback,
                "dry_fallback" if self._rir_dry_fallback else "silence",
                reason,
            )

    def _compute_normalized_rir(self, msg: PropagatedEventMsg) -> tuple[np.ndarray, str]:
        """Return the one shared pyroomacoustics treatment for any source."""
        source = (
            float(msg.source_position.x),
            float(msg.source_position.y),
            self._source_height(msg),
        )
        listener = (
            float(msg.listener_position.x),
            float(msg.listener_position.y),
            self._listener_height(msg),
        )
        if self._pra_adapter is None:
            raise LookupError("pyroomacoustics_adapter_not_initialized")
        propagation_reason = str(msg.backend_fallback_reason).strip()
        if msg.used_backend_fallback and propagation_reason:
            raise LookupError(f"propagation_fallback:{propagation_reason}")
        source_zone = str(msg.source_zone).strip()
        listener_zone = str(msg.listener_zone).strip()
        room = self._room_for_event(msg)
        if room is not None:
            rir = self._pra_adapter.compute_rir(
                room,
                source_position_m=source,
                listener_position_m=listener,
            )
            playback_backend = "pyroomacoustics_same_room"
        elif (
            source_zone
            and listener_zone
            and source_zone != listener_zone
            and self._portal_coupler is not None
        ):
            coupling = self._portal_coupler.compute(
                source_zone=source_zone,
                listener_zone=listener_zone,
                source_position_m=source,
                listener_position_m=listener,
            )
            rir = coupling.rir
            playback_backend = (
                "pyroomacoustics_one_door"
                if coupling.route is None or coupling.route.hop_count == 1
                else "pyroomacoustics_multi_portal"
            )
        else:
            if propagation_reason:
                raise LookupError(f"propagation_fallback:{propagation_reason}")
            if not source_zone or not listener_zone:
                raise LookupError("missing_source_or_listener_zone_metadata")
            raise LookupError("no_same_room_or_portal_route_rir")
        impulse = np.asarray(rir.samples, dtype=np.float32)
        if rir.global_delay_samples > 0:
            impulse = np.pad(
                impulse,
                (int(rir.global_delay_samples), 0),
            )
        # Propagation metadata supplies distance/portal level exactly once;
        # the normalized RIR supplies only direct/reflection/reverb shape.
        impulse_peak = float(np.max(np.abs(impulse)))
        if impulse_peak <= 0.0 or not np.isfinite(impulse_peak):
            raise ValueError("RIR has no finite non-zero impulse")
        return impulse / impulse_peak, playback_backend

    def destroy_node(self) -> bool:
        self._asset_loader.shutdown(wait=False, cancel_futures=True)
        self._world_loader.shutdown(wait=False, cancel_futures=True)
        self._mixer.close()
        return super().destroy_node()
