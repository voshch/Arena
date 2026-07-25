from __future__ import annotations
from pathlib import Path
from dataclasses import replace
from concurrent.futures import Future, ThreadPoolExecutor
import time
import traceback

import numpy as np
from scipy.signal import fftconvolve
from shapely.geometry import Polygon
from task_generator.auditory.asset_lib import AcousticAssetCatalog
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from collections import defaultdict, deque
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from task_generator.auditory.audio_mixer import AudioMixer
from arena_simulation_setup.tree.World import WorldIdentifier
from task_generator.auditory.acoustic_room_spec import (
    AcousticRoomSpec,
    AcousticRoomSpecBuilder,
    AcousticRoomSpecConfig,
)
from task_generator.auditory.acoustic_world_graph import AcousticWorldGraph
from task_generator.auditory.material_catalog import AcousticMaterialCatalog
from task_generator.auditory.pyroomacoustics_adapter import (
    PyroomacousticsAdapter,
    PyroomacousticsConfig,
)
from task_generator.auditory.portal_coupling import (
    MultiPortalRirCoupler,
    PortalCouplingConfig,
)
from task_generator_msgs.msg import EpisodeRecord, HeardSoundEvent, SoundEvent
from task_generator.auditory.qos_profiles import (
    acoustic_metadata_qos,
    transient_event_qos,
)

FOOTSTEP_VARIANT_TAGS = frozenset({"default", "walnut_planks", "oak_planks", "marble_tile", "smooth_concrete", "ceramic_tile"})

class HumanSoundPlaybackNode(Node):
    def __init__(self) -> None:
        super().__init__("human_sound_playback")

        share_dir = Path(get_package_share_directory("task_generator"))
        self.declare_parameter("sound_events_topic", "human_sound_events")
        self.declare_parameter("sound_dir", str(share_dir / "sounds"))
        self.declare_parameter("asset_catalog",str(share_dir / "config" / "auditory" / "acoustic_assets.yaml"))
        self.declare_parameter("output_sample_rate", 44100)
        self.declare_parameter("output_channels", 2)
        self.declare_parameter("block_size", 1024)
        self.declare_parameter("audio_device", "")
        self.declare_parameter("master_gain_db", 0.0)
        self.declare_parameter("episode_topic", "state/episode")
        self.declare_parameter("use_rir", False)
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
        self.declare_parameter("listener_robot_name", "jackal")
        self.declare_parameter("world_topic", "state/world")
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
        self.declare_parameter("motor_playback_mode", "sequence")
        self.declare_parameter("motor_single_asset_id", "motor")
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
        device = str(self.get_parameter("audio_device").value).strip() or None

        self._catalog = AcousticAssetCatalog(
            config_path=Path(str(self.get_parameter("asset_catalog").value)),
            sound_dir=Path(str(self.get_parameter("sound_dir").value)),
            output_sample_rate=sample_rate,
            output_channels=channels,
        )
        footstep_asset = self._catalog.require("footstep")
        greeting_asset = self._catalog.require("greeting")
        self.get_logger().info(
            "registered lazy pedestrian audio assets: "
            f"footstep_variants={len(footstep_asset.variants)}, "
            f"greeting_variants={len(greeting_asset.variants)}, "
            f"sound_dir={self.get_parameter('sound_dir').value!r}"
        )
        self._asset_loader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="audio_asset_loader",
        )
        self._pending_asset_loads: deque[
            tuple[Future, object, object, str | None]
        ] = deque()
        self._cancelled_motor_starts: set[str] = set()
        self._asset_poll_timer = self.create_timer(
            0.01,
            self._poll_asset_loads,
        )

        self._mixer = AudioMixer(
            sample_rate=sample_rate,
            channels=channels,
            block_size=int(self.get_parameter("block_size").value),
            device=device,
            master_gain_db=float(self.get_parameter("master_gain_db").value),
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
        self._world_graph: AcousticWorldGraph | None = None
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
            heard_topic = str(
                self.get_parameter("heard_sound_events_topic").value
            )
            self.create_subscription(
                HeardSoundEvent,
                heard_topic,
                self._cb_heard_sound,
                transient_event_qos(),
            )
            self.get_logger().info(
                "RIR audio rendering enabled; listening for "
                f"robot:{self.get_parameter('listener_robot_name').value}"
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

    def _cb_world(self, msg) -> None:
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
            self._pra_adapter,
        )

    @staticmethod
    def _load_room_specs(
        world_name: str,
        ceiling_height_m: float,
        adjacency_tolerance_m: float,
        derive_opening_portals: bool,
        minimum_opening_width_m: float,
        door_portal_loss_db: float,
        opening_portal_loss_db: float,
        adapter: PyroomacousticsAdapter | None,
    ) -> tuple[
        str,
        tuple[AcousticRoomSpec, ...],
        AcousticWorldGraph,
        float,
        str,
    ]:
        world = WorldIdentifier(world_name).resolve_sync().load()
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
        warmup_seconds = 0.0
        warmup_error = ""
        if adapter is not None and specs:
            try:
                polygon = Polygon(specs[0].corners_xy)
                point = polygon.representative_point()
                started = time.perf_counter()
                adapter.compute_rir(
                    specs[0],
                    source_position_m=(point.x, point.y, 0.5),
                    listener_position_m=(point.x, point.y, 1.5),
                )
                warmup_seconds = time.perf_counter() - started
            except Exception as exc:
                # Warming the optional cache must never make the acoustic
                # scene unavailable. A real event can still render or use
                # the configured dry fallback.
                warmup_error = f"{type(exc).__name__}: {exc}"
        return world_name, specs, graph, warmup_seconds, warmup_error

    def _poll_room_load(self) -> None:
        if self._world_load_future is None or not self._world_load_future.done():
            return
        future = self._world_load_future
        self._world_load_future = None
        try:
            (
                self._world_name,
                self._room_specs,
                self._world_graph,
                warmup_seconds,
                warmup_error,
            ) = future.result()
            self._portal_coupler = (
                MultiPortalRirCoupler(
                    self._pra_adapter,
                    self._world_graph,
                    world_name=self._world_name,
                    config=self._portal_coupling_config(),
                )
                if self._pra_adapter is not None else None
            )
            self.get_logger().info(
                f"loaded {len(self._room_specs)} acoustic rooms for "
                f"{self._world_name!r}; "
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
                self._process_heard_sound(
                    self._pending_heard_events.popleft()
                )
        except Exception as exc:
            self.get_logger().error(f"failed to load acoustic rooms: {exc!r}")

    def _cb_heard_sound(self, msg: HeardSoundEvent) -> None:
        expected = "robot:" + str(
            self.get_parameter("listener_robot_name").value
        ).strip()
        self._heard_received += 1
        if msg.listener_id != expected:
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
        self._mixer.stop_all()
        while self._pending_asset_loads:
            future, _, _, _ = self._pending_asset_loads.popleft()
            future.cancel()
        self._cancelled_motor_starts.clear()

    def _cb_sound_event(self, msg: SoundEvent) -> None:
        try:
            self._play_event(msg)
        except Exception:
            self.get_logger().error(
                "unhandled exception while processing SoundEvent "
                f"{msg.event_id!r}:\n{traceback.format_exc()}"
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

    def _play_event(self, msg) -> None:
        asset_id = msg.asset_id.strip() or msg.sound_type.strip()

        motor_voice_id = f"motor:{int(msg.source_agent_id)}"
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

        key = (int(msg.source_agent_id), asset_id)

        occurrence = self._occurrences[key]
        self._occurrences[key] += 1

        required_tags = frozenset()

        if asset_id == "footstep":
            semantic_tags = tuple(
                str(tag) for tag in getattr(msg, "semantic_tags", ())
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

    def _play_loaded_sample(self, msg, asset, sample) -> None:
        asset_id = asset.asset_id
        rendered = sample
        playback_backend = "dry"
        playback_fallback_reason = ""
        if self._use_rir:
            rendered, playback_backend, playback_fallback_reason = (
                self._render_with_rir(msg, sample)
            )

        if hasattr(msg, "received_volume_db"):
            # HeardSoundEvent already contains distance, room, and occlusion
            # effects. Use it for amplitude instead of the source level, so
            # distant humans are quieter than nearby humans.
            playback_gain_db = (
                float(msg.received_volume_db)
                - asset.reference_level_db
            )
        else:
            playback_gain_db = (
                float(msg.source_volume_db)
                - asset.reference_level_db
            )

        playback_gain_db += asset.playback_gain_db
        if playback_gain_db < self._minimum_playback_gain_db:
            self._heard_filtered += 1
            return

        self._mixer.play(rendered,loop=bool(getattr(msg, "loop", False) or asset.loop),gain_db=playback_gain_db)
        self._played_events += 1
        self.get_logger().info(
            f"playing {asset_id} / {sample.sample_id}: "
            f"{rendered.duration_sec:.3f}s, "
            f"propagation_backend="
            f"{getattr(msg, 'propagation_backend', 'direct_event')!r}, "
            f"playback_backend={playback_backend!r}, "
            f"playback_fallback={playback_fallback_reason!r}, "
            f"gain={playback_gain_db:.1f} dB"
        )

    def _schedule_motor_sequence(self, msg, voice_id: str) -> None:
        self._cancelled_motor_starts.discard(voice_id)
        selected_segments = []
        for asset_id in ("motor_start", "motor_loop", "motor_stop"):
            key = (int(msg.source_agent_id), asset_id)
            occurrence = self._occurrences[key]
            self._occurrences[key] += 1
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

    def _schedule_single_motor_loop(self, msg, voice_id: str) -> None:
        self._cancelled_motor_starts.discard(voice_id)
        asset_id = str(
            self.get_parameter("motor_single_asset_id").value
        ).strip()
        key = (int(msg.source_agent_id), asset_id)
        occurrence = self._occurrences[key]
        self._occurrences[key] += 1
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
        msg,
        voice_id: str,
        asset,
        sample,
    ) -> None:
        playback_gain_db = self._event_playback_gain_db(msg, asset)
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
        msg,
        voice_id: str,
        assets,
        samples,
    ) -> None:
        start_asset = assets[0]
        playback_gain_db = self._event_playback_gain_db(msg, start_asset)
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
        )
        self._played_events += 1
        self.get_logger().info(
            f"playing motor sequence for {msg.source_agent_name!r}: "
            "intro -> loop -> outro; "
            f"propagation_backend="
            f"{getattr(msg, 'propagation_backend', 'direct_event')!r}, "
            f"playback_backends={sorted(set(playback_backends))}, "
            f"playback_fallbacks="
            f"{sorted(set(reason for reason in playback_fallback_reasons if reason))}"
        )

    @staticmethod
    def _event_playback_gain_db(msg, asset) -> float:
        if hasattr(msg, "received_volume_db"):
            level_db = float(msg.received_volume_db)
        else:
            level_db = float(msg.source_volume_db)
        return level_db - asset.reference_level_db + asset.playback_gain_db

    @staticmethod
    def _source_height(msg) -> float:
        sound = f"{msg.sound_type} {msg.label} " \
            f"{' '.join(str(tag) for tag in getattr(msg, 'semantic_tags', ())) }".lower()
        if "foot" in sound or "step" in sound:
            return 0.05
        if "motor" in sound or "robot" in sound:
            return 0.25
        return 1.60

    def _room_for_event(self, msg) -> AcousticRoomSpec | None:
        if not self._room_specs:
            return None
        source_zone = str(getattr(msg, "source_zone", "")).strip()
        listener_zone = str(getattr(msg, "listener_zone", "")).strip()
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

    def _render_with_rir(self, msg, sample):
        source = (
            float(msg.source_position.x),
            float(msg.source_position.y),
            self._source_height(msg),
        )
        listener = (
            float(msg.listener_position.x),
            float(msg.listener_position.y),
            0.35,
        )
        try:
            if self._pra_adapter is None:
                raise LookupError("pyroomacoustics_adapter_not_initialized")
            propagation_reason = str(
                getattr(msg, "backend_fallback_reason", "")
            ).strip()
            if bool(getattr(msg, "used_backend_fallback", False)) and (
                propagation_reason
            ):
                raise LookupError(
                    f"propagation_fallback:{propagation_reason}"
                )
            source_zone = str(getattr(msg, "source_zone", "")).strip()
            listener_zone = str(getattr(msg, "listener_zone", "")).strip()
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
                    if coupling.route is None
                    or coupling.route.hop_count == 1
                    else "pyroomacoustics_multi_portal"
                )
            else:
                if propagation_reason:
                    raise LookupError(
                        f"propagation_fallback:{propagation_reason}"
                    )
                if not source_zone or not listener_zone:
                    raise LookupError(
                        "missing_source_or_listener_zone_metadata"
                    )
                raise LookupError("no_same_room_or_portal_route_rir")
            impulse = np.asarray(rir.samples, dtype=np.float32)
            if rir.global_delay_samples > 0:
                impulse = np.pad(
                    impulse,
                    (int(rir.global_delay_samples), 0),
                )
            # Apply propagation level through HeardSoundEvent below. Normalize
            # the RIR here so distance attenuation is not applied twice; the
            # impulse shape still supplies direct sound and reverberation.
            impulse_peak = float(np.max(np.abs(impulse)))
            if impulse_peak <= 0.0 or not np.isfinite(impulse_peak):
                raise ValueError("RIR has no finite non-zero impulse")
            impulse = impulse / impulse_peak
            channels = [
                fftconvolve(sample.samples[:, channel], impulse, mode="full")
                for channel in range(sample.samples.shape[1])
            ]
            rendered = np.stack(channels, axis=1).astype(np.float32)
            return (
                replace(
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
                    f"for {msg.asset_id!r}: {exc}; "
                    f"using "
                    f"{'dry sample' if self._rir_dry_fallback else 'silence'}"
                )
            fallback = (
                sample
                if self._rir_dry_fallback
                else replace(sample, samples=np.zeros_like(sample.samples))
            )
            return (
                fallback,
                "dry_fallback" if self._rir_dry_fallback else "silence",
                reason,
            )
    
    def destroy_node(self) -> bool:
        self._asset_loader.shutdown(wait=False, cancel_futures=True)
        self._world_loader.shutdown(wait=False, cancel_futures=True)
        self._mixer.close()
        return super().destroy_node()
        

def main() -> None:
    rclpy.init()
    node = HumanSoundPlaybackNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
