from __future__ import annotations

import math
import time
import zlib
from collections.abc import Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import attrs
import rclpy
from ament_index_python.packages import get_package_share_directory
from arena_simulation_setup.tree.World import WorldIdentifier
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import ColorRGBA, String
from task_generator_msgs.msg import (
    ContinuousAudioSourceState,
    ContinuousHeardSoundState,
    HeardSoundEvent,
)
from visualization_msgs.msg import Marker, MarkerArray

from .acoustic_plot import (
    AcousticPlotDashboard,
    AcousticPlotSnapshot,
    Position3D,
    make_static_room_spec,
)
from .acoustic_room_spec import (
    AcousticRoomSpec,
    AcousticRoomSpecBuilder,
    AcousticRoomSpecConfig,
)
from .acoustic_world_graph import AcousticWorldGraph
from .material_catalog import AcousticMaterialCatalog
from .portal_coupling import MultiPortalRirCoupler, PortalCouplingConfig
from .pyroomacoustics_adapter import (
    PyroomacousticsAdapter,
    PyroomacousticsConfig,
)
from .qos_profiles import (
    acoustic_metadata_qos,
    continuous_audio_qos,
    transient_event_qos,
)

#: Both heard-event flavours the visualizer draws.
HeardEventMsg = ContinuousHeardSoundState | HeardSoundEvent


@attrs.frozen
class _LoadedAcousticWorld:
    name: str
    room_specs: tuple[AcousticRoomSpec, ...]
    graph: AcousticWorldGraph
    adapter: PyroomacousticsAdapter | None
    portal_coupler: MultiPortalRirCoupler | None


@attrs.frozen
class _PlotRequest:
    world: _LoadedAcousticWorld
    source_position_m: Position3D
    listener_position_m: Position3D
    source_zone: str
    listener_zone: str
    traversed_zones: tuple[str, ...]
    portal_positions_m: tuple[Position3D, ...]
    backend: str
    label: str


class SoundPropagationVisualizer(Node):
    """Publish RViz geometry and optionally plot live acoustic RIRs."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("sound_propagation_visualizer", **kwargs)
        share = Path(get_package_share_directory("task_generator"))
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
        self.declare_parameter(
            "continuous_heard_sounds_topic",
            "continuous_heard_sounds",
        )
        self.declare_parameter(
            "pedestrian_marker_topic",
            "pedestrian_sound_propagation_markers",
        )
        self.declare_parameter(
            "robot_marker_topic",
            "robot_sound_propagation_markers",
        )
        self.declare_parameter("room_marker_topic", "acoustic_room_markers")
        self.declare_parameter(
            "environment_source_marker_topic",
            "environment_audio_source_markers",
        )
        self.declare_parameter(
            "continuous_audio_sources_topic",
            "continuous_audio_sources",
        )
        self.declare_parameter("continuous_marker_lifetime_sec", 0.5)
        self.declare_parameter("continuous_listener_id", "")
        self.declare_parameter("world_topic", "state/world")
        self.declare_parameter("robot_fleet_topic", "state/robots")
        self.declare_parameter("sync_robot_listener_to_tf", False)
        self.declare_parameter("marker_lifetime_sec", 2.0)
        self.declare_parameter("path_z_m", 1.0)

        self.declare_parameter("plot_mode", "off")
        self.declare_parameter("plot_listener_id", "")
        self.declare_parameter("plot_update_rate_hz", 2.0)
        self.declare_parameter("plot_position_quantization_m", 0.10)
        self.declare_parameter("plot_energy_bin_ms", 5.0)
        self.declare_parameter("plot_early_window_sec", 0.08)
        self.declare_parameter(
            "material_catalog",
            str(share / "config" / "auditory" / "acoustic_materials.yaml"),
        )
        self.declare_parameter("rir_sample_rate_hz", 44100)
        self.declare_parameter("rir_max_order", 3)
        self.declare_parameter("rir_temperature_c", 20.0)
        self.declare_parameter("rir_relative_humidity_percent", 50.0)
        self.declare_parameter("rir_ceiling_height_m", 3.0)
        self.declare_parameter("rir_cache_position_quantization_m", 0.10)
        self.declare_parameter("rir_cache_size", 512)
        self.declare_parameter("portal_adjacency_tolerance_m", 0.2)
        self.declare_parameter("portal_inset_m", 0.03)
        self.declare_parameter("portal_loss_db", 3.0)
        self.declare_parameter("opening_portal_loss_db", 0.5)
        self.declare_parameter("derive_opening_portals", True)
        self.declare_parameter("minimum_opening_width_m", 0.2)
        self.declare_parameter("enable_multi_portal_rir", True)
        self.declare_parameter("max_portal_hops", 4)
        self.declare_parameter("route_distance_loss_db_per_m", 0.05)
        self.declare_parameter("portal_source_early_window_sec", 0.08)
        self.declare_parameter("portal_max_rir_duration_sec", 2.0)
        self.declare_parameter("portal_position_quantization_m", 0.10)
        self.declare_parameter("portal_rir_cache_size", 256)

        self.declare_parameter(
            "static_room_corners_xy",
            [0.0, 0.0, 8.0, 0.0, 8.0, 6.0, 0.0, 6.0],
        )
        self.declare_parameter("static_room_height_m", 3.0)
        self.declare_parameter(
            "static_source_position_m",
            [2.0, 2.0, 1.60],
        )
        self.declare_parameter(
            "static_listener_position_m",
            [6.0, 4.0, 0.35],
        )
        self.declare_parameter(
            "static_wall_material_id",
            "Acoustic_Default_Wall",
        )
        self.declare_parameter(
            "static_floor_material_id",
            "Acoustic_Default_Floor",
        )
        self.declare_parameter(
            "static_ceiling_material_id",
            "Acoustic_Default_Ceiling",
        )

        self._previous_portal_count = {
            "pedestrian": 0,
            "robot": 0,
        }
        self._pedestrian_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("pedestrian_marker_topic").value),
            10,
        )
        self._robot_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("robot_marker_topic").value),
            10,
        )
        self._room_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("room_marker_topic").value),
            acoustic_metadata_qos(),
        )
        self._environment_source_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("environment_source_marker_topic").value),
            acoustic_metadata_qos(depth=32),
        )
        self.create_subscription(
            HeardSoundEvent,
            str(self.get_parameter("heard_sound_events_topic").value),
            self._callback,
            transient_event_qos(),
        )
        self.create_subscription(
            ContinuousAudioSourceState,
            str(self.get_parameter("continuous_audio_sources_topic").value),
            self._on_continuous_source,
            continuous_audio_qos(),
        )
        self.create_subscription(
            ContinuousHeardSoundState,
            str(self.get_parameter("continuous_heard_sounds_topic").value),
            self._on_continuous_sound,
            continuous_audio_qos(),
        )
        self.create_subscription(
            String,
            str(self.get_parameter("world_topic").value),
            self._on_world,
            acoustic_metadata_qos(),
        )

        self._plot_mode = str(self.get_parameter("plot_mode").value).strip()
        if self._plot_mode not in {"off", "static", "live"}:
            raise ValueError("plot_mode must be 'off', 'static', or 'live'")
        self._dashboard: AcousticPlotDashboard | None = None
        if self._plot_mode != "off":
            self._dashboard = AcousticPlotDashboard(
                energy_bin_ms=float(self.get_parameter("plot_energy_bin_ms").value),
                early_window_sec=float(self.get_parameter("plot_early_window_sec").value),
            )

        self._world_loader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="acoustic_visualizer_world",
        )
        self._rir_worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="acoustic_visualizer_rir",
        )
        self._world_future: Future[_LoadedAcousticWorld] | None = None
        self._rir_future: Future[AcousticPlotSnapshot] | None = None
        self._loaded_world: _LoadedAcousticWorld | None = None
        self._pending_world_name = ""
        self._pending_plot_request: _PlotRequest | None = None
        self._last_plot_signature: tuple[Hashable, ...] | None = None
        self._last_plot_submit_time = 0.0
        self._selected_listener_id = ""
        self._continuous_visual_listener_id = ""
        self._last_frame = ""
        self._room_marker_frame = ""
        self._reported_plot_errors: set[str] = set()
        self._background_timer = self.create_timer(
            0.05,
            self._poll_background_work,
        )

        if self._plot_mode == "static":
            self._start_static_plot()

    def _callback(self, msg: HeardSoundEvent) -> None:
        if bool(msg.used_backend_fallback) and str(msg.backend_fallback_reason).strip() == "acoustic_scene_not_loaded":
            return
        backend = str(msg.propagation_backend).strip() or "unknown"
        output = self._listener_output(str(msg.listener_id))
        if output is None:
            self.get_logger().warning(f"cannot visualize sound for unknown listener {msg.listener_id!r}")
            return
        publisher, listener_kind, color = output
        frame = str(msg.header.frame_id).strip()
        if not frame:
            self.get_logger().warning("cannot visualize sound without the odometry frame ID")
            return
        self._last_frame = frame
        self._publish_room_geometry(frame)
        marker_id = 0
        lifetime = float(self.get_parameter("marker_lifetime_sec").value)

        source = Point(
            x=float(msg.source_position.x),
            y=float(msg.source_position.y),
            z=self._source_height(msg),
        )
        listener = Point(
            x=float(msg.listener_position.x),
            y=float(msg.listener_position.y),
            z=self._listener_height(
                str(msg.listener_id),
                float(msg.listener_position.z),
            ),
        )
        portal_points = [
            Point(
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
            )
            for point in msg.portal_positions
        ]
        if not portal_points and str(msg.portal_id).strip():
            portal_points = [
                Point(
                    x=float(msg.portal_position.x),
                    y=float(msg.portal_position.y),
                    z=float(msg.portal_position.z),
                )
            ]
        points = [source, *portal_points, listener]

        path = self._marker(frame, marker_id, Marker.LINE_STRIP, lifetime)
        path.ns = f"{listener_kind}_sound_propagation_path"
        path.scale.x = 0.07
        path.color = color
        path.points = points

        source_marker = self._marker(frame, marker_id + 1, Marker.SPHERE, lifetime)
        source_marker.ns = f"{listener_kind}_sound_source"
        source_marker.pose.position = source
        source_marker.scale.x = source_marker.scale.y = source_marker.scale.z = 0.22
        source_marker.color = color

        listener_marker = self._marker(frame, marker_id + 2, Marker.SPHERE, lifetime)
        listener_marker.ns = f"{listener_kind}_sound_listener"
        listener_marker.pose.position = listener
        listener_marker.scale.x = listener_marker.scale.y = listener_marker.scale.z = 0.22
        listener_marker.color = color

        text = self._marker(frame, marker_id + 3, Marker.TEXT_VIEW_FACING, lifetime)
        text.ns = f"{listener_kind}_sound_propagation_backend"
        text.pose.position = listener
        text.pose.position.z += 0.35
        text.scale.z = 0.24
        text.color = color
        text.text = backend
        portal_ids = list(msg.portal_ids)
        if not portal_ids and msg.portal_id:
            portal_ids = [msg.portal_id]
        if portal_ids:
            text.text += f"\n{len(portal_ids)} portal(s), loss={float(msg.portal_route_loss_db):.1f} dB"
            text.text += "\n" + " -> ".join(portal_ids)
        if msg.used_backend_fallback:
            text.text += f"\nfallback: {msg.backend_fallback_reason}"

        markers = [path, source_marker, listener_marker, text]
        for index, portal_point in enumerate(portal_points):
            portal = self._marker(
                frame,
                marker_id + 4 + index,
                Marker.CUBE,
                lifetime,
            )
            portal.ns = f"{listener_kind}_acoustic_portal"
            portal.pose.position = portal_point
            portal.scale.x = portal.scale.y = portal.scale.z = 0.28
            portal.color = color
            markers.append(portal)
        previous_portals = self._previous_portal_count.get(listener_kind, 0)
        for index in range(len(portal_points), previous_portals):
            stale = self._marker(
                frame,
                marker_id + 4 + index,
                Marker.CUBE,
                0.0,
            )
            stale.ns = f"{listener_kind}_acoustic_portal"
            stale.action = Marker.DELETE
            markers.append(stale)
        self._previous_portal_count[listener_kind] = len(portal_points)
        publisher.publish(MarkerArray(markers=markers))
        self._consider_live_plot(msg)

    def _on_continuous_sound(self, msg: ContinuousHeardSoundState) -> None:
        if not msg.active:
            return
        frame = str(msg.header.frame_id).strip()
        if frame:
            self._last_frame = frame
            self._publish_room_geometry(frame)
        self._publish_continuous_path(msg)
        self._consider_live_plot(msg)

    def _on_continuous_source(self, msg: ContinuousAudioSourceState) -> None:
        if str(msg.source_backend) != "wav_loop":
            return
        frame = str(msg.header.frame_id).strip()
        if not frame:
            return
        lifetime = float(self.get_parameter("continuous_marker_lifetime_sec").value)
        base_id = self._stable_marker_base(msg.source_id, 2)
        sound_type = str(msg.sound_type).lower()
        if not msg.active:
            color = ColorRGBA(r=0.45, g=0.45, b=0.45, a=0.65)
        elif "alarm" in sound_type or "siren" in sound_type:
            color = ColorRGBA(r=1.0, g=0.08, b=0.05, a=0.95)
        else:
            color = ColorRGBA(r=0.05, g=0.75, b=0.95, a=0.92)

        source = self._marker(frame, base_id, Marker.CUBE, lifetime)
        source.ns = "environment_audio_sources"
        source.pose.position = Point(
            x=float(msg.source_position.x),
            y=float(msg.source_position.y),
            z=float(msg.source_position.z),
        )
        source.pose.orientation.z = math.sin(float(msg.source_yaw) / 2.0)
        source.pose.orientation.w = math.cos(float(msg.source_yaw) / 2.0)
        source.scale.x = 0.42
        source.scale.y = 0.26
        source.scale.z = 0.22
        source.color = color

        label = self._marker(
            frame,
            base_id + 1,
            Marker.TEXT_VIEW_FACING,
            lifetime,
        )
        label.ns = "environment_audio_source_labels"
        label.pose.position = Point(
            x=float(msg.source_position.x),
            y=float(msg.source_position.y),
            z=float(msg.source_position.z) + 0.32,
        )
        label.scale.z = 0.22
        label.color = color
        state = "ACTIVE" if msg.active else "OFF"
        label.text = f"{msg.label or msg.group_id} / {msg.source_agent_name} [{state}]"
        self._environment_source_publisher.publish(MarkerArray(markers=[source, label]))

    def _publish_continuous_path(
        self,
        msg: ContinuousHeardSoundState,
    ) -> None:
        configured = str(self.get_parameter("continuous_listener_id").value).strip()
        listener_id = str(msg.listener_id)
        if configured:
            if listener_id != configured:
                return
        else:
            if not self._continuous_visual_listener_id:
                self._continuous_visual_listener_id = listener_id
            if listener_id != self._continuous_visual_listener_id:
                return
        output = self._listener_output(listener_id)
        if output is None:
            return
        publisher, listener_kind, color = output
        frame = str(msg.header.frame_id).strip()
        if not frame:
            return
        lifetime = float(self.get_parameter("continuous_marker_lifetime_sec").value)
        base_id = self._stable_marker_base(
            f"{msg.source_id}|{listener_id}",
            4 + len(msg.portal_positions),
        )
        source = Point(
            x=float(msg.source_position.x),
            y=float(msg.source_position.y),
            z=self._source_height(msg),
        )
        listener = Point(
            x=float(msg.listener_position.x),
            y=float(msg.listener_position.y),
            z=self._listener_height(
                listener_id,
                float(msg.listener_position.z),
            ),
        )
        portals = [Point(x=float(point.x), y=float(point.y), z=float(point.z)) for point in msg.portal_positions]

        path = self._marker(frame, base_id, Marker.LINE_STRIP, lifetime)
        path.ns = f"{listener_kind}_continuous_audio_paths"
        path.scale.x = 0.07
        path.color = color
        path.points = [source, *portals, listener]

        source_marker = self._marker(
            frame,
            base_id + 1,
            Marker.SPHERE,
            lifetime,
        )
        source_marker.ns = f"{listener_kind}_continuous_audio_sources"
        source_marker.pose.position = source
        source_marker.scale.x = 0.22
        source_marker.scale.y = 0.22
        source_marker.scale.z = 0.22
        source_marker.color = color

        listener_marker = self._marker(
            frame,
            base_id + 2,
            Marker.SPHERE,
            lifetime,
        )
        listener_marker.ns = f"{listener_kind}_continuous_audio_listeners"
        listener_marker.pose.position = listener
        listener_marker.scale.x = 0.22
        listener_marker.scale.y = 0.22
        listener_marker.scale.z = 0.22
        listener_marker.color = color

        label = self._marker(
            frame,
            base_id + 3,
            Marker.TEXT_VIEW_FACING,
            lifetime,
        )
        label.ns = f"{listener_kind}_continuous_audio_labels"
        label.pose.position = listener
        label.pose.position.z += 0.35
        label.scale.z = 0.22
        label.color = color
        label.text = f"{msg.label or msg.sound_type}: {msg.propagation_backend}\n{float(msg.direct_delay_sec) * 1000.0:.1f} ms, {len(portals)} portal(s)"

        markers = [path, source_marker, listener_marker, label]
        for index, point in enumerate(portals):
            portal = self._marker(
                frame,
                base_id + 4 + index,
                Marker.CUBE,
                lifetime,
            )
            portal.ns = f"{listener_kind}_continuous_audio_portals"
            portal.pose.position = point
            portal.scale.x = 0.28
            portal.scale.y = 0.28
            portal.scale.z = 0.28
            portal.color = color
            markers.append(portal)
        publisher.publish(MarkerArray(markers=markers))

    def _on_world(self, msg: String) -> None:
        world_name = str(msg.data).strip()
        if not world_name:
            return
        if self._loaded_world is not None and world_name == self._loaded_world.name:
            return
        self._pending_world_name = world_name
        self._pending_plot_request = None
        self._last_plot_signature = None
        if self._world_future is not None and not self._world_future.done():
            return
        self._start_world_load(world_name)

    def _start_world_load(self, world_name: str) -> None:
        self.get_logger().info(f"loading acoustic geometry for visualization of {world_name!r}")
        self._world_future = self._world_loader.submit(
            self._load_world,
            world_name,
            float(self.get_parameter("rir_ceiling_height_m").value),
            float(self.get_parameter("portal_adjacency_tolerance_m").value),
            bool(self.get_parameter("derive_opening_portals").value),
            float(self.get_parameter("minimum_opening_width_m").value),
            float(self.get_parameter("portal_loss_db").value),
            float(self.get_parameter("opening_portal_loss_db").value),
            self._plot_mode == "live",
            Path(str(self.get_parameter("material_catalog").value)),
            self._rir_config(),
            self._portal_config(),
        )

    @staticmethod
    def _load_world(
        world_name: str,
        ceiling_height_m: float,
        adjacency_tolerance_m: float,
        derive_opening_portals: bool,
        minimum_opening_width_m: float,
        door_portal_loss_db: float,
        opening_portal_loss_db: float,
        enable_rir: bool,
        material_catalog_path: Path,
        rir_config: PyroomacousticsConfig,
        portal_config: PortalCouplingConfig,
    ) -> _LoadedAcousticWorld:
        world_view = WorldIdentifier(world_name).resolve_sync()
        world = world_view.load()
        level_origins = world_view.level_origins()
        if level_origins is not None:
            world = world.compact_world(level_origins)
        room_specs = AcousticRoomSpecBuilder(AcousticRoomSpecConfig(ceiling_height_m=ceiling_height_m)).from_world(world)
        graph = AcousticWorldGraph.from_world(
            world,
            room_specs,
            adjacency_tolerance_m=adjacency_tolerance_m,
            derive_opening_portals=derive_opening_portals,
            minimum_opening_width_m=minimum_opening_width_m,
            door_portal_loss_db=door_portal_loss_db,
            opening_portal_loss_db=opening_portal_loss_db,
        )
        adapter = (
            PyroomacousticsAdapter(
                AcousticMaterialCatalog(material_catalog_path),
                rir_config,
            )
            if enable_rir
            else None
        )
        portal_coupler = (
            MultiPortalRirCoupler(
                adapter,
                graph,
                world_name=world_name,
                config=portal_config,
            )
            if adapter is not None
            else None
        )
        return _LoadedAcousticWorld(
            name=world_name,
            room_specs=room_specs,
            graph=graph,
            adapter=adapter,
            portal_coupler=portal_coupler,
        )

    def _poll_background_work(self) -> None:
        if self._dashboard is not None:
            self._dashboard.pump_events()
        self._poll_world_load()
        self._poll_rir()
        self._submit_pending_plot()

    def _poll_world_load(self) -> None:
        if self._world_future is None or not self._world_future.done():
            return
        future = self._world_future
        self._world_future = None
        try:
            loaded_world = future.result()
        except Exception as exc:
            self.get_logger().error(f"failed to load acoustic visualization geometry: {exc!r}")
            return
        if self._room_marker_frame:
            self._clear_room_geometry(self._room_marker_frame)
        self._loaded_world = loaded_world
        self._room_marker_frame = ""
        if self._last_frame:
            self._publish_room_geometry(self._last_frame)
        self.get_logger().info(f"loaded {len(self._loaded_world.room_specs)} acoustic rooms for visualization of {self._loaded_world.name!r}")
        if self._pending_world_name and self._pending_world_name != self._loaded_world.name:
            self._start_world_load(self._pending_world_name)

    def _poll_rir(self) -> None:
        if self._rir_future is None or not self._rir_future.done():
            return
        future = self._rir_future
        self._rir_future = None
        try:
            snapshot = future.result()
            if self._dashboard is not None:
                self._dashboard.update(snapshot)
        except Exception as exc:
            self._last_plot_signature = None
            key = f"{type(exc).__name__}: {exc}"
            if key not in self._reported_plot_errors:
                self._reported_plot_errors.add(key)
                self.get_logger().warning(f"RIR plot update failed: {key}")

    def _consider_live_plot(self, msg: HeardEventMsg) -> None:
        if self._plot_mode != "live" or self._loaded_world is None:
            return
        if not str(msg.header.frame_id).strip():
            return
        if self._pending_world_name and self._pending_world_name != self._loaded_world.name:
            return
        listener_id = str(msg.listener_id)
        configured_listener = str(self.get_parameter("plot_listener_id").value).strip()
        if configured_listener:
            if listener_id != configured_listener:
                return
        else:
            if not listener_id.startswith("robot:"):
                return
            if not self._selected_listener_id:
                self._selected_listener_id = listener_id
                self.get_logger().info(f"live RIR plot selected listener {listener_id!r}")
            if listener_id != self._selected_listener_id:
                return

        if bool(msg.used_backend_fallback):
            reason = str(msg.backend_fallback_reason).strip() or "unknown"
            key = f"fallback:{reason}"
            if key not in self._reported_plot_errors:
                self._reported_plot_errors.add(key)
                self.get_logger().warning(f"live RIR plot skipped propagation fallback: {reason}")
            return

        graph = self._loaded_world.graph
        source_x = float(msg.source_position.x)
        source_y = float(msg.source_position.y)
        listener_x = float(msg.listener_position.x)
        listener_y = float(msg.listener_position.y)
        source_zone = str(msg.source_zone).strip() or (graph.zone_at_xy(source_x, source_y) or "")
        listener_zone = str(msg.listener_zone).strip() or (graph.zone_at_xy(listener_x, listener_y) or "")
        if not source_zone or not listener_zone:
            key = "source_or_listener_outside_acoustic_zones"
            if key not in self._reported_plot_errors:
                self._reported_plot_errors.add(key)
                self.get_logger().warning("live RIR plot skipped because the source or listener is outside the acoustic zones")
            return

        source = (source_x, source_y, self._source_height(msg))
        listener = (
            listener_x,
            listener_y,
            self._listener_height(
                listener_id,
                float(msg.listener_position.z),
            ),
        )
        portal_positions = tuple((float(point.x), float(point.y), float(point.z)) for point in msg.portal_positions)
        request = _PlotRequest(
            world=self._loaded_world,
            source_position_m=source,
            listener_position_m=listener,
            source_zone=source_zone,
            listener_zone=listener_zone,
            traversed_zones=tuple(str(zone) for zone in msg.traversed_zones),
            portal_positions_m=portal_positions,
            backend=str(msg.propagation_backend).strip() or "pyroomacoustics",
            label=listener_id,
        )
        quantization = max(
            float(self.get_parameter("plot_position_quantization_m").value),
            1e-6,
        )
        signature = (
            self._loaded_world.name,
            source_zone,
            listener_zone,
            tuple(round(value / quantization) for value in source),
            tuple(round(value / quantization) for value in listener),
        )
        if signature == self._last_plot_signature:
            return
        self._last_plot_signature = signature
        self._pending_plot_request = request

    def _submit_pending_plot(self) -> None:
        if self._pending_plot_request is None or self._rir_future is not None:
            return
        update_rate = max(
            float(self.get_parameter("plot_update_rate_hz").value),
            0.1,
        )
        now = time.monotonic()
        if now - self._last_plot_submit_time < 1.0 / update_rate:
            return
        request = self._pending_plot_request
        self._pending_plot_request = None
        self._last_plot_submit_time = now
        self._rir_future = self._rir_worker.submit(
            self._compute_live_snapshot,
            request,
        )

    @staticmethod
    def _compute_live_snapshot(request: _PlotRequest) -> AcousticPlotSnapshot:
        adapter = request.world.adapter
        if adapter is None:
            raise LookupError("pyroomacoustics adapter is not initialized")
        traversed_zones = request.traversed_zones
        portal_positions = request.portal_positions_m
        backend = request.backend
        if request.source_zone == request.listener_zone:
            room = request.world.graph.room(request.source_zone)
            if room is None:
                raise LookupError(f"no acoustic room for zone {request.source_zone!r}")
            rir = adapter.compute_rir(
                room,
                source_position_m=request.source_position_m,
                listener_position_m=request.listener_position_m,
            )
            traversed_zones = (request.source_zone,)
            portal_positions = ()
            backend = "pyroomacoustics_same_room"
        else:
            coupler = request.world.portal_coupler
            if coupler is None:
                raise LookupError("portal RIR coupler is not initialized")
            result = coupler.compute(
                source_zone=request.source_zone,
                listener_zone=request.listener_zone,
                source_position_m=request.source_position_m,
                listener_position_m=request.listener_position_m,
            )
            rir = result.rir
            if result.route is not None:
                traversed_zones = result.route.zones
                backend = "pyroomacoustics_one_door" if result.route.hop_count == 1 else "pyroomacoustics_multi_portal"
            portal_positions = result.portal_positions
        return AcousticPlotSnapshot(
            room_specs=request.world.room_specs,
            source_position_m=request.source_position_m,
            listener_position_m=request.listener_position_m,
            rir=rir,
            backend=backend,
            source_zone=request.source_zone,
            listener_zone=request.listener_zone,
            traversed_zones=traversed_zones,
            portal_positions_m=portal_positions,
            label=request.label,
        )

    def _start_static_plot(self) -> None:
        corners = tuple(float(value) for value in self.get_parameter("static_room_corners_xy").value)
        if len(corners) < 6 or len(corners) % 2:
            raise ValueError("static_room_corners_xy must contain at least three x,y pairs")
        corner_pairs = tuple(zip(corners[::2], corners[1::2], strict=True))
        specification = make_static_room_spec(
            corner_pairs,
            ceiling_height_m=float(self.get_parameter("static_room_height_m").value),
            wall_material_id=str(self.get_parameter("static_wall_material_id").value),
            floor_material_id=str(self.get_parameter("static_floor_material_id").value),
            ceiling_material_id=str(self.get_parameter("static_ceiling_material_id").value),
        )
        source = self._position_parameter("static_source_position_m")
        listener = self._position_parameter("static_listener_position_m")
        adapter = PyroomacousticsAdapter(
            AcousticMaterialCatalog(Path(str(self.get_parameter("material_catalog").value))),
            self._rir_config(),
        )
        self._rir_future = self._rir_worker.submit(
            self._compute_static_snapshot,
            adapter,
            specification,
            source,
            listener,
        )

    @staticmethod
    def _compute_static_snapshot(
        adapter: PyroomacousticsAdapter,
        specification: AcousticRoomSpec,
        source: Position3D,
        listener: Position3D,
    ) -> AcousticPlotSnapshot:
        rir = adapter.compute_rir(
            specification,
            source_position_m=source,
            listener_position_m=listener,
        )
        return AcousticPlotSnapshot(
            room_specs=(specification,),
            source_position_m=source,
            listener_position_m=listener,
            rir=rir,
            backend="pyroomacoustics_static",
            source_zone=specification.zone_name,
            listener_zone=specification.zone_name,
            traversed_zones=(specification.zone_name,),
            label="fixed source and listener",
        )

    def _publish_room_geometry(self, frame: str) -> None:
        if self._loaded_world is None or not frame or frame == self._room_marker_frame:
            return
        markers: list[Marker] = []
        marker_id = 0
        for room in self._loaded_world.room_specs:
            floor = self._marker(frame, marker_id, Marker.LINE_STRIP, 0.0)
            marker_id += 1
            floor.ns = "acoustic_zone_floor"
            floor.scale.x = 0.045
            floor.color = ColorRGBA(r=0.15, g=0.55, b=0.85, a=0.75)
            floor.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in (*room.corners_xy, room.corners_xy[0])]
            markers.append(floor)

            walls = self._marker(frame, marker_id, Marker.LINE_LIST, 0.0)
            marker_id += 1
            walls.ns = "acoustic_room_walls"
            walls.scale.x = 0.035
            walls.color = ColorRGBA(r=0.55, g=0.60, b=0.65, a=0.65)
            for boundary in room.boundary:
                sx, sy = boundary.start
                ex, ey = boundary.end
                height = room.ceiling_height_m
                walls.points.extend(
                    [
                        Point(x=sx, y=sy, z=0.0),
                        Point(x=ex, y=ey, z=0.0),
                        Point(x=sx, y=sy, z=height),
                        Point(x=ex, y=ey, z=height),
                        Point(x=sx, y=sy, z=0.0),
                        Point(x=sx, y=sy, z=height),
                    ]
                )
            markers.append(walls)

            label = self._marker(frame, marker_id, Marker.TEXT_VIEW_FACING, 0.0)
            marker_id += 1
            label.ns = "acoustic_zone_names"
            corners = room.corners_xy
            label.pose.position = Point(
                x=sum(point[0] for point in corners) / len(corners),
                y=sum(point[1] for point in corners) / len(corners),
                z=0.12,
            )
            label.scale.z = 0.22
            label.color = ColorRGBA(r=0.10, g=0.25, b=0.45, a=0.90)
            label.text = room.zone_name
            markers.append(label)
        self._room_publisher.publish(MarkerArray(markers=markers))
        self._room_marker_frame = frame

    def _clear_room_geometry(self, frame: str) -> None:
        marker = self._marker(frame, 0, Marker.DELETEALL, 0.0)
        marker.action = Marker.DELETEALL
        self._room_publisher.publish(MarkerArray(markers=[marker]))

    def _rir_config(self) -> PyroomacousticsConfig:
        return PyroomacousticsConfig(
            sample_rate_hz=int(self.get_parameter("rir_sample_rate_hz").value),
            max_order=int(self.get_parameter("rir_max_order").value),
            temperature_c=float(self.get_parameter("rir_temperature_c").value),
            relative_humidity_percent=float(self.get_parameter("rir_relative_humidity_percent").value),
            cache_position_quantization_m=float(self.get_parameter("rir_cache_position_quantization_m").value),
            cache_size=int(self.get_parameter("rir_cache_size").value),
        )

    def _portal_config(self) -> PortalCouplingConfig:
        return PortalCouplingConfig(
            portal_inset_m=float(self.get_parameter("portal_inset_m").value),
            portal_loss_db=float(self.get_parameter("portal_loss_db").value),
            source_room_early_window_sec=float(self.get_parameter("portal_source_early_window_sec").value),
            maximum_rir_duration_sec=float(self.get_parameter("portal_max_rir_duration_sec").value),
            position_quantization_m=float(self.get_parameter("portal_position_quantization_m").value),
            cache_size=int(self.get_parameter("portal_rir_cache_size").value),
            opening_portal_loss_db=float(self.get_parameter("opening_portal_loss_db").value),
            max_portal_hops=(int(self.get_parameter("max_portal_hops").value) if bool(self.get_parameter("enable_multi_portal_rir").value) else 1),
            route_distance_loss_db_per_m=float(self.get_parameter("route_distance_loss_db_per_m").value),
        )

    def _position_parameter(self, name: str) -> Position3D:
        values = tuple(float(value) for value in self.get_parameter(name).value)
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly x, y, and z")
        return values

    @staticmethod
    def _source_height(msg: HeardEventMsg) -> float:
        if isinstance(msg, ContinuousHeardSoundState) and str(msg.source_backend) == "wav_loop":
            return float(msg.source_position.z)
        sound_type = str(msg.sound_type).lower()
        if "foot" in sound_type or "step" in sound_type:
            return 0.05
        if "motor" in sound_type or "robot" in sound_type:
            return 0.25
        return 1.60

    @staticmethod
    def _listener_height(listener_id: str, listener_z: float = 0.0) -> float:
        if listener_id.startswith("microphone") or listener_id.endswith("_mic"):
            return listener_z
        return 0.35 if listener_id.startswith("robot:") else 1.60

    @staticmethod
    def _stable_marker_base(key: str, width: int) -> int:
        maximum = max((2**31 - 1) // max(width, 1), 1)
        return (zlib.crc32(key.encode()) % maximum) * max(width, 1)

    def _marker(
        self,
        frame: str,
        marker_id: int,
        marker_type: int,
        lifetime_sec: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        seconds = max(lifetime_sec, 0.0)
        marker.lifetime.sec = int(seconds)
        marker.lifetime.nanosec = int((seconds % 1.0) * 1_000_000_000)
        return marker

    def _listener_output(
        self,
        listener_id: str,
    ) -> tuple[Publisher, str, ColorRGBA] | None:
        if listener_id.startswith("agent:"):
            return (
                self._pedestrian_publisher,
                "pedestrian",
                ColorRGBA(r=0.10, g=0.45, b=1.0, a=0.92),
            )
        if listener_id.startswith("robot:"):
            return (
                self._robot_publisher,
                "robot",
                ColorRGBA(r=0.65, g=0.20, b=1.0, a=0.92),
            )
        if listener_id.startswith("microphone") or listener_id.endswith("_mic"):
            listener_kind = listener_id.replace(":", "_").replace("/", "_")
            return (
                self._robot_publisher,
                listener_kind,
                ColorRGBA(r=0.10, g=0.85, b=0.55, a=0.92),
            )
        return None

    def destroy_node(self) -> bool:
        self._world_loader.shutdown(wait=False, cancel_futures=True)
        self._rir_worker.shutdown(wait=False, cancel_futures=True)
        if self._dashboard is not None:
            self._dashboard.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SoundPropagationVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
