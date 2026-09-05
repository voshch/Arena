from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from collections.abc import Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_people_msgs.msg import Pedestrians
from arena_robots.Robot import RobotIdentifier
from arena_simulation_setup.tree.World import (
    MICROPHONE_PLACEMENT_TOLERANCE_M,
    WorldIdentifier,
)
from geometry_msgs.msg import Point, PoseStamped, Transform
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from shapely.geometry import Point as ShapelyPoint
from std_msgs.msg import ColorRGBA, String
from task_generator_msgs.msg import (
    AcousticPath,
    ContinuousAudioSourceState,
    ContinuousHeardSoundState,
    EpisodeRecord,
    HeardSoundEvent,
    RobotFleet,
    SoundEvent,
)
from task_generator_msgs.srv import RemoveMicrophone, SpawnMicrophone
from visualization_msgs.msg import Marker, MarkerArray

from task_generator.auditory.acoustic_frame import (
    realize_acoustic_geometry,
    runtime_acoustic_offset,
)
from task_generator.auditory.acoustic_room_spec import (
    AcousticRoomSpec,
    AcousticRoomSpecBuilder,
    AcousticRoomSpecConfig,
)
from task_generator.auditory.acoustic_scene import AcousticScene
from task_generator.auditory.acoustic_world_graph import (
    AcousticPortalRoute,
    AcousticWorldGraph,
)
from task_generator.auditory.material_catalog import AcousticMaterialCatalog
from task_generator.auditory.microphone_config import (
    WorldMicrophoneSpec,
    parse_robot_microphones,
    world_microphones,
)
from task_generator.auditory.portal_coupling import (
    MultiPortalRirCoupler,
    PortalCouplingConfig,
    PortalCouplingResult,
)
from task_generator.auditory.propagation import Level3Propagation
from task_generator.auditory.pyroomacoustics_adapter import (
    PyroomacousticsAdapter,
    PyroomacousticsConfig,
    PyroomacousticsUnavailableError,
    RoomImpulseResponse,
)
from task_generator.auditory.qos_profiles import (
    acoustic_metadata_qos,
    continuous_audio_qos,
    transient_event_qos,
)


class SoundPropagationNode(Node):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("sound_propagation_node", **kwargs)

        self.declare_parameter("sound_events_topic", "human_sound_events")
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
        self.declare_parameter(
            "continuous_audio_sources_topic",
            "continuous_audio_sources",
        )
        self.declare_parameter(
            "continuous_heard_sounds_topic",
            "continuous_heard_sounds",
        )
        self.declare_parameter("robot_microphones", "[]")
        self.declare_parameter("viewport_down_projection_height_m", 1.6)
        self.declare_parameter("enable_propagation", True)
        self.declare_parameter("active_microphone_id", "")
        self.declare_parameter(
            "microphone_listeners_topic",
            "microphone_listeners",
        )
        self.declare_parameter(
            "microphone_marker_topic",
            "microphone_markers",
        )
        self.declare_parameter("arena_peds_topic", "arena_peds")
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("robot_fleet_topic", "state/robots")
        self.declare_parameter("default_hearing_threshold_db", 10.0)
        self.declare_parameter("minimum_propagation_distance_m", 1.0)
        self.declare_parameter("self_hearing_distance_m", 0.3)
        self.declare_parameter("occlusion_penalty_db", 20.0)
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("publish_inaudible", True)
        self.declare_parameter("robots_hear_self", True)
        self.declare_parameter("world_topic", "state/world")
        self.declare_parameter("episode_topic", "state/episode")
        self.declare_parameter("odom_topic_template", "{namespace}/{name}_velocity_controller/odom")
        self.declare_parameter("robot_listener_frame", "")
        self.declare_parameter("propagation_backend", "level3")
        self.declare_parameter("pyroom_sample_rate_hz", 44100)
        self.declare_parameter("pyroom_max_order", 3)
        self.declare_parameter("pyroom_temperature_c", 20.0)
        self.declare_parameter("pyroom_relative_humidity_percent", 50.0)
        self.declare_parameter("pyroom_ceiling_height_m", 3.0)
        self.declare_parameter(
            "pyroom_cache_position_quantization_m",
            0.10,
        )
        self.declare_parameter("pyroom_cache_size", 512)
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
        self.declare_parameter("validate_zone_coverage", True)
        self.declare_parameter("zone_coverage_stride_cells", 10)
        self.declare_parameter("zone_coverage_tolerance_m", 0.10)
        self.declare_parameter("buffer_events_until_scene_loaded", True)
        self.declare_parameter("scene_event_buffer_size", 128)
        self.declare_parameter("ped_hearing", False)
        self.declare_parameter("compute_rir_in_propagation", True)
        self._scene: AcousticScene | None = None
        self._authored_scene: AcousticScene | None = None
        self._world_name = ""
        self._pending_world_name = ""
        self._world_load_future: Future | None = None
        self._world_loader = ThreadPoolExecutor(max_workers=1)
        self._world_subscription = self.create_subscription(
            String,
            str(self.get_parameter("world_topic").value),
            self._cb_world,
            acoustic_metadata_qos(),
        )
        self._episode_subscription = self.create_subscription(
            EpisodeRecord,
            str(self.get_parameter("episode_topic").value),
            self._cb_episode_world,
            acoustic_metadata_qos(depth=20),
        )
        self._world_load_timer = self.create_timer(0.1, self._poll_world_load)
        self._peds: dict[int, object] = {}
        self._peds_frame_id = "map"
        self._robots: dict[str, tuple[Point, str]] = {}
        self._robot_base_frames: dict[str, str] = {}
        self._robot_microphone_specs = parse_robot_microphones(str(self.get_parameter("robot_microphones").value))
        self._robot_microphones: dict[str, tuple[Point, str]] = {}
        self._world_microphones: dict[str, WorldMicrophoneSpec] = {}
        self._spawned_microphones: dict[str, tuple[Point, str]] = {}
        self._viewport_microphones: dict[str, tuple[Point, str]] = {}
        self._spawned_microphone_index = 0
        self._last_continuous_outputs: dict[tuple[str, str], ContinuousHeardSoundState] = {}
        self._continuous_propagation_signatures: dict[tuple[str, str], tuple[Hashable, ...]] = {}
        self._missing_microphone_robots: set[str] = set()
        self._acoustic_offset: tuple[float, float] | None = None
        self._episode_id = -1
        self._odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._tf_buffer = tf2_ros.Buffer(node=self)
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer,
            self,
        )
        self._transform_warning_times: dict[tuple[str, str, str], float] = {}
        self._map: OccupancyGrid | None = None
        self._world_graph: AcousticWorldGraph | None = None
        self._authored_room_specs: tuple[AcousticRoomSpec, ...] = ()
        self._authored_world_graph: AcousticWorldGraph | None = None
        self._portal_coupler: MultiPortalRirCoupler | None = None
        self._coverage_signature: tuple[Hashable, ...] | None = None
        self._authored_map_origin: tuple[float, float] | None = None
        self._acoustic_alignment_signature: tuple[Hashable, ...] | None = None
        self._reported_routes: set[tuple[str, str, str, str]] = set()
        self._pending_events: deque[tuple[SoundEvent, dict[str, Point]]] = deque()
        self._odom_subs: dict[
            tuple[str, str],
            rclpy.subscription.Subscription,
        ] = {}
        sound_events_topic = str(self.get_parameter("sound_events_topic").value)
        heard_sound_events_topic = str(self.get_parameter("heard_sound_events_topic").value)
        peds_topic = str(self.get_parameter("arena_peds_topic").value)
        map_topic = str(self.get_parameter("map_topic").value)
        robot_fleet_topic = str(self.get_parameter("robot_fleet_topic").value)
        self._heard_pub = self.create_publisher(HeardSoundEvent, heard_sound_events_topic, transient_event_qos())
        self._continuous_heard_pub = self.create_publisher(
            ContinuousHeardSoundState,
            str(self.get_parameter("continuous_heard_sounds_topic").value),
            continuous_audio_qos(),
        )
        self._microphone_registry_pub = self.create_publisher(
            String,
            str(self.get_parameter("microphone_listeners_topic").value),
            acoustic_metadata_qos(),
        )
        self._microphone_marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("microphone_marker_topic").value),
            acoustic_metadata_qos(),
        )
        self._publish_microphone_registry()
        self._spawn_microphone_service = self.create_service(
            SpawnMicrophone,
            "runtime/spawn_microphone",
            self._spawn_microphone,
        )
        self._remove_microphone_service = self.create_service(
            RemoveMicrophone,
            "runtime/remove_microphone",
            self._remove_microphone,
        )
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._microphone_marker_timer = self.create_timer(
            0.25,
            self._publish_microphone_markers,
        )
        self.create_subscription(SoundEvent, sound_events_topic, self._cb_sound_event, transient_event_qos())
        self.create_subscription(
            ContinuousAudioSourceState,
            str(self.get_parameter("continuous_audio_sources_topic").value),
            self._cb_continuous_source,
            continuous_audio_qos(),
        )
        self.create_subscription(Pedestrians, peds_topic, self._cb_peds, 10)
        self.create_subscription(
            PoseStamped,
            "/arena/viewport/camera_pose",
            self._cb_viewport_camera_pose,
            10,
        )
        self.create_subscription(
            OccupancyGrid,
            map_topic,
            self._cb_map,
            acoustic_metadata_qos(),
        )
        self.create_subscription(RobotFleet, robot_fleet_topic, self._cb_robot_fleet, acoustic_metadata_qos())
        share = Path(get_package_share_directory("task_generator"))
        materials = AcousticMaterialCatalog(share / "config" / "auditory" / "acoustic_materials.yaml")
        self._room_specs = ()
        backend = str(self.get_parameter("propagation_backend").value)
        self._requested_backend = backend

        self._pra_adapter = None

        if backend == "pyroomacoustics":
            try:
                # The adapter keeps PRA optional for the Level3 backend, but
                # fail fast when the user explicitly selected this backend.
                import pyroomacoustics  # noqa: F401

                pra_config = PyroomacousticsConfig(
                    sample_rate_hz=int(self.get_parameter("pyroom_sample_rate_hz").value),
                    max_order=int(self.get_parameter("pyroom_max_order").value),
                    temperature_c=float(self.get_parameter("pyroom_temperature_c").value),
                    relative_humidity_percent=float(self.get_parameter("pyroom_relative_humidity_percent").value),
                    cache_position_quantization_m=float(self.get_parameter("pyroom_cache_position_quantization_m").value),
                    cache_size=int(self.get_parameter("pyroom_cache_size").value),
                )
                self._pra_adapter = PyroomacousticsAdapter(
                    materials,
                    pra_config,
                )
            except (ImportError, PyroomacousticsUnavailableError, ValueError) as exc:
                raise RuntimeError(f"pyroomacoustics backend requested but could not be initialized: {exc}") from exc
        elif backend != "level3":
            raise ValueError("propagation_backend must be 'level3' or 'pyroomacoustics'")
        self._propagation = Level3Propagation(materials)

    def _on_set_parameters(
        self,
        parameters: list[Parameter],
    ) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name != "enable_propagation":
                if parameter.name == "active_microphone_id":
                    if parameter.type_ != Parameter.Type.STRING:
                        return SetParametersResult(
                            successful=False,
                            reason="active_microphone_id must be a string",
                        )
                    self._stop_continuous_outputs(str(parameter.value).strip())
                continue
            if parameter.type_ != Parameter.Type.BOOL:
                return SetParametersResult(
                    successful=False,
                    reason="enable_propagation must be a boolean",
                )
            if not bool(parameter.value):
                self._stop_continuous_outputs()
        return SetParametersResult(successful=True)

    def _stop_continuous_outputs(
        self,
        active_microphone_id: str | None = None,
    ) -> None:
        microphone_ids = set(self._robot_microphones) | set(self._world_microphones) | set(self._spawned_microphones) | set(self._viewport_microphones)
        for key, previous in tuple(self._last_continuous_outputs.items()):
            if active_microphone_id is not None:
                if key[1] not in microphone_ids:
                    continue
                if key[1] == active_microphone_id:
                    continue
            stopped = copy.deepcopy(previous)
            stopped.header.stamp = self.get_clock().now().to_msg()
            stopped.active = False
            stopped.audible = False
            self._continuous_propagation_signatures.pop(key, None)
            self._last_continuous_outputs[key] = stopped
            self._continuous_heard_pub.publish(stopped)

    def _cb_peds(self, msg: Pedestrians) -> None:
        self._peds = {int(p.id): p for p in msg.pedestrians}
        self._peds_frame_id = str(msg.header.frame_id).strip() or "map"

    def _cb_viewport_camera_pose(self, msg: PoseStamped) -> None:
        frame_id = str(msg.header.frame_id).strip().lstrip("/")
        position = msg.pose.position
        if not frame_id or not all(math.isfinite(value) for value in (position.x, position.y, position.z)):
            return
        down_projection_height = float(self.get_parameter("viewport_down_projection_height_m").value)
        if not math.isfinite(down_projection_height):
            return
        publish_registry = not self._viewport_microphones
        self._viewport_microphones = {
            "microphone:viewport:down_projection": (
                Point(
                    x=float(position.x),
                    y=float(position.y),
                    z=down_projection_height,
                ),
                frame_id,
            ),
            "microphone:viewport:projective_center": (
                Point(
                    x=float(position.x),
                    y=float(position.y),
                    z=float(position.z),
                ),
                frame_id,
            ),
        }
        if publish_registry:
            self._publish_microphone_registry()

    def _cb_world(self, msg: String) -> None:
        world_name = msg.data.strip()
        if not world_name or world_name == self._world_name:
            return

        self._pending_world_name = world_name

        if self._world_load_future is not None and not self._world_load_future.done():
            return

        self._start_world_load(world_name)

    def _cb_episode_world(self, msg: EpisodeRecord) -> None:
        """Recover the world even if a launch missed the standalone state topic."""
        episode_id = int(msg.episode_id)
        if episode_id != self._episode_id:
            self._episode_id = episode_id
            self._spawned_microphones.clear()
            self._spawned_microphone_index = 0
            self._last_continuous_outputs.clear()
            self._continuous_propagation_signatures.clear()
            self._publish_microphone_registry()
        world_name = str(msg.world).strip()
        if world_name:
            self._cb_world(String(data=world_name))

    def _start_world_load(self, world_name: str) -> None:
        self.get_logger().info(f"loading acoustic scene for world {world_name!r}")
        self._world_microphones.clear()
        self._spawned_microphones.clear()
        self._spawned_microphone_index = 0
        self._last_continuous_outputs.clear()
        self._continuous_propagation_signatures.clear()
        self._publish_microphone_registry()

        self._world_load_future = self._world_loader.submit(
            self._load_acoustic_scene,
            world_name,
            float(self.get_parameter("pyroom_ceiling_height_m").value),
            float(self.get_parameter("portal_adjacency_tolerance_m").value),
            bool(self.get_parameter("derive_opening_portals").value),
            float(self.get_parameter("minimum_opening_width_m").value),
            float(self.get_parameter("portal_loss_db").value),
            float(self.get_parameter("opening_portal_loss_db").value),
        )

    @staticmethod
    def _load_acoustic_scene(
        world_name: str,
        ceiling_height_m: float,
        adjacency_tolerance_m: float,
        derive_opening_portals: bool,
        minimum_opening_width_m: float,
        door_portal_loss_db: float,
        opening_portal_loss_db: float,
    ) -> tuple[
        str,
        AcousticScene,
        tuple[AcousticRoomSpec, ...],
        AcousticWorldGraph,
        tuple[float, float] | None,
        tuple[WorldMicrophoneSpec, ...],
    ]:
        world_view = WorldIdentifier(world_name).resolve_sync()
        world_description = world_view.load()
        authored_map_origin = None
        level_origins = world_view.level_origins()
        microphones = world_microphones(
            world_description,
            ceiling_height_m,
            level_origins=level_origins,
        )
        if level_origins is not None:
            world_description = world_description.compact_world(level_origins)
            _, authored_map_origin = world_description.render_grid()
        else:
            for level_id in sorted(world_description.levels):
                map_yaml = Path(world_view.path) / str(level_id) / "map.yaml"
                if not map_yaml.exists():
                    continue
                map_config = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
                origin = map_config.get("origin", (0.0, 0.0, 0.0))
                authored_map_origin = (float(origin[0]), float(origin[1]))
                break

        scene = AcousticScene.from_world(world_description)

        room_config = AcousticRoomSpecConfig(
            ceiling_height_m=ceiling_height_m,
        )
        room_specs = AcousticRoomSpecBuilder(room_config).from_world(world_description)
        graph = AcousticWorldGraph.from_world(
            world_description,
            room_specs,
            adjacency_tolerance_m=adjacency_tolerance_m,
            derive_opening_portals=derive_opening_portals,
            minimum_opening_width_m=minimum_opening_width_m,
            door_portal_loss_db=door_portal_loss_db,
            opening_portal_loss_db=opening_portal_loss_db,
        )

        return (
            world_name,
            scene,
            room_specs,
            graph,
            authored_map_origin,
            microphones,
        )

    def _poll_world_load(self) -> None:
        if self._world_load_future is None:
            return

        if not self._world_load_future.done():
            return

        future = self._world_load_future
        self._world_load_future = None

        try:
            (
                world_name,
                scene,
                room_specs,
                graph,
                authored_map_origin,
                microphones,
            ) = future.result()
        except Exception as exc:
            self.get_logger().error(f"failed to load acoustic scene: {exc!r}")
            return

        self._world_name = world_name
        self._authored_scene = scene
        self._authored_room_specs = room_specs
        self._authored_world_graph = graph
        self._authored_map_origin = authored_map_origin
        if not scene.zones:
            self.get_logger().warning(f"world {world_name!r} has no authored acoustic zones, using map-based distance and occlusion propagation")
        if authored_map_origin is None:
            self.get_logger().error(f"cannot realize acoustic scene for {world_name!r}: no level map.yaml origin is available")
            self._world_microphones.clear()
        else:
            self._world_microphones = {microphone.listener_id: microphone for microphone in microphones}
        self._publish_microphone_registry()
        self._scene = None
        self._room_specs = ()
        self._world_graph = None
        self._portal_coupler = None
        self._coverage_signature = None
        self._acoustic_alignment_signature = None
        self._acoustic_offset = None
        self._realize_acoustic_geometry()

        if self._pending_world_name and self._pending_world_name != self._world_name:
            self._start_world_load(self._pending_world_name)

    def destroy_node(self) -> bool:
        self._world_loader.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _cb_map(self, msg: OccupancyGrid) -> None:
        self._map = msg
        self._realize_acoustic_geometry()
        self._validate_acoustic_zone_coverage()

    def _realize_acoustic_geometry(self) -> None:
        if self._map is None or self._authored_map_origin is None or self._authored_scene is None or self._authored_world_graph is None:
            return
        offset = runtime_acoustic_offset(
            self._map,
            self._authored_map_origin,
        )
        signature = (self._world_name, *offset)
        if signature == self._acoustic_alignment_signature:
            return
        self._acoustic_alignment_signature = signature
        self._acoustic_offset = offset
        (
            self._scene,
            self._room_specs,
            self._world_graph,
        ) = realize_acoustic_geometry(
            self._authored_scene,
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
            if self._pra_adapter is not None
            else None
        )
        self._continuous_propagation_signatures.clear()
        self._coverage_signature = None
        graph = self._world_graph
        self.get_logger().info(
            f"realized acoustic scene for world {self._world_name!r} in "
            f"runtime map frame {self._map.header.frame_id!r} with "
            f"offset=({offset[0]:.2f},{offset[1]:.2f}), "
            f"rooms={len(self._room_specs)}, "
            f"door_portals="
            f"{sum(p.portal_kind == 'door' for p in graph.portals)}, "
            f"opening_portals="
            f"{sum(p.portal_kind == 'opening' for p in graph.portals)}, "
            f"components={len(graph.connected_components())}, "
            f"unpaired_doors={len(graph.unpaired_doors)}"
        )
        for door in graph.unpaired_doors:
            self.get_logger().info(f"acoustic door {door.door_name!r} in {door.owner_zone!r} was not paired: {door.reason}")
        for portal in graph.portals:
            self.get_logger().info(f"acoustic portal {portal.portal_id}: {portal.zone_a!r} <-> {portal.zone_b!r}, kind={portal.portal_kind!r}, material={portal.material_id!r}, loss={portal.loss_db} dB")
        self._validate_acoustic_zone_coverage()
        while self._pending_events:
            event, listeners = self._pending_events.popleft()
            self._publish_event_to_listeners(event, listeners)

    def _portal_coupling_config(self) -> PortalCouplingConfig:
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

    def _validate_acoustic_zone_coverage(self) -> None:
        if not bool(self.get_parameter("validate_zone_coverage").value) or self._scene is None or not self._scene.zones or self._map is None:
            return
        info = self._map.info
        stride = max(int(self.get_parameter("zone_coverage_stride_cells").value), 1)
        coverage_tolerance = max(
            float(self.get_parameter("zone_coverage_tolerance_m").value),
            0.0,
        )
        signature = (
            self._world_name,
            info.width,
            info.height,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
            stride,
            coverage_tolerance,
            self._authored_map_origin,
        )
        if signature == self._coverage_signature:
            return
        self._coverage_signature = signature

        orientation = info.origin.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        traversable = 0
        uncovered: list[tuple[float, float]] = []
        for grid_y in range(0, info.height, stride):
            for grid_x in range(0, info.width, stride):
                index = grid_y * info.width + grid_x
                value = self._map.data[index]
                if value < 0 or value >= occupied_threshold:
                    continue
                local_x = (grid_x + 0.5) * info.resolution
                local_y = (grid_y + 0.5) * info.resolution
                world_x = info.origin.position.x + cos_yaw * local_x - sin_yaw * local_y
                world_y = info.origin.position.y + sin_yaw * local_x + cos_yaw * local_y
                traversable += 1
                sample = ShapelyPoint(world_x, world_y)
                if not any(zone.polygon.buffer(coverage_tolerance).covers(sample) for zone in self._scene.zones):
                    if len(uncovered) < 8:
                        uncovered.append((world_x, world_y))

        if not uncovered:
            self.get_logger().info(f"acoustic zone coverage validated on {traversable} sampled traversable cells (stride={stride})")
        else:
            examples = ", ".join(f"({x:.2f},{y:.2f})" for x, y in uncovered)
            map_width_m = info.width * info.resolution
            map_height_m = info.height * info.resolution
            zone_bounds = ", ".join(f"{zone.name}={tuple(round(v, 2) for v in zone.polygon.bounds)}" for zone in self._scene.zones)
            self.get_logger().warning(
                "traversable map locations exist outside all acoustic zones; "
                f"map(frame={self._map.header.frame_id!r}, "
                f"cells={info.width}x{info.height}, "
                f"size={map_width_m:.2f}x{map_height_m:.2f} m, "
                f"resolution={info.resolution:.3f}, "
                f"origin=({info.origin.position.x:.2f},"
                f"{info.origin.position.y:.2f}), yaw={yaw:.3f}); "
                f"zone bounds: {zone_bounds}; first samples: {examples}. "
                "These events will log an explicit backend fallback reason."
            )

    def _cb_robot_fleet(self, msg: RobotFleet) -> None:
        desired: set[tuple[str, str]] = set()
        active_names: set[str] = set()
        robot_frame_prefixes: dict[str, str] = {}

        for state in msg.robots:
            robot = state.descriptor
            name = str(robot.name)
            namespace = str(robot.ns).rstrip("/")
            active_names.add(name)
            robot_frame_prefixes[name] = str(robot.frame)
            listener_id = f"robot:{name}"
            base_frame_id = self._robot_base_frame(
                str(robot.model),
                str(robot.frame),
            )
            self._robot_base_frames[listener_id] = base_frame_id

            topics = self._robot_odom_topics(
                model_name=str(robot.model),
                namespace=namespace,
                robot_name=name,
            )
            for topic in topics:
                key = (name, topic)
                desired.add(key)
                if key not in self._odom_subs:
                    self._odom_subs[key] = self.create_subscription(
                        Odometry,
                        topic,
                        lambda odom, robot_name=name: self._cb_robot_odom(robot_name, odom),
                        self._odom_qos,
                    )

        for key in set(self._odom_subs) - desired:
            self.destroy_subscription(self._odom_subs.pop(key))

        for listener_id in list(self._robots):
            name = listener_id.removeprefix("robot:")
            if name not in active_names:
                self._robots.pop(listener_id, None)
                self._robot_base_frames.pop(listener_id, None)

        automatic_microphones = {
            f"{name}_mic": (
                Point(z=0.35),
                self._robot_base_frames[f"robot:{name}"],
            )
            for name in active_names
        }
        configured_microphones = {
            spec.listener_id: (
                Point(),
                spec.resolve_frame(robot_frame_prefixes[spec.robot]),
            )
            for spec in self._robot_microphone_specs
            if spec.robot in robot_frame_prefixes
        }
        self._robot_microphones = {
            **automatic_microphones,
            **configured_microphones,
        }
        configured_robots = {spec.robot for spec in self._robot_microphone_specs}
        missing_robots = sorted(configured_robots - active_names)
        if set(missing_robots) != self._missing_microphone_robots:
            self._missing_microphone_robots = set(missing_robots)
        else:
            missing_robots = []
        if missing_robots:
            self.get_logger().warning(f"configured microphones reference robots absent from state/robots: {missing_robots}")
        self._publish_microphone_registry()

    def _publish_microphone_registry(self) -> None:
        listener_ids = sorted(set(self._robot_microphones) | set(self._world_microphones) | set(self._spawned_microphones) | set(self._viewport_microphones))
        self._microphone_registry_pub.publish(String(data=json.dumps(listener_ids, separators=(",", ":"))))
        self._publish_microphone_markers()

    def _publish_microphone_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        markers = []
        color = ColorRGBA(r=0.12, g=0.95, b=0.45, a=0.85)
        for index, (listener_id, (position, frame_id)) in enumerate(sorted(self._microphone_marker_poses().items())):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "acoustic_microphones"
            marker.id = index * 2
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            marker.color = color
            marker.lifetime.sec = 1
            apex = Point(
                x=float(position.x) + 0.28,
                y=float(position.y),
                z=float(position.z),
            )
            base_a = Point(
                x=float(position.x) - 0.14,
                y=float(position.y) - 0.16,
                z=float(position.z) - 0.12,
            )
            base_b = Point(
                x=float(position.x) - 0.14,
                y=float(position.y) + 0.16,
                z=float(position.z) - 0.12,
            )
            base_c = Point(
                x=float(position.x) - 0.14,
                y=float(position.y),
                z=float(position.z) + 0.18,
            )
            marker.points = [
                apex,
                base_a,
                base_b,
                apex,
                base_b,
                base_c,
                apex,
                base_c,
                base_a,
                base_a,
                base_c,
                base_b,
            ]

            label = Marker()
            label.header = marker.header
            label.ns = "acoustic_microphone_labels"
            label.id = index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = Point(
                x=float(position.x),
                y=float(position.y),
                z=float(position.z) + 0.35,
            )
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color = ColorRGBA(r=0.05, g=0.35, b=0.12, a=1.0)
            label.lifetime = marker.lifetime
            label.text = listener_id
            markers.extend((marker, label))
        self._microphone_marker_pub.publish(MarkerArray(markers=markers))

    def _microphone_marker_poses(
        self,
    ) -> dict[str, tuple[Point, str]]:
        poses = {listener_id: (position, frame_id) for listener_id, (position, frame_id) in (self._robot_microphones.items())}
        poses.update(self._spawned_microphones)
        map_frame = str(self._map.header.frame_id).strip() if self._map is not None else "map"
        for listener_id, position in self._all_microphone_positions().items():
            if listener_id in poses:
                continue
            poses[listener_id] = (position, map_frame)
        return poses

    def _spawn_microphone(
        self,
        request: SpawnMicrophone.Request,
        response: SpawnMicrophone.Response,
    ) -> SpawnMicrophone.Response:
        placement = str(request.placement).strip().lower()
        if not placement or ":" in placement:
            response.error_msg = "placement must be non-empty and contain no ':'"
            return response
        point = request.position.point
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            response.error_msg = "microphone position must be finite"
            return response
        source_frame = str(request.position.header.frame_id).strip().lstrip("/")
        if not source_frame:
            response.error_msg = "microphone position requires a frame ID"
            return response
        attached_frame = str(request.attached_frame).strip().lstrip("/")
        stored_position = Point(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
        )
        stored_frame = source_frame
        if attached_frame:
            attached_position = self._point_between_frames(
                point,
                source_frame,
                attached_frame,
                "spawned microphone attachment",
            )
            if attached_position is None:
                response.error_msg = f"cannot transform clicked position into TF frame {attached_frame!r}"
                return response
            stored_position = attached_position
            stored_frame = attached_frame
        transformed = self._point_in_acoustic_frame(
            point,
            source_frame,
            "spawned microphone",
        )
        zone = (
            next(
                (candidate for candidate in self._scene.zones if candidate.polygon.buffer(MICROPHONE_PLACEMENT_TOLERANCE_M).covers(ShapelyPoint(transformed.x, transformed.y))),
                None,
            )
            if self._scene is not None and transformed is not None
            else None
        )
        room_spec = next(
            (candidate for candidate in self._room_specs if zone is not None and candidate.zone_name == zone.name),
            None,
        )
        validation_position = transformed or stored_position
        if validation_position.z < -MICROPHONE_PLACEMENT_TOLERANCE_M:
            response.error_msg = "microphone height cannot be below the floor"
            return response
        if room_spec is not None and validation_position.z > room_spec.ceiling_height_m + MICROPHONE_PLACEMENT_TOLERANCE_M:
            response.error_msg = f"microphone height exceeds zone {zone.name!r} ceiling at {room_spec.ceiling_height_m:.2f} m"
            return response
        if room_spec is None and validation_position.z > float(self.get_parameter("pyroom_ceiling_height_m").value) + MICROPHONE_PLACEMENT_TOLERANCE_M:
            response.error_msg = "microphone height exceeds the default ceiling"
            return response

        existing = set(self._robot_microphones) | set(self._world_microphones) | set(self._spawned_microphones) | set(self._viewport_microphones)
        while True:
            self._spawned_microphone_index += 1
            listener_id = f"microphone{self._spawned_microphone_index}"
            if listener_id not in existing:
                break

        self._spawned_microphones[listener_id] = (
            Point(
                x=float(stored_position.x),
                y=float(stored_position.y),
                z=float(stored_position.z),
            ),
            stored_frame,
        )
        self._publish_microphone_registry()
        response.listener_id = listener_id
        response.zone = zone.name if zone is not None else ""
        response.attached_frame = attached_frame
        response.success = True
        log_position = transformed or stored_position
        self.get_logger().info(f"spawned microphone {listener_id!r} at ({log_position.x:.2f}, {log_position.y:.2f}, {log_position.z:.2f})")
        return response

    def _remove_microphone(
        self,
        request: RemoveMicrophone.Request,
        response: RemoveMicrophone.Response,
    ) -> RemoveMicrophone.Response:
        listener_id = str(request.listener_id).strip()
        if listener_id not in self._spawned_microphones:
            if listener_id in self._world_microphones:
                response.error_msg = "world-authored microphones cannot be removed"
            elif listener_id in self._robot_microphones:
                response.error_msg = "robot-attached microphones cannot be removed"
            elif listener_id in self._viewport_microphones:
                response.error_msg = "viewport microphones cannot be removed"
            else:
                response.error_msg = f"unknown runtime microphone {listener_id!r}"
            return response
        self._spawned_microphones.pop(listener_id)
        for key in tuple(self._last_continuous_outputs):
            if key[1] == listener_id:
                stopped = self._last_continuous_outputs.pop(key)
                self._continuous_propagation_signatures.pop(key, None)
                stopped.header.stamp = self.get_clock().now().to_msg()
                stopped.active = False
                stopped.audible = False
                self._continuous_heard_pub.publish(stopped)
        self._publish_microphone_registry()
        response.success = True
        return response

    def _cb_robot_odom(
        self,
        robot_name: str,
        _msg: Odometry,
    ) -> None:
        listener_id = f"robot:{robot_name}"
        frame_id = self._robot_base_frames.get(listener_id)
        if frame_id is None:
            return
        self._robots[listener_id] = (
            Point(),
            frame_id,
        )

    def _cb_sound_event(self, event: SoundEvent) -> None:
        if not bool(self.get_parameter("enable_propagation").value):
            return
        if not event.sound_type.strip():
            return

        if self._scene is not None and not self._transform_event_source(event):
            return

        listeners = self._listeners_for_event(event)

        if self._requested_backend == "pyroomacoustics" and self._scene is None and bool(self.get_parameter("buffer_events_until_scene_loaded").value):
            maximum = max(
                int(self.get_parameter("scene_event_buffer_size").value),
                1,
            )
            if len(self._pending_events) >= maximum:
                dropped, _ = self._pending_events.popleft()
                self.get_logger().warning(f"acoustic scene event buffer full; dropping {dropped.event_id!r}")
            snapshots = {
                listener_id: Point(
                    x=float(position.x),
                    y=float(position.y),
                    z=float(position.z),
                )
                for listener_id, position in listeners.items()
            }
            self._pending_events.append((event, snapshots))
            return

        self._publish_event_to_listeners(event, listeners)

    def _cb_continuous_source(
        self,
        state: ContinuousAudioSourceState,
    ) -> None:
        """Propagate persistent source state through the same acoustic model."""
        if not bool(self.get_parameter("enable_propagation").value):
            return
        if not state.sound_type.strip():
            return
        # Continuous publishers refresh their state. Waiting for the next
        # update avoids an unbounded second scene-load buffer.
        if self._requested_backend == "pyroomacoustics" and self._scene is None:
            return

        event = SoundEvent()
        event.header = copy.deepcopy(state.header)
        event.event_id = state.source_id
        event.source_agent_id = state.source_agent_id
        event.source_agent_name = state.source_agent_name
        event.sound_type = state.sound_type
        event.label = state.label or state.sound_type
        event.asset_id = state.asset_id or state.source_backend
        event.source_position = state.source_position
        event.source_yaw = state.source_yaw
        event.source_volume_db = state.source_volume_db
        event.semantic_tags = ["continuous", *state.semantic_tags]
        event.loop = state.loop

        if not self._transform_event_source(event):
            return

        listeners = self._listeners_for_event(event)
        for listener_id, listener_pos in listeners.items():
            key = (state.source_id, listener_id)
            signature = self._continuous_propagation_signature(
                state,
                event.source_position,
                listener_pos,
            )
            output = None
            if signature == self._continuous_propagation_signatures.get(key):
                previous = self._last_continuous_outputs.get(key)
                if previous is not None:
                    output = copy.deepcopy(previous)

            if output is None:
                heard = self._calculate_heard_event(
                    event,
                    listener_id,
                    listener_pos,
                )
                if not heard.audible and not bool(self.get_parameter("publish_inaudible").value):
                    continue
                output = ContinuousHeardSoundState()
                output.received_volume_db = heard.received_volume_db
                output.direct_delay_sec = heard.direct_delay_sec
                output.audible = heard.audible
                output.occluded = heard.occluded
                output.source_zone = heard.source_zone
                output.listener_zone = heard.listener_zone
                output.propagation_backend = heard.propagation_backend
                output.used_backend_fallback = heard.used_backend_fallback
                output.backend_fallback_reason = heard.backend_fallback_reason
                output.portal_ids = heard.portal_ids
                output.traversed_zones = heard.traversed_zones
                output.portal_positions = heard.portal_positions
                output.portal_hop_count = heard.portal_hop_count
                output.portal_route_loss_db = heard.portal_route_loss_db
                self._continuous_propagation_signatures[key] = signature

            output.header = event.header
            output.source_id = state.source_id
            output.listener_id = listener_id
            output.source_agent_id = state.source_agent_id
            output.source_agent_name = state.source_agent_name
            output.source_model = state.source_model
            output.sound_type = state.sound_type
            output.source_backend = state.source_backend
            output.group_id = state.group_id
            output.asset_id = state.asset_id
            output.label = state.label
            output.loop = state.loop
            output.program_start_time = state.program_start_time
            output.source_position = event.source_position
            output.listener_position = listener_pos
            output.linear_velocity_mps = state.linear_velocity_mps
            output.angular_velocity_radps = state.angular_velocity_radps
            output.left_velocity_mps = state.left_velocity_mps
            output.right_velocity_mps = state.right_velocity_mps
            output.source_volume_db = state.source_volume_db
            output.active = state.active
            output.deterministic_seed = state.deterministic_seed
            self._last_continuous_outputs[key] = copy.deepcopy(output)
            self._continuous_heard_pub.publish(output)

    @staticmethod
    def _continuous_propagation_signature(
        state: ContinuousAudioSourceState,
        source_position: Point,
        listener_position: Point,
    ) -> tuple[Hashable, ...]:
        return (
            int(state.source_agent_id),
            state.source_agent_name,
            state.source_model,
            state.sound_type,
            state.label,
            tuple(state.semantic_tags),
            float(state.source_volume_db),
            float(source_position.x),
            float(source_position.y),
            float(source_position.z),
            float(state.source_yaw),
            float(listener_position.x),
            float(listener_position.y),
            float(listener_position.z),
        )

    def _listeners_for_event(self, event: SoundEvent) -> dict[str, Point]:
        """Snapshot listeners according to the configured hearing policy."""
        listeners: dict[str, Point] = {}

        if bool(self.get_parameter("ped_hearing").value):
            for agent_id, ped in self._peds.items():
                if agent_id == event.source_agent_id:
                    continue
                listener_id = f"agent:{agent_id}"
                position = self._point_in_acoustic_frame(
                    ped.pose.position,
                    self._peds_frame_id,
                    listener_id,
                )
                if position is not None:
                    listeners[listener_id] = position

        for listener_id, (position, frame_id) in self._robots.items():
            transformed = self._point_in_acoustic_frame(
                position,
                frame_id,
                listener_id,
            )
            if transformed is not None:
                listeners[listener_id] = transformed

        listeners.update(self._microphone_positions())

        if not bool(self.get_parameter("robots_hear_self").value):
            listeners.pop(f"robot:{event.source_agent_name}", None)

        return listeners

    def _microphone_positions(self) -> dict[str, Point]:
        listeners = self._all_microphone_positions()
        selected = str(self.get_parameter("active_microphone_id").value).strip()
        if selected not in listeners:
            return {}
        return {selected: listeners[selected]}

    def _all_microphone_positions(self) -> dict[str, Point]:
        listeners: dict[str, Point] = {}

        for listener_id, (position, frame_id) in self._robot_microphones.items():
            transformed = self._point_in_acoustic_frame(
                position,
                frame_id,
                listener_id,
            )
            if transformed is not None:
                listeners[listener_id] = transformed

        for listener_id, microphone in self._world_microphones.items():
            position = Point(
                x=microphone.position[0],
                y=microphone.position[1],
                z=microphone.position[2],
            )
            frame_id = microphone.frame
            if frame_id == "map":
                if self._map is None or self._acoustic_offset is None:
                    continue
                position.x += self._acoustic_offset[0]
                position.y += self._acoustic_offset[1]
                frame_id = self._map.header.frame_id
            transformed = self._point_in_acoustic_frame(
                position,
                frame_id,
                listener_id,
            )
            if transformed is None or not self._valid_world_microphone(
                microphone,
                transformed,
            ):
                continue
            listeners[listener_id] = transformed

        for listener_id, (position, frame_id) in self._spawned_microphones.items():
            transformed = self._point_in_acoustic_frame(
                position,
                frame_id,
                listener_id,
            )
            if transformed is not None:
                listeners[listener_id] = transformed

        for listener_id, (position, frame_id) in self._viewport_microphones.items():
            transformed = self._point_in_acoustic_frame(
                position,
                frame_id,
                listener_id,
            )
            if transformed is not None:
                listeners[listener_id] = transformed

        return listeners

    def _valid_world_microphone(
        self,
        microphone: WorldMicrophoneSpec,
        position: Point,
    ) -> bool:
        if self._scene is None:
            return False
        zone = next(
            (candidate for candidate in self._scene.zones if candidate.name == microphone.zone),
            None,
        )
        in_zone = zone is not None and zone.polygon.buffer(MICROPHONE_PLACEMENT_TOLERANCE_M).covers(ShapelyPoint(position.x, position.y))
        if not in_zone:
            self._warn_transform_unavailable(
                microphone.listener_id,
                microphone.frame,
                self._map.header.frame_id if self._map is not None else "map",
                f"resolved outside declared zone {microphone.zone!r}",
            )
            return False
        if microphone.ceiling_height_m is not None and not math.isclose(
            position.z,
            microphone.ceiling_height_m,
            abs_tol=MICROPHONE_PLACEMENT_TOLERANCE_M,
        ):
            self._warn_transform_unavailable(
                microphone.listener_id,
                microphone.frame,
                self._map.header.frame_id if self._map is not None else "map",
                "resolved height does not match the declared zone ceiling",
            )
            return False
        return True

    def _transform_event_source(self, event: SoundEvent) -> bool:
        source_frame = str(event.header.frame_id).strip()
        source = self._point_in_acoustic_frame(
            event.source_position,
            source_frame,
            event.event_id or event.source_agent_name or "sound source",
        )
        if source is None:
            return False
        event.source_position = source
        assert self._map is not None
        event.header.frame_id = self._map.header.frame_id
        return True

    def _robot_base_frame(self, model_name: str, frame_prefix: str) -> str:
        prefix = frame_prefix.strip("/")
        configured_listener_frame = str(self.get_parameter("robot_listener_frame").value).strip()
        try:
            base_frame = RobotIdentifier(model_name).resolve_sync().model_params.base_frame.strip("/")
        except Exception as exc:
            base_frame = "base_link"
            self.get_logger().warning(f"could not resolve base frame for robot model {model_name!r}: {exc}, using {base_frame!r}")

        if configured_listener_frame:
            configured_listener_frame = configured_listener_frame.strip("/")
            if "{" in configured_listener_frame:
                try:
                    configured_listener_frame = configured_listener_frame.format(
                        prefix=prefix,
                        base_frame=base_frame,
                    ).strip("/")
                except KeyError as exc:
                    self.get_logger().warning(f"invalid robot_listener_frame template {configured_listener_frame!r}: missing {exc}")
            if not configured_listener_frame:
                return "/".join(part for part in (prefix, base_frame) if part)
            if "/" in configured_listener_frame:
                return configured_listener_frame
            return "/".join(part for part in (prefix, configured_listener_frame) if part)

        return "/".join(part for part in (prefix, base_frame) if part)

    def _robot_odom_topics(
        self,
        *,
        model_name: str,
        namespace: str,
        robot_name: str,
    ) -> tuple[str, ...]:
        topics = [
            str(self.get_parameter("odom_topic_template").value).format(
                namespace=namespace,
                name=robot_name,
            ),
            f"{namespace}/odom",
        ]
        try:
            control = RobotIdentifier(model_name).resolve_sync().model_params.control
            model_odom = control.odom_topic.strip("/") if control is not None else ""
            if model_odom:
                topics.append(f"{namespace}/{model_odom}")
        except Exception as exc:
            self.get_logger().warning(f"could not resolve odometry topic for {model_name!r}: {exc}")
        return tuple(dict.fromkeys(topic for topic in topics if topic))

    def _point_in_acoustic_frame(
        self,
        point: Point,
        source_frame: str,
        entity_id: str,
    ) -> Point | None:
        if self._map is None:
            return None
        target_frame = str(self._map.header.frame_id).strip()
        return self._point_between_frames(
            point,
            source_frame,
            target_frame,
            entity_id,
        )

    def _point_between_frames(
        self,
        point: Point,
        source_frame: str,
        target_frame: str,
        entity_id: str,
    ) -> Point | None:
        source_frame = source_frame.strip().lstrip("/")
        target_frame = target_frame.strip().lstrip("/")
        if not source_frame or not target_frame:
            self._warn_transform_unavailable(
                entity_id,
                source_frame,
                target_frame,
                "missing frame ID",
            )
            return None
        if source_frame == target_frame:
            return Point(
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
            )
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self._warn_transform_unavailable(
                entity_id,
                source_frame,
                target_frame,
                str(exc),
            )
            return None
        return self._apply_transform(point, transform.transform)

    def _warn_transform_unavailable(
        self,
        entity_id: str,
        source_frame: str,
        target_frame: str,
        reason: str,
    ) -> None:
        key = (entity_id, source_frame, target_frame)
        now = time.monotonic()
        if now - self._transform_warning_times.get(key, -float("inf")) < 5.0:
            return
        self._transform_warning_times[key] = now
        self.get_logger().warning(f"cannot transform acoustic position for {entity_id!r} from {source_frame!r} to runtime map frame {target_frame!r}: {reason}")

    @staticmethod
    def _apply_transform(point: Point, transform: Transform) -> Point:
        rotation = transform.rotation
        translation = transform.translation
        qx = float(rotation.x)
        qy = float(rotation.y)
        qz = float(rotation.z)
        qw = float(rotation.w)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm > 0.0:
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm

        x = float(point.x)
        y = float(point.y)
        z = float(point.z)
        uv_x = qy * z - qz * y
        uv_y = qz * x - qx * z
        uv_z = qx * y - qy * x
        uuv_x = qy * uv_z - qz * uv_y
        uuv_y = qz * uv_x - qx * uv_z
        uuv_z = qx * uv_y - qy * uv_x
        return Point(
            x=x + 2.0 * (qw * uv_x + uuv_x) + float(translation.x),
            y=y + 2.0 * (qw * uv_y + uuv_y) + float(translation.y),
            z=z + 2.0 * (qw * uv_z + uuv_z) + float(translation.z),
        )

    def _publish_event_to_listeners(
        self,
        event: SoundEvent,
        listeners: dict[str, Point],
    ) -> None:
        if not self._transform_event_source(event):
            return
        for listener_id, listener_pos in listeners.items():
            heard = self._calculate_heard_event(event, listener_id, listener_pos)
            if heard.audible or bool(self.get_parameter("publish_inaudible").value):
                self._heard_pub.publish(heard)

    def _effective_sound_distance(
        self,
        geometric_distance: float,
        event: SoundEvent,
        listener_id: str,
    ) -> float:
        if listener_id == f"robot:{event.source_agent_name}":
            return max(float(self.get_parameter("self_hearing_distance_m").value), 1e-3)

        return max(geometric_distance, float(self.get_parameter("minimum_propagation_distance_m").value), 1e-3)

    def _calculate_legacy_event(self, event: SoundEvent, listener_id: str, listener_pos: Point) -> HeardSoundEvent:
        dx = event.source_position.x - listener_pos.x
        dy = event.source_position.y - listener_pos.y
        # distance = max(math.hypot(dx, dy), 1.0)
        # distance_loss = 20.0 * math.log10(distance)
        geometric_distance = math.hypot(dx, dy)
        effective_distance = self._effective_sound_distance(
            geometric_distance,
            event,
            listener_id,
        )
        distance_loss = 20.0 * math.log10(effective_distance)

        occluded = self._is_occluded(event.source_position, listener_pos)
        occlusion_penalty = float(self.get_parameter("occlusion_penalty_db").value) if occluded else 0.0
        received = event.source_volume_db - distance_loss - occlusion_penalty
        threshold = float(self.get_parameter("default_hearing_threshold_db").value)

        msg = HeardSoundEvent()
        msg.header = event.header
        msg.event_id = event.event_id
        msg.listener_id = listener_id
        msg.source_agent_id = event.source_agent_id
        msg.source_agent_name = event.source_agent_name
        msg.sound_type = event.sound_type
        msg.label = event.label
        msg.asset_id = event.asset_id
        msg.source_position = event.source_position
        msg.listener_position = listener_pos
        msg.distance = float(geometric_distance)
        msg.source_volume_db = event.source_volume_db
        msg.received_volume_db = float(received)
        msg.hearing_threshold_db = float(threshold)
        msg.audible = received >= threshold
        msg.occluded = occluded
        msg.bearing_rad = float(math.atan2(dy, dx))
        msg.direct_delay_sec = float(effective_distance / 343.0)
        msg.propagation_level = 0
        msg.reverb_rt60_sec = 0.0
        msg.reverb_gain_db = 0.0
        msg.source_zone = ""
        msg.listener_zone = ""
        return msg

    def _calculate_heard_event(self, event: SoundEvent, listener_id: str, listener_pos: Point) -> HeardSoundEvent:
        if self._scene is None or not self._scene.zones:
            msg = self._calculate_legacy_event(event, listener_id, listener_pos)
            return self._finalize_backend(
                msg,
                backend="legacy_distance_occlusion",
                used_fallback=self._requested_backend == "pyroomacoustics",
                fallback_reason=(("acoustic_scene_not_loaded" if self._scene is None else "acoustic_scene_has_no_zones") if self._requested_backend == "pyroomacoustics" else ""),
            )

        if listener_id == f"robot:{event.source_agent_name}" and self._pra_adapter is None:
            msg = self._calculate_legacy_event(event, listener_id, listener_pos)
            return self._finalize_backend(
                msg,
                backend="legacy_self_hearing",
                used_fallback=True,
                fallback_reason="level3_self_hearing_uses_legacy_distance",
            )

        fallback_reason = ""
        deferred_same_room = False
        deferred_route = None
        compute_rir_here = bool(self.get_parameter("compute_rir_in_propagation").value)
        if self._pra_adapter is not None:
            source_zone = self._scene.zone_at(event.source_position)
            listener_zone = self._scene.zone_at(listener_pos)
            if source_zone is None:
                fallback_reason = "source_outside_acoustic_zones"
            elif listener_zone is None:
                fallback_reason = "listener_outside_acoustic_zones"
            elif source_zone.name == listener_zone.name:
                room_spec = next(
                    (spec for spec in self._room_specs if spec.zone_name == source_zone.name),
                    None,
                )
                if room_spec is None:
                    fallback_reason = "same_zone_has_no_room_spec"
                elif not compute_rir_here:
                    deferred_same_room = True
                else:
                    try:
                        return self._finalize_backend(
                            self._calculate_pyroom_event(event, listener_id, listener_pos, room_spec),
                            backend="pyroomacoustics_same_room",
                            used_fallback=False,
                            fallback_reason="",
                        )
                    except Exception as exc:
                        fallback_reason = "same_room_rir_error:" + type(exc).__name__
                        self.get_logger().warning(f"pyroomacoustics same-room RIR failed for {event.source_agent_name}->{listener_id} in {room_spec.zone_name!r}: {exc}")
            else:
                max_hops = int(self.get_parameter("max_portal_hops").value) if bool(self.get_parameter("enable_multi_portal_rir").value) else 1
                route = (
                    self._world_graph.find_portal_route(
                        source_zone.name,
                        listener_zone.name,
                        source_xy=(event.source_position.x, event.source_position.y),
                        listener_xy=(listener_pos.x, listener_pos.y),
                        max_portals=max_hops,
                        distance_loss_db_per_m=float(self.get_parameter("route_distance_loss_db_per_m").value),
                    )
                    if self._world_graph is not None
                    else None
                )
                if route is None:
                    unrestricted = (
                        self._world_graph.find_portal_route(
                            source_zone.name,
                            listener_zone.name,
                            source_xy=(
                                event.source_position.x,
                                event.source_position.y,
                            ),
                            listener_xy=(listener_pos.x, listener_pos.y),
                            max_portals=max(len(self._room_specs) - 1, 1),
                            distance_loss_db_per_m=float(self.get_parameter("route_distance_loss_db_per_m").value),
                        )
                        if self._world_graph is not None
                        else None
                    )
                    fallback_reason = "portal_route_exceeds_max_hops" if unrestricted is not None else "no_portal_route_between_zones"
                elif not compute_rir_here:
                    deferred_route = route
                elif self._portal_coupler is None:
                    fallback_reason = "portal_coupler_not_initialized"
                else:
                    try:
                        msg = self._calculate_portal_event(
                            event,
                            listener_id,
                            listener_pos,
                            source_zone.name,
                            listener_zone.name,
                        )
                        return self._finalize_backend(
                            msg,
                            backend=("pyroomacoustics_one_door" if msg.portal_hop_count == 1 else "pyroomacoustics_multi_portal"),
                            used_fallback=False,
                            fallback_reason="",
                            portal_id=msg.portal_id,
                        )
                    except Exception as exc:
                        fallback_reason = "portal_route_rir_error:" + type(exc).__name__
                        self.get_logger().warning(f"pyroomacoustics portal-route coupling failed for {source_zone.name!r}->{listener_zone.name!r} through {[p.portal_id for p in route.portals]!r}: {exc}")

        result = self._propagation.calculate(
            self._scene,
            event.source_position,
            listener_pos,
            event.source_volume_db,
        )
        dx = event.source_position.x - listener_pos.x
        dy = event.source_position.y - listener_pos.y
        distance = max(math.hypot(dx, dy), 1.0)
        threshold = float(self.get_parameter("default_hearing_threshold_db").value)

        msg = HeardSoundEvent()
        msg.header = event.header
        msg.event_id = event.event_id
        msg.listener_id = listener_id
        msg.source_agent_id = event.source_agent_id
        msg.source_agent_name = event.source_agent_name
        msg.sound_type = event.sound_type
        msg.label = event.label
        msg.asset_id = event.asset_id
        msg.source_position = event.source_position
        msg.listener_position = listener_pos
        msg.distance = float(distance)
        msg.bearing_rad = float(math.atan2(dy, dx))
        msg.source_volume_db = event.source_volume_db
        msg.received_volume_db = result.received_volume_db
        msg.hearing_threshold_db = threshold
        msg.audible = result.received_volume_db >= threshold
        msg.occluded = result.occluded
        msg.propagation_level = 3
        msg.direct_delay_sec = result.direct_delay_sec
        msg.reverb_rt60_sec = result.rt60_sec
        msg.reverb_gain_db = result.reverb_gain_db
        msg.source_zone = result.source_zone
        msg.listener_zone = result.listener_zone

        for path in result.paths:
            path_msg = AcousticPath()
            path_msg.delay.sec = int(path.delay_sec)
            path_msg.delay.nanosec = int((path.delay_sec % 1.0) * 1_000_000_000)
            path_msg.gain_db = path.gain_db
            path_msg.bearing_rad = path.bearing_rad
            path_msg.interaction_type = path.interaction_type
            path_msg.material_id = path.material_id

            if path.reflection_point is not None:
                path_msg.reflection_point.x = path.reflection_point[0]
                path_msg.reflection_point.y = path.reflection_point[1]

            msg.early_paths.append(path_msg)

        if deferred_route is not None:
            self._apply_deferred_route_metadata(msg, deferred_route)

        return self._finalize_backend(
            msg,
            backend=("level3_rir_deferred_same_room" if deferred_same_room else "level3_rir_deferred_portal" if deferred_route is not None else "level3"),
            used_fallback=(self._requested_backend == "pyroomacoustics" and not deferred_same_room and deferred_route is None),
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _apply_deferred_route_metadata(
        msg: HeardSoundEvent,
        route: AcousticPortalRoute,
    ) -> None:
        msg.portal_ids = [portal.portal_id for portal in route.portals]
        msg.traversed_zones = list(route.zones)
        msg.portal_hop_count = len(route.portals)
        msg.portal_route_loss_db = float(sum(portal.loss_db or 0.0 for portal in route.portals))
        if not route.portals:
            return
        first = route.portals[0]
        msg.portal_id = first.portal_id
        msg.portal_position.x = first.center_xy[0]
        msg.portal_position.y = first.center_xy[1]
        msg.portal_position.z = 0.5 * first.height_m
        for portal in route.portals:
            point = Point()
            point.x = portal.center_xy[0]
            point.y = portal.center_xy[1]
            point.z = 0.5 * portal.height_m
            msg.portal_positions.append(point)
            path = AcousticPath()
            path.delay.sec = int(msg.direct_delay_sec)
            path.delay.nanosec = int((msg.direct_delay_sec % 1.0) * 1_000_000_000)
            path.gain_db = -float(portal.loss_db or 0.0)
            path.bearing_rad = float(
                math.atan2(
                    portal.center_xy[1] - msg.listener_position.y,
                    portal.center_xy[0] - msg.listener_position.x,
                )
            )
            path.reflection_point = point
            path.interaction_type = f"portal_{portal.portal_kind}"
            path.material_id = portal.material_id
            msg.early_paths.append(path)

    def _finalize_backend(
        self,
        msg: HeardSoundEvent,
        *,
        backend: str,
        used_fallback: bool,
        fallback_reason: str,
        portal_id: str = "",
    ) -> HeardSoundEvent:
        msg.propagation_backend = backend
        msg.used_backend_fallback = bool(used_fallback)
        msg.backend_fallback_reason = fallback_reason
        if portal_id:
            msg.portal_id = portal_id
        route = (
            backend,
            fallback_reason,
            str(msg.source_zone),
            str(msg.listener_zone),
        )
        if route not in self._reported_routes:
            self._reported_routes.add(route)
            detail = f", fallback={fallback_reason!r}" if used_fallback else ""
            portal_detail = f", portal={msg.portal_id!r}" if msg.portal_id else ""
            message = (
                f"actual propagation backend={backend!r} for {msg.source_zone!r}->{msg.listener_zone!r}{portal_detail}{detail}; source=({msg.source_position.x:.2f},{msg.source_position.y:.2f}) name={msg.source_agent_name!r}, listener={msg.listener_id!r}@({msg.listener_position.x:.2f},{msg.listener_position.y:.2f})"
            )
            if used_fallback:
                self.get_logger().warning(message)
            else:
                self.get_logger().info(message)
        return msg

    @staticmethod
    def _source_height(event: SoundEvent) -> float:
        if "static" in event.semantic_tags:
            return float(event.source_position.z)
        sound_type = f"{event.sound_type} {event.label}".lower()
        if "foot" in sound_type or "step" in sound_type:
            return 0.05
        if "motor" in sound_type or "robot" in sound_type:
            return 0.25
        return 1.60

    def _listener_height(
        self,
        listener_id: str,
        listener_position: Point,
    ) -> float:
        # The robot odometry pose is a ground-contact position, not the
        # microphone position. Human listeners use an approximate ear height.
        if listener_id in (set(self._robot_microphones) | set(self._world_microphones) | set(self._spawned_microphones) | set(self._viewport_microphones)):
            return float(listener_position.z)
        return 0.35 if listener_id.startswith("robot:") else 1.60

    def _calculate_pyroom_event(
        self,
        event: SoundEvent,
        listener_id: str,
        listener_position: Point,
        room_spec: AcousticRoomSpec,
    ) -> HeardSoundEvent:
        assert self._pra_adapter is not None

        source_position = (
            event.source_position.x,
            event.source_position.y,
            self._source_height(event),
        )
        listener_position_3d = (
            listener_position.x,
            listener_position.y,
            self._listener_height(listener_id, listener_position),
        )
        rir = self._pra_adapter.compute_rir(
            room_spec,
            source_position_m=source_position,
            listener_position_m=listener_position_3d,
        )
        return self._calculate_rir_event(
            event,
            listener_id,
            listener_position,
            rir,
            source_zone=room_spec.zone_name,
            listener_zone=room_spec.zone_name,
            portal_result=None,
        )

    def _calculate_portal_event(
        self,
        event: SoundEvent,
        listener_id: str,
        listener_position: Point,
        source_zone: str,
        listener_zone: str,
    ) -> HeardSoundEvent:
        assert self._portal_coupler is not None
        result = self._portal_coupler.compute(
            source_zone=source_zone,
            listener_zone=listener_zone,
            source_position_m=(
                event.source_position.x,
                event.source_position.y,
                self._source_height(event),
            ),
            listener_position_m=(
                listener_position.x,
                listener_position.y,
                self._listener_height(listener_id, listener_position),
            ),
        )
        self.get_logger().debug(f"portal RIR cache: entries={self._portal_coupler.cache_entries}, hits={result.cache_hits}, misses={result.cache_misses}; route_entries={self._portal_coupler.route_cache_entries}, route_hits={self._portal_coupler.route_cache_hits}, route_misses={self._portal_coupler.route_cache_misses}")
        return self._calculate_rir_event(
            event,
            listener_id,
            listener_position,
            result.rir,
            source_zone=source_zone,
            listener_zone=listener_zone,
            portal_result=result,
        )

    def _calculate_rir_event(
        self,
        event: SoundEvent,
        listener_id: str,
        listener_position: Point,
        rir: RoomImpulseResponse,
        *,
        source_zone: str,
        listener_zone: str,
        portal_result: PortalCouplingResult | None,
    ) -> HeardSoundEvent:

        samples = np.asarray(rir.samples, dtype=np.float64)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("RIR contains no finite samples")

        peak_index = int(np.argmax(np.abs(samples)))
        peak_amplitude = max(float(np.max(np.abs(samples))), 1e-12)
        gain_db = 20.0 * math.log10(peak_amplitude)
        delay_sec = (rir.global_delay_samples + peak_index) / float(rir.sample_rate_hz)

        # Report a conservative late-energy estimate as reverb gain. The
        # message schema has no RIR field; audio convolution is a later stage.
        direct_end = min(peak_index + 1, samples.size)
        late_energy = float(np.sqrt(np.mean(samples[direct_end:] ** 2))) if direct_end < samples.size else 0.0
        reverb_gain_db = 20.0 * math.log10(max(late_energy / peak_amplitude, 1e-12)) if late_energy > 0.0 else -120.0

        dx = event.source_position.x - listener_position.x
        dy = event.source_position.y - listener_position.y
        distance = max(math.hypot(dx, dy), 1.0)
        # For a coupled RIR the wall crossing is intentional: the two room
        # responses and portal loss already model the source-door-listener
        # path. Applying the straight-line map penalty would attenuate it a
        # second time merely because source and listener occupy two rooms.
        occluded = False if portal_result is not None else self._is_occluded(event.source_position, listener_position)
        occlusion_penalty = float(self.get_parameter("occlusion_penalty_db").value) if occluded else 0.0
        threshold = float(self.get_parameter("default_hearing_threshold_db").value)

        msg = HeardSoundEvent()
        msg.header = event.header
        msg.event_id = event.event_id
        msg.listener_id = listener_id
        msg.source_agent_id = event.source_agent_id
        msg.source_agent_name = event.source_agent_name
        msg.sound_type = event.sound_type
        msg.label = event.label
        msg.asset_id = event.asset_id
        msg.source_position = event.source_position
        msg.listener_position = listener_position
        msg.distance = float(distance)
        msg.bearing_rad = float(math.atan2(dy, dx))
        msg.source_volume_db = event.source_volume_db
        msg.received_volume_db = float(event.source_volume_db + gain_db - occlusion_penalty)
        msg.hearing_threshold_db = threshold
        msg.audible = msg.received_volume_db >= threshold
        msg.occluded = occluded
        msg.propagation_level = 3
        msg.direct_delay_sec = float(delay_sec)
        msg.reverb_rt60_sec = 0.0
        msg.reverb_gain_db = float(reverb_gain_db)
        msg.source_zone = source_zone
        msg.listener_zone = listener_zone
        if portal_result is not None:
            route = portal_result.route
            portals = route.portals if route is not None else (portal_result.portal,)
            zones = route.zones if route is not None else (source_zone, listener_zone)
            portal = portals[0]
            msg.portal_id = portal.portal_id
            msg.portal_position.x = portal.center_xy[0]
            msg.portal_position.y = portal.center_xy[1]
            msg.portal_position.z = 0.5 * portal.height_m
            msg.portal_ids = [item.portal_id for item in portals]
            msg.traversed_zones = list(zones)
            msg.portal_hop_count = len(portals)
            msg.portal_route_loss_db = float(portal_result.applied_portal_loss_db)
            for item in portals:
                portal_point = Point()
                portal_point.x = item.center_xy[0]
                portal_point.y = item.center_xy[1]
                portal_point.z = 0.5 * item.height_m
                msg.portal_positions.append(portal_point)

                path = AcousticPath()
                path.delay.sec = int(delay_sec)
                path.delay.nanosec = int((delay_sec % 1.0) * 1_000_000_000)
                path.gain_db = float(gain_db)
                path.bearing_rad = float(
                    math.atan2(
                        item.center_xy[1] - listener_position.y,
                        item.center_xy[0] - listener_position.x,
                    )
                )
                path.reflection_point = portal_point
                path.interaction_type = f"portal_{item.portal_kind}"
                path.material_id = item.material_id
                msg.early_paths.append(path)
        return msg

    def _is_occluded(self, source: Point, listener: Point) -> bool:
        if self._map is None:
            return False

        a = self._world_to_grid(source)
        b = self._world_to_grid(listener)
        if a is None or b is None:
            return False

        occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        for x, y in self._bresenham(a[0], a[1], b[0], b[1]):
            idx = y * self._map.info.width + x
            if 0 <= idx < len(self._map.data) and self._map.data[idx] >= occupied_threshold:
                return True

        return False

    def _world_to_grid(self, point: Point) -> tuple[int, int] | None:
        assert self._map is not None
        origin = self._map.info.origin.position
        resolution = self._map.info.resolution
        dx = point.x - origin.x
        dy = point.y - origin.y
        orientation = self._map.info.origin.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        x = math.floor(local_x / resolution + 1e-9)
        y = math.floor(local_y / resolution + 1e-9)

        if x < 0 or y < 0 or x >= self._map.info.width or y >= self._map.info.height:
            return None
        return x, y

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


def main() -> None:
    rclpy.init()
    node = SoundPropagationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
