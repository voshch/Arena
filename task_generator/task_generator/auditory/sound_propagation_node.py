from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from arena_people_msgs.msg import Pedestrians
from arena_simulation_setup.tree.World import WorldIdentifier
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import String
from task_generator.auditory.acoustic_scene import AcousticScene
from task_generator.auditory.acoustic_world_graph import AcousticWorldGraph
from task_generator.auditory.material_catalog import AcousticMaterialCatalog
from task_generator.auditory.propagation import Level3Propagation
from task_generator_msgs.msg import AcousticPath, HeardSoundEvent, RobotFleet, SoundEvent
from task_generator.auditory.qos_profiles import acoustic_metadata_qos, transient_event_qos
from task_generator.auditory.acoustic_room_spec import (
    AcousticRoomSpec,
    AcousticRoomSpecBuilder,
    AcousticRoomSpecConfig,
)
from task_generator.auditory.pyroomacoustics_adapter import (
    PyroomacousticsAdapter,
    PyroomacousticsConfig,
    PyroomacousticsUnavailableError,
    RoomImpulseResponse,
)
from task_generator.auditory.portal_coupling import (
    OneDoorRirCoupler,
    PortalCouplingConfig,
    PortalCouplingResult,
)



class SoundPropagationNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("sound_propagation_node", **kwargs)

        self.declare_parameter("sound_events_topic", "human_sound_events")
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
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
        self.declare_parameter("odom_topic_template", "{namespace}/{name}_velocity_controller/odom")
        self.declare_parameter("propagation_backend", "level3")
        self.declare_parameter("pyroom_sample_rate_hz", 44100)
        self.declare_parameter("pyroom_max_order", 3)
        self.declare_parameter("pyroom_temperature_c", 20.0)
        self.declare_parameter("pyroom_relative_humidity_percent", 50.0)
        self.declare_parameter("pyroom_ceiling_height_m", 3.0)
        self.declare_parameter("portal_adjacency_tolerance_m", 0.08)
        self.declare_parameter("portal_inset_m", 0.03)
        self.declare_parameter("portal_loss_db", 3.0)
        self.declare_parameter("portal_source_early_window_sec", 0.08)
        self.declare_parameter("portal_max_rir_duration_sec", 2.0)
        self.declare_parameter("portal_position_quantization_m", 0.10)
        self.declare_parameter("portal_rir_cache_size", 256)
        self.declare_parameter("validate_zone_coverage", True)
        self.declare_parameter("zone_coverage_stride_cells", 10)
        self._scene: AcousticScene | None = None
        self._world_name = ""
        self._pending_world_name = ""
        self._world_load_future: Future | None = None
        self._world_loader = ThreadPoolExecutor(max_workers=1)
        self.create_subscription(String, str(self.get_parameter("world_topic").value), self._cb_world, 1)
        self._world_load_timer = self.create_timer(0.1, self._poll_world_load)
        self._peds: dict[int, object] = {}
        self._robots: dict[str, Point] = {}
        self._map: OccupancyGrid | None = None
        self._world_graph: AcousticWorldGraph | None = None
        self._portal_coupler: OneDoorRirCoupler | None = None
        self._coverage_signature: tuple[object, ...] | None = None
        self._reported_routes: set[tuple[str, str, str, str]] = set()
        self._odom_subs = []
        sound_events_topic = str(self.get_parameter("sound_events_topic").value)
        heard_sound_events_topic = str(self.get_parameter("heard_sound_events_topic").value)
        peds_topic = str(self.get_parameter("arena_peds_topic").value)
        map_topic = str(self.get_parameter("map_topic").value)
        robot_fleet_topic = str(self.get_parameter("robot_fleet_topic").value)
        self._heard_pub = self.create_publisher( HeardSoundEvent,  heard_sound_events_topic, transient_event_qos())
        self.create_subscription(SoundEvent, sound_events_topic, self._cb_sound_event, transient_event_qos())
        self.create_subscription(Pedestrians, peds_topic, self._cb_peds, 10)
        self.create_subscription(OccupancyGrid, map_topic, self._cb_map, 1)
        self.create_subscription(RobotFleet, robot_fleet_topic, self._cb_robot_fleet, acoustic_metadata_qos())
        share = Path(get_package_share_directory("task_generator"))
        materials = AcousticMaterialCatalog(share / "config" / "auditory" / "acoustic_materials.yaml")
        self._room_specs = ()
        backend = str(
            self.get_parameter("propagation_backend").value
        )
        self._requested_backend = backend

        self._pra_adapter = None

        if backend == "pyroomacoustics":
            try:
                # The adapter keeps PRA optional for the Level3 backend, but
                # fail fast when the user explicitly selected this backend.
                import pyroomacoustics  # noqa: F401

                pra_config = PyroomacousticsConfig(
                    sample_rate_hz=int(
                        self.get_parameter("pyroom_sample_rate_hz").value
                    ),
                    max_order=int(
                        self.get_parameter("pyroom_max_order").value
                    ),
                    temperature_c=float(
                        self.get_parameter("pyroom_temperature_c").value
                    ),
                    relative_humidity_percent=float(
                        self.get_parameter(
                            "pyroom_relative_humidity_percent"
                        ).value
                    ),
                )
                self._pra_adapter = PyroomacousticsAdapter(
                    materials,
                    pra_config,
                )
            except (ImportError, PyroomacousticsUnavailableError, ValueError) as exc:
                raise RuntimeError(
                    "pyroomacoustics backend requested but could not be "
                    f"initialized: {exc}"
                ) from exc
        elif backend != "level3":
            raise ValueError(
                "propagation_backend must be 'level3' or "
                "'pyroomacoustics'"
            )
        self._propagation = Level3Propagation(materials)

    def _cb_peds(self, msg: Pedestrians) -> None:
        self._peds = {int(p.id): p for p in msg.pedestrians}
    
    def _cb_world(self, msg: String) -> None:
        world_name = msg.data.strip()
        if not world_name or world_name == self._world_name:
            return

        self._pending_world_name = world_name

        if self._world_load_future is not None and not self._world_load_future.done():
            return

        self._start_world_load(world_name)
    
    def _start_world_load(self, world_name: str) -> None:
        self.get_logger().info(f"loading acoustic scene for world {world_name!r}")

        self._world_load_future = self._world_loader.submit(
            self._load_acoustic_scene,
            world_name,
            float(self.get_parameter("pyroom_ceiling_height_m").value),
            float(self.get_parameter("portal_adjacency_tolerance_m").value),
        )
    
    @staticmethod
    def _load_acoustic_scene(
        world_name: str,
        ceiling_height_m: float,
        adjacency_tolerance_m: float,
    ) -> tuple[
        str,
        AcousticScene,
        tuple[AcousticRoomSpec, ...],
        AcousticWorldGraph,
    ]:
        world_view = WorldIdentifier(world_name).resolve_sync()
        world_description = world_view.load()

        scene = AcousticScene.from_world(world_description)

        room_config = AcousticRoomSpecConfig(
            ceiling_height_m=ceiling_height_m,
        )
        room_specs = AcousticRoomSpecBuilder(
            room_config
        ).from_world(world_description)
        graph = AcousticWorldGraph.from_world(
            world_description,
            room_specs,
            adjacency_tolerance_m=adjacency_tolerance_m,
        )

        return world_name, scene, room_specs, graph
    
    def _poll_world_load(self) -> None:
        if self._world_load_future is None:
            return

        if not self._world_load_future.done():
            return

        future = self._world_load_future
        self._world_load_future = None

        try:
            world_name, scene, room_specs, graph = future.result()
        except Exception as exc:
            self.get_logger().error(f"failed to load acoustic scene: {exc!r}")
            return

        self._scene = scene
        self._room_specs = room_specs
        self._world_graph = graph
        self._world_name = world_name
        self._coverage_signature = None
        self._portal_coupler = (
            OneDoorRirCoupler(
                self._pra_adapter,
                graph,
                world_name=world_name,
                config=self._portal_coupling_config(),
            )
            if self._pra_adapter is not None else None
        )
        self.get_logger().info(
            f"loaded acoustic scene for world {world_name!r} "
            f"({len(room_specs)} rooms, {len(graph.portals)} paired portals, "
            f"{len(graph.unpaired_doors)} unpaired doors)"
        )
        for door in graph.unpaired_doors:
            self.get_logger().warning(
                f"acoustic door {door.door_name!r} in {door.owner_zone!r} "
                f"was not paired: {door.reason}"
            )
        for portal in graph.portals:
            self.get_logger().info(
                f"acoustic portal {portal.portal_id}: {portal.zone_a!r} "
                f"<-> {portal.zone_b!r}, material={portal.material_id!r}"
            )
        self._validate_acoustic_zone_coverage()

        if self._pending_world_name and self._pending_world_name != self._world_name:
            self._start_world_load(self._pending_world_name)
    
    def destroy_node(self) -> bool:
        self._world_loader.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _cb_map(self, msg: OccupancyGrid) -> None:
        self._map = msg
        self._validate_acoustic_zone_coverage()

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
        )

    def _validate_acoustic_zone_coverage(self) -> None:
        if (
            not bool(self.get_parameter("validate_zone_coverage").value)
            or self._scene is None
            or self._map is None
        ):
            return
        info = self._map.info
        stride = max(
            int(self.get_parameter("zone_coverage_stride_cells").value), 1
        )
        signature = (
            self._world_name,
            info.width,
            info.height,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
            stride,
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
                if self._scene.zone_at_xy(world_x, world_y) is None:
                    if len(uncovered) < 8:
                        uncovered.append((world_x, world_y))

        if not uncovered:
            self.get_logger().info(
                f"acoustic zone coverage validated on {traversable} sampled "
                f"traversable cells (stride={stride})"
            )
        else:
            examples = ", ".join(f"({x:.2f},{y:.2f})" for x, y in uncovered)
            self.get_logger().warning(
                "traversable map locations exist outside all acoustic zones; "
                f"first samples: {examples}. These events will log an explicit "
                "backend fallback reason."
            )

    def _cb_robot_fleet(self, msg: RobotFleet) -> None:
        for robot in msg.robots:
            # topic = f"{robot.ns}/odom"
            topic = str(self.get_parameter("odom_topic_template").value).format(namespace=str(robot.ns).rstrip("/"),name=str(robot.name))
            sub = self.create_subscription(Odometry, topic, lambda odom, name=robot.name: self._cb_robot_odom(name, odom), 10)
            self._odom_subs.append(sub)

    def _cb_robot_odom(self, robot_name: str, msg: Odometry) -> None:
        self._robots[f"robot:{robot_name}"] = msg.pose.pose.position
        self.get_logger().info(f"robot listener updated: robot:{robot_name}")


    def _cb_sound_event(self, event: SoundEvent) -> None:
        if not event.sound_type.strip():
            return

        listeners: dict[str, Point] = {}

        for agent_id, ped in self._peds.items():
            if agent_id == event.source_agent_id:
                continue
            listeners[f"agent:{agent_id}"] = ped.pose.position

        listeners.update(self._robots)

        if not bool(self.get_parameter("robots_hear_self").value):
            listeners.pop(f"robot:{event.source_agent_name}", None)

        for listener_id, listener_pos in listeners.items():
            heard = self._calculate_heard_event(event, listener_id, listener_pos)
            if heard.audible or bool(self.get_parameter("publish_inaudible").value):
                self._heard_pub.publish(heard)
    
    def _effective_sound_distance(self, geometric_distance: float, event: SoundEvent, listener_id: str,) -> float:
        if listener_id == f"robot:{event.source_agent_name}":
            return max(float(self.get_parameter("self_hearing_distance_m").value),1e-3)

        return max(geometric_distance, float(self.get_parameter("minimum_propagation_distance_m").value),1e-3)


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
        occlusion_penalty = (float(self.get_parameter("occlusion_penalty_db").value) if occluded else 0.0)
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


    def _calculate_heard_event(
        self, event: SoundEvent, listener_id: str, listener_pos: Point
    ) -> HeardSoundEvent:
        if self._scene is None:
            msg = self._calculate_legacy_event(event, listener_id, listener_pos)
            return self._finalize_backend(
                msg,
                backend="legacy_distance_occlusion",
                used_fallback=self._requested_backend == "pyroomacoustics",
                fallback_reason=(
                    "acoustic_scene_not_loaded"
                    if self._requested_backend == "pyroomacoustics" else ""
                ),
            )

        if (
            listener_id == f"robot:{event.source_agent_name}"
            and self._pra_adapter is None
        ):
            msg = self._calculate_legacy_event(event, listener_id, listener_pos)
            return self._finalize_backend(
                msg,
                backend="legacy_self_hearing",
                used_fallback=True,
                fallback_reason="level3_self_hearing_uses_legacy_distance",
            )

        fallback_reason = ""
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
                else:
                    try:
                        return self._finalize_backend(
                            self._calculate_pyroom_event(
                                event, listener_id, listener_pos, room_spec
                            ),
                            backend="pyroomacoustics_same_room",
                            used_fallback=False,
                            fallback_reason="",
                        )
                    except Exception as exc:
                        fallback_reason = "same_room_rir_error:" + type(exc).__name__
                        self.get_logger().warning(
                            "pyroomacoustics same-room RIR failed for "
                            f"{event.source_agent_name}->{listener_id} in "
                            f"{room_spec.zone_name!r}: {exc}"
                        )
            else:
                portal = (
                    self._world_graph.direct_portal(
                        source_zone.name,
                        listener_zone.name,
                        source_xy=(event.source_position.x, event.source_position.y),
                        listener_xy=(listener_pos.x, listener_pos.y),
                    )
                    if self._world_graph is not None else None
                )
                if portal is None:
                    fallback_reason = "no_direct_portal_between_zones"
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
                            backend="pyroomacoustics_one_door",
                            used_fallback=False,
                            fallback_reason="",
                            portal_id=msg.portal_id,
                        )
                    except Exception as exc:
                        fallback_reason = "one_door_rir_error:" + type(exc).__name__
                        self.get_logger().warning(
                            "pyroomacoustics one-door coupling failed for "
                            f"{source_zone.name!r}->{listener_zone.name!r} "
                            f"through {portal.portal_id!r}: {exc}"
                        )

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

        return self._finalize_backend(
            msg,
            backend="level3",
            used_fallback=self._requested_backend == "pyroomacoustics",
            fallback_reason=fallback_reason,
        )

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
            self.get_logger().info(
                f"actual propagation backend={backend!r} for "
                f"{msg.source_zone!r}->{msg.listener_zone!r}"
                f"{portal_detail}{detail}"
            )
        return msg

    @staticmethod
    def _source_height(event: SoundEvent) -> float:
        sound_type = f"{event.sound_type} {event.label}".lower()
        if "foot" in sound_type or "step" in sound_type:
            return 0.05
        return 1.60

    @staticmethod
    def _listener_height(listener_id: str) -> float:
        # The robot odometry pose is a ground-contact position, not the
        # microphone position. Human listeners use an approximate ear height.
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
            self._listener_height(listener_id),
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
                self._listener_height(listener_id),
            ),
        )
        self.get_logger().debug(
            f"portal RIR cache: entries={self._portal_coupler.cache_entries}, "
            f"hits={result.cache_hits}, misses={result.cache_misses}"
        )
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
        delay_sec = (
            rir.global_delay_samples + peak_index
        ) / float(rir.sample_rate_hz)

        # Report a conservative late-energy estimate as reverb gain. The
        # message schema has no RIR field; audio convolution is a later stage.
        direct_end = min(peak_index + 1, samples.size)
        late_energy = float(np.sqrt(np.mean(samples[direct_end:] ** 2))) \
            if direct_end < samples.size else 0.0
        reverb_gain_db = (
            20.0 * math.log10(max(late_energy / peak_amplitude, 1e-12))
            if late_energy > 0.0 else -120.0
        )

        dx = event.source_position.x - listener_position.x
        dy = event.source_position.y - listener_position.y
        distance = max(math.hypot(dx, dy), 1.0)
        # For a coupled RIR the wall crossing is intentional: the two room
        # responses and portal loss already model the source-door-listener
        # path. Applying the straight-line map penalty would attenuate it a
        # second time merely because source and listener occupy two rooms.
        occluded = (
            False
            if portal_result is not None
            else self._is_occluded(event.source_position, listener_position)
        )
        occlusion_penalty = (
            float(self.get_parameter("occlusion_penalty_db").value)
            if occluded else 0.0
        )
        threshold = float(
            self.get_parameter("default_hearing_threshold_db").value
        )

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
        msg.received_volume_db = float(
            event.source_volume_db + gain_db - occlusion_penalty
        )
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
            portal = portal_result.portal
            msg.portal_id = portal.portal_id
            msg.portal_position.x = portal.center_xy[0]
            msg.portal_position.y = portal.center_xy[1]
            msg.portal_position.z = 0.5 * portal.height_m
            path = AcousticPath()
            path.delay.sec = int(delay_sec)
            path.delay.nanosec = int((delay_sec % 1.0) * 1_000_000_000)
            path.gain_db = float(gain_db)
            path.bearing_rad = float(
                math.atan2(
                    portal.center_xy[1] - listener_position.y,
                    portal.center_xy[0] - listener_position.x,
                )
            )
            path.reflection_point = msg.portal_position
            path.interaction_type = "portal"
            path.material_id = portal.material_id
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

        x = int((point.x - origin.x) / resolution)
        y = int((point.y - origin.y) / resolution)

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
