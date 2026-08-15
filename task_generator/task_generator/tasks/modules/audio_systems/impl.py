from __future__ import annotations

import hashlib
import math
import threading
import traceback

import attrs
import rclpy
import tf2_ros
import yaml
from arena_simulation_setup.shared import Obstacle, Position
from arena_simulation_setup.tree.World import WorldDescription, WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import (
    AudioEmitter,
    AudioSystem,
    ScenarioAudio,
)
from arena_simulation_setup.utils.cattrs import converter
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Vector3
from rclpy.clock import Clock, ClockType
from task_generator_msgs.msg import (
    AudioSystemState,
    ContinuousAudioSourceState,
)
from task_generator_msgs.srv import (
    RemoveAudioSystem,
    SetAudioSystem,
    SpawnAudioSource,
)

from task_generator.auditory.qos_profiles import (
    acoustic_metadata_qos,
    continuous_audio_qos,
)
from task_generator.tasks.modules import TM_Module


@attrs.frozen
class _ResolvedEmitter:
    source_id: str
    name: str
    position: Point
    yaw: float
    source_volume_db: float


@attrs.define
class _RuntimeSystem:
    specification: AudioSystem
    emitters: tuple[_ResolvedEmitter, ...]
    active: bool
    program_start_time: Time


class Mod_AudioSystems(TM_Module):
    """Publish scenario and launch-configured static audio emitters."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._systems_lock = threading.RLock()
        self._systems: dict[str, _RuntimeSystem] = {}
        self._runtime_system_ids: set[str] = set()
        self._pending_system_states: list[
            tuple[str, _RuntimeSystem, bool]
        ] = []
        self._source_publisher = self.node.create_publisher(
            ContinuousAudioSourceState,
            self.node.service_namespace("continuous_audio_sources"),
            continuous_audio_qos(),
        )
        self._state_publisher = self.node.create_publisher(
            AudioSystemState,
            self.node.service_namespace("audio_system_states"),
            acoustic_metadata_qos(depth=32),
        )
        self._service = self.node.create_service(
            SetAudioSystem,
            self.node.service_namespace("runtime", "set_audio_system"),
            self._set_audio_system,
        )
        self._spawn_service = self.node.create_service(
            SpawnAudioSource,
            self.node.service_namespace("runtime", "spawn_audio_source"),
            self._spawn_audio_source,
        )
        self._remove_service = self.node.create_service(
            RemoveAudioSystem,
            self.node.service_namespace("runtime", "remove_audio_system"),
            self._remove_audio_system,
        )
        self._timer = self.node.create_timer(
            0.1,
            self._publish_sources,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def after_reset(self) -> None:
        with self._systems_lock:
            self._publish_removed_sources()
            self._systems.clear()
            self._runtime_system_ids.clear()

        world_name = self._ctx.world_manager.loaded_world
        scenario_name = str(
            self.node.get_parameter("task.scenario.file").value
        ).strip()
        world_view = WorldIdentifier(world_name).resolve_sync()
        specifications = self._configured_systems()
        available_scenarios = {
            identifier.shortname
            for identifier in world_view.scenario.listall()
        }
        if scenario_name in available_scenarios:
            scenario_view = world_view.scenario(scenario_name).resolve_sync()
            specifications = [
                *scenario_view.load_audio().systems,
                *specifications,
            ]
        world = world_view.load()

        names = [specification.name for specification in specifications]
        duplicates = sorted(
            name for name in set(names) if names.count(name) > 1
        )
        if duplicates:
            raise ValueError(
                "static audio system names must be unique across scenario "
                f"and launch configuration: {duplicates}"
            )

        with self._systems_lock:
            for specification in specifications:
                emitters = self._resolve_emitters(specification, world)
                start_time = self.node.get_clock().now().to_msg()
                self._systems[specification.name] = _RuntimeSystem(
                    specification=specification,
                    emitters=emitters,
                    active=specification.initially_active,
                    program_start_time=start_time,
                )

            self._publish_system_states()
            loaded_system_count = len(self._systems)
        origin = (
            f"scenario {scenario_name!r} and launch configuration"
            if scenario_name in available_scenarios
            else "launch configuration"
        )
        self._logger.info(
            f"loaded {loaded_system_count} static audio system(s) from {origin}"
        )

    def _configured_systems(self) -> list[AudioSystem]:
        if not self.node.has_parameter("static_audio_devices"):
            return []
        raw = str(
            self.node.get_parameter("static_audio_devices").value
        ).strip()
        parsed = yaml.safe_load(raw) if raw else []
        if parsed is None:
            parsed = []
        if isinstance(parsed, dict):
            parsed = parsed.get("systems", [])
        if not isinstance(parsed, list):
            raise ValueError(
                "static_audio_devices must be a YAML list or a mapping with "
                "a systems list"
            )
        return converter.structure(
            {"systems": parsed},
            ScenarioAudio,
        ).systems

    def _resolve_emitters(
        self,
        system: AudioSystem,
        world: WorldDescription,
    ) -> tuple[_ResolvedEmitter, ...]:
        indexed_entities: dict[str, list[tuple[str, Obstacle]]] = {}
        for level_id, level in world.levels.items():
            for entity in level.all_static_entities:
                indexed_entities.setdefault(entity.name, []).append(
                    (str(level_id), entity)
                )

        resolved: list[_ResolvedEmitter] = []
        for emitter in system.emitters:
            yaw = 0.0
            level_id = emitter.level
            if emitter.entity_ref:
                matches = indexed_entities.get(emitter.entity_ref, [])
                if not matches:
                    raise ValueError(
                        f"audio emitter {system.name!r}/{emitter.name!r} "
                        f"references unknown static entity "
                        f"{emitter.entity_ref!r}"
                    )
                if len(matches) > 1:
                    raise ValueError(
                        f"audio emitter {system.name!r}/{emitter.name!r} "
                        f"references ambiguous static entity "
                        f"{emitter.entity_ref!r}"
                    )
                level_id, entity = matches[0]
                yaw = float(entity.pose.orientation.to_yaw())
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                position = Position(
                    x=(
                        entity.pose.position.x
                        + cos_yaw * emitter.offset.x
                        - sin_yaw * emitter.offset.y
                    ),
                    y=(
                        entity.pose.position.y
                        + sin_yaw * emitter.offset.x
                        + cos_yaw * emitter.offset.y
                    ),
                    z=entity.pose.position.z + emitter.offset.z,
                )
            else:
                if emitter.position is None:
                    raise RuntimeError("validated audio emitter has no position")
                if not level_id:
                    if len(world.levels) != 1:
                        raise ValueError(
                            f"audio emitter {system.name!r}/{emitter.name!r} "
                            "requires level in a multi-level world"
                        )
                    level_id = next(iter(world.levels))
                if level_id not in world.levels:
                    raise ValueError(
                        f"audio emitter {system.name!r}/{emitter.name!r} "
                        f"references unknown level {level_id!r}"
                    )
                position = emitter.position + emitter.offset

            realized = self.node._realizer.realize(position, level_id)
            resolved.append(
                _ResolvedEmitter(
                    source_id=f"environment:{system.name}:{emitter.name}",
                    name=emitter.name,
                    position=realized.to_msg(),
                    yaw=yaw,
                    source_volume_db=emitter.source_volume_db,
                )
            )
        return tuple(resolved)

    def _set_audio_system(
        self,
        request: SetAudioSystem.Request,
        response: SetAudioSystem.Response,
    ) -> SetAudioSystem.Response:
        system_id = str(request.system_id).strip()
        with self._systems_lock:
            runtime = self._systems.get(system_id)
            if runtime is None:
                response.error_msg = f"unknown audio system {system_id!r}"
                return response
            active = bool(request.active)
            if active and not runtime.active:
                runtime.program_start_time = self.node.get_clock().now().to_msg()
            runtime.active = active
            self._queue_system_state(system_id, runtime)
        response.success = True
        return response

    def _spawn_audio_source(
        self,
        request: SpawnAudioSource.Request,
        response: SpawnAudioSource.Response,
    ) -> SpawnAudioSource.Response:
        try:
            return self._spawn_audio_source_impl(request, response)
        except Exception as exc:
            self._logger.error(
                "spawning runtime audio source failed:\n"
                f"{traceback.format_exc()}"
            )
            response.error_msg = f"{type(exc).__name__}: {exc}"
            return response

    def _spawn_audio_source_impl(
        self,
        request: SpawnAudioSource.Request,
        response: SpawnAudioSource.Response,
    ) -> SpawnAudioSource.Response:
        mode = str(request.mode).strip().lower()
        settings = {
            "music": ("radio_loop", 62.0, ("radio", "music")),
            "alarm": ("alarm_loop", 88.0, ("alarm", "emergency")),
        }.get(mode)
        if settings is None:
            response.error_msg = "mode must be 'music' or 'alarm'"
            return response
        default_asset_id, default_volume_db, semantic_tags = settings
        if bool(request.customize_playback):
            asset_id = str(request.asset_id).strip() or default_asset_id
            source_volume_db = float(request.source_volume_db)
            if not math.isfinite(source_volume_db):
                response.error_msg = "source volume must be finite"
                return response
            loop = bool(request.loop)
            initially_active = bool(request.initially_active)
        else:
            asset_id = default_asset_id
            source_volume_db = default_volume_db
            loop = True
            initially_active = True

        frame_id = str(request.pose.header.frame_id).strip().lstrip("/")
        if not frame_id:
            response.error_msg = "audio source pose requires a frame ID"
            return response

        position = request.pose.pose.position
        orientation = request.pose.pose.orientation
        values = (
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        if not all(math.isfinite(value) for value in values):
            response.error_msg = "audio source pose must be finite"
            return response
        orientation_norm = math.sqrt(
            orientation.x**2
            + orientation.y**2
            + orientation.z**2
            + orientation.w**2
        )
        if orientation_norm <= 1e-9:
            response.error_msg = "audio source orientation must be valid"
            return response

        map_position = Point(
            x=float(position.x),
            y=float(position.y),
            z=float(position.z),
        )
        map_orientation = (
            float(orientation.x) / orientation_norm,
            float(orientation.y) / orientation_norm,
            float(orientation.z) / orientation_norm,
            float(orientation.w) / orientation_norm,
        )
        if frame_id != "map":
            try:
                transform = self.node.tf_buffer.lookup_transform(
                    "map",
                    frame_id,
                    rclpy.time.Time(),
                ).transform
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                response.error_msg = (
                    f"cannot transform audio source from {frame_id!r} "
                    f"to 'map': {exc}"
                )
                return response
            transform_orientation = (
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            )
            map_position = self._transform_position(
                map_position,
                transform_orientation,
                transform.translation,
            )
            map_orientation = self._multiply_quaternions(
                transform_orientation,
                map_orientation,
            )

        if map_position.z < 0.0:
            response.error_msg = "audio source height cannot be below the floor"
            return response

        with self._systems_lock:
            index = 1
            while f"runtime_{mode}_{index}" in self._systems:
                index += 1
            system_id = f"runtime_{mode}_{index}"
            emitter_name = "radio" if mode == "music" else "alarm"
            emitter = AudioEmitter(
                name=emitter_name,
                position=Position(
                    float(map_position.x),
                    float(map_position.y),
                    float(map_position.z),
                ),
                source_volume_db=source_volume_db,
            )
            specification = AudioSystem(
                name=system_id,
                sound_type=mode,
                asset_id=asset_id,
                emitters=[emitter],
                loop=loop,
                initially_active=initially_active,
                semantic_tags=["runtime", *semantic_tags],
            )
            qx, qy, qz, qw = map_orientation
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy**2 + qz**2),
            )
            resolved = _ResolvedEmitter(
                source_id=f"environment:{system_id}:{emitter_name}",
                name=emitter_name,
                position=Point(
                    x=float(map_position.x),
                    y=float(map_position.y),
                    z=float(map_position.z),
                ),
                yaw=float(yaw),
                source_volume_db=source_volume_db,
            )
            runtime = _RuntimeSystem(
                specification=specification,
                emitters=(resolved,),
                active=initially_active,
                program_start_time=self.node.get_clock().now().to_msg(),
            )
            self._systems[system_id] = runtime
            self._runtime_system_ids.add(system_id)
            self._queue_system_state(system_id, runtime)
        response.system_id = system_id
        response.success = True
        self._logger.info(
            f"spawned {mode} source {system_id!r} at "
            f"({map_position.x:.2f}, {map_position.y:.2f}, "
            f"{map_position.z:.2f})"
        )
        return response

    def _remove_audio_system(
        self,
        request: RemoveAudioSystem.Request,
        response: RemoveAudioSystem.Response,
    ) -> RemoveAudioSystem.Response:
        system_id = str(request.system_id).strip()
        with self._systems_lock:
            runtime = self._systems.get(system_id)
            if runtime is None:
                response.error_msg = f"unknown audio system {system_id!r}"
                return response
            if system_id not in self._runtime_system_ids:
                response.error_msg = (
                    "scenario and launch-configured audio systems cannot be removed"
                )
                return response
            runtime.active = False
            self._queue_system_state(system_id, runtime, removed=True)
            self._systems.pop(system_id)
            self._runtime_system_ids.remove(system_id)
        response.success = True
        return response

    @staticmethod
    def _multiply_quaternions(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )

    @staticmethod
    def _transform_position(
        position: Point,
        rotation: tuple[float, float, float, float],
        translation: Vector3,
    ) -> Point:
        qx, qy, qz, qw = rotation
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm > 0.0:
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm
        x = float(position.x)
        y = float(position.y)
        z = float(position.z)
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

    def _publish_sources(self) -> None:
        try:
            self._publish_sources_impl()
        except Exception:
            self._logger.error(
                "publishing static audio state failed:\n"
                f"{traceback.format_exc()}"
            )

    def _publish_sources_impl(self) -> None:
        with self._systems_lock:
            systems = tuple(attrs.evolve(runtime) for runtime in self._systems.values())
            pending_states = tuple(self._pending_system_states)
        for _, runtime, removed in pending_states:
            if removed:
                self._publish_runtime_system(runtime)
        for runtime in systems:
            self._publish_runtime_system(runtime)
        for system_id, runtime, removed in pending_states:
            self._publish_system_state(system_id, runtime, removed=removed)
        with self._systems_lock:
            del self._pending_system_states[:len(pending_states)]

    def _publish_runtime_system(self, runtime: _RuntimeSystem) -> None:
        specification = runtime.specification
        stamp = self.node.get_clock().now().to_msg()
        tags = [
            "environment",
            "static",
            specification.sound_type,
            *map(str, specification.semantic_tags),
        ]
        for emitter in runtime.emitters:
            msg = ContinuousAudioSourceState()
            msg.header.stamp = stamp
            msg.header.frame_id = "map"
            msg.source_id = emitter.source_id
            msg.source_agent_id = self._numeric_id(emitter.source_id)
            msg.source_agent_name = emitter.name
            msg.source_model = "static_audio_source"
            msg.sound_type = specification.sound_type
            msg.source_backend = "wav_loop"
            msg.system_id = specification.name
            msg.asset_id = specification.asset_id
            msg.label = specification.name
            msg.loop = specification.loop
            msg.program_start_time = runtime.program_start_time
            msg.source_position = emitter.position
            msg.source_yaw = emitter.yaw
            msg.source_volume_db = emitter.source_volume_db
            msg.reference_distance_m = specification.reference_distance_m
            msg.active = runtime.active
            msg.deterministic_seed = self._deterministic_seed(emitter.source_id)
            msg.semantic_tags = tags
            self._source_publisher.publish(msg)

    def _publish_removed_sources(self) -> None:
        with self._systems_lock:
            for system_id, runtime in self._systems.items():
                runtime.active = False
                self._queue_system_state(system_id, runtime, removed=True)

    def _publish_system_states(self) -> None:
        with self._systems_lock:
            for system_id, runtime in self._systems.items():
                self._queue_system_state(system_id, runtime)

    def _queue_system_state(
        self,
        system_id: str,
        runtime: _RuntimeSystem,
        removed: bool = False,
    ) -> None:
        with self._systems_lock:
            self._pending_system_states.append(
                (system_id, attrs.evolve(runtime), removed)
            )

    def _publish_system_state(
        self,
        system_id: str,
        runtime: _RuntimeSystem,
        removed: bool = False,
    ) -> None:
        specification = runtime.specification
        msg = AudioSystemState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.system_id = system_id
        msg.sound_type = specification.sound_type
        msg.asset_id = specification.asset_id
        msg.loop = specification.loop
        msg.active = runtime.active
        msg.program_start_time = runtime.program_start_time
        msg.emitter_ids = (
            []
            if removed
            else [emitter.source_id for emitter in runtime.emitters]
        )
        self._state_publisher.publish(msg)

    @staticmethod
    def _numeric_id(source_id: str) -> int:
        digest = hashlib.blake2b(source_id.encode(), digest_size=4).digest()
        return int.from_bytes(digest, "little") & 0x7FFFFFFF

    @staticmethod
    def _deterministic_seed(source_id: str) -> int:
        digest = hashlib.blake2b(source_id.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little")
