from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import yaml
from arena_simulation_setup.shared import Obstacle, Position
from arena_simulation_setup.tree.World import WorldDescription, WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import AudioSystem, ScenarioAudio
from arena_simulation_setup.utils.cattrs import converter
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from task_generator_msgs.msg import (
    AudioSystemState,
    ContinuousAudioSourceState,
)
from task_generator_msgs.srv import SetAudioSystem

from task_generator.auditory.qos_profiles import (
    acoustic_metadata_qos,
    continuous_audio_qos,
)
from task_generator.tasks.modules import TM_Module


@dataclass(frozen=True)
class _ResolvedEmitter:
    source_id: str
    name: str
    position: Point
    yaw: float
    source_volume_db: float


@dataclass
class _RuntimeSystem:
    specification: AudioSystem
    emitters: tuple[_ResolvedEmitter, ...]
    active: bool
    program_start_time: Time


class Mod_AudioSystems(TM_Module):
    """Publish scenario and launch-configured static audio emitters."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._systems: dict[str, _RuntimeSystem] = {}
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
        self._timer = self.node.create_timer(0.1, self._publish_sources)

    def after_reset(self) -> None:
        self._publish_removed_sources()
        self._systems.clear()

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

        for specification in specifications:
            emitters = self._resolve_emitters(specification, world)
            start_time = self.node.get_clock().now().to_msg()
            self._systems[specification.name] = _RuntimeSystem(
                specification=specification,
                emitters=emitters,
                active=specification.initially_active,
                program_start_time=start_time,
            )

        self._publish_sources()
        self._publish_system_states()
        origin = (
            f"scenario {scenario_name!r} and launch configuration"
            if scenario_name in available_scenarios
            else "launch configuration"
        )
        self._logger.info(
            f"loaded {len(self._systems)} static audio system(s) from {origin}"
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
        runtime = self._systems.get(system_id)
        if runtime is None:
            response.error_msg = f"unknown audio system {system_id!r}"
            return response
        active = bool(request.active)
        if active and not runtime.active:
            runtime.program_start_time = self.node.get_clock().now().to_msg()
        runtime.active = active
        self._publish_runtime_system(runtime)
        self._publish_system_state(system_id, runtime)
        response.success = True
        return response

    def _publish_sources(self) -> None:
        for runtime in self._systems.values():
            self._publish_runtime_system(runtime)

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
        for system_id, runtime in self._systems.items():
            runtime.active = False
            self._publish_runtime_system(runtime)
            self._publish_system_state(system_id, runtime, removed=True)

    def _publish_system_states(self) -> None:
        for system_id, runtime in self._systems.items():
            self._publish_system_state(system_id, runtime)

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
