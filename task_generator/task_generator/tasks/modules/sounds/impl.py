from __future__ import annotations

import hashlib
import math
import traceback
from collections.abc import Sequence
from pathlib import Path

import attrs
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_simulation_setup.shared import Obstacle, Position, SemanticCfg, Sound
from arena_simulation_setup.tree.World import WorldDescription, WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import Scenario
from arena_simulation_setup.utils.cattrs import converter
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Vector3
from rclpy.clock import Clock, ClockType
from task_generator_msgs.msg import ContinuousAudioSourceState
from task_generator_msgs.srv import RemoveSound, SpawnSound

from task_generator.auditory.qos_profiles import continuous_audio_qos
from task_generator.tasks.modules import TM_Module


def _index_static_entities(world: WorldDescription, scenario_static: Sequence[Obstacle] = ()) -> dict[str, list[tuple[str, Obstacle]]]:
    """Every static entity (world levels plus the episode's scenario), keyed by bare name, alongside its level."""
    indexed: dict[str, list[tuple[str, Obstacle]]] = {}
    for level_id, level in world.levels.items():
        for entity in level.all_static_entities:
            indexed.setdefault(entity.name, []).append((str(level_id), entity))
    for entity in scenario_static:
        indexed.setdefault(entity.name, []).append((entity.level_id or "", entity))
    return indexed


def _resolve_sound_placement(
    sound: Sound,
    world: WorldDescription,
    indexed_entities: dict[str, list[tuple[str, Obstacle]]],
    context_level_id: str | None,
) -> tuple[Position, float, str]:
    """Level-local (position, yaw, level_id) for a Sound: context_level_id pins a world-embedded sound to its structural level, None falls back to sound.level or the world's sole level."""
    if sound.entity_ref:
        matches = indexed_entities.get(sound.entity_ref, [])
        if not matches:
            raise ValueError(f"sound {sound.name!r} references unknown static entity {sound.entity_ref!r}")
        if len(matches) > 1:
            raise ValueError(f"sound {sound.name!r} references ambiguous static entity {sound.entity_ref!r}")
        level_id, entity = matches[0]
        yaw = float(entity.pose.orientation.to_yaw())
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        position = Position(
            x=entity.pose.position.x + cos_yaw * sound.offset.x - sin_yaw * sound.offset.y,
            y=entity.pose.position.y + sin_yaw * sound.offset.x + cos_yaw * sound.offset.y,
            z=entity.pose.position.z + sound.offset.z,
        )
        return position, yaw, level_id

    if sound.position is None:
        raise RuntimeError("validated sound has no position")
    if context_level_id is not None:
        level_id = context_level_id
    else:
        level_id = sound.level
        if not level_id:
            if len(world.levels) != 1:
                raise ValueError(f"sound {sound.name!r} requires level in a multi-level world")
            level_id = next(iter(world.levels))
    if level_id not in world.levels:
        raise ValueError(f"sound {sound.name!r} references unknown level {level_id!r}")
    return sound.position + sound.offset, 0.0, level_id


def _merge_params(cfgs: Sequence[SemanticCfg]) -> dict:
    params: dict = {}
    for cfg in cfgs:
        params.update(cfg.params)
    return params


def _sound_group_id(cfgs: Sequence[SemanticCfg], realized_name: str) -> str:
    """Wire group_id: the sound_on regime name when set (groups sirens sharing it), else the entity name."""
    sound_on = _merge_params(cfgs).get("sound_on")
    return str(sound_on) if sound_on else realized_name


@attrs.frozen
class _CatalogEntry:
    sound_type: str
    tags: tuple[str, ...]
    reference_level_db: float


def _realize_frame(env_frame: str, frame: str) -> str:
    """Env-prefixed TF frame, left alone when the author already wrote the prefix."""
    env_frame = env_frame.strip("/")
    if not env_frame or frame == env_frame or frame.startswith(f"{env_frame}/"):
        return frame
    return f"{env_frame}/{frame}"


def _parse_catalog(raw: dict) -> dict[str, _CatalogEntry]:
    """asset_id -> catalog entry from a parsed acoustic_assets.yaml document."""
    return {
        str(asset_id): _CatalogEntry(
            sound_type=str(entry.get("category", asset_id)),
            tags=tuple(str(tag) for tag in entry.get("semantic_tags", [])),
            reference_level_db=float(entry["reference_level_db"]),
        )
        for asset_id, entry in raw.get("assets", {}).items()
    }


def _catalog_lookup(catalog: dict[str, _CatalogEntry], asset_id: str) -> tuple[str, tuple[str, ...], bool]:
    """(sound_type, tags, found) for asset_id, unknown assets fall back to (asset_id, (), False)."""
    entry = catalog.get(asset_id)
    if entry is None:
        return asset_id, (), False
    return entry.sound_type, entry.tags, True


def _catalog_default(catalog: dict[str, _CatalogEntry], sound_type: str) -> tuple[str, float] | None:
    """(asset_id, reference_level_db) of the first catalog asset of this sound_type."""
    for asset_id, entry in catalog.items():
        if entry.sound_type == sound_type:
            return asset_id, entry.reference_level_db
    return None


@attrs.frozen
class _ResolvedSound:
    """Renderer-resolved wire metadata for one sound entity, keyed by its realized name."""

    name: str
    group_id: str
    asset_id: str
    sound_type: str
    tags: tuple[str, ...]
    loop: bool
    reference_distance_m: float
    frame_id: str
    position: Point
    yaw: float


@attrs.define
class _SoundState:
    """Per-entity edge-detection state for the publish timer."""

    last_sounding: bool = False
    program_start_time: Time = attrs.field(factory=Time)


class Mod_Sounds(TM_Module):
    """Render engine-owned sound state (world, launch-configured and runtime-spawned) onto the wire.

    All state lives on the node's event loop: service and timer callbacks marshal via wait_for."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._sounds: dict[str, _ResolvedSound] = {}
        self._sound_state: dict[str, _SoundState] = {}
        self._attached: set[str] = set()
        self._runtime: set[str] = set()

        catalog_path = Path(get_package_share_directory("task_generator")) / "config" / "auditory" / "acoustic_assets.yaml"
        self._catalog = _parse_catalog(yaml.safe_load(catalog_path.read_text()))

        self._source_publisher = self.node.create_publisher(
            ContinuousAudioSourceState,
            self.node.service_namespace("continuous_audio_sources"),
            continuous_audio_qos(),
        )
        self._spawn_service = self.node.create_service(
            SpawnSound,
            self.node.service_namespace("runtime", "spawn_sound"),
            self._spawn_sound,
        )
        self._remove_service = self.node.create_service(
            RemoveSound,
            self.node.service_namespace("runtime", "remove_sound"),
            self._remove_sound,
        )
        self._timer = self.node.create_timer(
            0.1,
            self._publish_sources,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def after_reset(self) -> None:
        for entity in self._attached:
            self.node._simulator.detach_semantics(entity)
        self._sound_state.clear()
        self._runtime.clear()

        world_name = self._ctx.world_manager.loaded_world
        world_view = WorldIdentifier(world_name).resolve_sync()
        world = world_view.load()
        scenario = self._active_scenario()
        indexed_entities = _index_static_entities(world, scenario.static if scenario is not None else ())

        resolved: dict[str, _ResolvedSound] = {}
        for level_id, level in world.levels.items():
            for snd in level.all_sounds:
                realized_name = self.node._realizer.realize(snd, level_id).name
                resolved[realized_name] = self._resolve_and_build(snd, world, indexed_entities, level_id, realized_name)

        attached: set[str] = set()
        episode_sounds = list(scenario.sounds) if scenario is not None else []
        for snd in [*episode_sounds, *self._configured_sounds()]:
            realized_name = self.node._realizer.realize(snd).name
            resolved[realized_name] = self._resolve_and_build(snd, world, indexed_entities, None, realized_name)
            self.node._simulator.attach_semantics("sound", realized_name, snd.semantics)
            attached.add(realized_name)

        self._sounds = resolved
        self._attached = attached
        self._logger.info(f"loaded {len(resolved)} sound(s), {len(episode_sounds)} scenario, {len(attached) - len(episode_sounds)} launch")

    def _active_scenario(self) -> Scenario | None:
        from task_generator.tasks.obstacles.scenario.impl import TM_Scenario

        tm = self._task.tm_obstacles
        return tm.scenario if isinstance(tm, TM_Scenario) else None

    def _resolve_and_build(
        self,
        snd: Sound,
        world: WorldDescription,
        indexed_entities: dict[str, list[tuple[str, Obstacle]]],
        context_level_id: str | None,
        realized_name: str,
    ) -> _ResolvedSound:
        if snd.frame:
            frame_id = _realize_frame(self.node._realizer.realize(), snd.frame)
            return self._build_resolved(snd, realized_name, snd.offset.to_msg(), 0.0, frame_id)
        position, yaw, level_id = _resolve_sound_placement(snd, world, indexed_entities, context_level_id)
        map_position = self.node._realizer.realize(position, level_id)
        return self._build_resolved(snd, realized_name, map_position.to_msg(), yaw, "map")

    def _build_resolved(self, snd: Sound, realized_name: str, position: Point, yaw: float, frame_id: str) -> _ResolvedSound:
        sound_type, catalog_tags, found = _catalog_lookup(self._catalog, snd.asset_id)
        if not found:
            self._logger.warning(f"sound {snd.name!r}: asset_id {snd.asset_id!r} is not in the acoustic catalog")
        return _ResolvedSound(
            name=snd.name,
            group_id=_sound_group_id(snd.semantics, realized_name),
            asset_id=snd.asset_id,
            sound_type=sound_type,
            tags=("environment", "static", sound_type, *catalog_tags),
            loop=snd.loop,
            reference_distance_m=snd.reference_distance_m,
            frame_id=frame_id,
            position=position,
            yaw=yaw,
        )

    def _configured_sounds(self) -> list[Sound]:
        if not self.node.has_parameter("static_sounds"):
            return []
        raw = str(self.node.get_parameter("static_sounds").value).strip()
        parsed = yaml.safe_load(raw) if raw else []
        if parsed is None:
            parsed = []
        if not isinstance(parsed, list):
            raise ValueError("static_sounds must be a YAML list of sound entries")
        return converter.structure(parsed, list[Sound])

    def _spawn_sound(
        self,
        request: SpawnSound.Request,
        response: SpawnSound.Response,
    ) -> SpawnSound.Response:
        try:
            return self._spawn_sound_impl(request, response)
        except Exception as exc:
            self._logger.error(f"spawning runtime sound failed:\n{traceback.format_exc()}")
            response.error_msg = f"{type(exc).__name__}: {exc}"
            return response

    def _spawn_sound_impl(
        self,
        request: SpawnSound.Request,
        response: SpawnSound.Response,
    ) -> SpawnSound.Response:
        mode = str(request.mode).strip().lower()
        default = _catalog_default(self._catalog, mode) if mode in ("music", "alarm") else None
        if default is None:
            response.error_msg = "mode must be 'music' or 'alarm'"
            return response
        default_asset_id, default_volume_db = default
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
            response.error_msg = "sound pose requires a frame ID"
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
            response.error_msg = "sound pose must be finite"
            return response
        orientation_norm = math.sqrt(orientation.x**2 + orientation.y**2 + orientation.z**2 + orientation.w**2)
        if orientation_norm <= 1e-9:
            response.error_msg = "sound orientation must be valid"
            return response

        attach_to_frame = bool(request.attach_to_frame)
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
        if frame_id != "map" and not attach_to_frame:
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
                response.error_msg = f"cannot transform sound from {frame_id!r} to 'map': {exc}"
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

        if map_position.z < 0.0 and not attach_to_frame:
            response.error_msg = "sound height cannot be below the floor"
            return response

        qx, qy, qz, qw = map_orientation
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy**2 + qz**2),
        )

        async def _spawn() -> str | None:
            index = 1
            entity_name = f"runtime_{mode}_{index}"
            realized_name = self.node._realizer.prefix(entity_name)
            while realized_name in self._attached:
                index += 1
                entity_name = f"runtime_{mode}_{index}"
                realized_name = self.node._realizer.prefix(entity_name)
            local = Position(float(map_position.x), float(map_position.y), float(map_position.z))
            sound = Sound(
                name=entity_name,
                asset_id=asset_id,
                frame=frame_id if attach_to_frame else "",
                offset=local if attach_to_frame else Position(0.0, 0.0, 0.0),
                position=None if attach_to_frame else local,
                loop=loop,
                semantics=[
                    SemanticCfg(role="predicate", name="sounding"),
                    SemanticCfg(role="state", name="volume_db", value=source_volume_db),
                ],
            )
            if not self.node._simulator.attach_semantics("sound", realized_name, sound.semantics):
                return None
            if initially_active:
                self.node._simulator.set_semantic_value(realized_name, "sounding", "true")
            self._attached.add(realized_name)
            self._runtime.add(realized_name)
            wire_frame = _realize_frame(self.node._realizer.realize(), frame_id) if attach_to_frame else "map"
            self._sounds[realized_name] = self._build_resolved(sound, realized_name, map_position, float(yaw), wire_frame)
            return realized_name

        realized_name = self.node.wait_for(_spawn())
        if realized_name is None:
            response.error_msg = "semantics engine refused the sound"
            return response

        response.entity = realized_name
        response.success = True
        self._logger.info(f"spawned {mode} source {realized_name!r} at ({map_position.x:.2f}, {map_position.y:.2f}, {map_position.z:.2f})")
        return response

    def _remove_sound(
        self,
        request: RemoveSound.Request,
        response: RemoveSound.Response,
    ) -> RemoveSound.Response:
        entity_name = str(request.entity).strip()

        async def _remove() -> bool:
            if entity_name not in self._runtime:
                return False
            self._sounds.pop(entity_name, None)
            self._sound_state.pop(entity_name, None)
            self._attached.discard(entity_name)
            self._runtime.discard(entity_name)
            self.node._simulator.detach_semantics(entity_name)
            return True

        if not self.node.wait_for(_remove()):
            response.error_msg = f"unknown or non-removable sound {entity_name!r}"
            return response
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
            self._logger.error(f"publishing static audio state failed:\n{traceback.format_exc()}")

    def _publish_sources_impl(self) -> None:
        for msg in self.node.wait_for(self._source_msgs()):
            self._source_publisher.publish(msg)

    async def _source_msgs(self) -> list[ContinuousAudioSourceState]:
        stamp = self.node.get_clock().now().to_msg()
        msgs: list[ContinuousAudioSourceState] = []
        for snap in self.node._simulator.semantics_snapshot():
            if snap.kind != "sound":
                continue
            resolved = self._sounds.get(snap.entity)
            if resolved is None:
                continue
            sounding = bool(snap.predicates.get("sounding", False))
            volume_db = float(snap.continuous.get("volume_db", 0.0))
            state = self._sound_state.setdefault(snap.entity, _SoundState())
            if sounding and not state.last_sounding:
                state.program_start_time = self.node.get_clock().now().to_msg()
            state.last_sounding = sounding

            msg = ContinuousAudioSourceState()
            msg.header.stamp = stamp
            msg.header.frame_id = resolved.frame_id
            msg.source_id = f"environment:{snap.entity}"
            msg.source_agent_id = self._numeric_id(snap.entity)
            msg.source_agent_name = resolved.name
            msg.source_model = "static_audio_source"
            msg.sound_type = resolved.sound_type
            msg.source_backend = "wav_loop"
            msg.group_id = resolved.group_id
            msg.asset_id = resolved.asset_id
            msg.label = resolved.name
            msg.loop = resolved.loop
            msg.program_start_time = state.program_start_time
            msg.source_position = resolved.position
            msg.source_yaw = resolved.yaw
            msg.source_volume_db = volume_db
            msg.reference_distance_m = resolved.reference_distance_m
            msg.active = sounding
            msg.deterministic_seed = self._deterministic_seed(snap.entity)
            msg.semantic_tags = list(resolved.tags)
            msgs.append(msg)
        return msgs

    @staticmethod
    def _numeric_id(source_id: str) -> int:
        digest = hashlib.blake2b(source_id.encode(), digest_size=4).digest()
        return int.from_bytes(digest, "little") & 0x7FFFFFFF

    @staticmethod
    def _deterministic_seed(source_id: str) -> int:
        digest = hashlib.blake2b(source_id.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little")
