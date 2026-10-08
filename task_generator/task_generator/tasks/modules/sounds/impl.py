from __future__ import annotations

import math
import traceback
import typing
from collections.abc import Sequence

import attrs
import rclpy
import tf2_ros
import yaml
from arena_simulation_setup.shared import Obstacle, Position, SemanticCfg, Sound
from arena_simulation_setup.tree.assets.sound_catalog import AgentKind, SoundLibrary, selection_seed
from arena_simulation_setup.tree.World import WorldDescription, WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import Scenario
from arena_simulation_setup.utils.cattrs import converter
from geometry_msgs.msg import Point, Vector3
from geometry_msgs.msg import Pose as PoseMsg
from rclpy.clock import Clock, ClockType
from task_generator_msgs.srv import RemoveSound, SpawnSound
from visualization_msgs.msg import Marker

from task_generator.interactive import Apply, Menu, planar_marker, visual
from task_generator.shared import Orientation
from task_generator.simulators.acoustics import SoundEmission
from task_generator.tasks.modules import TM_Module
from task_generator.tasks.modules.sounds import REMOVE_SOUND, SPAWN_SOUND

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.assets.sound_catalog import SoundAsset


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


def _has_initial_sounding(cfgs: Sequence[SemanticCfg]) -> bool:
    """True when a sound_on regime or an explicit sounding value decides whether the sound plays."""
    if _merge_params(cfgs).get("sound_on"):
        return True
    return any(cfg.name == "sounding" and cfg.value is not None for cfg in cfgs)


def _sounding_by_default(snd: Sound) -> Sound:
    """Launch-defined sounds play from the start unless the entry decides otherwise."""
    if _has_initial_sounding(snd.semantics):
        return snd
    sounding = next((cfg for cfg in snd.semantics if cfg.name == "sounding"), None)
    if sounding is None:
        sounding = SemanticCfg(role="predicate", name="sounding")
        snd.semantics.append(sounding)
    sounding.value = True
    return snd


def _sound_group_id(cfgs: Sequence[SemanticCfg], realized_name: str) -> str:
    """Wire group_id: the sound_on regime name when set (groups sirens sharing it), else the entity name."""
    sound_on = _merge_params(cfgs).get("sound_on")
    return str(sound_on) if sound_on else realized_name


def _realize_frame(env_frame: str, frame: str) -> str:
    """Env-prefixed TF frame, left alone when the author already wrote the prefix."""
    env_frame = env_frame.strip("/")
    if not env_frame or frame == env_frame or frame.startswith(f"{env_frame}/"):
        return frame
    return f"{env_frame}/{frame}"


@attrs.frozen
class _ResolvedSound:
    """Wire metadata for one sound entity, keyed by its realized name, asset and variant resolved."""

    name: str
    group_id: str
    asset_id: str
    kind: str
    variant_id: str
    model: str
    tags: tuple[str, ...]
    loop: bool
    reference_distance_m: float
    level_db: float
    seed: int
    frame_id: str
    position: Point
    yaw: float


@attrs.define
class _SoundState:
    """Per-entity edge-detection state for the publish timer."""

    last_sounding: bool = False
    program_start_ns: int = 0
    volume_db: float = 0.0


class Mod_Sounds(TM_Module):
    """Hand engine-owned sound state (world, launch-configured and runtime-spawned) to the acoustics simulator.

    All state lives on the node's event loop: service and timer callbacks run there via ros_callback."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._sounds: dict[str, _ResolvedSound] = {}
        self._sound_state: dict[str, _SoundState] = {}
        self._attached: set[str] = set()
        self._runtime: set[str] = set()
        self._warned_inert: set[str] = set()
        self._library = SoundLibrary.default()

        self._spawn_service = self.node.create_service(
            SpawnSound,
            self.node.service_namespace(SPAWN_SOUND),
            self._spawn_sound,
        )
        self._remove_service = self.node.create_service(
            RemoveSound,
            self.node.service_namespace(REMOVE_SOUND),
            self._remove_sound,
        )
        self._timer = self.node.create_timer(
            0.1,
            self.node.ros_callback(self._publish_sources),
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def after_reset(self) -> None:
        for entity in self._attached:
            self.node._simulator.detach_semantics(entity)
        self._sound_state.clear()
        self._runtime.clear()
        self.node._acoustics_simulator.clear_sounds()

        world_name = self._ctx.world_manager.loaded_world
        world_view = WorldIdentifier(world_name).resolve_sync()
        self._library.use_world(world_view.path)
        world = world_view.load()
        scenario = self._active_scenario()
        indexed_entities = _index_static_entities(world, scenario.static if scenario is not None else ())

        resolved: dict[str, _ResolvedSound] = {}
        for level_id, level in world.levels.items():
            for snd in level.all_sounds:
                realized_name = self.node._realizer.realize(snd, level_id).name
                self._warn_if_inert(snd, realized_name)
                built = self._resolve_and_build(snd, world, indexed_entities, level_id, realized_name)
                if built is not None:
                    resolved[realized_name] = built

        attached: set[str] = set()
        episode_sounds = list(scenario.sounds) if scenario is not None else []
        for snd in [*episode_sounds, *self._configured_sounds()]:
            realized_name = self.node._realizer.realize(snd).name
            self._warn_if_inert(snd, realized_name)
            built = self._resolve_and_build(snd, world, indexed_entities, None, realized_name)
            if built is not None:
                resolved[realized_name] = built
            self.node._simulator.attach_semantics("sound", realized_name, snd.semantics)
            attached.add(realized_name)

        self._sounds = resolved
        self._attached = attached
        self.node.register_sound_levels({name: sound.level_db for name, sound in resolved.items()})
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
    ) -> _ResolvedSound | None:
        """None when the sound's asset does not resolve, logged as an error."""
        try:
            asset = self._library.asset(snd.asset_id)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            self._logger.error(f"sound {snd.name!r} stays silent: {exc}")
            return None
        if snd.frame:
            frame_id = _realize_frame(self.node._realizer.realize(), snd.frame)
            return self._build_resolved(snd, asset, realized_name, snd.offset.to_msg(), 0.0, frame_id)
        position, yaw, level_id = _resolve_sound_placement(snd, world, indexed_entities, context_level_id)
        map_position = self.node._realizer.realize(position, level_id)
        return self._build_resolved(snd, asset, realized_name, map_position.to_msg(), yaw, "map")

    def _build_resolved(self, snd: Sound, asset: SoundAsset, realized_name: str, position: Point, yaw: float, frame_id: str) -> _ResolvedSound:
        group_id = _sound_group_id(snd.semantics, realized_name)
        seed = selection_seed(group_id)
        variant = asset.select(context={}, seed=seed)
        return _ResolvedSound(
            name=snd.name,
            group_id=group_id,
            asset_id=asset.id,
            kind=asset.kind,
            variant_id=variant.id,
            model=variant.model,
            tags=variant.tags,
            loop=snd.loop,
            reference_distance_m=snd.reference_distance_m,
            level_db=asset.level_db,
            seed=seed,
            frame_id=frame_id,
            position=position,
            yaw=yaw,
        )

    def _configured_sounds(self) -> list[Sound]:
        if not self.node.has_parameter("auditory.static_sounds"):
            return []
        raw = str(self.node.get_parameter("auditory.static_sounds").value).strip()
        parsed = yaml.safe_load(raw) if raw else []
        if parsed is None:
            parsed = []
        if not isinstance(parsed, list):
            raise ValueError("auditory.static_sounds must be a YAML list of sound entries")
        return [_sounding_by_default(snd) for snd in converter.structure(parsed, list[Sound])]

    def _warn_if_inert(self, snd: Sound, realized_name: str) -> None:
        if _has_initial_sounding(snd.semantics) or realized_name in self._warned_inert:
            return
        self._warned_inert.add(realized_name)
        self._logger.warning(f"sound {snd.name!r} has neither sound_on nor a sounding value, so it stays silent until toggled: ros2 service call {self.node.service_namespace('semantics', 'set')} task_generator_msgs/srv/SetSemantic \"{{entity: {realized_name}, field: sounding, value: 'true'}}\"")

    async def _spawn_sound(
        self,
        request: SpawnSound.Request,
        response: SpawnSound.Response,
    ) -> SpawnSound.Response:
        try:
            async with self.node._reset_lock:
                return await self._spawn_sound_impl(request, response)
        except ValueError as exc:
            self._logger.error(f"spawning runtime sound failed:\n{traceback.format_exc()}")
            response.error_msg = f"{type(exc).__name__}: {exc}"
            return response

    async def _spawn_sound_impl(
        self,
        request: SpawnSound.Request,
        response: SpawnSound.Response,
    ) -> SpawnSound.Response:
        kind = str(request.kind).strip().lower()
        accepted = self._library.kinds_of(AgentKind.ENVIRONMENT)
        if kind not in accepted:
            response.error_msg = f"unknown environment sound kind {kind!r}, expected one of {sorted(accepted)}"
            return response
        try:
            asset = self._library.default_asset(kind)
            if bool(request.customize_playback) and str(request.asset_id).strip():
                asset = self._library.asset(str(request.asset_id))
        except (KeyError, ValueError, FileNotFoundError) as exc:
            response.error_msg = str(exc)
            return response
        if asset.kind != kind:
            response.error_msg = f"sound asset {asset.id!r} is of kind {asset.kind!r}, not {kind!r}"
            return response
        if bool(request.customize_playback):
            source_volume_db = float(request.source_volume_db)
            if not math.isfinite(source_volume_db):
                response.error_msg = "source volume must be finite"
                return response
            loop = bool(request.loop)
            initially_active = bool(request.initially_active)
        else:
            source_volume_db = asset.level_db
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
            entity_name = f"runtime_{kind}_{index}"
            realized_name = self.node._realizer.prefix(entity_name)
            while realized_name in self._attached:
                index += 1
                entity_name = f"runtime_{kind}_{index}"
                realized_name = self.node._realizer.prefix(entity_name)
            local = Position(float(map_position.x), float(map_position.y), float(map_position.z))
            sound = Sound(
                name=entity_name,
                asset_id=asset.id,
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
            self._sounds[realized_name] = self._build_resolved(sound, asset, realized_name, map_position, float(yaw), wire_frame)
            if not attach_to_frame:
                self._put_handle(realized_name, map_position, float(yaw))
            return realized_name

        realized_name = await _spawn()
        if realized_name is None:
            response.error_msg = "semantics engine refused the sound"
            return response

        response.entity = realized_name
        response.success = True
        self._logger.info(f"spawned {kind} source {realized_name!r} at ({map_position.x:.2f}, {map_position.y:.2f}, {map_position.z:.2f})")
        return response

    def _put_handle(self, entity: str, position: Point, yaw: float) -> None:
        pose = PoseMsg(position=position, orientation=Orientation.from_yaw(yaw).to_msg())

        async def on_pose(moved: PoseMsg) -> None:
            self._move_sound(entity, moved)

        async def remove() -> None:
            self._remove_runtime(entity)

        self.node.markers.put(
            planar_marker(entity, pose, description=entity, scale=0.6, visuals=[visual(Marker.CUBE, (0.25, 0.25, 0.25), (0.9, 0.8, 0.2, 0.9))]),
            on_pose=on_pose,
            apply=Apply.LIVE,
            menu=[Menu("Remove", remove)],
        )

    def _move_sound(self, entity: str, pose: PoseMsg) -> None:
        resolved = self._sounds[entity]
        self._sounds[entity] = attrs.evolve(resolved, position=pose.position, yaw=Orientation.from_msg(pose.orientation).to_yaw())

    def _remove_runtime(self, entity: str) -> bool:
        if entity not in self._runtime:
            return False
        self._sounds.pop(entity, None)
        self._sound_state.pop(entity, None)
        self._attached.discard(entity)
        self._runtime.discard(entity)
        self.node.markers.erase(entity)
        self.node._simulator.detach_semantics(entity)
        return True

    async def _remove_sound(
        self,
        request: RemoveSound.Request,
        response: RemoveSound.Response,
    ) -> RemoveSound.Response:
        entity_name = str(request.entity).strip()
        async with self.node._reset_lock:
            removed = self._remove_runtime(entity_name)
        if not removed:
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

    async def _publish_sources(self) -> None:
        try:
            self.node._acoustics_simulator.emit_sounds(await self._emissions())
        except (TypeError, ValueError):
            self._logger.error(f"publishing static audio state failed:\n{traceback.format_exc()}")

    async def _emissions(self) -> list[SoundEmission]:
        emissions: list[SoundEmission] = []
        for snap in self.node._simulator.semantics_snapshot():
            if snap.kind != "sound":
                continue
            resolved = self._sounds.get(snap.entity)
            if resolved is None:
                continue
            sounding = bool(snap.predicates.get("sounding", False))
            state = self._sound_state.setdefault(snap.entity, _SoundState())
            if sounding and not state.last_sounding:
                state.program_start_ns = self.node.get_clock().now().nanoseconds
            state.last_sounding = sounding
            state.volume_db = float(snap.continuous.get("volume_db", resolved.level_db))
            emissions.append(
                SoundEmission(
                    entity=snap.entity,
                    name=resolved.name,
                    group_id=resolved.group_id,
                    kind=resolved.kind,
                    asset_id=resolved.asset_id,
                    variant_id=resolved.variant_id,
                    model=resolved.model,
                    tags=resolved.tags,
                    loop=resolved.loop,
                    reference_distance_m=resolved.reference_distance_m,
                    level_db=state.volume_db,
                    seed=resolved.seed,
                    frame_id=resolved.frame_id,
                    position=(resolved.position.x, resolved.position.y, resolved.position.z),
                    yaw=resolved.yaw,
                    active=sounding,
                    program_start_ns=state.program_start_ns,
                )
            )
        return emissions
