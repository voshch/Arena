import asyncio
import json
import typing
from collections.abc import Callable, Collection, Sequence
from typing import Any

import attrs
import rclpy.qos
import shapely
from arena_runtime._node import NodeInterface
from arena_runtime.sim import BaseSim
from arena_runtime.sim._semantics import _SEMANTIC_KINDS
from arena_simulation_setup.shared import Ceiling, Light
from arena_simulation_setup.tree.World import LevelDescription, WorldDescription
from arena_simulation_setup.tree.World.World import _render_door_polygons, _render_elevator_polygons, entity_lights
from rcl_interfaces.msg import Parameter as ParameterMsg
from std_msgs.msg import String

from task_generator.manager.collision_grid import CollisionGrid
from task_generator.manager.realizer import Realizer
from task_generator.manager.world_manager.utils import WorldMap
from task_generator.manager.world_manager.world_manager import WORLD_ENTITY_PREFIX
from task_generator.shared import (
    Door,
    DynamicObstacle,
    Elevator,
    Obstacle,
    Pose,
    Region,
    Robot,
    SemanticCfg,
    Wall,
)
from task_generator.simulators.human import BaseHumanSimulator
from task_generator.simulators.human.utils import ObstacleLayer
from task_generator.utils.flags import ObstaclesOptim, obstacles_optim_level


def _kind_vocab(kind_cls: type) -> frozenset[str]:
    """All state and predicate field names a runtime semantic kind exposes."""
    return frozenset((*kind_cls.DISCRETE, *kind_cls.CONTINUOUS, *kind_cls.PREDICATES))


_ATTACHABLE_TO_HOST = frozenset({'gate', 'pressure_plate'})


def _route_semantics(cfgs: Sequence[SemanticCfg], host_kind: str) -> dict[str, list[SemanticCfg]]:
    """Route a mechanism entry's cfgs to their owning scripted kind, raising on
    host-vocabulary or unattachable names (mechanism state publishes intrinsically)."""
    host_vocab = _kind_vocab(_SEMANTIC_KINDS[host_kind])
    extras: dict[str, list[SemanticCfg]] = {}
    for cfg in cfgs:
        if cfg.name in host_vocab:
            raise ValueError(f"semantics: every {host_kind} publishes {cfg.name!r} intrinsically, remove the annotation")
        for kname in _ATTACHABLE_TO_HOST:
            if cfg.name in _kind_vocab(_SEMANTIC_KINDS[kname]):
                extras.setdefault(kname, []).append(cfg)
                break
        else:
            raise ValueError(f"semantics: {cfg.name!r} is not attachable to a {host_kind} host")
    return extras


_OCCUPANCY_CAP_VOCAB = _kind_vocab(_SEMANTIC_KINDS["occupancy_cap"])


def authored_statics(level: LevelDescription) -> LevelDescription:
    """The level with its static entities under their authored names, as lights and their entity_ref name them."""
    return attrs.evolve(
        level,
        zones=[attrs.evolve(zone, entities=attrs.evolve(zone.entities, static=[attrs.evolve(entity, name=entity.name.removeprefix(WORLD_ENTITY_PREFIX)) for entity in zone.entities.static])) for zone in level.zones],
    )


async def world_lights(level: LevelDescription) -> list[Light]:
    """The lights of a level under authored names, each owner under the name its static entity spawns with."""
    return [attrs.evolve(light, owner=f'{WORLD_ENTITY_PREFIX}{light.owner}') if light.owner else light for light in await authored_statics(level).all_lights()]


STAGE_AMBIENT_TOPIC = '/arena/stage_ambient'
STAGE_AMBIENT_ENV = 'env_0'


def stage_ambient(lights: Sequence[Light]) -> list[dict]:
    """The dome and sun lights of a world as comparable settings, sorted, names left out."""
    return sorted(
        ({'fixture': light.fixture, 'lux': light.lux, 'cct_K': light.cct_K, 'direction': list(light.direction), 'level': light.level, 'lit': light.lit, 'light_on': light.light_on} for light in lights if light.spec.ambient),
        key=json.dumps,
    )


def stage_ambient_conflict(env: str, mine: list[dict], stage: list[dict]) -> str | None:
    """Why an env renders under other dome and sun lights than its world declares, None when they agree."""
    if mine == stage:
        return None

    def described(ambient: list[dict]) -> str:
        return ', '.join(f"{a['fixture']} {a['lux']:g} lux" for a in ambient) or 'no dome or sun (default lighting)'

    return f'dome and sun light the whole stage and {STAGE_AMBIENT_ENV} sets them: {env} renders under {described(stage)}, its world declares {described(mine)}'


class EnvironmentManager(NodeInterface):
    _human_simulator: BaseHumanSimulator
    _simulator: BaseSim
    _realizer: Realizer

    _collision_grid: CollisionGrid | None
    _static_polygons: dict[str, shapely.Polygon]

    def __init__(
        self,
        *args: object,
        simulator: BaseSim,
        human_simulator: BaseHumanSimulator,
        realizer: Realizer,
        **kwargs: object,
    ):
        super().__init__(*args, **kwargs)

        self._simulator = simulator
        self._human_simulator = human_simulator
        self._realizer = realizer

        self._collision_grid = None
        self._static_polygons = {}
        self._attached_semantic_entities: set[str] = set()
        self._world_light_names: set[str] = set()
        self._world_level_ids: list[str] = []
        self._episode_lights: dict[str, list[str]] = {}
        self._episode_light_owners: set[str] | None = None
        self._stage_ambient: list[dict] | None = None
        self._own_ambient: list[dict] | None = None
        self._stage_ambient_warned: tuple[str, str] | None = None
        qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        if self._env_name == STAGE_AMBIENT_ENV:
            self._stage_ambient_pub = self.node.create_publisher(String, STAGE_AMBIENT_TOPIC, qos)
        else:
            self._stage_ambient_sub = self.node.create_subscription(String, STAGE_AMBIENT_TOPIC, self._on_stage_ambient, qos)

    @property
    def _env_name(self) -> str:
        return str(self._realizer.prefix()).strip('/')

    def _on_stage_ambient(self, msg: String) -> None:
        self._stage_ambient = json.loads(msg.data)
        self._check_stage_ambient()

    def _declare_ambient(self, lights: Sequence[Light]) -> None:
        self._own_ambient = stage_ambient(lights)
        if self._env_name == STAGE_AMBIENT_ENV:
            self._stage_ambient_pub.publish(String(data=json.dumps(self._own_ambient)))
            return
        self._check_stage_ambient()

    def _check_stage_ambient(self) -> None:
        if self._own_ambient is None or self._stage_ambient is None:
            return
        conflict = stage_ambient_conflict(self._env_name, self._own_ambient, self._stage_ambient)
        key = (json.dumps(self._own_ambient), json.dumps(self._stage_ambient))
        if conflict is not None and key != self._stage_ambient_warned:
            self._logger.warning(conflict)
            self._stage_ambient_warned = key

    def _detach_extra_semantics(self) -> None:
        """Detach the non-door/non-elevator semantics attached for the previous world."""
        for entity in self._attached_semantic_entities:
            self._simulator.detach_semantics(entity)
        self._attached_semantic_entities.clear()

    @property
    def _skip_obstacles(self) -> bool:
        """optim.obstacles=none: silently skip all static obstacle spawns."""
        return obstacles_optim_level(self.node) >= ObstaclesOptim.NONE

    def realize(self, target: object) -> object:
        return self._realizer.realize(target)

    def ezilear(self, target: Pose) -> Pose:
        return self._realizer.ezilear(target)

    @property
    def collision_grid(self) -> CollisionGrid | None:
        """Map-frame labelled occupancy of the loaded world plus every spawned static, None until a world is loaded."""
        return self._collision_grid

    @property
    def static_polygons(self) -> dict[str, shapely.Polygon]:
        """Map-frame footprint polygons of every static obstacle currently spawned,
        keyed by obstacle name. Includes both WORLD-layer entities and INUSE
        episode-spawned obstacles. Pedestrians are not included, consumers should
        read those from the `arena_peds` topic."""
        return self._static_polygons

    async def _cache_polygons(self, obstacles: Sequence[Obstacle]) -> None:
        polys = await asyncio.gather(*(o.footprint() for o in obstacles))
        for obstacle, poly in zip(obstacles, polys, strict=True):
            if poly is None:
                continue
            self._static_polygons[obstacle.name] = poly
            if self._collision_grid is not None:
                self._collision_grid.stamp(obstacle.name, poly)

    def _sync_static_polygons(self) -> None:
        """Drop polygons whose obstacle is no longer registered in the human simulator."""
        alive = set(self._human_simulator._known_obstacles.keys())
        self._static_polygons = {n: p for n, p in self._static_polygons.items() if n in alive}
        if self._collision_grid is not None:
            self._collision_grid.sync(self._static_polygons.keys())

    async def spawn_world_obstacles(
        self,
        world: WorldDescription | LevelDescription,
        level_id: str = "",
        detected_walls: dict[str, Sequence[Wall]] | None = None,
        world_map: WorldMap | None = None,
    ):
        """
        Loads given obstacles into the simulator,
        the map file is retrieved from launch parameter "world"

        Args:
            detected_walls: per-level occupancy-derived walls fed to the human-sim as collision geometry (populated only under debug.map_source:=disk).
            world_map: the loaded occupancy map, seeds `collision_grid`.
        """
        await self._spawn_world_obstacles(world, level_id, detected_walls, world_map)

    async def _spawn_world_obstacles(
        self,
        world: WorldDescription | LevelDescription,
        level_id: str = "",
        detected_walls: dict[str, Sequence[Wall]] | None = None,
        world_map: WorldMap | None = None,
    ) -> None:

        def _match_level_id(fid: str | None) -> bool:
            target_id = level_id
            if target_id == "":
                return True
            else:
                if fid is not None:
                    return target_id == fid
                else:
                    return False

        _world = WorldDescription.from_levels(world) if isinstance(world, LevelDescription) else world

        self._detach_extra_semantics()
        await self._simulator.remove_lights()
        self._episode_lights.clear()
        self._world_light_names.clear()
        self._world_level_ids = [str(fid) for fid in _world.levels if _match_level_id(fid)]
        self.node._clear_semantic_entities()
        # (kind, realized entity, cfgs, polygon) attaches deferred until geometry exists.
        pending: list[tuple[str, str, list[SemanticCfg], list[tuple[float, float]] | None]] = []

        walls_list: list[Wall] = []
        collision_walls: list[Wall] = []
        lintels: list[Wall] = []
        for fid, level in _world.levels.items():
            if not _match_level_id(fid):
                continue
            walls_list.extend(self._realizer.realize(w, fid) for w in await level.closed_walls())
            lintels.extend(self._realizer.realize(w, fid) for w in await level.door_lintels())
            if detected_walls and detected_walls.get(fid):
                collision_walls.extend(self._realizer.realize(w, fid) for w in detected_walls[fid])
        walls = tuple(walls_list)
        doors_list: list[Door] = []
        for fid, level in _world.levels.items():
            if not _match_level_id(fid):
                continue
            for d in level.all_doors:
                realized = self._realizer.realize(d, fid)
                self.node._register_semantic_entity(d.name, realized.name)
                extras = _route_semantics(realized.semantics, "door")
                doors_list.append(realized)
                pending.extend((kind, realized.name, cfgs, None) for kind, cfgs in extras.items())
        doors = tuple(doors_list)
        floors = tuple(self._realizer.realize(f, fid) for fid, level in _world.levels.items() if _match_level_id(fid) for f in level.all_floors)
        ceilings: list[Ceiling] = []
        for fid, level in _world.levels.items():
            if not _match_level_id(fid):
                continue
            for ceiling in await level.all_ceilings():
                ceilings.append(self._realizer.realize(ceiling, fid))
        lights: list[Light] = []
        for fid, level in _world.levels.items():
            if not _match_level_id(fid):
                continue
            for light in await world_lights(level.with_lighting(self.node.conf.Arena.WORLD_LIGHTING.value or 'authored')):
                realized_l = self._realizer.realize(light, fid)
                self.node._register_semantic_entity(light.name, realized_l.name)
                self._world_light_names.add(light.name)
                lights.append(realized_l)
                pending.append(("light", realized_l.name, realized_l.semantics, None))
        elevators_list: list[Elevator] = []
        for fid, level in _world.levels.items():
            if not _match_level_id(fid):
                continue
            for e in level.all_elevators:
                realized_e = self._realizer.realize(e, fid)
                self.node._register_semantic_entity(e.name, realized_e.name)
                extras = _route_semantics(realized_e.semantics, "elevator")
                elevators_list.append(realized_e)
                pending.extend((kind, realized_e.name, cfgs, None) for kind, cfgs in extras.items())
            for sched in level.all_schedules:
                realized_s = self._realizer.realize(sched, fid)
                self.node._register_semantic_entity(sched.name, realized_s.name)
                if realized_s.semantics:
                    pending.append(("schedule", realized_s.name, list(realized_s.semantics), None))
            for sig in level.all_signals:
                realized_sig = self._realizer.realize(sig, fid)
                self.node._register_semantic_entity(sig.name, realized_sig.name)
                if realized_sig.semantics:
                    pending.append(("signal", realized_sig.name, list(realized_sig.semantics), None))
            for snd in level.all_sounds:
                realized_snd = self._realizer.realize(snd, fid)
                self.node._register_semantic_entity(snd.name, realized_snd.name)
                if realized_snd.semantics:
                    pending.append(("sound", realized_snd.name, list(realized_snd.semantics), None))
            for zone in level.zones:
                if not zone.semantics:
                    continue
                zone_name = self._realizer.prefix(zone.name, fid)
                self.node._register_semantic_entity(zone.name, zone_name)
                occ = [cfg for cfg in zone.semantics if cfg.name in _OCCUPANCY_CAP_VOCAB]
                if occ:
                    pending.append(("occupancy_cap", zone_name, occ, self._realizer.realize_polygon(zone.corners, fid)))
        elevators = tuple(elevators_list)
        statics = tuple(self._realizer.realize(s, fid) for fid, level in _world.levels.items() if _match_level_id(fid) for s in level.all_static_entities)
        if self._skip_obstacles:
            statics = ()

        if world_map is not None:
            cutouts = [p for d in doors for p in _render_door_polygons(d)] + [p for e in elevators for p in _render_elevator_polygons(e)]
            self._collision_grid = CollisionGrid.build(world_map, origin=self._realizer.realize(world_map.origin), walls=walls, cutouts=cutouts)

        await self._cache_polygons(statics)

        futures: list[typing.Awaitable] = []
        if floors:
            futures.append(self._simulator.spawn_floors(floors))
        if ceilings:
            futures.append(self._simulator.spawn_ceilings(ceilings))
        if walls or doors or collision_walls:
            futures.append(self._human_simulator.spawn_world(walls, doors, collision_walls=tuple(collision_walls)))
        if lintels:
            futures.append(self._simulator.spawn_walls(lintels, clear_existing=False))
        futures.append(self._human_simulator.spawn_obstacles(statics, layer=ObstacleLayer.WORLD))
        if elevators:
            futures.append(self._simulator.spawn_elevators(elevators))

        await asyncio.gather(*futures)
        if lights:
            await self._simulator.spawn_lights(lights)
        self._declare_ambient(lights)

        # Door/elevator kinds self-attach inside their spawn helpers. The scripted,
        # position, and zone kinds have no geometry spawn, so attach them here once
        # their host geometry exists (gate/plate read the spawned door runtime).
        for kind, entity, cfgs, polygon in pending:
            self._simulator.attach_semantics(kind, entity, cfgs, polygon=polygon)
            self._attached_semantic_entities.add(entity)

    async def spawn_dynamic_obstacles(self, setups: Collection[DynamicObstacle]):
        """
        Loads given dynamic obstacles into the simulator.
        """
        realized = tuple(self._realizer.realize(obstacle, obstacle.level_id or "") for obstacle in setups)
        await self._human_simulator.spawn_dynamic_obstacles(realized)

    async def spawn_obstacles(self, setups: Collection[Obstacle]):
        """
        Loads given obstacles into the simulator.
        """
        if self._skip_obstacles:
            return
        realized = tuple(self._realizer.realize(obstacle, obstacle.level_id or "") for obstacle in setups)
        await self._cache_polygons(realized)
        await self._human_simulator.spawn_obstacles(realized)
        await self._spawn_episode_lights(setups)

    async def _spawn_episode_lights(self, setups: Collection[Obstacle]) -> None:
        """Spawn the lights the objects of episode obstacles carry, replacing those of an obstacle spawned before."""
        await self._remove_episode_lights([obstacle.name for obstacle in setups if obstacle.name in self._episode_lights])
        lights: list[Light] = []
        for obstacle in setups:
            if obstacle.name.startswith(WORLD_ENTITY_PREFIX):
                continue
            if self._episode_light_owners is not None:
                self._episode_light_owners.add(obstacle.name)
            names: list[str] = []
            for light in await entity_lights(obstacle):
                if light.name in self._world_light_names:
                    self._logger.warning(f"light {light.name!r} of obstacle {obstacle.name!r} is also a world light, skipped")
                    continue
                realized = self._realizer.realize(light, obstacle.level_id or (self._world_level_ids[0] if len(self._world_level_ids) == 1 else ""))
                self.node._register_semantic_entity(light.name, realized.name)
                lights.append(realized)
                names.append(realized.name)
            if names:
                self._episode_lights[obstacle.name] = names
        if not lights:
            return
        await self._simulator.spawn_lights(lights)
        for light in lights:
            self._simulator.attach_semantics("light", light.name, light.semantics)

    async def _remove_episode_lights(self, owners: Collection[str]) -> None:
        names = [name for owner in owners for name in self._episode_lights.pop(owner, ())]
        if not names:
            return
        for name in names:
            self._simulator.detach_semantics(name)
        await self._simulator.remove_lights(names)

    async def move_obstacles(self, setups: Collection[Obstacle]):
        """Moves already spawned static obstacles and rebuilds the collision grid without their old footprints."""
        await self.spawn_obstacles(setups)
        self._sync_static_polygons()

    async def spawn_robot(self, robots: Sequence[Robot]) -> Sequence[Robot]:
        """
        Loads given robot into the simulator
        """
        await self._human_simulator.spawn_robot(robots=tuple(map(self.realize, robots)))
        return robots

    def robot_controllers(self, robot: Robot) -> list[str]:
        return self._simulator.robot_controllers(robot)

    async def move_robot(self, robots: Sequence[Robot]) -> Sequence[bool]:
        """
        Moves given robot
        """
        return await self._human_simulator.move_robot(tuple(map(self.realize, robots)))

    async def remove_robot(self, robots: Sequence[Robot]) -> Sequence[bool]:
        """
        Deletes given robot
        """
        return await self._human_simulator.remove_robot(tuple(map(self.realize, robots)))

    async def respawn(self, callback: Callable[[], typing.Awaitable[Any]]):
        """
        Unuse obstacles, (re-)use them in callback, finally remove unused obstacles
        @callback: Function to call between unuse and remove
        """
        await self._human_simulator.unuse_obstacles()
        self._episode_light_owners = set()
        try:
            await callback()
        finally:
            owners, self._episode_light_owners = self._episode_light_owners, None
        await self._remove_episode_lights([owner for owner in self._episode_lights if owner not in owners])
        await self._human_simulator.remove_obstacles(purge=ObstacleLayer.UNUSED)
        self._sync_static_polygons()

    async def respawn_world(self, world: WorldDescription, detected_walls: dict[str, Sequence[Wall]] | None = None):
        """
        Replace world obstacles atomically: old items are only cleaned
        up after new ones have been spawned successfully.
        """
        old_walls, old_doors = self._human_simulator.unuse_world()
        await self._simulator.remove_mechanisms()
        await self._simulator.remove_world()
        await self.spawn_world_obstacles(world, detected_walls=detected_walls)
        self._human_simulator.remove_stale_world(old_walls, old_doors)
        await self._human_simulator.remove_obstacles(purge=ObstacleLayer.UNUSED)

    async def setup_regions(self, regions: Sequence[Region]) -> bool:
        """
        Configure regions (sources/sinks) on the human simulator.
        """
        return await self._human_simulator.setup_regions(regions)

    async def configure_contact(self, mode: str, standing_distance: float) -> None:
        """Contact mode of the human simulator for the episode about to spawn."""
        await self._human_simulator.configure_contact(mode, standing_distance)

    async def configure_gestures(self, mode: str) -> None:
        """Gesture mode of the human simulator for the episode about to spawn."""
        await self._human_simulator.configure_gestures(mode)

    async def configure_humans(self, params: Sequence[ParameterMsg]) -> list[ParameterMsg]:
        """
        Apply episode-level params on the human simulator and return the accepted ones.
        """
        return await self._human_simulator.configure(params)

    async def remove_all_regions(self) -> bool:
        """
        Remove all tracked regions from the human simulator.
        """
        return await self._human_simulator.remove_all_regions()

    async def reset(self, purge: ObstacleLayer = ObstacleLayer.INUSE):
        """
        Unuse and remove all obstacles
        """
        await self._human_simulator.remove_obstacles(purge=purge)
        await self._remove_episode_lights(list(self._episode_lights))
        self._sync_static_polygons()
        if purge >= ObstacleLayer.WORLD:
            self._collision_grid = None

    async def step(self, n: int = 1) -> bool:
        return await self._simulator.step(n)

    async def before_reset_episode(self) -> bool:
        await self._human_simulator.pause()
        result = await self._simulator.before_reset_episode()
        self._simulator.reset_semantics()
        self.node.reset_timeline()
        return result

    async def after_reset_episode(self) -> bool:
        try:
            return await self._simulator.after_reset_episode()
        finally:
            await self._human_simulator.unpause()
