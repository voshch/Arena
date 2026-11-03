import asyncio
import json
import typing
from collections.abc import Callable, Collection, Sequence
from typing import Any

import attrs
import rclpy.qos
import shapely
from arena_simulation_setup.shared import Ceiling, Light, Placement, Stairs
from arena_simulation_setup.shared.navmesh import NavLevel, nav_floors, nav_walls
from arena_simulation_setup.tree.World import Level, World, entity_lights
from arena_simulation_setup.tree.World.model import render_door_polygons, render_elevator_polygons
from arena_simulation_setup.tree.World.Scenario import TimelineEntry
from arena_simulation_setup.utils.affordance import AffordanceSlot
from rcl_interfaces.msg import Parameter as ParameterMsg
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from arena_runtime._node import NodeInterface
from arena_runtime.env.affordance_markers import affordance_markers
from arena_runtime.env.collision_grid import CollisionGrid
from arena_runtime.env.realizer import Realizer
from arena_runtime.env.world_manager.utils import WorldMap
from arena_runtime.flags import ObstaclesOptim, obstacles_optim_level
from arena_runtime.humans import BaseHumanSimulator
from arena_runtime.humans.utils import ObstacleLayer
from arena_runtime.shared import (
    WORLD_ENTITY_PREFIX,
    Door,
    DynamicObstacle,
    Elevator,
    Entity,
    Obstacle,
    Pose,
    Position,
    Region,
    Robot,
    SemanticCfg,
    Wall,
)
from arena_runtime.sim import BaseSim
from arena_runtime.sim._semantics import _SEMANTIC_KINDS

EntityT = typing.TypeVar('EntityT', bound=Entity)


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


def _realized_plate(realizer: Realizer, cfg: SemanticCfg, level_id: str) -> SemanticCfg:
    """A plate cfg with its authored position in the map frame agents are tracked in and its driven door under its realized name."""
    params = dict(cfg.params)
    if 'position' in params:
        x, y = params['position'][:2]
        params['position'] = list(realizer.realize_xy(x, y, level_id))
    if params.get('drives'):
        params['drives'] = realizer.realize_reference(params['drives'], level_id)
    return attrs.evolve(cfg, params=params)


def _attachments(realizer: Realizer, extras: dict[str, list[SemanticCfg]], level_id: str) -> list[tuple[str, list[SemanticCfg]]]:
    return [(kind, [_realized_plate(realizer, cfg, level_id) for cfg in cfgs] if kind == 'pressure_plate' else cfgs) for kind, cfgs in extras.items()]


_OCCUPANCY_CAP_VOCAB = _kind_vocab(_SEMANTIC_KINDS["occupancy_cap"])


def authored_statics(level: Level) -> Level:
    """The level with its objects under their authored names, as lights attached to them name them."""
    return attrs.evolve(level, objects=[attrs.evolve(entity, name=entity.name.removeprefix(WORLD_ENTITY_PREFIX)) for entity in level.objects])


async def world_lights(level: Level) -> list[Light]:
    """The lights of a level under authored names, each owner under the name its static entity spawns with."""
    return [attrs.evolve(light, owner=f'{WORLD_ENTITY_PREFIX}{light.owner}') if light.owner else light for light in await authored_statics(level).all_lights()]


def admit_lights(lights: Sequence[Light], taken: Collection[str]) -> tuple[list[Light], list[Light]]:
    """Split lights into those whose name is free and those whose name is taken or repeats earlier in the batch."""
    seen = set(taken)
    admitted: list[Light] = []
    refused: list[Light] = []
    for light in lights:
        (refused if light.name in seen else admitted).append(light)
        seen.add(light.name)
    return admitted, refused


STAGE_AMBIENT_TOPIC = '/arena/stage_ambient'
MODE_LIGHT_OWNER = ''
STAGE_AMBIENT_ENV = 'env_0'


def stage_ambient(lights: Sequence[Light]) -> list[dict]:
    """The dome and sun lights of a world as comparable settings, sorted, names left out."""
    return sorted(
        ({'fixture': light.fixture, 'lux': light.lux, 'cct_K': light.cct_K, 'direction': list(light.direction), 'dim': light.dim, 'lit': light.lit, 'light_on': light.light_on} for light in lights if light.spec.ambient),
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
    _nav_levels: tuple[NavLevel, ...]
    _static_polygons: dict[str, shapely.Polygon]
    _affordances: dict[str, tuple[str, Pose, list[AffordanceSlot]]]

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
        self._nav_levels = ()
        self._static_polygons = {}
        self._affordances = {}
        self._affordance_markers_pub = self.node.create_publisher(
            MarkerArray,
            'affordance_markers',
            rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE),
        )
        self._human_simulator.on_held_seats_change(self._publish_affordance_markers)
        self._attached_semantic_entities: set[str] = set()
        self._world_light_names: set[str] = set()
        self._episode_lights: dict[str, dict[str, str]] = {}
        self._episode_light_owners: set[str] | None = None
        self._robot_lights: dict[str, list[Light]] = {}
        self._carried_lights: dict[str, dict[str, str]] = {}
        self._stage_ambient: list[dict] | None = None
        self._own_ambient: list[dict] | None = None
        self._stage_ambient_warned: tuple[str, str] | None = None
        qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        if self._env_name == STAGE_AMBIENT_ENV:
            self._stage_ambient_pub = self.node.create_publisher(String, STAGE_AMBIENT_TOPIC, qos)
        else:
            self._stage_ambient_sub = self.node.create_subscription(String, STAGE_AMBIENT_TOPIC, self._on_stage_ambient, qos)

    @property
    def stacks_levels(self) -> bool:
        """Whether the simulator can place levels above one another."""
        return self._simulator.STACKS_LEVELS

    @property
    def simulator_name(self) -> str:
        return self._simulator.SIM_NAME

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

    @typing.overload
    def realize(self, target: str) -> str: ...

    @typing.overload
    def realize(self, target: Position) -> Position: ...

    @typing.overload
    def realize(self, target: Pose) -> Pose: ...

    @typing.overload
    def realize(self, target: EntityT) -> EntityT: ...

    def realize(self, target: str | Position | Pose | Entity) -> str | Position | Pose | Entity:
        return self._realizer.realize(target)

    def ezilear(self, target: Pose) -> Pose:
        return self._realizer.ezilear(target)

    def placement(self) -> Placement:
        """The map from the abstract frame into the simulator frame for this env."""
        return self._realizer.placement()

    @property
    def nav_levels(self) -> tuple[NavLevel, ...]:
        """Floors, blocking walls, stair halves and doors under their authored names of every level of the spawned world, in the simulator frame."""
        return self._nav_levels

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
        slots = await asyncio.gather(*(o.affordance_slots() for o in obstacles), return_exceptions=True)
        for obstacle, found in zip(obstacles, slots, strict=True):
            if isinstance(found, ValueError):
                self._logger.warning(f"Ignoring the affordances of '{obstacle.name}': {found}")
                found = []
            elif isinstance(found, BaseException):
                raise found
            if found:
                self._affordances[obstacle.name] = (obstacle.sim_path, obstacle.pose, found)
            else:
                self._affordances.pop(obstacle.name, None)
        self._publish_affordance_markers()

    def _sync_static_polygons(self) -> None:
        """Drop polygons and affordance slots whose obstacle is no longer registered in the human simulator."""
        alive = set(self._human_simulator._known_obstacles.keys())
        self._static_polygons = {n: p for n, p in self._static_polygons.items() if n in alive}
        self._affordances = {n: placed for n, placed in self._affordances.items() if n in alive}
        if self._collision_grid is not None:
            self._collision_grid.sync(self._static_polygons.keys())
        self._publish_affordance_markers()

    def _publish_affordance_markers(self) -> None:
        """Republish the latched slot overlay of every spawned obstacle, occupancy from the seats agents hold on it."""
        held = self._human_simulator.held_seats()
        objects = ((name, pose, slots, held.get(sim_path, [])) for name, (sim_path, pose, slots) in sorted(self._affordances.items()))
        self._affordance_markers_pub.publish(MarkerArray(markers=affordance_markers(objects, self.node.sim_time.to_msg())))

    async def spawn_world_obstacles(
        self,
        world: World | Level,
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
        world: World | Level,
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

        _world = World.from_levels(world) if isinstance(world, Level) else world

        self._detach_extra_semantics()
        for name in (name for names in self._carried_lights.values() for name in names.values()):
            self._simulator.detach_semantics(name)
        await self._simulator.remove_lights()
        self._carried_lights.clear()
        self._episode_lights.clear()
        self._world_light_names.clear()
        self._realizer.reset_levels([str(fid) for fid in _world.levels if _match_level_id(fid)])
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
                pending.extend((kind, realized.name, cfgs, None) for kind, cfgs in _attachments(self._realizer, extras, fid))
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
                pending.extend((kind, realized_e.name, cfgs, None) for kind, cfgs in _attachments(self._realizer, extras, fid))
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
                    pending.append(("occupancy_cap", zone_name, occ, self._realizer.realize_polygon(zone.polygon, fid)))
        for sched in _world.schedules:
            realized_s = self._realizer.realize(sched)
            self.node._register_semantic_entity(sched.name, realized_s.name)
            if realized_s.semantics:
                pending.append(("schedule", realized_s.name, list(realized_s.semantics), None))
        for sig in _world.signals:
            realized_sig = self._realizer.realize(sig)
            self.node._register_semantic_entity(sig.name, realized_sig.name)
            if realized_sig.semantics:
                pending.append(("signal", realized_sig.name, list(realized_sig.semantics), None))
        elevators = tuple(elevators_list)
        stairs_list: list[Stairs] = []
        level_stairs: dict[str, list[Stairs]] = {}
        for fid, level in _world.levels.items():
            if not _match_level_id(fid):
                continue
            level_stairs[fid] = []
            for stair in level.stairs:
                realized_f = self._realizer.realize(stair, fid)
                self.node._register_semantic_entity(stair.name, realized_f.name)
                stairs_list.append(realized_f)
                level_stairs[fid].append(realized_f)
        stairs = tuple(stairs_list)
        self._nav_levels = tuple(
            NavLevel(
                level=str(fid),
                floors=tuple(self._realizer.place(self._realizer.realize(f, fid), fid) for f in nav_floors(level)),
                walls=tuple(self._realizer.place(self._realizer.realize(w, fid), fid) for w in nav_walls(level)),
                stairs=tuple(self._realizer.place(stair, fid) for stair in level_stairs[fid]),
                doors=tuple(self._realizer.place(attrs.evolve(self._realizer.realize(d, fid), name=d.name), fid) for d in level.all_doors),
            )
            for fid, level in _world.levels.items()
            if _match_level_id(fid)
        )
        self._human_simulator.spawn_stairs(stairs)
        statics = tuple(self._realizer.realize(s, fid) for fid, level in _world.levels.items() if _match_level_id(fid) for s in level.all_static_entities)
        if self._skip_obstacles:
            statics = ()
        for static in statics:
            self.node._register_semantic_entity(static.name.removeprefix(WORLD_ENTITY_PREFIX), self._realizer.prefix(static.name))

        if world_map is not None:
            cutouts = [p for d in doors for p in render_door_polygons(d)] + [p for e in elevators for p in render_elevator_polygons(e)]
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
        futures.append(self._spawn_elevators(elevators))
        if stairs:
            futures.append(self._simulator.spawn_stairs(stairs))

        await asyncio.gather(*futures)
        if lights:
            await self._simulator.spawn_lights(lights)
        self._declare_ambient(lights)
        for robot in self._robot_lights:
            await self._spawn_carried_lights(robot)

        # Door/elevator kinds self-attach inside their spawn helpers. The scripted,
        # position, and zone kinds have no geometry spawn, so attach them here once
        # their host geometry exists (gate/plate read the spawned door runtime).
        for kind, entity, cfgs, polygon in pending:
            self._simulator.attach_semantics(kind, entity, cfgs, polygon=polygon)
            self._attached_semantic_entities.add(entity)

    async def _spawn_elevators(self, elevators: Sequence[Elevator]) -> None:
        """Hand the cabins to the human simulator, then to the physics sim."""
        await self._human_simulator.spawn_elevators(elevators)
        if elevators:
            await self._simulator.spawn_elevators(elevators)

    async def spawn_dynamic_obstacles(self, setups: Collection[DynamicObstacle]):
        """
        Loads given dynamic obstacles into the simulator.
        """
        realized = tuple(self._realizer.realize(obstacle, obstacle.level) for obstacle in setups)
        await self._human_simulator.spawn_dynamic_obstacles(realized)

    async def spawn_obstacles(self, setups: Collection[Obstacle]):
        """
        Loads given obstacles into the simulator.
        """
        if self._skip_obstacles:
            return
        realized = tuple(self._realizer.realize(obstacle, obstacle.level) for obstacle in setups)
        for obstacle in realized:
            if not obstacle.name.startswith(WORLD_ENTITY_PREFIX):
                self.node._register_semantic_entity(obstacle.name, self._realizer.prefix(obstacle.name))
        await self._cache_polygons(realized)
        await self._human_simulator.spawn_obstacles(realized)
        await self._spawn_episode_lights(setups)

    async def _spawn_episode_lights(self, setups: Collection[Obstacle]) -> None:
        """Spawn the lights the objects of episode obstacles carry, replacing those of an obstacle spawned before."""
        await self._spawn_owned_lights([(obstacle.name, await entity_lights(obstacle), obstacle.level) for obstacle in setups if not obstacle.name.startswith(WORLD_ENTITY_PREFIX)])

    async def spawn_mode_lights(self, lights: Sequence[Light]) -> None:
        """Spawn the lights the obstacle mode declares for this episode, replacing those of the episode before."""
        await self._spawn_owned_lights([(MODE_LIGHT_OWNER, list(lights), "")])

    async def _spawn_owned_lights(self, owned: Sequence[tuple[str, list[Light], str]]) -> None:
        await self._remove_episode_lights([owner for owner, _, _ in owned if owner in self._episode_lights])
        lights: list[Light] = []
        for owner, owner_lights, level_id in owned:
            if self._episode_light_owners is not None:
                self._episode_light_owners.add(owner)
            names: dict[str, str] = {}
            for light in self._admit(owner_lights, f'obstacle {owner!r}' if owner else 'the obstacle mode'):
                realized = self._realizer.realize(light, level_id)
                self.node._register_semantic_entity(light.name, realized.name)
                lights.append(realized)
                names[light.name] = realized.name
            if names:
                self._episode_lights[owner] = names
        if not lights:
            return
        await self._simulator.spawn_lights(lights)
        for light in lights:
            self._simulator.attach_semantics("light", light.name, light.semantics)

    def _admit(self, lights: Sequence[Light], owner: str) -> list[Light]:
        """The lights whose name no spawned world, robot or episode light holds, the others logged and left out."""
        taken = {*self._world_light_names, *(name for names in (*self._carried_lights.values(), *self._episode_lights.values()) for name in names)}
        admitted, refused = admit_lights(lights, taken)
        for light in refused:
            self._logger.warning(f"light {light.name!r} of {owner} shares its name with another light, skipped")
        return admitted

    async def _remove_episode_lights(self, owners: Collection[str]) -> None:
        names = [name for owner in owners for name in self._episode_lights.pop(owner, {}).values()]
        if not names:
            return
        for name in names:
            self._simulator.detach_semantics(name)
        await self._simulator.remove_lights(names)

    async def _spawn_carried_lights(self, robot: str) -> None:
        lights = self._admit(self._robot_lights[robot], f'robot {robot!r}')
        if not lights:
            return
        realized = [self._realizer.realize(light) for light in lights]
        self._carried_lights[robot] = {light.name: realized_l.name for light, realized_l in zip(lights, realized, strict=True)}
        for name, realized_name in self._carried_lights[robot].items():
            self.node._register_semantic_entity(name, realized_name)
        await self._simulator.spawn_lights(realized)
        for light in realized:
            self._simulator.attach_semantics("light", light.name, light.semantics)

    async def spawn_robot_lights(self, robot: str, lights: Sequence[Light]) -> None:
        """Spawn the lights a robot carries, respawned with the world until remove_robot_lights."""
        await self.remove_robot_lights(robot)
        self._robot_lights[robot] = list(lights)
        await self._spawn_carried_lights(robot)

    async def remove_robot_lights(self, robot: str) -> None:
        self._robot_lights.pop(robot, None)
        names = list(self._carried_lights.pop(robot, {}).values())
        if not names:
            return
        for name in names:
            self._simulator.detach_semantics(name)
        await self._simulator.remove_lights(names)

    async def move_obstacles(self, setups: Collection[Obstacle]):
        """Moves already spawned static obstacles and rebuilds the collision grid without their old footprints."""
        await self.spawn_obstacles(setups)
        self._sync_static_polygons()

    async def notify_stimulus(self, agent: int | str, stimulus: str, intensity: float) -> bool:
        return await self._human_simulator.notify_stimulus(agent, stimulus, intensity)

    def check_stimuli(self, scenario: str, humans: Sequence[DynamicObstacle], timeline: Sequence[TimelineEntry]) -> None:
        self._human_simulator.check_stimuli(scenario, humans, timeline)

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

    async def respawn_world(self, world: World, detected_walls: dict[str, Sequence[Wall]] | None = None):
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
        realized = [attrs.evolve(region, polygon=[self._realizer.realize(vertex) for vertex in region.polygon]) for region in regions]
        return await self._human_simulator.setup_regions(realized)

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
        await self._human_simulator.remove_all_regions()
        result = await self._simulator.before_reset_episode()
        self._simulator.reset_semantics()
        self.node.reset_timeline()
        return result

    async def after_reset_episode(self) -> bool:
        try:
            return await self._simulator.after_reset_episode()
        finally:
            await self._human_simulator.unpause()
