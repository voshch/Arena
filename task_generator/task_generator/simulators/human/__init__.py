from __future__ import annotations

import abc
import asyncio
import itertools
import math
import typing
from collections.abc import Iterable, Mapping, Sequence

import attrs
import rclpy.publisher
import rclpy.qos
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry
from arena_rclpy_mixins.shared import Namespace
from arena_runtime._node import NodeInterface
from arena_runtime.sim import BaseSim
from arena_simulation_setup.tree.assets.Pedestrian import PedestrianIdentifier
from arena_simulation_setup.utils.models import ModelType
from visualization_msgs.msg import MarkerArray

from task_generator.constants import Constants
from task_generator.manager.realizer import Realizer
from task_generator.shared import Door, DynamicObstacle, Obstacle, Region, Robot, Wall
from task_generator.simulators.human.gait import GaitGenerator
from task_generator.simulators.human.utils import (
    KnownObstacle,
    KnownObstacles,
    ObstacleLayer,
)


class BaseHumanSimulator(NodeInterface, abc.ABC):
    _arena_peds_publisher: rclpy.publisher.Publisher
    _marker_publisher: rclpy.publisher.Publisher
    _static_marker_publisher: rclpy.publisher.Publisher
    _known_obstacles: KnownObstacles
    _known_walls: KnownObstacles[Wall]
    _known_doors: KnownObstacles[Door]

    @classmethod
    def _register_task_modes(cls) -> None:
        """Register simulator-specific obstacle task modes.

        Called during __init__. Subclasses override this to register
        task modes specific to their simulator (e.g. TM_Prompt).
        """

    def __init__(self, *args: object, namespace: Namespace, simulator: BaseSim, realizer: Realizer, **kwargs: object) -> None:
        """
        Initialize human simulator.

        Args:
            namespace: global namespace
            simulator: Simulator instance
            realizer: per-env Realizer (used to stamp sim_path on runtime-spawned obstacles)
        """
        super().__init__(*args, **kwargs)
        self._register_task_modes()
        self._simulator = simulator
        self._namespace = namespace
        self._realizer = realizer

        self._known_obstacles = KnownObstacles[Obstacle]()
        self._known_walls = KnownObstacles[Wall]()
        self._known_doors = KnownObstacles[Door]()
        self._wall_counter = itertools.count()
        self._known_regions: dict[str, Region] = {}
        self._warned_unresolved_models: set[str] = set()
        self._ped_model_uris: dict[str, str] = {}
        self._episode_driver: str = ""
        self._spawned_driver: str | None = None

        self._arena_peds_publisher = self.node.create_publisher(Pedestrians, self._namespace("arena_peds"), 10)
        self._marker_publisher = self.node.create_publisher(
            MarkerArray,
            self._namespace("pedestrian_markers", "extra"),
            rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )
        self._static_marker_publisher = self.node.create_publisher(
            MarkerArray,
            self._namespace("pedestrian_markers", "static"),
            rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        self._ped_positions_xy: dict[str, tuple[float, float]] = {}
        self._gait = GaitGenerator()
        self._gait_prev_stamp: dict[int, float] = {}
        self.node.create_subscription(
            Pedestrians,
            self._namespace("arena_peds"),
            self._on_arena_peds,
            10,
        )
        self._simulator.attach_human_simulator(self)

    def publish_arena_peds(self, msg: Pedestrians) -> None:
        """Fill bare-name joint_state for each ped missing one, then publish."""
        now = self.node.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        stamp = now.to_msg()

        current_ids: set[int] = {ped.id for ped in msg.pedestrians}
        for stale in sorted(set(self._gait_prev_stamp) - current_ids):
            self._gait.forget(stale)
            del self._gait_prev_stamp[stale]

        for ped in msg.pedestrians:
            ped.model_uri = self._ped_model_uris.get(ped.name, "")
            if ped.joint_state.name:
                continue
            prev = self._gait_prev_stamp.get(ped.id)
            dt = (now_sec - prev) if prev is not None and now_sec > prev else 0.1
            self._gait_prev_stamp[ped.id] = now_sec
            speed = math.hypot(ped.twist.linear.x, ped.twist.linear.y)
            angles = self._gait.compute(ped.id, ped.animation_state, speed, dt)
            ped.joint_state = self._gait.joint_state(angles, stamp=stamp)

        self._arena_peds_publisher.publish(msg)

    def publish_markers(self, markers: MarkerArray) -> None:
        """Publish a transient debug-overlay MarkerArray on `pedestrian_markers/extra`."""
        self._marker_publisher.publish(markers)

    def publish_static_markers(self, markers: MarkerArray) -> None:
        """Publish a latched MarkerArray on `pedestrian_markers/static` (TRANSIENT_LOCAL, late subscribers catch up)."""
        self._static_marker_publisher.publish(markers)

    def _on_arena_peds(self, msg: Pedestrians) -> None:
        self._ped_positions_xy = {p.name: (p.pose.position.x, p.pose.position.y) for p in msg.pedestrians}

    def pedestrian_positions_xy(self) -> Iterable[tuple[str, tuple[float, float]]]:
        return list(self._ped_positions_xy.items())

    async def pedestrian_teleport(self, destinations: Mapping[str, tuple[float, float]]) -> bool:
        """Teleport tracked pedestrians to given (x, y). Default impl asks the sim to move them."""
        if not destinations:
            return True
        peds_msg = Pedestrians()
        for name, (x, y) in destinations.items():
            ped = Pedestrian()
            ped.name = name
            cur = self._ped_positions_xy.get(name)
            ped.pose.position.x = x
            ped.pose.position.y = y
            ped.pose.position.z = 0.0
            peds_msg.pedestrians.append(ped)
            if cur is not None:
                self._ped_positions_xy[name] = (x, y)
        results = await self._simulator.pedestrian_update(peds_msg)
        return all(results)

    async def spawn_obstacles(self, obstacles: Sequence[Obstacle], layer: ObstacleLayer = ObstacleLayer.INUSE):
        """Spawns static obstacles.

        Args:
            obstacles (Sequence[Obstacle]): Static obstacles to spawn.
            layer (ObstacleLayer, optional): Layer to assign to spawned obstacles. Defaults to ObstacleLayer.INUSE.
        """
        self._logger.debug(f"spawning {len(obstacles)} static obstacles")

        futures: list[typing.Awaitable] = []
        to_register: list[KnownObstacle[Obstacle]] = []
        to_move: list[Obstacle] = []

        for obstacle in obstacles:
            if (known := self._known_obstacles.get(obstacle.name)) is not None:
                known.obstacle = obstacle
                to_move.append(known.obstacle)
                known.layer = layer
            else:
                known = self._known_obstacles.create_or_get(
                    name=obstacle.name,
                    obstacle=obstacle,
                )
            if not known.spawned:
                to_register.append(known)
        if to_move:
            futures.append(self._simulator.obstacle_move(to_move))

        to_spawn: list[Obstacle] = []
        for known, obstacle in zip(
            to_register,
            await self._spawn_obstacles_impl([known.obstacle for known in to_register]),
            strict=False,
        ):
            if not obstacle:
                continue
            known.obstacle = obstacle
            known.spawned = True

            if known.layer == ObstacleLayer.UNUSED:
                to_spawn.append(known.obstacle)
            known.layer = layer

        if to_spawn:
            futures.append(self._simulator.obstacle_spawn(to_spawn))
        await asyncio.gather(*futures)

    async def spawn_dynamic_obstacles(self, obstacles: typing.Sequence[DynamicObstacle]):
        """Spawns dynamic obstacles.

        Args:
            obstacles (typing.Sequence[DynamicObstacle]): Dynamic obstacles to spawn.
        """
        self._logger.debug(f"spawning {len(obstacles)} dynamic obstacles")

        if obstacles:
            driver_set = self.node.conf.Human.DRIVER_SET.value
            if driver_set:
                self._episode_driver = self.node.conf.General.RNG.once(
                    "humansim",
                    "driver",
                    factory=lambda: str(
                        self.node.conf.General.RNG.stream("humansim", "driver").choice(driver_set)
                    ),
                )
                # WARN so it survives the env nodes' default WARN log level — this line is
                # experiment-critical telemetry (driver-stratified eval needs it in logs).
                self._logger.warning(f"episode driver: {self._episode_driver}")
            else:
                self._episode_driver = ""

        futures: list[typing.Awaitable] = []
        to_register: list[KnownObstacle[DynamicObstacle]] = []
        to_move: list[DynamicObstacle] = []
        to_respawn: list[DynamicObstacle] = []

        for obstacle in obstacles:
            if (known := self._known_obstacles.get(obstacle.name)) is not None:
                if known.spawned and self._episode_driver != self._spawned_driver:
                    to_respawn.append(known.obstacle)
                    known.spawned = False
                    known.layer = ObstacleLayer.UNUSED
                else:
                    known.layer = ObstacleLayer.INUSE
                known.obstacle = obstacle
                if known.spawned:
                    to_move.append(known.obstacle)
            else:
                known = self._known_obstacles.create_or_get(name=obstacle.name, obstacle=obstacle)
            if not known.spawned:
                to_register.append(known)

        if to_respawn:
            await self._simulator.pedestrian_delete(to_respawn)
        if to_move:
            futures.append(self._simulator.pedestrian_move(to_move))

        to_spawn: list[DynamicObstacle] = []
        for known, obstacle in zip(
            to_register,
            await self._spawn_dynamic_obstacles_impl([known.obstacle for known in to_register]),
            strict=False,
        ):
            self._logger.debug(f"Spawned dynamic obstacle: {obstacle}")
            if not obstacle:
                continue

            known.obstacle = obstacle
            known.spawned = True

            if known.layer == ObstacleLayer.UNUSED:
                to_spawn.append(known.obstacle)
            known.layer = ObstacleLayer.INUSE

        if to_spawn:
            futures.append(self._simulator.pedestrian_spawn(await self._ensure_spawnable(to_spawn)))
        await asyncio.gather(*futures)

        if obstacles:
            self._spawned_driver = self._episode_driver

    _PEDESTRIAN_FALLBACK: typing.ClassVar[str] = "arenian"

    async def _ensure_spawnable(self, obstacles: Sequence[DynamicObstacle]) -> Sequence[DynamicObstacle]:
        """Swap unresolvable ped models for _PEDESTRIAN_FALLBACK and record each
        ped's resolved SDF path, which publish_arena_peds stamps as model_uri so
        PedSkeletonPlugin can create the actor."""

        async def _sdf_path(obs: DynamicObstacle) -> str | None:
            view = await obs.model.resolve()
            model = await view.model.get(ModelType.SDF)
            if model.type is ModelType.UNKNOWN or model.path is None:
                return None
            return str(model.path)

        async def _resolve(obs: DynamicObstacle) -> DynamicObstacle:
            try:
                path = await _sdf_path(obs)
                if path is not None:
                    self._record_ped_model(obs, path)
                    return obs
            except Exception as e:
                if obs.model.name not in self._warned_unresolved_models:
                    self._warned_unresolved_models.add(obs.model.name)
                    self._logger.warning(
                        f"pedestrian model {obs.model.name!r} unresolved ({e}); using fallback",
                    )
            else:
                if obs.model.name not in self._warned_unresolved_models:
                    self._warned_unresolved_models.add(obs.model.name)
                    self._logger.warning(
                        f"pedestrian model {obs.model.name!r} has no SDF; using fallback",
                    )
            fallback = attrs.evolve(obs, model=PedestrianIdentifier.parse(self._PEDESTRIAN_FALLBACK))
            try:
                fallback_path = await _sdf_path(fallback)
            except Exception:
                fallback_path = None
            self._record_ped_model(fallback, fallback_path or "")
            return fallback

        return await asyncio.gather(*(_resolve(o) for o in obstacles))

    def _record_ped_model(self, obs: DynamicObstacle, model_uri: str) -> None:
        """Map both a ped's name and sim_path to its SDF path, since the published
        ped name is one or the other depending on the simulator."""
        self._ped_model_uris[obs.name] = model_uri
        self._ped_model_uris[obs.sim_path] = model_uri

    async def spawn_world(
        self,
        walls: Sequence[Wall],
        doors: Sequence[Door],
    ):
        """Spawns world elements.

        Args:
            walls (Sequence[Wall]): Walls to spawn.
            doors (Sequence[Door]): Doors to spawn.
        """
        self._logger.debug(f"spawning {len(walls)} walls and {len(doors)} doors")

        wall_map: dict[str, Wall] = {}
        for wall in walls:
            name = f"wall_{next(self._wall_counter)}"
            self._known_walls.create_or_get(
                name=name,
                obstacle=wall,
                layer=ObstacleLayer.WORLD,
            )
            wall_map[name] = wall

        door_map: dict[str, Door] = {}
        for door in doors:
            self._known_doors.create_or_get(
                name=door.name,
                obstacle=door,
                layer=ObstacleLayer.WORLD,
            )
            door_map[door.name] = door

        await asyncio.gather(
            self._simulator.spawn_doors(doors),
            self._simulator.spawn_walls(walls, clear_existing=False),
            self._spawn_walls_impl(wall_map),
            self._spawn_doors_impl(door_map),
        )

    @staticmethod
    def _door_wall_name(door_name: str) -> str:
        return f"__door_{door_name}"

    async def update_doors(
        self,
        doors: Sequence[tuple[Door, bool]],
    ):
        """Update door states. True = open (no wall), False = closed (wall spawned).

        Args:
            doors (Sequence[tuple[Door, bool]]): Door/state pairs.
        """
        to_spawn: dict[str, Wall] = {}
        to_remove: list[str] = []

        for door, is_open in doors:
            name = self._door_wall_name(door.name)
            if is_open:
                if name in self._known_walls:
                    to_remove.append(name)
            else:
                if name not in self._known_walls:
                    wall = Wall(start=door.start, end=door.end)
                    self._known_walls.create_or_get(
                        name=name,
                        obstacle=wall,
                        layer=ObstacleLayer.WORLD,
                    )
                    to_spawn[name] = wall

        futures: list[typing.Awaitable] = []
        if to_spawn:
            self._logger.debug(f"closing {len(to_spawn)} doors: {list(to_spawn)}")
            futures.append(self._spawn_walls_impl(to_spawn))
        if to_remove:
            self._logger.debug(f"opening {len(to_remove)} doors: {to_remove}")
            for name in to_remove:
                self._known_walls.forget(name)
            futures.append(self._remove_walls_impl(to_remove))
        await asyncio.gather(*futures)

    async def unuse_obstacles(self):
        """
        Prepares obstacles for reuse or removal.
        """
        self._logger.debug("unusing obstacles")

        obstacle_names = [name for name, known in self._known_obstacles.items() if not isinstance(known.obstacle, DynamicObstacle) and known.layer == ObstacleLayer.UNUSED]

        await asyncio.gather(
            self._remove_pedestrians_impl(),
            self._remove_obstacles_impl(obstacle_names),
        )

        for name in obstacle_names:
            self._known_obstacles.forget(name)

        for obstacle in self._known_obstacles.values():
            if obstacle.layer == ObstacleLayer.INUSE:
                obstacle.spawned = False
                obstacle.layer = ObstacleLayer.UNUSED

    def unuse_world(self) -> tuple[frozenset[str], frozenset[str]]:
        """Mark world-level items for replacement.

        Downgrades WORLD-layer obstacles to UNUSED so they can be
        reclaimed after new world obstacles are spawned.

        Returns:
            Tuple of (old_wall_names, old_door_names) for stale-entry cleanup.
        """
        self._logger.debug("unusing world")

        old_walls = frozenset(self._known_walls.keys())
        old_doors = frozenset(self._known_doors.keys())

        for obstacle in self._known_obstacles.values():
            if obstacle.layer >= ObstacleLayer.WORLD:
                obstacle.spawned = False
                obstacle.layer = ObstacleLayer.UNUSED

        return old_walls, old_doors

    def remove_stale_world(
        self,
        old_walls: frozenset[str],
        old_doors: frozenset[str],
    ):
        """Forget wall/door tracking entries that were not reused during respawn.

        Args:
            old_walls: Wall names from before the respawn.
            old_doors: Door names from before the respawn.
        """
        for name in old_walls:
            if name in self._known_walls:
                self._known_walls.forget(name)
        for name in old_doors:
            if name in self._known_doors:
                self._known_doors.forget(name)

    async def remove_walls(
        self,
        walls: Sequence[Wall],
    ):
        """Removes specific walls from the simulation.

        Args:
            walls (Sequence[Wall]): Walls to remove (matched by start/end coordinates).
        """
        matched: dict[str, None] = {}
        for wall in walls:
            for name, known in self._known_walls.items():
                if known.obstacle.start == wall.start and known.obstacle.end == wall.end:
                    matched[name] = None
                    break

        if not matched:
            return

        names = tuple(matched)
        self._logger.debug(f"removing {len(names)} walls: {names}")
        for name in names:
            self._known_walls.forget(name)
        await self._remove_walls_impl(names)

    async def remove_doors(
        self,
        doors: Sequence[Door],
    ):
        """Removes specific doors from the simulation.

        Args:
            doors (Sequence[Door]): Doors to remove (matched by name).
        """
        names = [door.name for door in doors if door.name in self._known_doors]
        if not names:
            return

        self._logger.debug(f"removing {len(names)} doors: {names}")
        for name in names:
            self._known_doors.forget(name)
        await self._remove_doors_impl(names)

    async def setup_regions(self, regions: typing.Sequence[Region]) -> bool:
        """Configure regions (sources/sinks) for dynamic agent spawning.

        Args:
            regions: Already-resolved Region objects with concrete polygons.
        """
        for region in regions:
            self._known_regions[region.name] = region
        return await self._add_regions_impl(regions)

    async def remove_all_regions(self) -> bool:
        """Remove all tracked regions."""
        regions = tuple(self._known_regions.values())
        self._known_regions.clear()
        return await self._remove_regions_impl(regions)

    async def remove_obstacles(self, purge: ObstacleLayer = ObstacleLayer.UNUSED):
        """Removes obstacles from simulator.

        Args:
            purge (ObstacleLayer, optional): Level of obstacles to remove. Defaults to ObstacleLayer.UNUSED.
        """
        self._logger.debug(f"removing obstacles (level {purge})")
        futures: list[typing.Awaitable] = []

        stale_walls = [name for name, known in self._known_walls.items() if purge >= known.layer]
        stale_doors = [name for name, known in self._known_doors.items() if purge >= known.layer]

        if purge >= ObstacleLayer.WORLD:
            futures.append(self._simulator.remove_world())

        if stale_walls:
            futures.append(self._remove_walls_impl(stale_walls))
            for name in stale_walls:
                self._known_walls.forget(name)
        if stale_doors:
            futures.append(self._remove_doors_impl(stale_doors))
            for name in stale_doors:
                self._known_doors.forget(name)

        static: list[Obstacle] = []
        dynamic: list[DynamicObstacle] = []
        for oid, known in list(self._known_obstacles.items()):
            if purge >= known.layer:  # tmp: always respawn all dynamic obstacles
                if isinstance(known.obstacle, DynamicObstacle):
                    dynamic.append(known.obstacle)
                else:
                    static.append(known.obstacle)
                self._known_obstacles.forget(name=oid)

        static_names = [o.name for o in static]
        if static_names:
            futures.append(self._remove_obstacles_impl(static_names))
        futures.append(self._simulator.obstacle_delete(static))
        futures.append(self._simulator.pedestrian_delete(dynamic))
        await asyncio.gather(*futures)

    async def _sim_then_impl(
        self,
        sim_call: typing.Callable[[Sequence[Robot]], typing.Awaitable[Sequence[bool]]],
        impl_call: typing.Callable[[Sequence[Robot]], typing.Awaitable[Sequence[bool]]],
        robots: Sequence[Robot],
    ) -> tuple[bool, ...]:
        sim_success = await sim_call(robots)
        succeeded = tuple(r for r, s in zip(robots, sim_success, strict=False) if s)
        human_success = await impl_call(succeeded)
        human_iter = iter(human_success)
        return tuple(s and next(human_iter) for s in sim_success)

    async def spawn_robot(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        """Spawns robots.

        Args:
            robots (Sequence[Robot]): Robots to spawn.

        Returns:
            Sequence[bool]: Success of each robot spawn.
        """
        self._logger.debug(f"spawning {len(robots)} robots")
        return await self._sim_then_impl(
            self._simulator.robot_spawn,
            self._spawn_robot_impl,
            robots,
        )

    async def remove_robot(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        """Removes robots from the simulation.

        Args:
            robots (Sequence[Robot]): Robots to remove.

        Returns:
            Sequence[bool]: Success of each robot removal.
        """
        self._logger.debug(f"removing {len(robots)} robots")
        return await self._sim_then_impl(
            self._simulator.robot_delete,
            self._remove_robot_impl,
            robots,
        )

    async def move_robot(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        """Moves robots.

        Args:
            robots (Sequence[Robot]): Robots to move.

        Returns:
            Sequence[bool]: Success of each robot move.
        """
        self._logger.debug(f"moving {len(robots)} robots")
        return await self._sim_then_impl(
            self._simulator.robot_move,
            self._move_robot_impl,
            robots,
        )

    # impl

    async def pause(self):
        pass

    async def unpause(self):
        pass

    @abc.abstractmethod
    async def _spawn_obstacles_impl(
        self,
        obstacles: Sequence[Obstacle],
    ) -> Sequence[Obstacle | None]: ...

    @abc.abstractmethod
    async def _spawn_dynamic_obstacles_impl(
        self,
        obstacles: Sequence[DynamicObstacle],
    ) -> Sequence[DynamicObstacle | None]: ...

    @abc.abstractmethod
    async def _remove_obstacles_impl(
        self,
        names: Sequence[str],
    ) -> bool: ...

    @abc.abstractmethod
    async def _remove_pedestrians_impl(
        self,
    ) -> bool: ...

    @abc.abstractmethod
    async def _spawn_walls_impl(
        self,
        walls: Mapping[str, Wall],
    ) -> bool: ...

    @abc.abstractmethod
    async def _spawn_doors_impl(
        self,
        doors: Mapping[str, Door],
    ) -> bool: ...

    @abc.abstractmethod
    async def _remove_walls_impl(
        self,
        names: Sequence[str],
    ) -> bool: ...

    @abc.abstractmethod
    async def _remove_doors_impl(
        self,
        names: Sequence[str],
    ) -> bool: ...

    @abc.abstractmethod
    async def _spawn_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]: ...

    @abc.abstractmethod
    async def _remove_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]: ...

    @abc.abstractmethod
    async def _move_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]: ...

    @abc.abstractmethod
    async def _add_regions_impl(self, regions: Sequence[Region]) -> bool: ...

    @abc.abstractmethod
    async def _remove_regions_impl(self, regions: Sequence[Region]) -> bool: ...


HumanSimulatorRegistry = Registry[Constants.HumanSimulator, BaseHumanSimulator]()


@HumanSimulatorRegistry.register(Constants.HumanSimulator.DUMMY)
async def dummy(**kwargs: object) -> BaseHumanSimulator:
    from .dummy import DummyHumanSimulator

    return DummyHumanSimulator(**kwargs)


@HumanSimulatorRegistry.register(Constants.HumanSimulator.HUNAV)
async def lazy_hunavsim(**kwargs: object) -> BaseHumanSimulator:
    from .hunav.hunav import HunavHumanSimulator

    return await HunavHumanSimulator.create(**kwargs)


@HumanSimulatorRegistry.register(Constants.HumanSimulator.ISAAC)
async def isaacsim(**kwargs: object) -> BaseHumanSimulator:
    from .isaac import IsaacHumanSimulator

    return IsaacHumanSimulator(**kwargs)


@HumanSimulatorRegistry.register(Constants.HumanSimulator.ARENA)
async def arenasim(**kwargs: object) -> BaseHumanSimulator:
    from .arena_humansim.arena_humansim import ArenaHumanSimulator

    return await ArenaHumanSimulator.create(**kwargs)
