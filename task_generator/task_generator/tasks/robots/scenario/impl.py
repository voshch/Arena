import asyncio
import typing
from collections.abc import Iterable, Iterator, Mapping, Sequence

import attrs
from arena_rclpy_mixins.params import ParamHandle
from arena_runtime.env.task_api import GoToPhase, LoopPhase, Realizer, TaskPhase, TaskRequest
from arena_runtime.env.task_layer import FleetEntry, ResetPlan, RobotReset, TaskContext
from arena_runtime.shared import Pose, Position, PositionRadius
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import RobotGoal

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.scenario import ScenarioRobotParams


def to_map_frame(pose: Pose, zone: str, level: str, realizer: Realizer) -> Pose:
    """A pose shifted from the level it is placed on into the flattened map frame."""
    ox, oy = realizer.get_level_origin(realizer.placement_level(zone, level))
    return Pose(
        position=Position(pose.position.x + ox, pose.position.y + oy, pose.position.z),
        orientation=pose.orientation,
    )


def localize_phase(phase: TaskPhase, goal_level: str, realizer: Realizer) -> TaskPhase:
    """A scenario phase with its goto poses in the map frame."""
    if isinstance(phase, LoopPhase):
        return attrs.evolve(phase, phases=[localize_phase(inner, goal_level, realizer) for inner in phase.phases])
    if isinstance(phase, GoToPhase) and phase.pose is not None:
        return attrs.evolve(phase, pose=to_map_frame(phase.pose, phase.pose_zone, goal_level, realizer))
    return phase


def goto_poses(phase: TaskPhase) -> Iterator[Pose]:
    """Every goto pose of a phase, loops included."""
    if isinstance(phase, LoopPhase):
        for inner in phase.phases:
            yield from goto_poses(inner)
    elif isinstance(phase, GoToPhase) and phase.pose is not None:
        yield phase.pose


def declared_fleet(entries: Iterable[Mapping[str, object]]) -> list[FleetEntry] | None:
    """The fleet of a scenario's raw robot entries, None unless there is one and every entry names a model."""
    items = list(entries)
    if not items or not all(item.get("model") for item in items):
        return None
    return [FleetEntry(model=str(item["model"]), name=str(item.get("name", ""))) for item in items]


def pair_robots(robots: Sequence[tuple[str, str]], entries: Sequence[RobotGoal]) -> dict[int, int]:
    """Robot index to entry index: by name, then unnamed entries by model, then the rest in order."""
    pairs: dict[int, int] = {}
    paired: set[int] = set()
    for entry_index, entry in enumerate(entries):
        if entry.name:
            match = next((index for index, (name, _) in enumerate(robots) if name == entry.name and index not in pairs), None)
            if match is not None:
                pairs[match] = entry_index
                paired.add(entry_index)
    for entry_index, entry in enumerate(entries):
        if entry.name or not entry.model:
            continue
        match = next((index for index, (_, model) in enumerate(robots) if model == entry.model and index not in pairs), None)
        if match is not None:
            pairs[match] = entry_index
            paired.add(entry_index)
    free_robots = [index for index in range(len(robots)) if index not in pairs]
    free_entries = [index for index, entry in enumerate(entries) if index not in paired and not entry.name]
    pairs.update(zip(free_robots, free_entries, strict=False))
    return pairs


def idle_robots(robots: Sequence[tuple[str, str]], entries: Sequence[RobotGoal]) -> set[str]:
    """Robots with no task: without a paired entry or on an entry without phases."""
    pairs = pair_robots(robots, entries)
    return {name for index, (name, _) in enumerate(robots) if index not in pairs or not entries[pairs[index]].phases}


class TM_Scenario(TM_Robots):
    _config: ParamHandle[str]
    _linger_after_completion: ParamHandle[bool]
    _idle_robots: set[str]
    _mechanism_map: ParamHandle[bool]

    def robot_goals(self, entries: Sequence[RobotGoal], humans: frozenset[str]) -> list[RobotGoal]:
        """The scenario's robot entries as this mode runs them."""
        del humans
        return list(entries)

    @classmethod
    async def fleet_for(cls, ctx: TaskContext, scenario_file: str) -> Sequence[FleetEntry] | None:
        if not scenario_file:
            return None
        raw = WorldIdentifier(ctx.world_manager.loaded_world).resolve_sync().scenario(scenario_file).resolve_sync().read()
        return declared_fleet(raw.get("robots") or [])

    async def reset(self) -> ResetPlan:
        plan = await super().reset()

        zone_conv = self._ctx.world_manager.world_compacted().zone_converter(self._ctx.conf.RNG.stream("robots", "scenario"))
        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(self._config.value).resolve_sync()
        scenario = scenario_view.load(converter=zone_conv)
        humans = frozenset(human.name for human in scenario.humans)
        SCENARIO_ROBOTS = self.robot_goals(scenario.robots, humans)

        live_robots = list(self._ctx.robots.values())
        pairs = pair_robots([(robot.name, robot.model_name) for robot in live_robots], SCENARIO_ROBOTS)

        self._idle_robots = idle_robots([(robot.name, robot.model_name) for robot in live_robots], SCENARIO_ROBOTS)

        if len(pairs) < len(live_robots) and SCENARIO_ROBOTS:
            self._logger.warn("Robot setup contains more robots than the scenario file.", once=True)

        if len(pairs) < len(SCENARIO_ROBOTS):
            self._logger.warn("Scenario file contains more robots than setup.", once=True)

        managed_robots = [live_robots[index] for index in sorted(pairs)]
        paired_entries = [SCENARIO_ROBOTS[pairs[index]] for index in sorted(pairs)]

        realizer = self._ctx.realizer

        for robot, config in zip(managed_robots, paired_entries, strict=True):
            start_pose = to_map_frame(config.start, config.start_zone, config.level, realizer)
            entry = plan.robots[robot.name] = RobotReset(start_pose=start_pose, humans=humans)

            phases = [localize_phase(phase, config.goal_level, realizer) for phase in config.phases]
            forbidden: list[PositionRadius] = [PositionRadius(x=pose.position.x, y=pose.position.y, radius=robot.safe_distance) for pose in (start_pose, *(goto for phase in phases for goto in goto_poses(phase)))]

            # An empty phase list intentionally describes a stationary robot.
            # Do not submit it to RobotManager: an empty TaskRequest has no
            # dispatchable phase.  Keep the scenario active until its normal
            # timeout (or until another non-idle robot finishes) so that
            # recordings can capture pedestrians passing a parked robot.
            if phases:
                entry.task = TaskRequest(phases=phases, conditions=list(config.conditions))
            plan.forbidden_zones.extend(forbidden)

        agents = [name for robot in managed_robots for name in (robot.name, robot.robot.sim_path) if name]
        self._ctx.world_manager.set_mechanism_map(bool(self._mechanism_map.value), agents)
        return plan

    async def teardown(self) -> None:
        self._ctx.world_manager.set_mechanism_map(False, [])
        await super().teardown()

    @property
    async def done(self) -> bool:
        """Treat empty scenario phase lists as parked, not completed, robots."""
        active_robots = [manager for name, manager in self._ctx.robots.items() if name not in self._idle_robots]
        if not active_robots:
            return False
        if self._linger_after_completion.value:
            return False
        return all(await asyncio.gather(*(manager.is_done for manager in active_robots)))

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Robots.__init__(self, **kwargs)

        params = ScenarioRobotParams(self.node, namespace=self.namespace())
        self._config = params.FILE
        self._linger_after_completion = params.LINGER_AFTER_COMPLETION
        self._mechanism_map = params.MECHANISM_MAP
