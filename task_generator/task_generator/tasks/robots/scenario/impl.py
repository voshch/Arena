import asyncio
import typing
from collections.abc import Iterator, Sequence

import attrs
from arena_rclpy_mixins.params import ParamHandle
from arena_runtime.env.task_api import GoToPhase, LoopPhase, Realizer, TaskPhase, TaskRequest
from arena_runtime.env.task_layer import ResetPlan, RobotReset
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


def idle_robots(names: Sequence[str], entries: Sequence[RobotGoal]) -> set[str]:
    """Robots with no task: past the scenario's entries or on an entry without phases."""
    return {name for index, name in enumerate(names) if index >= len(entries) or not entries[index].phases}


class TM_Scenario(TM_Robots):
    _config: ParamHandle[str]
    _linger_after_completion: ParamHandle[bool]
    _idle_robots: set[str]
    _mechanism_map: ParamHandle[bool]

    def robot_goals(self, entries: Sequence[RobotGoal], humans: frozenset[str]) -> list[RobotGoal]:
        """The scenario's robot entries as this mode runs them."""
        del humans
        return list(entries)

    async def reset(self) -> ResetPlan:
        plan = await super().reset()

        zone_conv = self._ctx.world_manager.world_compacted().zone_converter(self._ctx.conf.RNG.stream("robots", "scenario"))
        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(self._config.value).resolve_sync()
        scenario = scenario_view.load(converter=zone_conv)
        humans = frozenset(human.name for human in scenario.humans)
        SCENARIO_ROBOTS = self.robot_goals(scenario.robots, humans)

        managed_robots = list(self._ctx.robots.values())

        scenario_robots_length = len(SCENARIO_ROBOTS)
        setup_robot_length = len(managed_robots)

        self._idle_robots = idle_robots([robot.name for robot in managed_robots], SCENARIO_ROBOTS)

        if setup_robot_length > scenario_robots_length:
            managed_robots = managed_robots[:scenario_robots_length]
            if SCENARIO_ROBOTS:
                self._logger.warn("Robot setup contains more robots than the scenario file.", once=True)

        if scenario_robots_length > setup_robot_length:
            SCENARIO_ROBOTS = SCENARIO_ROBOTS[:setup_robot_length]
            self._logger.warn("Scenario file contains more robots than setup.", once=True)

        realizer = self._ctx.realizer

        for robot, config in zip(managed_robots, SCENARIO_ROBOTS, strict=False):
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
