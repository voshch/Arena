import typing
from collections.abc import Sequence

import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_runtime.env.task_api import GoToPhase
from arena_simulation_setup.tree.World.Scenario import RobotGoal

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, default_scenario
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.scenario.impl import TM_Scenario

SCENARIO_NS = _REGISTRY_NAMESPACE("scenario")


def search_target(humans: Sequence[str], configured: str) -> str:
    """The human to find: the configured one, else the scenario's only human."""
    if configured:
        if configured not in humans:
            raise ValueError(f"search target '{configured}' is not a human of the scenario, expected one of {sorted(humans)}")
        return configured
    if len(humans) != 1:
        raise ValueError(f"search needs task.search.target to pick one of {len(humans)} scenario humans {sorted(humans)}")
    return humans[0]


def search_goal(entry: RobotGoal, target: str, radius: float) -> RobotGoal:
    """A scenario robot entry as one judge-only goto on the target, its pose the robot's own start."""
    if entry.phases:
        raise ValueError("a search scenario gives its robots a start only, since their phases would reach the planner")
    phase = GoToPhase(pose=entry.start, target=target, tolerance_radius=radius, pose_zone=entry.start_zone, dispatch=False)
    return attrs.evolve(entry, goal_level=entry.level, phases=[phase])


class TM_Search(TM_Scenario):
    """Scenario robots that look for a hidden human whose pose never reaches the planner."""

    _target: ROSParamT[str]
    _radius: ROSParamT[float]

    def robot_goals(self, entries: Sequence[RobotGoal], humans: frozenset[str]) -> list[RobotGoal]:
        target = search_target(sorted(humans), self._target.value)
        return [search_goal(entry, target, self._radius.value) for entry in entries]

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._config = self.node.ROSParam[str](SCENARIO_NS("file"), default_scenario(self._ctx.world_manager.loaded_world))
        self._linger_after_completion = self.node.ROSParam[bool](self.namespace("linger_after_completion"), False)
        self._mechanism_map = self.node.ROSParam[bool](self.namespace("mechanism_map"), False)
        self._target = self.node.ROSParam[str](self.namespace("target"), "")
        self._radius = self.node.ROSParam[float](self.namespace("radius"), 1.5)
