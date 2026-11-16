import typing
from collections.abc import Sequence

import attrs
from arena_runtime.env.task_api import GoToPhase
from arena_simulation_setup.tree.World.Scenario import RobotGoal

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.obstacles.scenario import ScenarioParams
from task_generator.tasks.registry import _REGISTRY_NAMESPACE
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.scenario.impl import TM_Scenario
from task_generator.tasks.robots.search import SearchParams

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


def found_until(target: str, radius: float, line_of_sight: bool, dwell_s: float) -> str | None:
    """The `until` a search goto holds on top of arrival, None when arrival alone finds the target."""
    found = f"robot sees {target} within {radius}" if line_of_sight else f"robot within {radius} of {target}"
    if dwell_s > 0.0:
        return f"{found} for {dwell_s}"
    return found if line_of_sight else None


def search_goal(entry: RobotGoal, target: str, radius: float, *, line_of_sight: bool = False, dwell_s: float = 0.0) -> RobotGoal:
    """A scenario robot entry as one judge-only goto on the target, its pose the robot's own start."""
    if entry.phases:
        raise ValueError("a search scenario gives its robots a start only, since their phases would reach the planner")
    phase = GoToPhase(pose=entry.start, target=target, tolerance_radius=radius, pose_zone=entry.start_zone, dispatch=False, until=found_until(target, radius, line_of_sight, dwell_s))
    return attrs.evolve(entry, goal_level=entry.level, phases=[phase])


class TM_Search(TM_Scenario):
    """Scenario robots that look for a hidden human whose pose never reaches the planner."""

    _params: SearchParams

    def robot_goals(self, entries: Sequence[RobotGoal], humans: frozenset[str]) -> list[RobotGoal]:
        target = search_target(sorted(humans), self._params.TARGET.value)
        return [search_goal(entry, target, self._params.RADIUS.value, line_of_sight=self._params.LINE_OF_SIGHT.value, dwell_s=self._params.DWELL_S.value) for entry in entries]

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._params = SearchParams(self.node, namespace=self.namespace())
        self._config = ScenarioParams(self.node, namespace=SCENARIO_NS).FILE
        self._linger_after_completion = self._params.LINGER_AFTER_COMPLETION
        self._mechanism_map = self._params.MECHANISM_MAP
