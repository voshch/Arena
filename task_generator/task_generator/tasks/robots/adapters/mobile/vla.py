"""VLA planner adapter: a bridge planner conditioned on a goal pose, a language instruction, or both."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import attrs
from arena_robots.Sensor import SensorType
from arena_simulation_setup.shared.route import WORDINGS, instruct

from task_generator.manager.world_manager.world_manager import WORLD_ENTITY_PREFIX
from task_generator.tasks.robots.adapters.mobile.drl import DrlAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from arena_simulation_setup.tree.World import LevelDescription
    from rclpy.impl.rcutils_logger import RcutilsLogger

    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.shared import Pose
    from task_generator.tasks.robots.request import GoToPhase


def localized_instruction(phase: GoToPhase, here: Pose | None, ezilear: Callable[[Pose], Pose], level: LevelDescription, wording: str = "", fallback: str = "") -> dict[str, str]:
    """Instruction for the env-frame `phase` worded from env-frame pose `here`, both taken into `level`'s frame by `ezilear`, `fallback` standing in for missing phase text."""
    local = attrs.evolve(phase, text=phase.text or fallback, pose=None if phase.pose is None else ezilear(phase.pose))
    return instruct(local, level, None if here is None else ezilear(here).to_2d(), wording, WORLD_ENTITY_PREFIX)


class VlaAdapter(DrlAdapter):
    kind: ClassVar[str] = "vla"

    def __init__(
        self,
        robot_manager: RobotManager,
        *,
        goal: str = "",
        instruction: str = "",
        wording: str = "",
        **kwargs: object,
    ) -> None:
        self._goal = str(goal)
        self._fallback_instruction = str(instruction)
        self._wording = str(wording)
        if self._wording not in ("", *WORDINGS):
            raise ValueError(f"VlaAdapter: robot.mobile.wording {self._wording!r} unknown, expected 'goal' (name the target) or 'route' (walking directions), unset means route for a bare pose and goal for a named target")
        self._revealed: frozenset[str] = frozenset()
        self._instruction: dict[str, str] | None = None
        super().__init__(robot_manager, **kwargs)

    @property
    def goal_inputs(self) -> tuple[str, ...]:
        from arena_planners import goal  # noqa: PLC0415

        return tuple(name for name in goal.GOAL_INPUTS if name in self._revealed)

    @property
    def instruction(self) -> dict[str, str] | None:
        return self._instruction

    def _adapt_manifest(self, manifest: dict) -> dict:
        from arena_planners import goal  # noqa: PLC0415
        from arena_planners.resolver import ResolverError  # noqa: PLC0415

        accepted = goal.declared(manifest)
        if accepted is None:
            raise ResolverError(f"VlaAdapter: planner {self._planner_name!r} declares no goal_inputs, run it with robot.mobile:=drl")
        try:
            self._revealed = goal.reveal(self._goal, accepted)
        except ValueError as exc:
            raise ValueError(f"VlaAdapter: robot.mobile.{exc}") from exc
        if "pose" in self._revealed:
            return manifest
        return goal.withhold_pose(manifest)

    def _instruct(self, phase: GoToPhase) -> dict[str, str]:
        """Instruction for the dispatched `phase`, worded from the robot's current pose."""
        return localized_instruction(phase, self.rm.pose, self.rm._environment_manager.ezilear, self.rm.node._world_manager.world_compacted(), self._wording, self._fallback_instruction)  # pylint: disable=protected-access

    def _initial_state(self, phase: GoToPhase) -> dict:
        from arena_planners import goal  # noqa: PLC0415

        self._instruction = self._instruct(phase) if "instruction" in self._revealed else None
        x, y, theta = phase.pose.to_2d()
        return goal.initial_state({"x": x, "y": y, "theta": theta}, self._instruction["text"] if self._instruction else "", self._revealed)

    def _bind_sensor_topics(self, sensors: list, namespace: str, logger: RcutilsLogger) -> dict:
        if not any(spec.type == SensorType.IMAGE for spec in sensors):
            raise RuntimeError(f"VlaAdapter: planner {self._planner_name!r} needs an image sensor, robot sensors are {sorted({str(s.type) for s in sensors})}")
        return super()._bind_sensor_topics(sensors, namespace, logger)


__all__ = ["VlaAdapter", "localized_instruction"]
