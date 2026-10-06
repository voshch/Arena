"""VLA planner adapter: a bridge planner conditioned on a goal pose, a language instruction, or both."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arena_robots.Sensor import SensorType

from task_generator.tasks.robots.adapters.mobile.drl import DrlAdapter

if TYPE_CHECKING:
    from rclpy.impl.rcutils_logger import RcutilsLogger

    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.tasks.robots.request import GoToPhase


class VlaAdapter(DrlAdapter):
    kind: ClassVar[str] = "vla"

    def __init__(
        self,
        robot_manager: RobotManager,
        *,
        goal: str = "",
        instruction: str = "",
        **kwargs: object,
    ) -> None:
        self._goal = str(goal)
        self._fallback_instruction = str(instruction)
        self._revealed: frozenset[str] = frozenset()
        super().__init__(robot_manager, **kwargs)

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

    def _initial_state(self, phase: GoToPhase) -> dict:
        from arena_planners import goal  # noqa: PLC0415
        from arena_simulation_setup.shared.render import render_phase  # noqa: PLC0415

        instruction = phase.text or self._fallback_instruction or render_phase(phase)
        x, y, theta = phase.pose.to_2d()
        return goal.initial_state({"x": x, "y": y, "theta": theta}, instruction, self._revealed)

    def _bind_sensor_topics(self, sensors: list, namespace: str, logger: RcutilsLogger) -> dict:
        if not any(spec.type == SensorType.IMAGE for spec in sensors):
            raise RuntimeError(f"VlaAdapter: planner {self._planner_name!r} needs an image sensor, robot sensors are {sorted({str(s.type) for s in sensors})}")
        return super()._bind_sensor_topics(sensors, namespace, logger)


__all__ = ["VlaAdapter"]
