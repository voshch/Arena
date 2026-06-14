import math

from arena_rclpy_mixins.declarations import declare_string
from arena_rclpy_mixins.ROSParamServer import ROSParamT

from task_generator.shared import Orientation, Pose
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import TaskRequest, VLAPhase

_INSTRUCTION_DEFAULT = "go to a corner and stay there"


class TM_VLA(TM_Robots):
    """Thin VLA task mode: feeds the instruction param to each robot as a VLAPhase. Inference,
    waypoint following, and visualization live in the robot's `vla` mobile adapter."""

    _instruction_p: ROSParamT[str]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        declare_string(self.node, str(self.namespace("instruction")), _INSTRUCTION_DEFAULT, label="Instruction", description="Language instruction sent to the VLA.")
        self._instruction_p = self.node.ROSParam[str](self.namespace("instruction"))

    async def reset(self, **kwargs: object) -> None:
        await super().reset(**kwargs)

        instruction = self._instruction_p.value

        biggest_robot = max((r.safe_distance for r in self._ctx.robots.values()), default=0.5)
        n = len(self._ctx.robots)
        positions = self._ctx.world_manager.get_positions_on_map(n=n, safe_dist=biggest_robot)
        orientations = 2 * math.pi * self.node.conf.General.RNG.value.random(n)
        for (name, _robot), pos, ori in zip(self._ctx.robots.items(), positions, orientations, strict=False):
            self._start_poses[name] = Pose(pos, Orientation.from_yaw(ori))

        for name, robot in self._ctx.robots.items():
            if robot.mobile_adapter_kind != "vla":
                raise RuntimeError(f"TM_VLA requires the 'vla' mobile adapter; robot {name!r} is on {robot.mobile_adapter_kind!r}. Leave mobile:= unset (autoselected) or set mobile:=vla.")
            await robot.submit_task(TaskRequest(phases=[VLAPhase(instruction=instruction)]))
