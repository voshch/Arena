import numpy as np
import rclpy
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_runtime.constants import SimSimulator

from . import Constants
from .rng import EpisodeRng


def Configuration(server: ROSParamServer) -> type:

    def _positive_or_inf(v: float) -> float:
        return v if v >= 0 else float('inf')

    class Config:
        """
        Combined Task Config
        """

        class Arena:
            SIM = server.ROSParam[SimSimulator]('sim', SimSimulator.DUMMY.value, parse=SimSimulator)

            HUMAN = server.ROSParam[Constants.HumanSimulator]('human', Constants.HumanSimulator.DUMMY.value, parse=Constants.HumanSimulator)

            WORLD = server.ROSParam[str](
                'world',
                type_=rclpy.Parameter.Type.STRING,
            )

        class General:
            """
            General Task Configuration
            """

            WAIT_FOR_SERVICE_TIMEOUT = server.ROSParam[float](
                'timeout_wait_for_service',
                30,
            )

            MAX_RESET_FAIL_TIMES = server.ROSParam[int](
                'max_reset_fail_times',
                10,
            )

            RNG = EpisodeRng()

            DESIRED_EPISODES = server.ROSParam[float](
                'episodes',
                -1,
                parse=_positive_or_inf,
            )

        class Obstacles:
            OBSTACLE_MAX_RADIUS = server.ROSParam[float](
                'obstacle_max_radius',
                15,
                parse=_positive_or_inf,
            )

            SAFE_DIST = server.ROSParam[float](
                'obstacle_safe_dist',
                0.35,
            )

        class Robot:
            GOAL_TOLERANCE_RADIUS = server.ROSParam[float]('goal_tolerance_radius', 1.0)

            GOAL_TOLERANCE_ANGLE = server.ROSParam[float](
                'goal_tolerance_angle',
                30.0 * np.pi / 180.0,
            )

            SPAWN_ROBOT_SAFE_DIST = server.ROSParam[float](
                'robot_safe_dist',
                0.25,
            )

            TIMEOUT = server.ROSParam[float](
                'timeout',
                -1,
                parse=_positive_or_inf,
            )

            READY_TIMEOUT = server.ROSParam[float](
                'robot.ready_timeout',
                -1,
                parse=_positive_or_inf,
            )

            MOBILE_ADAPTER = server.ROSParam[str](
                'robot.mobile_adapter',
                'nav2',
            )

            ARM_ADAPTER = server.ROSParam[str](
                'robot.arm_adapter',
                'moveit',
            )

        class TaskMode:
            TM_ROBOTS = server.ROSParam[Constants.TaskMode.TM_Robots](
                'tm_robots',
                Constants.TaskMode.TM_Robots.default().value,
                parse=Constants.TaskMode.TM_Robots,
            )

            TM_OBSTACLES = server.ROSParam[Constants.TaskMode.TM_Obstacles](
                'tm_obstacles',
                Constants.TaskMode.TM_Obstacles.default().value,
                parse=Constants.TaskMode.TM_Obstacles,
            )

            TM_CONFIG = server.ROSParam[str]('tm_config', '')

            TM_MODULES = server.ROSParam[set[Constants.TaskMode.TM_Module]]('tm_modules', ','.join([m.value for m in Constants.TaskMode.TM_Module.default()]), parse=lambda x: {Constants.TaskMode.TM_Module(m) for m in x.split(',') if m != ''})

    return Config


# def lp(parameter: str, fallback: Any) -> Callable[[Optional[Any]], Any]:
#     """
#     load parameter
#     """
#     val = fallback

#     def gen():
#         return val

#     if isinstance(val, list):
#         lo, hi = val[:2]

#         def new_gen():
#             return min(
#                 hi,
#                 max(
#                     lo,
#                     Config.General.RNG.normal((hi + lo) / 2, (hi - lo) / 6)
#                 )
#             )
#         gen = new_gen

#     return lambda x: x if x is not None else gen()
