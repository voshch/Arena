import numpy as np
import rclpy
import rclpy.node
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_runtime.constants import SimSimulator

from . import Constants
from .rng import EpisodeRng

EPISODE_PARAMS: dict[str, str] = {
    'task.episode.count': 'Stop the env after N episodes (-1 = run forever).',
    'task.episode.auto_reset': 'true = standalone: node auto-advances episodes. false = managed: external controller drives resets via lifecycle/reset_episode.',
    'task.episode.fail_on_collision': 'true = abort the episode (FAILED) when the robot footprint contacts a wall, static obstacle, or pedestrian.',
    'task.episode.timeout': 'Episode time limit in sim seconds (-1 = none).',
    'task.episode.timeout.robot_ready': 'Seconds to wait for robot adapters to become ready (-1 = unbounded).',
    'task.episode.reset.max_fails': 'Consecutive reset failures before the env gives up.',
    'task.episode.spawn.robot_clearance': 'Clearance in metres added to the robot radius when placing robot spawns and goals.',
    'task.episode.spawn.obstacle_clearance': 'Clearance in metres around scenario obstacle placements.',
    'task.episode.spawn.obstacle_max_radius': 'Largest obstacle radius in metres (-1 = unbounded).',
    'task.episode.goto_pose.tolerance.radius': 'Goal position tolerance in metres for goto_pose phases.',
    'task.episode.goto_pose.tolerance.angle': 'Goal heading tolerance in radians for goto_pose phases.',
    'task.episode.goto_pose.timeout.no_progress': 'Fail a goto_pose episode after this many sim seconds without goal progress (-1 = off).',
    'task.episode.goto_pose.hold_time': 'Sim seconds a robot must park at a goto_pose goal before the phase counts as met (0 = arrival suffices).',
    'task.episode.goto_pose.signal': 'Signal the robot must send to end a goto_pose phase, such as arrived (empty = Arena judges arrival).',
}

DEPRECATED_PARAMS: dict[str, str] = {
    'episodes': 'task.episode.count',
    'auto_reset': 'task.episode.auto_reset',
    'fail_on_collision': 'task.episode.fail_on_collision',
    'timeout': 'task.episode.timeout',
    'robot.ready_timeout': 'task.episode.timeout.robot_ready',
    'max_reset_fail_times': 'task.episode.reset.max_fails',
    'robot_safe_dist': 'task.episode.spawn.robot_clearance',
    'obstacle_safe_dist': 'task.episode.spawn.obstacle_clearance',
    'obstacle_max_radius': 'task.episode.spawn.obstacle_max_radius',
    'goal_tolerance_radius': 'task.episode.goto_pose.tolerance.radius',
    'goal_tolerance_angle': 'task.episode.goto_pose.tolerance.angle',
    'no_progress_timeout': 'task.episode.goto_pose.timeout.no_progress',
    'tm_robots': 'task.robots',
    'tm_obstacles': 'task.obstacles',
    'tm_config': 'task.config',
    'tm_modules': 'task.modules',
    'train_mode': 'robot.train',
    'static_sounds': 'auditory.static_sounds',
}


def migrate_deprecated_params(node: rclpy.node.Node) -> list[str]:
    """Copy each deprecated parameter given at startup onto its replacement unless that is set too, warn, return the deprecated names found."""
    found = [old for old in DEPRECATED_PARAMS if node.has_parameter(old)]
    for old in found:
        new = DEPRECATED_PARAMS[old]
        if not node.has_parameter(new):
            node.declare_parameter(new, node.get_parameter(old).value)
        node.get_logger().warning(f"parameter {old!r} is deprecated, use {new!r}")
    return found


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

            ACOUSTICS = server.ROSParam[Constants.AcousticsSimulator]('acoustics', Constants.AcousticsSimulator.NONE.value, parse=Constants.AcousticsSimulator)

            WORLD = server.ROSParam[str](
                'world',
                type_=rclpy.Parameter.Type.STRING,
            )

        class General:
            """
            General Task Configuration
            """

            MAX_RESET_FAIL_TIMES = server.ROSParam[int](
                'task.episode.reset.max_fails',
                10,
            )

            RNG = EpisodeRng()

            DESIRED_EPISODES = server.ROSParam[float](
                'task.episode.count',
                -1,
                parse=_positive_or_inf,
            )

        class Obstacles:
            OBSTACLE_MAX_RADIUS = server.ROSParam[float](
                'task.episode.spawn.obstacle_max_radius',
                15,
                parse=_positive_or_inf,
            )

            SAFE_DIST = server.ROSParam[float](
                'task.episode.spawn.obstacle_clearance',
                0.35,
            )

        class Robot:
            GOAL_TOLERANCE_RADIUS = server.ROSParam[float]('task.episode.goto_pose.tolerance.radius', 1.0)

            GOAL_TOLERANCE_ANGLE = server.ROSParam[float](
                'task.episode.goto_pose.tolerance.angle',
                30.0 * np.pi / 180.0,
            )

            GOAL_HOLD_TIME = server.ROSParam[float]('task.episode.goto_pose.hold_time', 0.0)

            GOAL_SIGNAL = server.ROSParam[str]('task.episode.goto_pose.signal', '')

            SPAWN_ROBOT_SAFE_DIST = server.ROSParam[float](
                'task.episode.spawn.robot_clearance',
                0.25,
            )

            TIMEOUT = server.ROSParam[float](
                'task.episode.timeout',
                -1,
                parse=_positive_or_inf,
            )

            NO_PROGRESS_TIMEOUT = server.ROSParam[float](
                'task.episode.goto_pose.timeout.no_progress',
                -1,
                parse=_positive_or_inf,
            )

            READY_TIMEOUT = server.ROSParam[float](
                'task.episode.timeout.robot_ready',
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

            HEARING = server.ROSParam[str](
                'robot.hearing',
                'none',
            )

        class TaskMode:
            TM_ROBOTS = server.ROSParam[Constants.TaskMode.TM_Robots](
                'task.robots',
                Constants.TaskMode.TM_Robots.default().value,
                parse=Constants.TaskMode.TM_Robots,
            )

            TM_OBSTACLES = server.ROSParam[Constants.TaskMode.TM_Obstacles](
                'task.obstacles',
                Constants.TaskMode.TM_Obstacles.default().value,
                parse=Constants.TaskMode.TM_Obstacles,
            )

            TM_CONFIG = server.ROSParam[str]('task.config', '')

            TM_MODULES = server.ROSParam[set[Constants.TaskMode.TM_Module]]('task.modules', ','.join([m.value for m in Constants.TaskMode.TM_Module.default()]), parse=lambda x: {Constants.TaskMode.TM_Module(m) for m in x.split(',') if m != ''})

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
