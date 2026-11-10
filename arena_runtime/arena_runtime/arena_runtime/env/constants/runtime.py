import numpy as np
import rclpy
import rclpy.node
from arena_bringup.constants import SimSimulator
from arena_rclpy_mixins.params import Choices, Param, ParamGroup, Range
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_simulation_setup import params as world_params

from arena_runtime import hearing
from arena_runtime.env.task_layer import legacy_mode_params, mode_param

from . import Constants
from .rng import EpisodeRng

RUN_EPISODE_BUSY = 'busy'
RUN_EPISODE_SHUTTING_DOWN = 'shutting down'

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
    **legacy_mode_params(),
    'tm_config': mode_param('config'),
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


def _positive_or_inf(v: object) -> float:
    if not isinstance(v, (int, float)):
        raise TypeError(f'expected a number, got {v!r}')
    return v if v >= 0 else float('inf')


class ArenaParams(ParamGroup):
    SIM = Param[SimSimulator]('sim', SimSimulator.DUMMY, 'Simulator backend the runtime drives.')
    HUMAN = Param[Constants.HumanSimulator]('human', Constants.HumanSimulator.DUMMY, 'Pedestrian simulator of the env.')
    ACOUSTICS = Param[Constants.AcousticsSimulator]('acoustics', Constants.AcousticsSimulator.NONE, 'Acoustics backend of the env.')
    WORLD = Param[str]('world', None, 'World the env starts in.', type_=rclpy.Parameter.Type.STRING)
    WORLD_LIGHTING = world_params.WORLD_LIGHTING
    WORLD_LEVELS = world_params.WORLD_LEVELS
    KG_DERIVE = world_params.KG_DERIVE


class GeneralParams(ParamGroup):
    """General Task Configuration"""

    MAX_RESET_FAIL_TIMES = Param[int]('task.episode.reset.max_fails', 10, 'Consecutive reset failures before the env gives up.')
    RNG = EpisodeRng()
    DESIRED_EPISODES = Param[float]('task.episode.count', -1.0, 'Stop the env after N episodes (-1 = run forever).', parse=_positive_or_inf)
    LOG_LEVEL = Param[str]('log_level', '', 'Log level rule set applied to the env and its robot nodes (empty = leave unchanged).')


class ObstaclesParams(ParamGroup):
    OBSTACLE_MAX_RADIUS = Param[float]('task.episode.spawn.obstacle_max_radius', 15.0, 'Largest obstacle radius in metres (-1 = unbounded).', parse=_positive_or_inf)
    SAFE_DIST = Param[float]('task.episode.spawn.obstacle_clearance', 0.35, 'Clearance in metres around scenario obstacle placements.')


class RobotParams(ParamGroup):
    GOAL_TOLERANCE_RADIUS = Param[float]('task.episode.goto_pose.tolerance.radius', 1.0, 'Goal position tolerance in metres for goto_pose phases.')
    GOAL_TOLERANCE_ANGLE = Param[float]('task.episode.goto_pose.tolerance.angle', 30.0 * np.pi / 180.0, 'Goal heading tolerance in radians for goto_pose phases.')
    GOAL_HOLD_TIME = Param[float]('task.episode.goto_pose.hold_time', 0.0, 'Sim seconds a robot must park at a goto_pose goal before the phase counts as met (0 = arrival suffices).')
    GOAL_SIGNAL = Param[str]('task.episode.goto_pose.signal', '', 'Signal the robot must send to end a goto_pose phase, such as arrived (empty = Arena judges arrival).')
    NODE_PING_PERIOD = Param[float]('task.episode.node_ping.period', 10.0, 'Wall seconds between pings of the nodes of this env, also the time a node has to answer.')
    NODE_PING_MISSES = Param[int]('task.episode.node_ping.misses', 3, 'Pings in a row a node may miss before it counts as hung, 0 turns the check off.')
    SPAWN_ROBOT_SAFE_DIST = Param[float]('task.episode.spawn.robot_clearance', 0.25, 'Clearance in metres added to the robot radius when placing robot spawns and goals.')
    TIMEOUT = Param[float]('task.episode.timeout', -1.0, 'Episode time limit in sim seconds (-1 = none).', parse=_positive_or_inf)
    NO_PROGRESS_TIMEOUT = Param[float]('task.episode.goto_pose.timeout.no_progress', -1.0, 'Fail a goto_pose episode after this many sim seconds without goal progress (-1 = off).', parse=_positive_or_inf)
    READY_TIMEOUT = Param[float]('task.episode.timeout.robot_ready', -1.0, 'Seconds to wait for robot adapters to become ready (-1 = unbounded).', parse=_positive_or_inf)
    MOBILE_ADAPTER = Param[str]('robot.mobile_adapter', 'nav2', 'Default mobile-cap adapter kind of the fleet robots.')
    ARM_ADAPTER = Param[str]('robot.arm_adapter', 'moveit', 'Default arm-cap adapter kind of the fleet robots.')
    LIFT_ADAPTER = Param[str]('robot.lift_adapter', 'direct', 'Default lift-cap adapter kind of the fleet robots.')
    GRIPPER_ADAPTER = Param[str]('robot.gripper_adapter', 'direct', 'Default gripper-cap adapter kind of the fleet robots.')
    HEARING = Param[str]('robot.hearing', hearing.NONE, 'Robot hearing front-end of every fleet robot.', constraint=Choices(hearing.NONE, *hearing.FRONTENDS))


class HumanParams(ParamGroup):
    ROBOTS_VISIBLE = Param[bool]('human.robots.visible', True, 'false = the human backend leaves the robots out of what it tells its pedestrians.')


class EpisodeParams(ParamGroup):
    AUTO_RESET = Param[bool]('task.episode.auto_reset', True, 'true = standalone: node auto-advances episodes. false = managed: external controller drives resets via lifecycle/reset_episode.')
    FAIL_ON_COLLISION = Param[bool]('task.episode.fail_on_collision', False, 'true = abort the episode (FAILED) when the robot footprint contacts a wall, static obstacle, or pedestrian.')
    RUN_SEED = Param[str]('run_seed', '', 'Hex seed of the episode RNG (empty = random at startup).')
    QUEUE_DEPTH = Param[int]('episode_queue_depth', 10, 'Publisher depth of the state/queue topic.', constraint=Range(1, 100))


_TASK_EPISODE_GROUPS = (GeneralParams, ObstaclesParams, RobotParams, EpisodeParams)

EPISODE_PARAMS: dict[str, Param[object]] = {name: param for group in _TASK_EPISODE_GROUPS for name, param in group.by_name().items() if name.startswith('task.episode.')}


class Config:
    """Combined Task Config"""

    def __init__(self, server: ROSParamServer) -> None:
        self.Arena = ArenaParams(server)
        self.General = GeneralParams(server)
        self.Obstacles = ObstaclesParams(server)
        self.Robot = RobotParams(server)
        self.Human = HumanParams(server)


def Configuration(server: ROSParamServer) -> Config:
    return Config(server)
