import enum
import os
import typing

import attrs
import geometry_msgs.msg
import hunav_msgs.msg
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_simulation_setup.tree.assets.Human import HumanIdentifier

from task_generator.shared import DynamicObstacle, Pose, Position


@attrs.define
class PositionH(Position):
    h: float = attrs.field(default=0.0, converter=float)


class Goals(list[Position]):
    @classmethod
    def parse(cls, obj: dict) -> "Goals":
        waypoints = [
            Position(
                x=waypoint.get('x', 0.0),
                y=waypoint.get('y', 0.0),
                z=waypoint.get('z', 0.0),
            )
            for waypoint in (obj.get(wpname) for wpname in obj['goals'])
            if waypoint is not None
        ]
        return cls(waypoints)

    def as_poses(self) -> list[geometry_msgs.msg.Pose]:
        return [Pose(p).to_msg() for p in self]


@attrs.define()
class HunavDynamicObstacle:
    @attrs.define()
    class Behavior:
        type: int
        state: int
        configuration: int
        duration: float = attrs.field(converter=float)
        vel: float = attrs.field(converter=float)
        dist: float = attrs.field(converter=float)
        social_force_factor: float = attrs.field(converter=float)
        goal_force_factor: float = attrs.field(converter=float)
        obstacle_force_factor: float = attrs.field(converter=float)
        other_force_factor: float = attrs.field(converter=float)
        once: bool = False

        _default: typing.ClassVar["HunavDynamicObstacle.Behavior"]

        @classmethod
        def parse(cls, obj: dict) -> "HunavDynamicObstacle.Behavior":
            return cls(
                type=obj.get('type', cls._default.type),
                state=obj.get('state', cls._default.state),
                configuration=obj.get('configuration', cls._default.configuration),
                duration=obj.get('duration', cls._default.duration),
                once=obj.get('once', cls._default.once),
                vel=obj.get('vel', cls._default.vel),
                dist=obj.get('dist', cls._default.dist),
                social_force_factor=obj.get('social_force_factor', cls._default.social_force_factor),
                goal_force_factor=obj.get('goal_force_factor', cls._default.goal_force_factor),
                obstacle_force_factor=obj.get('obstacle_force_factor', cls._default.obstacle_force_factor),
                other_force_factor=obj.get('other_force_factor', cls._default.other_force_factor),
            )

        def to_msg(self) -> hunav_msgs.msg.AgentBehavior:
            behavior_msg = hunav_msgs.msg.AgentBehavior()
            behavior_msg.type = self.type
            behavior_msg.configuration = self.configuration
            behavior_msg.duration = self.duration
            behavior_msg.once = self.once
            behavior_msg.vel = self.vel
            behavior_msg.dist = self.dist
            behavior_msg.goal_force_factor = self.goal_force_factor
            behavior_msg.obstacle_force_factor = self.obstacle_force_factor
            behavior_msg.social_force_factor = self.social_force_factor
            behavior_msg.other_force_factor = self.other_force_factor
            return behavior_msg

    id: int
    type: int = attrs.field(converter=lambda v: v if isinstance(v, int) else 1)
    skin: int
    name: str
    group_id: int
    init_pose: PositionH
    yaw: float
    model: HumanIdentifier
    goals: Goals
    extra: dict
    velocity: None
    desired_velocity: float
    radius: float
    linear_vel: float
    angular_vel: float

    behavior: Behavior
    behavior_tree: str

    cyclic_goals: bool
    goal_radius: float
    closest_obs: list

    _default: typing.ClassVar["HunavDynamicObstacle"]

    @classmethod
    def from_dynamic_obstacle(cls, obj: DynamicObstacle, extra: dict | None = None) -> "HunavDynamicObstacle":
        if extra is None:
            extra = {}
        extra = {**obj.extra, **extra}

        if 'goals' in extra:
            waypoints = Goals.parse(extra)
        else:
            waypoints = Goals(
                [
                    Position(
                        x=waypoint.x,
                        y=waypoint.y,
                    )
                    for waypoint in obj.waypoints
                ]
            )

        if 'behavior' in extra:
            behavior = cls.Behavior.parse(extra['behavior'])
        else:
            behavior = cls.Behavior._default

        behavior_tree: str = cls._default.behavior_tree
        if 'behavior_tree' in extra:
            behavior_tree = extra['behavior_tree']
            if behavior_tree.startswith('./') and obj.included_from:
                behavior_tree = str(obj.included_from / behavior_tree)

        return cls(
            name=obj.sim_path,
            init_pose=PositionH(
                x=extra.get('position', {}).get('x', obj.pose.position.x),
                y=extra.get('position', {}).get('y', obj.pose.position.y),
                z=extra.get('position', {}).get('z', cls._default.init_pose.z),
                h=extra.get('position', {}).get('h', cls._default.init_pose.h),
            ),
            yaw=0.0,
            model=obj.model,
            goals=waypoints,
            velocity=extra.get('velocity', cls._default.velocity),
            desired_velocity=extra.get('desired_velocity', cls._default.desired_velocity),
            radius=extra.get('radius', cls._default.radius),
            linear_vel=extra.get('linear_vel', cls._default.linear_vel),
            angular_vel=extra.get('angular_vel', cls._default.angular_vel),
            behavior=behavior,
            behavior_tree=behavior_tree,
            cyclic_goals=extra.get('cyclic_goals', cls._default.cyclic_goals),
            goal_radius=extra.get('goal_radius', cls._default.goal_radius),
            closest_obs=[],
            extra=extra,
            id=extra.get('id', cls._default.id),
            type=extra.get('type', cls._default.type),
            skin=extra.get('skin', cls._default.skin),
            group_id=extra.get('group_id', cls._default.group_id),
        )

    def to_msg(self) -> hunav_msgs.msg.Agent:
        agent_msg = hunav_msgs.msg.Agent()
        agent_msg.id = self.id
        agent_msg.name = self.name
        agent_msg.type = self.type
        agent_msg.skin = self.skin
        agent_msg.group_id = self.group_id
        agent_msg.desired_velocity = self.desired_velocity
        # self._logger.info(f"=== spawn_dynamic_obstacles_desired_velocity: {agent_msg.desired_velocity}===")
        agent_msg.radius = self.radius

        # Set position
        agent_msg.position = geometry_msgs.msg.Pose()
        agent_msg.position.position.x = self.init_pose.x
        agent_msg.position.position.y = self.init_pose.y
        agent_msg.position.position.z = 1.250000
        agent_msg.yaw = self.yaw

        # Set behavior
        agent_msg.behavior = self.behavior.to_msg()
        agent_msg.behavior_tree = self.behavior_tree

        # Set goals
        agent_msg.goal_radius = self.goal_radius
        agent_msg.cyclic_goals = self.cyclic_goals
        if self.goals:
            agent_msg.goals = self.goals.as_poses()
        else:
            # Default goals if none exist
            goals = [(-3.133759, -4.166653, 1.250000), (0.997901, -4.131655, 1.250000), (-0.227549, -20.187146, 1.250000)]
            agent_msg.goals = [geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=x, y=y)) for x, y, _ in goals]

        return agent_msg

    @classmethod
    def parse(cls, obj: dict) -> "HunavDynamicObstacle":

        if 'goals' in obj:
            waypoints = Goals.parse(obj)
        else:
            waypoints = cls._default.goals

        return cls(
            name=obj.get('name', cls._default.name),
            extra=obj.get('extra', cls._default.extra),
            model=obj.get('model', cls._default.model),
            goals=waypoints,
            init_pose=PositionH(
                x=obj.get('init_pose', {}).get('x', cls._default.init_pose.x),
                y=obj.get('init_pose', {}).get('y', cls._default.init_pose.y),
                z=obj.get('init_pose', {}).get('z', cls._default.init_pose.z),
                h=obj.get('init_pose', {}).get('h', cls._default.init_pose.h),
            ),
            yaw=obj.get('yaw', cls._default.yaw),
            id=obj.get("id", cls._default.id),
            behavior=cls.Behavior.parse(obj.get('behavior', {})),
            behavior_tree=obj.get('behavior_tree', cls._default.behavior_tree),
            type=obj.get('type', cls._default.type),
            skin=obj.get('skin', cls._default.skin),
            group_id=obj.get('group_id', cls._default.group_id),
            velocity=None,
            desired_velocity=obj.get('max_vel', cls._default.desired_velocity),
            radius=obj.get('radius', cls._default.radius),
            linear_vel=cls._default.linear_vel,
            angular_vel=cls._default.angular_vel,
            cyclic_goals=obj.get('cyclic_goals', cls._default.cyclic_goals),
            goal_radius=obj.get('goal_radius', cls._default.goal_radius),
            closest_obs=[],
        )


def _load_config(filename: str = "default.yaml") -> "HunavDynamicObstacle":
    """Load config from YAML file in arena_simulation_setup configs."""

    config_path = os.path.join(get_package_share_directory("arena_simulation_setup"), "configs", "hunav", filename)

    try:
        with open(config_path) as f:
            agent_config = yaml.safe_load(f)

        assert isinstance(agent_config, dict), f"Top-level structure in {config_path} must be a mapping"

        return HunavDynamicObstacle.parse(agent_config)

    except Exception as e:
        raise RuntimeError(f"Error loading config from {config_path}") from e


HunavDynamicObstacle.Behavior._default = HunavDynamicObstacle.Behavior(
    type=0,
    state=0,
    configuration=0,
    duration=0,
    once=False,
    vel=0.0,
    dist=0.0,
    social_force_factor=0.0,
    goal_force_factor=0.0,
    obstacle_force_factor=0.0,
    other_force_factor=0.0,
)

HunavDynamicObstacle._default = HunavDynamicObstacle(
    init_pose=PositionH(x=0, y=0),
    name='',
    model=HumanIdentifier(''),
    extra={},
    goals=Goals(),
    id=0,
    type=1,
    skin=0,
    group_id=0,
    yaw=0.0,
    velocity=None,
    desired_velocity=0.0,
    radius=0.0,
    linear_vel=0.0,
    angular_vel=0.0,
    behavior=HunavDynamicObstacle.Behavior._default,
    behavior_tree='default.xml',
    cyclic_goals=False,
    goal_radius=0.0,
    closest_obs=[],
)


HunavDynamicObstacle._default = _load_config()
HunavDynamicObstacle.Behavior._default = HunavDynamicObstacle._default.behavior


# Animation configuration (from WorldGenerator)
SKIN_TYPES: dict[int, str] = {0: 'elegant_man.dae', 1: 'casual_man.dae', 2: 'elegant_woman.dae', 3: 'regular_man.dae', 4: 'worker_man.dae', 5: 'walk.dae'}


class ANIMATION_TYPES(enum.StrEnum):
    WALK = ('07_01-walk.bvh',)
    WALK_FORWARD = ('69_02_walk_forward.bvh',)
    NORMAL_WAIT = ('137_28-normal_wait.bvh',)
    WALK_CHILDISH = ('142_01-walk_childist.bvh',)
    SLOW_WALK = ('07_04-slow_walk.bvh',)
    WALK_SCARED = ('142_17-walk_scared.bvh',)
    WALK_ANGRY = '17_01-walk_with_anger.bvh'
