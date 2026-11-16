import math
import pathlib
import typing
from collections.abc import Callable

import rclpy
import rclpy.action
import rclpy.action.client
import rclpy.publisher
import rclpy.qos
import rclpy.task
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins import ArenaMixinNode
from arena_robots.Robot import RobotIdentifier
from arena_runtime_msgs.action import RunTask
from arena_runtime_msgs.msg import EnvRegistry, RobotFleet, SemanticSnapshot
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header

from arena_fleet.config import load_cost, load_rules
from arena_fleet.dispatcher import INFOS, Command, DoorState, FleetDispatcher, Info, Pose2, PublishPlan, SendGoal, Stop
from arena_fleet.params import FleetParams
from arena_fleet.site import Site
from arena_fleet.view import MapOfDynamics

FOLLOW_STEP_M = 0.05
ENVS_TOPIC = "/arena/state/envs"
LATCHED = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)


def _config(value: str, name: str) -> pathlib.Path:
    return pathlib.Path(value) if value else pathlib.Path(get_package_share_directory("arena_fleet")) / "configs" / name


def _required(value: str, name: str) -> pathlib.Path:
    if not value:
        raise ValueError(f"parameter {name} is required")
    return pathlib.Path(value)


def _radius(model: str) -> float:
    mobile = RobotIdentifier(model).resolve_sync().mobile
    if mobile is None or mobile.radius is None:
        raise ValueError(f"robot model {model} has no mobile radius")
    return float(mobile.radius)


def _yaw(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class FleetNode(ArenaMixinNode):
    """Runs the fleet dispatcher next to an Arena env whose robots are parked by a scenario."""

    def __init__(self) -> None:
        super().__init__("arena_fleet", parameter_overrides=[rclpy.Parameter("use_sim_time", value=True)])
        self.p = p = FleetParams(self)
        info = p.INFO.value
        if info not in INFOS:
            raise ValueError(f"info {info!r} not in {INFOS}")
        self.people_info: Info = typing.cast(Info, info)
        self.site = Site.load(_required(p.WORLD.value, "world"), _required(p.RESOURCES.value, "resources"), _required(p.SCENARIO.value, "scenario"))
        self.mod = MapOfDynamics.load(pathlib.Path(p.MOD.value)) if p.MOD.value else None
        self.cost, self.rules = load_cost(_config(p.COST.value, "cost.yaml")), load_rules(_config(p.RULES.value, "rules.yaml"))
        self.out = _required(p.OUT.value, "out")
        self.dwb = p.LOCAL.value == "dwb"
        self.tick_s = p.TICK_S.value
        self.core: FleetDispatcher | None = None
        self.env_fqn: str | None = None
        self.raw: dict[str, Pose2] = {}
        self.prefix: str | None = None
        self.door_entities: dict[str, DoorState] = {}
        self.client: rclpy.action.ActionClient | None = None
        self.plan_pub: dict[str, rclpy.publisher.Publisher] = {}
        self.follow: dict[str, rclpy.action.ActionClient] = {}
        self.follow_seq: dict[str, int] = {}
        self.follow_handle: dict[str, rclpy.action.client.ClientGoalHandle | None] = {}
        self.create_subscription(EnvRegistry, ENVS_TOPIC, self.on_envs, LATCHED)

    async def setup(self) -> None:
        self.create_timer(self.tick_s, self.on_tick)

    async def teardown(self) -> None:
        if self.core is not None:
            self.core.close()

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_envs(self, msg: EnvRegistry) -> None:
        """Attach once to the env with our env_id: its fleet, door semantics, pedestrians and run_task action."""
        if self.env_fqn is not None:
            return
        record = next((r for r in msg.envs if r.env_id == self.p.ENV_ID.value), None)
        if record is None:
            return
        self.env_fqn = fqn = str(record.fqn)
        self.create_subscription(RobotFleet, f"{fqn}/state/robots", self.on_fleet, LATCHED)
        self.create_subscription(SemanticSnapshot, f"{fqn}/state/semantics", self.on_semantics, LATCHED)
        self.create_subscription(Pedestrians, f"{fqn.rsplit('/', 1)[0]}/arena_peds", self.on_peds, 10)
        self.client = rclpy.action.ActionClient(self, RunTask, f"{fqn}/runtime/run_task")
        self.get_logger().info(f"attached to env {record.env_id} at {fqn}")

    def on_fleet(self, msg: RobotFleet) -> None:
        """The dispatcher drives the env's fleet as it stands when the fleet is first published."""
        robots = [(str(s.descriptor.name), str(s.descriptor.ns).rstrip("/")) for s in msg.robots if str(s.descriptor.name).strip()]
        if self.core is not None or not robots:
            return
        names = [name for name, _ in robots]
        self.prefix = next(str(s.descriptor.frame) for s in msg.robots if str(s.descriptor.name).strip()).rsplit("/", 1)[0]
        robot_r = max(_radius(str(s.descriptor.model)) for s in msg.robots if str(s.descriptor.name).strip())
        self.core = FleetDispatcher(self.site, self.cost, self.rules, names=names, robot_r=robot_r, info=self.people_info, profile=self.p.PROFILE.value, seed=self.p.SEED.value, out=self.out, duration=self.p.DURATION_S.value, mod=self.mod, samples=self.p.SAMPLES.value)
        for name, ns in robots:
            self.plan_pub[name] = self.create_publisher(Path, f"{ns}/plan", 10)
            if self.dwb:
                self.follow[name] = rclpy.action.ActionClient(self, FollowPath, f"{ns}/follow_path")
            self.follow_seq[name], self.follow_handle[name] = 0, None
            self.create_subscription(Odometry, f"{ns}/odom", self.odom_callback(name), 10)
        self.get_logger().info(f"fleet {names}")

    def odom_callback(self, n: str) -> Callable[[Odometry], None]:
        def on_odom(msg: Odometry) -> None:
            p = msg.pose.pose.position
            self.raw[n] = (p.x, p.y, _yaw(msg))
            if self.core is not None:
                self.core.set_pose(n, *self.raw[n])

        return on_odom

    def on_peds(self, msg: Pedestrians) -> None:
        if self.core is not None:
            self.core.set_peds({str(p.id): (p.pose.position.x, p.pose.position.y, p.twist.linear.x, p.twist.linear.y) for p in msg.pedestrians})

    def on_semantics(self, msg: SemanticSnapshot) -> None:
        for e in msg.entities:
            if e.kind == "door":
                states = dict(zip(e.discrete_names, e.discrete_values, strict=True))
                values = dict(zip(e.continuous_names, e.continuous_values, strict=True))
                self.door_entities[str(e.entity)] = DoorState(str(states["state"]), float(values["progress"]))

    def doors(self) -> dict[str, DoorState]:
        """World doors of the site's level by name, from the env's semantic entities <env prefix>/<door>/<level>."""
        found = ((leaf.name, self.door_entities.get(f"{self.prefix}/{leaf.name}/{self.site.level}")) for leaf in self.site.leaves)
        return {name: state for name, state in found if state is not None}

    def try_setup(self, core: FleetDispatcher) -> bool:
        if self.client is None or not self.client.server_is_ready():
            return False
        if not core.try_setup(self.raw):
            return False
        self.get_logger().info(f"offset {core.off}, profile {core.profile}, info {core.info}")
        return True

    def on_tick(self) -> None:
        core = self.core
        if core is None:
            return
        core.set_time(self.now_s())
        if core.off is None and not self.try_setup(core):
            return
        core.set_doors(self.doors())
        core.tick()
        for cmd in core.drain():
            self.execute(cmd)
        if core.finished:
            self.get_logger().info(f"episode over, trips {core.trips_done}")
            core.close()
            self.request_shutdown("episode over")

    def execute(self, cmd: Command) -> None:
        match cmd:
            case PublishPlan(robot=n, path=path):
                header = Header(frame_id="map", stamp=self.get_clock().now().to_msg())
                poses = [self.pose_msg(header, *self.to_map(p)) for p in path]
                self.plan_pub[n].publish(Path(header=header, poses=poses))
                if self.dwb:
                    self.send_follow(n, header, poses)
            case SendGoal(robot=n, goal=goal, tolerance=tol):
                if not self.dwb:
                    self.run_task(n, yaml.safe_dump([{"goto": [*self.to_map(goal), 0.0], "tolerance_radius": tol}], default_flow_style=True))
            case Stop(robot=n):
                if self.dwb:
                    self.cancel_follow(n)
                else:
                    self.run_task(n, "[]")

    def to_map(self, p: tuple[float, float]) -> tuple[float, float]:
        off = self.core.off if self.core is not None and self.core.off is not None else (0.0, 0.0)
        return round(p[0] + off[0], 3), round(p[1] + off[1], 3)

    @staticmethod
    def pose_msg(header: Header, x: float, y: float, yaw: float = 0.0) -> PoseStamped:
        pose = PoseStamped(header=header)
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return pose

    def run_task(self, n: str, task_yaml: str) -> None:
        if self.client is None:
            return
        future = self.client.send_goal_async(RunTask.Goal(robot=n, task_yaml=task_yaml))

        def on_goal(f: rclpy.task.Future, n: str = n) -> None:
            handle = typing.cast(rclpy.action.client.ClientGoalHandle, f.result())
            if not handle.accepted and self.core is not None:
                self.core.log("rejected", robot=n)

        future.add_done_callback(on_goal)

    def send_follow(self, n: str, header: Header, poses: list[PoseStamped]) -> None:
        """Plan resampled to FOLLOW_STEP_M as a FollowPath goal; a newer goal preempts the running one."""
        client, core = self.follow[n], self.core
        if core is None:
            return
        if not client.server_is_ready():
            core.log("follow_not_ready", robot=n)
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in poses]
        dense: list[PoseStamped] = []
        for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
            k, th = max(int(math.dist((ax, ay), (bx, by)) / FOLLOW_STEP_M), 1), math.atan2(by - ay, bx - ax)
            dense += [self.pose_msg(header, ax + (bx - ax) * i / k, ay + (by - ay) * i / k, th) for i in range(k)]
        if poses:
            dense.append(PoseStamped(header=header, pose=poses[-1].pose))
        self.follow_seq[n] += 1
        seq = self.follow_seq[n]

        def on_goal(f: rclpy.task.Future, n: str = n, seq: int = seq) -> None:
            handle = typing.cast(rclpy.action.client.ClientGoalHandle, f.result())
            if not handle.accepted:
                core.log("rejected", robot=n)
            elif seq != self.follow_seq[n]:
                if core.rob[n].goal is None:
                    handle.cancel_goal_async()
            else:
                self.follow_handle[n] = handle

        client.send_goal_async(FollowPath.Goal(path=Path(header=header, poses=dense), controller_id="FollowPath")).add_done_callback(on_goal)

    def cancel_follow(self, n: str) -> None:
        self.follow_seq[n] += 1
        handle = self.follow_handle[n]
        if handle is not None:
            handle.cancel_goal_async()
            self.follow_handle[n] = None
