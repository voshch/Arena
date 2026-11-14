import math
import pathlib
import typing
from collections.abc import Callable

import rclpy
import rclpy.action
import rclpy.action.client
import rclpy.action.graph
import rclpy.task
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins import ArenaMixinNode
from arena_runtime_msgs.action import RunTask
from flatsim_msgs.msg import EntityPoses
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header

from arena_fleet.config import load_cost, load_rules
from arena_fleet.dispatcher import INFOS, Command, FleetDispatcher, Info, Pose2, PublishPlan, SendGoal, Stop, robot_names
from arena_fleet.params import FleetParams
from arena_fleet.site import Site
from arena_fleet.view import MapOfDynamics

FOLLOW_STEP_M = 0.05


def _config(value: str, name: str) -> pathlib.Path:
    return pathlib.Path(value) if value else pathlib.Path(get_package_share_directory("arena_fleet")) / "configs" / name


def _required(value: str, name: str) -> pathlib.Path:
    if not value:
        raise ValueError(f"parameter {name} is required")
    return pathlib.Path(value)


def _yaw(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class FleetNode(ArenaMixinNode):
    """Runs the fleet dispatcher next to an Arena env whose robots are parked by a scenario."""

    def __init__(self) -> None:
        super().__init__("arena_fleet", parameter_overrides=[rclpy.Parameter("use_sim_time", value=True)])
        p = FleetParams(self)
        info = p.INFO.value
        if info not in INFOS:
            raise ValueError(f"info {info!r} not in {INFOS}")
        site = Site.load(_required(p.WORLD.value, "world"), _required(p.RESOURCES.value, "resources"), _required(p.SCENARIO.value, "scenario"))
        mod = MapOfDynamics.load(pathlib.Path(p.MOD.value)) if p.MOD.value else None
        self.names = robot_names(p.ROBOT.value, p.N.value)
        self.core = FleetDispatcher(
            site, load_cost(_config(p.COST.value, "cost.yaml")), load_rules(_config(p.RULES.value, "rules.yaml")), names=self.names, info=typing.cast(Info, info), profile=p.PROFILE.value, seed=p.SEED.value, out=_required(p.OUT.value, "out"), duration=p.DURATION_S.value, mod=mod, samples=p.SAMPLES.value
        )
        self.dwb = p.LOCAL.value == "dwb"
        self.tick_s = p.TICK_S.value
        ns = p.NS.value.rstrip("/")
        self.raw: dict[str, Pose2] = {}
        self.client: rclpy.action.ActionClient | None = None
        self.plan_pub = {n: self.create_publisher(Path, f"{ns}/arena_env_node/{n}/plan", 10) for n in self.names}
        self.follow = {n: rclpy.action.ActionClient(self, FollowPath, f"{ns}/arena_env_node/{n}/follow_path") for n in self.names} if self.dwb else {}
        self.follow_seq = dict.fromkeys(self.names, 0)
        self.follow_handle: dict[str, rclpy.action.client.ClientGoalHandle | None] = dict.fromkeys(self.names, None)
        for n in self.names:
            self.create_subscription(Odometry, f"{ns}/arena_env_node/{n}/odom", self.odom_callback(n), 10)
        self.create_subscription(Pedestrians, f"{ns}/arena_peds", self.on_peds, 10)
        self.create_subscription(EntityPoses, "/flatsim/poses", self.on_leaves, 10)

    async def setup(self) -> None:
        self.create_timer(self.tick_s, self.on_tick)

    async def teardown(self) -> None:
        self.core.close()

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, n: str) -> Callable[[Odometry], None]:
        def on_odom(msg: Odometry) -> None:
            p = msg.pose.pose.position
            self.raw[n] = (p.x, p.y, _yaw(msg))
            self.core.set_pose(n, *self.raw[n])

        return on_odom

    def on_peds(self, msg: Pedestrians) -> None:
        self.core.set_peds({str(p.id): (p.pose.position.x, p.pose.position.y, p.twist.linear.x, p.twist.linear.y) for p in msg.pedestrians})

    def on_leaves(self, msg: EntityPoses) -> None:
        self.core.set_leaves([(p.name, p.x, p.y) for p in msg.poses])

    def try_setup(self) -> bool:
        if self.client is None:
            names = [name for name, _ in rclpy.action.graph.get_action_names_and_types(self) if name.endswith("runtime/run_task")]
            if not names:
                return False
            self.client = rclpy.action.ActionClient(self, RunTask, names[0])
        if not self.client.server_is_ready():
            return False
        if not self.core.try_setup(self.raw):
            return False
        self.get_logger().info(f"offset {self.core.off}, profile {self.core.profile}, info {self.core.info}")
        return True

    def on_tick(self) -> None:
        self.core.set_time(self.now_s())
        if self.core.off is None and not self.try_setup():
            return
        self.core.tick()
        for cmd in self.core.drain():
            self.execute(cmd)
        if self.core.finished:
            self.get_logger().info(f"episode over, trips {self.core.trips_done}")
            self.core.close()
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
        off = self.core.off or (0.0, 0.0)
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
            if not handle.accepted:
                self.core.log("rejected", robot=n)

        future.add_done_callback(on_goal)

    def send_follow(self, n: str, header: Header, poses: list[PoseStamped]) -> None:
        """Plan resampled to FOLLOW_STEP_M as a FollowPath goal; a newer goal preempts the running one."""
        client = self.follow[n]
        if not client.server_is_ready():
            self.core.log("follow_not_ready", robot=n)
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
                self.core.log("rejected", robot=n)
            elif seq != self.follow_seq[n]:
                if self.core.rob[n].goal is None:
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
