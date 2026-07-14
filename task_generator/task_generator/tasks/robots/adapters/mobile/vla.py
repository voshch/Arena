"""VLA adapter: the central bridge for language-conditioned navigation.

It owns the vla_server lifecycle, the per-robot inference loop, the per-action-type handlers, and the
intent visualization. The task mode submits a single `VLAPhase` carrying the instruction; everything
else (image, inference, waypoint following, completion) lives here. Waypoints are followed through the
inherited nav2 stack; a future cmd_vel form publishes velocity directly.
"""

from __future__ import annotations

import asyncio
import io
import math
from typing import TYPE_CHECKING, ClassVar

import httpx
import numpy as np
import sensor_msgs.msg
import visualization_msgs.msg
from arena_robots.bringup.mobile.nav2 import Nav2Bringup
from arena_robots.clients.goto_pose import GotoPoseClient
from arena_robots.Sensor import SensorType
from arena_viz.kinds import DisplayKind
from PIL import Image
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.robots.adapters import AdapterDisplayHint, AdapterMeta
from task_generator.tasks.robots.adapters.mobile.nav2 import Nav2Adapter
from task_generator.tasks.robots.request import GoToPhase, TaskPhase, VLAPhase
from task_generator.tasks.robots.vla.contract import MobileAction, Response, Waypoints, parse

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.tasks.robots.adapters import ResetContext

_MODEL = "omnivla-edge"

_INFERENCE_INTERVAL = 2.0  # seconds between inference cycles
_WAYPOINT_PROXIMITY_THRESHOLD = 0.2  # meter, threshold before handing the final goal to nav2
_MAX_INVALID_STREAK = 3  # consecutive all-invalid cycles before forcing proximity handoff


@AdapterMeta.attach(
    accepts={GoToPhase.kind},
    bringup=Nav2Bringup,
    client=GotoPoseClient,
    cap="mobile",
    displays=[
        AdapterDisplayHint(
            name="VLA Intent",
            topic="{ns}/vla/intent",
            topic_type="visualization_msgs/MarkerArray",
            kind=DisplayKind.MARKER_ARRAY,
        ),
    ],
)
class VLAAdapter(Nav2Adapter):
    kind: ClassVar[str] = "vla"

    def __init__(self, robot_manager: RobotManager, **bringup_kwargs: object) -> None:
        super().__init__(robot_manager, **bringup_kwargs)
        self._instruction: str | None = None
        self._latest_image: sensor_msgs.msg.Image | None = None
        self._image_sub: object = None
        self._near_goal = False
        self._invalid_streak = 0
        self._last_dispatch = robot_manager.node.sim_time
        self._loop_task: asyncio.Task | None = None
        self._vla_endpoint: str | None = None
        self._http: httpx.AsyncClient | None = None
        self._intent_pub = robot_manager.node.create_publisher(
            visualization_msgs.msg.MarkerArray,
            str(robot_manager.namespace("vla/intent")),
            1,
        )
        # per-action-type handlers; extend with cmd_vel when that form lands on the wire
        self._mobile_handlers = {Waypoints: self._handle_waypoints}

    # ------------------------------------------------------------------
    # vla_server lifecycle
    # ------------------------------------------------------------------

    async def ensure_services(self) -> None:
        await super().ensure_services()
        if self._vla_endpoint is not None:
            return
        # capture both streams: the port rides stdout, docker chatter rides stderr (the feature folds
        # compose output into stderr). Stay silent on success, surface stderr only on failure.
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            "arena feature vla_server launch",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            for line in err.decode(errors="replace").splitlines():
                self.rm.node.get_logger().error(f"[vla_server] {line}")
            raise RuntimeError("[VLAAdapter] vla_server launch failed (see logs above)")

        daemon = f"http://127.0.0.1:{out.decode().strip()}"
        async with httpx.AsyncClient(base_url=daemon, timeout=180.0) as client:
            resp = await client.post("/ensure", json={"model": _MODEL})
            resp.raise_for_status()
            self._vla_endpoint = f"http://127.0.0.1:{resp.json()['port']}"
        self._http = httpx.AsyncClient(base_url=self._vla_endpoint, timeout=10.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        self._cancel_loop()
        await super().on_reset(robot, ctx)  # cancels nav2 goal, teleports, clears costmap

        self._near_goal = False
        self._invalid_streak = 0
        self._latest_image = None
        self._last_dispatch = robot.node.sim_time

        if self._http is not None:
            await self._http.post("/reset", data={"session": self._session()})

        if self._image_sub is None:
            topic = self._find_image_topic(robot)
            if topic is None:
                robot.node.get_logger().warn(f"[VLAAdapter:{robot.name}] no image sensor in model_params; VLA disabled for this robot")
            else:
                qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
                self._image_sub = robot.node.create_subscription(sensor_msgs.msg.Image, topic, self._on_image, qos)
                robot.node.get_logger().info(f"[VLAAdapter:{robot.name}] subscribed to {topic}")

        if self._instruction is not None:
            self._start_loop(robot)

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        if not isinstance(phase, VLAPhase):
            await super().dispatch_phase(phase, robot)  # plain GOTO_POSE passthrough (nav2)
            return
        self._instruction = phase.instruction
        self._near_goal = False
        self._invalid_streak = 0
        self._last_dispatch = robot.node.sim_time
        self._start_loop(robot)

    def is_phase_done(self, phase: TaskPhase, robot: RobotManager) -> bool | None:
        if not isinstance(phase, VLAPhase):
            return super().is_phase_done(phase, robot)
        if self._near_goal:
            return self.client.is_done() is True
        timeout = robot.node.conf.Robot.TIMEOUT.value
        if (robot.node.sim_time.sec - self._last_dispatch.sec) >= timeout:
            return True
        return False

    async def teardown(self) -> None:
        self._cancel_loop()
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        await super().teardown()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    def _start_loop(self, robot: RobotManager) -> None:
        self._cancel_loop()
        self._loop_task = asyncio.ensure_future(self._run_loop(robot))

        def _on_done(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                import traceback  # noqa: PLC0415

                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                robot.node.get_logger().error(f"[VLAAdapter:{robot.name}] inference loop crashed:\n{tb}")

        self._loop_task.add_done_callback(_on_done)

    def _cancel_loop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            self._loop_task = None

    async def _run_loop(self, robot: RobotManager) -> None:
        while True:
            await asyncio.sleep(_INFERENCE_INTERVAL)
            if self._near_goal:
                continue  # waiting on nav2 to finish the approach
            await self._infer_and_dispatch(robot)

    async def _infer_and_dispatch(self, robot: RobotManager) -> None:
        image = self._latest_image
        current_pose = robot.pose
        if image is None or current_pose is None:
            robot.node.get_logger().warn(f"[VLAAdapter:{robot.name}] no image or pose; skipping cycle")
            return

        response = await self._act(image)
        if response is None:
            return

        self._publish_intent(robot, current_pose, response.meta.intent)

        handler = self._mobile_handlers.get(type(response.actions.mobile))
        if handler is None:
            return
        await handler(response.actions.mobile, robot, current_pose)

    async def _act(self, image: sensor_msgs.msg.Image) -> Response | None:
        arr = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        if image.encoding == "bgr8":
            arr = arr[:, :, ::-1].copy()  # PIL expects rgb
        buf = io.BytesIO()
        Image.fromarray(arr).resize((512, 512)).save(buf, format="JPEG")

        assert self._http is not None
        try:
            resp = await self._http.post(
                "/act",
                files={"image": ("frame.jpg", buf.getvalue(), "image/jpeg")},
                data={"instruction": self._instruction, "session": self._session()},
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            self.rm.node.get_logger().warn(f"[VLAAdapter] server unreachable at {self._vla_endpoint}")
            return None
        return parse(resp.json())

    # ------------------------------------------------------------------
    # Per-action-type handlers
    # ------------------------------------------------------------------

    async def _handle_waypoints(self, action: MobileAction, robot: RobotManager, current_pose: Pose) -> None:
        env = robot._environment_manager  # noqa: SLF001

        phases = [GoToPhase(pose=env.ezilear(self._to_pose(current_pose, (wp.x, wp.y)))) for wp in action.steps]
        phases = [p for p in phases if self._is_valid_pose(robot, p.pose.position.x, p.pose.position.y)]

        if not phases:
            self._invalid_streak += 1
            robot.node.get_logger().warn(f"[VLAAdapter:{robot.name}] all waypoints invalid ({self._invalid_streak}/{_MAX_INVALID_STREAK})")
            if self._invalid_streak >= _MAX_INVALID_STREAK:
                robot.node.get_logger().warn(f"[VLAAdapter:{robot.name}] invalid streak limit reached, forcing proximity handoff")
                self._near_goal = True
            return

        goal = phases[-1]  # furthest valid waypoint
        realized = [current_pose, env.realize(goal.pose)]
        total_dist = math.hypot(realized[1].position.x - realized[0].position.x, realized[1].position.y - realized[0].position.y)
        self._near_goal = total_dist < _WAYPOINT_PROXIMITY_THRESHOLD
        if self._near_goal:
            robot.node.get_logger().info(f"[VLAAdapter:{robot.name}] proximity mode, total_dist={total_dist:.2f}m < {_WAYPOINT_PROXIMITY_THRESHOLD}m, handing final goal to nav2")
        else:
            robot.node.get_logger().debug(f"[VLAAdapter:{robot.name}] goal ({realized[1].position.x:.2f},{realized[1].position.y:.2f})")

        self._invalid_streak = 0
        self._last_dispatch = robot.node.sim_time
        await super().dispatch_phase(GoToPhase(pose=realized[1]), robot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session(self) -> str:
        return str(self.rm.namespace)

    def _on_image(self, msg: sensor_msgs.msg.Image) -> None:
        self._latest_image = msg

    def _find_image_topic(self, robot: RobotManager) -> str | None:
        image_sensors = [s for s in robot.robot_view.model_params.sensors if s.type == SensorType.IMAGE]
        if not image_sensors:
            return None
        preferred = next((s for s in image_sensors if "front" in s.name.lower()), image_sensors[0])
        return str(robot.namespace(preferred.topic.removeprefix("${namespace}/")))

    def _is_valid_pose(self, robot: RobotManager, x: float, y: float) -> bool:
        from task_generator.manager.world_manager.utils import WorldOccupancy  # noqa: PLC0415

        world_map = robot.node._world_manager.map  # noqa: SLF001
        row, col = world_map.tf_pos2grid(Position(x=x, y=y))
        rows, cols = world_map.occupancy.grid.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        return bool(WorldOccupancy.not_full(world_map.occupancy.grid)[int(row), int(col)])

    def _to_pose(self, current: Pose, action: tuple[float, float]) -> Pose:
        dx, dy = action
        yaw = current.orientation.to_yaw()
        move_dx = dx * math.cos(yaw) - dy * math.sin(yaw)
        move_dy = dx * math.sin(yaw) + dy * math.cos(yaw)
        new_x = current.position.x + move_dx
        new_y = current.position.y + move_dy
        return Pose(Position(new_x, new_y), Orientation.from_yaw(math.atan2(move_dy, move_dx)))

    def _publish_intent(self, robot: RobotManager, current_pose: Pose, intent: str) -> None:
        markers = visualization_msgs.msg.MarkerArray()
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = robot.node.sim_time.to_msg()
        marker.ns = "vla_intent"
        marker.id = 0
        marker.type = visualization_msgs.msg.Marker.TEXT_VIEW_FACING
        marker.action = visualization_msgs.msg.Marker.ADD if intent else visualization_msgs.msg.Marker.DELETE
        marker.pose.position.x = current_pose.position.x
        marker.pose.position.y = current_pose.position.y
        marker.pose.position.z = 1.5
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = intent
        markers.markers.append(marker)
        self._intent_pub.publish(markers)


__all__ = ["VLAAdapter"]
