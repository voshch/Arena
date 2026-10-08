"""ROS surface of the viewport camera: the /arena/viewport services and topics every sim backend advertises, over one env."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import geometry_msgs.msg
import mujoco
import rclpy.task
import rclpy.time
import sensor_msgs.msg
import tf2_ros
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from viewport_control_msgs.msg import ViewportView
from viewport_control_msgs.srv import ViewportCapture, ViewportSetProjection, ViewportSetReferenceFrame, ViewportSetView

from arena_mujoco.sensors.core import get_renderer, make_stamp
from arena_mujoco.services.SpawnCeilings import CEILING_GROUP
from arena_mujoco.services.SpawnUrdf import frame_body

from .camera import CAPTURE_HEIGHT, CAPTURE_WIDTH, Lens, orbit_of, pose_of, render
from .controller import Keyframe, Pose, ViewportController

if TYPE_CHECKING:
    from collections.abc import Callable

    import rclpy.node

    from arena_mujoco.scene import SceneStore

NAMESPACE = '/arena/viewport'
POSE_FRAME = 'map'
CAPTURE_TIMEOUT_S = 15.0

_POSE_PERIOD_S = 0.1
_START = pose_of((0.0, 0.0, 0.0), 10.0, 90.0, -45.0)

STREAM_QOS = QoSProfile(depth=64, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


def _pose(msg: geometry_msgs.msg.Pose) -> Pose:
    return Pose(
        (msg.position.x, msg.position.y, msg.position.z),
        (msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z),
    )


class Viewport:
    """Free camera over env_id, driven through the viewport contract and mirrored onto the passive viewer while one is open."""

    def __init__(self, node: rclpy.node.Node, store: SceneStore, env_id: int, viewer: Callable[[], object | None]) -> None:
        self._node = node
        self._store = store
        self._env_id = env_id
        self._viewer = viewer
        self._controller = ViewportController()
        self._pose = _START
        self._lens = Lens()
        self._option = mujoco.MjvOption()
        self._option.geomgroup[CEILING_GROUP] = 0
        self._mirrored: tuple[tuple[float, float, float], float, float] | None = None
        self._gated: list[tuple[float, float, rclpy.task.Future]] = []
        self._tf: tf2_ros.Buffer | None = None
        self._tf_listener: tf2_ros.TransformListener | None = None
        self._warned_entity = ''
        self._pose_due = 0.0

        node.create_service(ViewportSetView, f'{NAMESPACE}/set_view', self._cb_set_view)
        node.create_service(ViewportSetReferenceFrame, f'{NAMESPACE}/set_reference_frame', self._cb_set_reference_frame)
        node.create_service(ViewportSetProjection, f'{NAMESPACE}/set_projection', self._cb_set_projection)
        node.create_service(ViewportCapture, f'{NAMESPACE}/capture', self._cb_capture, callback_group=MutuallyExclusiveCallbackGroup())
        node.create_subscription(ViewportView, f'{NAMESPACE}/cmd_view', self._cb_cmd_view, STREAM_QOS)
        self._pose_pub = node.create_publisher(geometry_msgs.msg.PoseStamped, f'{NAMESPACE}/camera_pose', 10)

    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _cb_set_view(self, request: ViewportSetView.Request, response: ViewportSetView.Response) -> ViewportSetView.Response:
        eye = (request.eye.x, request.eye.y, request.eye.z)
        target = (request.target.x, request.target.y, request.target.z)
        self._controller.set_view(eye, target, request.fov)
        response.success = True
        response.message = 'ok'
        return response

    def _cb_set_reference_frame(self, request: ViewportSetReferenceFrame.Request, response: ViewportSetReferenceFrame.Response) -> ViewportSetReferenceFrame.Response:
        response.message = self._controller.set_reference_frame(request.entity, _pose(request.pose), request.has_pose, request.mode)
        self._warned_entity = ''
        response.success = True
        return response

    def _cb_set_projection(self, request: ViewportSetProjection.Request, response: ViewportSetProjection.Response) -> ViewportSetProjection.Response:
        response.success = self._controller.set_projection(request.projection)
        response.message = 'ok' if response.success else "projection must be 'perspective' or 'orthographic'"
        return response

    def _cb_cmd_view(self, msg: ViewportView) -> None:
        keyframe = Keyframe(
            time=rclpy.time.Time.from_msg(msg.target_time).nanoseconds * 1e-9,
            local=_pose(msg.pose),
            world_orientation=msg.world_orientation,
            fov=msg.fov,
        )
        self._controller.push_keyframe(keyframe, self._now())

    async def _cb_capture(self, request: ViewportCapture.Request, response: ViewportCapture.Response) -> ViewportCapture.Response:
        """Snap to the requested pose once sim time reached min_sim_time, render and return the frame."""
        min_sim_time = rclpy.time.Time.from_msg(request.min_sim_time).nanoseconds * 1e-9
        if min_sim_time > 0.0 and self._store.time + 1e-6 < min_sim_time:
            reached = rclpy.task.Future()
            self._gated.append((min_sim_time, time.monotonic() + CAPTURE_TIMEOUT_S, reached))
            await reached
            if not reached.result():
                response.success = False
                response.message = 'capture timed out, min_sim_time not reached'
                return response
        self._controller.set_local(_pose(request.pose), request.world_orientation, request.fov)
        self.apply()
        model = self._store.get_model(self._env_id)
        data = self._store.get_data(self._env_id)
        renderer = None if model is None else get_renderer(self._env_id, model, CAPTURE_WIDTH, CAPTURE_HEIGHT)
        if renderer is None or data is None:
            response.success = False
            response.message = 'no offscreen renderer for the viewport'
            return response
        pixels = render(renderer, data, self._pose, self._lens, self._option)
        image = sensor_msgs.msg.Image()
        image.header.stamp = make_stamp(self._store.time)
        image.header.frame_id = POSE_FRAME
        image.width = CAPTURE_WIDTH
        image.height = CAPTURE_HEIGHT
        image.encoding = 'rgb8'
        image.is_bigendian = False
        image.step = CAPTURE_WIDTH * 3
        image.data = pixels.tobytes()
        response.success = True
        response.message = 'ok'
        response.image = image
        return response

    def release(self) -> None:
        """Answer the captures whose min_sim_time the sim reached, and fail the ones waiting past their deadline."""
        if not self._gated:
            return
        sim_time = self._store.time
        now = time.monotonic()
        waiting = []
        for min_sim_time, deadline, reached in self._gated:
            if sim_time + 1e-6 >= min_sim_time:
                reached.set_result(True)
            elif now > deadline:
                reached.set_result(False)
            else:
                waiting.append((min_sim_time, deadline, reached))
        self._gated = waiting

    def _tf_pose(self, frame: str) -> Pose | None:
        if self._tf is None:
            self._tf = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf, self._node, spin_thread=False)
        try:
            tf = self._tf.lookup_transform(POSE_FRAME, frame, rclpy.time.Time()).transform
        except tf2_ros.TransformException:
            return None
        return Pose(
            (tf.translation.x, tf.translation.y, tf.translation.z),
            (tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z),
        )

    def _sample_reference(self) -> None:
        """Resolve the tracked entity's world pose: a body of the env by name, a robot link by its TF frame id, else a TF frame."""
        entity = self._controller.tracked_entity
        if not entity:
            return
        body = self._store.get_body_pose(self._env_id, entity)
        if body is None:
            link = frame_body(self._env_id, entity)
            body = None if link is None else self._store.get_body_pose(self._env_id, link)
        pose = Pose(*body) if body is not None else self._tf_pose(entity)
        if pose is None:
            if self._warned_entity != entity:
                self._warned_entity = entity
                self._node.get_logger().warning(f"tracked entity '{entity}' is neither a body of env {self._env_id} nor a TF frame, camera holds the world frame")
            return
        self._warned_entity = ''
        self._controller.set_reference_target(pose)

    def _seen(self) -> Pose:
        """Where the camera stands: the commanded pose, or the viewer's own once the user moved it."""
        viewer = self._viewer()
        if viewer is None:
            return self._pose
        with viewer.lock():
            orbit = (tuple(viewer.cam.lookat), viewer.cam.azimuth, viewer.cam.elevation)
            distance = viewer.cam.distance
        if orbit == self._mirrored:
            return self._pose
        self._mirrored = None
        return pose_of(orbit[0], distance, orbit[1], orbit[2])

    def _mirror(self) -> None:
        viewer = self._viewer()
        if viewer is None:
            return
        with viewer.lock():
            lookat, azimuth, elevation = orbit_of(self._pose, viewer.cam.distance)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = lookat
            viewer.cam.azimuth = azimuth
            viewer.cam.elevation = elevation
            self._mirrored = (tuple(viewer.cam.lookat), viewer.cam.azimuth, viewer.cam.elevation)

    def apply(self) -> None:
        """Drive the camera one frame and publish its pose every _POSE_PERIOD_S."""
        self._sample_reference()
        seen = self._seen()
        frame = self._controller.apply(self._now(), seen)
        if self._controller.released is not None:
            offset, angle = self._controller.released
            self._controller.released = None
            self._node.get_logger().warning(f'camera moved off its tracked pose by {offset:.3f} m / {angle:.4f} rad, releasing the tracked entity')
        if frame.projection is not None:
            self._lens.orthographic = frame.projection == 'orthographic'
        if frame.fov is not None and frame.fov > 0.0:
            self._lens.hfov = frame.fov
        if frame.pose is not None:
            self._pose = frame.pose
            self._mirror()
        else:
            self._pose = seen

        now = time.monotonic()
        if now < self._pose_due:
            return
        self._pose_due = now + _POSE_PERIOD_S
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = POSE_FRAME
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self._pose.position
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = self._pose.orientation
        self._pose_pub.publish(msg)
