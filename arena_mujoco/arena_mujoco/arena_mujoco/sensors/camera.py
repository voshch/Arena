"""Camera publishers: color and depth images, camera info and depth point clouds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from arena_robots.Sensor import SensorType
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from arena_mujoco.context import get_scene_store
from arena_mujoco.sensors.core import SensorPublisher, get_renderer, make_stamp, register_sensor, sensor_pose

if TYPE_CHECKING:
    import rclpy.node
    from arena_robots.Sensor import SensorSpec

    from arena_mujoco.services.SpawnUrdf import RobotEntry, UrdfSensor

_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_HFOV = 1.047
_DEFAULT_NEAR = 0.1
_DEFAULT_FAR = 100.0
_DEFAULT_RATE_HZ = 30.0

_SPEC_SUFFIXES = ('_image', '_info', '_depth', '_points')

_LOOK_ALONG_X = np.array([0.5, 0.5, -0.5, -0.5])


def _camera_key(spec: SensorSpec) -> str:
    """Name the specs of one physical camera share."""
    if spec.sensor is not None:
        return spec.sensor
    for suffix in _SPEC_SUFFIXES:
        if spec.name.endswith(suffix):
            return spec.name.removesuffix(suffix)
    return spec.name


def _camera_name(element_prefix: str, spec: SensorSpec) -> str:
    return f'{element_prefix}{_camera_key(spec)}_cam'


def _resolution(urdf_sensor: UrdfSensor | None) -> tuple[int, int]:
    if urdf_sensor is None or urdf_sensor.width is None or urdf_sensor.height is None:
        return _DEFAULT_WIDTH, _DEFAULT_HEIGHT
    return urdf_sensor.width, urdf_sensor.height


def _horizontal_fov(urdf_sensor: UrdfSensor | None) -> float:
    if urdf_sensor is None or urdf_sensor.horizontal_fov is None:
        return _DEFAULT_HFOV
    return urdf_sensor.horizontal_fov


class CameraRig:
    """One MuJoCo camera, rendered at most once per sim time for every publisher that shares it."""

    def __init__(self, env_id: int, camera_name: str, urdf_sensor: UrdfSensor | None) -> None:
        self.env_id = env_id
        self.camera_name = camera_name
        self.width, self.height = _resolution(urdf_sensor)
        self.focal = (self.width / 2.0) / math.tan(_horizontal_fov(urdf_sensor) / 2.0)
        self.near = _DEFAULT_NEAR if urdf_sensor is None or urdf_sensor.near is None else urdf_sensor.near
        self.far = _DEFAULT_FAR if urdf_sensor is None or urdf_sensor.far is None else urdf_sensor.far
        rate = None if urdf_sensor is None else urdf_sensor.update_rate
        self.rate_hz = rate if rate else _DEFAULT_RATE_HZ
        self.users = 0
        self._model: mujoco.MjModel | None = None
        self._camera_id = -1
        self._time = -1.0
        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None

    def _bind(self, model: mujoco.MjModel) -> mujoco.Renderer | None:
        now = get_scene_store().time
        if self._model is not model:
            self._model = model
            self._camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
            self._time = -1.0
        if self._time != now:
            self._time = now
            self._color = None
            self._depth = None
        if self._camera_id < 0:
            return None
        return get_renderer(self.env_id, model, self.width, self.height)

    def color(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
        """rgb8 frame of the current sim time, None without a renderer."""
        renderer = self._bind(model)
        if renderer is None:
            return None
        if self._color is None:
            renderer.update_scene(data, camera=self._camera_id)
            self._color = renderer.render()
        return self._color

    def depth(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
        """Planar depth in meters of the current sim time, inf past far and -inf before near."""
        renderer = self._bind(model)
        if renderer is None:
            return None
        if self._depth is None:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=self._camera_id)
            depth = renderer.render()
            renderer.disable_depth_rendering()
            depth[depth > self.far] = np.inf
            depth[depth < self.near] = -np.inf
            self._depth = depth
        return self._depth


_rigs: dict[tuple[int, str], CameraRig] = {}


def body_rays(width: int, height: int, focal: float) -> np.ndarray:
    """Per-pixel directions (height, width, 3) in body axes, scaled to unit depth along x."""
    u = (np.arange(width) + 0.5 - width / 2.0) / focal
    v = (np.arange(height) + 0.5 - height / 2.0) / focal
    left, up = np.meshgrid(-u, -v)
    return np.stack([np.ones_like(left), left, up], axis=-1).astype(np.float32)


class _CameraPublisher(SensorPublisher):
    """Publisher on a camera shared by every spec with the same backing sensor."""

    _MSG_TYPE: type

    def __init__(
        self,
        node: rclpy.node.Node,
        env_id: int,
        spec: SensorSpec,
        frame_id: str,
        robot: RobotEntry,
        element_prefix: str,
    ) -> None:
        super().__init__(node, env_id, spec, frame_id, robot, element_prefix)
        self._rig_key = (env_id, _camera_name(element_prefix, spec))
        if self._rig_key not in _rigs:
            _rigs[self._rig_key] = CameraRig(env_id, self._rig_key[1], self.urdf_sensor)
        self._rig = _rigs[self._rig_key]
        self._rig.users += 1
        self._pub = node.create_publisher(self._MSG_TYPE, spec.topic, 10)

    def rebind(self) -> None:
        """The rig rebinds itself when the model changes."""

    def destroy(self) -> None:
        super().destroy()
        self._rig.users -= 1
        if self._rig.users == 0:
            del _rigs[self._rig_key]

    @classmethod
    def add_model_elements(
        cls,
        spec_builder: mujoco.MjSpec,
        mount_body: mujoco.MjsBody,
        sensor_spec: SensorSpec,
        element_prefix: str,
        urdf_sensor: UrdfSensor | None,
    ) -> None:
        """Add the shared camera once, looking along the sensor pose's x axis with z up."""
        name = _camera_name(element_prefix, sensor_spec)
        if spec_builder.camera(name) is not None:
            return
        width, height = _resolution(urdf_sensor)
        pos, sensor_quat = sensor_pose(urdf_sensor)
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, sensor_quat, _LOOK_ALONG_X)
        camera = mount_body.add_camera()
        camera.name = name
        camera.pos = pos
        camera.quat = quat.tolist()
        camera.fovy = math.degrees(2.0 * math.atan(math.tan(_horizontal_fov(urdf_sensor) / 2.0) * height / width))
        offscreen = spec_builder.visual.global_
        offscreen.offwidth = max(offscreen.offwidth, width)
        offscreen.offheight = max(offscreen.offheight, height)

    @property
    def rate_hz(self) -> float:
        return self._rig.rate_hz

    def _header(self) -> Header:
        return Header(stamp=make_stamp(get_scene_store().time), frame_id=self.frame_id)

    def _image(self, pixels: np.ndarray, encoding: str) -> Image:
        msg = Image()
        msg.header = self._header()
        msg.height = self._rig.height
        msg.width = self._rig.width
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = pixels.nbytes // self._rig.height
        msg.data = pixels.tobytes()
        return msg


@register_sensor(SensorType.IMAGE)
class CameraImagePublisher(_CameraPublisher):
    """Color frames as sensor_msgs/Image (rgb8), rendered only while subscribed."""

    _MSG_TYPE = Image

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._pub.get_subscription_count() == 0:
            return
        color = self._rig.color(model, data)
        if color is not None:
            self._pub.publish(self._image(color, 'rgb8'))


@register_sensor(SensorType.DEPTH)
class CameraDepthPublisher(_CameraPublisher):
    """Planar depth in meters as sensor_msgs/Image (32FC1), rendered only while subscribed."""

    _MSG_TYPE = Image

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._pub.get_subscription_count() == 0:
            return
        depth = self._rig.depth(model, data)
        if depth is not None:
            self._pub.publish(self._image(depth, '32FC1'))


@register_sensor(SensorType.POINTCLOUD, camera=True)
class CameraPointCloudPublisher(_CameraPublisher):
    """Depth returns as an XYZ cloud in the sensor frame's body axes (x forward, z up)."""

    _MSG_TYPE = PointCloud2

    def __init__(
        self,
        node: rclpy.node.Node,
        env_id: int,
        spec: SensorSpec,
        frame_id: str,
        robot: RobotEntry,
        element_prefix: str,
    ) -> None:
        super().__init__(node, env_id, spec, frame_id, robot, element_prefix)
        self._rays = body_rays(self._rig.width, self._rig.height, self._rig.focal)

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._pub.get_subscription_count() == 0:
            return
        depth = self._rig.depth(model, data)
        if depth is None:
            return
        hit = np.isfinite(depth)
        points = self._rays[hit] * depth[hit, None]
        self._pub.publish(point_cloud2.create_cloud_xyz32(self._header(), points))


@register_sensor(SensorType.CAMERA_INFO)
class CameraInfoPublisher(_CameraPublisher):
    """Pinhole intrinsics of the shared camera as sensor_msgs/CameraInfo."""

    _MSG_TYPE = CameraInfo

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model, data
        rig = self._rig
        cx = rig.width / 2.0
        cy = rig.height / 2.0
        msg = CameraInfo()
        msg.header = self._header()
        msg.height = rig.height
        msg.width = rig.width
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [rig.focal, 0.0, cx, 0.0, rig.focal, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [rig.focal, 0.0, cx, 0.0, 0.0, rig.focal, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._pub.publish(msg)


__all__ = [
    'CameraDepthPublisher',
    'CameraImagePublisher',
    'CameraInfoPublisher',
    'CameraPointCloudPublisher',
]
