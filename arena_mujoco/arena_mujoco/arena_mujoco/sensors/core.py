"""Sensor framework: spec ingestion, publisher base, renderer pool, hook wiring."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import attrs
import mujoco
import numpy as np
from builtin_interfaces.msg import Time

from arena_mujoco import hooks
from arena_mujoco.context import get_scene_store

if TYPE_CHECKING:
    from collections.abc import Callable

    import rclpy.node
    import rclpy.publisher
    from arena_robots.Sensor import SensorSpec

    from arena_mujoco.services.SpawnUrdf import RobotEntry, UrdfSensor


_DEFAULT_RATE_HZ = 30.0

_RATE_SLACK_S = 1e-6


def make_stamp(sim_time: float) -> Time:
    """Sim time in seconds as a message stamp."""
    sec = int(sim_time)
    return Time(sec=sec, nanosec=int(round((sim_time - sec) * 1e9)))


def backing_sensor(robot: RobotEntry, spec: SensorSpec) -> UrdfSensor | None:
    """The URDF <sensor> element a spec is backed by, None without one."""
    if spec.sensor is None:
        return None
    return robot.urdf_sensors.get(spec.sensor)


def sensor_pose(urdf_sensor: UrdfSensor | None) -> tuple[list[float], np.ndarray]:
    """Position and quaternion (wxyz) of a URDF sensor on its link, identity without one."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    if urdf_sensor is None:
        return [0.0, 0.0, 0.0], quat
    mujoco.mju_euler2Quat(quat, np.array(urdf_sensor.pose[3:]), 'XYZ')
    return list(urdf_sensor.pose[:3]), quat


class SensorPublisher(abc.ABC):
    """One sensor instance bound to one robot in one env."""

    def __init__(
        self,
        node: rclpy.node.Node,
        env_id: int,
        spec: SensorSpec,
        frame_id: str,
        robot: RobotEntry,
        element_prefix: str,
    ) -> None:
        self.node = node
        self.env_id = env_id
        self.spec = spec
        self.robot_model = robot.robot_model
        self.urdf_sensor = backing_sensor(robot, spec)
        self.element_prefix = element_prefix
        self._pub: rclpy.publisher.Publisher
        self._bound_model: mujoco.MjModel | None = None
        self.frame_id = frame_id

    @abc.abstractmethod
    def rebind(self) -> None:
        """Drop ids cached against the previous model, a recompile renumbers every element."""

    @classmethod
    @abc.abstractmethod
    def add_model_elements(
        cls,
        spec_builder: mujoco.MjSpec,
        mount_body: mujoco.MjsBody,
        sensor_spec: SensorSpec,
        element_prefix: str,
        urdf_sensor: UrdfSensor | None,
    ) -> None:
        """Add this sensor's MuJoCo elements to the env spec before recompile."""

    def destroy(self) -> None:
        """Release the ROS entities of a despawned robot's sensor."""
        self.node.destroy_publisher(self._pub)

    @abc.abstractmethod
    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Read the sensor's MuJoCo data and publish the ROS message for one tick."""

    @property
    def rate_hz(self) -> float:
        """Publish cadence for this sensor; subclasses may override per spec."""
        return _DEFAULT_RATE_HZ


_REGISTRY: dict[tuple[str, bool], type[SensorPublisher]] = {}


def register_sensor(sensor_type: str, *, camera: bool = False) -> Callable[[type[SensorPublisher]], type[SensorPublisher]]:
    """Class decorator binding a SensorType, camera-backed or not, to its SensorPublisher subclass."""

    def _decorate(cls: type[SensorPublisher]) -> type[SensorPublisher]:
        key = (str(sensor_type), camera)
        if key in _REGISTRY:
            raise ValueError(f'sensor type {key!r} already registered to {_REGISTRY[key].__name__}')
        _REGISTRY[key] = cls
        return cls

    return _decorate


def get_publisher_class(sensor_type: str, camera: bool) -> type[SensorPublisher] | None:
    """Return the publisher class for a SensorType value, the camera-backed one if registered."""
    return _REGISTRY.get((str(sensor_type), camera)) or _REGISTRY.get((str(sensor_type), False))


class _RendererPool:
    """Lazy, per-(env, resolution) cache of offscreen mujoco.Renderer handles."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int, int], tuple[mujoco.MjModel, mujoco.Renderer | None]] = {}

    def get(self, env_id: int, model: mujoco.MjModel, width: int, height: int) -> mujoco.Renderer | None:
        key = (int(env_id), int(width), int(height))
        cached = self._entries.get(key)
        if cached is not None:
            bound_model, renderer = cached
            if bound_model is model:
                return renderer
            self._close(renderer)
        renderer = self._make(model, int(width), int(height))
        self._entries[key] = (model, renderer)
        return renderer

    @staticmethod
    def _make(model: mujoco.MjModel, width: int, height: int) -> mujoco.Renderer | None:
        try:
            return mujoco.Renderer(model, height, width)
        except Exception as exc:  # noqa: BLE001 - any GL/EGL failure means no rendering here
            hooks.get_node().get_logger().warning(f'offscreen renderer unavailable ({width}x{height}): {exc}')
            return None

    @staticmethod
    def _close(renderer: mujoco.Renderer | None) -> None:
        if renderer is None:
            return
        try:
            renderer.close()
        except Exception:  # noqa: BLE001 - EGL teardown is noisy, the renderer is being discarded anyway
            pass


_renderer_pool = _RendererPool()


def get_renderer(env_id: int, model: mujoco.MjModel, width: int, height: int) -> mujoco.Renderer | None:
    """Return a shared offscreen renderer for (env, model, w, h), or None if unavailable."""
    return _renderer_pool.get(env_id, model, width, height)


_active: dict[tuple[int, str], list[SensorPublisher]] = {}


_CAMERA_TYPES = frozenset({'image', 'depth', 'camera_info'})


def _element_prefix(robot_entry: RobotEntry) -> str:
    return f'{robot_entry.prim_name}_sensor_'


def _planned_sensors(
    robot_entry: RobotEntry,
) -> list[tuple[SensorSpec, UrdfSensor | None, type[SensorPublisher]]]:
    """Every declared sensor with a publisher class, paired with its backing URDF sensor, names made unique."""
    seen: dict[str, int] = {}
    specs = []
    for spec in robot_entry.sensors:
        count = seen.get(spec.name, 0)
        seen[spec.name] = count + 1
        specs.append(spec if count == 0 else attrs.evolve(spec, name=f'{spec.name}_{count}'))
    camera_sensors = {spec.sensor for spec in specs if spec.sensor is not None and str(spec.type) in _CAMERA_TYPES}
    planned = []
    for spec in specs:
        urdf_sensor = backing_sensor(robot_entry, spec)
        camera = spec.sensor in camera_sensors or (urdf_sensor is not None and urdf_sensor.is_camera)
        cls = get_publisher_class(spec.type, camera)
        if cls is not None:
            planned.append((spec, urdf_sensor, cls))
    return planned


def _sensor_pre_compile(
    env_id: int,
    spec: mujoco.MjSpec,
    robot_root_body: mujoco.MjsBody,
    robot_entry: RobotEntry,
    joint_names: list[str],
) -> None:
    """pre-compile hook: add every declared sensor's MuJoCo elements to the spec."""
    del joint_names
    from arena_mujoco.scene import sanitize_id

    link_prefix = robot_root_body.name.removesuffix(sanitize_id(robot_entry.base_frame))
    for sensor_spec, urdf_sensor, cls in _planned_sensors(robot_entry):
        links = (sensor_spec.frame,) if urdf_sensor is None else (urdf_sensor.link, sensor_spec.frame)
        mounts = (spec.body(link_prefix + sanitize_id(link)) for link in links)
        cls.add_model_elements(
            spec,
            next((mount for mount in mounts if mount is not None), robot_root_body),
            sensor_spec,
            _element_prefix(robot_entry),
            urdf_sensor,
        )


def _sensor_post_spawn(
    env_id: int,
    name: str,
    robot_entry: RobotEntry,
    joint_names: list[str],
) -> None:
    """post-spawn hook: instantiate a publisher per declared sensor with a known type."""
    del joint_names, name
    node = hooks.get_node()
    publishers: list[SensorPublisher] = []
    for sensor_spec, _urdf_sensor, cls in _planned_sensors(robot_entry):
        frame_id = f'{robot_entry.tf_prefix}{sensor_spec.frame}'
        publishers.append(cls(node, env_id, sensor_spec, frame_id, robot_entry, _element_prefix(robot_entry)))
    if publishers:
        _active[(env_id, robot_entry.tf_prefix)] = publishers


def _sensor_despawn(env_id: int, robot_entry: RobotEntry) -> None:
    for pub in _active.pop((env_id, robot_entry.tf_prefix), []):
        _next_due.pop(id(pub), None)
        pub.destroy()


def _sensor_pump() -> None:
    """pump: publish every active sensor whose rate window has elapsed."""
    store = get_scene_store()
    now = store.time
    for (env_id, _tf_prefix), publishers in _active.items():
        model = store.get_model(env_id)
        data = store.get_data(env_id)
        if model is None or data is None:
            continue
        for pub in publishers:
            due = _next_due.get(id(pub), 0.0)
            if now < due - _RATE_SLACK_S:
                continue
            period = 1.0 / pub.rate_hz
            _next_due[id(pub)] = due + period if due + period > now - period else now + period
            if pub._bound_model is not model:
                pub.rebind()
                pub._bound_model = model
            pub.publish(model, data)


_next_due: dict[int, float] = {}

_installed = False


def install_sensor_hooks() -> None:
    """Register the sensor pre-compile, post-spawn, and pump hooks once."""
    global _installed
    if _installed:
        return
    hooks.register_pre_compile(_sensor_pre_compile)
    hooks.register_post_spawn(_sensor_post_spawn)
    hooks.register_despawn(_sensor_despawn)
    hooks.register_pump(_sensor_pump)
    _installed = True
