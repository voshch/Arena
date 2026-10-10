"""Ray-cast lidar publishers: LaserScan and the multi-ring PointCloud2."""

from __future__ import annotations

import abc
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import sensor_msgs.msg
import std_msgs.msg
from sensor_msgs_py import point_cloud2

from arena_mujoco.context import get_scene_store
from arena_mujoco.sensors.core import SensorPublisher, make_stamp, register_sensor, sensor_pose

if TYPE_CHECKING:
    from collections.abc import Sequence

    import rclpy.node
    from arena_robots.Sensor import SensorSpec

    from arena_mujoco.services.SpawnUrdf import RobotEntry, UrdfSensor

_DEFAULT_RANGE_MIN = 0.05
_DEFAULT_RANGE_MAX = 10.0
_DEFAULT_RATE_HZ = 10.0

_SELF_GROUP = 5
_RAY_GROUPS = np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8)

_CAST_WORKERS = min(8, os.cpu_count() or 1)
_MIN_RAYS_PER_WORKER = 512
_cast_pool = ThreadPoolExecutor(max_workers=_CAST_WORKERS, thread_name_prefix='mj_ray')


@dataclass(frozen=True)
class LidarGeometry:
    """Beam layout of one lidar, end angles included."""

    angle_min: float
    angle_max: float
    num_beams: int
    elevations: tuple[float, ...]
    range_min: float
    range_max: float
    rate_hz: float

    @property
    def angle_increment(self) -> float:
        return (self.angle_max - self.angle_min) / (self.num_beams - 1) if self.num_beams > 1 else 0.0

    @property
    def scan_elevation(self) -> float:
        """Elevation of the ring a LaserScan carries, the one ros_gz_bridge picks from a multi-ring scan."""
        return self.elevations[len(self.elevations) // 2]

    def directions(self, elevations: tuple[float, ...]) -> np.ndarray:
        """Unit ray directions in the sensor frame, ring after ring, shape (rings * beams, 3)."""
        azimuth = np.linspace(self.angle_min, self.angle_max, self.num_beams)
        elevation = np.asarray(elevations)[:, None]
        return np.stack(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation) * np.ones_like(azimuth),
            ],
            axis=-1,
        ).reshape(-1, 3)


def lidar_geometry(robot_model: str, urdf_sensor: UrdfSensor | None) -> LidarGeometry:
    """Beam layout from the URDF sensor element, else from the robot's caps/mobile.yaml laser block, else a 360 beam ring."""
    if urdf_sensor is not None and urdf_sensor.horizontal is not None:
        horizontal = urdf_sensor.horizontal
        vertical = urdf_sensor.vertical
        elevations = (0.0,) if vertical is None else tuple(np.linspace(vertical.min_angle, vertical.max_angle, vertical.samples).tolist())
        return LidarGeometry(
            angle_min=horizontal.min_angle,
            angle_max=horizontal.max_angle,
            num_beams=horizontal.samples,
            elevations=elevations,
            range_min=_DEFAULT_RANGE_MIN if urdf_sensor.near is None else urdf_sensor.near,
            range_max=_DEFAULT_RANGE_MAX if urdf_sensor.far is None else urdf_sensor.far,
            rate_hz=urdf_sensor.update_rate or _DEFAULT_RATE_HZ,
        )

    from arena_robots.Robot import RobotIdentifier

    mobile = RobotIdentifier(robot_model).resolve_sync().mobile
    laser = mobile.laser if mobile is not None else None
    if laser is None:
        return LidarGeometry(-math.pi, math.pi, 360, (0.0,), _DEFAULT_RANGE_MIN, _DEFAULT_RANGE_MAX, _DEFAULT_RATE_HZ)
    return LidarGeometry(
        angle_min=laser.angle.min,
        angle_max=laser.angle.max,
        num_beams=laser.num_beams,
        elevations=(0.0,),
        range_min=_DEFAULT_RANGE_MIN,
        range_max=laser.range,
        rate_hz=float(laser.update_rate),
    )


def own_geoms(model: mujoco.MjModel, site_id: int) -> np.ndarray:
    """Ids of every geom in the kinematic tree that carries the site."""
    root = model.body_rootid[model.site_bodyid[site_id]]
    return np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)


def worker_datas(model: mujoco.MjModel, rays: int) -> list[mujoco.MjData]:
    """One MjData per thread worth using for a fan of that many rays, none for a small fan."""
    count = min(_CAST_WORKERS, rays // _MIN_RAYS_PER_WORKER)
    return [mujoco.MjData(model) for _ in range(count)] if count > 1 else []


def _cast(model: mujoco.MjModel, data: mujoco.MjData, origin: np.ndarray, rays: np.ndarray, cutoff: float) -> np.ndarray:
    dist = np.empty(len(rays), dtype=np.float64)
    geomid = np.empty(len(rays), dtype=np.int32)
    mujoco.mj_multiRay(model, data, origin, rays.ravel(), _RAY_GROUPS, 1, -1, geomid, dist, None, len(rays), cutoff)
    return dist


def _cast_on(model: mujoco.MjModel, data: mujoco.MjData, worker: mujoco.MjData, origin: np.ndarray, rays: np.ndarray, cutoff: float) -> np.ndarray:
    mujoco.mj_copyData(worker, model, data)
    return _cast(model, worker, origin, rays, cutoff)


def cast_rays(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    rays: np.ndarray,
    hidden: np.ndarray,
    range_min: float,
    range_max: float,
    workers: Sequence[mujoco.MjData] = (),
) -> np.ndarray:
    """Range per sensor-frame ray from the site, the hidden geoms unseen: +inf without a hit in range_max, -inf below range_min, cast on one thread per worker."""
    origin = np.array(data.site_xpos[site_id], dtype=np.float64)
    xmat = np.array(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    world_rays = rays @ xmat.T
    groups = model.geom_group[hidden].copy()
    model.geom_group[hidden] = _SELF_GROUP
    if len(workers) > 1:
        dist = np.empty(len(rays), dtype=np.float64)
        chunks = [np.ascontiguousarray(world_rays[start :: len(workers)]) for start in range(len(workers))]
        parts = _cast_pool.map(lambda worker, chunk: _cast_on(model, data, worker, origin, chunk, range_max), workers, chunks)
        for start, part in enumerate(parts):
            dist[start :: len(workers)] = part
    else:
        dist = _cast(model, data, origin, np.ascontiguousarray(world_rays), range_max)
    model.geom_group[hidden] = groups
    ranges = dist.astype(np.float32)
    ranges[(dist < 0.0) | (dist > range_max)] = np.inf
    ranges[(dist >= 0.0) & (dist < range_min)] = -np.inf
    return ranges


class _LidarPublisher(SensorPublisher):
    """Ray fan from a site at the URDF sensor pose, blind to the robot that carries it."""

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
        self.geometry = lidar_geometry(robot.robot_model, self.urdf_sensor)
        self._rays = self.geometry.directions(self._elevations())
        self._pub = node.create_publisher(self._MSG_TYPE, spec.topic, 10)
        self._site_name = f"{element_prefix}{spec.name}_site"
        self._site_id: int = -1
        self._own_geoms: np.ndarray | None = None
        self._workers: list[mujoco.MjData] = []

    @abc.abstractmethod
    def _elevations(self) -> tuple[float, ...]:
        """Ring elevations this publisher casts."""

    def rebind(self) -> None:
        self._site_id = -1
        self._own_geoms = None
        self._workers = []

    @classmethod
    def add_model_elements(
        cls,
        spec_builder: mujoco.MjSpec,
        mount_body: mujoco.MjsBody,
        sensor_spec: SensorSpec,
        element_prefix: str,
        urdf_sensor: UrdfSensor | None,
    ) -> None:
        """Add a site at the sensor pose."""
        del spec_builder
        pos, quat = sensor_pose(urdf_sensor)
        mount_body.add_site(name=f"{element_prefix}{sensor_spec.name}_site", pos=pos, quat=quat.tolist())

    @property
    def rate_hz(self) -> float:
        return self.geometry.rate_hz

    def _cast(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
        """Range per ray of this publisher, None before the site exists."""
        if self._site_id < 0:
            self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._site_name)
            if self._site_id < 0:
                return None
        if self._own_geoms is None:
            self._own_geoms = own_geoms(model, self._site_id)
            self._workers = worker_datas(model, len(self._rays))
        return cast_rays(model, data, self._site_id, self._rays, self._own_geoms, self.geometry.range_min, self.geometry.range_max, self._workers)

    def _header(self) -> std_msgs.msg.Header:
        return std_msgs.msg.Header(stamp=make_stamp(get_scene_store().time), frame_id=self.frame_id)


@register_sensor("laserscan")
class LaserScanPublisher(_LidarPublisher):
    """One ring of the lidar as sensor_msgs/LaserScan."""

    _MSG_TYPE = sensor_msgs.msg.LaserScan

    def _elevations(self) -> tuple[float, ...]:
        return (self.geometry.scan_elevation,)

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        ranges = self._cast(model, data)
        if ranges is None:
            return
        geometry = self.geometry
        msg = sensor_msgs.msg.LaserScan()
        msg.header = self._header()
        msg.angle_min = float(geometry.angle_min)
        msg.angle_max = float(geometry.angle_max)
        msg.angle_increment = geometry.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / geometry.rate_hz
        msg.range_min = float(geometry.range_min)
        msg.range_max = float(geometry.range_max)
        msg.ranges = ranges.tolist()
        msg.intensities = []
        self._pub.publish(msg)


@register_sensor("pointcloud")
class PointCloudPublisher(_LidarPublisher):
    """Every ring of the lidar as an XYZ cloud in the sensor frame, cast only while subscribed."""

    _MSG_TYPE = sensor_msgs.msg.PointCloud2

    def _elevations(self) -> tuple[float, ...]:
        return self.geometry.elevations

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._pub.get_subscription_count() == 0:
            return
        ranges = self._cast(model, data)
        if ranges is None:
            return
        hit = np.isfinite(ranges)
        points = (self._rays[hit] * ranges[hit, None]).astype(np.float32)
        self._pub.publish(point_cloud2.create_cloud_xyz32(self._header(), points))


__all__ = ["LaserScanPublisher", "LidarGeometry", "PointCloudPublisher", "cast_rays", "lidar_geometry", "own_geoms", "worker_datas"]
