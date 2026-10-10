import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from arena_robots.Sensor import SensorSpec, SensorType
from std_msgs.msg import Header

from arena_mujoco import context
from arena_mujoco.scene import SceneStore
from arena_mujoco.sensors.camera import (
    CameraImagePublisher,
    CameraPointCloudPublisher,
    CameraRig,
    body_rays,
)
from arena_mujoco.sensors.contact import contacts_message, outside_contacts
from arena_mujoco.sensors.core import get_publisher_class
from arena_mujoco.sensors.laser import LidarGeometry, PointCloudPublisher, cast_rays, lidar_geometry, own_geoms, worker_datas
from arena_mujoco.services.SpawnUrdf import UrdfSensor, urdf_sensors

_URDF = """<robot name="r">
  <link name="cam-link"/>
  <gazebo reference="cam-link">
    <sensor name="front" type="rgbd_camera">
      <pose>0.1 0 0.2 0 0.5 0</pose>
      <update_rate>15</update_rate>
      <camera>
        <horizontal_fov>1.25</horizontal_fov>
        <image><width>320</width><height>240</height></image>
        <clip><near>0.3</near><far>8</far></clip>
      </camera>
    </sensor>
  </gazebo>
  <gazebo reference="scan_link">
    <sensor name="scan" type="gpu_lidar"><update_rate>10</update_rate></sensor>
  </gazebo>
  <joint name="imu_joint" type="fixed"><parent link="cam-link"/><child link="imu_frame"/></joint>
  <gazebo reference="imu_joint">
    <sensor name="imu" type="imu"><update_rate>100</update_rate></sensor>
  </gazebo>
</robot>"""

_PREFIX = 'robot_sensor_'
_CAMERA = f'{_PREFIX}front_cam'
_IMAGE = SensorSpec(name='front_image', type=SensorType.IMAGE, topic='image', frame='cam_link', sensor='front')


def _camera(pose: tuple[float, ...] = (0.0,) * 6, far: float = 8.0) -> UrdfSensor:
    return UrdfSensor(
        kind='rgbd_camera',
        link='cam_link',
        pose=pose,
        update_rate=15.0,
        horizontal_fov=1.25,
        width=320,
        height=240,
        near=0.3,
        far=far,
    )


def _box(name: str, pos: tuple[float, float, float], half: tuple[float, float, float], rgba: list[float], free: bool = False):
    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name=name, pos=list(pos))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), rgba=rgba)
        if free:
            body.add_freejoint().name = f'{name}_free'
        return body

    return build


def _carrier(sensor: UrdfSensor, height: float):
    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name='cam_link', pos=[0.0, 0.0, height])
        CameraImagePublisher.add_model_elements(spec, body, _IMAGE, _PREFIX, sensor)
        return body

    return build


@pytest.fixture
def store() -> SceneStore:
    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    return store


@pytest.fixture
def offscreen(store: SceneStore) -> None:
    try:
        mujoco.Renderer(store.get_model(0), 8, 8).close()
    except Exception as exc:  # noqa: BLE001 - any GL failure means this machine cannot render
        pytest.skip(f'no offscreen GL: {exc}')


def _scene(store: SceneStore, sensor: UrdfSensor, height: float = 0.5) -> tuple[CameraRig, mujoco.MjModel, mujoco.MjData]:
    store.add_body(0, _carrier(sensor, height))
    store.add_body(0, _box('target', (2.0, 0.5, 0.5), (0.2, 0.2, 0.5), [1.0, 0.0, 0.0, 1.0]))
    store.add_body(0, _box('wall', (4.0, 0.0, 1.0), (0.1, 5.0, 1.0), [0.0, 0.0, 1.0, 1.0]))
    store.recompile(0)
    model, data = store.get_model(0), store.get_data(0)
    mujoco.mj_forward(model, data)
    return CameraRig(0, _CAMERA, sensor), model, data


def _red(color: np.ndarray) -> np.ndarray:
    return (color[:, :, 0] > 120) & (color[:, :, 1] < 60) & (color[:, :, 2] < 60)


def test_urdf_sensor_fields_come_from_the_gazebo_block():
    sensors = urdf_sensors(ET.fromstring(_URDF))
    front = sensors['front']
    assert front.is_camera
    assert front.link == 'cam_link'
    assert front.pose == (0.1, 0.0, 0.2, 0.0, 0.5, 0.0)
    assert (front.width, front.height) == (320, 240)
    assert (front.horizontal_fov, front.near, front.far, front.update_rate) == (1.25, 0.3, 8.0, 15.0)
    assert not sensors['scan'].is_camera
    assert sensors['scan'].width is None
    assert sensors['imu'].link == 'imu_frame'


def test_pointcloud_class_follows_the_backing_sensor():
    assert get_publisher_class(SensorType.POINTCLOUD, True) is CameraPointCloudPublisher
    assert get_publisher_class(SensorType.POINTCLOUD, False) is PointCloudPublisher
    assert get_publisher_class(SensorType.IMAGE, True) is CameraImagePublisher


def test_specs_of_one_sensor_share_one_camera_sized_from_the_urdf(store: SceneStore):
    depth = SensorSpec(name='front_depth', type=SensorType.DEPTH, topic='depth', frame='cam_link', sensor='front')

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name='cam_link')
        CameraImagePublisher.add_model_elements(spec, body, _IMAGE, _PREFIX, _camera())
        CameraImagePublisher.add_model_elements(spec, body, depth, _PREFIX, _camera())
        return body

    store.add_body(0, build)
    store.recompile(0)
    model = store.get_model(0)
    assert model.ncam == 1
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0) == _CAMERA
    assert model.vis.global_.offwidth >= 320
    assert model.vis.global_.offheight >= 240
    assert math.radians(model.cam_fovy[0]) == pytest.approx(2.0 * math.atan(math.tan(1.25 / 2.0) * 240 / 320))


def test_camera_looks_along_x_with_left_on_the_left(store: SceneStore, offscreen: None):
    rig, model, data = _scene(store, _camera())
    color = rig.color(model, data)
    assert color.shape == (240, 320, 3)
    rows, columns = np.nonzero(_red(color))
    assert len(columns) > 500
    assert columns.mean() < 130
    assert rows.mean() == pytest.approx(120, abs=4)


def test_depth_is_planar_meters_clipped_to_the_sensor_range(store: SceneStore, offscreen: None):
    rig, model, data = _scene(store, _camera())
    depth = rig.depth(model, data)
    assert depth.dtype == np.float32
    assert depth[120, 160] == pytest.approx(3.9, abs=0.02)
    assert depth[120, 300] == pytest.approx(3.9, abs=0.02)

    short = CameraRig(0, _CAMERA, _camera(far=3.0))
    assert np.isposinf(short.depth(model, data)[120, 160])


def test_sensor_pose_pitch_tilts_the_camera_down(store: SceneStore, offscreen: None):
    pitch = 0.5
    rig, model, data = _scene(store, _camera(pose=(0.0, 0.0, 0.0, 0.0, pitch, 0.0)), height=1.0)
    depth = rig.depth(model, data)
    assert depth[120, 160] == pytest.approx(1.0 / math.sin(pitch), abs=0.03)


def test_cloud_points_are_in_body_axes(store: SceneStore, offscreen: None):
    rig, model, data = _scene(store, _camera())
    points = body_rays(rig.width, rig.height, rig.focal) * rig.depth(model, data)[:, :, None]
    target = points[_red(rig.color(model, data))]
    assert target[:, 0].mean() == pytest.approx(1.8, abs=0.03)
    assert target[:, 1].mean() == pytest.approx(0.5, abs=0.05)
    assert target[:, 2].min() == pytest.approx(-0.5, abs=0.05)


def test_contact_reports_touching_bodies_outside_the_own_tree(store: SceneStore):
    grey = [0.5, 0.5, 0.5, 1.0]
    store.add_body(0, _box('resting', (0.0, 0.0, 0.1), (0.1, 0.1, 0.1), grey, free=True))
    store.add_body(0, _box('stacked', (0.0, 0.0, 0.3), (0.1, 0.1, 0.1), grey, free=True))
    store.add_body(0, _box('hovering', (3.0, 0.0, 2.0), (0.1, 0.1, 0.1), grey))
    store.recompile(0)
    store.step(100)
    model, data = store.get_model(0), store.get_data(0)

    def touching(name: str) -> list[int]:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return sorted({int(model.geom_bodyid[other]) for _, _, other in outside_contacts(model, data, body)})

    resting = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'resting')
    stacked = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'stacked')
    assert touching('resting') == [0, stacked]
    assert touching('stacked') == [resting]
    assert touching('hovering') == []

    message = contacts_message(model, data, stacked, Header(frame_id='bumper'))
    (contact,) = message.contacts
    assert (contact.collision1.name, contact.collision2.name) == ('stacked', 'resting')
    assert len(contact.positions) == len(contact.depths) == 4
    assert all(position.z == pytest.approx(0.2, abs=0.01) for position in contact.positions)
    assert all(normal.z == pytest.approx(1.0) for normal in contact.normals)
    assert contacts_message(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hovering'), Header()).contacts == []


_LIDAR_URDF = """<robot name="r">
  <gazebo reference="lidar_link">
    <sensor name="gpu_lidar" type="gpu_lidar">
      <pose>0 0 0.142 0 0 0</pose>
      <update_rate>10</update_rate>
      <lidar>
        <scan>
          <horizontal><samples>640</samples><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
          <vertical><samples>16</samples><min_angle>-0.261799</min_angle><max_angle>0.261799</max_angle></vertical>
        </scan>
        <range><min>0.08</min><max>12.0</max></range>
      </lidar>
    </sensor>
  </gazebo>
</robot>"""


def test_lidar_layout_comes_from_its_own_urdf_element():
    sensor = urdf_sensors(ET.fromstring(_LIDAR_URDF))['gpu_lidar']
    geometry = lidar_geometry('unused', sensor)
    assert (geometry.num_beams, geometry.range_min, geometry.range_max, geometry.rate_hz) == (640, 0.08, 12.0, 10.0)
    assert geometry.angle_increment == pytest.approx(2 * 3.14159 / 639)
    assert len(geometry.elevations) == 16
    assert geometry.scan_elevation == pytest.approx(-0.261799 + 8 * 2 * 0.261799 / 15)
    rays = geometry.directions(geometry.elevations)
    assert rays.shape == (16 * 640, 3)
    assert np.linalg.norm(rays, axis=1) == pytest.approx(1.0)
    assert rays[0] == pytest.approx((-math.cos(0.261799), 0.0, -math.sin(0.261799)), abs=1e-5)


def test_rays_skip_the_carrier_and_range_limits_follow_rep_117(store: SceneStore):
    def carrier(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name='base', pos=[0.0, 0.0, 0.5])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.3, 0.3])
        body.add_site(name='lidar', pos=[0.0, 0.0, 0.142])
        return body

    store.add_body(0, carrier)
    store.add_body(0, _box('wall', (3.0, 0.0, 1.0), (0.1, 5.0, 1.0), [0.0, 0.0, 1.0, 1.0]))
    store.add_body(0, _box('pole', (0.0, 0.34, 1.0), (0.01, 0.01, 1.0), [0.0, 1.0, 0.0, 1.0]))
    store.recompile(0)
    model, data = store.get_model(0), store.get_data(0)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'lidar')

    down = math.radians(-30.0)
    rays = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [math.cos(down), 0.0, math.sin(down)],
        ]
    )
    ranges = cast_rays(model, data, site, rays, own_geoms(model, site), range_min=0.5, range_max=8.0)
    assert ranges[0] == pytest.approx(2.9, abs=1e-3)
    assert np.isposinf(ranges[1])
    assert np.isneginf(ranges[2])
    assert ranges[3] == pytest.approx(0.642 / math.sin(math.radians(30.0)), abs=1e-3)
    assert model.geom_group[own_geoms(model, site)].tolist() == [0]


def test_a_fan_cast_across_threads_keeps_ray_order(store: SceneStore):
    def carrier(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name='base', pos=[0.0, 0.0, 0.5])
        body.add_site(name='lidar')
        return body

    def dome(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name='dome', pos=[0.0, 0.0, 0.5])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.3, 0.0, 0.0], contype=0, conaffinity=0)
        body.add_freejoint().name = 'dome_free'
        return body

    store.add_body(0, carrier)
    store.add_body(0, dome)
    store.add_body(0, _box('wall', (1.0, 0.0, 1.0), (0.1, 5.0, 1.0), [0.0, 0.0, 1.0, 1.0]))
    store.add_body(0, _box('pole', (-1.0, 1.0, 1.0), (0.05, 0.05, 1.0), [0.0, 1.0, 0.0, 1.0]))
    store.recompile(0)
    model, data = store.get_model(0), store.get_data(0)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'lidar')
    nothing = np.array([], dtype=int)

    rings = tuple(np.linspace(-0.26, 0.26, 16).tolist())
    rays = LidarGeometry(-math.pi, math.pi, 640, rings, 0.05, 8.0, 10.0).directions(rings)
    assert len(worker_datas(model, len(rays))) in (0, 2, 4, 8)
    assert worker_datas(model, 640) == []
    workers = [mujoco.MjData(model) for _ in range(4)]
    fan = cast_rays(model, data, site, rays, nothing, range_min=0.05, range_max=8.0, workers=workers)
    one_by_one = np.concatenate([cast_rays(model, data, site, rays[i : i + 64], nothing, range_min=0.05, range_max=8.0) for i in range(0, len(rays), 64)])
    assert np.isfinite(fan).sum() > 500
    np.testing.assert_array_equal(fan, one_by_one)


def test_near_plane_stays_at_a_centimeter_with_bodies_parked_far_away(store: SceneStore, offscreen: None):
    store.alloc_mocap(0, 2)
    rig, model, data = _scene(store, _camera())
    assert model.vis.map.znear * model.stat.extent == pytest.approx(0.01)
    assert rig.depth(model, data)[120, 160] == pytest.approx(3.9, abs=0.02)
