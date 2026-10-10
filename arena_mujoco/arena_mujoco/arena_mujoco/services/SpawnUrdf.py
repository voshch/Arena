"""Real SpawnUrdf handler: load a URDF, attach it into the env spec, register its mapping."""

from __future__ import annotations

import logging
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
from arena_mujoco_msgs.srv import SpawnUrdf
from arena_robots.Sensor import SensorSpec

from arena_mujoco import hooks
from arena_mujoco.context import get_scene_store
from arena_mujoco.mesh_mat import MeshPart, mesh_parts, part_material
from arena_mujoco.rollers import add_rollers, roller_wheels
from arena_mujoco.scene import AssetCache, RejectedBodies, sanitize_id

from .utils import Service

if TYPE_CHECKING:
    from geometry_msgs.msg import Pose

_DROP_TAGS = ('ros2_control', 'transmission', 'gazebo')

_parts_by_file: dict[str, tuple[MeshPart, ...]] = {}

_logger = logging.getLogger(__name__)

_RGBA_UNSET = (0.5, 0.5, 0.5, 1.0)

_JOINT_ARMATURE = 0.2


_CAMERA_KINDS = frozenset({'camera', 'depth', 'depth_camera', 'rgbd_camera'})


@dataclass(frozen=True)
class ScanAxis:
    """Ray fan of a lidar along one axis, both end angles included."""

    samples: int
    min_angle: float
    max_angle: float


@dataclass(frozen=True)
class UrdfSensor:
    """One <gazebo reference><sensor> element of the robot URDF."""

    kind: str
    link: str
    pose: tuple[float, float, float, float, float, float]
    update_rate: float | None
    horizontal_fov: float | None
    width: int | None
    height: int | None
    near: float | None
    far: float | None
    horizontal: ScanAxis | None = None
    vertical: ScanAxis | None = None

    @property
    def is_camera(self) -> bool:
        return self.kind in _CAMERA_KINDS


def _number(element: ET.Element, path: str) -> float | None:
    text = element.findtext(path)
    return float(text) if text and text.strip() else None


def _whole(element: ET.Element, path: str) -> int | None:
    value = _number(element, path)
    return None if value is None else int(value)


def _scan_axis(element: ET.Element, axis: str) -> ScanAxis | None:
    samples = _whole(element, f'.//scan/{axis}/samples')
    low = _number(element, f'.//scan/{axis}/min_angle')
    high = _number(element, f'.//scan/{axis}/max_angle')
    if samples is None or low is None or high is None:
        return None
    return ScanAxis(samples=samples, min_angle=low, max_angle=high)


def _first_number(element: ET.Element, *paths: str) -> float | None:
    for path in paths:
        value = _number(element, path)
        if value is not None:
            return value
    return None


def urdf_sensors(root: ET.Element) -> dict[str, UrdfSensor]:
    """Sensor declarations of a URDF by <sensor> name, one referenced to a joint sits on that joint's child link."""
    child_links = {joint.attrib.get('name'): child.attrib.get('link') for joint in root.findall('joint') if (child := joint.find('child')) is not None}
    sensors: dict[str, UrdfSensor] = {}
    for gazebo in root.iter('gazebo'):
        reference = gazebo.attrib.get('reference')
        link = child_links.get(reference) or reference
        if not link:
            continue
        for element in gazebo.iter('sensor'):
            name = element.attrib.get('name')
            if not name:
                continue
            x, y, z, roll, pitch, yaw = (float(v) for v in element.findtext('pose', '0 0 0 0 0 0').split())
            sensors[name] = UrdfSensor(
                kind=element.attrib.get('type', ''),
                link=link.replace('-', '_'),
                pose=(x, y, z, roll, pitch, yaw),
                update_rate=_number(element, 'update_rate'),
                horizontal_fov=_number(element, './/horizontal_fov'),
                width=_whole(element, './/image/width'),
                height=_whole(element, './/image/height'),
                near=_first_number(element, './/clip/near', './/range/min'),
                far=_first_number(element, './/clip/far', './/range/max'),
                horizontal=_scan_axis(element, 'horizontal'),
                vertical=_scan_axis(element, 'vertical'),
            )
    return sensors


@dataclass(frozen=True)
class RobotEntry:
    """Spawn record mapping a robot's ROS names to its MJCF and TF ids."""

    robot_model: str
    tf_prefix: str
    base_frame: str
    prim_name: str
    localization: bool
    joint_names: tuple[str, ...]
    joint_states_topic: str
    cmd_vel_topic: str
    odom_topic: str
    command_interfaces: dict[str, tuple[str, ...]]
    urdf_sensors: dict[str, UrdfSensor]
    sensors: tuple[SensorSpec, ...]


_robots: dict[tuple[int, str], RobotEntry] = {}


def get_robot(env_id: int, name: str) -> RobotEntry | None:
    """Return the spawn record for (env_id, name), or None if no robot was spawned under it."""
    return _robots.get((env_id, name))


def frame_body(env_id: int, frame: str) -> str | None:
    """Body id of the robot link a TF frame id names, None when no robot of env_id owns the frame."""
    for (robot_env, _), entry in _robots.items():
        if robot_env == env_id and entry.tf_prefix and frame.startswith(entry.tf_prefix):
            prefix = entry.prim_name[: len(entry.prim_name) - len(entry.base_frame)]
            return f'{prefix}{frame[len(entry.tf_prefix) :].replace("-", "_")}'
    return None


def forget_robots(env_id: int, prefix: str) -> None:
    """Fire the despawn hooks for every robot of env_id whose root body id starts with prefix."""
    key = sanitize_id(prefix) if prefix else ''
    for robot_key, entry in list(_robots.items()):
        if robot_key[0] == env_id and entry.prim_name.startswith(key):
            hooks.fire_despawn(env_id, entry)
            del _robots[robot_key]


def _apply_mesh_parts(robot_spec: mujoco.MjSpec) -> None:
    """Give every mesh geom its file's material and add a sibling geom per further material of that file."""
    cache = AssetCache()
    for mesh in list(robot_spec.meshes):
        parts = _parts_by_file.get(mesh.file)
        if parts is None:
            continue
        names = [mesh.name]
        for index, part in enumerate(parts[1:], start=1):
            extra = robot_spec.add_mesh()
            extra.name = f'{mesh.name}_part{index}'
            extra.file = part.file
            extra.scale = list(mesh.scale)
            names.append(extra.name)
        for geom in list(robot_spec.geoms):
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH or geom.meshname != mesh.name:
                continue
            for name, part in zip(names, parts, strict=True):
                target = geom
                if name != mesh.name:
                    target = geom.parent.add_geom()
                    target.type = mujoco.mjtGeom.mjGEOM_MESH
                    target.meshname = name
                    target.pos = list(geom.pos)
                    target.quat = list(geom.quat)
                    target.contype = geom.contype
                    target.conaffinity = geom.conaffinity
                    target.group = geom.group
                    target.density = geom.density
                material = part_material(robot_spec, cache, part)
                if material is not None:
                    target.material = material
                    target.rgba = list(_RGBA_UNSET)
                elif target is not geom:
                    target.rgba = list(geom.rgba)


def _mujoco_compat(urdf_path: str) -> tuple[str, dict[str, tuple[str, ...]], dict[str, UrdfSensor]]:
    """Rewrite urdf_path into a MuJoCo-friendly URDF, return (temp path, command interfaces, sensors)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    sensors = urdf_sensors(root)

    command_interfaces: dict[str, tuple[str, ...]] = {}
    for rc in root.iter('ros2_control'):
        for joint_el in rc.findall('joint'):
            jname = joint_el.attrib.get('name')
            if not jname:
                continue
            ifaces = tuple(ci.attrib['name'] for ci in joint_el.findall('command_interface') if ci.attrib.get('name'))
            command_interfaces[jname.replace('-', '_')] = ifaces

    for tag in _DROP_TAGS:
        for el in root.findall(tag):
            root.remove(el)

    link_map: dict[str, str] = {}
    joint_map: dict[str, str] = {}
    for el in root.iter():
        if el.tag == 'link':
            name = el.attrib.get('name')
            if name and '-' in name:
                link_map[name] = name.replace('-', '_')
        elif el.tag == 'joint':
            name = el.attrib.get('name')
            if name and '-' in name:
                joint_map[name] = name.replace('-', '_')

    for el in root.iter():
        if el.tag in ('link', 'joint') and '-' in el.attrib.get('name', ''):
            el.attrib['name'] = el.attrib['name'].replace('-', '_')
        elif el.tag in ('parent', 'child'):
            link = el.attrib.get('link')
            if link in link_map:
                el.attrib['link'] = link_map[link]
        elif el.tag == 'mimic':
            joint = el.attrib.get('joint')
            if joint in joint_map:
                el.attrib['joint'] = joint_map[joint]

    for link in root.findall('link'):
        for shape in [*link.findall('visual'), *link.findall('collision')]:
            mesh = shape.find('geometry/mesh')
            filename = '' if mesh is None else mesh.attrib.get('filename', '').removeprefix('file://')
            if not filename:
                continue
            try:
                parts = mesh_parts(filename)
            except Exception as exc:  # noqa: BLE001 - an unreadable mesh drops that shape, as Gazebo does
                _logger.warning(f'{link.attrib.get("name")} {shape.tag} dropped, mesh {filename!r} unreadable: {exc!r}')
                link.remove(shape)
                continue
            if not parts:
                link.remove(shape)
                continue
            _parts_by_file[parts[0].file] = parts
            mesh.attrib['filename'] = parts[0].file

    for joint_el in root.findall('joint'):
        for tag in ('parent', 'child'):
            for extra in joint_el.findall(tag)[1:]:
                joint_el.remove(extra)

    compiler = ET.SubElement(ET.SubElement(root, 'mujoco'), 'compiler')
    compiler.set('discardvisual', 'false')
    compiler.set('balanceinertia', 'true')
    compiler.set('fusestatic', 'false')
    compiler.set('strippath', 'false')

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_mjcompat.urdf', mode='w')
    tree.write(tmp.name, encoding='unicode', xml_declaration=True)
    tmp.close()
    return tmp.name, command_interfaces, sensors


def _pose_to_mujoco(
    pose: Pose,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Map a geometry_msgs/Pose (orientation xyzw) to MuJoCo (pos, quat wxyz)."""
    p = pose.position
    o = pose.orientation
    return (p.x, p.y, p.z), (o.w, o.x, o.y, o.z)


def _attach_robot(
    spec: mujoco.MjSpec,
    cache: AssetCache,
    robot_spec: mujoco.MjSpec,
    prefix: str,
    base_frame: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    localization: bool,
) -> mujoco.MjsBody:
    """Attach robot_spec under spec's worldbody with prefix, return the attached root body."""
    known = {asset.name for asset in (*spec.textures, *spec.materials, *spec.meshes)}
    frame = spec.worldbody.add_frame(pos=list(pos), quat=list(quat))
    spec.attach(robot_spec, prefix=prefix, frame=frame)
    for asset in (*spec.textures, *spec.materials, *spec.meshes):
        if asset.name not in known:
            cache.add(None, asset)

    base = spec.body(f'{prefix}{base_frame}')
    if base is None:
        raise RuntimeError(f"attached robot base '{prefix}{base_frame}' not found after attach")

    if localization:
        base.add_freejoint().name = f'{prefix}{base_frame}_free'

    return base


def _spawn_urdf_cb(
    request: SpawnUrdf.Request,
    response: SpawnUrdf.Response,
) -> SpawnUrdf.Response:
    store = get_scene_store()
    env_id = int(request.env_id)
    store.ensure_env(env_id)

    compat_path, command_interfaces, sensors = _mujoco_compat(request.urdf_path)
    robot_spec = mujoco.MjSpec.from_file(compat_path)
    _apply_mesh_parts(robot_spec)

    for joint in robot_spec.joints:
        joint.armature = max(joint.armature, _JOINT_ARMATURE)

    joint_names = tuple(joint.name for joint in robot_spec.joints if joint.name)
    add_rollers(robot_spec, roller_wheels(ET.parse(request.urdf_path).getroot()))

    prefix = f'{sanitize_id(request.name)}_'
    pos, quat = _pose_to_mujoco(request.pose)

    captured: dict[str, mujoco.MjSpec | mujoco.MjsBody] = {}
    prim_name = store.add_body(
        env_id,
        lambda spec: _attach(
            captured,
            spec,
            store.asset_cache(env_id),
            robot_spec,
            prefix,
            request.base_frame.replace('-', '_'),
            pos,
            quat,
            request.localization,
        ),
    )

    entry = RobotEntry(
        robot_model=request.robot_model,
        tf_prefix=request.tf_prefix,
        base_frame=request.base_frame,
        prim_name=prim_name,
        localization=request.localization,
        joint_names=joint_names,
        joint_states_topic=request.joint_states_topic,
        cmd_vel_topic=request.cmd_vel_topic,
        odom_topic=request.odom_topic,
        command_interfaces=command_interfaces,
        urdf_sensors=sensors,
        sensors=tuple(SensorSpec(name=item.name, type=item.type, topic=item.topic, frame=item.frame, sensor=item.sensor or None) for item in request.sensors),
    )

    try:
        hooks.fire_pre_compile(env_id, captured['spec'], captured['root'], entry, list(joint_names))
    except ValueError as exc:
        store.discard_pending(env_id)
        hooks.get_node().get_logger().error(f'robot {request.name!r} not spawned: {exc}')
        return response
    try:
        store.recompile(env_id)
    except RejectedBodies as exc:
        hooks.get_node().get_logger().error(f'robot {request.name!r} not spawned: {exc}')
        return response

    _robots[(env_id, request.name)] = entry
    hooks.fire_post_spawn(env_id, request.name, entry, list(joint_names))

    response.path = prim_name
    return response


def _attach(
    captured: dict[str, mujoco.MjSpec | mujoco.MjsBody],
    spec: mujoco.MjSpec,
    cache: AssetCache,
    robot_spec: mujoco.MjSpec,
    prefix: str,
    base_frame: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    localization: bool,
) -> mujoco.MjsBody:
    """Attach and capture spec plus root body so the caller can hand them to pre-compile hooks."""
    attached = _attach_robot(spec, cache, robot_spec, prefix, base_frame, pos, quat, localization)
    captured['spec'] = spec
    captured['root'] = attached
    return attached


spawn_urdf_service = Service(
    srv_type=SpawnUrdf,
    srv_name='mujoco/SpawnUrdf',
    callback=_spawn_urdf_cb,
)

__all__ = ['frame_body', 'get_robot', 'spawn_urdf_service']
