import asyncio
import functools
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import aiofiles
import attrs

from . import Model, ModelProvider, ModelType

_PACKAGE_URI = "package://"

ET.register_namespace("gz", "http://gazebosim.org/schema")

_OPTIM_MAP: dict[str, frozenset[str]] = {
    'no_camera': frozenset({'camera', 'depth', 'rgbd_camera'}),
    'no_lidar': frozenset({'ray', 'gpu_lidar'}),
}


def _strip_sensors(root: ET.Element, tokens: set[str]) -> None:
    disabled_types: set[str] = set()
    unknown: set[str] = set()
    for tok in tokens:
        m = _OPTIM_MAP.get(tok)
        if m is None:
            unknown.add(tok)
        else:
            disabled_types |= m
    if unknown:
        print(f"[urdf.optim] ignoring unknown token(s): {sorted(unknown)}", file=sys.stderr)
    if not disabled_types:
        return
    for parent in root.iter():
        for child in list(parent):
            if child.tag.rpartition('}')[-1] == 'sensor' and child.attrib.get('type') in disabled_types:
                parent.remove(child)


def _patch_sensor_topics(root: ET.Element, patches: list[tuple[str, str, str]]) -> None:
    for sensor_name, child_path, value in patches:
        sensor = next(
            (elem for elem in root.iter() if elem.tag.rpartition('}')[-1] == 'sensor' and elem.attrib.get('name') == sensor_name),
            None,
        )
        if sensor is None:
            print(f"[urdf.sensor_topics] no <sensor name={sensor_name!r}> in this URDF variant, skipping patch", file=sys.stderr)
            continue
        node = sensor
        for part in child_path.split('/'):
            child = next((c for c in node if c.tag.rpartition('}')[-1] == part), None)
            if child is None:
                child = ET.SubElement(node, part)
            node = child
        node.text = value


def _ensure_effort_state(root: ET.Element) -> None:
    for control in root.iter():
        if control.tag.rpartition('}')[-1] != 'ros2_control':
            continue
        for joint in control:
            if joint.tag.rpartition('}')[-1] != 'joint':
                continue
            has_effort = any(child.tag.rpartition('}')[-1] == 'state_interface' and child.attrib.get('name') == 'effort' for child in joint)
            if not has_effort:
                ET.SubElement(joint, 'state_interface', {'name': 'effort'})


_GAZEBO_ROS2_CONTROL_PLUGIN = 'gz_ros2_control/GazeboSimSystem'


def _control_plugin_text(control: ET.Element) -> str | None:
    plugin = next((el for el in control.iter() if el.tag.rpartition('}')[-1] == 'plugin'), None)
    return plugin.text if plugin is not None else None


_LOCKABLE_JOINT_TYPES = frozenset({'prismatic', 'revolute'})
_JOINT_MOTION_CHILDREN = frozenset({'axis', 'limit', 'dynamics', 'calibration', 'safety_controller'})


def _lock_passive_joints(root: ET.Element) -> None:
    """Convert every ``<joint>`` with ``<limit effort="0">`` to a fixed joint.
    Zero effort makes a joint unactuatable per the URDF spec, so it only exists as
    passive compliance (e.g. create3's sprung wheel_drop suspension), and neither
    simulator models the spring: the joint collapses under gravity and the chassis
    beaches on its belly."""
    for joint in root.iter('joint'):
        if joint.attrib.get('type') not in _LOCKABLE_JOINT_TYPES:
            continue
        limit = joint.find('limit')
        if limit is None or float(limit.attrib.get('effort', 'inf')) != 0.0:
            continue
        joint.set('type', 'fixed')
        for child in [el for el in joint if el.tag in _JOINT_MOTION_CHILDREN]:
            joint.remove(child)
        print(f"[urdf.joints] locked passive zero-effort joint {joint.attrib.get('name')!r}", file=sys.stderr)


_HW_PLUGIN_RESOURCE = 'hardware_interface__pluginlib__plugin'


@functools.cache
def _registered_hardware_plugins() -> frozenset[str]:
    """Hardware plugin classes declared in the ament index, i.e. exactly the set a
    controller_manager in this environment can load."""
    try:
        from ament_index_python.resources import get_resource, get_resources
    except ModuleNotFoundError:
        return frozenset()
    names: set[str] = set()
    for pkg in get_resources(_HW_PLUGIN_RESOURCE):
        content, prefix = get_resource(_HW_PLUGIN_RESOURCE, pkg)
        for rel in filter(None, (line.strip() for line in content.splitlines())):
            try:
                manifest = ET.parse(os.path.join(prefix, rel))
            except (OSError, ET.ParseError) as e:
                print(f"[urdf.ros2_control] skipping unreadable plugin manifest of {pkg}: {e}", file=sys.stderr)
                continue
            for cls in manifest.getroot().iter('class'):
                name = cls.attrib.get('name') or cls.attrib.get('type')
                if name:
                    names.add(name)
    return frozenset(names)


def _strip_unregistered_ros2_control(root: ET.Element, registered: frozenset[str]) -> None:
    """Drop every top-level ``<ros2_control>`` block whose hardware plugin is not in
    ``registered`` (vendored descriptions leak gazebo-classic blocks that crash the
    external controller_manager), plus the classic ``libgazebo_ros2_control.so`` host
    plugin. The canonical ``gz_ros2_control/GazeboSimSystem`` block always survives:
    gazebo consumes it in-sim, the isaac adapter rewrites it. No-op when ``registered``
    is empty (no ament index to trust)."""
    if not registered:
        print("[urdf.ros2_control] no registered hardware plugins discovered, keeping all <ros2_control> blocks", file=sys.stderr)
        return
    keep = registered | {_GAZEBO_ROS2_CONTROL_PLUGIN}
    for control in [el for el in root if el.tag.rpartition('}')[-1] == 'ros2_control']:
        plugin = _control_plugin_text(control)
        if plugin is None or plugin.strip() not in keep:
            root.remove(control)
            print(f"[urdf.ros2_control] stripped <ros2_control name={control.attrib.get('name')!r}>: no loadable hardware plugin ({plugin!r})", file=sys.stderr)
    for gazebo in [el for el in root if el.tag.rpartition('}')[-1] == 'gazebo']:
        classic = [el for el in gazebo if el.tag.rpartition('}')[-1] == 'plugin' and el.attrib.get('filename') == 'libgazebo_ros2_control.so']
        if not classic:
            continue
        for plugin_el in classic:
            gazebo.remove(plugin_el)
        if len(gazebo) == 0:
            root.remove(gazebo)
        print("[urdf.ros2_control] stripped classic libgazebo_ros2_control.so plugin element", file=sys.stderr)


def _joint_element(joint: dict) -> ET.Element:
    el = ET.Element('joint', {'name': str(joint['name'])})
    # explicit jazzy ros2_control mimic attribute, kinematics come from the URDF mimic tag
    if joint.get('mimic'):
        el.set('mimic', 'true')
    for iface in joint.get('command_interfaces', []):
        ET.SubElement(el, 'command_interface', {'name': str(iface)})
    for iface in joint.get('state_interfaces', []):
        ET.SubElement(el, 'state_interface', {'name': str(iface)})
    return el


def _inject_ros2_control_joints(root: ET.Element, joints: list[dict]) -> None:
    """Merge every top-level ``<ros2_control>`` tag in ``root`` into the one whose
    hardware plugin is ``gz_ros2_control/GazeboSimSystem`` (the chassis-emitted tag;
    arena_robots.catalog.render_wrapper_xacro renders the chassis and every arm
    placement's component normally, so a joint-bearing part's xacro may add its own
    native ``ros2_control`` tag too). Appends ``joints`` (a resolved arm placement's
    control-joint patch, ``arena_robots.catalog.render_control_joints``, computed from
    ``resolved_assembly`` independently of this URDF's rendered content) as ``<joint>``
    elements onto the chassis tag, then drops every other ``<ros2_control>`` tag (an
    arm component's own native tag, superseded by the injected patch). No-op when
    ``joints`` is empty."""
    if not joints:
        return
    control_tags = [el for el in root if el.tag.rpartition('}')[-1] == 'ros2_control']
    chassis_tag = next((tag for tag in control_tags if _control_plugin_text(tag) == _GAZEBO_ROS2_CONTROL_PLUGIN), None)
    if chassis_tag is None:
        raise RuntimeError(f"no <ros2_control> tag with <plugin>{_GAZEBO_ROS2_CONTROL_PLUGIN}</plugin> to merge the control-joint patch into")
    for joint in joints:
        chassis_tag.append(_joint_element(joint))
    for tag in control_tags:
        if tag is not chassis_tag:
            root.remove(tag)


class ModelProvider_URDF(ModelProvider.provides(ModelType.URDF)):
    @classmethod
    async def load(cls, model_dir: Path, model: str, loader_args: dict | None) -> Model:

        loader_args = dict(loader_args) if loader_args else {}
        optim_raw = loader_args.pop('optim', '') or ''
        optim_tokens: set[str] = {t.strip() for t in optim_raw.split(',') if t.strip()}
        sensor_patches = loader_args.pop('sensor_topic_patches', None) or []
        wrapper = loader_args.pop('xacro_wrapper', None)
        control_joint_patch = loader_args.pop('control_joint_patch', None) or []

        base_path = model_dir / "urdf"
        xacro_path = base_path / f"{model}.urdf.xacro"

        if not xacro_path.is_file():
            raise FileNotFoundError(f"Xacro file for model {model} not found at {xacro_path}")

        def to_string(v: object) -> str:
            if attrs.has(type(v)):
                v = attrs.asdict(v)
            if isinstance(v, dict):
                return json.dumps(v)
            return str(v)

        wrapper_path: Path | None = None
        if wrapper:
            async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".urdf.xacro", mode="w") as wf:
                await wf.write(wrapper)
                wrapper_path = Path(wf.name)

        cmd = [
            "ros2",
            "run",
            "xacro",
            "xacro",
            str(wrapper_path if wrapper_path is not None else xacro_path),
            *(f"{k}:={to_string(v)}" for k, v in loader_args.items() if v is not None),
        ]

        try:
            process = await asyncio.subprocess.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode or -1, cmd, output=stdout + stderr)
            base_dir = str(base_path)
            root = ET.fromstring(stdout)

            for elem in root.iter():
                if 'filename' not in elem.attrib:
                    continue
                if elem.tag.rpartition('}')[-1] == 'plugin':
                    continue
                original_path = elem.attrib['filename']

                if original_path.startswith(_PACKAGE_URI):
                    pkg, _, sub = original_path[len(_PACKAGE_URI) :].partition('/')
                    abs_path: str | None = None
                    try:
                        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
                    except ModuleNotFoundError:
                        pass
                    else:
                        try:
                            abs_path = os.path.join(get_package_share_directory(pkg), sub)
                        except PackageNotFoundError:
                            pass
                    if abs_path is None:
                        abs_path = os.path.abspath(os.path.join(base_dir, sub))
                    elem.attrib['filename'] = f"file://{abs_path}"
                    print(f"Resolved {original_path} -> file://{abs_path}")
                    continue

                if original_path.startswith("file://"):
                    continue

                if not os.path.isabs(original_path):
                    abs_path = os.path.abspath(os.path.join(base_dir, original_path))
                    elem.attrib['filename'] = f"file://{abs_path}"
                    print(f"Updated relative path to absolute: {original_path} -> file://{abs_path}")
                    continue

                elem.attrib['filename'] = f"file://{original_path}"

            if optim_tokens:
                _strip_sensors(root, optim_tokens)

            _lock_passive_joints(root)
            _strip_unregistered_ros2_control(root, _registered_hardware_plugins())
            _inject_ros2_control_joints(root, control_joint_patch)
            _ensure_effort_state(root)
            _patch_sensor_topics(root, sensor_patches)

            ser = ET.tostring(root, encoding="utf-8", method="xml", xml_declaration=True)
            async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode="wb") as tmp:
                await tmp.write(ser)
                print(f"Converted URDF saved to temporary file: {tmp.name}")

            return Model(type=ModelType.URDF, name=model, description=ser.decode("utf-8"), path=Path(tmp.name))

        except subprocess.CalledProcessError as e:
            print(f"error processing model {model} URDF file {xacro_path}. refusing to load.\n{e}\n{e.output.decode('utf-8')}", file=sys.stderr)
            print(f"Command executed: {' '.join(cmd)}", file=sys.stderr)
            raise
        finally:
            if wrapper_path is not None:
                wrapper_path.unlink(missing_ok=True)
