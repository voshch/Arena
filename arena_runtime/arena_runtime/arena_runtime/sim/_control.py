"""Shared launch helpers for the ros2_control plane (Gazebo + Isaac)."""

import os
import tempfile
import typing
from collections.abc import Sequence

import launch_ros.actions
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from arena_robots.assembly import ResolvedAssembly
from arena_robots.catalog import Catalog, render_effective_control

if typing.TYPE_CHECKING:
    from arena_robots.Robot import RobotView


def load_control_yaml(config_uri: str) -> dict:
    """Resolve a ``control.config`` URI (``package://...`` or a bare path) and
    load the raw YAML as a dict. No frame prefixing; the caller merges/prefixes."""
    if config_uri.startswith("package://"):
        pkg, _, sub = config_uri[len("package://") :].partition('/')
        try:
            src_path = os.path.join(get_package_share_directory(pkg), sub)
        except PackageNotFoundError as e:
            raise FileNotFoundError(f"control.config package '{pkg}' not found") from e
    else:
        src_path = config_uri

    with open(src_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"control config at {src_path} must be a mapping; got {type(data).__name__}")
    return data


def _prefix_frames(obj: object, frame_prefix: str) -> object:
    if isinstance(obj, dict):
        return {k: (f"{frame_prefix}{v}" if isinstance(k, str) and k.endswith("_frame_id") and isinstance(v, str) and "/" not in v else _prefix_frames(v, frame_prefix)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_prefix_frames(x, frame_prefix) for x in obj]
    return obj


def render_ros2_control_yaml(
    config_uri: str,
    sim_path: str,
    frame_prefix: str,
    *,
    data: dict | None = None,
) -> str:
    """Render and write a ros2_control YAML to /tmp.

    Prefixes bare frame_id values and writes the result to a deterministic temp
    path. Returns the absolute output path. ``data`` overrides the config loaded
    from ``config_uri`` (the caller's merged effective control dict); when
    omitted, loads ``config_uri`` unchanged.
    """
    rendered = _prefix_frames(data if data is not None else load_control_yaml(config_uri), frame_prefix)
    if isinstance(rendered, dict):
        rendered = {(k if k.startswith('/') else f'/**/{k}'): v for k, v in rendered.items()}

    out_path = os.path.join(tempfile.gettempdir(), f"arena_control_{sim_path.replace('/', '_')}.yaml")
    with open(out_path, 'w') as f:
        yaml.safe_dump(rendered, f, sort_keys=False)
    return out_path


def effective_control_yaml(
    resolved: ResolvedAssembly | None,
    config_uri: str,
    sim_path: str,
    frame_prefix: str,
    *,
    catalog: Catalog | None = None,
    prefix: str = 'robot_',
) -> str:
    """Render+write the ros2_control YAML for one robot instance, merging
    ``resolved``'s placement control blocks into the chassis config first.
    Robots without an assembly (``resolved`` is ``None``) take the raw chassis
    config unchanged."""
    if resolved is None:
        return render_ros2_control_yaml(config_uri, sim_path, frame_prefix)
    base_control = load_control_yaml(config_uri)
    merged, _ = render_effective_control(resolved, base_control, catalog if catalog is not None else Catalog(), prefix=prefix)
    return render_ros2_control_yaml(config_uri, sim_path, frame_prefix, data=merged)


def effective_controllers(
    resolved: ResolvedAssembly | None,
    controllers: Sequence[str],
    *,
    catalog: Catalog | None = None,
    prefix: str = 'robot_',
) -> list[str]:
    """``controllers`` plus every placement's extra controller name. Robots
    without an assembly (``resolved`` is ``None``) return ``controllers``
    unchanged."""
    if resolved is None:
        return list(controllers)
    _, extra = render_effective_control(resolved, {}, catalog if catalog is not None else Catalog(), prefix=prefix)
    return [*controllers, *extra]


def robot_controllers(robot_config: "RobotView", resolved: ResolvedAssembly | None) -> list[str]:
    """Controller names the ros2_control plane spawns for this robot, empty when it is not ros2_control-driven."""
    control_spec = robot_config.model_params.control
    if control_spec is None or not control_spec.is_ros2_control:
        return []
    prefix = robot_config.assembly.prefix if robot_config.assembly is not None else 'robot_'
    return effective_controllers(resolved, control_spec.controllers, prefix=prefix)


def odom_relay_node(odom_topic: str) -> launch_ros.actions.Node:
    """Relay `<odom_topic>` (controller-specific) -> `odom` for the Arena convention."""
    return launch_ros.actions.Node(
        package='topic_tools',
        executable='relay',
        name='odom_relay',
        output='screen',
        arguments=[odom_topic, 'odom'],
        parameters=[{'use_sim_time': True}],
    )


def twist_stamper_node(cmd_vel_topic: str, frame_id: str) -> launch_ros.actions.Node:
    """Stamp `cmd_vel` (Twist) -> `cmd_vel_out` (TwistStamped) for stamped controllers."""
    return launch_ros.actions.Node(
        package='twist_stamper',
        executable='twist_stamper',
        name='twist_stamper',
        output='screen',
        remappings=[
            ('cmd_vel_in', 'cmd_vel'),
            ('cmd_vel_out', cmd_vel_topic),
        ],
        parameters=[
            {
                'use_sim_time': True,
                'frame_id': frame_id,
            }
        ],
    )
