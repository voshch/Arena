#! /usr/bin/env python3

import asyncio
import math
import os
import signal
import sys
import tempfile
import time
from collections.abc import Callable, Sequence

import arena_bringup.extensions.NodeLogLevelExtension as NodeLogLevelExtension
import launch
import launch_ros.actions
import rcl_interfaces.msg
import rcl_interfaces.srv
import rclpy
import rclpy.parameter
import rclpy.qos
import rclpy.task
import rclpy.time
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.shared import FrameNamespace
from arena_robots.moveit_factory import build_moveit_params
from arena_robots.Robot import RobotIdentifier
from arena_viz import DisplayKind
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rviz_display_control_msgs.msg import DisplaySet, DisplaySpec
from task_generator_msgs.msg import AdapterVizManifest, RobotDescriptor, RobotFleet

from rviz_utils.renderers import REGISTRY

#: DRI render nodes. Present only when the container was given `devices: /dev/dri` and the
#: video/render GIDs (see docker-compose.yaml).
_DRI_PATH = '/dev/dri'

#: Orbit distance per metre of static-map span for the map view. 50 m framed the 35 m
#: hospital_1 map well, and that ratio holds for the other worlds.
_MAP_VIEW_DISTANCE_PER_M = 1.45
#: How long to wait for the env's static map before falling back to the fixed frame.
_MAP_WAIT_S = 5.0
# The map view is worth waiting for: the grid is latched once the world is loaded, well after
# the `prefix` parameter exists when RViz and the simulation start together.
_MAP_VIEW_WAIT_S = 120.0
#: The map view before maps were framed automatically; still used when no map arrives.
#: Seconds a robot view waits for the fleet to be registered before falling back to the map.
_FLEET_WAIT_S = 120.0
_FALLBACK_MAP_VIEW: dict[str, object] = {'Distance': 50.0, 'Focal Point': {'X': 15.0, 'Y': 10.0, 'Z': 0.0}}


#: Metres of free border kept around the occupied cells when a map is framed on its walls.
_MAP_FRAME_MARGIN_M = 1.0


def map_view_frame(info: MapMetaData, offset: tuple[float, float, float] = (0.0, 0.0, 0.0), data: Sequence[int] | None = None) -> dict[str, object]:
    """Focal point and orbit distance that frame the static map - its *occupied* cells when
    ``data`` is given, else the whole image (a generated world sits in one corner of a map
    image several times its size, and framing the image left it small and off-centre).

    ``offset`` is the (x, y, yaw) of the grid's frame expressed in the view's fixed frame; the
    grid is published in its own ``map`` frame, which the env's fixed frame may translate.
    """
    width = info.width * info.resolution
    height = info.height * info.resolution
    cx = info.origin.position.x + width / 2
    cy = info.origin.position.y + height / 2
    if data is not None and info.width and info.height:
        occupied = [i for i, v in enumerate(data) if v is not None and v > 50]
        if occupied:
            rows = [i // info.width for i in occupied]
            cols = [i % info.width for i in occupied]
            x0 = info.origin.position.x + min(cols) * info.resolution - _MAP_FRAME_MARGIN_M
            x1 = info.origin.position.x + (max(cols) + 1) * info.resolution + _MAP_FRAME_MARGIN_M
            y0 = info.origin.position.y + min(rows) * info.resolution - _MAP_FRAME_MARGIN_M
            y1 = info.origin.position.y + (max(rows) + 1) * info.resolution + _MAP_FRAME_MARGIN_M
            width, height = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tx, ty, yaw = offset
    fx = tx + math.cos(yaw) * cx - math.sin(yaw) * cy
    fy = ty + math.sin(yaw) * cx + math.cos(yaw) * cy
    return {
        'Distance': round(max(width, height, 1.0) * _MAP_VIEW_DISTANCE_PER_M, 1),
        'Focal Point': {'X': round(fx, 2), 'Y': round(fy, 2), 'Z': 0.0},
    }


def _gl_env() -> dict[str, str]:
    """GL environment for rviz2.

    Without a render node Mesa picks the `iris` driver, fails with
    "glx: failed to create dri3 screen", and ABORTS instead of falling back - which reads as
    "RViz just never opened", because the supervisor node keeps running. Forcing software GL
    is the documented workaround for that.

    But software GL is expensive: measured at 185% CPU for rviz2 alone on
    arena_arena_002, against 27% for a GPU-accelerated `gz sim` in the same run. It was the
    dominant cost of every non-headless session and froze the workstation more than once.

    So force it only when there is genuinely no render node to use. With
    `/dev/dri/renderD*` present and readable, iris works and rviz2 renders on the GPU.
    Set LIBGL_ALWAYS_SOFTWARE=1 in the environment to override this and pin software GL
    regardless - the launch honours an explicit setting.
    """
    explicit = os.environ.get('LIBGL_ALWAYS_SOFTWARE')
    if explicit is not None:
        return {'LIBGL_ALWAYS_SOFTWARE': explicit}

    try:
        render_nodes = [n for n in os.listdir(_DRI_PATH) if n.startswith('renderD')]
    except OSError:
        render_nodes = []

    usable = any(os.access(os.path.join(_DRI_PATH, n), os.R_OK | os.W_OK) for n in render_nodes)
    return {} if usable else {'LIBGL_ALWAYS_SOFTWARE': '1'}


class ConfigFileGenerator(ArenaMixinNode):
    _robots: list[RobotDescriptor]
    _viz_manifest: AdapterVizManifest | None
    _node_params: list[rcl_interfaces.msg.Parameter]
    _display_set_pub: rclpy.publisher.Publisher
    _frame_prefix: str
    _env_id: int
    _map_view: dict[str, object]

    def __init__(self, TASKGEN_NODE: str = '/task_generator_node'):
        super().__init__('rviz_config_generator')

        self._TASKGEN_NODE = TASKGEN_NODE
        self.declare_parameter('view', 'map')
        self.declare_parameter('robot', 0)
        self.declare_parameter('view_distance', 8.0)

    async def _await_param(
        self,
        client: rclpy.client.Client,
        param_name: str,
        test_fn: Callable[[rcl_interfaces.msg.ParameterValue], bool] | None = None,
        interval: float = 1.0,
    ) -> rcl_interfaces.msg.ParameterValue:
        """Block until parameter passes test function."""
        while True:
            self.get_logger().info(f'waiting for {param_name} to be set')
            req = rcl_interfaces.srv.GetParameters.Request(names=[param_name])
            params = await self.await_ros(client.call_async(req))
            if params and params.values:
                value = params.values[0]
                if (not test_fn) or test_fn(value):
                    self.get_logger().info(f'param {param_name} is set')
                    return value
            await asyncio.sleep(interval)

    async def setup(self) -> None:
        TASKGEN_PARAM_SRV = os.path.join(self._TASKGEN_NODE, 'get_parameters')

        get_parameters_cli = self.create_client(rcl_interfaces.srv.GetParameters, TASKGEN_PARAM_SRV)
        self.get_logger().info(f'waiting for service {TASKGEN_PARAM_SRV} to become available')
        await self.wait_for_service_async(get_parameters_cli)
        self.get_logger().info(f'service {TASKGEN_PARAM_SRV} is available')

        self._frame_prefix = (await self._await_param(get_parameters_cli, 'prefix')).string_value
        self._env_id = (await self._await_param(get_parameters_cli, 'env_id')).integer_value
        self._map_view = await self._await_map_view()
        self._robots = []
        self._viz_manifest = None
        self._node_params = []

        _manifest_qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        self._display_set_pub = self.create_publisher(DisplaySet, os.path.join(self._TASKGEN_NODE, 'state', 'display_set'), qos_profile=_manifest_qos)

        self.create_subscription(
            AdapterVizManifest,
            os.path.join(self._TASKGEN_NODE, 'state', 'viz_manifest'),
            self._on_viz_manifest,
            qos_profile=_manifest_qos,
        )
        self.create_subscription(
            RobotFleet,
            os.path.join(self._TASKGEN_NODE, 'state', 'robots'),
            self._on_robots,
            qos_profile=_manifest_qos,
        )

        if str(self.get_parameter('view').value) in ('robot', 'robot3p'):
            # The fleet arrives on a latched topic once the robots are registered, after the map.
            deadline = time.monotonic() + _FLEET_WAIT_S
            while not self._robots and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
            if not self._robots:
                self.get_logger().warning(f'no robot registered within {_FLEET_WAIT_S:.0f}s; the robot view falls back to the map view')

        config_file = self.create_config()

        rviz_parameters: list[dict[str, object]] = [{"use_sim_time": True}]

        launch_task = await self._launch_manager.launch_description(
            launch.LaunchDescription(
                [
                    NodeLogLevelExtension.SetGlobalLogLevelAction(rclpy.logging.get_logger_effective_level(self.get_logger().name).name.lower()),
                    launch_ros.actions.Node(
                        package="rviz2",
                        executable="rviz2",
                        name="rviz2",
                        namespace=self._TASKGEN_NODE,
                        arguments=['-d', config_file],
                        parameters=rviz_parameters,
                        output="screen",
                        additional_env=_gl_env(),
                    ),
                ]
            )
        )
        await launch_task
        self.get_logger().info('rviz2 exited, shutting down supervisor')
        os.kill(os.getpid(), signal.SIGINT)

    def _on_viz_manifest(self, msg: AdapterVizManifest) -> None:
        self._viz_manifest = msg
        self._rebuild_display_set()

    def _on_robots(self, msg: RobotFleet) -> None:
        self._robots = [state.descriptor for state in msg.robots]
        self._node_params = self._build_node_params()
        self._rebuild_display_set()

    def _rebuild_display_set(self) -> None:
        if self._viz_manifest is None:
            return

        specs: list[DisplaySpec] = []
        robots_by_ns = {robot.ns: robot for robot in self._robots}

        for d in self._viz_manifest.env_displays:
            try:
                renderer = REGISTRY[DisplayKind(d.kind)]
            except KeyError:
                self.get_logger().warning(f"no rviz renderer for kind {d.kind!r}, skipping {d.name!r}")
                continue
            full_dict = renderer(d, None)
            if full_dict is None:
                continue
            specs.append(DisplaySpec(
                id=f"env::{d.group}::{d.kind}::{d.topic}",
                group_path=d.group,
                class_id=full_dict["Class"],
                config=yaml.dump(full_dict),
                require_topic=d.topic if d.topic_must_exist else "",
            ))

        for entry in self._viz_manifest.entries:
            robot = robots_by_ns.get(entry.robot_ns)
            if robot is None:
                self.get_logger().warning(f"manifest entry for unknown robot ns {entry.robot_ns!r}, skipping")
                continue
            for d in entry.displays:
                try:
                    renderer = REGISTRY[DisplayKind(d.kind)]
                except KeyError:
                    self.get_logger().warning(f"no rviz renderer for kind {d.kind!r}, skipping {d.name!r}")
                    continue
                full_dict = renderer(d, robot)
                if full_dict is None:
                    continue
                specs.append(DisplaySpec(
                    id=f"robot::{entry.robot_ns}::{d.kind}::{d.topic}",
                    group_path=f"Robot: {robot.name}",
                    class_id=full_dict["Class"],
                    config=yaml.dump(full_dict),
                    require_topic=d.topic if d.topic_must_exist else "",
                ))

        self._display_set_pub.publish(DisplaySet(root_group="Arena", displays=specs, node_params=self._node_params))

    def _build_node_params(self) -> list[rcl_interfaces.msg.Parameter]:
        return self._flatten_params(self._collect_moveit_params())

    def _flatten_params(self, params: dict[str, object], prefix: str = "") -> list[rcl_interfaces.msg.Parameter]:
        out: list[rcl_interfaces.msg.Parameter] = []
        for key, value in params.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                out.extend(self._flatten_params(value, name))
            elif value is not None:
                try:
                    out.append(rclpy.parameter.Parameter(name, value=value).to_parameter_msg())
                except (TypeError, ValueError):
                    self.get_logger().warning(f"skipping non-parameter value at {name!r}")
        return out

    def _collect_moveit_params(self) -> dict[str, object]:
        """Mirror each arm robot's MoveIt config into rviz2 under a robot-named
        prefix. Per-display ``Robot Description`` properties point at
        ``<robot>.robot_description`` so every arm robot gets its own
        Trajectory/PlanningScene display."""
        combined: dict[str, object] = {}
        arm_names: list[str] = []
        for robot in self._robots:
            tf_prefix = FrameNamespace(robot.frame).raw()
            tf_prefix = tf_prefix + "/" if tf_prefix else ""
            params = build_moveit_params(robot.model, tf_prefix=tf_prefix)
            if params is None:
                continue
            arm_names.append(robot.name)
            for key, value in params.items():
                combined[f"{robot.name}.{key}"] = value
        if arm_names:
            self.get_logger().info(f"injecting MoveIt params into rviz2 for: {arm_names}")
        return combined

    async def _await_map_view(self) -> dict[str, object]:
        """Frame the map view on the env's static map (latched by map_server), or fall back."""
        topic = os.path.join(self._TASKGEN_NODE, 'map')
        wait_s = _MAP_VIEW_WAIT_S if str(self.get_parameter('view').value) == 'map' else _MAP_WAIT_S
        future: rclpy.task.Future = rclpy.task.Future()

        def on_map(msg: OccupancyGrid) -> None:
            if not future.done():
                future.set_result(msg)

        qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        sub = self.create_subscription(OccupancyGrid, topic, on_map, qos_profile=qos)
        try:
            grid: OccupancyGrid = await asyncio.wait_for(self.await_ros(future), timeout=wait_s)
        except TimeoutError:
            self.get_logger().warning(f'no static map on {topic} within {wait_s}s; using the fixed map view')
            return dict(_FALLBACK_MAP_VIEW)
        finally:
            self.destroy_subscription(sub)

        fixed_frame = FrameNamespace(self._frame_prefix).tf('map')
        offset = (0.0, 0.0, 0.0)
        if grid.header.frame_id and grid.header.frame_id != fixed_frame:
            offset = await self._frame_offset(fixed_frame, grid.header.frame_id)
        view = map_view_frame(grid.info, offset, data=grid.data)
        self.get_logger().info(f"map view framed on {view['Focal Point']} ({fixed_frame}) at distance {view['Distance']}")
        return view

    async def _frame_offset(self, target: str, source: str) -> tuple[float, float, float]:
        """(x, y, yaw) of ``source`` in ``target`` from tf, or the identity if tf never answers."""
        buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(buffer, self, spin_thread=False)
        try:
            deadline = asyncio.get_running_loop().time() + _MAP_WAIT_S
            while not buffer.can_transform(target, source, rclpy.time.Time()):
                if asyncio.get_running_loop().time() > deadline:
                    self.get_logger().warning(f'no transform {target} <- {source} within {_MAP_WAIT_S}s; framing the map in its own frame')
                    return (0.0, 0.0, 0.0)
                await asyncio.sleep(0.1)
            t = buffer.lookup_transform(target, source, rclpy.time.Time()).transform
            q = t.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (t.translation.x, t.translation.y, yaw)
        finally:
            listener.unregister()

    def create_config(self) -> str:
        skeleton = self._read_default_file()
        skeleton["Visualization Manager"]["Views"]["Current"] = self._build_view()
        file_path = self._tmp_config_file(skeleton, prefix=f"env{self._env_id}_")
        self.get_logger().info(f'created config file at {file_path}')
        return file_path

    def _target_robot_frame(self) -> str | None:
        if not self._robots:
            self.get_logger().warning('view requested a robot target frame, but fleet is empty, falling back to map view')
            return None
        idx = self.get_parameter('robot').value
        try:
            robot = self._robots[idx]
        except IndexError:
            self.get_logger().warning(f'robot index {idx} out of range (fleet size {len(self._robots)}), ignoring')
            return None
        base_frame = RobotIdentifier(robot.model).resolve_sync().model_params.base_frame
        prefix = FrameNamespace(robot.frame).raw()
        return f'{prefix}/{base_frame}' if prefix else base_frame

    def _build_view(self) -> dict[str, object]:
        view = str(self.get_parameter('view').value)

        target = '<Fixed Frame>'
        if view in ('robot', 'robot3p'):
            robot_frame = self._target_robot_frame()
            if robot_frame is None:
                view = 'map'
            else:
                target = robot_frame

        base: dict[str, object] = {
            'Class': 'rviz_viewport_control/ViewportControl',
            'Name': 'Current View',
            'Near Clip Distance': 0.01,
            'Target Frame': target,
            'Value': True,
        }

        distance = float(self.get_parameter('view_distance').value or 8.0)
        if view == 'robot':
            return {**base, 'Distance': distance, 'Focal Point': {'X': 0.0, 'Y': 0.0, 'Z': 0.0}, 'Pitch': 0.9, 'Yaw': 3.14}
        if view == 'robot3p':
            return {**base, 'Distance': distance, 'Focal Point': {'X': 0.0, 'Y': 0.0, 'Z': 0.0}, 'Pitch': 0.5, 'Yaw': 3.14}
        return {**base, **self._map_view, 'Pitch': 0.9, 'Yaw': 3.8}

    def _read_default_file(self) -> dict[str, object]:
        package_path = get_package_share_directory("rviz_utils")
        file_path = os.path.join(package_path, "config", "rviz_default.rviz")

        fixed_frame = FrameNamespace(self._frame_prefix).tf('map')

        with open(file_path) as file:
            content = file.read()
            # i'm lazy, bite me
            content = content.format(
                task_generator_node=self._TASKGEN_NODE,
                fixed_frame=fixed_frame,
            )
            return yaml.safe_load(content)

    @classmethod
    def _tmp_config_file(cls, config_file: dict[str, object], prefix: str = "") -> str:
        f = tempfile.NamedTemporaryFile('w', delete=False, prefix=prefix)
        yaml.dump(config_file, f)
        f.close()
        return f.name


def main():
    cli_args = rclpy.utilities.remove_ros_args(sys.argv)
    ConfigFileGenerator.run_main(*cli_args[1:])


if __name__ == "__main__":
    main()
