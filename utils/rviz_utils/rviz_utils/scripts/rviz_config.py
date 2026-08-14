#! /usr/bin/env python3

import asyncio
import os
import signal
import sys
import tempfile
from collections.abc import Callable

import arena_bringup.extensions.NodeLogLevelExtension as NodeLogLevelExtension
import launch
import launch_ros.actions
import rcl_interfaces.msg
import rcl_interfaces.srv
import rclpy
import rclpy.parameter
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.shared import FrameNamespace
from arena_robots.moveit_factory import build_moveit_params
from arena_robots.Robot import RobotIdentifier
from arena_viz import DisplayKind
from rviz_display_control_msgs.msg import DisplaySet, DisplaySpec
from task_generator_msgs.msg import AdapterVizManifest, RobotDescriptor, RobotFleet

from rviz_utils.renderers import REGISTRY


class ConfigFileGenerator(ArenaMixinNode):
    _robots: list[RobotDescriptor]
    _viz_manifest: AdapterVizManifest | None
    _node_params: list[rcl_interfaces.msg.Parameter]
    _display_set_pub: rclpy.publisher.Publisher
    _frame_prefix: str
    _env_id: int

    def __init__(self, TASKGEN_NODE: str = '/task_generator_node'):
        super().__init__('rviz_config_generator')

        self._TASKGEN_NODE = TASKGEN_NODE
        self.declare_parameter('view', 'map')
        self.declare_parameter('robot', 0)

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

        if view == 'robot':
            return {**base, 'Distance': 8.0, 'Focal Point': {'X': 0.0, 'Y': 0.0, 'Z': 0.0}, 'Pitch': 0.9, 'Yaw': 3.14}
        if view == 'robot3p':
            return {**base, 'Distance': 8.0, 'Focal Point': {'X': 0.0, 'Y': 0.0, 'Z': 0.0}, 'Pitch': 0.5, 'Yaw': 3.14}
        return {**base, 'Distance': 50.0, 'Focal Point': {'X': 15.0, 'Y': 10.0, 'Z': 0.0}, 'Pitch': 0.9, 'Yaw': 3.8}

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
