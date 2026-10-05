#! /usr/bin/env python3
"""Bridge node: subscribe per arena_viz manifest entry, log to rerun, serve via web."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import typing
import urllib.parse

import rcl_interfaces.msg
import rcl_interfaces.srv
import rclpy
import rerun as rr
from arena_rclpy_mixins import ArenaMixinNode
from arena_viz import DisplayKind
from task_generator_msgs.msg import AdapterVizManifest, RobotDescriptor, RobotFleet

from rerun_utils.renderers import REGISTRY, RendererCtx
from rerun_utils.tf_mirror import TFMirror


class RerunBridge(ArenaMixinNode):
    robots: list[RobotDescriptor]
    viz_manifest: AdapterVizManifest
    _frame_prefix: str
    _env_id: int
    _tf_namespace: str

    def __init__(self, TASKGEN_NODE: str = '/task_generator_node') -> None:
        super().__init__('rerun_bridge')
        self._TASKGEN_NODE = TASKGEN_NODE
        self.declare_parameter('web_port', 9090)
        self.declare_parameter('grpc_port', 9876)

    async def _await_param(
        self,
        client: rclpy.client.Client,
        param_name: str,
        test_fn: typing.Callable[[typing.Any], bool] | None = None,
        interval: float = 1.0,
    ) -> rcl_interfaces.msg.ParameterValue:
        while True:
            self.get_logger().info(f'waiting for {param_name} to be set')
            req = rcl_interfaces.srv.GetParameters.Request(names=[param_name])
            params = await self.await_ros(client.call_async(req))
            if params and params.values:
                value = params.values[0]
                if (not test_fn) or test_fn(value):
                    return value
            await asyncio.sleep(interval)

    async def _await_latched(self, msg_type, topic: str):
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        sub = self.create_subscription(
            msg_type,
            topic,
            lambda msg: loop.call_soon_threadsafe(future.set_result, msg) if not future.done() else None,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        try:
            self.get_logger().info(f'waiting for first {topic} message')
            return await future
        finally:
            self.destroy_subscription(sub)

    async def setup(self) -> None:
        TASKGEN_PARAM_SRV = os.path.join(self._TASKGEN_NODE, 'get_parameters')
        cli = self.create_client(rcl_interfaces.srv.GetParameters, TASKGEN_PARAM_SRV)
        await self.wait_for_service_async(cli)

        await self._await_param(cli, 'initialized', lambda x: x.bool_value)
        self._frame_prefix = (await self._await_param(cli, 'prefix')).string_value
        self._env_id = (await self._await_param(cli, 'env_id')).integer_value
        self._tf_namespace = (await self._await_param(cli, 'tf_namespace')).string_value
        self.robots = [
            state.descriptor
            for state in (await self._await_latched(
                RobotFleet, os.path.join(self._TASKGEN_NODE, 'state', 'robots'))).robots
        ]
        self.viz_manifest = await self._await_latched(
            AdapterVizManifest, os.path.join(self._TASKGEN_NODE, 'state', 'viz_manifest'))

        web_port = int(self.get_parameter('web_port').value)
        grpc_port = int(self.get_parameter('grpc_port').value)
        rec_id = f"arena_env_{self._env_id}"
        rr.init(rec_id, spawn=False)
        grpc_uri = rr.serve_grpc(
            grpc_port=grpc_port,
            server_memory_limit="2GiB",
        )
        rr.serve_web_viewer(
            web_port=web_port,
            open_browser=False,
            connect_to=grpc_uri,
        )
        viewer_url = f"http://localhost:{web_port}/?url={urllib.parse.quote(grpc_uri, safe='')}"
        self.get_logger().info(f"rerun web viewer: open {viewer_url}")

        self._tf_mirror = TFMirror(self, self._env_id, self._tf_namespace)
        self._ctx = RendererCtx(env_id=self._env_id, node=self)
        self._dispatch_manifest()

        loop = asyncio.get_running_loop()
        stop = loop.create_future()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: stop.set_result(None) if not stop.done() else None)
        await stop

    def _dispatch_manifest(self) -> None:
        robots_by_ns = {r.ns: r for r in self.robots}

        for d in self.viz_manifest.env_displays:
            self._invoke(d, robot=None)

        for entry in self.viz_manifest.entries:
            robot = robots_by_ns.get(entry.robot_ns)
            if robot is None:
                self.get_logger().warning(f"manifest entry for unknown robot ns {entry.robot_ns!r}")
                continue
            for d in entry.displays:
                self._invoke(d, robot=robot)

    def _invoke(self, display, robot: RobotDescriptor | None) -> None:
        try:
            kind = DisplayKind(display.kind)
        except ValueError:
            self.get_logger().warning(f"unknown kind {display.kind!r}, skipping {display.name!r}")
            return
        renderer = REGISTRY.get(kind)
        if renderer is None:
            self.get_logger().warning(f"no rerun renderer for kind {kind!r}, skipping {display.name!r}")
            return
        try:
            renderer(display, robot, self._ctx)
        except Exception as e:
            self.get_logger().error(f"renderer {kind!r} for {display.name!r} failed: {e}")


def main() -> None:
    cli_args = rclpy.utilities.remove_ros_args(sys.argv)
    RerunBridge.run_main(*cli_args[1:])


if __name__ == "__main__":
    main()
