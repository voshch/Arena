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
import rclpy.client
import rclpy.clock
import rclpy.qos
import rclpy.utilities
import rerun as rr
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.params import declare
from arena_runtime_msgs.msg import AdapterVizManifest, RobotDescriptor, RobotFleet
from arena_viz import DisplayKind, StyleSpec, env_namespace

from rerun_utils.blueprint import blueprint
from rerun_utils.entity_paths import display_path
from rerun_utils.renderers import REGISTRY, RendererCtx
from rerun_utils.params import GRPC_PORT, MAX_RATE_HZ, MEMORY_LIMIT, WEB_PORT
from rerun_utils.stream import Stream, recording
from rerun_utils.tf_mirror import TFMirror


class RerunBridge(ArenaMixinNode):
    robots: list[RobotDescriptor]
    viz_manifest: AdapterVizManifest
    _frame_prefix: str
    _env_id: int
    _tf_namespace: str

    def __init__(self, TASKGEN_NODE: str = '/arena_env_node') -> None:
        super().__init__('rerun_bridge')
        self._TASKGEN_NODE = TASKGEN_NODE
        self._web_port = declare(self, WEB_PORT)
        self._grpc_port = declare(self, GRPC_PORT)
        self._memory_limit = declare(self, MEMORY_LIMIT)
        self._max_rate_hz = declare(self, MAX_RATE_HZ)
        self._dispatched: list[tuple[DisplayKind, str, str]] = []

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

        def on_msg(msg: object) -> None:
            if not future.done():
                loop.call_soon_threadsafe(future.set_result, msg)

        sub = self.create_subscription(
            msg_type,
            topic,
            on_msg,
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

        web_port = self._web_port.value
        grpc_port = self._grpc_port.value
        rec = recording(f"arena_env_{self._env_id}", self._max_rate_hz.value)
        grpc_uri = rr.serve_grpc(
            grpc_port=grpc_port,
            server_memory_limit=self._memory_limit.value,
            recording=rec,
        )
        rr.serve_web_viewer(
            web_port=web_port,
            open_browser=False,
            connect_to=grpc_uri,
        )
        viewer_url = f"http://localhost:{web_port}/?url={urllib.parse.quote(grpc_uri, safe='')}"
        self.get_logger().info(f"rerun web viewer: open {viewer_url}")

        stream = Stream(rec, self._max_rate_hz.value, lambda: self.sim_time.to_nanoseconds() * 1e-9)
        if stream.period > 0:
            self.create_timer(stream.period, stream.flush, clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME))
        self._tf_mirror = TFMirror(self, stream, self._env_id, self._tf_namespace)
        self._ctx = RendererCtx(env_id=self._env_id, node=self, stream=stream, env_namespace=env_namespace(self._TASKGEN_NODE))
        self._dispatch_manifest()
        rr.send_blueprint(blueprint(self._dispatched), recording=rec)

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
            return
        if StyleSpec.from_json(display.style_json).enabled:
            name = display.name if robot is None else f"{robot.name}/{display.name}"
            self._dispatched.append((kind, display_path(self._env_id, robot, display.name), name))


def main() -> None:
    cli_args = rclpy.utilities.remove_ros_args(sys.argv)
    RerunBridge.run_main(*cli_args[1:])


if __name__ == "__main__":
    main()
