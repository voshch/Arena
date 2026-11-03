"""ConfigFileGenerator._await_param against task_generator parameters that arrive late or whose first response is lost."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import rclpy.node
    from arena_rclpy_mixins import ClientWrapper

    from rviz_utils.scripts.rviz_config import ConfigFileGenerator

PARAM_VALUE = "env_0"


def _run_with_generator(make_server: Callable[[str], rclpy.node.Node], body: Callable[[ConfigFileGenerator, ClientWrapper], Awaitable[None]], ns: str) -> None:
    import rclpy.executors
    import rcl_interfaces.srv

    from rviz_utils.scripts.rviz_config import ConfigFileGenerator

    async def main() -> None:
        server = make_server(ns)
        node = ConfigFileGenerator(f"{ns}/arena_env_node")
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        executor.add_node(server)
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        try:
            client = node.create_client_wrapper(rcl_interfaces.srv.GetParameters, f"{ns}/arena_env_node/get_parameters")
            assert await node.wait_for_service_async(client.client, timeout=10.0)
            await body(node, client)
        finally:
            executor.shutdown()
            spin.join(timeout=5.0)
            node.destroy_node()
            server.destroy_node()

    asyncio.run(main())


def test_await_param_succeeds_when_param_is_declared_after_first_query() -> None:
    import rclpy.node

    declared = threading.Event()

    def make_server(ns: str) -> rclpy.node.Node:
        server = rclpy.node.Node("arena_env_node", namespace=ns)

        def declare() -> None:
            server.declare_parameter("prefix", PARAM_VALUE)
            declared.set()

        threading.Timer(1.5, declare).start()
        return server

    async def body(node: ConfigFileGenerator, client: ClientWrapper) -> None:
        start = time.monotonic()
        value = await asyncio.wait_for(node._await_param(client, "prefix", interval=0.2), timeout=15.0)
        assert declared.is_set()
        assert value.string_value == PARAM_VALUE
        assert time.monotonic() - start >= 1.0

    _run_with_generator(make_server, body, "/rviz_cfg_wait_late")


def test_await_param_retries_when_first_response_never_arrives() -> None:
    import rcl_interfaces.msg
    import rcl_interfaces.srv
    import rclpy.callback_groups
    import rclpy.node

    release = threading.Event()
    requests: list[float] = []

    def make_server(ns: str) -> rclpy.node.Node:
        server = rclpy.node.Node("arena_env_node", namespace=ns, start_parameter_services=False)

        def on_get(request: rcl_interfaces.srv.GetParameters.Request, response: rcl_interfaces.srv.GetParameters.Response) -> rcl_interfaces.srv.GetParameters.Response:
            requests.append(time.monotonic())
            if len(requests) == 1:
                release.wait()
            response.values = [rcl_interfaces.msg.ParameterValue(type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING, string_value=PARAM_VALUE) for _ in request.names]
            return response

        server.create_service(rcl_interfaces.srv.GetParameters, f"{ns}/arena_env_node/get_parameters", on_get, callback_group=rclpy.callback_groups.ReentrantCallbackGroup())
        return server

    async def body(node: ConfigFileGenerator, client: ClientWrapper) -> None:
        try:
            value = await asyncio.wait_for(node._await_param(client, "prefix", interval=0.1, call_timeout=0.5), timeout=10.0)
        finally:
            release.set()
        assert value.string_value == PARAM_VALUE
        assert len(requests) >= 2

    _run_with_generator(make_server, body, "/rviz_cfg_wait_lost")


def test_await_param_skips_not_set_values() -> None:
    import rcl_interfaces.msg
    import rcl_interfaces.srv
    import rclpy.node

    requests: list[float] = []

    def make_server(ns: str) -> rclpy.node.Node:
        server = rclpy.node.Node("arena_env_node", namespace=ns, start_parameter_services=False)

        def on_get(request: rcl_interfaces.srv.GetParameters.Request, response: rcl_interfaces.srv.GetParameters.Response) -> rcl_interfaces.srv.GetParameters.Response:
            requests.append(time.monotonic())
            if len(requests) < 3:
                response.values = [rcl_interfaces.msg.ParameterValue() for _ in request.names]
            else:
                response.values = [rcl_interfaces.msg.ParameterValue(type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING, string_value=PARAM_VALUE) for _ in request.names]
            return response

        server.create_service(rcl_interfaces.srv.GetParameters, f"{ns}/arena_env_node/get_parameters", on_get)
        return server

    async def body(node: ConfigFileGenerator, client: ClientWrapper) -> None:
        value = await asyncio.wait_for(node._await_param(client, "prefix", interval=0.1), timeout=10.0)
        assert value.string_value == PARAM_VALUE
        assert len(requests) == 3

    _run_with_generator(make_server, body, "/rviz_cfg_wait_unset")
