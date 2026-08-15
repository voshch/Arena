from __future__ import annotations

import asyncio
import logging

import rclpy
import rclpy.executors
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
)

from .resources import build_resources_list, read_resource_content
from .ros_bridge import RosBridge
from .tools import build_tools_list, dispatch_tool_call

logger = logging.getLogger(__name__)

_DISCOVERY_TIMEOUT = 10.0


async def _wait_for_bridge(bridge: RosBridge) -> None:
    """Wait briefly for initial state/episode to arrive from latched topic."""
    deadline = asyncio.get_event_loop().time() + _DISCOVERY_TIMEOUT
    while bridge.current_episode is None:
        if asyncio.get_event_loop().time() >= deadline:
            logger.warning("state/episode not received within %.1fs; continuing anyway", _DISCOVERY_TIMEOUT)
            break
        await asyncio.sleep(0.1)


async def _run(bridge: RosBridge) -> None:
    await _wait_for_bridge(bridge)

    server = Server("task_generator_mcp")

    async def handle_list_tools(_ctx: object, _params: PaginatedRequestParams) -> ListToolsResult:
        return ListToolsResult(tools=build_tools_list())

    server.add_request_handler("tools/list", PaginatedRequestParams, handle_list_tools)

    async def handle_call_tool(_ctx: object, params: CallToolRequestParams) -> CallToolResult:
        return await dispatch_tool_call(params, bridge)

    server.add_request_handler("tools/call", CallToolRequestParams, handle_call_tool)

    async def handle_list_resources(_ctx: object, _params: PaginatedRequestParams) -> ListResourcesResult:
        return ListResourcesResult(resources=build_resources_list())

    server.add_request_handler("resources/list", PaginatedRequestParams, handle_list_resources)

    async def handle_read_resource(_ctx: object, params: ReadResourceRequestParams) -> ReadResourceResult:
        return read_resource_content(params, bridge)

    server.add_request_handler("resources/read", ReadResourceRequestParams, handle_read_resource)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def _amain() -> None:
    loop = asyncio.get_running_loop()
    executor = rclpy.executors.MultiThreadedExecutor()
    bridge = RosBridge()
    executor.add_node(bridge)
    spin_task = loop.run_in_executor(None, executor.spin)
    try:
        await _run(bridge)
    finally:
        executor.shutdown()
        await spin_task
        bridge.destroy_node()


def main() -> None:
    rclpy.init()
    try:
        asyncio.run(_amain())
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
