"""The map_server bringup lands in the task generator namespace, also when relaunched."""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest


@pytest.fixture(scope="module", autouse=True)
def _require_ament() -> None:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")


def _node_levels(description) -> dict[str, str | None]:
    import launch
    from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension
    from launch_ros.actions import Node

    ctx = launch.LaunchContext()
    levels: dict[str, str | None] = {}

    def walk(entity) -> None:
        if isinstance(entity, Node):
            entity._perform_substitutions(ctx)
            args, _ = NodeLogLevelExtension().prepare_for_execute(ctx, {}, entity)
            levels[entity.node_name] = args[1][0].text if args else None
        elif isinstance(entity, launch.LaunchDescription):
            for child in entity.entities:
                walk(child)
        elif isinstance(entity, launch.Action):
            for child in entity.execute(ctx) or []:
                walk(child)

    walk(description)
    return levels


def test_map_server_bringup_is_under_the_namespace() -> None:
    from arena_runtime.env.world_manager.world_manager_ros import map_server_launch_description

    assert set(_node_levels(map_server_launch_description("/arena/env_3/arena_env_node", "warn"))) == {
        "/arena/env_3/arena_env_node/map_server",
        "/arena/env_3/arena_env_node/lifecycle_manager_map_server",
    }


def test_map_server_bringup_logs_by_the_env_rule_set() -> None:
    from arena_runtime.env.world_manager.world_manager_ros import map_server_launch_description

    levels = _node_levels(map_server_launch_description("/arena/env_3/arena_env_node", "{**/map_server:info, warn}"))
    assert levels == {
        "/arena/env_3/arena_env_node/map_server": "info",
        "/arena/env_3/arena_env_node/lifecycle_manager_map_server": "warn",
    }


def test_relaunch_replaces_the_bringup_in_the_namespace() -> None:
    import lifecycle_msgs.msg
    import rclpy
    from arena_rclpy_mixins.LifecycleClient import AsyncLifecycleClient
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    from arena_runtime.env.world_manager.world_manager_ros import MapServerHandler

    namespace = f"/map_server_relaunch_{os.getpid()}"
    context = Context()
    rclpy.init(context=context)

    async def main() -> tuple[bool, bool, bool, list[tuple[str, str]]]:
        node = AsyncLifecycleClient("probe", namespace=namespace, context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()
        handler = MapServerHandler(node=node)
        map_server = f"{namespace}/probe/map_server"
        active = lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE
        try:
            await handler.launch_map_server("warn")
            first = handler._map_server_launch
            assert first is not None
            first_active = await node.wait_for_lifecycle_state_async(map_server, active, timeout=90.0)
            await handler.launch_map_server("warn")
            first_gone = first.task.done()
            relaunch_active = await node.wait_for_lifecycle_state_async(map_server, active, timeout=90.0)
            deadline = time.monotonic() + 10.0
            while True:
                map_servers = [(name, ns) for name, ns in node.get_node_names_and_namespaces() if name == "map_server" and ns in ("/", f"{namespace}/probe")]
                if len(map_servers) == 1 or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.5)
            return first_active, first_gone, relaunch_active, map_servers
        finally:
            await node.kill_launches()
            executor.shutdown()
            node.destroy_node()

    try:
        first_active, first_gone, relaunch_active, map_servers = asyncio.run(main())
    finally:
        rclpy.shutdown(context=context)

    assert first_active
    assert first_gone
    assert relaunch_active
    assert map_servers == [("map_server", f"{namespace}/probe")]
