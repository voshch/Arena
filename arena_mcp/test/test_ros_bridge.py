from __future__ import annotations

import asyncio
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from arena_mcp.core.errors import ToolError
from arena_mcp.core.ros import LATCHED, ROS, profile


def _peer():
    node = rclpy.create_node("arena_mcp_bridge_peer")
    node.declare_parameter("timeout", -1)
    node.declare_parameter("ratio", 0.5)
    node.declare_parameter("label", "x")
    node.create_service(Trigger, "/arena_mcp_test/trigger", lambda request, response: Trigger.Response(success=True, message="pong"))
    publisher = node.create_publisher(String, "/arena_mcp_test/latched", profile(LATCHED, 1))
    publisher.publish(String(data="kept"))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return node, executor


def test_bridge_calls_reads_latched_and_coerces_params():
    async def scenario():
        await ROS.node()
        node, executor = _peer()
        try:
            response = await ROS.call(Trigger, "/arena_mcp_test/trigger", ready=10.0)
            latched = await ROS.latest(String, "/arena_mcp_test/latched", timeout=10.0)
            start = time.monotonic()
            with pytest.raises(ToolError, match="not available"):
                await ROS.call(Trigger, "/arena_mcp_test/nobody_serves_this", ready=0.5)
            fast = time.monotonic() - start
            peer = "/arena_mcp_bridge_peer"
            set_result = await ROS.set_params(peer, {"timeout": 30, "ratio": 2, "label": 7})
            with pytest.raises(ToolError, match="not a valid int"):
                await ROS.set_params(peer, {"timeout": 1.5})
            values = await ROS.get_params(peer, ["timeout", "ratio", "label"])
            with pytest.raises(ToolError, match="does not declare nope"):
                await ROS.get_params(peer, ["timeout", "nope"])
            with pytest.raises(ToolError, match="does not declare nope"):
                await ROS.set_params(peer, {"nope": 1})
            shallow = await ROS.cache(String, "/arena_mcp_test/latched", depth=1)
            deep = await ROS.cache(String, "/arena_mcp_test/latched", depth=50)
            assert deep is shallow and deep.history.maxlen == 50 and deep.last.data == "kept"
            graph = await ROS.graph()
            own = await ROS.node()
            for _ in range(20):
                lingering = own.get_client_names_and_types_by_node(own.get_name(), own.get_namespace())
                if not lingering:
                    break
                await asyncio.sleep(0.1)
            ready_absent = await ROS.ready(Trigger, "/arena_mcp_test/nobody_serves_this", timeout=0.3)
            ready_typed = await ROS.ready(Trigger, "/arena_mcp_test/trigger", timeout=5.0)
            ready_mistyped = await ROS.ready(SetBool, "/arena_mcp_test/trigger", timeout=0.3)
            lingering_after_probe = own.get_client_names_and_types_by_node(own.get_name(), own.get_namespace())
            return response, latched, fast, set_result, values, graph, lingering, (ready_absent, ready_typed, ready_mistyped), lingering_after_probe
        finally:
            executor.shutdown()
            node.destroy_node()
            await ROS.shutdown()

    response, latched, fast, set_result, values, graph, lingering, ready_absent, lingering_after_probe = asyncio.run(scenario())
    assert response.success and response.message == "pong"
    assert latched.data == "kept"
    assert fast < 10.0
    assert all(entry["ok"] for entry in set_result.values())
    assert values == {"timeout": 30, "ratio": 2.0, "label": "7"}
    assert "/arena_mcp_bridge_peer" in graph["nodes"]
    assert lingering == [] and lingering_after_probe == []
    assert ready_absent == (False, True, False)
