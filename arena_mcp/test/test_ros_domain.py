from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from example_interfaces.action import Fibonacci
from mcp.server.mcpserver import Image
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Int32, String
from std_srvs.srv import SetBool

from arena_mcp.core import JOBS, ROS
from arena_mcp.core.errors import ToolError
from arena_mcp.core.ros import LATCHED, profile
from arena_mcp.domains import ros

NS = "/arena_mcp_ros_test"


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    return tmp_path


def test_resolve_infers_the_kind_and_lists_alternatives():
    assert ros.resolve("std_msgs/String") == ("std_msgs/msg/String", "msg")
    assert ros.resolve("/std_srvs/srv/SetBool") == ("std_srvs/srv/SetBool", "srv")
    assert ros.resolve("example_interfaces/Fibonacci") == ("example_interfaces/action/Fibonacci", "action")
    with pytest.raises(ToolError, match="no interface") as missing:
        ros.resolve("std_srvs/srv/Nope")
    assert "std_srvs/srv/Trigger" in missing.value.hint
    with pytest.raises(ToolError, match="is a msg, a srv is needed"):
        ros.resolve("std_msgs/msg/String", "srv")
    with pytest.raises(ToolError, match="no interface package"):
        ros.resolve("no_such_pkg_xyz/msg/Thing")
    with pytest.raises(ToolError, match="not an interface name"):
        ros.resolve("String")


def test_interface_shows_definition_and_fields():
    srv = ros.ros_interface("std_srvs/SetBool")
    assert srv["kind"] == "srv" and "bool data" in srv["definition"] and "---" in srv["definition"]
    assert srv["fields"] == {"request": {"data": "boolean"}, "response": {"success": "boolean", "message": "string"}}
    action = ros.ros_interface("example_interfaces/action/Fibonacci")
    assert action["fields"]["feedback"] == {"sequence": "sequence<int32>"}
    msg = ros.ros_interface("std_msgs/msg/String")
    assert msg["fields"] == {"data": "string"} and msg["path"].endswith("String.msg")


def test_clip_cuts_long_lists_and_names_them():
    clipped = []
    out = ros.clip({"ranges": list(range(10)), "header": {"frame_id": "x"}, "points": [{"v": [1, 2, 3]}]}, 2, "", clipped)
    assert out == {"ranges": [0, 1], "header": {"frame_id": "x"}, "points": [{"v": [1, 2]}]}
    assert clipped == ["ranges (10 items)", "points[].v (3 items)"]


class Peer:
    """A node serving and publishing what the ros domain is tested against."""

    def __init__(self):
        self.node = rclpy.create_node("peer", namespace=NS)
        self.node.declare_parameter("rate", 2.0, ParameterDescriptor(description="a double"))
        self.node.declare_parameter("label", "x")
        self.received = []
        self.cancelled = threading.Event()
        group = ReentrantCallbackGroup()
        self.node.create_service(SetBool, f"{NS}/toggle", self._toggle)
        self.chatter = self.node.create_publisher(String, f"{NS}/chatter", 10)
        self.camera = self.node.create_publisher(ImageMsg, f"{NS}/camera", 10)
        latched = self.node.create_publisher(String, f"{NS}/state", profile(LATCHED, 1))
        latched.publish(String(data="ready"))
        self._latched = latched
        self.node.create_subscription(Int32, f"{NS}/command", self.received.append, 10)
        self.count = 0
        self.node.create_timer(0.05, self._tick)
        ActionServer(self.node, Fibonacci, f"{NS}/fibonacci", self._execute, cancel_callback=lambda goal: CancelResponse.ACCEPT, callback_group=group)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def _toggle(self, request, response):
        response.success = request.data
        response.message = "on" if request.data else "off"
        return response

    def _tick(self):
        self.count += 1
        self.chatter.publish(String(data=f"n{self.count}"))
        image = ImageMsg(height=2, width=3, encoding="rgb8", step=9)
        image.data = bytes(range(18))
        self.camera.publish(image)

    def _execute(self, goal_handle):
        sequence = [0, 1]
        for _ in range(goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.cancelled.set()
                return Fibonacci.Result(sequence=sequence)
            sequence.append(sequence[-1] + sequence[-2])
            goal_handle.publish_feedback(Fibonacci.Feedback(sequence=sequence))
            time.sleep(0.05)
        goal_handle.succeed()
        return Fibonacci.Result(sequence=sequence)

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()


def test_ros_tools_against_a_live_peer():
    async def scenario():
        await ROS.node()
        peer = Peer()
        try:
            graph = await ros.ros_graph(prefix=NS)
            hidden = await ros.ros_graph(prefix=NS, hidden=True)
            called = await ros.ros_call(f"{NS}/toggle", request={"data": True})
            with pytest.raises(ToolError, match="cannot build"):
                await ros.ros_call(f"{NS}/toggle", "std_srvs/srv/SetBool", {"nope": 1})
            start = time.monotonic()
            with pytest.raises(ToolError, match="not available"):
                await ros.ros_call(f"{NS}/nobody", "std_srvs/srv/SetBool")
            fast = time.monotonic() - start
            echoed = await ros.ros_echo(f"{NS}/chatter", count=3)
            state = await ros.ros_echo(f"{NS}/state", qos="latched")
            picture = await ros.ros_echo(f"{NS}/camera", "sensor_msgs/Image")
            with pytest.raises(ToolError, match="nothing received"):
                await ros.ros_echo(f"{NS}/state", "std_msgs/msg/String", timeout_s=0.5)
            published = await ros.ros_publish(f"{NS}/command", "std_msgs/Int32", {"data": 7})
            deadline = time.monotonic() + 30.0
            while not peer.received and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            params = await ros.ros_params(f"{NS[1:]}/peer")
            updated = await ros.ros_set_params(f"{NS}/peer", {"rate": 3, "label": "y"})
            after = await ros.ros_params(f"{NS}/peer", ["rate", "label"])
            done = await ros.ros_action(f"{NS}/fibonacci", {"order": 5})
            finished = await JOBS.wait(done["id"], 30)
            long = await ros.ros_action(f"{NS}/fibonacci", {"order": 400}, "example_interfaces/action/Fibonacci")
            deadline = time.monotonic() + 30.0
            while not JOBS.get(long["id"]).progress.get("feedback_count") and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            cancelled = await JOBS.cancel(long["id"])
            server_saw_cancel = await asyncio.to_thread(peer.cancelled.wait, 20.0)
            return graph, hidden, called, fast, echoed, state, picture, published, list(peer.received), params, updated, after, finished, cancelled, server_saw_cancel
        finally:
            peer.close()
            await ROS.shutdown()

    graph, hidden, called, fast, echoed, state, picture, published, received, params, updated, after, finished, cancelled, server_saw_cancel = asyncio.run(scenario())
    assert f"{NS}/peer" in graph["nodes"]
    assert graph["topics"][f"{NS}/chatter"] == ["std_msgs/msg/String"]
    assert graph["services"][f"{NS}/toggle"] == ["std_srvs/srv/SetBool"]
    assert graph["actions"][f"{NS}/fibonacci"] == ["example_interfaces/action/Fibonacci"]
    assert not [name for name in graph["topics"] if "/_action/" in name]
    assert [name for name in hidden["topics"] if "/_action/" in name]
    assert called == {"service": f"{NS}/toggle", "type": "std_srvs/srv/SetBool", "response": {"success": True, "message": "on"}}
    assert fast < 10.0
    assert len(echoed["messages"]) == 3 and all(message["data"].startswith("n") for message in echoed["messages"])
    assert state["messages"] == [{"data": "ready"}]
    envelope, image = picture
    assert isinstance(image, Image)
    assert envelope["messages"][0]["encoding"] == "rgb8" and "data" not in envelope["messages"][0]
    assert published["subscribers"] == 1 and received[0].data == 7
    assert params["node"] == f"{NS}/peer" and params["params"]["rate"] == 2.0
    assert all(result["ok"] for result in updated["results"].values())
    assert after["params"] == {"rate": 3.0, "label": "y"}
    assert finished.state == "succeeded"
    assert finished.result == {"status": "succeeded", "result": {"sequence": [0, 1, 1, 2, 3, 5, 8]}}
    assert finished.progress["feedback_count"] >= 1 and finished.progress["feedback"]["sequence"][:3] == [0, 1, 1]
    assert cancelled.state == "cancelled" and server_saw_cancel


def test_echo_and_publish_validate_arguments():
    async def scenario():
        with pytest.raises(ToolError, match="count"):
            await ros.ros_echo("/x", "std_msgs/String", count=0)
        with pytest.raises(ToolError, match="unknown qos"):
            await ros.ros_echo("/x", "std_msgs/String", qos="best")
        with pytest.raises(ToolError, match="is a srv, a msg is needed"):
            await ros.ros_publish("/x", "std_srvs/srv/SetBool", {})
        with pytest.raises(ToolError, match="values is empty"):
            await ros.ros_set_params("/x", {})

    asyncio.run(scenario())
