"""The task generator's static broadcaster replaces a resent child frame."""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def node():
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.node import Node

    context = Context()
    rclpy.init(context=context)
    probe = Node("static_tf_probe", context=context)
    yield probe
    probe.destroy_node()
    rclpy.shutdown(context=context)


def _transform(child: str, x: float, y: float):
    import geometry_msgs.msg

    t = geometry_msgs.msg.TransformStamped()
    t.header.frame_id = "map"
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.w = 1.0
    return t


def _latched(node) -> list[tuple[str, float, float]]:
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from tf2_msgs.msg import TFMessage

    received: list[TFMessage] = []
    qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(TFMessage, "/tf_static", received.append, qos)
    executor = SingleThreadedExecutor(context=node.context)
    executor.add_node(node)
    deadline = time.monotonic() + 5.0
    while not received and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
    executor.remove_node(node)
    node.destroy_subscription(sub)
    assert len(received) == 1
    return [(t.child_frame_id, t.transform.translation.x, t.transform.translation.y) for t in received[0].transforms]


def test_resent_anchor_replaces_the_first_one(node):
    from task_generator.utils.static_tf import StaticTransformBroadcaster

    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(_transform("env_0/map", 4.975, 152.075))
    broadcaster.sendTransform([_transform("env_0/jackal", 0.0, 0.0)])
    broadcaster.sendTransform(_transform("env_0/map", 773.354, 5.0))

    assert _latched(node) == [("env_0/map", 773.354, 5.0), ("env_0/jackal", 0.0, 0.0)]
