from __future__ import annotations

import rclpy
import rclpy.node
from human_steering.driver import resolve_namespace
from std_msgs.msg import Empty


def test_resolve_namespace_finds_manifest_topic() -> None:
    node = rclpy.node.Node("human_steering_test_publisher")
    discoverer = rclpy.node.Node("human_steering_test_discoverer")
    try:
        # any message type will do: resolve_namespace only cares about the topic
        # name suffix, matching the arena viz manifest-suffix convention.
        node.create_publisher(Empty, "/arena/env_7/arena_env_node/state/viz_manifest", 1)

        namespaces = None
        for _ in range(50):
            rclpy.spin_once(discoverer, timeout_sec=0.1)
            namespaces = resolve_namespace(discoverer, target="env_7")
            if namespaces is not None:
                break

        assert namespaces is not None
        assert namespaces.node_ns == "/arena/env_7/arena_env_node"
        assert namespaces.env_ns == "/arena/env_7"
        assert namespaces.map_topic == "/arena/env_7/arena_env_node/map"
    finally:
        node.destroy_node()
        discoverer.destroy_node()


def test_resolve_namespace_returns_none_without_target_match() -> None:
    discoverer = rclpy.node.Node("human_steering_test_no_match")
    try:
        assert resolve_namespace(discoverer, target="no_such_env") is None
    finally:
        discoverer.destroy_node()
