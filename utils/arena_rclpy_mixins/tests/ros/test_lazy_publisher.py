from __future__ import annotations

import time

import rclpy
import rclpy.node
from arena_rclpy_mixins.lazy import LazyPublisher, LazySubscription
from arena_rclpy_mixins.qos import latched
from std_msgs.msg import String


def _spin_until(node: rclpy.node.Node, done, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    return done()


def test_builds_nothing_without_a_subscriber():
    node = rclpy.create_node("lazy_publisher_idle")
    built: list[str] = []
    try:
        pub: LazyPublisher[String] = LazyPublisher(node.create_publisher(String, "lazy_idle", 10))
        assert not pub.wanted
        assert not pub.publish(lambda: built.append("x") or String(data="x"))
        assert built == []
    finally:
        node.destroy_node()


def test_publishes_once_a_subscriber_joins():
    node = rclpy.create_node("lazy_publisher_watched")
    received: list[str] = []
    try:
        pub: LazyPublisher[String] = LazyPublisher(node.create_publisher(String, "lazy_watched", 10))
        node.create_subscription(String, "lazy_watched", lambda m: received.append(m.data), 10)
        assert _spin_until(node, lambda: pub.wanted)
        assert pub.publish(lambda: String(data="hello"))
        assert _spin_until(node, lambda: received == ["hello"])
    finally:
        node.destroy_node()


def test_latched_publisher_sends_without_a_subscriber():
    node = rclpy.create_node("lazy_publisher_latched")
    received: list[str] = []
    try:
        pub: LazyPublisher[String] = LazyPublisher(node.create_publisher(String, "lazy_latched", latched(1)))
        assert pub.wanted
        assert pub.publish(lambda: String(data="kept"))
        node.create_subscription(String, "lazy_latched", lambda m: received.append(m.data), latched(1))
        assert _spin_until(node, lambda: received == ["kept"])
    finally:
        node.destroy_node()


def test_subscription_follows_the_publisher_it_feeds():
    node = rclpy.create_node("lazy_subscription_relay")
    received: list[str] = []
    try:
        upstream = node.create_publisher(String, "lazy_relay_in", 10)
        out: LazyPublisher[String] = LazyPublisher(node.create_publisher(String, "lazy_relay_out", 10))
        relay = LazySubscription(node, out, String, "lazy_relay_in", lambda m: out.publish(lambda: m), 10, period_s=0.05)
        assert not relay.active
        assert upstream.get_subscription_count() == 0
        viewer = node.create_subscription(String, "lazy_relay_out", lambda m: received.append(m.data), 10)
        assert _spin_until(node, lambda: upstream.get_subscription_count() == 1)
        assert relay.active
        upstream.publish(String(data="through"))
        assert _spin_until(node, lambda: received == ["through"])
        node.destroy_subscription(viewer)
        assert _spin_until(node, lambda: upstream.get_subscription_count() == 0)
        assert not relay.active
    finally:
        node.destroy_node()


def test_subscription_lives_while_any_fed_publisher_is_wanted():
    node = rclpy.create_node("lazy_subscription_fanout")
    try:
        upstream = node.create_publisher(String, "lazy_fanout_in", 10)
        first: LazyPublisher[String] = LazyPublisher(node.create_publisher(String, "lazy_fanout_a", 10))
        second: LazyPublisher[String] = LazyPublisher(node.create_publisher(String, "lazy_fanout_b", 10))
        relay = LazySubscription(node, (first, second), String, "lazy_fanout_in", lambda _m: None, 10, period_s=0.05)
        assert not relay.active
        viewer = node.create_subscription(String, "lazy_fanout_b", lambda _m: None, 10)
        assert _spin_until(node, lambda: upstream.get_subscription_count() == 1)
        node.destroy_subscription(viewer)
        assert _spin_until(node, lambda: upstream.get_subscription_count() == 0)
        assert not relay.active
    finally:
        node.destroy_node()
