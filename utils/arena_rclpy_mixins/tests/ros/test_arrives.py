from __future__ import annotations

import asyncio
import threading

import rclpy
from geometry_msgs.msg import PoseStamped


def _make_node(name: str):
    from arena_rclpy_mixins.Async import AsyncNode

    return AsyncNode(name)


def _spin_in_background(node, stop: threading.Event) -> threading.Thread:
    def _spin() -> None:
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread


async def _republish_until_done(pub, messages, waiter: asyncio.Future) -> None:
    while not waiter.done():
        for message in messages:
            pub.publish(message)
        await asyncio.sleep(0.05)


def test_arrives_returns_first_message():
    async def main():
        node = _make_node("arrives_basic")
        stop = threading.Event()
        thread = _spin_in_background(node, stop)
        try:
            pub = node.create_publisher(PoseStamped, "arrives_basic_topic", 10)

            msg = PoseStamped()
            msg.header.stamp.sec = 42
            waiter = asyncio.ensure_future(node.arrives("arrives_basic_topic", PoseStamped, warn_period=0.01))
            publisher = asyncio.ensure_future(_republish_until_done(pub, [msg], waiter))
            result = await asyncio.wait_for(waiter, timeout=30.0)
            await publisher
            assert result.header.stamp.sec == 42
        finally:
            stop.set()
            thread.join(timeout=1.0)
            node.destroy_node()

    asyncio.run(main())


def test_arrives_filters_on_min_stamp():
    async def main():
        from arena_rclpy_mixins.Time import Time

        node = _make_node("arrives_min_stamp")
        stop = threading.Event()
        thread = _spin_in_background(node, stop)
        try:
            pub = node.create_publisher(PoseStamped, "arrives_min_stamp_topic", 10)

            stale = PoseStamped()
            stale.header.stamp.sec = 1
            fresh = PoseStamped()
            fresh.header.stamp.sec = 5
            waiter = asyncio.ensure_future(
                node.arrives(
                    "arrives_min_stamp_topic",
                    PoseStamped,
                    min_stamp=Time(sec=3, nanosec=0),
                    warn_period=0.01,
                )
            )
            publisher = asyncio.ensure_future(_republish_until_done(pub, [stale, fresh], waiter))
            result = await asyncio.wait_for(waiter, timeout=30.0)
            await publisher
            assert result.header.stamp.sec == 5
        finally:
            stop.set()
            thread.join(timeout=1.0)
            node.destroy_node()

    asyncio.run(main())
