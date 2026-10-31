"""Spin helpers for ROS-tier tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from types import ModuleType

from rclpy.node import Node


def spin_until(rclpy: ModuleType, nodes: Iterable[Node], predicate: Callable[[], bool], timeout_sec: float = 10.0) -> None:
    """Spin nodes round-robin until predicate holds, raise AssertionError at the deadline."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


def spin_in_background(rclpy: ModuleType, node: Node, stop: threading.Event) -> threading.Thread:
    """Spin node on a daemon thread until stop is set."""

    def _spin() -> None:
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread
