"""The drive panel's face path against live faces/tracked and view topics."""

from __future__ import annotations

import math
import time

import pytest
import rclpy
import rclpy.executors
import rclpy.node
from arena_people_msgs.msg import FaceView
from hri_msgs.msg import IdsList
from sensor_msgs.msg import CameraInfo

from arena_cam.drive import Driver, EntityRoster
from arena_cam.fly import Mode
from arena_cam.surfaces import TargetSelection

_TRACKED = "/arena/env_77/humans/faces/tracked"
_FACE = "env_77_agent_2"
_VIEW = f"/arena/env_77/humans/faces/{_FACE}/view"


@pytest.fixture(scope="module", autouse=True)
def _rclpy():
    rclpy.init()
    yield
    rclpy.shutdown()


def _spin_until(executor: rclpy.executors.Executor, done, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
    return done()


def test_framing_a_face_looks_through_it_until_back_to_world():
    producer = rclpy.create_node("drive_faces_producer")
    panel = rclpy.create_node("drive_faces_panel")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(producer)
    executor.add_node(panel)
    try:
        tracked = producer.create_publisher(IdsList, _TRACKED, 10)
        view = producer.create_publisher(FaceView, _VIEW, 10)
        f = 256.0 / math.tan(math.radians(25.0))
        producer.create_timer(0.05, lambda: tracked.publish(IdsList(ids=[_FACE])))
        info = CameraInfo(width=512, height=512, k=[f, 0.0, 256.0, 0.0, f, 256.0, 0.0, 0.0, 1.0])
        producer.create_timer(0.05, lambda: view.publish(FaceView(info=info, clip_near=0.12)))

        roster = EntityRoster(panel)
        driver = Driver(panel, TargetSelection(include_sim=False, viz_all=False, viz_env=None), roster=roster)

        def listed() -> bool:
            roster.refresh()
            return f"face_{_FACE}" in roster.names()

        assert _spin_until(executor, listed)
        driver.entity = f"face_{_FACE}"
        driver.frame()
        assert driver.anchor == f"face_{_FACE}"
        assert driver.fly.mode is Mode.FLY
        assert driver.fly.pos == (0.0, 0.0, 0.0)
        assert driver.fly.forward_vec() == pytest.approx((1.0, 0.0, 0.0))
        assert _spin_until(executor, lambda: view.get_subscription_count() == 1)
        assert _spin_until(executor, lambda: driver.fly.fov == pytest.approx(math.radians(50.0)))
        assert "fov 50 deg" in driver.status
        assert driver.next_frame(0.03).clip_near == 0.12

        driver.to_world()
        assert driver.anchor == ""
        assert driver.next_frame(0.03).clip_near < 0.0
        assert _spin_until(executor, lambda: view.get_subscription_count() == 0)
    finally:
        executor.shutdown()
        panel.destroy_node()
        producer.destroy_node()


def test_robot_frames_have_no_view():
    node = rclpy.create_node("drive_faces_roster")
    try:
        assert EntityRoster(node).view_topic("env_0/jackal/base_link") is None
        assert EntityRoster(node).view_topic(f"face_{_FACE}") is None
    finally:
        node.destroy_node()
