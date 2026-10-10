"""hri_producer face frames and views against a live Pedestrians stream."""

from __future__ import annotations

import time

import rclpy
import rclpy.executors
import rclpy.node
from arena_people_msgs.msg import FaceView, Pedestrian, Pedestrians
from hri_msgs.msg import IdsList
from tf2_msgs.msg import TFMessage

from rviz_utils.scripts.hri_producer import HriProducer

_FACE = "env_agent_4"
_VIEW = f"/humans/faces/{_FACE}/view"


def _spin_until(executor: rclpy.executors.Executor, done, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
    return done()


def _spin_for(executor: rclpy.executors.Executor, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)


def test_face_frames_follow_the_view_subscription():
    producer = HriProducer()
    host = rclpy.create_node("hri_faces_host")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(producer)
    executor.add_node(host)
    tracked: list[list[str]] = []
    frames: list[TFMessage] = []
    views: list[FaceView] = []
    try:
        peds = host.create_publisher(Pedestrians, "/arena_peds", 10)

        def publish() -> None:
            msg = Pedestrians()
            msg.header.frame_id = "map"
            ped = Pedestrian(name="ped_4", id=4, animation_state=Pedestrian.IDLE)
            ped.pose.position.x, ped.pose.position.y = 3.0, -1.0
            ped.pose.orientation.w = 1.0
            msg.pedestrians = [ped]
            peds.publish(msg)

        host.create_timer(0.05, publish)
        host.create_subscription(TFMessage, "/tf", frames.append, 100)
        roster = host.create_subscription(IdsList, "/humans/faces/tracked", lambda msg: tracked.append(list(msg.ids)), 10)

        assert _spin_until(executor, lambda: [_FACE] in tracked)
        _spin_for(executor, 1.0)
        assert not [t for msg in frames for t in msg.transforms if t.child_frame_id == f"face_{_FACE}"]

        view = host.create_subscription(FaceView, _VIEW, views.append, 10)
        assert _spin_until(executor, lambda: views and any(t.child_frame_id == f"face_{_FACE}" for msg in frames for t in msg.transforms))
        assert views[-1].info.header.frame_id == f"gaze_{_FACE}"
        assert views[-1].clip_near == 0.15
        face = next(t for msg in reversed(frames) for t in msg.transforms if t.child_frame_id == f"face_{_FACE}")
        assert face.header.frame_id == "map"
        assert abs(face.transform.translation.x - 3.1) < 0.1
        assert abs(face.transform.translation.y + 1.0) < 0.1
        assert 1.4 < face.transform.translation.z < 1.7
        gaze = next(t for msg in reversed(frames) for t in msg.transforms if t.child_frame_id == f"gaze_{_FACE}")
        assert gaze.header.frame_id == f"face_{_FACE}"

        host.destroy_subscription(view)
        _spin_for(executor, 0.5)
        frames.clear()
        _spin_for(executor, 1.0)
        assert not [t for msg in frames for t in msg.transforms if t.child_frame_id.startswith(("face_", "gaze_"))]

        host.destroy_subscription(roster)
        tracked.clear()
        assert _spin_until(executor, lambda: not [i for i in host.get_subscriptions_info_by_topic("/arena_peds") if i.node_name == producer.get_name()])
    finally:
        executor.shutdown()
        producer.destroy_node()
        host.destroy_node()
