"""hri_producer joint frames against a real robot_state_publisher on the same joint_states."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import rclpy
import rclpy.executors
import rclpy.node
import yaml
from ament_index_python.packages import get_package_prefix
from arena_people_msgs.msg import Pedestrian, Pedestrians
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from rviz_utils.scripts.hri_producer import HriProducer

_BODY = "env_agent_4"
_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _spin_until(executor: rclpy.executors.Executor, done, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
    return done()


def _start_rsp(urdf: str) -> tuple[subprocess.Popen[bytes], str]:
    handle, params = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(handle, "w") as out:
        yaml.safe_dump({"/**": {"ros__parameters": {"robot_description": urdf}}}, out, width=10**9)
    binary = f"{get_package_prefix('robot_state_publisher')}/lib/robot_state_publisher/robot_state_publisher"
    cmd = [binary, "--ros-args", "--log-level", "warn", "--params-file", params, "--remap", "__node:=rsp_parity", "--remap", f"joint_states:=/humans/bodies/{_BODY}/joint_states", "--remap", "/tf:=/rsp/tf", "--remap", "/tf_static:=/rsp/tf_static", "--remap", "robot_description:=/rsp/robot_description"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL), params


def _key(tf) -> tuple[str, str]:
    return tf.header.frame_id, tf.child_frame_id


def _assert_same_transform(ours, theirs) -> None:
    a, b = ours.transform, theirs.transform
    assert abs(a.translation.x - b.translation.x) < 1e-6
    assert abs(a.translation.y - b.translation.y) < 1e-6
    assert abs(a.translation.z - b.translation.z) < 1e-6
    dot = a.rotation.x * b.rotation.x + a.rotation.y * b.rotation.y + a.rotation.z * b.rotation.z + a.rotation.w * b.rotation.w
    assert abs(abs(dot) - 1.0) < 1e-6


def test_joint_frames_match_robot_state_publisher():
    producer = HriProducer()
    host = rclpy.create_node("hri_bones_host")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(producer)
    executor.add_node(host)
    urdfs: list[String] = []
    ours: dict[int, dict[tuple[str, str], object]] = {}
    theirs: dict[int, dict[tuple[str, str], object]] = {}
    ours_static: list[TFMessage] = []
    theirs_static: list[TFMessage] = []
    states: list[JointState] = []
    rsp = None
    params = None

    def collect(store):
        def on_tf(msg: TFMessage) -> None:
            for tf in msg.transforms:
                stamp = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
                store.setdefault(stamp, {})[_key(tf)] = tf

        return on_tf

    try:
        peds = host.create_publisher(Pedestrians, "/arena_peds", 10)

        def publish() -> None:
            msg = Pedestrians()
            msg.header.frame_id = "map"
            ped = Pedestrian(name="ped_4", id=4, animation_state=Pedestrian.WALKING)
            ped.pose.orientation.w = 1.0
            ped.twist.linear.x = 1.0
            msg.pedestrians = [ped]
            peds.publish(msg)

        host.create_timer(0.05, publish)
        host.create_subscription(TFMessage, "/humans/tf", collect(ours), 100)
        host.create_subscription(TFMessage, "/rsp/tf", collect(theirs), 100)
        host.create_subscription(TFMessage, "/tf_static", ours_static.append, _LATCHED)
        host.create_subscription(TFMessage, "/rsp/tf_static", theirs_static.append, _LATCHED)
        host.create_subscription(String, f"/humans/bodies/{_BODY}/urdf", urdfs.append, _LATCHED)
        host.create_subscription(JointState, f"/humans/bodies/{_BODY}/joint_states", states.append, 10)

        assert _spin_until(executor, lambda: urdfs and states)
        rsp, params = _start_rsp(urdfs[0].data)

        def bones(stamp: int) -> dict[tuple[str, str], object]:
            return {key: tf for key, tf in ours[stamp].items() if key[0] != "map" and key[1] != "ped_4"}

        def common() -> list[int]:
            return [stamp for stamp in theirs if stamp in ours and len(theirs[stamp]) == len(bones(stamp)) == len(states[0].name)]

        assert _spin_until(executor, lambda: len(common()) >= 5 and ours_static and theirs_static)

        states_moved = {tuple(state.position) for state in states}
        assert len(states_moved) > 1
        for stamp in common():
            assert bones(stamp).keys() == theirs[stamp].keys()
            for key, tf in bones(stamp).items():
                _assert_same_transform(tf, theirs[stamp][key])

        ours_fixed = {_key(tf): tf for tf in ours_static[-1].transforms}
        theirs_fixed = {_key(tf): tf for tf in theirs_static[-1].transforms}
        assert ours_fixed.keys() == theirs_fixed.keys()
        assert ours_fixed
        for key, tf in ours_fixed.items():
            _assert_same_transform(tf, theirs_fixed[key])
    finally:
        if rsp is not None:
            rsp.terminate()
            rsp.wait(timeout=10.0)
        if params is not None:
            os.unlink(params)
        executor.shutdown()
        producer.destroy_node()
        host.destroy_node()


def test_producer_starts_no_child_process():
    producer = HriProducer()
    host = rclpy.create_node("hri_children_host")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(producer)
    executor.add_node(host)
    bones: list[TFMessage] = []
    try:
        peds = host.create_publisher(Pedestrians, "/arena_peds", 10)

        def publish() -> None:
            msg = Pedestrians()
            msg.header.frame_id = "map"
            ped = Pedestrian(name="ped_4", id=4, animation_state=Pedestrian.IDLE)
            ped.pose.orientation.w = 1.0
            msg.pedestrians = [ped]
            peds.publish(msg)

        host.create_timer(0.05, publish)
        host.create_subscription(TFMessage, "/humans/tf", bones.append, 100)
        assert _spin_until(executor, lambda: any(len(msg.transforms) > 2 for msg in bones))
        pid = os.getpid()
        children = [int(c) for t in os.listdir(f"/proc/{pid}/task") for c in open(f"/proc/{pid}/task/{t}/children").read().split()]
        assert children == []
    finally:
        executor.shutdown()
        producer.destroy_node()
        host.destroy_node()
