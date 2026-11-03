from __future__ import annotations

import types

import pytest

geometry_msgs = pytest.importorskip("geometry_msgs.msg")

from arena_runtime.env.realizer import Realizer
from arena_runtime.env.episode_node import EpisodeNode, _is_map_frame


def _stamped(frame: str, x: float, y: float) -> object:
    msg = geometry_msgs.PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


@pytest.mark.parametrize(("frame", "accepted"), [("", True), ("map", True), ("env_0/map", True), ("odom", False), ("env_0/jackal/base_link", False), ("mapper", False)])
def test_map_frames_are_the_plain_and_the_env_namespaced_map(frame, accepted):
    assert _is_map_frame(frame) is accepted


def test_spawn_poses_are_shifted_from_the_map_frame_into_the_world_frame():
    holder = types.SimpleNamespace(_realizer=Realizer(Realizer._Configuration(x=5.0, y=7.0, prefix="env_0")))
    pose = EpisodeNode._pose_from_request(holder, _stamped("env_0/map", 6.0, 10.0))
    assert (pose.position.x, pose.position.y) == pytest.approx((1.0, 3.0))
    with pytest.raises(ValueError, match="map frame"):
        EpisodeNode._pose_from_request(holder, _stamped("odom", 6.0, 10.0))


def test_submitted_goto_poses_are_shifted_from_the_map_frame_into_the_world_frame():
    from arena_simulation_setup.shared.task import TaskRequest

    holder = types.SimpleNamespace(_realizer=Realizer(Realizer._Configuration(x=5.0, y=7.0, prefix="env_0")))
    holder._ezilear_phases = lambda phases: EpisodeNode._ezilear_phases(holder, phases)
    task = TaskRequest.parse([{"goto": [6.0, 10.0, 0.0]}, {"loop": [{"goto": [5.0, 7.0, 0.0]}, {"goto": "kitchen"}]}])
    EpisodeNode._ezilear_phases(holder, task.phases)
    assert (task.phases[0].pose.position.x, task.phases[0].pose.position.y) == pytest.approx((1.0, 3.0))
    first, named = task.phases[1].phases
    assert (first.pose.position.x, first.pose.position.y) == pytest.approx((0.0, 0.0))
    assert named.pose is None and named.target == "kitchen"
