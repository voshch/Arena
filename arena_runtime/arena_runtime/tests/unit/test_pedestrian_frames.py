from __future__ import annotations

import pytest

pytest.importorskip("arena_people_msgs")

from arena_people_msgs.msg import Pedestrian
from builtin_interfaces.msg import Time

from arena_runtime.humans import BaseHumanSimulator, ped_transform


def test_a_pedestrian_frame_sits_under_its_published_name():
    assert BaseHumanSimulator.pedestrian_frame("env_0/host") == "env_0/host/base_link"


def test_a_pedestrian_transform_carries_its_pose_from_the_stream_frame():
    ped = Pedestrian()
    ped.name = "env_0/host"
    ped.pose.position.x, ped.pose.position.y, ped.pose.position.z = 1.5, -2.0, 0.25
    ped.pose.orientation.z, ped.pose.orientation.w = 0.6, 0.8
    transform = ped_transform(ped, "map", BaseHumanSimulator.pedestrian_frame(ped.name), Time(sec=7))

    assert (transform.header.frame_id, transform.child_frame_id, transform.header.stamp.sec) == ("map", "env_0/host/base_link", 7)
    translation, rotation = transform.transform.translation, transform.transform.rotation
    assert (translation.x, translation.y, translation.z) == (1.5, -2.0, 0.25)
    assert (rotation.z, rotation.w) == (0.6, 0.8)
