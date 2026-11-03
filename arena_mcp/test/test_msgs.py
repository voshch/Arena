from __future__ import annotations

import importlib
import inspect
import math

import pytest

pytest.importorskip("rosidl_runtime_py")

from arena_mcp.core.errors import ToolError
from arena_mcp.core.msgs import constants, fill, pose, pose_dict, to_dict

PACKAGES = ["arena_runtime_msgs", "arena_runtime_msgs", "arena_robots_msgs", "world_generator_msgs"]


def _message_types():
    for package in PACKAGES:
        try:
            module = importlib.import_module(f"{package}.msg")
        except ImportError:
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__.startswith(f"{package}.msg"):
                yield pytest.param(cls, id=f"{package}/{name}")


@pytest.mark.parametrize("msg_type", list(_message_types()))
def test_every_message_round_trips_through_dicts(msg_type):
    data = to_dict(msg_type())
    assert to_dict(fill(msg_type, data)) == data


def test_fill_rejects_unknown_fields_with_the_field_list():
    from arena_runtime_msgs.msg import RobotDescriptor

    with pytest.raises(ToolError, match="fields: name, model"):
        fill(RobotDescriptor, {"nmae": "jackal"})


def test_constants_and_pose_helpers():
    from arena_runtime_msgs.msg import EpisodeRecord

    names = constants(EpisodeRecord)
    assert names
    value = pose(1.0, -2.0, yaw=math.pi / 2)
    out = pose_dict(value)
    assert out["x"] == 1.0 and out["y"] == -2.0 and math.isclose(out["yaw"], math.pi / 2)
