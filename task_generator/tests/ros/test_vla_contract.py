from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def test_parse_waypoints():
    from task_generator.tasks.robots.vla.contract import Waypoints, parse

    resp = parse({"actions": {"mobile": {"waypoints": [{"x": 1.0, "y": 2.0, "yaw": 0.5}, {"x": 3.0, "y": 4.0, "yaw": -0.5}]}}})

    assert isinstance(resp.actions.mobile, Waypoints)
    assert [(w.x, w.y, w.yaw) for w in resp.actions.mobile.steps] == [(1.0, 2.0, 0.5), (3.0, 4.0, -0.5)]


def test_parse_meta_absent_defaults_empty_intent():
    from task_generator.tasks.robots.vla.contract import parse

    resp = parse({"actions": {"mobile": {"waypoints": []}}})

    assert resp.meta.intent == ""


def test_parse_meta_intent_populated():
    from task_generator.tasks.robots.vla.contract import parse

    resp = parse({"actions": {"mobile": {"waypoints": []}}, "meta": {"intent": "head for the door"}})

    assert resp.meta.intent == "head for the door"


def test_parse_mobile_absent():
    from task_generator.tasks.robots.vla.contract import parse

    resp = parse({"actions": {}})

    assert resp.actions.mobile is None


def test_parse_unknown_mobile_form_raises():
    from task_generator.tasks.robots.vla.contract import parse

    with pytest.raises(ValueError, match="cmd_vel"):
        parse({"actions": {"mobile": {"cmd_vel": [{"vx": 0.1, "wz": 0.0}]}}})


def test_parse_rejects_multi_key_union():
    from task_generator.tasks.robots.vla.contract import parse

    # the cap union is key-tagged: exactly one form per cap
    with pytest.raises(ValueError):
        parse({"actions": {"mobile": {"waypoints": [], "cmd_vel": []}}})
