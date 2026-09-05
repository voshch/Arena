from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


@pytest.fixture()
def stub_node():
    from arena_runtime.constants import SimSimulator

    class _FakeConf:
        class Robot:
            class TIMEOUT:
                value = 60
        class Arena:
            class SIM:
                value = SimSimulator.GAZEBO

    class _FakeLogger:
        def get_child(self, name):
            return self
        def debug(self, *a, **kw): ...
        def info(self, *a, **kw): ...
        def warn(self, *a, **kw): ...
        def error(self, *a, **kw): ...

    return SimpleNamespace(
        conf=_FakeConf(),
        sim_time=SimpleNamespace(sec=0),
        get_logger=lambda: _FakeLogger(),
    )


def _make_robot(name, model_name, mobile="nav2"):
    from arena_robots.Robot import RobotIdentifier
    from task_generator.shared import Pose, Robot
    return Robot(
        name=name,
        pose=Pose(),
        model=RobotIdentifier.parse(model_name),
        adapters={"mobile": mobile},
        extra={},
    )


def test_compatible_same_config():
    r1 = _make_robot("r1", "turtlebot3_burger")
    r2 = _make_robot("r2", "turtlebot3_burger")
    assert r1.compatible(r2) is True


def test_compatible_different_model():
    r1 = _make_robot("r1", "turtlebot3_burger")
    r2 = _make_robot("r2", "jackal")
    assert r1.compatible(r2) is False


def test_compatible_different_mobile_adapter():
    r1 = _make_robot("r1", "turtlebot3_burger", mobile="nav2")
    r2 = _make_robot("r2", "turtlebot3_burger", mobile="none")
    assert r1.compatible(r2) is False


def test_compatible_different_parts():
    from arena_robots.Robot import RobotIdentifier
    from task_generator.shared import Pose, Robot
    r1 = Robot(
        name="r1",
        pose=Pose(),
        model=RobotIdentifier.parse("turtlebot3_burger"),
        adapters={"mobile": "nav2"},
        parts={},
        extra={},
    )
    r2 = Robot(
        name="r2",
        pose=Pose(),
        model=RobotIdentifier.parse("turtlebot3_burger"),
        adapters={"mobile": "nav2"},
        parts={"lidar": ["sick"]},
        extra={},
    )
    assert r1.compatible(r2) is False


def test_parse_minimal_value(stub_node):
    from task_generator.shared import Robot
    value = {"name": "bot1", "model": "turtlebot3_burger"}
    robot = Robot.parse(value, node=stub_node)
    assert "mobile" not in robot.adapters


def test_parse_adapters_block_sets_adapters(stub_node):
    from task_generator.shared import Robot
    value = {
        "name": "bot2",
        "model": "turtlebot3_burger",
        "adapters": {"mobile": "none"},
    }
    robot = Robot.parse(value, node=stub_node)
    assert robot.adapters["mobile"] == "none"


def test_parse_flat_mobile_key_not_routed_to_adapters(stub_node):
    """Flat `mobile: x` is a morphology key (§2.1), not adapter sugar; parse
    leaves it out of `adapters` (only `mobile.adapter=` / the `adapters` block routes there)."""
    from task_generator.shared import Robot
    value = {
        "name": "bot2b",
        "model": "turtlebot3_burger",
        "mobile": "none",
    }
    robot = Robot.parse(value, node=stub_node)
    assert "mobile" not in robot.adapters


def test_parse_unknown_adapter_cap_raises(stub_node):
    from task_generator.shared import Robot
    value = {
        "name": "bot2c",
        "model": "turtlebot3_burger",
        "adapters": {"nonexistent_cap": "nav2"},
    }
    with pytest.raises(RuntimeError):
        Robot.parse(value, node=stub_node)


def test_parse_unknown_adapter_kind_raises(stub_node):
    from task_generator.shared import Robot
    value = {
        "name": "bot2d",
        "model": "turtlebot3_burger",
        "adapters": {"mobile": "nonexistent_kind"},
    }
    with pytest.raises(RuntimeError):
        Robot.parse(value, node=stub_node)


def test_parse_extra_dict_preserved(stub_node):
    from task_generator.shared import Robot
    value = {"name": "bot3", "model": "turtlebot3_burger", "custom_key": "custom_val"}
    robot = Robot.parse(value, node=stub_node)
    assert "custom_key" in robot.extra
    assert robot.extra["custom_key"] == "custom_val"


def test_parse_name_set_correctly(stub_node):
    from task_generator.shared import Robot
    value = {"name": "my_robot", "model": "turtlebot3_burger"}
    robot = Robot.parse(value, node=stub_node)
    assert robot.name == "my_robot"


def test_parse_default_pos_is_zero(stub_node):
    from task_generator.shared import Robot
    value = {"name": "bot4", "model": "turtlebot3_burger"}
    robot = Robot.parse(value, node=stub_node)
    assert robot.pose.position.x == 0.0
    assert robot.pose.position.y == 0.0


def test_frame_sim_path_branch():
    from arena_robots.Robot import RobotIdentifier
    from task_generator.shared import Pose, Robot
    robot = Robot(
        name="r",
        pose=Pose(),
        model=RobotIdentifier.parse("turtlebot3_burger"),
        extra={},
    )
    robot.sim_path = "simulation/r"
    frame = robot.frame
    assert "simulation/r" in str(frame)


def test_frame_name_fallback():
    from arena_robots.Robot import RobotIdentifier
    from task_generator.shared import Pose, Robot
    robot = Robot(
        name="robot_a",
        pose=Pose(),
        model=RobotIdentifier.parse("turtlebot3_burger"),
        extra={},
    )
    frame = robot.frame
    assert "robot_a" in str(frame)


def test_from_setup_delegates_to_parse(stub_node):
    from arena_robots.SetupFile import Config
    from task_generator.shared import Robot
    setup = Config(robot="turtlebot3_burger", name="setup_bot")
    robot = Robot.from_setup(setup, node=stub_node)
    assert robot.name == "setup_bot"
    assert robot.model.name == "turtlebot3_burger"


def test_eq_equal_when_all_fields_match():
    r1 = _make_robot("bot", "turtlebot3_burger")
    r2 = _make_robot("bot", "turtlebot3_burger")
    assert r1 == r2


def test_eq_not_equal_when_name_differs():
    r1 = _make_robot("bot_a", "turtlebot3_burger")
    r2 = _make_robot("bot_b", "turtlebot3_burger")
    assert r1 != r2


def test_eq_not_equal_when_model_differs():
    r1 = _make_robot("bot", "turtlebot3_burger")
    r2 = _make_robot("bot", "jackal")
    assert r1 != r2


def test_eq_not_equal_to_non_robot():
    r = _make_robot("bot", "turtlebot3_burger")
    assert (r == "some string") is False


def test_frame_empty_name_fallback_to_empty_string():
    from arena_robots.Robot import RobotIdentifier
    from task_generator.shared import Pose, Robot
    robot = Robot(
        name="",
        pose=Pose(),
        model=RobotIdentifier.parse("turtlebot3_burger"),
        extra={},
    )
    assert str(robot.frame) == ""


def test_parse_empty_parts_no_assembly_resolved_assembly_none(stub_node):
    from task_generator.shared import Robot
    value = {"name": "bot5", "model": "turtlebot3_burger"}
    robot = Robot.parse(value, node=stub_node)
    assert robot.resolved_assembly is None


def test_parse_empty_parts_with_assembly_resolves_default(stub_node):
    from task_generator.shared import Robot
    value = {"name": "bot5b", "model": "jackal"}
    robot = Robot.parse(value, node=stub_node)
    assert robot.resolved_assembly is not None


def test_parse_parts_without_assembly_raises(stub_node):
    from task_generator.shared import Robot
    value = {
        "name": "bot6",
        "model": "rbkairos_plus",
        "parts": {"lidar": ["sick"]},
    }
    with pytest.raises(RuntimeError, match="requires an assembly.yaml"):
        Robot.parse(value, node=stub_node)


def test_parse_frames_valid_mount_bakes_override_into_resolved_assembly(stub_node):
    """A deployment frames override on a declared, populated mount is baked onto
    resolved_assembly, winning over the mount's own declared frame (jackal's `front`
    declares frame: lidar; the override replaces it)."""
    from task_generator.shared import Robot
    value = {"name": "bot7", "model": "jackal", "frames": {"front": "custom_link"}}
    robot = Robot.parse(value, node=stub_node)
    front = next(p for p in robot.resolved_assembly.placements if p.mount.name == "front")
    assert front.mount.frame == "custom_link"


def test_parse_frames_unknown_mount_raises(stub_node):
    from task_generator.shared import Robot
    value = {"name": "bot8", "model": "jackal", "frames": {"bogus": "x"}}
    with pytest.raises(RuntimeError, match="unknown mount"):
        Robot.parse(value, node=stub_node)


def test_parse_frames_without_assembly_raises(stub_node):
    from task_generator.shared import Robot
    value = {"name": "bot9", "model": "rbkairos_plus", "frames": {"front": "x"}}
    with pytest.raises(RuntimeError, match="requires an assembly.yaml"):
        Robot.parse(value, node=stub_node)
