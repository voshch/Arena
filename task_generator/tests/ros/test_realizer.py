from __future__ import annotations

import math
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


@pytest.fixture()
def realizer():
    from task_generator.manager.realizer import Realizer
    config = Realizer._Configuration(x=1.0, y=2.0, prefix="world")
    return Realizer(config)


@pytest.fixture()
def zero_realizer():
    from task_generator.manager.realizer import Realizer
    config = Realizer._Configuration(x=0.0, y=0.0, prefix="")
    return Realizer(config)


@pytest.fixture()
def _pose():
    from arena_simulation_setup.utils.geometry import Pose
    return Pose


@pytest.fixture()
def _position():
    from arena_simulation_setup.utils.geometry import Position
    return Position


@pytest.fixture()
def _orientation():
    from arena_simulation_setup.utils.geometry import Orientation
    return Orientation


def test_realize_none_returns_prefixed_namespace(realizer):
    result = realizer.realize(None)
    assert isinstance(result, str)
    assert "world" in result


def test_realize_none_zero_prefix_returns_empty_or_slash(zero_realizer):
    result = zero_realizer.realize(None)
    assert isinstance(result, str)


def test_realize_str_adds_prefix(realizer):
    result = realizer.realize("robot1")
    assert isinstance(result, str)
    assert "robot1" in result
    assert "world" in result


def test_realize_str_zero_prefix(zero_realizer):
    result = zero_realizer.realize("robot1")
    assert "robot1" in result


def test_realize_position_translates_xy(_position, realizer):
    from arena_simulation_setup.utils.geometry import Position
    p = Position(x=3.0, y=4.0, z=5.0)
    result = realizer.realize(p)
    assert isinstance(result, Position)
    assert math.isclose(result.x, 4.0)
    assert math.isclose(result.y, 6.0)
    assert math.isclose(result.z, 5.0)


def test_realize_position_zero_offset_unchanged(zero_realizer, _position):
    from arena_simulation_setup.utils.geometry import Position
    p = Position(x=7.0, y=-3.0, z=1.0)
    result = zero_realizer.realize(p)
    assert math.isclose(result.x, 7.0)
    assert math.isclose(result.y, -3.0)


def test_realize_pose_translates_position(_pose, realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation
    p = Pose(Position(0.0, 0.0), Orientation.from_yaw(0.5))
    result = realizer.realize(p)
    assert math.isclose(result.position.x, 1.0)
    assert math.isclose(result.position.y, 2.0)
    assert math.isclose(result.orientation.to_yaw(), 0.5, abs_tol=1e-6)


def test_realize_pose_orientation_preserved(realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation
    yaw = 1.2
    p = Pose(Position(1.0, 1.0), Orientation.from_yaw(yaw))
    result = realizer.realize(p)
    assert math.isclose(result.orientation.to_yaw(), yaw, abs_tol=1e-6)


def test_ezilear_translates_pose_by_negative_offset(realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation
    p = Pose(Position(5.0, 7.0), Orientation.from_yaw(0.5))
    result = realizer.ezilear(p)
    assert math.isclose(result.position.x, 4.0)
    assert math.isclose(result.position.y, 5.0)
    assert math.isclose(result.orientation.to_yaw(), 0.5, abs_tol=1e-6)


def test_ezilear_round_trips_with_realize(realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation
    p = Pose(Position(3.0, -2.5), Orientation.from_yaw(1.2))
    assert math.isclose(realizer.ezilear(realizer.realize(p)).position.x, p.position.x)
    assert math.isclose(realizer.ezilear(realizer.realize(p)).position.y, p.position.y)
    assert math.isclose(realizer.realize(realizer.ezilear(p)).position.x, p.position.x)
    assert math.isclose(realizer.realize(realizer.ezilear(p)).position.y, p.position.y)


def test_realize_wall_translates_start_end(realizer):
    from arena_simulation_setup.shared import Wall
    from arena_simulation_setup.utils.geometry import Position
    w = Wall(start=Position(0, 0), end=Position(1, 0))
    result = realizer.realize(w)
    assert math.isclose(result.start.x, 1.0)
    assert math.isclose(result.start.y, 2.0)
    assert math.isclose(result.end.x, 2.0)
    assert math.isclose(result.end.y, 2.0)


def test_realize_door_translates_and_prefixes_name(realizer):
    from arena_simulation_setup.shared import Door
    from arena_simulation_setup.utils.geometry import Position
    d = Door(start=Position(0, 0), end=Position(1, 0), name="door1", extra={})
    result = realizer.realize(d)
    assert "door1" in result.name
    assert math.isclose(result.start.x, 1.0)
    assert math.isclose(result.end.x, 2.0)


def test_realize_floor_translates_and_prefixes_name(realizer):
    from arena_simulation_setup.shared import Floor
    from arena_simulation_setup.utils.geometry import Position
    f = Floor(pos=Position(0, 0), name="floor1", extra={})
    result = realizer.realize(f)
    assert "floor1" in result.name
    assert math.isclose(result.pos.x, 1.0)
    assert math.isclose(result.pos.y, 2.0)


def test_realize_door_preserves_semantics_with_prefixed_name(realizer):
    from arena_simulation_setup.shared import Door
    from arena_simulation_setup.utils.geometry import Position
    d = Door(start=Position(0, 0), end=Position(1, 0), name="door1", semantics=[{"preset": "gate"}], extra={})
    result = realizer.realize(d)
    assert "door1" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in d.semantics]


def test_realize_elevator_translates_and_prefixes(realizer):
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position
    e = Elevator(position=Position(0, 0), name="elev1", extra={})
    result = realizer.realize(e)
    assert "elev1" in result.name
    assert math.isclose(result.position.x, 1.0)


def test_realize_elevator_preserves_semantics_with_prefixed_name(realizer):
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position
    e = Elevator(position=Position(0, 0), name="elev1", semantics=[{"preset": "pressure_plate"}], extra={})
    result = realizer.realize(e)
    assert "elev1" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in e.semantics]


def test_realize_elevator_no_destination(realizer):
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position
    e = Elevator(position=Position(0, 0), destination="", name="elev2", extra={})
    result = realizer.realize(e)
    assert result.destination == ""


def test_realize_elevator_with_destination_prefixed(realizer):
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position
    e = Elevator(position=Position(0, 0), destination="floor2", name="elev3", extra={})
    result = realizer.realize(e)
    assert "floor2" in result.destination


def test_realize_dynamic_obstacle_translates_pose_and_waypoints(realizer):
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.utils.geometry import Position, Pose
    obs = DynamicObstacle(
        name="human1",
        pose=Pose(Position(0, 0)),
        model="human",
        waypoints=[Position(1, 0), Position(2, 1)],
        extra={},
    )
    result = realizer.realize(obs)
    assert math.isclose(result.pose.position.x, 1.0)
    assert math.isclose(result.pose.position.y, 2.0)
    assert len(result.waypoints) == 2
    assert math.isclose(result.waypoints[0].x, 2.0)
    assert math.isclose(result.waypoints[1].x, 3.0)
    assert math.isclose(result.waypoints[1].y, 3.0)


def test_realize_dynamic_obstacle_empty_waypoints(realizer):
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.utils.geometry import Position, Pose
    obs = DynamicObstacle(
        name="human2",
        pose=Pose(Position(0, 0)),
        model="human",
        waypoints=[],
        extra={},
    )
    result = realizer.realize(obs)
    assert result.waypoints == []


def test_realize_position_inverse_round_trip(realizer):
    from arena_simulation_setup.utils.geometry import Position
    p = Position(x=5.0, y=-3.0, z=1.0)
    forward = realizer._realize_position(p)
    back = realizer._realize_position_inv(forward)
    assert math.isclose(back.x, p.x, abs_tol=1e-9)
    assert math.isclose(back.y, p.y, abs_tol=1e-9)
    assert math.isclose(back.z, p.z, abs_tol=1e-9)


def test_realize_unknown_type_raises(realizer):
    with pytest.raises(TypeError, match="realization not implemented"):
        realizer.realize(object())


def test_realize_schedule_prefixes_name(realizer):
    from arena_simulation_setup.shared import Schedule
    s = Schedule(name="fire_alarm", semantics=[{"preset": "schedule"}], extra={})
    result = realizer.realize(s)
    assert "fire_alarm" in result.name
    assert "world" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in s.semantics]


def test_realize_signal_prefixes_name(realizer):
    from arena_simulation_setup.shared import Signal
    s = Signal(name="crossing_1", semantics=[{"preset": "signal"}], extra={})
    result = realizer.realize(s)
    assert "crossing_1" in result.name
    assert "world" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in s.semantics]


def test_realize_polygon_translates_corners(realizer):
    from arena_simulation_setup.utils.geometry import Position
    corners = [Position(0, 0), Position(2, 0), Position(2, 2), Position(0, 2)]
    ring = realizer.realize_polygon(corners)
    assert ring == [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)]


def test_realize_polygon_zero_offset(zero_realizer):
    from arena_simulation_setup.utils.geometry import Position
    corners = [Position(1, 1), Position(3, 1), Position(3, 3)]
    ring = zero_realizer.realize_polygon(corners)
    assert ring == [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0)]


def test_entity_sim_path_set_after_realize(realizer):
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.utils.geometry import Pose, Position
    obs = Obstacle(
        name="box1",
        pose=Pose(Position(0, 0)),
        model="box",
        extra={},
    )
    result = realizer.realize(obs)
    assert "box1" in result.sim_path
