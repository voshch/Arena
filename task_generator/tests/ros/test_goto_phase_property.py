from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


_floats = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_pos_floats = st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False)
_angles = st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False)


def _stub(pose, tol_dist, tol_ang):
    robot_conf = SimpleNamespace(
        GOAL_TOLERANCE_RADIUS=SimpleNamespace(value=tol_dist),
        GOAL_TOLERANCE_ANGLE=SimpleNamespace(value=tol_ang),
    )
    return SimpleNamespace(
        pose=pose,
        controls_orientation=True,
        node=SimpleNamespace(conf=SimpleNamespace(Robot=robot_conf)),
    )


@given(_floats, _floats, _angles, _pos_floats, _pos_floats)
@settings(max_examples=100)
def test_is_satisfied_goal_equals_pose(gx, gy, gyaw, tol_dist, tol_ang):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    goal = Pose(Position(gx, gy), Orientation.from_yaw(gyaw))
    phase = GoToPhase(pose=goal, tolerance_radius=tol_dist, tolerance_angle=tol_ang)
    assert phase.is_satisfied(_stub(goal, tol_dist, tol_ang)) is True


@given(_angles)
@settings(max_examples=100)
def test_dyaw_normalization_within_minus_pi_to_pi(yaw):
    raw = yaw * 2
    dyaw = (raw + math.pi) % (2 * math.pi) - math.pi
    assert -math.pi <= dyaw <= math.pi


@given(_floats, _floats, _angles, _pos_floats)
@settings(max_examples=100)
def test_none_pose_always_false(gx, gy, gyaw, tol_dist):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    goal = Pose(Position(gx, gy), Orientation.from_yaw(gyaw))
    phase = GoToPhase(pose=goal, tolerance_radius=tol_dist, tolerance_angle=0.0)
    assert phase.is_satisfied(_stub(None, tol_dist, 0.0)) is False


@given(_floats, _floats, _angles, _pos_floats)
@settings(max_examples=100)
def test_large_distance_always_fails(gx, gy, gyaw, tol_dist):
    assume(tol_dist < 100.0)
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    goal = Pose(Position(gx, gy), Orientation.from_yaw(gyaw))
    far = Pose(Position(gx + 1000.0, gy + 1000.0), Orientation.from_yaw(gyaw))
    phase = GoToPhase(pose=goal, tolerance_radius=tol_dist, tolerance_angle=0.0)
    assert phase.is_satisfied(_stub(far, tol_dist, 0.0)) is False
