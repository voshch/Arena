from __future__ import annotations

import math
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _make_robot_manager_stub(pose, goal_tolerance_distance=0.3, goal_tolerance_angle=0.2):
    robot_conf = SimpleNamespace(
        GOAL_TOLERANCE_RADIUS=SimpleNamespace(value=goal_tolerance_distance),
        GOAL_TOLERANCE_ANGLE=SimpleNamespace(value=goal_tolerance_angle),
    )
    return SimpleNamespace(
        pose=pose,
        controls_orientation=True,
        node=SimpleNamespace(conf=SimpleNamespace(Robot=robot_conf)),
    )


@pytest.fixture()
def goal_pose():
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    return Pose(Position(1.0, 2.0), Orientation.from_yaw(0.0))


def test_is_satisfied_pose_none(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase

    phase = GoToPhase(pose=goal_pose)
    stub = _make_robot_manager_stub(pose=None)
    assert phase.is_satisfied(stub) is False


def test_is_satisfied_at_goal(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose)
    current = Pose(Position(1.0, 2.0), Orientation.from_yaw(0.0))
    stub = _make_robot_manager_stub(pose=current)
    assert phase.is_satisfied(stub) is True


def test_is_satisfied_below_distance_threshold(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose, tolerance_radius=0.5)
    close = Pose(Position(1.3, 2.0), Orientation.from_yaw(0.0))
    stub = _make_robot_manager_stub(pose=close)
    assert phase.is_satisfied(stub) is True


def test_is_satisfied_above_distance_threshold(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose, tolerance_radius=0.1)
    far = Pose(Position(2.0, 2.0), Orientation.from_yaw(0.0))
    stub = _make_robot_manager_stub(pose=far)
    assert phase.is_satisfied(stub) is False


def test_is_satisfied_uses_manager_default_tolerance(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose)
    just_within = Pose(Position(1.0 + 0.29, 2.0), Orientation.from_yaw(0.0))
    stub = _make_robot_manager_stub(pose=just_within, goal_tolerance_distance=0.3)
    assert phase.is_satisfied(stub) is True


def test_is_satisfied_per_phase_tolerance_overrides(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose, tolerance_radius=0.05)
    close = Pose(Position(1.1, 2.0), Orientation.from_yaw(0.0))
    stub = _make_robot_manager_stub(pose=close, goal_tolerance_distance=5.0)
    assert phase.is_satisfied(stub) is False


def test_is_satisfied_angle_within_tolerance(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose, tolerance_radius=1.0, tolerance_angle=0.5)
    current = Pose(Position(1.0, 2.0), Orientation.from_yaw(0.3))
    stub = _make_robot_manager_stub(pose=current)
    assert phase.is_satisfied(stub) is True


def test_is_satisfied_angle_above_tolerance(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose, tolerance_radius=1.0, tolerance_angle=0.1)
    current = Pose(Position(1.0, 2.0), Orientation.from_yaw(0.5))
    stub = _make_robot_manager_stub(pose=current)
    assert phase.is_satisfied(stub) is False


def test_is_satisfied_zero_angle_tolerance_skips_angle_check(goal_pose):
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    phase = GoToPhase(pose=goal_pose, tolerance_radius=1.0, tolerance_angle=0.0)
    current = Pose(Position(1.0, 2.0), Orientation.from_yaw(math.pi - 0.01))
    stub = _make_robot_manager_stub(pose=current)
    assert phase.is_satisfied(stub) is True


def test_is_satisfied_angle_wraps_at_pi():
    from task_generator.tasks.robots.request import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    goal = Pose(Position(0.0, 0.0), Orientation.from_yaw(math.pi - 0.05))
    phase = GoToPhase(pose=goal, tolerance_radius=1.0, tolerance_angle=0.2)
    current = Pose(Position(0.0, 0.0), Orientation.from_yaw(-math.pi + 0.05))
    stub = _make_robot_manager_stub(pose=current)
    assert phase.is_satisfied(stub) is True


def test_task_request_kind_empty():
    from task_generator.tasks.robots.request import TaskRequest

    req = TaskRequest(phases=[])
    assert req.kind is None


def test_task_request_kind_single():
    from task_generator.tasks.robots.request import GoToPhase, TaskRequest
    from arena_simulation_setup.utils.geometry import Pose

    phase = GoToPhase(pose=Pose())
    req = TaskRequest(phases=[phase])
    from arena_robots.task_kinds import TaskKind

    assert req.kind is TaskKind.GOTO_POSE


def test_task_request_kind_all_same():
    from task_generator.tasks.robots.request import GoToPhase, TaskRequest
    from arena_simulation_setup.utils.geometry import Pose

    phases = [GoToPhase(pose=Pose()), GoToPhase(pose=Pose())]
    req = TaskRequest(phases=phases)
    from arena_robots.task_kinds import TaskKind

    assert req.kind is TaskKind.GOTO_POSE


def test_task_request_kind_mixed_returns_none():
    from task_generator.tasks.robots.request import GoToPhase, TaskPhase, TaskRequest
    from arena_robots.task_kinds import TaskKind
    from arena_simulation_setup.utils.geometry import Pose
    import attrs

    @attrs.define
    class _OtherPhase(TaskPhase):
        kind = TaskKind.GOTO_POSE

        def is_satisfied(self, robot_manager):
            return False

    phase1 = GoToPhase(pose=Pose())
    req = TaskRequest(phases=[phase1, phase1])
    assert req.kind is TaskKind.GOTO_POSE
