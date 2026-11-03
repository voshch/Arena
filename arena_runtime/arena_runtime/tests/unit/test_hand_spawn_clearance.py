"""A hand-spawned robot without a pose is placed clear for its own footprint."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")

SAFE_DIST = 0.25


class _World:
    def __init__(self) -> None:
        from arena_runtime.env.constants.rng import EpisodeRng

        self.clearances: list[float] = []
        self.node = SimpleNamespace(conf=SimpleNamespace(General=SimpleNamespace(RNG=EpisodeRng())))

    def get_positions_on_map(self, n: int, safe_dist: float, level_id: str = "") -> list[object]:
        from arena_runtime.shared import Position

        del n, level_id
        self.clearances.append(safe_dist)
        return [Position(7.0, 8.0, 0.0)]


class _Robots:
    def __init__(self, *fleet: float) -> None:
        self.managers = {f"r{index}": SimpleNamespace(safe_distance=distance) for index, distance in enumerate(fleet)}
        self.pending: dict[str, object] = {}

    def next_name(self, model: str) -> str:
        return f"{model}_0"

    def add_pending(self, name: str, robot: object) -> None:
        self.pending[name] = robot


def _spawn(robots: _Robots, world: _World, pose: object = None, **args: str) -> object:
    from arena_runtime.env.episode_node import EpisodeNode

    node = SimpleNamespace(_robots_manager=robots, _world_manager=world, conf=SimpleNamespace(Robot=SimpleNamespace(SPAWN_ROBOT_SAFE_DIST=SimpleNamespace(value=SAFE_DIST))))
    return robots.pending[EpisodeNode._add_pending_robot(node, "jackal", None, pose, args)]


def _own_clearance() -> float:
    from arena_robots.Robot import RobotIdentifier
    from arena_runtime.env.robot_manager.robot_manager import footprint_radius

    return footprint_radius(RobotIdentifier.parse("jackal").resolve_sync()) + SAFE_DIST


def test_a_robot_bigger_than_the_fleet_gets_its_own_clearance() -> None:
    world = _World()
    robot = _spawn(_Robots(0.01), world)
    assert world.clearances == [pytest.approx(_own_clearance())]
    assert _own_clearance() > 0.01
    assert (robot.pose.position.x, robot.pose.position.y) == (7.0, 8.0)


def test_a_robot_on_an_empty_fleet_gets_its_own_clearance() -> None:
    world = _World()
    _spawn(_Robots(), world)
    assert world.clearances == [pytest.approx(_own_clearance())]


def test_a_robot_smaller_than_the_fleet_gets_the_clearance_of_the_biggest() -> None:
    world = _World()
    _spawn(_Robots(_own_clearance() + 1.0), world)
    assert world.clearances == [pytest.approx(_own_clearance() + 1.0)]


def test_a_given_pose_is_kept_and_samples_nothing() -> None:
    from arena_runtime.shared import Orientation, Pose, Position

    world = _World()
    robot = _spawn(_Robots(), world, pose=Pose(Position(1.0, 2.0, 0.0), Orientation.from_yaw(0.0)))
    assert world.clearances == []
    assert (robot.pose.position.x, robot.pose.position.y) == (1.0, 2.0)
