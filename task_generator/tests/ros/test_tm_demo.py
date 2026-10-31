from __future__ import annotations

import math

import pytest


def _origin() -> object:
    from task_generator.shared import Position

    return Position(0.0, 0.0, 0.0)


def _at(x: float, y: float) -> object:
    from task_generator.shared import Position

    return Position(x, y, 0.0)


@pytest.mark.parametrize(("cx", "cy"), [(0.0, 0.0), (5.0, 3.0)])
def test_vertices_stand_on_the_circle(cx: float, cy: float):
    from task_generator.tasks.robots.demo.impl import _Orientation, _vertices

    radius = 1.5
    poses = _vertices(_at(cx, cy), 4, radius, _Orientation.RADIAL_IN)
    assert len(poses) == 4
    for pose in poses:
        assert abs(math.hypot(pose.position.x - cx, pose.position.y - cy) - radius) < 1e-9


@pytest.mark.parametrize(("orientation", "sign"), [("RADIAL_IN", 1.0), ("RADIAL_OUT", -1.0)])
def test_vertices_radial_orientation_faces_the_center_or_away(orientation: str, sign: float):
    from task_generator.tasks.robots.demo.impl import _Orientation, _vertices

    cx, cy = 5.0, 3.0
    for pose in _vertices(_at(cx, cy), 4, 1.5, _Orientation[orientation]):
        x, y = pose.position.x, pose.position.y
        expected_yaw = math.atan2(sign * (cy - y), sign * (cx - x))
        diff = (pose.orientation.to_yaw() - expected_yaw + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-9


def test_vertices_tangent_points_to_next():
    from task_generator.tasks.robots.demo.impl import _Orientation, _vertices

    n = 3
    poses = _vertices(_at(2.0, 2.0), n, 2.0, _Orientation.TANGENT)
    for i, pose in enumerate(poses):
        nx = poses[(i + 1) % n].position.x
        ny = poses[(i + 1) % n].position.y
        expected_yaw = math.atan2(ny - pose.position.y, nx - pose.position.x)
        diff = (pose.orientation.to_yaw() - expected_yaw + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-9


@pytest.mark.parametrize(
    ("n", "radius", "orientation", "match"),
    [
        (2, 1.0, "TANGENT", "VERTICES"),
        (4, -1.0, "TANGENT", "RADIUS"),
        (4, 0.0, "TANGENT", "RADIUS"),
        (4, 1.0, None, "ORIENTATION"),
    ],
)
def test_vertices_refuse_bad_arguments(n: int, radius: float, orientation: str | None, match: str):
    from task_generator.tasks.robots.demo.impl import _Orientation, _vertices

    with pytest.raises(ValueError, match=match):
        _vertices(_origin(), n, radius, _Orientation[orientation] if orientation else "spiral")  # type: ignore[arg-type]


@pytest.mark.parametrize(("name", "picked"), [("<random>", None), ("", ""), ("wave", "wave")])
def test_pick_gesture(name: str, picked: str | None):
    from task_generator.tasks.robots.demo.impl import _pick_gesture

    assert _pick_gesture(name) == picked


@pytest.mark.parametrize("name", ["", "random", "<random>"])
def test_parse_gesture_canonicalizes_random(name: str):
    from task_generator.tasks.robots.demo.impl import _parse_gesture

    assert _parse_gesture(name) == "<random>"


def test_parse_gesture_wave_returns_wave():
    from arena_simulation_setup import ASS_DIR

    gestures_dir = ASS_DIR / "configs" / "gestures"
    if not gestures_dir.exists():
        pytest.skip("gestures config dir not present in test environment")

    from task_generator.tasks.robots.demo.impl import _parse_gesture

    assert _parse_gesture("wave") == "wave"


def test_parse_gesture_unknown_raises():
    from arena_simulation_setup import ASS_DIR

    gestures_dir = ASS_DIR / "configs" / "gestures"
    if not gestures_dir.exists():
        pytest.skip("gestures config dir not present in test environment")

    from task_generator.tasks.robots.demo.impl import _parse_gesture

    with pytest.raises(ValueError, match="unknown gesture"):
        _parse_gesture("nonexistent_gesture_xyz")
