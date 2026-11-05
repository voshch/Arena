"""facing_pose: humansim queues an object from behind its theta, so the theta it gets puts that side at the model's front (+x without one)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("rclpy")

from geometry_msgs.msg import Pose2D

from arena_runtime.humans.arena_humansim.arena_humansim import facing_pose


def _queue_side(pose: Pose2D) -> tuple[float, float]:
    """Where humansim puts the first person in line, one metre out: behind theta, as its line formation does."""
    return (round(pose.x + math.cos(pose.theta + math.pi), 6), round(pose.y + math.sin(pose.theta + math.pi), 6))


@pytest.mark.parametrize(
    ("face", "queue"),
    [("+x", (0.0, 1.0)), ("-x", (0.0, -1.0)), ("+y", (-1.0, 0.0)), ("-y", (1.0, 0.0))],
)
def test_the_queue_forms_at_the_annotated_front(face: str, queue: tuple[float, float]) -> None:
    turned = facing_pose(Pose2D(x=0.0, y=0.0, theta=math.pi / 2), face)
    assert _queue_side(turned) == pytest.approx(queue, abs=1e-6)


@pytest.mark.parametrize("face", [None, "", "up"])
def test_an_object_without_a_known_face_is_used_from_its_plus_x_side(face: str | None) -> None:
    pose = Pose2D(x=1.0, y=2.0, theta=0.3)
    assert _queue_side(facing_pose(pose, face)) == pytest.approx(_queue_side(facing_pose(pose, "+x")), abs=1e-6)
    assert _queue_side(facing_pose(pose, face)) == pytest.approx((1.0 + math.cos(0.3), 2.0 + math.sin(0.3)), abs=1e-6)


def test_the_turned_theta_stays_in_minus_pi_to_pi() -> None:
    for face in ("+x", "-x", "+y", "-y"):
        theta = facing_pose(Pose2D(theta=3.0), face).theta
        assert -math.pi <= theta <= math.pi
