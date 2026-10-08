"""Driven joints: rig.yaml drivers and the wheel angles integrated from a ped's poses."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from task_generator.simulators.human.drivers import DrivenJoints, Roller, load_drivers, parse_drivers, with_driven
from task_generator.simulators.human.possession import bare_joint_names_valid

_RIG = """
skeleton: cmu
drivers:
  l_wheel: {bone: WheelL, from: distance, radius: 0.3, lateral: 0.28, axis: [0.0, 1.0, 0.0]}
  r_wheel: {bone: WheelR, from: distance, radius: 0.3, lateral: -0.28, axis: [0.0, 1.0, 0.0]}
"""
_CHAIR = (Roller("l_wheel", 0.3, 0.28), Roller("r_wheel", 0.3, -0.28))


def _bundle(root: Path, rig: str | None) -> str:
    bundle = root / "chair"
    bundle.mkdir(parents=True)
    sdf = bundle / "chair.sdf"
    sdf.write_text("<sdf/>")
    if rig is not None:
        (bundle / "rig.yaml").write_text(rig)
    return str(sdf)


def test_drivers_load_from_the_rig_file_beside_the_actor_sdf(tmp_path: Path) -> None:
    assert load_drivers(_bundle(tmp_path, _RIG)) == _CHAIR


def test_models_without_a_rig_file_or_a_drivers_section_have_no_drivers(tmp_path: Path) -> None:
    assert load_drivers("") == ()
    assert load_drivers(_bundle(tmp_path / "a", None)) == ()
    assert load_drivers(_bundle(tmp_path / "b", "skeleton: cmu\n")) == ()


@pytest.mark.parametrize(
    ("rig", "message"),
    [
        ("- a\n", "a rig file is a mapping"),
        ("drivers: [l_wheel]\n", "drivers takes"),
        ("drivers: {l_wheel: {from: heading, radius: 0.3}}\n", "driver 'l_wheel' takes"),
        ("drivers: {l_wheel: {from: distance}}\n", "needs a radius above 0 m"),
        ("drivers: {l_wheel: {from: distance, radius: 0.0}}\n", "needs a radius above 0 m"),
    ],
)
def test_malformed_drivers_are_rejected_with_the_expected_form(rig: str, message: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=message):
        load_drivers(_bundle(tmp_path, rig))


def test_a_driver_without_lateral_sits_on_the_centerline() -> None:
    assert parse_drivers({"drivers": {"wheel": {"from": "distance", "radius": 0.1}}}, "rig") == (Roller("wheel", 0.1, 0.0),)


def test_straight_travel_turns_both_wheels_by_distance_over_radius() -> None:
    driven = DrivenJoints()
    assert driven.advance(7, _CHAIR, 1.0, 2.0, 0.5) == {"l_wheel": 0.0, "r_wheel": 0.0}
    angles: dict[str, float] = {}
    for step in range(1, 101):
        angles = driven.advance(7, _CHAIR, 1.0 + 0.05 * step * math.cos(0.5), 2.0 + 0.05 * step * math.sin(0.5), 0.5)
    assert angles["l_wheel"] == pytest.approx(5.0 / 0.3)
    assert angles["r_wheel"] == pytest.approx(5.0 / 0.3)
    assert angles["l_wheel"] > math.tau


def test_backing_up_turns_the_wheels_backward() -> None:
    driven = DrivenJoints()
    driven.advance(7, _CHAIR, 0.0, 0.0, math.pi / 2.0)
    angles = driven.advance(7, _CHAIR, 0.0, -0.3, math.pi / 2.0)
    assert angles == pytest.approx({"l_wheel": -1.0, "r_wheel": -1.0})


def test_a_pivot_in_place_counter_rotates_the_wheels() -> None:
    driven = DrivenJoints()
    driven.advance(7, _CHAIR, 0.0, 0.0, 3.0)
    angles = driven.advance(7, _CHAIR, 0.0, 0.0, -3.0)
    turn = math.tau - 6.0
    assert angles["l_wheel"] == pytest.approx(-turn * 0.28 / 0.3)
    assert angles["r_wheel"] == pytest.approx(turn * 0.28 / 0.3)


def test_a_left_arc_rolls_each_wheel_over_its_own_path_length() -> None:
    driven = DrivenJoints()
    radius = 2.0
    angles: dict[str, float] = {}
    steps = 400
    for step in range(steps + 1):
        yaw = (math.pi / 2.0) * step / steps
        angles = driven.advance(7, _CHAIR, radius * math.sin(yaw), radius * (1.0 - math.cos(yaw)), yaw)
    assert angles["l_wheel"] == pytest.approx((radius - 0.28) * (math.pi / 2.0) / 0.3, rel=1e-4)
    assert angles["r_wheel"] == pytest.approx((radius + 0.28) * (math.pi / 2.0) / 0.3, rel=1e-4)


def test_sideways_displacement_does_not_roll_the_wheels() -> None:
    driven = DrivenJoints()
    driven.advance(7, _CHAIR, 0.0, 0.0, 0.0)
    assert driven.advance(7, _CHAIR, 0.0, 0.4, 0.0) == pytest.approx({"l_wheel": 0.0, "r_wheel": 0.0})


def test_peds_integrate_independently_and_restart_after_forget() -> None:
    driven = DrivenJoints()
    driven.advance(1, _CHAIR, 0.0, 0.0, 0.0)
    driven.advance(2, _CHAIR, 0.0, 0.0, 0.0)
    assert driven.advance(1, _CHAIR, 0.6, 0.0, 0.0)["l_wheel"] == pytest.approx(2.0)
    assert driven.advance(2, _CHAIR, 0.3, 0.0, 0.0)["l_wheel"] == pytest.approx(1.0)
    assert driven.known() == {1, 2}
    driven.forget(1)
    assert driven.known() == {2}
    assert driven.advance(1, _CHAIR, 5.0, 5.0, 1.0) == {"l_wheel": 0.0, "r_wheel": 0.0}


def test_driven_angles_append_to_the_gait_joints_and_replace_their_own_earlier_values() -> None:
    names, positions = with_driven(["waist", "l_knee"], [0.1, -0.4], {"l_wheel": 3.0, "r_wheel": 4.0})
    assert (names, positions) == (["waist", "l_knee", "l_wheel", "r_wheel"], [0.1, -0.4, 3.0, 4.0])
    names, positions = with_driven(names, positions, {"l_wheel": 3.5, "r_wheel": 4.5})
    assert (names, positions) == (["waist", "l_knee", "l_wheel", "r_wheel"], [0.1, -0.4, 3.5, 4.5])
    assert with_driven([], [], {"l_wheel": 1.0}) == (["l_wheel"], [1.0])


def test_driven_joints_stay_outside_the_possession_claim_set() -> None:
    assert not bare_joint_names_valid(["l_wheel"])
