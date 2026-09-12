from __future__ import annotations

import numpy as np
import pytest

from task_generator.simulators.human import layer_quality as LQ
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.contract import ROS_JOINT_ORDER


def _feet(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(T, 4, 3) feet at height z, sliding along x."""
    t = len(z)
    out = np.zeros((t, 4, 3))
    out[..., 0] = x[:, None]
    out[..., 2] = z[:, None]
    return out


def test_standing_still_is_clean() -> None:
    t = 40
    feet = _feet(np.zeros(t), np.zeros(t))
    res = LQ.plausibility(feet, np.zeros((t, 5, 3)))
    assert res["skate_ratio"] == 0.0 and res["penetration_cm"] == 0.0 and res["lowest_foot_cm"] == 0.0
    assert res["jerk_m_s3"] == 0.0


def test_planted_feet_sliding_skate_and_sunk_feet_penetrate() -> None:
    t = 40
    slide = LQ.plausibility(_feet(np.zeros(t), np.arange(t) * 0.05), np.zeros((t, 5, 3)))
    assert slide["skate_ratio"] == pytest.approx(1.0)
    sunk = LQ.plausibility(_feet(np.full(t, -0.03), np.zeros(t)), np.zeros((t, 5, 3)))
    assert sunk["penetration_cm"] == pytest.approx(3.0) and sunk["penetration_max_cm"] == pytest.approx(3.0)
    lifted = LQ.plausibility(_feet(np.full(t, 0.2), np.arange(t) * 0.05), np.zeros((t, 5, 3)))
    assert np.isnan(lifted["skate_ratio"])  # no planted frame: sliding in the air is not skating
    assert lifted["lowest_foot_cm"] == pytest.approx(20.0)


def test_h3d_axes_to_z_up() -> None:
    p = np.array([[[1.0, 2.0, 3.0]]])  # x left, y up, z forward
    assert LQ.h3d_z_up(p).tolist() == [[[3.0, 1.0, 2.0]]]


def _frames(values: list[float], joint: str = "r_elbow") -> list[dict]:
    return [{"angles": {**dict.fromkeys(ROS_JOINT_ORDER, 0.0), joint: v}} for v in values]


def test_loop_seam_flags_a_wrap_jump() -> None:
    body = S.Body(1.65)
    smooth = LQ.loop_seam(_frames(list(np.sin(np.linspace(0, 2 * np.pi, 41))[:-1] * 0.5)), body, None)
    assert smooth["passes_gate"] and smooth["seam_over_p95"] < 1.5
    ramp = LQ.loop_seam(_frames(list(np.linspace(0.0, 2.4, 40))), body, None)
    assert not ramp["passes_gate"] and ramp["seam_link"] == "r_wrist"
    assert ramp["seam_joint_rad"] == pytest.approx(2.4)


def test_masked_joints_are_the_only_ones_played() -> None:
    body = S.Body(1.65)
    frames = [{"angles": {**dict.fromkeys(ROS_JOINT_ORDER, 0.0), "l_knee": -1.0, "r_elbow": 1.0}}]
    masked = LQ.rig_positions(frames, body, joints=("r_elbow",))
    rest = LQ.rig_positions(_frames([0.0]), body)
    assert np.allclose(masked["l_ankle"], rest["l_ankle"])  # the knee is outside the mask
    assert not np.allclose(masked["r_wrist"], rest["r_wrist"])


def test_point_hold_is_on_target() -> None:
    res = LQ.aim_errors("arm", (3.0, 1.0, 1.2), moving=False)
    assert res is not None and res["median_deg"] < 5.0


def test_halt_is_on_its_bearing_from_the_body() -> None:
    near = (1.5, 0.0, 1.2)
    body = LQ.aim_errors("halt", near, moving=False, origin="body")
    wrist = LQ.aim_errors("halt", near, moving=False)
    assert body is not None and wrist is not None
    assert body["median_deg"] < 2.0 < wrist["median_deg"]  # the hand sits off the body axis: parallax at close range


def test_reverse_loop_has_no_seam() -> None:
    body = S.Body(1.65)
    ramp = _frames(list(np.linspace(0.0, 2.4, 40)))
    assert not LQ.loop_seam(ramp, body, None)["passes_gate"]
    out_and_back = LQ.loop_seam(ramp, body, None, reverse=True)
    assert out_and_back["passes_gate"]  # the wrap lands on frame 1, one ordinary step away from frame 0
    assert out_and_back["seam_over_p95"] <= 1.5
