from __future__ import annotations

import numpy as np
import pytest

from task_generator.simulators.human.pointing import (
    HoldPose,
    PointAtGenerator,
    PointAtOptions,
    angles_from_direction,
    direction_from_angles,
    load_template,
)
from task_generator.simulators.human.pointing import contract as C
from task_generator.simulators.human.pointing import skeleton as S


def aim_of(clip, body, i):
    pos, _ = S.fk(clip.frames[i]["angles"], body)
    return S.unit(pos[f"{clip.side}_wrist"] - pos[f"{clip.side}_elbow"])


def joint_matrix(clip, names=C.ROS_JOINT_ORDER):
    return np.array([[fr["angles"][n] for n in names] for fr in clip.frames])


@pytest.fixture(scope="module")
def gen() -> PointAtGenerator:
    return PointAtGenerator()


# ---------------------------------------------------------------------------
# port of pointing/test_pointing.py


def test_arm_rot_inverse_round_trips():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(500):
        y, p, r = rng.uniform(-np.pi, np.pi, 3)
        m = S.arm_rot(y, p, r)
        worst = max(worst, float(np.abs(m - S.arm_rot(*S.arm_rot_inverse(m))).max()))
    assert worst < 1e-9


def test_slerp():
    a, b = S.rot_axis([0, 0, 1], 0.3), S.rot_axis([1, 0, 0], 1.1)
    half = S.slerp_rot(a, b, 0.5)
    ang = lambda m: float(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))  # noqa: E731
    assert abs(ang(a.T @ half) - ang(half.T @ b)) < 1e-9
    assert np.allclose(S.slerp_rot(a, b, 0.0), a) and np.allclose(S.slerp_rot(a, b, 1.0), b)


def test_rest_pose_geometry():
    body = S.Body(1.65)
    rest = dict.fromkeys(C.ROS_JOINT_ORDER, 0.0)
    pos, _ = S.fk(rest, body)
    assert abs(pos["l_ankle"][2]) < 1e-9
    assert abs(pos["l_wrist"][2] - (pos["l_shoulder"][2] - body.arm_reach)) < 1e-9
    assert abs(pos["l_shoulder"][1] + pos["r_shoulder"][1]) < 1e-12


def test_direction_angle_round_trip():
    d = direction_from_angles(37.0, -21.0)
    az, el = angles_from_direction(d)
    assert abs(az - 37) < 1e-9 and abs(el + 21) < 1e-9


def test_segment_distance_vs_brute_force():
    rng = np.random.default_rng(3)
    err = 0.0
    for _ in range(200):
        p0, p1, q0, q1 = rng.normal(size=(4, 3))
        ts = np.linspace(0, 1, 400)
        brute = min(
            np.linalg.norm((p0 + t * (p1 - p0)) - (q0 + u * (q1 - q0)))
            for t in ts for u in ts[::8]
        )
        err = max(err, S._segment_distance(p0, p1, q0, q1) - brute)
    assert err < 1e-6


@pytest.mark.parametrize("name", ["point_to_right"])
def test_template(name):
    t = load_template(name)
    assert t.w[0] == 0.0 and t.w.max() == 1.0
    assert (t.w >= 0).all() and (t.w <= 1).all()
    assert t.hold[0] <= t.settled[0] < t.settled[1] <= t.hold[1]
    assert t.side in ("l", "r")
    assert all(
        abs(t.base[i][d] + t.w[i] * (t.frames[t.peak]["angles"][d] - t.frames[0]["angles"][d]) - t.frames[i]["angles"][d]) < 1e-12
        for i in (0, t.peak, len(t.frames) - 1)
        for d in C.arm_dofs(t.side)
    )


def test_identity(gen):
    t = gen.template
    clip = gen.point_at(azimuth=np.degrees(gen.tpl_az), elevation=np.degrees(gen.tpl_el))
    worst = 0.0
    for i in range(len(clip.frames)):
        a, _ = S.fk(clip.frames[i]["angles"], gen.body)
        b, _ = S.fk(t.frames[i]["angles"], gen.body)
        worst = max(worst, max(np.linalg.norm(a[k] - b[k]) for k in a))
    assert worst < 0.05
    assert len(clip.frames) == len(t.frames)


def test_aim_over_cone(gen):
    worst = 0.0
    for el in range(-90, 91, 30):
        for az in range(-90, 91, 30):
            worst = max(worst, gen.point_at(azimuth=az, elevation=el).report["aim_error_deg"])
    assert worst <= 1.0


def test_aim_point_target(gen):
    clip = gen.point_at(target=(3.0, 1.2, 2.0))
    pos, _ = S.fk(clip.frames[clip.peak]["angles"], gen.body)
    want = S.unit(np.array([3.0, 1.2, 2.0]) - pos[f"{clip.side}_wrist"])
    err = np.degrees(np.arccos(np.clip(float(aim_of(clip, gen.body, clip.peak) @ want), -1, 1)))
    assert err < 1.0


def test_wire_contract(gen):
    clip = gen.point_at(azimuth=-55, elevation=35)
    f = clip.frames[0]
    assert list(f["angles"]) == C.ROS_JOINT_ORDER
    assert set(f) == {"angles", "root_xy_yaw", "animation_state", "t"}
    assert all(isinstance(v, float) for v in f["angles"].values())
    assert all(np.isfinite(v) for fr in clip.frames for v in fr["angles"].values())
    assert np.allclose(np.diff([fr["t"] for fr in clip.frames]), gen.template.dt)
    written = set(gen._blended_dofs(clip.side))
    assert all(
        C.saturation(n, v, limits=C.EMIT_LIMITS) == 0.0
        for fr in clip.frames for n, v in fr["angles"].items() if n in written
    )
    over = {n for fr in clip.frames for n, v in fr["angles"].items() if n in written and C.saturation(n, v) > 0.0}
    assert over <= {f"{s}_r_shoulder" for s in "lr"}
    assert any(C.saturation(n, v) > 0.0 for fr in clip.frames for n, v in fr["angles"].items() if n not in written)
    assert all(fr["angles"]["l_knee"] == gen.template.frames[i]["angles"]["l_knee"] for i, fr in enumerate(clip.frames))
    arr = clip.to_array()
    assert arr.dtype == object and arr.shape == (len(clip.frames),)


@pytest.mark.parametrize("az,el", [(0, 90), (-85, -70), (60, 40), (0, 0), (-40, 25)])
def test_continuity(gen, az, el):
    spherical = {f"{s}_{d}_shoulder" for s in "lr" for d in "ypr"}
    others = [n for n in C.ROS_JOINT_ORDER if n not in spherical]
    clip = gen.point_at(azimuth=az, elevation=el)
    m = joint_matrix(clip, others)
    assert float(np.abs(np.diff(m, axis=0)).max()) < 0.35
    aims = np.array([aim_of(clip, gen.body, i) for i in range(len(clip.frames))])
    step = np.degrees(np.arccos(np.clip((aims[1:] * aims[:-1]).sum(1), -1, 1))).max()
    assert step < 50.0
    pos = [S.fk(fr["angles"], gen.body)[0][f"{clip.side}_wrist"] for fr in clip.frames]
    hop = float(max(np.linalg.norm(b - a) for a, b in zip(pos, pos[1:], strict=False)))
    assert hop < 0.35


def test_unwrap_option(gen):
    tight = gen.point_at(azimuth=70, elevation=-60)
    loose = gen.point_at(azimuth=70, elevation=-60, unwrap_shoulder=True)
    assert tight.report["max_joint_step_rad"] > 5.5
    assert loose.report["max_joint_step_rad"] < 3.5
    for c in (tight, loose):
        assert c.report["aim_error_deg"] < 1.0
    a = [S.fk(f["angles"], gen.body)[0]["l_wrist"] for f in tight.frames]
    b = [S.fk(f["angles"], gen.body)[0]["l_wrist"] for f in loose.frames]
    assert max(np.linalg.norm(x - y) for x, y in zip(a, b, strict=True)) < 0.02


@pytest.mark.parametrize("az,el", [(60, 20), (75, -30), (45, 55)])
def test_symmetry(gen, az, el):
    a = gen.point_at(azimuth=az, elevation=el, hand="l").report
    b = gen.point_at(azimuth=-az, elevation=el, hand="r").report
    assert abs(a["clearance_m"] - b["clearance_m"]) < 0.02


def test_hand_choice(gen):
    assert gen.point_at(azimuth=60, elevation=0).side == "l"
    assert gen.point_at(azimuth=-60, elevation=0).side == "r"
    assert gen.point_at(azimuth=0, elevation=0).side == "r"
    assert gen.point_at(azimuth=170, elevation=90).side == gen.point_at(azimuth=-170, elevation=90).side
    assert gen.point_at(azimuth=60, elevation=0, hand="r").side == "r"


@pytest.mark.parametrize("kw", [{"target": (1.0, np.nan, 1.0)}, {"target": (np.inf, 0.0, 1.0)}, {}])
def test_degenerate_rejected(gen, kw):
    with pytest.raises(ValueError):
        gen.point_at(**kw)


def test_degenerate_fallbacks(gen):
    near = gen.point_at(target=(0.05, -0.10, 1.40))
    assert near.report["near_target_fallback"] and near.report["aim_error_deg"] < 1.0
    far = gen.point_at(target=(1e6, 2e5, 3e5))
    assert far.report["aim_error_deg"] < 1.0


@pytest.mark.parametrize("h", [1.20, 1.65, 2.10])
def test_heights(h):
    rep = PointAtGenerator(height=h).point_at(target=(2.5, -1.0, 1.3)).report
    assert rep["aim_error_deg"] < 1.0


def test_hold_stretch(gen):
    base = len(gen.point_at(azimuth=-30, elevation=20).frames)
    long = gen.point_at(azimuth=-30, elevation=20, hold_s=5.0)
    assert len(long.frames) > base + 60
    assert long.report["aim_error_deg"] < 1.0
    spherical = {"l_y_shoulder", "r_y_shoulder", "l_r_shoulder", "r_r_shoulder"}
    keep = [n for n in C.ROS_JOINT_ORDER if n not in spherical]
    assert float(np.abs(np.diff(joint_matrix(long, keep), axis=0)).max()) < 0.35


def test_head_passthrough(gen):
    tpl = gen.template
    worst = 0.0
    for az, el in ((0, 0), (-60, 0), (60, 30), (0, -70), (0, 85)):
        clip = gen.point_at(azimuth=az, elevation=el)
        for f, t in zip(clip.frames, tpl.frames, strict=True):
            for k in ("r_head", "y_head", "p_head"):
                worst = max(worst, abs(f["angles"][k] - t["angles"][k]))
    assert worst < 1e-12


# ---------------------------------------------------------------------------
# gesture chaining


def test_hold_range(gen):
    clip = gen.point_at(azimuth=-40, elevation=25)
    a, b = clip.hold_range
    assert 0 < a <= clip.peak <= b < len(clip.frames)
    assert (clip.envelope[a:b + 1] >= 0.999).all()
    assert clip.envelope[a - 1] < 0.999 and clip.envelope[b + 1] < 0.999
    stretched = gen.point_at(azimuth=-40, elevation=25, hold_s=3.0)
    a2, b2 = stretched.hold_range
    assert b2 - a2 > b - a


def test_hold_pose_round_trip(gen):
    clip = gen.point_at(target=(2.5, -1.0, 1.6))
    hp = gen.hold_pose(clip)
    assert isinstance(hp, HoldPose)
    assert hp.side == clip.side
    assert set(hp.angles) == set(gen._blended_dofs(clip.side))
    peak = clip.frames[clip.peak]["angles"]
    assert all(hp.angles[n] == peak[n] for n in hp.angles)
    assert hp.swivel == clip.report["swivel_rad"] and np.isfinite(hp.swivel)
    assert np.allclose(hp.target_dir, clip.target_dir)
    assert np.allclose(hp.target_point, (2.5, -1.0, 1.6))


def test_torso_yaw_assist_caps_near_15_deg(gen):
    from task_generator.simulators.human.pointing.generator import _torso_assist

    yaw, _ = _torso_assist(np.radians(60.0), 0.0, PointAtOptions())
    assert abs(np.degrees(yaw) - 15.0) < 0.5
    clip = gen.point_at(azimuth=60, elevation=0)
    i = gen.template.peak
    peak = clip.frames[i]["angles"]
    src = gen._source_pose(i, clip.side)
    delta = sum(peak[f"y_{s}"] - src[f"y_{s}"] for s in C.SPINE_SEGMENTS)
    d_yaw, _ = gen._assist_deltas(np.radians(60.0), 0.0, PointAtOptions())
    assert delta == pytest.approx(d_yaw, abs=1e-6)
    assert abs(np.degrees(d_yaw) - (15.0 - np.degrees(0.25 * gen.tpl_az * np.cos(gen.tpl_el)))) < 0.5


def test_aim_of_matches_point_at(gen):
    t = np.array([2.5, -1.0, 1.6])
    assert np.allclose(gen.aim_of(t), gen.point_at(target=t).target_dir)
