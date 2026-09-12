from __future__ import annotations

import math
import os

import numpy as np
import pytest

from task_generator.simulators.human.pointing import BakedPointAt, PointAtGenerator, PointAtOptions, PointTable, bake
from task_generator.simulators.human.pointing import contract as C
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.table import COLLAR_SCALE, anchor_index, table_path


@pytest.fixture(scope="module")
def gen() -> PointAtGenerator:
    return PointAtGenerator(options=PointAtOptions(collar_scale=COLLAR_SCALE))


@pytest.fixture(scope="module")
def baked() -> BakedPointAt:
    return BakedPointAt()


def aim_error_deg(clip, body: S.Body) -> float:
    ang = clip.frames[clip.peak]["angles"]
    pos, _ = S.fk(ang, body)
    side = clip.side
    achieved = S.unit(pos[f"{side}_wrist"] - pos[f"{side}_elbow"])
    want = S.unit(clip.target_point - pos[f"{side}_wrist"]) if clip.target_point is not None else S.unit(clip.target_dir)
    return math.degrees(math.acos(float(np.clip(achieved @ want, -1.0, 1.0))))


def max_step(clip) -> float:
    m = np.array([[f["angles"][n] for n in C.ROS_JOINT_ORDER] for f in clip.frames])
    return float(np.abs(np.diff(m, axis=0)).max())


def test_table_ships_for_the_default_template():
    path = table_path("point_to_right")
    assert os.path.exists(path), f"bake it: python3 -m task_generator.simulators.human.pointing.table ({path})"
    t = PointTable(path)
    assert t.template == "point_to_right" and t.height == pytest.approx(1.65) and t.collar_scale == COLLAR_SCALE
    assert t.az[0] == -180.0 and t.az[-1] == 180.0
    assert t.ok[:, :, :].mean() > 0.5
    # the wrapped column is the first column
    assert np.array_equal(t.quat[:, -1], t.quat[:, 0])


@pytest.mark.parametrize("az,el", [(-30.0, 0.0), (30.0, 0.0), (-60.0, 30.0), (0.0, -30.0), (-90.0, 20.0), (45.0, -20.0)])
def test_grid_nodes_reproduce_the_solver(gen: PointAtGenerator, baked: BakedPointAt, az: float, el: float):
    ref = gen.point_at(azimuth=az, elevation=el)
    got = baked.point_at(azimuth=az, elevation=el)
    assert got.side == ref.side
    a = anchor_index(gen)
    for n in gen._blended_dofs(ref.side):
        assert got.frames[a]["angles"][n] == pytest.approx(ref.frames[a]["angles"][n], abs=1e-3)
    assert aim_error_deg(got, gen.body) < 0.5
    assert max_step(got) < max_step(ref) + 0.2


@pytest.mark.parametrize("az,el", [(-42.5, 12.5), (17.5, -17.5), (-77.5, 32.5), (62.5, 7.5)])
def test_between_nodes_interpolates_close(baked: BakedPointAt, az: float, el: float):
    clip = baked.point_at(azimuth=az, elevation=el)
    assert aim_error_deg(clip, baked.body) < 4.0
    assert clip.report["ok"]


@pytest.mark.parametrize("target", [(2.0, -1.0, 1.2), (2.0, 1.0, 1.2), (1.5, -0.3, 0.4), (0.5, -2.0, 2.0)])
def test_point_targets_refine_through_the_wrist(baked: BakedPointAt, target: tuple[float, float, float]):
    clip = baked.point_at(target=target)
    assert aim_error_deg(clip, baked.body) < 4.0
    assert not clip.report["near_target_fallback"]


def test_retarget_is_continuous_and_holds(baked: BakedPointAt):
    first = baked.point_at(target=(2.0, -1.0, 1.2))
    hold = baked.hold_pose(first)
    second = baked.retarget(hold, target=(2.0, -0.2, 1.8), transition_s=0.4)
    assert second.side == first.side
    for n in hold.angles:
        assert second.frames[0]["angles"][n] == pytest.approx(hold.angles[n], abs=1e-6)
    assert max_step(second) < 0.6
    assert aim_error_deg(second, baked.body) < 4.0
    assert second.report["transition_frames"] == 8
    with pytest.raises(ValueError, match="hand switch"):
        baked.retarget(hold, target=(2.0, 1.5, 1.2))


def test_wrong_table_is_rejected():
    with pytest.raises(ValueError, match="baked for"):
        BakedPointAt(height=1.80)
    with pytest.raises(ValueError, match="baked for"):
        BakedPointAt(options=PointAtOptions(collar_scale=1.0))


def test_bake_tiny_grid_round_trips(tmp_path):
    path = bake(path=str(tmp_path / "t.npz"), collar_scale=1.0, az_step=90.0, el_step=45.0, el_max=45.0, workers=0)
    t = PointTable(path)
    assert t.linear.shape[:3] == (2, 5, 3) and t.collar_scale == 1.0
    b = BakedPointAt(table=t)
    assert b.opts.collar_scale == 1.0
    raw = PointAtGenerator()
    got = b.point_at(azimuth=-90.0, elevation=0.0)
    ref = raw.point_at(azimuth=-90.0, elevation=0.0)
    a = anchor_index(raw)
    for n in raw._blended_dofs(ref.side):
        assert got.frames[a]["angles"][n] == pytest.approx(ref.frames[a]["angles"][n], abs=1e-3)
