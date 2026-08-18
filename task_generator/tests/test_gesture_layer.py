from __future__ import annotations

import concurrent.futures
import math
from pathlib import Path

import numpy as np
import pytest

from task_generator.simulators.human.animation_mananager import AnimationManager
from task_generator.simulators.human.gestures import (
    BREATH_AMP_RAD,
    BREATH_HZ,
    COMPANION_RELEASE_LAG_S,
    FADE_S,
    GAZE_LEAD_S,
    GESTURES,
    HOLD_MIN_S,
    MOVE_MAX_S,
    MOVE_MIN_S,
    OMEGA_RAD_S,
    GestureClip,
    GestureLayer,
    GestureRequest,
    RETARGET_EPS_M,
    move_time,
    resample,
    world_to_local,
)
from task_generator.simulators.human.gestures.look import HEAD, LookGesture, LookHold
from task_generator.simulators.human.gestures.point import LIGHT_RETARGET_RAD, PointGesture
from task_generator.simulators.human.pointing.contract import LIMITS, arm_dofs

ANIMATIONS = Path(__file__).resolve().parents[1] / "task_generator" / "simulators" / "human" / "animations"
DT = 0.05
POSE = (1.0, 2.0, math.pi / 2)
LEAD_TICKS = int(round(GAZE_LEAD_S / DT))


class StubLogger:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(msg)

    def warning(self, msg: str) -> None:
        self.lines.append(msg)


@pytest.fixture()
def rig() -> tuple[AnimationManager, GestureLayer, StubLogger]:
    log = StubLogger()
    mgr = AnimationManager(ANIMATIONS, logger=log, fps=20.0)
    layer = GestureLayer(mgr, log, sync=True)
    mgr.gesture_hook = layer
    return mgr, layer, log


def req(kind: str, at: tuple[float, float, float], moving: bool = False, opts: dict | None = None) -> GestureRequest:
    return GestureRequest(kind=kind, at=at, pose=POSE, moving=moving, opts=opts or {})


def local_req(kind: str, local: tuple[float, float, float], moving: bool = False) -> GestureRequest:
    """Request whose target is given in the ped frame of POSE."""
    c, s = math.cos(POSE[2]), math.sin(POSE[2])
    return req(kind, (POSE[0] + c * local[0] - s * local[1], POSE[1] + s * local[0] + c * local[1], local[2]), moving=moving)


class StubExecutor:
    """Holds submitted jobs until the test lands them."""

    def __init__(self):
        self.pending: list[tuple[concurrent.futures.Future, object]] = []
        self.submitted = 0

    def submit(self, fn):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.pending.append((fut, fn))
        self.submitted += 1
        return fut

    def land(self, n: int | None = None) -> None:
        batch, self.pending = self.pending[:n], self.pending[n:] if n is not None else []
        for fut, fn in batch:
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except Exception as e:  # noqa: BLE001 - mirrors the pool, the layer classifies it
                fut.set_exception(e)


@pytest.fixture()
def async_rig(rig) -> tuple[AnimationManager, GestureLayer, StubLogger, StubExecutor]:
    mgr, layer, log = rig
    layer.sync = False
    layer._executor = StubExecutor()
    return mgr, layer, log, layer._executor


def tick(mgr: AnimationManager, agent: int, request: GestureRequest | None, n: int = 1) -> dict[str, float]:
    out = {}
    for _ in range(n):
        out = mgr.compute(agent, 0, 0.0, DT, gesture=request)
    return out


def arm(layer: GestureLayer, agent: int):
    return layer._agents[agent].slots["arm"]


def head(layer: GestureLayer, agent: int):
    return layer._agents[agent].slots["head"]


def arm_moved(angles: dict[str, float], side: str) -> bool:
    return any(abs(angles[j]) > 1e-3 for j in arm_dofs(side))


def to_hold(mgr: AnimationManager, layer: GestureLayer, agent: int, r: GestureRequest):
    """Start ``r`` and tick until the arm slot holds."""
    tick(mgr, agent, r, n=LEAD_TICKS + 1)
    st = arm(layer, agent)
    assert st.clip is not None
    tick(mgr, agent, r, n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    assert st.phase == "hold"
    return st


def test_world_to_local_rotates_into_ped_frame():
    local = world_to_local((1.0, 5.0, 1.2), POSE)
    assert np.allclose(local, [3.0, 0.0, 1.2], atol=1e-9)


def test_unknown_kind_warns_once_and_leaves_gait(rig):
    mgr, layer, log = rig
    tick(mgr, 1, req("shrug", (0.0, 4.0, 1.0)), n=3)
    assert sum("not registered" in line for line in log.lines) == 1
    assert 1 not in mgr._ped_blend


def test_point_ramps_holds_and_releases(rig):
    mgr, layer, log = rig
    r = req("point", (1.0, 5.0, 1.2))
    tick(mgr, 1, r)
    assert set(mgr._ped_blend[1]) == {"head"}  # gaze leads
    assert arm(layer, 1).phase == "requested"
    tick(mgr, 1, r, n=LEAD_TICKS)
    assert set(mgr._ped_blend[1]) == {"head", "arm"}
    st = arm(layer, 1)
    assert st.phase == "ramp"
    tick(mgr, 1, r, n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    assert st.phase == "hold"
    a = tick(mgr, 1, r)
    assert arm_moved(a, st.clip.side)
    tick(mgr, 1, r, n=int(HOLD_MIN_S / DT) + 1)
    tick(mgr, 1, None)
    assert st.phase == "release"
    tick(mgr, 1, None, n=len(st.clip.frames) + int((COMPANION_RELEASE_LAG_S + FADE_S) / DT) + 12)
    assert 1 not in layer._agents
    assert 1 not in mgr._ped_blend


def test_clear_before_min_hold_is_deferred(rig):
    mgr, layer, _ = rig
    r = req("point", (1.0, 5.0, 1.2))
    tick(mgr, 1, r, n=LEAD_TICKS + 1)
    st = arm(layer, 1)
    tick(mgr, 1, None)
    assert st.phase != "release"
    assert st.release_pending
    tick(mgr, 1, None, n=int((st.clip.hold_start / st.clip.fps + HOLD_MIN_S) / DT) + 2)
    assert st.phase == "release"


def test_retarget_same_hand_transitions_without_release(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, req("point", (-1.0, 4.0, 1.2)))
    first = st.clip
    tick(mgr, 1, req("point", (-1.5, 5.0, 0.6)))
    assert st.clip is not first
    assert st.phase == "transition"
    assert st.clip.side == first.side
    held = first.hold.angles
    start = st.clip.frames[0]["angles"]
    assert max(abs(start[j] - held[j]) for j in held) < 1e-6


def test_small_target_jitter_does_not_retarget(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, req("point", (-1.0, 4.0, 1.2)))
    clip = st.clip
    tick(mgr, 1, req("point", (-1.05, 4.05, 1.2)))
    assert st.clip is clip


def test_hand_switch_releases_then_restarts(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, req("point", (-1.0, 4.0, 1.2)))
    side = st.clip.side
    other = req("point", (3.0, 4.0, 1.2))
    tick(mgr, 1, other)
    assert st.phase == "release"
    tick(mgr, 1, other, n=len(st.clip.frames) + int((COMPANION_RELEASE_LAG_S + FADE_S) / DT) + 12 + LEAD_TICKS)
    new = arm(layer, 1)
    assert new is not st
    assert new.clip.side != side


def test_moving_ped_blends_arm_only(rig):
    mgr, layer, _ = rig
    tick(mgr, 1, req("point", (1.0, 5.0, 1.2), moving=True), n=LEAD_TICKS + 1)
    joints = mgr._ped_blend[1]["arm"].joints
    assert "waist" not in joints
    assert set(arm_dofs(arm(layer, 1).clip.side)) <= joints


def test_forget_drops_state_and_transients(rig):
    mgr, layer, _ = rig
    tick(mgr, 1, req("point", (1.0, 5.0, 1.2)), n=LEAD_TICKS + 1)
    assert any(name.startswith("gesture:point:1") for name in mgr.animations)
    assert any(name.startswith("gesture:look:1") for name in mgr.animations)
    layer.forget(1)
    mgr.forget(1)
    assert 1 not in layer._agents
    assert not any(name.startswith("gesture:") for name in mgr.animations)


def test_release_after_retarget_eases_from_the_new_hold_to_rest(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, req("point", (-0.3, 4.0, 0.3)))  # front, low
    first_release = st.release_frames
    high = req("point", (-2.5, 3.0, 2.2))  # side, high
    tick(mgr, 1, high)
    assert st.phase == "transition"
    assert st.release_frames is not first_release
    tick(mgr, 1, high, n=int(HOLD_MIN_S / DT) + int(st.clip.hold_start / st.clip.fps / DT) + 2)
    tick(mgr, 1, None)
    assert st.phase == "release"
    tail = mgr._ped_blend[1]["arm"].anim
    assert tail.n_frames > 5
    park = st.clip.frames[st.clip.hold_end]["angles"]
    first = tail.frames[0]["angles"]
    assert max(abs(first[j] - park[j]) for j in st.joints) < 0.1
    assert max(abs(v) for v in tail.frames[-1]["angles"].values()) < 1e-9
    side = st.clip.side
    for j in (f"{side}_p_shoulder", f"{side}_y_shoulder"):
        vals = np.array([f["angles"][j] for f in st.release_frames])
        assert vals[0] == park[j]
        assert np.all(np.diff(np.abs(vals)) <= 1e-9)
    # a hold that never retargeted keeps the recorded lowering arc
    st2 = to_hold(mgr, layer, 2, req("point", (-0.3, 4.0, 0.3)))
    tick(mgr, 2, req("point", (-0.3, 4.0, 0.3)), n=int(HOLD_MIN_S / DT) + 2)
    tick(mgr, 2, None)
    assert st2.phase == "release"
    assert st2.release_frames is st2.clip.release


def test_moving_flip_crossfades_spine_joints(rig):
    mgr, layer, _ = rig
    still, walking = req("point", (1.0, 5.0, 1.2)), req("point", (1.0, 5.0, 1.2), moving=True)
    st = to_hold(mgr, layer, 1, still)
    spine = [j for j in st.joints if j in ("waist", "spine", "chest", "y_waist", "y_spine", "y_chest")]
    assert spine and "waist" in mgr._ped_blend[1]["arm"].joints
    fade_ticks = int(round(FADE_S / DT))
    with_spine = tick(mgr, 1, still)
    ov = mgr._ped_blend[1]["arm"]
    seen = [tick(mgr, 1, walking) for _ in range(fade_ticks + 2)]
    assert mgr._ped_blend[1]["arm"] is ov  # same overlay, no reinstall
    assert not (set(spine) & ov.joints)
    assert st.joints == layer._gesture("point").joints(st.clip.side, True)
    steps = np.array([[abs(b[j] - a[j]) for j in spine] for a, b in zip([with_spine, *seen], seen)])
    assert steps.max() < 0.1
    assert max(abs(with_spine[j] - seen[-1][j]) for j in spine) > 0.05
    mid = seen[fade_ticks // 2]
    for j in spine:
        lo, hi = sorted((with_spine[j], seen[-1][j]))
        assert lo - 0.02 <= mid[j] <= hi + 0.02
    seen = [tick(mgr, 1, still) for _ in range(fade_ticks + 2)]
    assert mgr._ped_blend[1]["arm"] is ov
    assert set(spine) <= ov.joints
    assert st.joints == layer._gesture("point").joints(st.clip.side, False)
    steps = np.array([[abs(b[j] - a[j]) for j in spine] for a, b in zip(seen, seen[1:])])
    assert steps.max() < 0.1
    for j in spine:
        assert seen[-1][j] == pytest.approx(with_spine[j], abs=0.05)


def test_same_kind_request_waits_for_release_tail(rig):
    mgr, layer, _ = rig
    r = req("point", (1.0, 5.0, 1.2))
    st = to_hold(mgr, layer, 1, r)
    tick(mgr, 1, r, n=int(HOLD_MIN_S / DT) + 2)
    tick(mgr, 1, None)
    assert st.phase == "release"
    tail = mgr._ped_blend[1]["arm"].anim
    tick(mgr, 1, r, n=3)
    assert arm(layer, 1) is st
    assert mgr._ped_blend[1]["arm"].anim is tail
    tick(mgr, 1, r, n=tail.n_frames + int((COMPANION_RELEASE_LAG_S + FADE_S) / DT) + 12 + LEAD_TICKS)
    assert arm(layer, 1) is not st


class BrokenGesture:
    slot = "arm"
    REACH_IN = 10.0
    REACH_OUT = 10.0

    def __init__(self):
        self.starts = 0

    def joints(self, side: str, moving: bool) -> set[str]:
        return set()

    def reach(self, local) -> float:
        return 0.0

    def start(self, local, opts) -> GestureClip:
        self.starts += 1
        raise RuntimeError("boom")

    def retarget(self, hold, local, opts) -> GestureClip:
        raise ValueError("no")

    def companions(self) -> list[tuple[str, str]]:
        return []

    def breathing(self, side: str) -> dict[str, float]:
        return {}


def test_generator_fault_warns_once_and_drops(rig):
    mgr, layer, log = rig
    broken = BrokenGesture()
    layer._gestures["broken"] = broken
    GESTURES._loaders.setdefault("broken", BrokenGesture)
    try:
        tick(mgr, 1, req("broken", (1.0, 5.0, 1.2)), n=3)
        assert 1 not in layer._agents
        assert 1 not in mgr._ped_blend
        assert sum("failed" in line for line in log.lines) == 1
        assert broken.starts == 1
        tick(mgr, 1, req("broken", (1.0, 5.5, 1.2)), n=3)
        assert broken.starts == 2
        assert 1 not in layer._agents
        tick(mgr, 1, req("broken", (1.0, 5.5, 1.2), opts={"hand": "l"}), n=2)
        assert broken.starts == 3
        layer.forget(1)
        tick(mgr, 1, req("broken", (1.0, 5.5, 1.2), opts={"hand": "l"}), n=2)
        assert broken.starts == 4
    finally:
        GESTURES._loaders.pop("broken", None)
    assert sum("failed" in line for line in log.lines) == 2  # forget clears the warn-once set too


def test_target_behind_the_ped_is_not_pointed_at(rig):
    mgr, layer, _ = rig
    tick(mgr, 1, req("point", (1.0, -3.0, 1.2)), n=3)
    assert 1 not in layer._agents
    assert 1 not in mgr._ped_blend


# -- opts ---------------------------------------------------------------


def test_request_opts_default_empty():
    assert GestureRequest(kind="point", at=(0.0, 0.0, 0.0), pose=(0.0, 0.0, 0.0), moving=False).opts == {}


def test_hand_override_and_hand_change_restart(rig):
    mgr, layer, _ = rig
    left_side = req("point", (-1.0, 4.0, 1.2))  # target on the left of a ped facing +y
    st = to_hold(mgr, layer, 1, left_side)
    assert st.clip.side == "l"
    tick(mgr, 1, None, n=int(HOLD_MIN_S / DT) + 60)
    assert 1 not in layer._agents
    forced = req("point", (-1.0, 4.0, 1.2), opts={"hand": "right"})
    st = to_hold(mgr, layer, 1, forced)
    assert st.clip.side == "r"
    tick(mgr, 1, req("point", (-1.0, 4.0, 1.2), opts={"hand": "left"}))
    assert st.phase == "release"
    tick(mgr, 1, req("point", (-1.0, 4.0, 1.2), opts={"hand": "left"}), n=len(st.clip.frames) + 60)
    new = arm(layer, 1)
    assert new is not st and new.clip.side == "l"


def test_bogus_hand_falls_back_to_auto():
    g = PointGesture()
    assert g._options({"hand": "tentacle"}).hand == "auto"
    assert g._options({"hand": "l"}).hand == "l"
    assert g._options({}).hand == "auto"


# -- look ---------------------------------------------------------------


def head_dir(angles: dict[str, float]) -> np.ndarray:
    y, p = angles["y_head"], angles["p_head"]
    return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])


def test_look_frames_aim_within_limits_and_release_to_zero():
    g = LookGesture()
    local = np.array([2.0, 1.0, 0.4])
    clip = g.start(local, {})
    assert clip.hold_start == 5 and clip.hold_end == len(clip.frames) - 1
    assert set(clip.frames[0]["angles"]) == set(HEAD)
    want = local - g.anchor
    want /= np.linalg.norm(want)
    err = math.degrees(math.acos(float(np.clip(head_dir(clip.frames[-1]["angles"]) @ want, -1, 1))))
    assert err < 2.0
    for f in clip.frames + list(clip.release):
        for k in HEAD:
            lo, hi = LIMITS[k]
            assert lo <= f["angles"][k] <= hi
    assert clip.frames[0]["angles"]["y_head"] < clip.frames[-1]["angles"]["y_head"]  # eases in from rest
    assert all(abs(v) < 1e-12 for v in clip.release[-1]["angles"].values())
    assert isinstance(clip.hold, LookHold)
    again = g.retarget(clip.hold, np.array([2.0, -1.5, 2.5]), {})
    assert again.hold.yaw < 0 and again.hold.pitch > 0
    assert abs(again.frames[0]["angles"]["y_head"] - clip.hold.yaw) < 0.2 * abs(again.hold.yaw - clip.hold.yaw)
    assert again.frames[-1]["angles"]["y_head"] == again.hold.yaw
    far = g.start(np.array([0.5, 3.0, 5.0]), {})
    assert far.hold.yaw <= LIMITS["y_head"][1] and far.hold.pitch <= LIMITS["p_head"][1]


def test_look_as_primary(rig):
    mgr, layer, _ = rig
    r = req("look", (1.0, 5.0, 1.2))
    tick(mgr, 1, r)
    assert set(mgr._ped_blend[1]) == {"head"}
    st = head(layer, 1)
    assert st.primary and st.phase == "ramp"
    tick(mgr, 1, r, n=6)
    assert st.phase == "hold"
    a = tick(mgr, 1, r)
    assert abs(a["y_head"]) < 0.05 and abs(a["p_head"]) > 0.05  # straight ahead, slightly down
    tick(mgr, 1, r, n=int(HOLD_MIN_S / DT))
    tick(mgr, 1, None)
    assert st.phase == "release"
    tick(mgr, 1, None, n=20)
    assert 1 not in layer._agents


def test_point_installs_gaze_and_releases_it_after_the_arm(rig):
    mgr, layer, _ = rig
    r = req("point", (-1.0, 4.0, 1.2))
    st = to_hold(mgr, layer, 1, r)
    hd = head(layer, 1)
    assert not hd.primary and hd.phase == "hold"
    assert set(mgr._ped_blend[1]) == {"head", "arm"}
    tick(mgr, 1, req("point", (-1.5, 5.0, 0.6)))
    assert st.phase == "transition" and hd.phase == "transition"
    assert np.allclose(hd.local, st.local)
    tick(mgr, 1, req("point", (-1.5, 5.0, 0.6)), n=int(HOLD_MIN_S / DT) + int(st.clip.hold_start / st.clip.fps / DT) + 2)
    tick(mgr, 1, None)
    assert st.phase == "release" and hd.phase == "hold"
    lag_ticks = int(round(COMPANION_RELEASE_LAG_S / DT))
    tick(mgr, 1, None, n=lag_ticks - 1)
    assert hd.phase == "hold"
    tick(mgr, 1, None)
    assert hd.phase == "release"
    tick(mgr, 1, None, n=len(st.clip.frames) + 60)
    assert 1 not in layer._agents and 1 not in mgr._ped_blend


def test_gaze_companion_out_of_reach_leaves_arm_alone(rig):
    mgr, layer, _ = rig
    ped_local = np.array([0.5, 1.5, 1.2])  # az ~1.25 rad: inside point's reach, outside look's
    world = (POSE[0] + math.cos(POSE[2]) * ped_local[0] - math.sin(POSE[2]) * ped_local[1], POSE[1] + math.sin(POSE[2]) * ped_local[0] + math.cos(POSE[2]) * ped_local[1], 1.2)
    tick(mgr, 1, req("point", world), n=LEAD_TICKS + 1)
    assert set(layer._agents[1].slots) == {"arm"}
    assert arm(layer, 1).clip is not None
    assert arm(layer, 1).age == pytest.approx(DT * LEAD_TICKS)


# -- timing ---------------------------------------------------------------


def test_move_time_clamps():
    assert move_time(0.0) == MOVE_MIN_S
    assert move_time(10.0) == MOVE_MAX_S
    assert move_time(1.75) == pytest.approx(1.75 / OMEGA_RAD_S)


def test_resample_endpoints_and_length():
    frames = [{"angles": {"a": float(i)}, "t": i} for i in range(4)]
    out = resample(frames, 7)
    assert len(out) == 7
    assert out[0]["angles"]["a"] == 0.0 and out[-1]["angles"]["a"] == 3.0
    assert [f["angles"]["a"] for f in out] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert resample(frames, 2)[1]["angles"]["a"] == 3.0
    assert resample(frames, 0) == []


def test_ramp_length_scales_with_swept_angle():
    g = PointGesture()
    lo_t, hi_t = np.array([1.5, -0.4, 0.3]), np.array([0.05, -0.4, 5.0])
    low = g.start(lo_t, {})  # target low in front: short sweep
    up = g.start(hi_t, {})  # nearly straight up: long sweep
    raw = g._gen.point_at(target=hi_t, options=g._options({}))
    assert low.hold_start < raw.hold_range[0] < up.hold_start
    for clip, t in ((low, lo_t), (up, hi_t)):
        assert clip.hold_start == max(2, int(round(move_time(math.acos(-g._gen.aim_of(t)[2])) * 20.0)))
    assert len(up.release) == int(round(1.25 * (len(raw.frames) - raw.peak)))
    park = up.frames[up.hold_end]["angles"]
    assert park == raw.frames[raw.peak]["angles"]
    assert up.frames[up.hold_start - 1]["angles"] == raw.frames[raw.hold_range[0] - 1]["angles"]


def test_retarget_light_for_small_deltas_keeps_swivel(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, req("point", (-1.0, 4.0, 1.2)))
    hold = st.hold
    tick(mgr, 1, req("point", (-1.2, 4.0, 1.3)))
    assert st.phase == "transition"
    assert st.clip.report.get("light") is True
    assert st.clip.report["swivel_rad"] == hold.swivel
    assert st.clip.report["aim_error_deg"] < 2.0
    tr = st.clip.report["transition_frames"]
    assert tr == int(round(MOVE_MIN_S * 20.0))
    tick(mgr, 1, req("point", (-1.2, 4.0, 1.3)), n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    assert st.phase == "hold"
    tick(mgr, 1, req("point", (0.2, 2.5, 5.0)))
    assert st.phase == "transition"
    assert "light" not in st.clip.report
    assert st.clip.report["transition_frames"] > tr


def test_light_threshold_constant():
    assert LIGHT_RETARGET_RAD == pytest.approx(math.radians(20.0))


def test_breathing_loop_is_bounded_and_periodic(rig):
    mgr, layer, _ = rig
    r = req("point", (1.0, 5.0, 1.2))
    st = to_hold(mgr, layer, 1, r)
    anim = mgr._ped_blend[1]["arm"].anim
    assert anim.loop and anim.loop_from == st.clip.hold_end + 1
    period = int(round(20.0 / BREATH_HZ))
    assert anim.n_frames == st.clip.hold_end + 1 + period
    park = st.clip.frames[st.clip.hold_end]["angles"]
    side = st.clip.side
    tail = anim.frames[anim.loop_from :]
    for j in (f"{side}_p_shoulder", f"{side}_elbow"):
        vals = np.array([f["angles"][j] - park[j] for f in tail])
        assert np.abs(vals).max() <= BREATH_AMP_RAD + 1e-9
        assert np.abs(vals).max() > 0.5 * BREATH_AMP_RAD
        # one full period: the wrap from the last tail frame back to the first is one frame step
        step = np.abs(np.diff(np.concatenate([vals, vals[:1]])))
        assert step.max() < 2.0 * BREATH_AMP_RAD * math.sin(math.pi / period) + 1e-9
    other = [j for j in arm_dofs(side) if j not in (f"{side}_p_shoulder", f"{side}_elbow")]
    for f in tail:
        assert all(f["angles"][j] == park[j] for j in other)
    # ramp frames untouched, drift eases in from hold_start
    assert anim.frames[st.clip.hold_start]["angles"] == st.clip.frames[st.clip.hold_start]["angles"]
    assert anim.frames[0]["angles"] == st.clip.frames[0]["angles"]
    # long hold keeps sampling the loop, joints stay bounded
    tick(mgr, 1, r, n=int((st.clip.hold_end - st.clip.hold_start) / st.clip.fps / DT) + 3)
    seen = []
    for _ in range(3 * period):
        seen.append(tick(mgr, 1, r)[f"{side}_elbow"])
    seen = np.array(seen)
    assert seen.max() - seen.min() <= 2.0 * BREATH_AMP_RAD + 1e-6
    assert seen.max() - seen.min() > BREATH_AMP_RAD


def test_breathing_phase_differs_per_agent(rig):
    mgr, layer, _ = rig
    r = req("point", (1.0, 5.0, 1.2))
    st1 = to_hold(mgr, layer, 1, r)
    st2 = to_hold(mgr, layer, 2, r)
    a1 = mgr._ped_blend[1]["arm"].anim.frames[st1.clip.hold_end + 3]["angles"]
    a2 = mgr._ped_blend[2]["arm"].anim.frames[st2.clip.hold_end + 3]["angles"]
    side = st1.clip.side
    assert a1[f"{side}_elbow"] != a2[f"{side}_elbow"]


# -- hysteresis -----------------------------------------------------------


def test_reach_hysteresis_in_and_out(rig):
    mgr, layer, _ = rig
    g = layer._gesture("point")

    def at(az: float) -> GestureRequest:
        local = np.array([2.0 * math.cos(az), 2.0 * math.sin(az), 1.2])
        world = (POSE[0] + math.cos(POSE[2]) * local[0] - math.sin(POSE[2]) * local[1], POSE[1] + math.sin(POSE[2]) * local[0] + math.cos(POSE[2]) * local[1], 1.2)
        return req("point", world)

    tick(mgr, 1, at(g.REACH_IN + 0.05), n=3)
    assert 1 not in layer._agents  # outside REACH_IN: never starts
    st = to_hold(mgr, layer, 1, at(g.REACH_IN - 0.05))
    tick(mgr, 1, at(g.REACH_OUT - 0.05))  # past REACH_IN but inside REACH_OUT: keeps going (retargets)
    assert st.phase in ("hold", "transition")
    tick(mgr, 1, at(g.REACH_OUT - 0.05), n=int(HOLD_MIN_S / DT) + 40)
    tick(mgr, 1, at(g.REACH_OUT + 0.05))
    assert st.phase == "release"
    assert g.REACH_IN < g.REACH_OUT
    look = layer._gesture("look")
    assert look.REACH_IN < look.REACH_OUT < g.REACH_IN


# -- retarget mid crossfade -----------------------------------------------


def test_retarget_install_carries_the_moving_crossfade(rig):
    mgr, layer, _ = rig
    still = req("point", (1.0, 5.0, 1.2))
    st = to_hold(mgr, layer, 1, still)
    spine = sorted(j for j in st.joints if j not in arm_dofs(st.clip.side))
    assert spine
    ov = mgr._ped_blend[1]["arm"]
    fade_ticks = int(round(FADE_S / DT))
    seen = [tick(mgr, 1, still)]
    seen.append(tick(mgr, 1, req("point", (1.0, 5.0, 1.2), moving=True)))
    assert set(spine) <= ov.joints and all(j in ov.ramps for j in spine)
    moved = req("point", (1.0 - 2.0 * RETARGET_EPS_M, 5.0, 1.2), moving=True)
    seen.append(tick(mgr, 1, moved))
    assert st.phase == "transition"
    new = mgr._ped_blend[1]["arm"]
    assert new is not ov
    assert set(spine) <= new.joints and all(j in new.ramps for j in spine)
    assert st.joints == layer._gesture("point").joints(st.clip.side, True)
    seen += [tick(mgr, 1, moved) for _ in range(fade_ticks)]
    assert not (set(spine) & new.joints)
    steps = np.array([[abs(b[j] - a[j]) for j in spine] for a, b in zip(seen, seen[1:])])
    assert steps.max() < 0.1
    assert max(abs(seen[0][j] - seen[-1][j]) for j in spine) > 0.05


# -- async executor -----------------------------------------------------------


def test_companion_release_survives_a_pending_job(async_rig):
    mgr, layer, _, ex = async_rig
    a = local_req("point", (1.0, 0.5, 1.2))
    tick(mgr, 1, a)
    ex.land()
    tick(mgr, 1, a, n=LEAD_TICKS + 1)
    st = arm(layer, 1)
    tick(mgr, 1, a, n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    hd = head(layer, 1)
    assert st.phase == "hold" and hd.phase == "hold"
    b = local_req("point", (0.8, 1.0, 1.0))
    tick(mgr, 1, b)
    assert st.job is not None and hd.job is not None
    ex.land(1)  # arm lands, head stays pending
    tick(mgr, 1, b)
    assert st.phase == "transition" and hd.job is not None
    tick(mgr, 1, b, n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    assert st.phase == "hold" and hd.job is not None and hd.clip is not None
    c = local_req("point", (0.3, 1.5, 1.2))  # inside point's reach, outside look's
    tick(mgr, 1, c)
    assert st.job is not None and hd.release_at is not None
    tick(mgr, 1, c, n=2)
    assert hd.phase == "hold" and hd.job is not None
    ex.land()
    tick(mgr, 1, c)
    assert hd.phase == "release"
    assert st.phase == "transition"


def test_one_tick_request_never_installs_in_async(async_rig):
    mgr, layer, _, ex = async_rig
    r = req("point", (1.0, 5.0, 1.2))
    tick(mgr, 1, r)
    assert set(layer._agents[1].slots) == {"head", "arm"}
    assert ex.pending
    tick(mgr, 1, None)
    assert 1 not in layer._agents
    assert all(fut.cancelled() for fut, _ in ex.pending)
    ex.land()
    tick(mgr, 1, None, n=3)
    assert 1 not in layer._agents
    assert 1 not in mgr._ped_blend
    assert not any(name.startswith("gesture:") for name in mgr.animations)


def test_one_tick_request_never_installs_in_sync(rig):
    mgr, layer, _ = rig
    tick(mgr, 1, req("point", (1.0, 5.0, 1.2)))
    assert set(mgr._ped_blend[1]) == {"head"}
    tick(mgr, 1, None)
    assert "arm" not in layer._agents[1].slots
    tick(mgr, 1, None, n=40)
    assert 1 not in layer._agents and 1 not in mgr._ped_blend


def test_failing_generator_is_not_resubmitted_until_the_request_changes(async_rig):
    mgr, layer, log, ex = async_rig
    broken = BrokenGesture()
    layer._gestures["broken"] = broken
    GESTURES._loaders.setdefault("broken", BrokenGesture)
    try:
        r = req("broken", (1.0, 5.0, 1.2))
        tick(mgr, 1, r)
        ex.land()
        tick(mgr, 1, r, n=2)
        assert ex.submitted == 1 and broken.starts == 1
        assert 1 not in layer._agents
        assert sum("failed" in line for line in log.lines) == 1
        tick(mgr, 1, req("broken", (1.0, 5.04, 1.2)), n=2)  # within the 0.1 m latch
        assert ex.submitted == 1
        tick(mgr, 1, req("broken", (1.0, 5.5, 1.2)))
        assert ex.submitted == 2
        ex.land()
        tick(mgr, 1, req("broken", (1.0, 5.5, 1.2)), n=2)
        assert ex.submitted == 2
    finally:
        GESTURES._loaders.pop("broken", None)
