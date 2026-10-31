from __future__ import annotations

import math

import numpy as np
import pytest

from task_generator.simulators.human.animation_manager import JOINT_NAMES, AnimationManager
from task_generator.simulators.human.gestures import (
    BREATH_AMP_RAD,
    BREATH_HZ,
    FADE_S,
    HOLD_MIN_S,
    MOVE_MAX_S,
    MOVE_MIN_S,
    OMEGA_RAD_S,
    RETARGET_EPS_M,
    Channel,
    GestureClip,
    GestureLayer,
    GestureRequest,
    move_time,
    resample,
    world_to_local,
)
from task_generator.simulators.human.gestures.look import FOLLOW_OMEGA_RAD_S, HEAD, PEAK_RAD_S, LookGesture, LookHold
from task_generator.simulators.human.gestures.point import PointGesture
from task_generator.simulators.human.pointing.contract import LIMITS, arm_dofs

pytestmark = pytest.mark.clips("point_at_template", "wave", "beckon", "hug", "shake_hand", "talk_with_arm_gesture_v1")
DT = 0.05
POSE = (1.0, 2.0, math.pi / 2)  # facing +y: world -x is the ped's left


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
    mgr = AnimationManager(logger=log, fps=20.0)
    layer = GestureLayer(mgr, log)
    mgr.gesture_hook = layer
    return mgr, layer, log


def req(*channels: Channel, moving: bool = False) -> GestureRequest:
    return GestureRequest(channels=channels, pose=POSE, moving=moving)


def arm(at: tuple[float, float, float], slot: str = "arm", **opts: str) -> Channel:
    return Channel(slot, at, hand=opts.get("dominant", ""))


def head_ch(at: tuple[float, float, float]) -> Channel:
    return Channel("head", at)


def point(at: tuple[float, float, float], moving: bool = False, **opts: str) -> GestureRequest:
    return req(arm(at, **opts), moving=moving)


def tick(mgr: AnimationManager, agent: int, request: GestureRequest | None, n: int = 1) -> dict[str, float]:
    out = {}
    for _ in range(n):
        out = mgr.compute(agent, 0, 0.0, DT, gesture=request)
    return out


def arm_slot(layer: GestureLayer, agent: int):
    return layer._agents[agent].slots["arm"]


def head_slot(layer: GestureLayer, agent: int):
    return layer._agents[agent].slots["head"]


def arm_moved(angles: dict[str, float], side: str) -> bool:
    return any(abs(angles[j]) > 1e-3 for j in arm_dofs(side))


def to_hold(mgr: AnimationManager, layer: GestureLayer, agent: int, r: GestureRequest, slot: str = "arm"):
    """Start ``r`` and tick until ``slot`` holds."""
    tick(mgr, agent, r)
    st = layer._agents[agent].slots[slot]
    assert st.clip is not None
    tick(mgr, agent, r, n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    assert st.phase == "hold"
    return st


def settle(mgr: AnimationManager, agent: int, r: GestureRequest | None, st) -> None:
    """Tick past HOLD_MIN_S so a release is honoured at once."""
    tick(mgr, agent, r, n=int((st.clip.hold_start / st.clip.fps + HOLD_MIN_S) / DT) + 2)


def drain(mgr: AnimationManager, agent: int, r: GestureRequest | None, st, extra: int = 12) -> None:
    tick(mgr, agent, r, n=len(st.clip.frames) + int(FADE_S / DT) + extra)


def test_world_to_local_rotates_into_ped_frame():
    local = world_to_local((1.0, 5.0, 1.2), POSE)
    assert np.allclose(local, [3.0, 0.0, 1.2], atol=1e-9)


def test_unknown_slot_warns_once_and_leaves_gait(rig):
    mgr, layer, log = rig
    tick(mgr, 1, req(Channel("tail", (0.0, 4.0, 1.0))), n=3)
    assert sum("unknown" in line for line in log.lines) == 1
    assert 1 not in layer._agents
    assert 1 not in mgr._ped_blend


def test_point_ramps_holds_and_releases(rig):
    mgr, layer, log = rig
    r = point((1.0, 5.0, 1.2))
    tick(mgr, 1, r)
    assert set(mgr._ped_blend[1]) == {"arm"}  # no head companion
    st = arm_slot(layer, 1)
    assert st.phase == "ramp"
    tick(mgr, 1, r, n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    assert st.phase == "hold"
    a = tick(mgr, 1, r)
    assert arm_moved(a, st.clip.side)
    tick(mgr, 1, r, n=int(HOLD_MIN_S / DT) + 1)
    tick(mgr, 1, None)
    assert st.phase == "release"
    drain(mgr, 1, None, st)
    assert 1 not in layer._agents
    assert 1 not in mgr._ped_blend


def test_channel_disappearing_releases_the_slot(rig):
    mgr, layer, _ = rig
    both = req(arm((1.0, 5.0, 1.2)), head_ch((1.0, 5.0, 1.6)))
    st = to_hold(mgr, layer, 1, both)
    hd = head_slot(layer, 1)
    settle(mgr, 1, both, st)
    tick(mgr, 1, req(head_ch((1.0, 5.0, 1.6))))
    assert st.phase == "release" and hd.phase == "hold"
    drain(mgr, 1, req(head_ch((1.0, 5.0, 1.6))), st)
    assert set(layer._agents[1].slots) == {"head"}
    assert hd.phase == "hold"


def test_clear_before_min_hold_is_deferred(rig):
    mgr, layer, _ = rig
    r = point((1.0, 5.0, 1.2))
    tick(mgr, 1, r)
    st = arm_slot(layer, 1)
    tick(mgr, 1, None)
    assert st.phase != "release"
    assert st.release_pending
    tick(mgr, 1, None, n=int((st.clip.hold_start / st.clip.fps + HOLD_MIN_S) / DT) + 2)
    assert st.phase == "release"


def test_retarget_transitions_without_release(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, point((-1.0, 4.0, 1.2)))
    first = st.clip
    tick(mgr, 1, point((-1.5, 5.0, 0.6)))
    assert st.clip is not first
    assert st.phase == "transition"
    assert st.clip.side == first.side
    held = first.hold.angles
    start = st.clip.frames[0]["angles"]
    assert max(abs(start[j] - held[j]) for j in held) < 1e-6


def test_small_target_jitter_does_not_retarget(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, point((-1.0, 4.0, 1.2)))
    clip = st.clip
    tick(mgr, 1, point((-1.05, 4.05, 1.2)))
    assert st.clip is clip


# -- hand -----------------------------------------------------------------


def test_hand_resolved_once_and_kept_across_the_midline(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, point((-1.0, 4.0, 1.2)))  # left of the ped
    assert st.clip.side == "l" and st.bound["hand"] == "l"
    for at in ((1.0, 4.0, 1.2), (2.0, 4.0, 1.2)):  # midline, then right of the ped
        settle(mgr, 1, point(at), st)
        assert st.phase == "hold" and st.clip.side == "l"
    assert arm_slot(layer, 1) is st
    assert st.bound["hand"] == "l"


def test_dominant_opt_picks_the_midline_arm(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, point((1.0, 5.0, 1.2)))
    assert st.clip.side == "r"
    tick(mgr, 1, None, n=int(HOLD_MIN_S / DT) + 60)
    assert 1 not in layer._agents
    st = to_hold(mgr, layer, 1, point((1.0, 5.0, 1.2), dominant="l"))
    assert st.clip.side == "l"


def test_explicit_arm_slots_force_the_side(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, req(arm((-1.0, 4.0, 1.2), slot="arm_r")))
    assert st.channel == "arm_r" and st.slot == "arm" and st.clip.side == "r"
    tick(mgr, 1, None, n=int(HOLD_MIN_S / DT) + 60)
    assert 1 not in layer._agents
    st = to_hold(mgr, layer, 1, req(arm((2.0, 4.0, 1.2), slot="arm_l")))
    assert st.channel == "arm_l" and st.clip.side == "l"


def test_switching_arm_slot_releases_then_restarts(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, point((-1.0, 4.0, 1.2)))
    side = st.clip.side
    other = req(arm((2.0, 4.0, 1.2), slot="arm_r"))
    tick(mgr, 1, other)
    assert st.phase == "release"
    assert set(layer._agents[1].slots) == {"arm"}
    drain(mgr, 1, other, st)
    new = arm_slot(layer, 1)
    assert new is not st
    assert new.channel == "arm_r" and new.clip.side == "r" and side == "l"


def test_opts_change_restarts(rig):
    mgr, layer, _ = rig
    r = point((1.0, 5.0, 1.2))
    st = to_hold(mgr, layer, 1, r)
    changed = point((1.0, 5.0, 1.2), dominant="l")
    tick(mgr, 1, changed)
    assert st.phase == "release"
    drain(mgr, 1, changed, st)
    new = arm_slot(layer, 1)
    assert new is not st and new.opts == {"dominant": "l"} and new.clip.side == "l"


def test_bind_and_resolve_hand():
    g = PointGesture()
    assert g.bind("arm_l", np.array([2.0, -1.0, 1.2]), {}) == {"hand": "l"}
    assert g.bind("arm_r", np.array([2.0, 1.0, 1.2]), {}) == {"hand": "r"}
    assert g.resolve_hand(np.array([2.0, 1.0, 1.2]), {}) == "l"
    assert g.resolve_hand(np.array([2.0, -1.0, 1.2]), {}) == "r"
    assert g.resolve_hand(np.array([2.0, 0.0, 1.2]), {}) == "r"
    assert g.resolve_hand(np.array([2.0, 0.0, 1.2]), {"dominant": "l"}) == "l"
    assert g.bind("arm", np.array([2.0, 0.0, 1.2]), {"dominant": "l"}) == {"dominant": "l", "hand": "l"}
    assert g._options({"hand": "tentacle"}).hand == "auto"
    assert g._options({"hand": "l"}).hand == "l"
    assert g._options({}).hand == "auto" and g._options({}).dominant == "r"
    assert g._options({"dominant": "left"}).dominant == "l"


def test_forced_hand_retargets_across_the_midline_without_switch():
    g = PointGesture()
    bound = g.bind("arm", np.array([2.0, 1.0, 1.2]), {})
    assert bound["hand"] == "l"
    clip = g.start(np.array([2.0, 1.0, 1.2]), bound)
    again = g.retarget(clip.hold, np.array([2.0, -0.8, 1.2]), bound)
    assert again.side == "l"


# -- head + arm ------------------------------------------------------------


def test_head_and_arm_are_independent(rig):
    mgr, layer, _ = rig
    both = req(arm((-1.0, 4.0, 1.2)), head_ch((2.0, 4.0, 1.6)))
    st = to_hold(mgr, layer, 1, both)
    hd = head_slot(layer, 1)
    assert set(mgr._ped_blend[1]) == {"head", "arm"}
    assert not np.allclose(hd.local, st.local)
    assert hd.hold.yaw < 0 < world_to_local((-1.0, 4.0, 1.2), POSE)[1]
    tick(mgr, 1, req(arm((-1.5, 5.0, 0.6)), head_ch((2.0, 4.0, 1.6))))
    assert st.phase == "transition" and hd.phase == "hold"
    settle(mgr, 1, req(arm((-1.5, 5.0, 0.6)), head_ch((2.0, 4.0, 1.6))), st)
    tick(mgr, 1, req(arm((-1.5, 5.0, 0.6))))
    assert hd.phase == "release" and st.phase == "hold"
    tick(mgr, 1, req(arm((-1.5, 5.0, 0.6))), n=30)
    assert set(layer._agents[1].slots) == {"arm"} and st.phase == "hold"
    tick(mgr, 1, req(arm((-1.5, 5.0, 0.6)), head_ch((-2.0, 4.0, 1.6))))
    assert set(layer._agents[1].slots) == {"arm", "head"}
    assert head_slot(layer, 1) is not hd


def test_head_only_starts_no_arm(rig):
    mgr, layer, _ = rig
    r = req(head_ch((1.0, 5.0, 1.2)))
    tick(mgr, 1, r)
    assert set(mgr._ped_blend[1]) == {"head"}
    st = head_slot(layer, 1)
    assert st.phase == "hold"
    tick(mgr, 1, r, n=6)
    assert st.phase == "hold"
    a = tick(mgr, 1, r)
    assert abs(a["y_head"]) < 0.05 and abs(a["p_head"]) > 0.05  # straight ahead, slightly down
    tick(mgr, 1, r, n=int(HOLD_MIN_S / DT))
    tick(mgr, 1, None)
    assert st.phase == "release"
    tick(mgr, 1, None, n=30)
    assert 1 not in layer._agents


def test_second_arm_channel_is_ignored_with_one_warning(rig):
    mgr, layer, log = rig
    r = req(arm((-1.0, 4.0, 1.2)), arm((2.0, 4.0, 1.2), slot="arm_r"))
    st = to_hold(mgr, layer, 1, r)
    assert set(layer._agents[1].slots) == {"arm"} and st.channel == "arm"
    assert sum("several arm channels" in line for line in log.lines) == 1


# -- moving / joints -------------------------------------------------------


def test_moving_ped_blends_arm_only(rig):
    mgr, layer, _ = rig
    tick(mgr, 1, point((1.0, 5.0, 1.2), moving=True))
    joints = mgr._ped_blend[1]["arm"].joints
    assert "waist" not in joints
    assert set(arm_dofs(arm_slot(layer, 1).clip.side)) <= joints


def test_forget_drops_state_and_transients(rig):
    mgr, layer, _ = rig
    tick(mgr, 1, req(arm((1.0, 5.0, 1.2)), head_ch((1.0, 5.0, 1.6))))
    assert any(name.startswith("gesture:point:1") for name in mgr.animations)
    assert any(name.startswith("gesture:look:1") for name in mgr.animations)
    layer.forget(1)
    mgr.forget(1)
    assert 1 not in layer._agents
    assert not any(name.startswith("gesture:") for name in mgr.animations)


def test_release_after_retarget_eases_from_the_new_hold_to_rest(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, point((-0.3, 4.0, 0.3)))  # front, low
    first_release = st.release_frames
    high = point((-2.5, 3.0, 2.2))  # side, high
    tick(mgr, 1, high)
    assert st.phase == "transition"
    assert st.release_frames is not first_release
    settle(mgr, 1, high, st)
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
    st2 = to_hold(mgr, layer, 2, point((-0.3, 4.0, 0.3)))
    tick(mgr, 2, point((-0.3, 4.0, 0.3)), n=int(HOLD_MIN_S / DT) + 2)
    tick(mgr, 2, None)
    assert st2.phase == "release"
    assert st2.release_frames is st2.clip.release


def test_moving_flip_crossfades_spine_joints(rig):
    mgr, layer, _ = rig
    still, walking = point((1.0, 5.0, 1.2)), point((1.0, 5.0, 1.2), moving=True)
    st = to_hold(mgr, layer, 1, still)
    spine = [j for j in st.joints if j in ("waist", "spine", "chest", "y_waist", "y_spine", "y_chest")]
    assert spine and "waist" in mgr._ped_blend[1]["arm"].joints
    fade_ticks = int(round(FADE_S / DT))
    with_spine = tick(mgr, 1, still)
    ov = mgr._ped_blend[1]["arm"]
    seen = [tick(mgr, 1, walking) for _ in range(fade_ticks + 2)]
    ov = mgr._ped_blend[1]["arm"]  # re-solved against the upright torso, ramps carried over
    assert st.bound["moving"] is True and st.clip.report["upright"] is True
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
    ov = mgr._ped_blend[1]["arm"]
    assert st.bound["moving"] is False and st.clip.report["upright"] is False
    assert set(spine) <= ov.joints
    assert st.joints == layer._gesture("point").joints(st.clip.side, False)
    steps = np.array([[abs(b[j] - a[j]) for j in spine] for a, b in zip(seen, seen[1:])])
    assert steps.max() < 0.1
    for j in spine:
        assert seen[-1][j] == pytest.approx(with_spine[j], abs=0.05)


def test_moving_flip_rejoints_head_and_arm(rig):
    mgr, layer, _ = rig
    both = req(arm((1.0, 5.0, 1.2)), head_ch((1.0, 5.0, 1.6)))
    st = to_hold(mgr, layer, 1, both)
    hd = head_slot(layer, 1)
    tick(mgr, 1, req(arm((1.0, 5.0, 1.2)), head_ch((1.0, 5.0, 1.6)), moving=True))
    assert layer._agents[1].moving
    assert st.joints == layer._gesture("point").joints(st.clip.side, True)
    assert hd.joints == set(HEAD)


def test_same_channel_request_waits_for_release_tail(rig):
    mgr, layer, _ = rig
    r = point((1.0, 5.0, 1.2))
    st = to_hold(mgr, layer, 1, r)
    tick(mgr, 1, r, n=int(HOLD_MIN_S / DT) + 2)
    tick(mgr, 1, None)
    assert st.phase == "release"
    tail = mgr._ped_blend[1]["arm"].anim
    tick(mgr, 1, r, n=3)
    assert arm_slot(layer, 1) is st
    assert mgr._ped_blend[1]["arm"].anim is tail
    tick(mgr, 1, r, n=tail.n_frames + int(FADE_S / DT) + 12)
    assert arm_slot(layer, 1) is not st


class BrokenGesture:
    def __init__(self):
        self.starts = 0

    def joints(self, side: str, moving: bool) -> set[str]:
        return set()

    def bind(self, slot: str, local, opts: dict) -> dict:
        return dict(opts)

    def start(self, local, opts) -> GestureClip:
        self.starts += 1
        raise RuntimeError("boom")

    def retarget(self, hold, local, opts) -> GestureClip:
        raise ValueError("no")

    def breathing(self, side: str) -> dict[str, float]:
        return {}


def test_generator_fault_warns_once_and_drops(rig):
    mgr, layer, log = rig
    broken = BrokenGesture()
    layer._gestures["point"] = broken
    tick(mgr, 1, point((1.0, 5.0, 1.2)), n=3)
    assert 1 not in layer._agents
    assert 1 not in mgr._ped_blend
    assert sum("failed" in line for line in log.lines) == 1
    assert broken.starts == 1
    tick(mgr, 1, point((1.0, 5.5, 1.2)), n=3)
    assert broken.starts == 2
    assert 1 not in layer._agents
    tick(mgr, 1, point((1.0, 5.5, 1.2), dominant="l"), n=2)
    assert broken.starts == 3
    layer.forget(1)
    tick(mgr, 1, point((1.0, 5.5, 1.2), dominant="l"), n=2)
    assert broken.starts == 4
    assert sum("failed" in line for line in log.lines) == 2  # forget clears the warn-once set too


def test_arm_fault_leaves_the_head_alone(rig):
    mgr, layer, log = rig
    layer._gestures["point"] = BrokenGesture()
    r = req(arm((1.0, 5.0, 1.2)), head_ch((1.0, 5.0, 1.6)))
    tick(mgr, 1, r, n=3)
    assert set(layer._agents[1].slots) == {"head"}
    assert set(mgr._ped_blend[1]) == {"head"}


def test_channel_opts_default_empty():
    assert Channel("arm", (0.0, 0.0, 0.0)).opts == {}


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
    assert g.bind("head", local, {"x": 1}) == {"x": 1}


T_ON, T_STOP, T_OFF, T_END = 0.5, 4.0, 6.5, 9.0


def pass_by(dt: float) -> dict[str, np.ndarray]:
    """A walker with a target passing 1.5 m to its left at 1 m/s, then halting, then dropped: per-tick shown, aimed and gait head."""
    mgr = AnimationManager(logger=StubLogger(), fps=20.0)
    layer = GestureLayer(mgr, StubLogger())
    mgr.gesture_hook = layer
    gait = AnimationManager(logger=StubLogger(), fps=20.0)
    look = LookGesture()
    pose = (0.0, 0.0, 0.0)
    rows = []
    for k in range(1, int(round(T_END / dt)) + 1):
        t = k * dt
        at = (4.0 - min(max(t - T_ON, 0.0), T_STOP - T_ON), 1.5, 1.0)
        live = T_ON <= t + 1e-9 < T_OFF
        shown = mgr.compute(1, 1, 1.2, dt, gesture=GestureRequest(channels=(head_ch(at),) if live else (), pose=pose, moving=True))
        base = gait.compute(1, 1, 1.2, dt)
        aim = look.aim(world_to_local(at, pose))
        rows.append((t, shown["y_head"], shown["p_head"], aim.yaw, aim.pitch, base["y_head"], base["p_head"]))
    assert 1 not in layer._agents and 1 not in mgr._ped_blend
    t, yaw, pitch, aim_yaw, aim_pitch, gait_yaw, gait_pitch = np.array(rows).T
    return {"t": t, "yaw": yaw, "pitch": pitch, "aim_yaw": aim_yaw, "aim_pitch": aim_pitch, "gait_yaw": gait_yaw, "gait_pitch": gait_pitch}


def test_look_follows_a_passing_target_smoothly_at_any_tick_rate():
    w = FOLLOW_OMEGA_RAD_S
    runs = {}
    for dt in (1.0 / 30.0, 1.0 / 20.0):
        r = runs[dt] = pass_by(dt)
        t = r["t"]
        vel = np.diff(r["yaw"]) / dt
        acc = np.abs(np.diff(vel)) / dt
        at = t[2:]
        on, off = int(np.argmax(t + 1e-9 >= T_ON)), int(np.argmax(t + 1e-9 >= T_OFF)) - 1
        gap = max(abs(r["aim_yaw"][on] - r["gait_yaw"][on]), abs(r["yaw"][off] - r["gait_yaw"][off]))
        stop_rate = abs(r["aim_yaw"][t <= T_STOP][-1] - r["aim_yaw"][t <= T_STOP][-2]) / dt
        gait_acc = float(np.max(np.abs(np.diff(r["gait_yaw"], 2)))) / dt**2
        assert acc.max() <= w**2 * gap + w / math.e * stop_rate + gait_acc
        tracking = (at > T_ON + 1.5) & (at <= T_STOP)
        aim_acc = float(np.max(np.abs(np.diff(r["aim_yaw"], 2))[(at > T_ON + 1.0) & (at <= T_STOP)])) / dt**2
        settled = w**2 * gap * (1.0 + w * 1.5) * math.exp(-w * 1.5)
        assert acc[tracking].max() <= 1.1 * aim_acc + settled + 0.02
        speed = np.hypot(np.diff(r["yaw"]), np.diff(r["pitch"])) / dt
        assert speed.max() <= PEAK_RAD_S
        held = (t > T_OFF - 0.5) & (t + 1e-9 < T_OFF)
        assert np.degrees(np.abs(r["yaw"] - r["aim_yaw"])[held]).max() < 1.0
        assert np.degrees(np.abs(r["pitch"] - r["aim_pitch"])[held]).max() < 1.0
        assert r["yaw"][-1] == r["gait_yaw"][-1] and r["pitch"][-1] == r["gait_pitch"][-1]
    fast, slow = runs[1.0 / 30.0], runs[1.0 / 20.0]
    assert np.degrees(np.abs(fast["yaw"][2::3] - slow["yaw"][1::2])).max() < 0.5


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


def test_retarget_transition_scales_with_swept_angle(rig):
    mgr, layer, _ = rig
    st = to_hold(mgr, layer, 1, point((-1.0, 4.0, 1.2)))
    tick(mgr, 1, point((-1.2, 4.0, 1.3)))
    assert st.phase == "transition"
    assert st.clip.report["aim_error_deg"] < 3.0
    tr = st.clip.report["transition_frames"]
    assert tr == int(round(MOVE_MIN_S * 20.0))
    tick(mgr, 1, point((-1.2, 4.0, 1.3)), n=int(st.clip.hold_start / st.clip.fps / DT) + 2)
    assert st.phase == "hold"
    tick(mgr, 1, point((0.2, 2.5, 5.0)))
    assert st.phase == "transition"
    assert st.clip.report["transition_frames"] > tr


def test_breathing_loop_is_bounded_and_periodic(rig):
    mgr, layer, _ = rig
    r = point((1.0, 5.0, 1.2))
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
    r = point((1.0, 5.0, 1.2))
    st1 = to_hold(mgr, layer, 1, r)
    st2 = to_hold(mgr, layer, 2, r)
    a1 = mgr._ped_blend[1]["arm"].anim.frames[st1.clip.hold_end + 3]["angles"]
    a2 = mgr._ped_blend[2]["arm"].anim.frames[st2.clip.hold_end + 3]["angles"]
    side = st1.clip.side
    assert a1[f"{side}_elbow"] != a2[f"{side}_elbow"]


# -- retarget mid crossfade -----------------------------------------------


def test_retarget_install_carries_the_moving_crossfade(rig):
    mgr, layer, _ = rig
    still = point((1.0, 5.0, 1.2))
    st = to_hold(mgr, layer, 1, still)
    spine = sorted(j for j in st.joints if j not in arm_dofs(st.clip.side))
    assert spine
    ov = mgr._ped_blend[1]["arm"]
    fade_ticks = int(round(FADE_S / DT))
    seen = [tick(mgr, 1, still)]
    seen.append(tick(mgr, 1, point((1.0, 5.0, 1.2), moving=True)))
    assert set(spine) <= ov.joints and all(j in ov.ramps for j in spine)
    moved = point((1.0 - 2.0 * RETARGET_EPS_M, 5.0, 1.2), moving=True)
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


# -- body clips ----------------------------------------------------------------


def body(clip: str) -> Channel:
    return Channel("body", (0.0, 0.0, 0.0), clip=clip)


def test_clip_channel_plays_annotated_joints_and_loops(rig):
    mgr, layer, log = rig
    r = req(body("wave"))
    tick(mgr, 1, r)
    st = layer._agents[1].slots["body"]
    anim = mgr.animations["wave"]
    assert st.kind == "clip" and st.clip is not None and st.clip.loop is anim.loop
    assert st.joints == set(anim.joints) and "l_knee" not in st.joints and "r_elbow" in st.joints
    overlay = mgr._ped_blend[1]["body"]
    assert overlay.anim.loop and overlay.anim.n_frames == anim.n_frames
    angles = tick(mgr, 1, r, n=anim.n_frames + 5)
    assert set(angles) == set(JOINT_NAMES)
    assert "body" in layer._agents[1].slots
    assert not [line for line in log.lines if "failed" in line or "rejected" in line]


def test_clip_channel_whole_body_when_unannotated(rig):
    mgr, layer, _ = rig
    tick(mgr, 1, req(body("jump")))
    st = layer._agents[1].slots["body"]
    assert st.joints == set(JOINT_NAMES)
    assert not mgr._ped_blend[1]["body"].anim.loop


def test_clip_channel_releases_with_fade_and_unknown_clip_drops(rig):
    mgr, layer, log = rig
    r = req(body("beckon"))  # a loop without release frames: the slot fades out
    tick(mgr, 1, r, n=int(HOLD_MIN_S / DT) + mgr.clip("beckon").n_frames + 4)
    tick(mgr, 1, req(), n=int(FADE_S / DT) + 4)
    assert 1 not in layer._agents
    tick(mgr, 2, req(body("no_such_clip")), n=2)
    assert 2 not in layer._agents
    assert sum("rejected" in line for line in log.lines) == 1


def test_clip_and_point_share_a_ped(rig):
    mgr, layer, _ = rig
    r = req(arm((1.0, 5.0, 1.2)), body("wave"))
    tick(mgr, 1, r, n=3)
    slots = layer._agents[1].slots
    assert set(slots) == {"arm", "body"}


def body(clip: str, lock: bool = False) -> Channel:
    return Channel(slot="body", at=(0.0, 0.0, 0.0), clip=clip, lock=lock)


def _body_playhead(mgr: AnimationManager, agent: int) -> float:
    return next(playhead for slot, _, playhead, _ in mgr.overlays(agent) if slot == "body")


@pytest.mark.parametrize("clip", ["hug", "shake_hand"])
def test_locked_contact_clips_play_in_phase_across_partners(rig, clip: str) -> None:
    mgr, _, _ = rig
    a, b = 3, 200
    for _ in range(12):
        for agent in (a, b):
            mgr.compute(agent, 0, 0.0, DT, gesture=req(body(clip, lock=True)))
    assert _body_playhead(mgr, a) == pytest.approx(_body_playhead(mgr, b))


def test_unlocked_loop_clips_keep_the_per_agent_stagger(rig) -> None:
    mgr, _, _ = rig
    for agent in (3, 200):
        mgr.compute(agent, 0, 0.0, DT, gesture=req(body("beckon")))  # loops from its first frame, no intro
    assert _body_playhead(mgr, 3) != pytest.approx(_body_playhead(mgr, 200))


def test_phases_reports_kind_channel_phase_and_clip(rig) -> None:
    mgr, layer, _ = rig
    mgr.compute(7, 0, 0.0, DT, gesture=req(body("hug", lock=True)))
    assert layer.phases(7) == {"body": ("clip", "body", "ramp", "hug")}
    mgr.compute(7, 0, 0.0, DT, gesture=req(body("hug", lock=True)))
    assert layer.phases(7)["body"][2] == "hold"
    slots = mgr.overlays(7)
    assert [s for s, *_ in slots] == ["body"]
    _, overlay, playhead, weight = slots[0]
    assert overlay.loop and 0.0 <= playhead < overlay.anim.duration
    assert 0.0 < weight < 1.0
    seen = []
    for _ in range(int((HOLD_MIN_S + 2 * FADE_S) / DT) + 4):
        mgr.compute(7, 0, 0.0, DT, gesture=req())
        seen.append(layer.phases(7).get("body", (None, None, None, None))[2])
    assert seen[: int(HOLD_MIN_S / DT) - 2] == ["hold"] * (int(HOLD_MIN_S / DT) - 2)
    assert seen[-1] is None
    assert layer.phases(99) == {}


def test_starting_to_walk_keeps_a_canned_clip_playing(rig) -> None:
    mgr, _, _ = rig
    ch = body("hug", lock=True)
    prev, heads = None, []
    for i in range(60):
        moving = i >= 40
        angles = mgr.compute(9, 1 if moving else 0, 1.2 if moving else 0.0, DT, gesture=req(ch, moving=moving))
        heads.append(_body_playhead(mgr, 9))
        if prev is not None and i == 40:
            assert abs(angles["l_elbow"] - prev["l_elbow"]) < 0.1
        prev = angles
    assert heads[40] > heads[39]


def test_the_gate_covers_every_looping_canned_clip() -> None:
    from task_generator.simulators.human.gestures import qa

    loops = qa.looping_clips()
    assert "talk_with_arm_gesture" in loops and "hug" in loops
    cases = {c.name for c in qa.default_cases()}
    assert {f"loop_{name}" for name in loops} <= cases


def test_a_reverse_clip_keeps_its_wrap_through_the_layer() -> None:
    from task_generator.simulators.human.animation_manager import AnimationManager
    from task_generator.simulators.human.gestures.clip import ClipGesture

    mgr = AnimationManager(logger=StubLogger(), fps=20.0)
    clip = ClipGesture(mgr).start(np.zeros(3), {"clip": "talk_with_arm_gesture_v1"})
    assert clip.loop and clip.reverse


def test_a_clip_with_release_frames_plays_them_when_its_slot_ends(rig) -> None:
    mgr, layer, _ = rig
    down = [{"angles": {**dict.fromkeys(JOINT_NAMES, 0.0), "l_knee": -0.2 * k}} for k in range(8)]
    mgr.register_transient("squat_cut", down, fps=20.0, release=down[::-1])
    body = req(Channel("body", (0.0, 0.0, 0.0), clip="squat_cut"))
    st = to_hold(mgr, layer, 4, body, slot="body")
    assert st.clip.release == down[::-1]
    settle(mgr, 4, body, st)
    assert tick(mgr, 4, body)["l_knee"] == pytest.approx(-1.4, abs=0.05)
    knees = [tick(mgr, 4, req())["l_knee"] for _ in range(10)]
    assert st.phase == "release"
    assert knees[-1] > knees[0] + 0.5


def test_a_looping_clip_releases_from_where_its_loop_is(rig) -> None:
    """The release of a looping clip blends out of the pose the loop shows, not out of the clip's last frame."""
    mgr, layer, _ = rig
    ramp = [{"angles": {**dict.fromkeys(JOINT_NAMES, 0.0), "l_knee": -0.1 * k}} for k in range(20)]
    rest = [{"angles": dict.fromkeys(JOINT_NAMES, 0.0)} for _ in range(6)]
    mgr.register_transient("knee_loop", ramp, fps=20.0, loop=True, loop_from=4, release=rest)
    body = req(Channel("body", (0.0, 0.0, 0.0), clip="knee_loop"))
    st = to_hold(mgr, layer, 5, body, slot="body")
    settle(mgr, 5, body, st)
    shown = tick(mgr, 5, body)["l_knee"]
    for _ in range(40):  # stop mid-loop, well short of the last frame's -1.9
        if -1.0 < shown < -0.6:
            break
        shown = tick(mgr, 5, body)["l_knee"]
    assert -1.0 < shown < -0.6
    first = tick(mgr, 5, req())["l_knee"]
    assert st.phase == "release"
    assert first == pytest.approx(shown, abs=0.25)
