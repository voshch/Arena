from __future__ import annotations

from pathlib import Path

import pytest

from task_generator.simulators.human.animation_mananager import JOINT_NAMES, Animation, AnimationManager

ANIMATIONS = Path(__file__).resolve().parents[1] / "task_generator" / "simulators" / "human" / "animations"
ARMS = {"l_y_collar", "l_p_collar", "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "l_elbow", "r_y_collar", "r_p_collar", "r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow"}
FPS = 20.0
DT = 0.1


class StubLogger:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(msg)

    def warning(self, msg: str) -> None:
        self.lines.append(msg)


def const_frames(value: float, n: int) -> list[dict]:
    return [{"angles": dict.fromkeys(JOINT_NAMES, value)} for _ in range(n)]


@pytest.fixture()
def mgr() -> AnimationManager:
    m = AnimationManager(ANIMATIONS, logger=StubLogger(), fps=FPS)
    m.register_transient("zero_base", const_frames(0.0, 40), loop=True)
    m.map_state_to_animation(0, "zero_base")
    return m


def step(mgr: AnimationManager, agent: int, n: int = 1, dt: float = DT) -> dict[str, float]:
    out = {}
    for _ in range(n):
        out = mgr.compute(agent, 0, 0.0, dt)
    return out


def test_state_map_defaults():
    m = AnimationManager(ANIMATIONS, logger=StubLogger(), fps=FPS)
    assert m.state_to_animation_map == {0: "idle", 1: "walk", 2: "run", 3: "idle", 4: "idle", 5: "idle", 6: "idle"}
    for state in range(7):
        angles = m.compute(1, state, 1.0, DT)
        assert set(angles) == set(JOINT_NAMES)


def test_transient_register_drop_forget(mgr):
    anim = mgr.register_transient("t1", const_frames(0.5, 10), owner=7)
    assert isinstance(anim, Animation)
    assert anim.n_frames == 10 and anim.duration == pytest.approx(0.5) and anim.fps == FPS and not anim.loop
    assert mgr.animations["t1"] is anim
    slow = mgr.register_transient("t2", const_frames(0.5, 10), fps=10.0, loop=True, owner=7)
    assert slow.duration == pytest.approx(1.0) and slow.loop
    mgr.register_transient("t3", const_frames(0.5, 10), owner=8)
    mgr.drop_animation("t1")
    assert "t1" not in mgr.animations
    mgr.forget(7)
    assert "t2" not in mgr.animations
    assert "t3" in mgr.animations
    mgr.forget(8)
    assert "t3" not in mgr.animations


def test_envelope_fade_in_and_out(mgr):
    ov = mgr.register_transient("ov", const_frames(1.0, 40))  # 2.0 s, non-loop
    ended = []
    mgr.set_ped_blend(1, ov, blend_joints=ARMS, blend_weight=1.0, fade_in_s=0.5, fade_out_s=0.5, on_end=ended.append)
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.2)      # elapsed 0.1 / 0.5
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.4)
    assert step(mgr, 1, 3)["l_elbow"] == pytest.approx(1.0)   # elapsed 0.5
    assert step(mgr, 1, 5)["l_elbow"] == pytest.approx(1.0)   # mid clip, t = 1.0
    assert step(mgr, 1, 6)["l_elbow"] == pytest.approx(0.8)   # t = 1.6, remaining 0.4 / 0.5
    assert step(mgr, 1, 2)["l_elbow"] == pytest.approx(0.4)   # t = 1.8
    assert ended == []
    assert step(mgr, 1, 2)["l_elbow"] == pytest.approx(0.0)   # t = 2.0, faded out and dropped
    assert ended == [1]
    assert 1 not in mgr._ped_blend
    assert step(mgr, 1, 3)["l_elbow"] == pytest.approx(0.0)
    assert ended == [1]


def test_envelope_weight_scales(mgr):
    ov = mgr.register_transient("ov", const_frames(1.0, 40), loop=True)
    mgr.set_ped_blend(1, ov, blend_joints=ARMS, blend_weight=0.5, fade_in_s=1.0)
    assert step(mgr, 1)["r_elbow"] == pytest.approx(0.05)
    assert step(mgr, 1, 9)["r_elbow"] == pytest.approx(0.5)
    assert step(mgr, 1, 30)["r_elbow"] == pytest.approx(0.5)  # looping: no fade-out, no end


def test_on_end_fires_once_without_fade(mgr):
    ov = mgr.register_transient("ov", const_frames(1.0, 10))  # 0.5 s
    ended = []
    mgr.set_ped_blend(1, ov, blend_joints=ARMS, on_end=ended.append)
    step(mgr, 1, 4)
    assert ended == []
    step(mgr, 1)
    assert ended == [1]
    assert step(mgr, 1, 5)["l_elbow"] == pytest.approx(1.0)   # holds the last frame
    assert ended == [1]
    assert mgr._ped_blend[1]["arm"].anim is ov


def test_clear_ped_blend_with_fade(mgr):
    ov = mgr.register_transient("ov", const_frames(1.0, 40), loop=True)
    ended = []
    mgr.set_ped_blend(1, ov, blend_joints=ARMS, on_end=ended.append)
    assert step(mgr, 1, 3)["l_elbow"] == pytest.approx(1.0)
    mgr.clear_ped_blend(1, fade_out_s=0.5)
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.8)
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.6)
    mgr.clear_ped_blend(1, fade_out_s=5.0)                    # already releasing: no restart
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.4)
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.2)
    assert ended == []
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.0)
    assert ended == [1]
    assert 1 not in mgr._ped_blend
    step(mgr, 1, 3)
    assert ended == [1]


def test_clear_ped_blend_immediate(mgr):
    ov = mgr.register_transient("ov", const_frames(1.0, 40), loop=True)
    ended = []
    mgr.set_ped_blend(1, ov, blend_joints=ARMS, on_end=ended.append)
    step(mgr, 1, 2)
    mgr.clear_ped_blend(1)
    assert 1 not in mgr._ped_blend
    assert step(mgr, 1)["l_elbow"] == pytest.approx(0.0)
    assert ended == []
    mgr.set_ped_blend(1, ov, blend_joints=ARMS)
    step(mgr, 1)
    mgr.set_ped_blend(1, None)
    assert 1 not in mgr._ped_blend


def test_overlay_identity(mgr):
    a = mgr.register_transient("same", const_frames(1.0, 40))
    mgr.set_ped_blend(1, a, blend_joints=ARMS)
    step(mgr, 1, 5)
    assert mgr._overlay_playhead[(1, "arm")] == pytest.approx(0.5)
    mgr.set_ped_blend(1, a, blend_joints=ARMS)
    assert mgr._overlay_playhead[(1, "arm")] == pytest.approx(0.5)     # same object: untouched
    b = mgr.register_transient("same", const_frames(1.0, 40))
    mgr.set_ped_blend(1, b, blend_joints=ARMS)
    assert mgr._ped_blend[1]["arm"].anim is b
    assert mgr._overlay_playhead[(1, "arm")] == pytest.approx(0.0)     # different object, same name: reset
    mgr.set_ped_blend(1, "same", blend_joints=ARMS)           # name resolves to the current object
    assert mgr._ped_blend[1]["arm"].anim is b


def test_compute_gesture_calls_hook(mgr):
    calls = []
    mgr.gesture_hook = lambda agent_id, gesture, dt: calls.append((agent_id, gesture, dt))
    req = {"kind": "point", "target": (1.0, 2.0, 1.5)}
    mgr.compute(3, 0, 0.0, DT, gesture=req)
    assert calls == [(3, req, DT)]
    mgr.compute(3, 0, 0.0, DT)
    assert calls[-1] == (3, None, DT)
    mgr.gesture_hook = None
    mgr.compute(3, 0, 0.0, DT, gesture=req)
    assert len(calls) == 2


def test_slots_are_independent(mgr):
    arm = mgr.register_transient("arm", const_frames(1.0, 40), loop=True)
    head = mgr.register_transient("head", const_frames(0.5, 10))  # 0.5 s, non-loop
    ended = []
    mgr.set_ped_blend(1, arm, blend_joints=ARMS, slot="arm")
    mgr.set_ped_blend(1, head, blend_joints={"y_head"}, fade_in_s=0.5, on_end=ended.append, slot="head")
    assert set(mgr._ped_blend[1]) == {"arm", "head"}
    a = step(mgr, 1)
    assert a["l_elbow"] == pytest.approx(1.0)
    assert a["y_head"] == pytest.approx(0.1)                       # own envelope: 0.1 / 0.5
    assert mgr._overlay_playhead[(1, "arm")] != mgr._overlay_playhead[(1, "head")]
    a = step(mgr, 1, 4)
    assert a["y_head"] == pytest.approx(0.5) and a["l_elbow"] == pytest.approx(1.0)
    assert ended == [1]                                            # head clip ended, arm keeps looping
    mgr.clear_ped_blend(1, slot="head")
    assert set(mgr._ped_blend[1]) == {"arm"}
    assert (1, "head") not in mgr._overlay_playhead
    a = step(mgr, 1)
    assert a["y_head"] == pytest.approx(0.0) and a["l_elbow"] == pytest.approx(1.0)
    mgr.clear_ped_blend(1)
    assert 1 not in mgr._ped_blend
    assert not any(k[0] == 1 for k in mgr._overlay_playhead)


def test_clear_one_slot_with_fade_keeps_the_other(mgr):
    arm = mgr.register_transient("arm", const_frames(1.0, 40), loop=True)
    head = mgr.register_transient("head", const_frames(1.0, 40), loop=True)
    mgr.set_ped_blend(1, arm, blend_joints=ARMS, slot="arm")
    mgr.set_ped_blend(1, head, blend_joints={"y_head"}, slot="head")
    step(mgr, 1)
    mgr.clear_ped_blend(1, fade_out_s=0.5, slot="arm")
    a = step(mgr, 1)
    assert a["l_elbow"] == pytest.approx(0.8) and a["y_head"] == pytest.approx(1.0)
    a = step(mgr, 1, 4)
    assert a["l_elbow"] == pytest.approx(0.0) and a["y_head"] == pytest.approx(1.0)
    assert set(mgr._ped_blend[1]) == {"head"}


def test_forget_drops_all_slots(mgr):
    arm = mgr.register_transient("arm", const_frames(1.0, 40), loop=True, owner=1)
    head = mgr.register_transient("head", const_frames(1.0, 40), loop=True, owner=1)
    mgr.set_ped_blend(1, arm, blend_joints=ARMS, slot="arm")
    mgr.set_ped_blend(1, head, blend_joints={"y_head"}, slot="head")
    step(mgr, 1)
    mgr.forget(1)
    assert 1 not in mgr._ped_blend
    assert not any(k[0] == 1 for k in mgr._overlay_playhead)
    assert "arm" not in mgr.animations and "head" not in mgr.animations


def test_loop_from_plays_intro_once(mgr):
    frames = const_frames(0.0, 2) + const_frames(1.0, 2) + const_frames(2.0, 2)  # intro 0.2 s, loop tail 0.4 s
    ov = mgr.register_transient("intro", frames, loop=True, loop_from=2)
    mgr.set_ped_blend(7, ov, blend_joints=ARMS)
    assert mgr._overlay_playhead[(7, "arm")] == 0.0                # intro clips start at the head
    seq = [step(mgr, 7, dt=0.05)["l_elbow"] for _ in range(9)]
    assert seq == pytest.approx([0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0])  # wraps to loop_from, never back into the intro


def test_overlay_joints_flip_back_mid_fade_reramps_from_current_weight(mgr):
    ov = mgr.register_transient("ov", const_frames(1.0, 40), loop=True)
    mgr.set_ped_blend(1, ov, blend_joints=ARMS | {"waist"})
    assert step(mgr, 1)["waist"] == pytest.approx(1.0)
    mgr.set_overlay_joints(1, "arm", ARMS, fade_s=0.5)
    assert step(mgr, 1)["waist"] == pytest.approx(0.8)
    assert step(mgr, 1)["waist"] == pytest.approx(0.6)
    overlay = mgr._ped_blend[1]["arm"]
    assert overlay.joint_weight("waist") == pytest.approx(0.6)
    mgr.set_overlay_joints(1, "arm", ARMS | {"waist"}, fade_s=0.5)
    assert overlay.ramps["waist"][:2] == pytest.approx([0.6, 1.0])
    assert step(mgr, 1)["waist"] == pytest.approx(0.68)
    assert step(mgr, 1, 4)["waist"] == pytest.approx(1.0)
    assert "waist" not in overlay.ramps and "waist" in overlay.joints
    mgr.set_overlay_joints(1, "arm", ARMS, fade_s=0.5)
    step(mgr, 1, 5)
    assert "waist" not in overlay.joints and step(mgr, 1)["waist"] == pytest.approx(0.0)


def test_clear_slot_with_ramps_drops_it(mgr):
    ov = mgr.register_transient("ov", const_frames(1.0, 40), loop=True)
    mgr.set_ped_blend(1, ov, blend_joints=ARMS | {"waist"})
    step(mgr, 1)
    mgr.set_overlay_joints(1, "arm", ARMS, fade_s=0.5)
    step(mgr, 1)
    assert mgr._ped_blend[1]["arm"].ramps
    mgr.clear_ped_blend(1, slot="arm")
    assert 1 not in mgr._ped_blend
    a = step(mgr, 1)
    assert a["waist"] == pytest.approx(0.0) and a["l_elbow"] == pytest.approx(0.0)


def test_set_ped_blend_carry_ramps_preserves_ramps(mgr):
    first = mgr.register_transient("first", const_frames(1.0, 40), loop=True)
    second = mgr.register_transient("second", const_frames(1.0, 40), loop=True)
    mgr.set_ped_blend(1, first, blend_joints=ARMS | {"waist"})
    step(mgr, 1)
    mgr.set_overlay_joints(1, "arm", ARMS, fade_s=0.5)
    assert step(mgr, 1)["waist"] == pytest.approx(0.8)
    old = mgr._ped_blend[1]["arm"]
    mgr.set_ped_blend(1, second, blend_joints=ARMS, carry_ramps=True)
    new = mgr._ped_blend[1]["arm"]
    assert new is not old and new.anim is second
    assert new.joints == ARMS | {"waist"} and new.joints is not old.joints
    assert new.ramps == old.ramps and new.ramps["waist"] is not old.ramps["waist"]
    assert step(mgr, 1)["waist"] == pytest.approx(0.6)
    assert old.ramps["waist"][2] == pytest.approx(0.1)
    step(mgr, 1, 3)
    assert "waist" not in new.joints
    mgr.set_ped_blend(1, first, blend_joints=ARMS | {"waist"})
    assert mgr._ped_blend[1]["arm"].joints == ARMS | {"waist"} and not mgr._ped_blend[1]["arm"].ramps
