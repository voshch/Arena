from __future__ import annotations

import numpy as np
import pytest
from arena_simulation_setup.shared.body import SEMANTIC_DOFS, forward, load_template, semantic_to_body

from arena_runtime.humans.animation_manager import AnimationManager
from arena_runtime.humans.clips import NoClips
from arena_runtime.humans.hands import COUNT_CLIP, COUNT_EASE_S, COUNT_JOINTS, COUNT_STEP_S, FIST, GRIP, RAISED, SKIN_FINGER, Grip, _clearance, _digit_points, blend, clip_hands, clip_pose, count_hand, counted, grasp, held_pose, palm_frame, shape


class StubLogger:
    def info(self, message: str, /) -> None:
        del message

    def warning(self, message: str, /) -> None:
        del message


def test_hand_shapes_name_semantic_digit_dofs():
    hands = {**shape("l", GRIP), **shape("r", GRIP)}
    assert set(hands) <= set(SEMANTIC_DOFS)
    assert len(hands) == 12


def test_no_clip_or_a_clip_without_a_hand_shape_moves_no_digit():
    assert clip_pose([], 1.0) == {}
    assert clip_pose([("beckon", 1.0), ("", 1.0)], 1.0) == {}


def test_a_clip_hand_shape_fades_with_its_weight_from_neutral():
    assert clip_pose([("shake_hand", 0.5)], 0.0)["r_curl_index"] == pytest.approx(0.5 * GRIP[1])
    assert clip_pose([("shake_hand", 1.0)], 0.0) == shape("r", GRIP)
    assert "l_curl_index" not in clip_pose([("shake_hand", 1.0)], 0.0)


def test_counted_raises_index_first_and_thumb_last():
    assert counted(0) == FIST
    assert counted(1)[1] == RAISED
    assert counted(1)[0] == FIST[0]
    assert counted(4)[0] == FIST[0]
    assert counted(5) == (RAISED,) * 5


def test_count_hand_steps_through_zero_to_five_then_holds_five():
    settled = COUNT_EASE_S + 0.01
    shown = [sum(1 for digit in ("thumb", "index", "middle", "ring", "pinky") if count_hand(n * COUNT_STEP_S + settled)[f"r_curl_{digit}"] == pytest.approx(RAISED)) for n in range(7)]
    assert shown == [0, 1, 2, 3, 4, 5, 5]
    assert count_hand(5.5 * COUNT_STEP_S)["r_spread"] > 0.0


def test_count_eases_between_counts():
    mid = count_hand(COUNT_STEP_S + COUNT_EASE_S / 2.0)["r_curl_index"]
    assert RAISED < mid < FIST[1]


@pytest.mark.parametrize("clip", ["shake_hand", "shake_hand_v3"])
def test_handshake_grips_the_right_hand(clip: str) -> None:
    hand = clip_hands(clip)
    assert hand is not None
    assert hand(0.0) == shape("r", GRIP)


def test_unknown_clip_has_no_hand_shape():
    assert clip_hands("beckon") is None


def test_blend_weights_the_overlay_and_keeps_other_digits():
    base = shape("l", GRIP)
    out = blend({**base, "r_curl_index": 0.2}, shape("r", GRIP), 0.5)
    assert out["r_curl_index"] == pytest.approx(0.1 + 0.5 * GRIP[1])
    assert out["l_curl_index"] == GRIP[1]
    assert blend({}, shape("r", GRIP), 2.0)["r_curl_index"] == GRIP[1]


def test_manager_ships_the_counting_clip_on_the_right_arm():
    manager = AnimationManager(logger=StubLogger(), fps=20.0, clips=NoClips())
    anim = manager.clip(COUNT_CLIP)
    assert anim is not None
    assert anim.loop
    assert anim.joints == COUNT_JOINTS
    assert anim.duration == pytest.approx(7 * COUNT_STEP_S)
    assert set(anim.frames[0]["angles"]) == set(COUNT_JOINTS)


@pytest.mark.parametrize("side", ["l", "r"])
def test_a_point_raises_the_index_of_its_arm_and_curls_the_rest(side: str) -> None:
    pose = clip_pose([(f"point_{side}", 1.0)], 0.0)
    assert pose[f"{side}_curl_index"] == pytest.approx(RAISED)
    assert all(pose[f"{side}_curl_{digit}"] > 1.0 for digit in ("middle", "ring", "pinky"))
    assert pose[f"{side}_curl_thumb"] > 0.5
    assert not any(name.startswith("l_" if side == "r" else "r_") for name in pose)


def _posed(side: str, curls: tuple[float, ...]) -> dict[str, np.ndarray]:
    return forward(load_template(), semantic_to_body(shape(side, curls)))[0]


@pytest.mark.parametrize("side", ["l", "r"])
def test_palm_frame_faces_where_the_fingers_curl(side: str) -> None:
    flat, curled = _posed(side, (0.0,) * 5), _posed(side, (0.0, 0.8, 0.8, 0.8, 0.8))
    hand = {"l": "left", "r": "right"}[side]
    frame = palm_frame(flat, side)
    assert (curled[f"{hand}_middle3"] - flat[f"{hand}_middle3"]) @ frame.palm > 0.01
    assert abs(frame.across @ frame.along) < 1e-9 and abs(frame.palm @ frame.across) < 1e-9
    assert np.linalg.norm(palm_frame(curled, side).grip(0.0125) - frame.grip(0.0125)) < 1e-9


@pytest.mark.parametrize("side", ["l", "r"])
def test_grasp_closes_tighter_on_thinner_grips_and_clears_the_grip(side: str) -> None:
    thin, thick = grasp(side, 0.0125), grasp(side, 0.025)
    assert all(a > b for a, b in zip(thin[1:], thick[1:], strict=True))
    assert grasp("l", 0.0125) == grasp("r", 0.0125)
    frame = palm_frame(_posed(side, (0.0,) * 5), side)
    posed = _posed(side, thin)
    for digit in ("index", "middle", "ring", "pinky"):
        assert _clearance(_digit_points(posed, side, digit), frame.grip(0.0125), frame.across, 0.0125).min() >= SKIN_FINGER - 1e-3
    assert _clearance(_digit_points(posed, side, "thumb"), frame.grip(0.0125), frame.across, 0.0125).min() > 0.0


def test_held_pose_grips_only_the_holding_hands_and_a_clip_overrides_it() -> None:
    held = held_pose({"r": Grip(0.0125)})
    assert held == shape("r", grasp("r", 0.0125))
    assert held_pose({"r": Grip(0.0125, 45.0)}) == shape("r", grasp("r", 0.0125, 45.0))
    assert clip_pose([], 0.0, held) == held
    assert clip_pose([("shake_hand", 1.0)], 0.0, held) == shape("r", GRIP)
    assert held_pose({}) == {}
    assert held_pose({"l": Grip(0.008, 0.0, (0.4, 0.9, 1.0, 1.1, 1.2))}) == shape("l", (0.4, 0.9, 1.0, 1.1, 1.2))


def test_a_diagonal_grip_closes_the_fingers_around_the_cane_further_than_across_the_palm() -> None:
    across, diagonal = grasp("r", 0.0125), grasp("r", 0.0125, 45.0)
    assert diagonal[1] > across[1]
    assert all(0.0 <= c <= 1.6 for c in diagonal)


def test_grasp_on_the_template_is_grasp_and_wrap_closes_the_thumb_over_the_fingers():
    from arena_simulation_setup.shared.body import forward, load_template, semantic_to_body
    from arena_runtime.humans.hands import grasp_on

    template = load_template()
    pose = lambda angles: forward(template, semantic_to_body(angles))[0]  # noqa: E731
    assert grasp_on(pose, "r", 0.0125) == grasp("r", 0.0125)
    plain, wrapped = grasp_on(pose, "l", 0.008, 42.0), grasp_on(pose, "l", 0.008, 42.0, wrap=True)
    assert wrapped[1:] == plain[1:]  # the fingers close the same way
    assert 0.0 < wrapped[0] <= 1.2
