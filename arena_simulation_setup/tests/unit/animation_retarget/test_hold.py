"""--hold: arms re-solved per clip frame so the fist closes on a prop, on Arena's template body (synthetic walking clip)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from arena_simulation_setup.adapters.arena_animation_retarget import hold as H
from arena_simulation_setup.adapters.arena_animation_retarget.contract import WIRE
from arena_simulation_setup.adapters.arena_animation_retarget.retarget import Clip
from arena_simulation_setup.shared.actor_pose import TemplatePose
from arena_simulation_setup.shared.body import BODY_JOINTS, DEFAULT_STANCE, semantic_to_body

FPS = 20.0


def _walk(T: int = 40, period: float = 1.1, with_body: bool = False) -> Clip:
    """Legs swinging at one stride per `period` s (right hip leads at t = 0), arms hanging still."""
    t = np.arange(T) / FPS
    w = 2 * math.pi / period
    angles = {k: np.zeros(T) for k in WIRE}
    angles["r_r_hip"], angles["l_r_hip"] = 0.4 * np.sin(w * t + math.pi / 2), -0.4 * np.sin(w * t + math.pi / 2)
    angles["r_knee"], angles["l_knee"] = -0.3 * (1 + np.sin(w * t)), -0.3 * (1 - np.sin(w * t))
    body = None
    if with_body:
        body = np.stack([[np.asarray(semantic_to_body({k: float(v[i]) for k, v in angles.items()}, DEFAULT_STANCE)[j], dtype=float) for j in BODY_JOINTS] for i in range(T)])
    return Clip(angles=angles, t=t, root_xy_yaw=np.zeros((T, 3)), root_z=np.full(T, 0.9), root_quat=np.tile([0.0, 0, 0, 1], (T, 1)), body_rotations=body, body_pelvis=np.zeros((T, 3)) if with_body else None)


def _frames(clip: Clip) -> list[dict[str, float]]:
    return [{k: float(v[i]) for k, v in clip.angles.items()} for i in range(len(clip.t))]


def test_parse_takes_overrides_and_refuses_bad_specs() -> None:
    h = H.Hold.parse("iv:gz=1.0,angle=40,side=r")
    assert (h.task, h.params["gz"], h.params["angle"], h.sides) == ("iv", 1.0, 40.0, ("r",))
    assert H.Hold.parse("rollator").sides == ("l", "r")
    for bad in ("teapot", "iv:gz", "iv:side=lr", "frame:side=l"):
        with pytest.raises(ValueError):
            H.Hold.parse(bad)


def test_iv_pole_held_thumb_up_with_the_elbow_bent() -> None:
    clip, rep = H.apply(_walk(), H.Hold.parse("iv:starts=4"))
    assert rep["grip_mm"][1] < 3.0 and rep["axis_deg"][1] < 2.0
    p = H.TASKS["iv"]
    for a in _frames(clip)[::5]:
        f = H.HANDS.palm_frame(TemplatePose().positions(a), "l")
        assert f.axis(p["angle"]) @ np.array([0.0, 0.0, -1.0]) > 0.99  # the little finger down: thumb up
        assert p["elbow"] - 0.05 <= a["l_elbow"] <= p["elbow_hi"] + 0.05
    assert len(rep["curls"]["l"]) == 5 and clip.angles["l_curl_index"][0] == rep["curls"]["l"][1]
    assert H.annotation(rep)["hold"]["curls"] == rep["curls"]


def test_rollator_grips_both_hands_level() -> None:
    # the default grips fit the character they were measured for (RollatorFitted under casual_male_caucasian_old): the template's own spot
    clip, rep = H.apply(_walk(), H.Hold.parse("rollator:gx=0.30,gz=0.92,starts=4"))
    assert rep["sides"] == "lr" and rep["grip_mm"][1] < 3.0
    q = TemplatePose().positions(_frames(clip)[10])
    gl, gr = (H.HANDS.palm_frame(q, s).grip(0.018, -35.0) for s in "lr")
    assert gl[2] == pytest.approx(gr[2], abs=0.003) and gl[1] - gr[1] == pytest.approx(0.444, abs=0.005)


def test_the_cane_sweeps_with_the_steps() -> None:
    clip, rep = H.apply(_walk(T=60), H.Hold.parse("cane:starts=4"))
    assert rep["grip_mm"][0] < 5.0
    lead, side = [], []
    for a in _frames(clip):
        q = TemplatePose().positions(a)
        lead.append(q["right_ankle"][0] - q["left_ankle"][0])
        ax = H.HANDS.palm_frame(q, "r").axis(-60.0)
        side.append((ax if ax[2] < 0 else -ax)[1])  # the cane's direction from the fist to the tip, left component
    assert np.corrcoef(lead, side)[0, 1] > 0.8  # tip on the left while the right foot leads (two-point touch)


def test_the_left_hand_cane_sweeps_the_mirrored_way() -> None:
    clip, rep = H.apply(_walk(T=60), H.Hold.parse("cane:side=l,starts=4"))
    assert rep["grip_mm"][0] < 5.0
    lead, side = [], []
    for a in _frames(clip):
        q = TemplatePose().positions(a)
        lead.append(q["right_ankle"][0] - q["left_ankle"][0])
        ax = H.HANDS.palm_frame(q, "l").axis(-60.0)
        side.append((ax if ax[2] < 0 else -ax)[1])
    assert np.corrcoef(lead, side)[0, 1] < -0.8  # tip on the right while the right foot leads


def test_a_held_clip_drops_the_body_track_so_players_pose_its_angles() -> None:
    clip, _ = H.apply(_walk(T=12, with_body=True), H.Hold.parse("iv:starts=2"))
    assert clip.body_rotations is None and clip.body_pelvis is None


def test_misspelled_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        H.Hold.parse("rollator:widht=0.5")
    assert H.Hold.parse("cane:wrap=1,starts=3").params["starts"] == 3.0
