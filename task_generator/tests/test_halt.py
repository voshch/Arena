from __future__ import annotations

import math

import numpy as np
import pytest

from task_generator.simulators.human.gestures import BODY_HEIGHT
from task_generator.simulators.human.gestures.halt import HaltGesture
from task_generator.simulators.human.gestures.point import PointGesture
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.contract import ROS_JOINT_ORDER, SPINE_SEGMENTS, wrist_dofs

BODY = S.Body(BODY_HEIGHT)
TARGETS = ((3.0, 0.0, 1.0), (3.0, -2.0, 1.0), (0.0, 3.0, 1.0), (-1.0, -3.0, 1.0))


@pytest.fixture(scope="module")
def halt() -> HaltGesture:
    return HaltGesture()


def _hand(angles: dict, side: str, moving: bool) -> tuple[np.ndarray, np.ndarray]:
    """(finger direction, palm normal) as rendered: upright spine while walking."""
    pose = {**dict.fromkeys(ROS_JOINT_ORDER, 0.0), **angles}
    if moving:
        for seg in SPINE_SEGMENTS:
            pose[f"r_{seg}"] = pose[f"y_{seg}"] = pose[seg] = 0.0
    _, frames = S.fk(pose, BODY)
    hand = frames[f"{side}_hand"]
    return hand @ S.DOWN, hand @ np.array([0.0, -S.REFLECT[side], 0.0])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("hand", ["l", "r"])
@pytest.mark.parametrize("moving", [False, True])
def test_park_pushes_palm_at_target(halt: HaltGesture, target: tuple, hand: str, moving: bool) -> None:
    clip = halt.start(np.array(target), {"hand": hand, "moving": moving})
    fingers, palm = _hand(clip.frames[clip.hold_end]["angles"], clip.side, moving)
    bearing = S.unit(np.array([target[0], target[1], 0.0]))
    assert math.degrees(math.asin(fingers[2])) > 75.0, f"fingers {fingers} not raised"
    assert float(palm @ bearing) > 0.95, f"palm {palm} does not face the target bearing {bearing}"


@pytest.mark.parametrize("target", TARGETS)
def test_wrist_opens_from_and_closes_to_rest(halt: HaltGesture, target: tuple) -> None:
    clip = halt.start(np.array(target), {"hand": "r"})
    for name in wrist_dofs(clip.side):
        assert clip.frames[0]["angles"][name] == pytest.approx(0.0, abs=1e-9)
        assert clip.release[0]["angles"][name] == pytest.approx(clip.frames[clip.hold_end]["angles"][name], abs=0.05)
        assert clip.release[-1]["angles"][name] == pytest.approx(0.0, abs=1e-9)


def test_retarget_keeps_the_palm(halt: HaltGesture) -> None:
    first = halt.start(np.array([3.0, 0.0, 1.0]), {"hand": "r"})
    clip = halt.retarget(first.hold, np.array([3.0, -3.0, 1.0]), {"hand": "r"})
    for name in wrist_dofs("r"):
        assert clip.frames[0]["angles"][name] == pytest.approx(first.frames[first.hold_end]["angles"][name], abs=1e-6)
    fingers, palm = _hand(clip.frames[clip.hold_end]["angles"], "r", False)
    assert math.degrees(math.asin(fingers[2])) > 75.0
    assert float(palm @ S.unit(np.array([3.0, -3.0, 0.0]))) > 0.95


def test_halt_blends_wrists_point_does_not(halt: HaltGesture) -> None:
    point = PointGesture()
    for side in ("l", "r"):
        assert set(wrist_dofs(side)) <= halt.joints(side, False)
        assert not set(wrist_dofs(side)) & point.joints(side, False)
    clip = point.start(np.array([3.0, -2.0, 1.0]), {"hand": "r"})
    assert all(frame["angles"].get(name, 0.0) == 0.0 for frame in clip.frames for side in ("l", "r") for name in wrist_dofs(side))
