"""Hand shapes as digit DOFs of the body clips that declare one and of point gestures, every other hand holding its neutral stance."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
COUNT_CLIP = "count_fingers"
COUNT_STEP_S = 0.8
COUNT_JOINTS = ("r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow", "r_r_wrist", "r_wrist")


def shape(side: str, curls: Sequence[float], spread: float = 0.0) -> dict[str, float]:
    """Digit DOFs of one hand from five curls in thumb to pinky order and the finger spread."""
    return {**{f"{side}_curl_{digit}": curl for digit, curl in zip(DIGITS, curls, strict=True)}, f"{side}_spread": spread}


GRIP = (0.7, 1.2, 1.3, 1.35, 1.4)
CUP = (0.3, 0.5, 0.55, 0.6, 0.65)
FIST = (0.9, 1.5, 1.5, 1.5, 1.5)
RAISED = 0.1
POINT = (0.9, RAISED, 1.5, 1.5, 1.5)
POINT_CLIP = "point"
COUNT_EASE_S = 0.18


def blend(base: Mapping[str, float], over: Mapping[str, float], weight: float) -> dict[str, float]:
    """`over` laid on `base` at weight, digits missing from `over` kept."""
    w = max(0.0, min(1.0, weight))
    return {**base, **{name: (1.0 - w) * base.get(name, 0.0) + w * value for name, value in over.items()}}


def counted(n: int) -> tuple[float, ...]:
    """Curls of a hand showing n fingers, index first and thumb last, a fist at 0."""
    order = ("index", "middle", "ring", "pinky", "thumb")
    up = set(order[: max(0, min(5, n))])
    return tuple(RAISED if digit in up else FIST[i] for i, digit in enumerate(DIGITS))


def count_hand(t: float) -> dict[str, float]:
    """Right hand counting 0 to 5 at COUNT_STEP_S per finger, easing into each count, spread open at five."""
    step = int(t / COUNT_STEP_S)
    n = min(step % 7, 5)
    before = min((step - 1) % 7, 5)
    a = min(1.0, (t - step * COUNT_STEP_S) / COUNT_EASE_S)
    a = a * a * (3.0 - 2.0 * a)
    return blend(shape("r", counted(before), 0.25 if before == 5 else 0.0), shape("r", counted(n), 0.25 if n == 5 else 0.0), a)


CLIP_HANDS: dict[str, Callable[[float], dict[str, float]]] = {
    "shake_hand": lambda t: shape("r", GRIP),
    "hug": lambda t: {**shape("l", CUP, 0.2), **shape("r", CUP, 0.2)},
    COUNT_CLIP: count_hand,
    f"{POINT_CLIP}_l": lambda t: shape("l", POINT),
    f"{POINT_CLIP}_r": lambda t: shape("r", POINT),
}


def clip_hands(clip: str) -> Callable[[float], dict[str, float]] | None:
    """Hand shape of a body clip by name, versioned names like shake_hand_v3 included."""
    if clip in CLIP_HANDS:
        return CLIP_HANDS[clip]
    base = clip.rsplit("_v", 1)[0] if "_v" in clip else clip
    return CLIP_HANDS.get(base)


def clip_pose(clips: Iterable[tuple[str, float]], now: float) -> dict[str, float]:
    """Digit DOFs of the playing clips, (name, weight) pairs, at sim time `now`, none without a clip that declares a hand shape."""
    pose: dict[str, float] = {}
    for clip, weight in clips:
        hand = clip_hands(clip)
        if hand is not None:
            pose = blend(pose, hand(now), weight)
    return pose


def count_frames(fps: float) -> list[dict]:
    """Clip frames of the counting gesture over one count cycle: right forearm raised in front of the chest, palm out."""
    n = max(2, round(fps * 7 * COUNT_STEP_S))
    frames = []
    for i in range(n):
        sway = 0.03 * math.sin(2.0 * math.pi * i / n)
        angles = {"r_y_shoulder": 0.25, "r_p_shoulder": 0.55 + sway, "r_r_shoulder": 0.0, "r_elbow": 1.9, "r_r_wrist": 1.2, "r_wrist": -0.3}
        frames.append({"t": i / fps, "angles": angles, "root_xy_yaw": [0.0, 0.0, 0.0], "animation_state": 0})
    return frames
