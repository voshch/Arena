"""Hand shapes as digit DOFs of the body clips that declare one, of point gestures and of hands holding a prop, every other hand holding its neutral stance."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence

import attrs
import numpy as np
from arena_simulation_setup.shared.body import forward, load_template, semantic_to_body

from arena_assets.utils.geom import unit

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


def clip_pose(clips: Iterable[tuple[str, float]], now: float, base: Mapping[str, float] | None = None) -> dict[str, float]:
    """Digit DOFs of the playing clips, (name, weight) pairs, at sim time `now` laid on `base` (held props), none without either."""
    pose: dict[str, float] = dict(base or {})
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


SIDES = {"l": "left", "r": "right"}
SKIN_PALM = 0.022  # palm surface off the hand bones, median of the shipped actor meshes
SKIN_FINGER = 0.011  # finger radius, median of the shipped actor meshes
TIP_REACH = 0.8
GRIP_BACK = 0.011
PALM_MID = 0.04  # furthest a steep grip's centre moves back from the knuckles: mid-palm
CURL_CAP = (1.2, 1.6, 1.6, 1.6, 1.6)
CURL_STEPS = 161


@attrs.frozen
class Grip:
    """How a hand holds a prop: the radius (m) of the cylinder it closes on and the cylinder's angle (deg) in the palm, 0 across
    the palm (power grip on a handle or pole), toward the fingers for a diagonal grip (a long cane leaving between the
    fingertips and the little finger). `curls` (thumb to pinky) overrides the solved finger curls: grasp() closes the template hand, so an
    actor whose fingers are longer than the template's needs its own (solved on its rendered skeleton) to stop on the handle instead of in it."""

    radius: float
    angle: float = 0.0
    curls: tuple[float, ...] | None = None
    wrap: bool = False  # the thumb closes over the fingers (a power grip round a pole), see grasp_on


@attrs.frozen(eq=False)
class PalmFrame:
    """A hand's grip geometry from its joint positions: knuckle centre, unit axes across the palm (index to pinky), along the fingers and out of the palm."""

    knuckles: np.ndarray
    across: np.ndarray
    along: np.ndarray
    palm: np.ndarray
    half_width: float = 0.0

    def grip(self, radius: float, angle: float = 0.0) -> np.ndarray:
        """Centre of a cylinder of `radius` lying in the palm `angle` deg from across it, GRIP_BACK behind the finger root it reaches furthest
        toward the fingers: across the palm just behind the knuckles, a diagonal one further back so it still crosses every finger's base,
        a steep one (a long cane held along the index finger) through mid-palm."""
        back = GRIP_BACK + min(abs(math.tan(math.radians(angle))) * self.half_width, PALM_MID)
        return self.knuckles - back * self.along + (SKIN_PALM + radius) * self.palm

    def axis(self, angle: float = 0.0) -> np.ndarray:
        """Unit axis of a grip cylinder `angle` deg from across the palm (toward the little finger) toward the fingers."""
        a = math.radians(angle)
        return math.cos(a) * self.across + math.sin(a) * self.along


def palm_frame(positions: Mapping[str, np.ndarray], side: str) -> PalmFrame:
    """PalmFrame of a posed hand from body-frame or world joint positions (`forward`), whatever its digits' curls."""
    hand = SIDES[side]
    wrist = np.asarray(positions[f"{hand}_wrist"], dtype=float)
    roots = {digit: np.asarray(positions[f"{hand}_{digit}1"], dtype=float) for digit in DIGITS[1:]}
    knuckles = np.mean(list(roots.values()), axis=0)
    across = unit(roots["pinky"] - roots["index"])
    along = knuckles - wrist
    along = unit(along - (along @ across) * across)
    palm = np.cross(along, across) if side == "r" else np.cross(across, along)
    half_width = max(abs(float((root - knuckles) @ across)) for root in roots.values())
    return PalmFrame(knuckles=knuckles, across=across, along=along, palm=palm, half_width=half_width)


def _digit_points(positions: Mapping[str, np.ndarray], side: str, digit: str) -> np.ndarray:
    """Joint centres of a digit plus its extrapolated tip, sampled densely along the bones."""
    hand = SIDES[side]
    joints = [np.asarray(positions[f"{hand}_{digit}{k}"], dtype=float) for k in (1, 2, 3)]
    joints.append(joints[2] + TIP_REACH * (joints[2] - joints[1]))
    return np.concatenate([np.linspace(a, b, 8) for a, b in zip(joints[:-1], joints[1:], strict=True)])


def _clearance(points: np.ndarray, centre: np.ndarray, axis: np.ndarray, radius: float) -> np.ndarray:
    rel = points - centre
    rel = rel - np.outer(rel @ axis, axis)
    return np.linalg.norm(rel, axis=1) - radius


@functools.cache
def grasp(side: str, radius: float, angle: float = 0.0, wrap: bool = False) -> tuple[float, ...]:
    """Curls (thumb to pinky) of the template hand closing on a cylinder of `radius` m in its palm, `angle` deg from across it
    (PalmFrame.axis), see grasp_on."""
    template = load_template()
    return grasp_on(lambda angles: forward(template, semantic_to_body(angles))[0], side, radius, angle, wrap)


def grasp_on(positions_of: Callable[[Mapping[str, float]], Mapping[str, np.ndarray]], side: str, radius: float, angle: float = 0.0, wrap: bool = False) -> tuple[float, ...]:
    """Curls (thumb to pinky) of a hand closing on a cylinder of `radius` m in its palm, `angle` deg from across it (PalmFrame.axis), on the
    body `positions_of` poses from semantic angles (joint positions, e.g. shared.actor_pose.ActorPose(...).positions for a character's own
    hand): each finger curls until it first touches the surface; the thumb, whose curl sweeps it across the grip, settles where its tip keeps
    the most room from the grip and the closed index finger, or with `wrap` closes over the grip like the fingers, its last bone stopping at the
    first contact with the grip or the closed index finger (a power grip round a pole: the thumb over the fingers)."""
    steps = np.linspace(0.0, max(CURL_CAP), CURL_STEPS)
    poses = [(float(c), positions_of({f"{side}_curl_{digit}": float(c) for digit in DIGITS})) for c in steps]
    frame = palm_frame(poses[0][1], side)
    centre, axis = frame.grip(radius, angle), frame.axis(angle)

    def close(digit: str, cap: float, blocked: Callable[[np.ndarray], bool]) -> tuple[float, np.ndarray]:
        best, points = 0.0, _digit_points(poses[0][1], side, digit)
        for c, positions in poses:
            here = _digit_points(positions, side, digit)
            if c > cap + 1e-9 or blocked(here):
                break
            best, points = c, here
        return best, points

    curls, index = {}, np.zeros((0, 3))
    for digit, cap in zip(DIGITS[1:], CURL_CAP[1:], strict=True):
        curls[digit], points = close(digit, cap, lambda pts: bool(_clearance(pts, centre, axis, radius).min() < SKIN_FINGER))
        if digit == "index":
            index = points
    room = []
    for c, positions in poses:
        if c > CURL_CAP[0] + 1e-9:
            break
        tip = _digit_points(positions, side, "thumb")[-8:]  # the last bone to the tip: the thumb's root lies against the grip anyway
        to_grip = float(_clearance(tip, centre, axis, radius).min()) - SKIN_FINGER
        to_index = float(np.linalg.norm(tip[:, None] - index[None], axis=2).min()) - 2.0 * SKIN_FINGER
        room.append((round(min(to_grip, to_index), 4), c))
    curls["thumb"] = max(room)[1]
    if wrap:
        curls["thumb"] = 0.0
        for c, positions in poses:
            tip = _digit_points(positions, side, "thumb")[-8:]  # its root lies along a diagonal grip anyway
            if c > CURL_CAP[0] + 1e-9 or _clearance(tip, centre, axis, radius).min() < SKIN_FINGER or float(np.linalg.norm(tip[:, None] - index[None], axis=2).min()) < 2.0 * SKIN_FINGER:
                break
            curls["thumb"] = c
    return tuple(round(curls[digit], 3) for digit in DIGITS)


def held_pose(hold: Mapping[str, Grip]) -> dict[str, float]:
    """Digit DOFs of the hands that hold a prop, {side: Grip}."""
    return {name: value for side, g in hold.items() for name, value in shape(side, g.curls or grasp(side, round(g.radius, 4), round(g.angle, 1), g.wrap)).items()}
