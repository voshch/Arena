"""Body frame pose of a ped sitting on a seat or lying on its back on a bed.

The body frame has its origin at the soles under the pelvis and its z axis up the body, renderers plant the lowest sole
point at body height 0.
"""

from __future__ import annotations

import functools
import math
import typing
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import yaml

from arena_assets.utils import human
from arena_assets.utils.human import Legs, PelvisClearance, Proportions
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.profile import JOINT_NAMES

ANNOTATION_FILE = "annotation.yaml"
SIT_CLIP = "sit"
DEFAULT_SEAT_HEIGHT = 0.45  # m, a seat without a known surface
SQUAT_CLIP = "squat"
SEATED_HIP_FLEXION = 0.9  # rad, mean of l_r_hip and r_r_hip a seated pose reaches at least
SHIN_GAP = 0.05  # m, backs of the calves ahead of the seat's front edge
PELVIS_MARGIN = 0.08  # m, pelvis behind the seat's front edge at least
RECLINE_S = 0.6  # s, torso back to vertical once the pelvis is down
LEAN_S = 0.25  # s, torso forward again before rising
SEATED_ARMS = {"y_shoulder": 0.25, "p_shoulder": 0.3, "elbow": 0.8, "r_wrist": 1.4}  # rad, left arm, the right mirrors y_shoulder
TORSO_JOINTS = ("r_waist", "y_waist", "waist", "r_spine", "y_spine", "spine", "r_chest", "y_chest", "chest")
SETTLE_S = 0.4  # s, the body moves between where it stands and where it rests
LIE_S = 1.0  # s, the body moves between standing beside a bed and lying on it
_FACING = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


@attrs.frozen
class Support:
    """Furniture a ped rests on, its surface height above the ped's floor, and how far ahead of and behind the parked point its seat ends, in m."""

    kind: typing.Literal["sit", "lie"]
    height: float
    front: float | None = None
    back: float | None = None


TYPICAL_PROPORTIONS = Proportions(shoulders=0.46, shoulder_width=0.18, hip_width=0.105, upperarm=0.255, forearm=0.25, hand=0.195, head=0.2)


@attrs.frozen
class BodyMeasurements:
    """The asset annotation's measurements of a character."""

    clearance: PelvisClearance
    legs: Legs
    proportions: Proportions = TYPICAL_PROPORTIONS


TYPICAL_SEATED = (
    (0.0, 0.127),
    (0.1745, 0.134),
    (0.3491, 0.139),
    (0.5236, 0.143),
    (0.6981, 0.142),
    (0.8727, 0.14),
    (1.0472, 0.134),
    (1.2217, 0.125),
    (1.3963, 0.115),
    (1.5708, 0.105),
    (1.7453, 0.096),
    (1.9199, 0.089),
    (2.0944, 0.086),
    (2.2689, 0.083),
    (2.4435, 0.081),
)  # rad, m
TYPICAL_BODY = BodyMeasurements(clearance=PelvisClearance(sit=0.105, lie=0.119, seated=TYPICAL_SEATED), legs=Legs(hip=0.015, thigh=0.36, shin=0.375, ankle=0.08))
SEAT_FIT_STEPS = 4  # fixed-point steps between the seated pelvis height and the hip flexion it gives


def _read_skin(path: Path) -> human.Skin | None:
    """The skin in a COLLADA file, None when it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return human.read_skin(path)
    except (ET.ParseError, KeyError, ValueError):
        return None


def load_body(model_uri: str) -> BodyMeasurements | None:
    """Measurements in the annotation beside an actor SDF, else measured off its skin (meshes/<name>.dae), None without either.

    Proportions the annotation lacks come off the skin too, typical ones when it has none or an unknown rig.
    """
    if not model_uri:
        return None
    sdf = Path(model_uri)
    path = sdf.with_name(ANNOTATION_FILE)
    annotation = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    skin_path = sdf.parent / "meshes" / f"{sdf.stem}.dae"
    skin = _read_skin(skin_path) if not (annotation.get("pelvis_clearance") and annotation.get("legs") and annotation.get("proportions")) else None
    if annotation.get("proportions"):
        proportions = Proportions(**annotation["proportions"])
    else:
        try:
            proportions = human.proportions(skin) if skin is not None else TYPICAL_PROPORTIONS
        except (KeyError, ValueError):
            proportions = TYPICAL_PROPORTIONS
    if annotation.get("pelvis_clearance") and annotation.get("legs"):
        return BodyMeasurements(clearance=PelvisClearance(**annotation["pelvis_clearance"]), legs=Legs(**annotation["legs"]), proportions=proportions)
    forward = _FACING.get(str(annotation.get("face", "+x")))
    if forward is None or skin is None:
        return None
    return BodyMeasurements(clearance=human.pelvis_clearance(skin, np.array(forward)), legs=human.legs(skin), proportions=proportions)


def _skeleton(legs: Legs, proportions: Proportions | None = None) -> S.Body:
    """The pointing skeleton with the character's legs and, given them, its trunk and arms (the template's otherwise)."""
    body = S.Body()
    body.thigh, body.tibia, body.hip_height = legs.thigh, legs.shin, legs.thigh + legs.shin
    if proportions is not None:
        body.torso_height, body.spine_segment = proportions.shoulders, proportions.shoulders / 3
        body.neck_length, body.neck_shoulder, body.waist_length = proportions.head, proportions.shoulder_width, 2 * proportions.hip_width
        body.upperarm, body.forearm = proportions.upperarm, proportions.forearm
    return body


def _leg_fk(angles: Mapping[str, float], legs: Legs) -> tuple[dict, S.Body]:
    body = _skeleton(legs)
    pos, _ = S.fk(dict.fromkeys(JOINT_NAMES, 0.0) | dict(angles), body)
    return pos, body


def _planted(pos: dict, legs: Legs) -> float:
    return float(legs.hip + pos["pelvis"][2] - min(pos["l_ankle"][2], pos["r_ankle"][2]) + legs.ankle)


def planted_pelvis(angles: Mapping[str, float], legs: Legs) -> float:
    """Height of the pelvis joint above the soles in the pose the angles give."""
    return _planted(_leg_fk(angles, legs)[0], legs)


def ankle_reach(angles: Mapping[str, float], legs: Legs) -> float:
    """How far ahead of the pelvis the ankle joints are in the pose the angles give."""
    pos, _ = _leg_fk(angles, legs)
    return float(0.5 * (pos["l_ankle"][0] + pos["r_ankle"][0]) - pos["pelvis"][0])


@attrs.frozen
class SquatProfile:
    """Per frame of a squat clip: the planted pelvis height and the hip flexion."""

    pelvis: tuple[float, ...]
    hip: tuple[float, ...]

    def drop_frame(self, pelvis: float) -> float:
        """Fractional frame where the pelvis is first down at the target, the last when the clip never gets that low."""
        return _crossing([-p for p in self.pelvis], -pelvis)

    def seated_frame(self) -> float:
        """Fractional frame where the hips first reach SEATED_HIP_FLEXION."""
        return _crossing(list(self.hip), SEATED_HIP_FLEXION)


def _crossing(values: Sequence[float], level: float) -> float:
    """Fractional index where a series first reaches level, linear between frames, its last index when it never does."""
    if values[0] >= level:
        return 0.0
    for k in range(1, len(values)):
        if values[k] >= level:
            return k - 1 + (level - values[k - 1]) / (values[k] - values[k - 1])
    return float(len(values) - 1)


def frames_until(frames: Sequence[Mapping], position: float) -> list[dict]:
    """The frames up to a fractional frame, the last one blended linearly between its neighbors."""
    k = int(math.floor(position))
    out = [dict(frame) for frame in frames[: k + 1]]
    alpha = position - k
    if alpha > 1e-6 and k + 1 < len(frames):
        a, b = frames[k]["angles"], frames[k + 1]["angles"]
        out.append({**frames[k + 1], "angles": {j: (1.0 - alpha) * a[j] + alpha * b.get(j, a[j]) for j in a}})
    return out


@functools.cache
def _profile(frames: tuple[tuple[tuple[str, float], ...], ...], legs: Legs) -> SquatProfile:
    angles = [dict(frame) for frame in frames]
    return SquatProfile(
        pelvis=tuple(planted_pelvis(a, legs) for a in angles),
        hip=tuple(hip_flexion(a) for a in angles),
    )


def squat_profile(frames: Sequence[Mapping], legs: Legs) -> SquatProfile:
    """The profile of a squat clip's frames ({"angles": ...}) for a character's legs, cached."""
    return _profile(tuple(tuple(sorted(f["angles"].items())) for f in frames), legs)


def _clamped_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def leg_ik(ahead: float, below: float, legs: Legs) -> tuple[float, float]:
    """Hip flexion and knee angle in wire signs of a leg whose ankle joint is ahead of and below its hip joint, knee forward."""
    reach = max(abs(legs.thigh - legs.shin) + 1e-9, min(math.hypot(ahead, below), legs.thigh + legs.shin - 1e-9))
    bend = math.pi - _clamped_acos((legs.thigh**2 + legs.shin**2 - reach**2) / (2.0 * legs.thigh * legs.shin))
    hip = math.atan2(ahead, below) + _clamped_acos((legs.thigh**2 + reach**2 - legs.shin**2) / (2.0 * legs.thigh * reach))
    return hip, -bend


def seated_target(body: BodyMeasurements, height: float) -> float:
    """Pelvis height of a body seated on a surface at height, its seated clearance taken at the hip flexion that height gives."""
    target = height + body.clearance.sit
    for _ in range(SEAT_FIT_STEPS):
        seated = seated_legs(target, body.legs)
        target = height + body.clearance.sit_at(seated[0] if seated is not None else math.pi / 2.0)
    return target


def hip_flexion(angles: Mapping[str, float]) -> float:
    """Mean flexion of both hips in rad."""
    return 0.5 * (angles.get("l_r_hip", 0.0) + angles.get("r_r_hip", 0.0))


def seated_legs(pelvis: float, legs: Legs) -> tuple[float, float] | None:
    """Hip flexion and the thighs' forward reach of a seated pose with the pelvis at height pelvis and vertical shins on the floor, None when the shins cannot reach it."""
    drop = pelvis - legs.hip - legs.ankle - legs.shin
    if drop > legs.thigh:
        return None
    drop = max(drop, -legs.thigh)
    reach = math.sqrt(legs.thigh**2 - drop**2)
    return math.atan2(reach, drop), reach


def _with_legs(angles: Mapping[str, float], hip: float, knee: float) -> dict[str, float]:
    out = dict(angles)
    for side in ("l", "r"):
        out |= {f"{side}_y_hip": 0.0, f"{side}_p_hip": 0.0, f"{side}_r_hip": hip, f"{side}_knee": knee, f"{side}_y_ankle": 0.0, f"{side}_ankle": -(hip + knee)}
    return out


def _with_arms(angles: Mapping[str, float], weight: float) -> dict[str, float]:
    return dict(angles) | {f"{side}_{joint}": weight * (-value if side == "r" and joint == "y_shoulder" else value) for side in ("l", "r") for joint, value in SEATED_ARMS.items()}


def _with_torso(angles: Mapping[str, float], torso: Mapping[str, float], weight: float) -> dict[str, float]:
    return dict(angles) | {joint: weight * torso.get(joint, 0.0) for joint in TORSO_JOINTS}


def _smooth(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def _timed(angles: Sequence[dict[str, float]], fps: float) -> tuple[dict, ...]:
    return tuple({"t": k / fps, "angles": a} for k, a in enumerate(angles))


@attrs.frozen
class SeatPlan:
    """A sitter's motion: frames down onto the seat ending on the parked pose, frames back up, and where it stands.

    feet is how far ahead of the parked point the feet stay planted, None when they dangle; depth is where the pelvis ends,
    stand where it starts and ends standing; fold is the seated hip flexion; down is how long the pelvis takes to come down.
    """

    frames: tuple[dict, ...]
    release: tuple[dict, ...]
    feet: float | None
    depth: float
    stand: float
    fold: float
    down: float


def seat_stand(front: float) -> float:
    """How far ahead of a seat whose front edge is front m ahead a sitter stands before it sits, calves clear of the edge."""
    return front + SHIN_GAP + S.Body().r_leg


def seat_pelvis(body: BodyMeasurements, reach: float, front: float | None, back: float | None) -> float:
    """Where the pelvis sits ahead of the parked point: back against the seat's rear edge, forward as far as the calves need to clear its front edge, never closer to that edge than PELVIS_MARGIN."""
    if front is None:
        return 0.0
    clear = front + SHIN_GAP + S.Body().r_leg - reach
    rest = clear if back is None else body.clearance.lie - back
    return min(max(rest, clear), front - PELVIS_MARGIN)


def plan_seat(squat: Sequence[Mapping], body: BodyMeasurements, height: float, front: float | None, back: float | None, fps: float) -> SeatPlan:
    """Frames sitting down onto a seat of the given height along the squat's drop and torso, and back up, the shins dangling from a seat out of their reach."""
    legs = body.legs
    target = seated_target(body, height)
    stand = legs.hip + legs.thigh + legs.shin + legs.ankle
    profile = squat_profile(squat, legs)
    seated = seated_legs(target, legs)
    hip, reach = seated if seated is not None else (math.pi / 2.0, legs.thigh)
    depth = seat_pelvis(body, reach, front, back)
    torso = frames_until(squat, profile.seated_frame() if seated is None else profile.drop_frame(target))
    n = len(torso)
    down: list[dict[str, float]] = []
    for k, frame in enumerate(torso):
        angles = frame["angles"]
        if seated is None:
            u = _smooth(k / (n - 1)) if n > 1 else 1.0
            down.append(_with_arms(_with_legs(angles, u * hip, -u * hip), u))
            continue
        z = target if k == n - 1 else max(target, min(stand, planted_pelvis(angles, legs)))
        u = (stand - z) / (stand - target) if stand > target else 1.0
        x = depth + reach - u * reach
        down.append(_with_arms(_with_legs(angles, *leg_ik(depth + reach - x, z - legs.hip - legs.ankle, legs)), u))
    lean = {joint: down[-1].get(joint, 0.0) for joint in TORSO_JOINTS}
    recline = [_with_torso(down[-1], lean, 1.0 - _smooth(j / max(1, round(RECLINE_S * fps)))) for j in range(1, max(1, round(RECLINE_S * fps)) + 1)]
    rise = [_with_torso(down[-1], lean, _smooth(j / max(1, round(LEAN_S * fps)))) for j in range(max(1, round(LEAN_S * fps)) + 1)]
    feet = None if seated is None else depth + reach
    start = depth + reach if feet is not None else depth if front is None else max(depth, seat_stand(front))
    return SeatPlan(frames=_timed(down + recline, fps), release=_timed(rise + down[::-1], fps), feet=feet, depth=depth, stand=start, fold=hip, down=(n - 1) / fps)


def rise_seconds(plan: SeatPlan) -> float:
    """How long the plan's rise takes."""
    return float(plan.release[-1]["t"]) if plan.release else 0.0


def blend_pose(start: tuple[np.ndarray, np.ndarray], end: tuple[np.ndarray, np.ndarray], weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and xyzw orientation weight of the way from start to end, eased."""
    w = _smooth(min(1.0, max(0.0, weight)))
    q0, q1 = np.asarray(start[1], dtype=float), np.asarray(end[1], dtype=float)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    angle = math.acos(min(1.0, float(np.dot(q0, q1))))
    if angle < 1e-6:
        q = q1
    else:
        q = (math.sin((1.0 - w) * angle) * q0 + math.sin(w * angle) * q1) / math.sin(angle)
    return (1.0 - w) * np.asarray(start[0], dtype=float) + w * np.asarray(end[0], dtype=float), q / np.linalg.norm(q)


def seat_depth(plan: SeatPlan, angles: Mapping[str, float], legs: Legs) -> float:
    """How far ahead of the parked point the pelvis is now: under planted feet it follows the legs, dangling it slides from where it stood onto the seat as the hips fold."""
    if plan.feet is not None:
        return plan.feet - ankle_reach(angles, legs)
    folded = min(1.0, max(0.0, hip_flexion(angles) / plan.fold)) if plan.fold > 0.0 else 1.0
    return plan.stand + folded * (plan.depth - plan.stand)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


_SUPINE = np.array([0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])


def supported_pose(support: Support, body: BodyMeasurements, angles: Mapping[str, float], x: float, y: float, yaw: float, depth: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Body frame position and xyzw orientation of a ped parked at (x, y, yaw) on its support.

    A sitter faces yaw with its pelvis depth m ahead of (x, y). A lying ped has its pelvis over (x, y), its head pointing
    away from yaw and its feet toward it.
    """
    pelvis = planted_pelvis(angles, body.legs)
    upright = np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    if support.kind == "lie":
        position = np.array([x + pelvis * math.cos(yaw), y + pelvis * math.sin(yaw), support.height + body.clearance.lie])
        return position, _quat_mul(upright, _SUPINE)
    lift = max(0.0, support.height + body.clearance.sit_at(hip_flexion(angles)) - pelvis)
    return np.array([x + depth * math.cos(yaw), y + depth * math.sin(yaw), lift]), upright


# m, joint centre to the skin a lying or kneeling body rests on; the pelvis takes the model's measurement
_FLESH = {"waist": 0.11, "spine": 0.11, "chest": 0.11, "torso": 0.11, "neck": 0.07, "head": 0.1, "shoulder": 0.06, "elbow": 0.045, "wrist": 0.04, "hip": 0.09, "knee": 0.055}
_PALM = 0.45  # of the wrist to fingertip reach, to the palm's centre
_HAND_FLESH = (0.03, 0.01)  # m, under the palm's centre and the fingertips
_TOE = 0.15  # m, ankle to the ball of the foot, as the skeleton's foot point


def _rotate(q: Sequence[float] | np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vector v (..., 3) turned by the xyzw quaternion q."""
    u, w = np.asarray(q[:3], dtype=float), float(q[3])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _contacts(angles: Mapping[str, float], body: BodyMeasurements) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Body frame points a body can rest on and the skin depth under each, the heel and toe sole points, the pelvis joint.

    Joints carry their flesh; the hands their palm and fingertips; the feet a heel and toe point on the sole, which turn
    with the foot as the renderers' sole points do.
    """
    pos, frames = S.fk(dict.fromkeys(JOINT_NAMES, 0.0) | dict(angles), _skeleton(body.legs, body.proportions))
    points = [pos[k] for k in pos if k.split("_")[-1] in _FLESH]
    flesh = [_FLESH[k.split("_")[-1]] for k in pos if k.split("_")[-1] in _FLESH]
    points.append(pos["pelvis"])
    flesh.append(body.clearance.lie)
    soles = []
    for side in ("l", "r"):
        hand = frames[f"{side}_hand"]
        for reach, depth in zip((_PALM * body.proportions.hand, body.proportions.hand), _HAND_FLESH, strict=True):
            points.append(pos[f"{side}_wrist"] + hand @ (S.DOWN * reach))
            flesh.append(depth)
        foot, down = frames[f"{side}_foot"], frames[f"{side}_foot"] @ (S.DOWN * body.legs.ankle)
        soles += [pos[f"{side}_ankle"] + down, pos[f"{side}_ankle"] + foot @ (S.FWD * _TOE) + down]
    pelvis = pos["pelvis"] + np.array([0.0, 0.0, body.legs.hip])  # where the renderer puts the pelvis joint
    return np.array(points), np.array(flesh), np.array(soles), pelvis


def tilted_pose(body: BodyMeasurements, angles: Mapping[str, float], x: float, y: float, floor: float, yaw: float, tilt: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Body frame position and xyzw orientation of a ped whose clip tilts its pelvis (``tilt``, xyzw from its heading yaw).

    The pelvis stays over (x, y) and the body turns about it; its height puts the lowest skin (joints less their flesh,
    hands, soles) on the floor. The renderers plant the lowest sole point at body height 0, so upright the pose matches
    an untilted ped's.
    """
    points, flesh, soles, pelvis = _contacts(angles, body)
    turned = _rotate(tilt, np.concatenate([points, soles]) - pelvis)[:, 2]
    height = float(np.max(np.concatenate([flesh, np.zeros(len(soles))]) - turned))
    orientation = _quat_mul(np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]), np.asarray(tilt, dtype=float))
    lift = float(pelvis[2] - soles[:, 2].min())  # the renderer's pelvis height above the body frame origin
    origin = np.array([x, y, floor + height]) - _rotate(orientation, np.array([0.0, 0.0, lift]))
    return origin, orientation
