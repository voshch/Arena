"""URDF-faithful forward kinematics + collision proxies for the 36-DOF human.

Mirrors ``human_description/urdf/human-tpl.xacro`` (branch ``arena-v2``)
exactly: same bone lengths as a function of ``height``, same joint axes, same
chain order.  Shoulder triples are interpreted in the **wire (anatomical)**
convention of JOINTS.md section 1a, i.e. ``Rz(y) R((0,-1,0), p) R((0,0,-1), r)``
relative to the collar frame -- identical to convert_v2's ``SPEC_ARM``.  This is
what Arena's RViz adapter and the Isaac bone map consume, so a clip that renders
correctly here renders correctly there.

Frame convention everywhere: REP-155 body frame, ``x`` forward, ``y`` left,
``z`` up.  The FK world origin ("ped-local") is the point on the **floor**
directly under the pelvis, so ``z`` is height above ground.
"""

from __future__ import annotations

import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])
UP = np.array([0.0, 0.0, 1.0])
FWD = np.array([1.0, 0.0, 0.0])

SPINE = ("waist", "spine", "chest")
TORSO_AXIS = {
    "r": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 0.0, 1.0]),
    "p": np.array([0.0, 1.0, 0.0]),
}
COLLAR_AXES = {  # (protraction axis, elevation axis), mirrored like p_hip
    "l": (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
    "r": (np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])),
}
ARM_AXES = (  # wire/anatomical, both sides identical (JOINTS.md 1a)
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, -1.0]),
)
HIP_AXES = {
    "l": (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
    "r": (np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
}
ELBOW_AXIS = np.array([0.0, -1.0, 0.0])
REFLECT = {"l": 1.0, "r": -1.0}


def rot_axis(axis: np.ndarray | list[float] | tuple[float, ...], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def unit(v: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def perp_component(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Component of ``ref`` orthogonal to unit vector ``v``, normalised."""
    v = unit(v)
    ref = np.asarray(ref, dtype=float)
    return unit(ref - np.dot(ref, v) * v)


class Body:
    """Bone lengths from human-tpl.xacro (all proportional to ``height``)."""

    def __init__(self, height: float = 1.65):
        u = height / 7.5
        self.height = float(height)
        self.size_unit = u
        self.head_radius = u / 2
        self.neck_length = u * 0.25
        self.neck_shoulder = u * 0.8        # clavicle length
        self.upperarm = u * 1.6
        self.forearm = u
        self.torso_height = u * 2.75
        self.spine_segment = self.torso_height / 3
        self.waist_length = u * 1.2         # hip width
        self.thigh = u * 1.5
        self.tibia = u * 2.0
        self.limb_radius = 0.03
        self.hip_height = self.thigh + self.tibia
        self.arm_reach = self.upperarm + self.forearm
        # Collision proxy radii (soft-tissue envelope, not the URDF stick radii).
        # Calibrated against the three recorded pointing clips: real captured
        # motion is collision-free by construction, so every radius sum sits
        # just under the tightest natural approach in those clips (arm-vs-torso
        # 19.6 cm, arm-vs-head 18.7 cm, forearm-vs-pelvis and -vs-thigh 14.8 cm
        # at height 1.65).  A negative clearance therefore means "tighter than
        # any pose in the source captures", not "geometrically interpenetrating
        # the URDF sticks".
        self.r_torso = 0.549 * self.waist_length     # 0.145 @ 1.65 m
        self.r_pelvis = 0.379 * self.waist_length    # 0.100
        self.r_arm = 0.170 * self.waist_length       # 0.045
        self.r_leg = 0.341 * self.waist_length       # 0.090
        self.r_head = 0.909 * self.head_radius       # 0.100


def segment_rot(r: float, y: float, p: float) -> np.ndarray:
    return rot_axis(TORSO_AXIS["r"], r) @ rot_axis(TORSO_AXIS["y"], y) @ rot_axis(TORSO_AXIS["p"], p)


def collar_rot(side: str, y: float, p: float) -> np.ndarray:
    ay, ap = COLLAR_AXES[side]
    return rot_axis(ay, y) @ rot_axis(ap, p)


def arm_rot(y: float, p: float, r: float) -> np.ndarray:
    """Wire-convention shoulder rotation, relative to the collar frame."""
    a1, a2, a3 = ARM_AXES
    return rot_axis(a1, y) @ rot_axis(a2, p) @ rot_axis(a3, r)


def arm_rot_inverse(m: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`arm_rot`: recover the wire triple from a rotation.

    ``arm_rot(y, p, r) @ (0,0,-1) = (-m[0,2], -m[1,2], -m[2,2])`` gives the
    polar pair directly, and the twist then falls out of the residual rotation.
    """
    p = float(np.arccos(np.clip(m[2, 2], -1.0, 1.0)))
    if np.sin(p) > 1e-7:
        y = float(np.arctan2(-m[1, 2], -m[0, 2]))
    else:                                    # arm exactly up or down: y and r merge
        y = 0.0
    n = (rot_axis(ARM_AXES[0], y) @ rot_axis(ARM_AXES[1], p)).T @ m
    r = float(np.arctan2(-n[1, 0], n[0, 0]))
    return y, p, r


def slerp_rot(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two rotation matrices."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    rel = a.T @ b
    cos = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        return a
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:                              # 180 deg: pick any perpendicular axis
        w, v = np.linalg.eigh(rel + np.eye(3))
        axis = v[:, int(np.argmax(w))]
    return a @ rot_axis(axis, t * angle)


def head_rot(r: float, y: float, p: float) -> np.ndarray:
    return rot_axis([1.0, 0, 0], r) @ rot_axis([0, 0, 1.0], y) @ rot_axis([0, -1.0, 0], p)


def fk(angles: dict, body: Body, root_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[dict, dict]:
    """Full-body FK.  Returns ``(positions, frames)`` in the ped-local frame.

    ``positions`` maps link name -> world xyz, ``frames`` maps segment name ->
    3x3 rotation.  ``root_xy_yaw`` shifts/turns the whole body (the clip's own
    root track).  Pass the default for a ped-local solve.
    """
    a = angles
    rx, ry, yaw = root_xy_yaw
    root_r = rot_axis(UP, yaw)
    pelvis = np.array([rx, ry, body.hip_height])
    pos = {"pelvis": pelvis}
    frames = {"root": root_r}

    r = root_r
    cursor = pelvis
    for base in SPINE:
        r = r @ segment_rot(a[f"r_{base}"], a[f"y_{base}"], a[base])
        frames[base] = r
        pos[base] = cursor
        cursor = cursor + r @ (UP * body.spine_segment)
    torso = cursor
    pos["torso"] = torso
    frames["torso"] = r
    torso_r = r

    neck = torso + torso_r @ (UP * body.neck_length)
    pos["neck"] = neck
    h_r = torso_r @ head_rot(a["r_head"], a["y_head"], a["p_head"])
    frames["head"] = h_r
    pos["head"] = neck + h_r @ np.array([body.head_radius / 2, 0.0, body.head_radius])
    pos["eye"] = neck + h_r @ np.array([body.head_radius, 0.0, body.head_radius])

    for side in ("l", "r"):
        c_r = torso_r @ collar_rot(side, a[f"{side}_y_collar"], a[f"{side}_p_collar"])
        frames[f"{side}_collar"] = c_r
        sh = torso + c_r @ np.array([0.0, REFLECT[side] * body.neck_shoulder, 0.0])
        pos[f"{side}_shoulder"] = sh
        s_r = c_r @ arm_rot(
            a[f"{side}_y_shoulder"], a[f"{side}_p_shoulder"], a[f"{side}_r_shoulder"]
        )
        frames[f"{side}_upperarm"] = s_r
        el = sh + body.upperarm * (s_r @ DOWN)
        pos[f"{side}_elbow"] = el
        f_r = s_r @ rot_axis(ELBOW_AXIS, a[f"{side}_elbow"])
        frames[f"{side}_forearm"] = f_r
        pos[f"{side}_wrist"] = el + body.forearm * (f_r @ DOWN)

    for side in ("l", "r"):
        hip = pelvis + root_r @ np.array([0.0, REFLECT[side] * body.waist_length / 2, 0.0])
        pos[f"{side}_hip"] = hip
        x1, x2, x3 = HIP_AXES[side]
        h = (
            root_r
            @ rot_axis(x1, a[f"{side}_y_hip"])
            @ rot_axis(x2, a[f"{side}_p_hip"])
            @ rot_axis(x3, a[f"{side}_r_hip"])
        )
        knee = hip + body.thigh * (h @ DOWN)
        pos[f"{side}_knee"] = knee
        k = h @ rot_axis(ELBOW_AXIS, a[f"{side}_knee"])
        ankle = knee + body.tibia * (k @ DOWN)
        pos[f"{side}_ankle"] = ankle
        f = k @ rot_axis([0, 0, -1.0], a[f"{side}_y_ankle"]) @ rot_axis(ELBOW_AXIS, a[f"{side}_ankle"])
        frames[f"{side}_foot"] = f
        pos[f"{side}_foot"] = ankle + 0.15 * (f @ FWD)

    return pos, frames


STICK_EDGES = [
    ("pelvis", "waist"), ("waist", "spine"), ("spine", "chest"), ("chest", "torso"),
    ("torso", "neck"), ("neck", "head"),
    ("torso", "l_shoulder"), ("torso", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("pelvis", "l_hip"), ("pelvis", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_foot"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_foot"),
]


# ---------------------------------------------------------------------------
# capsule collision proxies
# ---------------------------------------------------------------------------


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest distance between two 3D segments (Ericson, RTCD 5.1.9).

    Degenerate (zero-length) segments are handled, so a point is passed as
    ``(p, p)``.
    """
    eps = 1e-12
    d1 = np.asarray(p1, float) - np.asarray(p0, float)
    d2 = np.asarray(q1, float) - np.asarray(q0, float)
    r = np.asarray(p0, float) - np.asarray(q0, float)
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = np.dot(d1, r)
        if e <= eps:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((np.asarray(p0, float) + s * d1) - (np.asarray(q0, float) + t * d2)))


def _lerp(p0: np.ndarray, p1: np.ndarray, f: float) -> np.ndarray:
    return p0 + f * (p1 - p0)


def collision_clearances(pos: dict, body: Body, side: str) -> dict[str, float]:
    """Signed clearance (m) between the pointing arm and the rest of the body.

    Positive = free space between the capsule surfaces, negative = interpenetration.
    The upper arm is trimmed to its distal 65% for torso/head tests: its proximal
    end is *supposed* to sit against the shoulder, and testing the full segment
    would report a permanent false collision there.
    """
    other = "l" if side == "r" else "r"
    sh, el, wr = (pos[f"{side}_{k}"] for k in ("shoulder", "elbow", "wrist"))
    upper_trim = (_lerp(sh, el, 0.35), el)
    fore = (el, wr)
    ra = body.r_arm

    torso_seg = (pos["pelvis"], pos["torso"])
    pelvis_seg = (pos["l_hip"], pos["r_hip"])
    head_pt = pos["head"]
    out = {}

    for tag, seg in (("upperarm", upper_trim), ("forearm", fore)):
        out[f"{tag}/torso"] = _segment_distance(*seg, *torso_seg) - (ra + body.r_torso)
        out[f"{tag}/pelvis"] = _segment_distance(*seg, *pelvis_seg) - (ra + body.r_pelvis)
        out[f"{tag}/head"] = _segment_distance(*seg, head_pt, head_pt) - (ra + body.r_head)
        out[f"{tag}/neck"] = (
            _segment_distance(*seg, pos["torso"], pos["neck"]) - (ra + 0.6 * body.r_torso)
        )
        for oth_tag, oth in (
            ("otherarm", (pos[f"{other}_shoulder"], pos[f"{other}_elbow"])),
            ("otherfore", (pos[f"{other}_elbow"], pos[f"{other}_wrist"])),
        ):
            out[f"{tag}/{oth_tag}"] = _segment_distance(*seg, *oth) - 2 * ra
        for leg_tag, leg in (
            ("thigh_l", (pos["l_hip"], pos["l_knee"])),
            ("thigh_r", (pos["r_hip"], pos["r_knee"])),
        ):
            out[f"{tag}/{leg_tag}"] = _segment_distance(*seg, *leg) - (ra + body.r_leg)
    # the hand must not go through the floor either
    out["wrist/floor"] = float(wr[2]) - ra
    out["elbow/floor"] = float(el[2]) - ra
    return out


def worst_clearance(pos: dict, body: Body, side: str) -> tuple[str, float]:
    cl = collision_clearances(pos, body, side)
    k = min(cl, key=cl.get)
    return k, cl[k]
