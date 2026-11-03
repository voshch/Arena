"""Arena body rig (arena_runtime humans/BODY.md): joint table, template skeleton, SMPL-X conversion, semantic DOFs and bundle rest offsets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import yaml

from arena_assets.rig import Rig, rig_of
from arena_simulation_setup import ASS_DIR

TEMPLATE_FORMAT = "arena-body-template/1"

BODY_PARENTS: dict[str, str | None] = {
    "pelvis": None,
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}
_DIGITS = ("index", "middle", "pinky", "ring", "thumb")
FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
OPTIONAL_PARENTS: dict[str, str] = {
    "jaw": "head",
    "left_eye_smplhf": "head",
    "right_eye_smplhf": "head",
    **{f"{side}_{digit}{k}": f"{side}_wrist" if k == 1 else f"{side}_{digit}{k - 1}" for side in ("left", "right") for digit in _DIGITS for k in (1, 2, 3)},
}
PARENTS: dict[str, str | None] = {**BODY_PARENTS, **OPTIONAL_PARENTS}
BODY_JOINTS: tuple[str, ...] = tuple(BODY_PARENTS)

SMPLX_TO_ARENA = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

SEMANTIC_DOFS: tuple[str, ...] = (
    "r_waist",
    "y_waist",
    "waist",
    "r_spine",
    "y_spine",
    "spine",
    "r_chest",
    "y_chest",
    "chest",
    "r_head",
    "y_head",
    "p_head",
    *(f"{s}_{dof}" for s in "lr" for dof in ("y_collar", "p_collar", "y_shoulder", "p_shoulder", "r_shoulder", "elbow")),
    *(f"{s}_{dof}" for s in "lr" for dof in ("y_hip", "p_hip", "r_hip", "knee")),
    *(f"{s}_{dof}" for s in "lr" for dof in ("y_ankle", "ankle")),
    *(f"{s}_{dof}" for s in "lr" for dof in ("r_wrist", "wrist")),
    *(f"{s}_{dof}" for s in "lr" for dof in (*(f"curl_{digit}" for digit in FINGERS), "spread")),
)

_X, _Y, _Z = np.eye(3)
_SIDES = {"l": "left", "r": "right"}
_MIRROR = {"l": 1.0, "r": -1.0}
_PRIMARY_CHILD = {"pelvis": "spine1", "spine3": "neck", "left_wrist": "left_middle1", "right_wrist": "right_middle1"}
_CURL_GAIN = (1.0, 1.0, 0.7)
_SPREAD_SPLIT = {"index": 1.0, "middle": 1.0 / 3.0, "ring": -1.0 / 3.0, "pinky": -1.0}
_WRIST_OF = {f"{side}_{digit}{k}": f"{side}_wrist" for side in _SIDES.values() for digit in FINGERS for k in (1, 2, 3)}
DIGIT_JOINTS: tuple[str, ...] = tuple(_WRIST_OF)
_EPS = 1e-9


def axis_rotation(axis: Sequence[float] | np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix of `angle` rad about `axis`."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def quat_to_matrix(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rotation matrix of an x y z w quaternion."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """x y z w quaternion of a rotation matrix, w >= 0."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q = q / np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def from_smplx_rotation(m: np.ndarray) -> np.ndarray:
    """An SMPL-X joint rotation matrix in Arena axes."""
    return SMPLX_TO_ARENA @ m @ SMPLX_TO_ARENA.T


def from_smplx_vector(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """An SMPL-X vector in Arena axes."""
    return SMPLX_TO_ARENA @ np.asarray(v, dtype=float)


@attrs.frozen(eq=False)
class Template:
    """Reference skeleton at the zero pose: parent and offset from it (m, Arena axes) per joint, parents first."""

    parents: Mapping[str, str | None]
    offsets: Mapping[str, np.ndarray]


def load_template(path: Path | None = None) -> Template:
    """The template skeleton yaml, configs/body/template.yaml by default."""
    path = path or ASS_DIR / "configs" / "body" / "template.yaml"
    doc = yaml.safe_load(path.read_text())
    if doc.get("format") != TEMPLATE_FORMAT:
        raise ValueError(f"{path}: format {doc.get('format')!r}, expected {TEMPLATE_FORMAT!r}")
    parents: dict[str, str | None] = {}
    offsets: dict[str, np.ndarray] = {}
    for name, spec in doc["joints"].items():
        parent = spec["parent"]
        if name not in PARENTS or parent != PARENTS[name]:
            raise ValueError(f"{path}: joint {name!r} with parent {parent!r}, the rig has {PARENTS.get(name)!r}")
        if parent is not None and parent not in parents:
            raise ValueError(f"{path}: joint {name!r} comes before its parent {parent!r}")
        parents[name] = parent
        offsets[name] = np.asarray(spec["offset"], dtype=float)
    return Template(parents=parents, offsets=offsets)


def _check_joints(names: Sequence[str] | Mapping[str, object]) -> None:
    unknown = sorted(set(names) - set(PARENTS))
    if unknown:
        raise ValueError(f"unknown body joints {unknown}, the rig has {list(PARENTS)}")


def forward(template: Template, rotations: Mapping[str, Sequence[float] | np.ndarray], root_offset: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Body-frame position and world rotation of every template joint under a pose of local x y z w quaternions."""
    _check_joints(rotations)
    positions: dict[str, np.ndarray] = {}
    frames: dict[str, np.ndarray] = {}
    for name, parent in template.parents.items():
        local = quat_to_matrix(rotations[name]) if name in rotations else np.eye(3)
        if parent is None:
            positions[name] = template.offsets[name] + np.asarray(root_offset, dtype=float)
            frames[name] = local
        else:
            positions[name] = positions[parent] + frames[parent] @ template.offsets[name]
            frames[name] = frames[parent] @ local
    return positions, frames


@attrs.frozen(eq=False)
class _Group:
    joint: str
    dofs: tuple[str, ...]
    axes: tuple[np.ndarray, ...]


def _groups() -> tuple[_Group, ...]:
    groups = [
        _Group("spine1", ("r_waist", "y_waist", "waist"), (_X, _Z, _Y)),
        _Group("spine2", ("r_spine", "y_spine", "spine"), (_X, _Z, _Y)),
        _Group("spine3", ("r_chest", "y_chest", "chest"), (_X, _Z, _Y)),
    ]
    for s, side in _SIDES.items():
        m = _MIRROR[s]
        groups += [
            _Group(f"{side}_collar", (f"{s}_y_collar", f"{s}_p_collar"), (-m * _Z, m * _X)),
            _Group(f"{side}_elbow", (f"{s}_elbow",), (-_Y,)),
            _Group(f"{side}_wrist", (f"{s}_r_wrist", f"{s}_wrist"), (-m * _Z, m * _X)),
            _Group(f"{side}_hip", (f"{s}_y_hip", f"{s}_p_hip", f"{s}_r_hip"), (-_Z, m * _X, -_Y)),
            _Group(f"{side}_knee", (f"{s}_knee",), (-_Y,)),
            _Group(f"{side}_ankle", (f"{s}_y_ankle", f"{s}_ankle"), (-_Z, -_Y)),
        ]
    return tuple(groups)


_GROUPS = _groups()
_HEAD_AXES = (_X, _Z, -_Y)
_NECK_YAW = 0.6
_NECK_PITCH = 0.35
HANG = {s: axis_rotation(_X, -_MIRROR[s] * math.pi / 2.0) for s in _SIDES}
DEFAULT_STANCE: dict[str, np.ndarray] = {
    **{f"{side}_{joint}": HANG[s] for s, side in _SIDES.items() for joint in ("shoulder", "elbow", "wrist")},
    **{joint: HANG[s] for s, side in _SIDES.items() for joint in _WRIST_OF if joint.startswith(side)},
}


def stance_of(rig: Rig | None) -> Mapping[str, np.ndarray]:
    """World joint rotations of a rig's stance, the default stance without one."""
    if rig is None:
        return DEFAULT_STANCE
    return {name: quat_to_matrix(q) for name, q in zip(rig.joints, rig.stance, strict=True)}


def load_stance(model_uri: str) -> Mapping[str, np.ndarray]:
    """Stance of the .rig beside an actor SDF, the default stance without the file."""
    return stance_of(rig_of(model_uri))


def _held(stance: Mapping[str, np.ndarray], joint: str) -> np.ndarray:
    """A joint's stance rotation, a digit without one hanging like its wrist."""
    return stance.get(joint, stance.get(_WRIST_OF.get(joint, ""), np.eye(3)))


def _digit_frame(stance: Mapping[str, np.ndarray], joint: str) -> np.ndarray:
    """Takes the digit DOF axes of the hanging rest onto the digit as the stance holds it."""
    return _held(stance, joint) @ HANG[joint[0]].T


def _curl_axis(s: str, digit: str) -> np.ndarray:
    return -_MIRROR[s] * (_Z if digit == "thumb" else _X)


def _digit(s: str, digit: str, k: int, angles: Mapping[str, float]) -> np.ndarray:
    m = axis_rotation(_curl_axis(s, digit), _CURL_GAIN[k - 1] * float(angles.get(f"{s}_curl_{digit}", 0.0)))
    if k == 1 and digit in _SPREAD_SPLIT:
        m = axis_rotation(-_Y, _SPREAD_SPLIT[digit] * float(angles.get(f"{s}_spread", 0.0))) @ m
    return m


def _arm(y: float, p: float, r: float) -> np.ndarray:
    return axis_rotation(_Z, y) @ axis_rotation(-_Y, p) @ axis_rotation(-_Z, r)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _arm_angles(m: np.ndarray) -> tuple[float, float, float]:
    """The (y, p, r) of JOINTS.md 1a whose arm rotation is `m`, the branch with the smaller |y|."""
    p = math.acos(float(np.clip(m[2, 2], -1.0, 1.0)))
    y = math.atan2(-m[1, 2], -m[0, 2]) if math.sin(p) > 1e-7 else 0.0
    n = (axis_rotation(_Z, y) @ axis_rotation(-_Y, p)).T @ m
    r = math.atan2(-n[1, 0], n[0, 0])
    if p > 1e-7 and abs(_wrap(y + math.pi)) < abs(y):
        return _wrap(y + math.pi), -p, _wrap(r + math.pi)
    return y, p, r


def _decompose(m: np.ndarray, axes: tuple[np.ndarray, ...]) -> tuple[float, ...]:
    """Angles about the signed coordinate axes whose rotations, composed in order, give `m`."""
    if len(axes) == 1:
        q = matrix_to_quat(m)
        return (_wrap(2.0 * math.atan2(float(q[:3] @ axes[0]), q[3])),)
    index = [int(np.argmax(np.abs(a))) for a in axes]
    signs = [float(a[i]) for a, i in zip(axes, index, strict=True)]
    i, j = index[0], index[1]
    k = index[2] if len(index) == 3 else 3 - i - j
    e = 1.0 if (j - i) % 3 == 1 else -1.0
    angles = (
        math.atan2(-e * m[j, k], m[k, k]),
        math.asin(float(np.clip(e * m[i, k], -1.0, 1.0))),
        math.atan2(-e * m[i, j], m[i, i]),
    )
    return tuple(angle * sign for angle, sign in zip(angles, signs, strict=False))


def joint_angles(joint: str, rotation: np.ndarray) -> dict[str, float]:
    """The semantic DOFs of a body joint for its rotation in the frame aligned with the body at the stance."""
    group = next((group for group in _GROUPS if group.joint == joint), None)
    if group is None:
        raise ValueError(f"joint {joint!r} has no plain DOF group, those are {[group.joint for group in _GROUPS]}")
    return dict(zip(group.dofs, _decompose(rotation, group.axes), strict=True))


def _semantic_rotations(angles: Mapping[str, float]) -> dict[str, np.ndarray]:
    """Each driven joint's rotation in its semantic frame, aligned with the body at the stance."""
    out: dict[str, np.ndarray] = {}
    for group in _GROUPS:
        m = np.eye(3)
        for dof, axis in zip(group.dofs, group.axes, strict=True):
            m = m @ axis_rotation(axis, float(angles.get(dof, 0.0)))
        out[group.joint] = m
    for s, side in _SIDES.items():
        out[f"{side}_shoulder"] = _arm(*(float(angles.get(f"{s}_{dof}", 0.0)) for dof in ("y_shoulder", "p_shoulder", "r_shoulder")))
    for s, side in _SIDES.items():
        for digit in FINGERS:
            for k in (1, 2, 3):
                out[f"{side}_{digit}{k}"] = _digit(s, digit, k, angles)
    roll, yaw, pitch = (float(angles.get(dof, 0.0)) for dof in ("r_head", "y_head", "p_head"))
    out["neck"] = axis_rotation(_Z, _NECK_YAW * yaw) @ axis_rotation(-_Y, _NECK_PITCH * pitch)
    out["head"] = axis_rotation(_X, roll) @ axis_rotation(_Z, (1.0 - _NECK_YAW) * yaw) @ axis_rotation(-_Y, (1.0 - _NECK_PITCH) * pitch)
    return out


JOINT_DOFS: dict[str, tuple[str, ...]] = {
    **{group.joint: group.dofs for group in _GROUPS},
    **{f"{side}_shoulder": (f"{s}_y_shoulder", f"{s}_p_shoulder", f"{s}_r_shoulder") for s, side in _SIDES.items()},
    "neck": ("y_head", "p_head"),
    "head": ("r_head", "y_head", "p_head"),
    **{f"{side}_{digit}{k}": (f"{s}_curl_{digit}", f"{s}_spread") if k == 1 and digit in _SPREAD_SPLIT else (f"{s}_curl_{digit}",) for s, side in _SIDES.items() for digit in FINGERS for k in (1, 2, 3)},
}


def semantic_to_body(angles: Mapping[str, float], stance: Mapping[str, np.ndarray] = DEFAULT_STANCE) -> dict[str, np.ndarray]:
    """Local x y z w quaternions of the body and digit joints for the JOINTS.md DOFs over a stance, missing DOFs at 0."""
    unknown = sorted(set(angles) - set(SEMANTIC_DOFS))
    if unknown:
        raise ValueError(f"unknown semantic DOFs {unknown}, JOINTS.md has {list(SEMANTIC_DOFS)}")
    semantic = _semantic_rotations(angles)
    identity = np.eye(3)
    out: dict[str, np.ndarray] = {}
    for joint, parent in BODY_PARENTS.items():
        above = stance.get(parent, identity) if parent is not None else identity
        out[joint] = matrix_to_quat(above.T @ semantic.get(joint, identity) @ stance.get(joint, identity))
    for joint in _WRIST_OF:
        frame = _digit_frame(stance, joint)
        out[joint] = matrix_to_quat(_held(stance, OPTIONAL_PARENTS[joint]).T @ frame @ semantic[joint] @ frame.T @ _held(stance, joint))
    return out


def body_to_semantic(rotations: Mapping[str, Sequence[float] | np.ndarray], stance: Mapping[str, np.ndarray] = DEFAULT_STANCE) -> dict[str, float]:
    """All JOINTS.md DOFs extracted from local x y z w joint quaternions over a stance, missing joints at the stance."""
    _check_joints(rotations)
    identity = np.eye(3)
    semantic: dict[str, np.ndarray] = {}
    for joint, parent in BODY_PARENTS.items():
        above = stance.get(parent, identity) if parent is not None else identity
        below = stance.get(joint, identity)
        semantic[joint] = above @ quat_to_matrix(rotations[joint]) @ below.T if joint in rotations else identity
    for joint in _WRIST_OF:
        frame = _digit_frame(stance, joint)
        semantic[joint] = frame.T @ _held(stance, OPTIONAL_PARENTS[joint]) @ quat_to_matrix(rotations[joint]) @ _held(stance, joint).T @ frame if joint in rotations else identity
    out: dict[str, float] = {}
    for group in _GROUPS:
        out.update(zip(group.dofs, _decompose(semantic[group.joint], group.axes), strict=True))
    for s, side in _SIDES.items():
        out.update(zip((f"{s}_y_shoulder", f"{s}_p_shoulder", f"{s}_r_shoulder"), _arm_angles(semantic[f"{side}_shoulder"]), strict=True))
    for s, side in _SIDES.items():
        spread = spread_weight = 0.0
        for digit in FINGERS:
            curl = curl_weight = 0.0
            for k, gain in enumerate(_CURL_GAIN, 1):
                joint = semantic[f"{side}_{digit}{k}"]
                if k == 1 and digit in _SPREAD_SPLIT:
                    turn, angle = _decompose(joint, (-_Y, _curl_axis(s, digit)))
                    spread += _SPREAD_SPLIT[digit] * turn
                    spread_weight += _SPREAD_SPLIT[digit] ** 2
                else:
                    (angle,) = _decompose(joint, (_curl_axis(s, digit),))
                curl += gain * angle
                curl_weight += gain**2
            out[f"{s}_curl_{digit}"] = curl / curl_weight
        out[f"{s}_spread"] = spread / spread_weight
    neck_yaw, neck_pitch = _decompose(semantic["neck"], (_Z, -_Y))
    yaw, pitch = neck_yaw / _NECK_YAW, neck_pitch / _NECK_PITCH
    rest = axis_rotation(_Z, (1.0 - _NECK_YAW) * yaw) @ axis_rotation(-_Y, (1.0 - _NECK_PITCH) * pitch)
    out["r_head"], out["y_head"], out["p_head"] = _decompose(semantic["head"] @ rest.T, _HEAD_AXES)[0], yaw, pitch
    return {dof: out[dof] for dof in SEMANTIC_DOFS}


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking the direction of `a` onto the direction of `b`."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(a @ b)
    if s < _EPS:
        if c > 0.0:
            return np.eye(3)
        other = _X if abs(a[0]) < 0.9 else _Y
        return axis_rotation(np.cross(a, other), math.pi)
    return axis_rotation(axis / s, math.atan2(s, c))


def rest_correction(template: Template, source: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each joint's `O` of a source skeleton whose rest frames are the identity, its rest offsets in Arena axes."""
    heads, _ = forward(Template(parents={joint: template.parents[joint] for joint in template.parents if joint in source}, offsets=source), {})
    identity = {joint: np.eye(3) for joint in source}
    offsets = rest_offsets(template, {joint: [joint] for joint in source}, heads, identity)
    return {joint: quat_to_matrix(pairs[0][1]) for joint, pairs in offsets.items()}


def rest_offsets(template: Template, bones: Mapping[str, Sequence[str]], bind_positions: Mapping[str, Sequence[float] | np.ndarray], bind_rotations: Mapping[str, np.ndarray]) -> dict[str, list[tuple[str, np.ndarray]]]:
    """`.rig` bone offset quaternions per joint: each bone's world rotation with the bind skeleton (bone heads and world rotations, body frame) swung into the template zero pose."""
    _check_joints(bones)
    for joint in bones:
        if joint not in template.parents:
            raise ValueError(f"joint {joint!r} is not in the template")
        parent = template.parents[joint]
        if parent is not None and parent not in bones:
            raise ValueError(f"joint {joint!r} is mapped but its parent {parent!r} is not")
    order = [joint for joint in template.parents if joint in bones]
    children: dict[str, list[str]] = {joint: [] for joint in order}
    for joint in order:
        parent = template.parents[joint]
        if parent is not None:
            children[parent].append(joint)
    position = {joint: np.asarray(bind_positions[bones[joint][0]], dtype=float) for joint in order}
    rotation = {bone: np.asarray(bind_rotations[bone], dtype=float) for joint in order for bone in bones[joint]}

    def subtree(joint: str) -> list[str]:
        out = [joint]
        for child in children[joint]:
            out += subtree(child)
        return out

    for joint in order:
        if not children[joint]:
            continue
        if len(children[joint]) == 1:
            child = children[joint][0]
        elif _PRIMARY_CHILD.get(joint) in children[joint]:
            child = _PRIMARY_CHILD[joint]
        else:
            raise ValueError(f"joint {joint!r} has children {children[joint]} and none of them sets its direction")
        swing = _align(position[child] - position[joint], template.offsets[child])
        pivot = position[joint]
        for member in subtree(joint):
            for bone in bones[member]:
                rotation[bone] = swing @ rotation[bone]
            if member != joint:
                position[member] = pivot + swing @ (position[member] - pivot)
    return {joint: [(bone, matrix_to_quat(rotation[bone])) for bone in bones[joint]] for joint in order}
