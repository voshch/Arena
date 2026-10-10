"""REP-155 face pose of a human_description body and the intrinsics of its central view."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml
from task_generator.simulators.human.drivers import RIG_FILE
from urdf_parser_py import urdf as urdf_parser

from .body_pool import _origin_matrix
from .rig import _rot_axis

# gaze_<id> in face_<id>: z along the face's x, x right, y down
GAZE_IN_FACE = (0.5, -0.5, 0.5, -0.5)


@dataclasses.dataclass(frozen=True)
class _ChainJoint:
    name: str
    origin: np.ndarray
    axis: np.ndarray | None


@dataclasses.dataclass(frozen=True)
class FaceChain:
    """The joints from `body_<id>` to `head_<id>` and the sellion in the head frame."""

    joints: tuple[_ChainJoint, ...]
    sellion: np.ndarray

    @classmethod
    def from_urdf(cls, urdf_xml: str, body_id: str) -> FaceChain:
        robot = urdf_parser.URDF.from_xml_string(urdf_xml)
        head = f"head_{body_id}"
        names = robot.get_chain(f"body_{body_id}", head, links=False)
        joints = []
        for name in names:
            joint = robot.joint_map[name]
            axis = np.asarray(joint.axis, dtype=float) if joint.type in ("revolute", "continuous") else None
            joints.append(_ChainJoint(name, _origin_matrix(joint.origin), axis))
        eyes = [visual.origin.xyz for visual in robot.link_map[head].visuals if visual.origin is not None and abs(visual.origin.xyz[1]) > 0.0]
        if len(eyes) != 2:
            raise ValueError(f"{head} has {len(eyes)} off-midline visuals, expected two eyes")
        return cls(tuple(joints), np.mean(np.asarray(eyes, dtype=float), axis=0))

    def pose(self, positions: Mapping[str, float]) -> np.ndarray:
        """4x4 transform of the face (sellion, x forward, z up) in `body_<id>`, missing joints at zero."""
        m = np.eye(4)
        for joint in self.joints:
            m = m @ joint.origin
            if joint.axis is not None:
                q = positions.get(joint.name, 0.0)
                if q:
                    r = np.eye(4)
                    r[:3, :3] = _rot_axis(joint.axis, q)
                    m = m @ r
        sellion = np.eye(4)
        sellion[:3, 3] = self.sellion
        return m @ sellion


def root_drop(model_uri: str) -> float:
    """Height in m the rig.yaml beside an actor SDF moves the body root, 0 without the file or the key."""
    if not model_uri:
        return 0.0
    rig_path = Path(model_uri).with_name(RIG_FILE)
    if not rig_path.is_file():
        return 0.0
    rig = yaml.safe_load(rig_path.read_text()) or {}
    return float(rig.get("root_offset", (0.0, 0.0, 0.0))[2])


def quat_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the rotation block of a 4x4 transform."""
    r = m[:3, :3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        return 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return (r[k, j] - r[j, k]) / s, q[0], q[1], q[2]


def matrix_from_pose(xyz: tuple[float, float, float], wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = wxyz
    m = np.eye(4)
    m[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    m[:3, 3] = xyz
    return m


def pinhole_k(fov: float, size: int) -> list[float]:
    """Row-major K of a square `size` image whose horizontal and vertical fov is `fov` rad."""
    f = size / 2.0 / math.tan(fov / 2.0)
    c = size / 2.0
    return [f, 0.0, c, 0.0, f, c, 0.0, 0.0, 1.0]
