"""SMPL-X body motion: the source side.

Everything is normalised to a :class:`Motion` in ROS axes (x forward, y left, z up), holding the 22 body joints' global
rotations and positions. Two frames matter:

* the **body frame** every SMPL-X joint rotation is relative to: the model's canonical frame (Y up, Z forward, X left),
  in which all rest rotations are the identity (T-pose). ``BODY_TO_ROS`` maps it to ROS axes.
* the **world frame** the sequence lives in: Y up for Kimodo and the SMPL-X model's own output, Z up for AMASS.

Supported ``.npz`` inputs (detected by their keys):

* rotation output, e.g. Kimodo: ``global_rot_mats`` (T, >=22, 3, 3) and ``posed_joints`` (T, >=22, 3); or
  ``local_rot_mats`` alone (positions come from forward kinematics).
* SMPL-X parameters, e.g. AMASS: ``poses`` (T, >=66 axis-angle) or ``root_orient``/``global_orient`` + ``pose_body``/
  ``body_pose`` (T, 63), with ``trans``/``transl`` and optional ``betas``.

Two-person files carry each person's arrays under a ``p0_`` / ``p1_`` key prefix (``load(..., person=i)``). A file may
name its world up axis in an ``up`` key ("y" / "z"), which ``up="auto"`` follows.

Hands, jaw and eyes are ignored: Arena's wire has no fingers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.transform import Rotation

JOINTS = ("pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist")
J = {n: i for i, n in enumerate(JOINTS)}
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

BODY_TO_ROS = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])  # SMPL-X canonical (X left, Y up, Z fwd) -> ROS
WORLD_TO_ROS = {"y": BODY_TO_ROS, "z": np.eye(3)}  # by the world's up axis

# Rest offsets of each joint from its parent, in the canonical body frame (m): the body Kimodo generates with, measured
# from its output (rigid to 1e-7 across clips). Used when a source carries no joint positions and no model is given;
# pass an SMPL-X model file to get the exact skeleton for a sequence's own betas.
REST_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0523, -0.09394, -0.02761],
        [-0.05719, -0.10655, -0.02222],
        [-0.0015, 0.11293, -0.02498],
        [0.05887, -0.41644, -0.00656],
        [-0.04807, -0.39756, -0.01406],
        [0.0069, 0.14564, -0.00686],
        [-0.04174, -0.43758, -0.02951],
        [0.01449, -0.44685, -0.01803],
        [-0.01033, 0.05608, 0.02112],
        [0.04929, -0.06528, 0.12626],
        [-0.04058, -0.06529, 0.12708],
        [-0.01103, 0.17137, -0.02883],
        [0.04772, 0.08764, -0.00838],
        [-0.04664, 0.08661, -0.01486],
        [0.02465, 0.17539, 0.02446],
        [0.12628, 0.05768, -0.01389],
        [-0.10934, 0.05367, -0.00912],
        [0.27291, -0.06985, -0.03909],
        [-0.29203, -0.03544, -0.02456],
        [0.27617, 0.02125, -0.00248],
        [-0.27188, -0.00483, -0.01645],
    ]
)


@dataclass
class Motion:
    """22-joint SMPL-X body motion in ROS axes."""

    global_rots: np.ndarray  # (T, 22, 3, 3): body-frame -> ROS world, i.e. world_to_ros @ R_global @ BODY_TO_ROS.T
    joints: np.ndarray  # (T, 22, 3) positions in the ROS world (m)
    fps: float
    rest_offsets: np.ndarray  # (22, 3) joint offsets from the parent in the canonical body frame (row 0 unused)

    def __len__(self) -> int:
        return len(self.global_rots)

    def rest_vector(self, joint: str, descendant: str) -> np.ndarray:
        """Rest-pose vector joint -> descendant (e.g. hip -> ankle), in ROS axes of the joint's own frame. Rest
        rotations are the identity, so it is the sum of the offsets along the chain (constant over the clip)."""
        v, j = np.zeros(3), J[descendant]
        while j != J[joint]:
            if j <= 0:
                raise ValueError(f"{descendant} is not below {joint}")
            v, j = v + self.rest_offsets[j], PARENTS[j]
        return BODY_TO_ROS @ v


def forward_kinematics(local_rots: np.ndarray, root_pos: np.ndarray, offsets: np.ndarray = REST_OFFSETS) -> tuple[np.ndarray, np.ndarray]:
    """Canonical-frame FK: local (T, 22, 3, 3) + pelvis position (T, 3) -> global rotations and joint positions."""
    T = len(local_rots)
    G = np.zeros((T, 22, 3, 3))
    P = np.zeros((T, 22, 3))
    for j, p in enumerate(PARENTS):
        if p < 0:
            G[:, j], P[:, j] = local_rots[:, j], root_pos
        else:
            G[:, j] = G[:, p] @ local_rots[:, j]
            P[:, j] = P[:, p] + np.einsum("tij,j->ti", G[:, p], offsets[j])
    return G, P


def rest_offsets_from_model(model_path: str | Path, betas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rest skeleton of an SMPL-X model file (``SMPLX_NEUTRAL.npz`` etc.) for the given shape.

    Returns (offsets (22, 3), pelvis rest position (3,)). Only numpy is needed, not the ``smplx`` package.
    """
    m = np.load(model_path, allow_pickle=True)
    v = m["v_template"].astype(float)
    if betas is not None and len(betas):
        b = np.asarray(betas, dtype=float).ravel()
        dirs = m["shapedirs"][:, :, : min(len(b), m["shapedirs"].shape[2])].astype(float)
        v = v + dirs @ b[: dirs.shape[2]]
    reg = m["J_regressor"]
    reg = reg.toarray() if issparse(reg) else np.asarray(reg, dtype=float)
    joints = reg[:22] @ v
    off = np.zeros((22, 3))
    for j, p in enumerate(PARENTS):
        if p >= 0:
            off[j] = joints[j] - joints[p]
    return off, joints[0]


def _first(z: NpzLike, *keys: str) -> np.ndarray | None:
    return next((z[k] for k in keys if k in z.files), None)


class _Keys:
    """An npz's arrays with one person's key prefix stripped (``p0_trans`` -> ``trans``); shared keys stay visible."""

    def __init__(self, z: np.lib.npyio.NpzFile, prefix: str) -> None:
        self._z = z
        self._keys = {k: k for k in z.files if not k.startswith(PERSON_PREFIXES)}
        self._keys.update({k[len(prefix) :]: k for k in z.files if k.startswith(prefix)})
        if not any(k.startswith(prefix) for k in z.files):
            raise ValueError(f"no keys with prefix {prefix!r} (keys: {', '.join(z.files)})")
        self.files = list(self._keys)

    def __getitem__(self, k: str) -> np.ndarray:
        return self._z[self._keys[k]]


NpzLike = np.lib.npyio.NpzFile | _Keys  # a loaded npz, or one person's view of it


PERSON_PREFIXES = ("p0_", "p1_")


def load(path: str | Path, up: str = "auto", fps: float | None = None, model: str | Path | None = None, person: int | None = None) -> Motion:
    """Read an SMPL-X motion file (see the module doc for the formats) into a :class:`Motion`.

    ``up``: the world's up axis, "y" or "z"; "auto" takes Y for rotation outputs (Kimodo) and Z for parameter files
    (AMASS). ``fps`` overrides the file's frame rate (default: the file's, else 30). ``model``: an SMPL-X model file for
    the exact rest skeleton (with the file's ``betas``); without it the built-in Kimodo body is used where needed.
    ``person``: read one person of a two-person file, whose arrays carry a ``p0_`` / ``p1_`` key prefix (the
    InterMask export, see :mod:`.pair`).
    """
    z = np.load(path)
    if person is not None:
        z = _Keys(z, PERSON_PREFIXES[person])
    if up == "auto" and "up" in z.files:  # a file may name its own world up axis
        up = str(z["up"]).lower()
    fps = float(fps or _first(z, "fps", "mocap_frame_rate", "mocap_framerate") or 30.0)
    offsets, pelvis0 = REST_OFFSETS, np.zeros(3)
    if model is not None:
        offsets, pelvis0 = rest_offsets_from_model(model, _first(z, "betas"))

    if "global_rot_mats" in z.files or "local_rot_mats" in z.files:
        up = "y" if up == "auto" else up
        if "global_rot_mats" in z.files and "posed_joints" in z.files:
            G = z["global_rot_mats"][:, :22].astype(float)
            P = z["posed_joints"][:, :22].astype(float)
            if model is None:  # measure the rest skeleton from the clip itself
                offsets = np.zeros((22, 3))
                for j, p in enumerate(PARENTS[1:], 1):
                    offsets[j] = G[0, p].T @ (P[0, j] - P[0, p])
        else:
            L = z["local_rot_mats"][:, :22].astype(float)
            root = _first(z, "root_positions", "trans", "transl")
            root = np.zeros((len(L), 3)) if root is None else root.astype(float)
            G, P = forward_kinematics(L, root, offsets)
    else:
        up = "z" if up == "auto" else up
        poses = _first(z, "poses", "fullpose")
        if poses is not None:
            aa = poses.reshape(len(poses), -1, 3)[:, :22].astype(float)
        else:
            root_aa = _first(z, "root_orient", "global_orient")
            body_aa = _first(z, "pose_body", "body_pose")
            if root_aa is None or body_aa is None:
                raise ValueError(f"{path}: no SMPL-X rotations found (keys: {', '.join(z.files)})")
            aa = np.concatenate([root_aa.reshape(-1, 1, 3), body_aa.reshape(len(body_aa), -1, 3)[:, :21]], 1).astype(float)
        T = len(aa)
        L = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
        trans = _first(z, "trans", "transl")
        trans = np.zeros((T, 3)) if trans is None else trans.reshape(T, 3).astype(float)
        G, P = forward_kinematics(L, trans + pelvis0, offsets)

    if up not in WORLD_TO_ROS:
        raise ValueError(f"up must be 'y' or 'z', got {up!r}")
    W = WORLD_TO_ROS[up]
    return Motion(global_rots=np.einsum("ij,tkjl,ml->tkim", W, G, BODY_TO_ROS), joints=P @ W.T, fps=fps, rest_offsets=np.asarray(offsets, dtype=float))
