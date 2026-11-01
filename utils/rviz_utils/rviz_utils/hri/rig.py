"""Body pose of a pedestrian on the wire as semantic angles or as joint positions of its RViz body."""

from __future__ import annotations

import functools
import math
import typing
from collections.abc import Mapping

import numpy as np
from arena_simulation_setup.shared.body import DEFAULT_STANCE, body_to_semantic, load_stance, semantic_to_body

from .urdf import body_positions

if typing.TYPE_CHECKING:
    from arena_people_msgs.msg import Pedestrian


def _rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula about an arbitrary (non-unit) axis."""
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(theta), math.sin(theta)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


@functools.cache
def _stance(model_uri: str) -> Mapping[str, np.ndarray]:
    """Stance of a ped model, the default one when its rig file is rejected."""
    try:
        return load_stance(model_uri)
    except (OSError, ValueError):
        return DEFAULT_STANCE


def wire_rotations(ped: Pedestrian) -> dict[str, tuple[float, float, float, float]]:
    """Local x y z w rotation of each joint of a ped's body pose."""
    return {joint: (q.x, q.y, q.z, q.w) for joint, q in zip(ped.body.joints, ped.body.rotations, strict=True)}


def wire_angles(ped: Pedestrian) -> dict[str, float]:
    """The semantic angles of a ped's body pose, the digits included."""
    return body_to_semantic(wire_rotations(ped), _stance(ped.model_uri))


def ped_positions(ped: Pedestrian, body_id: str) -> dict[str, float]:
    """Joint positions of a ped's RViz body from its body pose."""
    offset = ped.body.root_offset
    return body_positions(body_id, wire_rotations(ped), (offset.x, offset.y, offset.z))


def semantic_positions(angles: Mapping[str, float], model_uri: str, body_id: str) -> dict[str, float]:
    """Joint positions of an RViz body posed by semantic angles over a ped model's stance."""
    return body_positions(body_id, semantic_to_body(angles, _stance(model_uri)))
