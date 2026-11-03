"""The arena_peds wire: a ped's body pose (BODY.md) from its 40 semantic joint angles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from arena_people_msgs.msg import BodyPose, Pedestrian
from arena_simulation_setup.shared.body import SEMANTIC_DOFS, semantic_to_body
from geometry_msgs.msg import Quaternion, Vector3

from arena_runtime.humans.animation_manager import BodyLayer, nlerp

_SEMANTIC = frozenset(SEMANTIC_DOFS)


def fill_body(ped: Pedestrian, stance: Mapping[str, np.ndarray], layers: Sequence[BodyLayer] = ()) -> None:
    """Move the semantic DOFs of a ped's joint_state into its body pose, joint_state keeping only the driven joints, clip body groups blended over it."""
    angles = dict(zip(ped.joint_state.name, ped.joint_state.position, strict=True))
    semantic = {name: value for name, value in angles.items() if name in _SEMANTIC}
    if not semantic:
        return
    rotations = semantic_to_body(semantic, stance)
    pelvis = np.zeros(3)
    for layer in layers:
        for joint, (quat, weight) in layer.rotations.items():
            rotations[joint] = np.asarray(nlerp(rotations[joint].tolist(), quat, weight))
        if layer.pelvis is not None:
            pelvis = (1.0 - layer.weight) * pelvis + layer.weight * layer.pelvis
    ped.body = BodyPose(
        joints=list(rotations),
        rotations=[Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])) for q in rotations.values()],
        root_offset=Vector3(x=float(pelvis[0]), y=float(pelvis[1]), z=float(pelvis[2])),
    )
    driven = {name: value for name, value in angles.items() if name not in _SEMANTIC}
    ped.joint_state.name, ped.joint_state.position = list(driven), list(driven.values())
