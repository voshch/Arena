"""The arena_peds wire: a ped's body pose (BODY.md) from its semantic joint angles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from arena_people_msgs.msg import BodyPose, Pedestrian
from arena_simulation_setup.shared.body import POSED_JOINTS, SEMANTIC_DOFS, CompiledStance, semantic_to_bodies
from geometry_msgs.msg import Quaternion, Vector3

from arena_runtime.humans.animation_manager import BodyLayer, nlerp

_SEMANTIC = frozenset(SEMANTIC_DOFS)
_POSED_INDEX = {joint: i for i, joint in enumerate(POSED_JOINTS)}


def fill_body(ped: Pedestrian, stance: CompiledStance, layers: Sequence[BodyLayer] = ()) -> None:
    """Move the semantic DOFs of a ped's joint_state into its body pose, joint_state keeping only the driven joints, clip body groups blended over it."""
    fill_bodies([ped], [stance], [layers])


def fill_bodies(peds: Sequence[Pedestrian], stances: Sequence[CompiledStance], layers: Sequence[Sequence[BodyLayer]]) -> None:
    """`fill_body` of every ped in one batch, a ped without semantic DOFs left as it is."""
    posed: list[tuple[Pedestrian, Sequence[BodyLayer]]] = []
    rows: list[list[float]] = []
    posed_stances: list[CompiledStance] = []
    for ped, stance, ped_layers in zip(peds, stances, layers, strict=True):
        angles = dict(zip(ped.joint_state.name, ped.joint_state.position, strict=True))
        if _SEMANTIC.isdisjoint(angles):
            continue
        rows.append([float(angles.get(dof, 0.0)) for dof in SEMANTIC_DOFS])
        posed.append((ped, ped_layers))
        posed_stances.append(stance)
        driven = {name: value for name, value in angles.items() if name not in _SEMANTIC}
        ped.joint_state.name, ped.joint_state.position = list(driven), list(driven.values())
    if not posed:
        return
    for (ped, ped_layers), quats in zip(posed, semantic_to_bodies(np.array(rows), posed_stances), strict=True):
        pelvis = np.zeros(3)
        for layer in ped_layers:
            for joint, (quat, weight) in layer.rotations.items():
                i = _POSED_INDEX[joint]
                quats[i] = nlerp(quats[i].tolist(), quat, weight)
            if layer.pelvis is not None:
                pelvis = (1.0 - layer.weight) * pelvis + layer.weight * layer.pelvis
        ped.body = BodyPose(
            joints=list(POSED_JOINTS),
            rotations=[Quaternion(x=x, y=y, z=z, w=w) for x, y, z, w in quats.tolist()],
            root_offset=Vector3(x=float(pelvis[0]), y=float(pelvis[1]), z=float(pelvis[2])),
        )
