"""The torch FK is the contract FK: same positions and frames as skeleton.fk on random poses."""

import numpy as np
import torch

from arena_simulation_setup.adapters.arena_animation_retarget.contract import SK, WIRE
from arena_simulation_setup.adapters.arena_animation_retarget.fk_torch import ContractFK


def test_matches_contract_fk():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-1.5, 1.5, (16, len(WIRE)))
    fk = ContractFK(1.7, dtype=torch.float64)
    pos, fr = fk(torch.as_tensor(theta))
    body = SK.Body(1.7)
    names = {"l_hand": "l_hand", "r_hand": "r_hand", "head": "head", "l_foot": "l_foot", "r_foot": "r_foot"}
    for i, th in enumerate(theta):
        p_ref, f_ref = SK.fk(dict(zip(WIRE, th)), body)
        for k in ("pelvis", "torso", "neck", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_foot", "r_foot"):
            assert np.allclose(pos[k][i].numpy(), p_ref[k], atol=1e-9), k
        for k in names:
            assert np.allclose(fr[k][i].numpy(), f_ref[k], atol=1e-9), k
        for s in "lr":
            assert np.allclose(fr[f"{s}_upperarm"][i].numpy(), f_ref[f"{s}_upperarm"], atol=1e-9)
            assert np.allclose(fr[f"{s}_forearm"][i].numpy(), f_ref[f"{s}_forearm"], atol=1e-9)
