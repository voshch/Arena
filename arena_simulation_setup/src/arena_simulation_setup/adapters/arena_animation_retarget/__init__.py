"""Retarget SMPL-X body motion onto Arena's 40-DOF human wire clips (rotation retarget + whole-body IK on the GPU).

from arena_simulation_setup.adapters.arena_animation_retarget import load, retarget, retarget_batch, write_clip
clip = retarget(load("wave.npz"))                       # one motion
clips = retarget_batch([load(f) for f in files])        # many, one batched IK
write_clip(clip, "arena_simulation_setup/assets/Common/Animation", "wave_smplx", loop=True)
"""

from .clip import load_clip, write_clip
from .retarget import Clip, prepare, retarget, retarget_batch
from .smplx import Motion, load

__all__ = ["Clip", "Motion", "load", "load_clip", "prepare", "retarget", "retarget_batch", "write_clip"]
