"""Pedestrian animation clips: one joint-angle recording per asset directory.

    <domain>/Animation/<name>/
        annotation.yaml   playback (fps, loop, reverse, joints) and provenance, also the bucket's listing sentinel
        <name>.npz        joint_names (J,), angles (T, J), t (T,), root_xy_yaw (T, 3), animation_state (T,)
        <extra>.npz       optional data derived from the clip, e.g. the baked pointing table (table.npz)

Clips never ship in the repo: they resolve from the world/local asset dirs (work in progress, gitignored) and the
asset bucket (`arena asset push animation <name>` publishes one).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import attrs
import numpy as np
import yaml

from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    DynamicPaths,
    NetResolver,
    PathView,
)

ANNOTATION = 'annotation.yaml'


@attrs.frozen(eq=False)
class AnimationClip:
    """A clip's arrays and its annotation."""

    name: str
    joint_names: tuple[str, ...]
    angles: np.ndarray  # (T, J) radians, columns follow joint_names
    t: np.ndarray  # (T,) seconds
    root_xy_yaw: np.ndarray  # (T, 3)
    animation_state: np.ndarray  # (T,) Pedestrian.msg state
    meta: dict

    @property
    def fps(self) -> float:
        return float(self.meta.get('fps', 20.0))

    def frames(self) -> list[dict]:
        """Per-frame dicts ({'t', 'angles': {joint: rad}, 'root_xy_yaw', 'animation_state'}), the shape the players consume."""
        return [
            {
                't': float(self.t[k]),
                'angles': dict(zip(self.joint_names, self.angles[k].tolist(), strict=True)),
                'root_xy_yaw': self.root_xy_yaw[k].tolist(),
                'animation_state': int(self.animation_state[k]),
            }
            for k in range(len(self.t))
        ]


class AnimationView(PathView):
    """View around a resolved Animation asset directory."""

    @cached_property
    def meta(self) -> dict:
        return yaml.safe_load((self.path / ANNOTATION).read_text()) or {}

    @cached_property
    def clip(self) -> AnimationClip:
        with np.load(self.path / f'{self.path.name}.npz', allow_pickle=False) as z:
            return AnimationClip(
                name=self.path.name,
                joint_names=tuple(str(j) for j in z['joint_names']),
                angles=z['angles'],
                t=z['t'],
                root_xy_yaw=z['root_xy_yaw'],
                animation_state=z['animation_state'],
                meta=self.meta,
            )

    def extra(self, filename: str) -> Path:
        """A derived file stored next to the clip."""
        return self.path / filename


@attrs.define(eq=False, hash=False)
class AnimationIdentifier(DomainAssetIdentifier[AnimationView]):
    """Represents an identifier referencing an animation clip."""

    _asset_type = 'Animation'

    def load(self, path: Path, /, **kwargs: object) -> AnimationView:
        del kwargs  # unused
        return AnimationView(path)


def write_clip(directory: Path, frames: Sequence[dict], meta: dict) -> Path:
    """Write per-frame dicts (the players' shape) as an Animation asset at directory/, named after it."""
    name = directory.name
    joint_names = sorted({j for f in frames for j in f['angles']})
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f'{name}.npz',
        joint_names=np.array(joint_names),
        # a joint missing from a frame (recordings that predate a DOF) reads as 0.0
        angles=np.array([[f['angles'].get(j, 0.0) for j in joint_names] for f in frames], dtype=np.float64),
        t=np.array([f.get('t', k / meta.get('fps', 20.0)) for k, f in enumerate(frames)], dtype=np.float64),
        root_xy_yaw=np.array([f.get('root_xy_yaw', (0.0, 0.0, 0.0)) for f in frames], dtype=np.float64).reshape(len(frames), 3),
        animation_state=np.array([f.get('animation_state', 0) for f in frames], dtype=np.int16),
    )
    (directory / ANNOTATION).write_text(yaml.safe_dump({'name': name, 'path': f'Animation/{name}', **meta}, sort_keys=False))
    return directory


AnimationIdentifier.use(*DynamicPaths.as_resolvers(AnimationIdentifier))
AnimationIdentifier.use(*NetResolver.all(AnimationIdentifier, formats=()))
