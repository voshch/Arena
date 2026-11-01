"""Pedestrian animation clips: one joint-angle recording per asset directory."""

from __future__ import annotations

import functools
import json
import logging
import typing
from collections.abc import Iterator, Mapping, Sequence
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
ANIM_SUFFIX = '.anim'
ANIM_READER = 1  # highest anim/requires this code plays
_CLIP_ARRAYS = ('joint_names', 'angles', 't', 'root_xy_yaw', 'animation_state')

logger = logging.getLogger(__name__)
logger.propagate = True


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
    root_quat: np.ndarray | None = None  # (T, 4) xyzw pelvis orientation from the start heading, clips annotated pelvis_tilt
    legacy: bool = False  # read from the old <name>.npz

    @property
    def fps(self) -> float:
        return float(self.meta.get('fps', 20.0))

    def frames(self, joints: Sequence[str] = ()) -> list[dict]:
        """Per-frame dicts in the shape the players consume, `joints` the clip lacks filled with 0.0."""
        return [
            {
                't': float(self.t[k]),
                'angles': {**dict.fromkeys(joints, 0.0), **dict(zip(self.joint_names, self.angles[k].tolist(), strict=True))},
                'root_xy_yaw': self.root_xy_yaw[k].tolist(),
                'animation_state': int(self.animation_state[k]),
                **({'root_quat': self.root_quat[k].tolist()} if self.root_quat is not None else {}),
            }
            for k in range(len(self.t))
        ]


def from_start_heading(root_quat: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Pelvis orientations (T, 4 xyzw) given in each frame's heading frame, turned into the first frame's.

    A lying pelvis has no clear heading, so the per-frame heading can jump; the orientation it leaves is smooth.
    """
    half = 0.5 * (np.asarray(yaw, dtype=float) - float(yaw[0]))
    s, c = np.sin(half), np.cos(half)  # yaw quaternion (0, 0, s, c)
    x, y, z, w = np.asarray(root_quat, dtype=float).T
    q = np.stack([c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z], axis=1)
    return q


def write_anim(path: Path, arrays: Mapping[str, np.ndarray], settings: Mapping[str, object] | None = None, requires: int = 1) -> Path:
    """Write a .anim: `group/member` arrays, settings that override annotation.yaml, the oldest reader that can play it."""
    members: dict[str, typing.Any] = {'anim/requires': np.array(requires), 'anim/settings': np.array(json.dumps(dict(settings or {}))), **arrays}
    with path.open('wb') as file:
        np.savez_compressed(file, **members)
    return path


def add_anim_group(path: Path, group: str, arrays: Mapping[str, np.ndarray]) -> Path:
    """Add or replace one group of a .anim, every other member kept."""
    with np.load(path, allow_pickle=False) as z:
        members: dict[str, typing.Any] = {key: z[key] for key in z.files if not key.startswith(f'{group}/')}
    members |= {f'{group}/{key}': value for key, value in arrays.items()}
    with path.open('wb') as file:
        np.savez_compressed(file, **members)
    return path


def pack_anim(source: Path, target: Path, settings: Mapping[str, object] | None = None) -> Path:
    """Write the .anim of the .npz clip at source/ into target/, its table.npz as the table group."""
    name = source.name
    with np.load(source / f'{name}.npz', allow_pickle=False) as z:
        arrays = {f'clip/{key}': z[key] for key in _CLIP_ARRAYS}
        arrays |= {f'root/{key}': z[f'root_{key}'] for key in ('quat', 'z') if f'root_{key}' in z.files}
    if (source / 'table.npz').is_file():
        with np.load(source / 'table.npz', allow_pickle=False) as z:
            arrays |= {f'table/{key}': z[key] for key in z.files}
    target.mkdir(parents=True, exist_ok=True)
    return write_anim(target / f'{name}{ANIM_SUFFIX}', arrays, settings)


class AnimationView(PathView):
    """View around a resolved Animation asset directory."""

    @cached_property
    def anim(self) -> Path | None:
        """The clip's .anim, None without one this code can play."""
        path = self.path / f'{self.path.name}{ANIM_SUFFIX}'
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as z:
            requires = int(z['anim/requires'])
        if requires > ANIM_READER:
            logger.warning('%s needs reader %d, this code is %d: update Arena', path, requires, ANIM_READER)
            return None
        return path

    @cached_property
    def meta(self) -> dict:
        """The annotation, the .anim's settings over it, the annotation's `anim` block over those."""
        annotation = yaml.safe_load((self.path / ANNOTATION).read_text()) or {}
        if self.anim is None:
            return annotation
        with np.load(self.anim, allow_pickle=False) as z:
            return annotation | json.loads(str(z['anim/settings'])) | annotation.get('anim', {})

    @cached_property
    def clip(self) -> AnimationClip:
        """The clip; its root_quat only when the annotation says the body is pelvis-relative (pelvis_tilt), as clips
        that fold the tilt into the waist and hips carry one too."""
        legacy = self.anim is None
        clip_key, quat_key = ('{}', 'root_quat') if legacy else ('clip/{}', 'root/quat')
        with np.load(self.anim or self.path / f'{self.path.name}.npz', allow_pickle=False) as z:
            tilted = bool(self.meta.get('pelvis_tilt')) and quat_key in z.files
            root_xy_yaw = z[clip_key.format('root_xy_yaw')]
            return AnimationClip(
                name=self.path.name,
                joint_names=tuple(str(j) for j in z[clip_key.format('joint_names')]),
                angles=z[clip_key.format('angles')],
                t=z[clip_key.format('t')],
                root_xy_yaw=root_xy_yaw,
                animation_state=z[clip_key.format('animation_state')],
                meta=self.meta,
                root_quat=from_start_heading(z[quat_key], root_xy_yaw[:, 2]) if tilted else None,
                legacy=legacy,
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
    tilted = any('root_quat' in f for f in frames)
    root_xy_yaw = np.array([f.get('root_xy_yaw', (0.0, 0.0, 0.0)) for f in frames], dtype=np.float64).reshape(len(frames), 3)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        'clip/joint_names': np.array(joint_names),
        'clip/angles': np.array([[f['angles'].get(j, 0.0) for j in joint_names] for f in frames], dtype=np.float64),
        'clip/t': np.array([f.get('t', k / meta.get('fps', 20.0)) for k, f in enumerate(frames)], dtype=np.float64),
        'clip/root_xy_yaw': root_xy_yaw,
        'clip/animation_state': np.array([f.get('animation_state', 0) for f in frames], dtype=np.int16),
    }
    if tilted:
        # stored per frame heading, as the retarget writes it: the inverse of the loader's turn
        arrays['root/quat'] = from_start_heading(np.array([f.get('root_quat', (0.0, 0.0, 0.0, 1.0)) for f in frames], dtype=np.float64), -root_xy_yaw[:, 2])
    write_anim(directory / f'{name}{ANIM_SUFFIX}', arrays)
    (directory / ANNOTATION).write_text(yaml.safe_dump({'name': name, 'path': f'Animation/{name}', **meta, **({'pelvis_tilt': True} if tilted else {})}, sort_keys=False))
    return directory


def referenced_clips(config: object) -> Iterator[str]:
    """Clip names a humansim agent config plays: every attention clip plus the default clip of each interaction step."""
    if isinstance(config, list):
        for item in config:
            yield from referenced_clips(item)
        return
    if not isinstance(config, dict):
        return
    for key, value in config.items():
        if key == 'clip':
            name = value.get('name') if isinstance(value, dict) else value
            if isinstance(name, str) and name:
                yield name
        else:
            yield from referenced_clips(value)
    interaction, attention = config.get('interaction'), config.get('attention')
    if isinstance(interaction, str) and not (isinstance(attention, dict) and attention.get('clip') is not None):
        if default := _interaction_clip(interaction):
            yield default


@functools.cache
def _interaction_clip(interaction: str) -> str | None:
    """The clip an interaction kind plays while bound (humansim's InteractionKind.clip), None without arena_humansim."""
    try:
        from arena_humansim.core.interaction_kinds import InteractionType
    except ImportError:
        return None
    try:
        return InteractionType[interaction].kind.clip or None
    except KeyError:
        return None


AnimationIdentifier.use(*DynamicPaths.as_resolvers(AnimationIdentifier))
AnimationIdentifier.use(*NetResolver.all(AnimationIdentifier, formats=(ANIM_SUFFIX[1:], 'yaml')))
