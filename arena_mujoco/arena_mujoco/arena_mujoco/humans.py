"""Pedestrian visuals: an actor's skinned COLLADA mesh as MuJoCo skins on bones posed from its clips."""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from PIL import Image

from arena_mujoco.bone_map import BONE_MAP
from arena_mujoco.mesh_mat import MeshPart, cache_entry, part_material

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import collada

    from arena_mujoco.scene import AssetCache

_TEXTURE_PX = 512
_MIN_OPAQUE = 0.25
_RIG_FILE = 'rig.npz'


@dataclass(frozen=True)
class Actor:
    """Files and offset an actor SDF declares."""

    skin: Path
    clips: dict[str, Path]
    z_offset: float


def read_actor(sdf_path: str) -> Actor:
    """Skin mesh, clips by animation name and vertical offset of an <actor> SDF."""
    sdf = Path(sdf_path)
    actor = ET.parse(sdf).getroot().find('actor')
    if actor is None:
        raise ValueError(f'{sdf_path} holds no <actor>')
    skin = actor.findtext('skin/filename')
    if not skin:
        raise ValueError(f'{sdf_path} names no skin mesh')
    clips = {clip.get('name', ''): sdf.parent / file for clip in actor.findall('animation') if (file := clip.findtext('filename'))}
    pose = actor.findtext('pose', '0 0 0 0 0 0').split()
    return Actor(skin=sdf.parent / skin, clips=clips, z_offset=float(pose[2]))


def clip_frames(clip_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Key times of a COLLADA clip and its local joint transforms (keys, 4, 4) by joint name."""
    root = ET.parse(clip_path).getroot()
    namespace = root.tag.partition('}')[0] + '}' if root.tag.startswith('{') else ''
    by_id = {element.get('id'): element for element in root.iter() if element.get('id')}
    times = np.zeros(0)
    frames: dict[str, np.ndarray] = {}
    for channel in root.iter(f'{namespace}channel'):
        joint, _, attribute = channel.get('target', '').partition('/')
        if attribute != 'transform':
            continue
        sampler = by_id[channel.get('source', '').lstrip('#')]
        arrays = {entry.get('semantic'): by_id[entry.get('source', '').lstrip('#')].find(f'{namespace}float_array') for entry in sampler.findall(f'{namespace}input')}
        frames[joint] = np.array(arrays['OUTPUT'].text.split(), dtype=np.float64).reshape(-1, 4, 4)
        if not len(times):
            times = np.array(arrays['INPUT'].text.split(), dtype=np.float64)
    keys = min([len(times), *(len(matrices) for matrices in frames.values())])
    return times[:keys], {joint: matrices[:keys] for joint, matrices in frames.items()}


@dataclass(frozen=True)
class Clip:
    """Bone poses per key in the actor's model frame, the root's steady advance taken out."""

    times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    stride: float

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    def pose(self, cursor: float) -> tuple[np.ndarray, np.ndarray]:
        """Bone positions (bones, 3) and quaternions (bones, 4) cursor seconds into the looping clip."""
        if self.duration <= 0.0:
            return self.positions[0], self.quaternions[0]
        at = self.times[0] + cursor % self.duration
        high = min(int(np.searchsorted(self.times, at, side='right')), len(self.times) - 1)
        low = high - 1
        blend = (at - self.times[low]) / (self.times[high] - self.times[low])
        quaternions = (1.0 - blend) * self.quaternions[low] + blend * self.quaternions[high]
        return (
            (1.0 - blend) * self.positions[low] + blend * self.positions[high],
            quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True),
        )


@dataclass(frozen=True)
class SkinPart:
    """Triangles of one material in bind pose, with the vertices and weights each bone moves."""

    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None
    rgba: tuple[float, float, float, float] | None
    texture: str | None
    bone_vertices: tuple[np.ndarray, ...]
    bone_weights: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Rig:
    """An actor ready for MuJoCo: bone names, one skin part per material, its clips and its skeleton in the neutral stance."""

    bones: tuple[str, ...]
    parts: tuple[SkinPart, ...]
    clips: dict[str, Clip]
    z_offset: float
    node_parents: tuple[int, ...]
    neutral: np.ndarray
    bone_nodes: tuple[int, ...]
    inverse_binds: np.ndarray
    bind_positions: np.ndarray
    scale: float

    @property
    def layout(self) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
        """What two rigs must share to be posed in one batch."""
        return self.bones, self.node_parents, self.bone_nodes


@dataclass(frozen=True)
class _Node:
    parent: int
    rest: np.ndarray
    joint: str | None
    root: bool


def _skeleton(dae: collada.Collada) -> list[_Node]:
    """Scene nodes parent first, the topmost joint of each branch marked as root."""
    import collada

    nodes: list[_Node] = []

    def descend(node: collada.scene.SceneNode, parent: int, below_root: bool) -> None:
        if not isinstance(node, collada.scene.Node):
            return
        joint = node.xmlnode.get('type') == 'JOINT'
        nodes.append(_Node(parent, np.asarray(node.matrix, dtype=np.float64), (node.xmlnode.get('sid') or node.id) if joint else None, joint and not below_root))
        index = len(nodes) - 1
        for child in node.children:
            descend(child, index, below_root or joint)

    for node in dae.scene.nodes:
        descend(node, -1, False)
    return nodes


def _locals(skeleton: Sequence[_Node], frames: dict[str, np.ndarray], keys: int) -> tuple[np.ndarray, np.ndarray]:
    """Local transforms (nodes, keys, 4, 4) with the root's steady advance removed, plus that advance in the root's parent frame."""
    transforms = np.empty((len(skeleton), keys, 4, 4))
    advance = np.zeros(3)
    for index, node in enumerate(skeleton):
        clip = frames.get(node.joint) if node.joint is not None else None
        if clip is None:
            transforms[index] = node.rest
        elif node.root:
            offsets = clip[:, :3, 3] - clip[0, :3, 3]
            advance = offsets[-1]
            transforms[index] = clip
            transforms[index, :, :3, 3] = node.rest[:3, 3] + offsets - np.linspace(0.0, 1.0, keys)[:, None] * advance
        else:
            transforms[index] = clip
    return transforms, advance


def _worlds(parents: Sequence[int], transforms: np.ndarray, roots: np.ndarray | None = None) -> np.ndarray:
    """World transforms of parent-first nodes (nodes, n, 4, 4) from their local ones, top nodes standing at roots (n, 4, 4)."""
    worlds = np.empty_like(transforms)
    for index, parent in enumerate(parents):
        if parent >= 0:
            worlds[index] = worlds[parent] @ transforms[index]
        elif roots is not None:
            worlds[index] = roots @ transforms[index]
        else:
            worlds[index] = transforms[index]
    return worlds


def _stride(skeleton: Sequence[_Node], worlds: np.ndarray, advance: np.ndarray) -> float:
    """Length of the root's advance in the model frame."""
    for node in skeleton:
        if node.root:
            return float(np.linalg.norm(worlds[node.parent, 0, :3, :3] @ advance)) if node.parent >= 0 else float(np.linalg.norm(advance))
    return 0.0


def _orthonormal(linear: np.ndarray) -> np.ndarray:
    """Nearest rotation matrices to a stack of 3x3 blocks (..., 3, 3)."""
    u, _, vt = np.linalg.svd(linear)
    return u @ vt


def _rigid(matrices: np.ndarray) -> tuple[np.ndarray, float]:
    """wxyz quaternions and the common scale of a stack of similarity transforms (..., 4, 4)."""
    linear = matrices[..., :3, :3]
    scale = float(np.median(np.cbrt(np.linalg.det(linear))))
    rotations = np.ascontiguousarray(_orthonormal(linear)).reshape(-1, 9)
    quaternions = np.empty((len(rotations), 4))
    for quaternion, rotation in zip(quaternions, rotations, strict=True):
        mujoco.mju_mat2Quat(quaternion, rotation)
    return quaternions.reshape(*matrices.shape[:-2], 4), scale


def _continuous(quaternions: np.ndarray) -> np.ndarray:
    """Quaternions (keys, bones, 4) with signs flipped so consecutive keys of a bone never point apart."""
    flips = np.sum(quaternions[1:] * quaternions[:-1], axis=-1) < 0.0
    signs = np.where(np.logical_xor.accumulate(flips, axis=0), -1.0, 1.0)
    return np.concatenate([quaternions[:1], quaternions[1:] * signs[..., None]])


def _diffuse(material: collada.material.Material, mesh_dir: Path) -> tuple[tuple[float, float, float, float] | None, Image.Image | None]:
    """Diffuse color or image of a COLLADA material, both None for an image that is mostly transparent."""
    import collada

    diffuse = material.effect.diffuse
    if isinstance(diffuse, collada.material.Map):
        image = Image.open(mesh_dir / diffuse.sampler.surface.image.path)
        if 'A' in image.getbands() and (np.asarray(image.getchannel('A')) > 127).mean() < _MIN_OPAQUE:
            return None, None
        return (1.0, 1.0, 1.0, 1.0), image
    if diffuse is None:
        return None, None
    return tuple(float(channel) for channel in diffuse), None


def _write_rig(out_dir: Path, stem: str, actor: Actor) -> None:
    """Write the actor's skin parts, bone influences and clip poses as rig.npz into out_dir, textures beside it."""
    import collada

    dae = collada.Collada(str(actor.skin), ignore=[collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError])
    (skin,) = dae.controllers
    skeleton = _skeleton(dae)
    bones = [str(name) for name in skin.weight_joints.data[:, 0]]
    inverse_binds = np.stack([np.asarray(skin.joint_matrices[name], dtype=np.float64) for name in bones])

    parents = [node.parent for node in skeleton]
    bone_nodes = np.array([next(index for index, node in enumerate(skeleton) if node.joint == bone) for bone in bones])

    arrays: dict[str, np.ndarray] = {'bones': np.array(bones), 'clips': np.array(sorted(actor.clips) or ['rest'])}
    clips = {name: clip_frames(path) for name, path in actor.clips.items()} or {'rest': (np.zeros(1), {})}
    scale = 1.0
    for name, (times, frames) in clips.items():
        transforms, advance = _locals(skeleton, frames, len(times))
        worlds = _worlds(parents, transforms)
        joints = np.moveaxis(worlds[bone_nodes], 0, 1)
        quaternions, scale = _rigid(joints @ inverse_binds)
        arrays[f'clip_{name}_times'] = times
        arrays[f'clip_{name}_positions'] = joints[..., :3, 3]
        arrays[f'clip_{name}_quaternions'] = _continuous(quaternions)
        arrays[f'clip_{name}_stride'] = np.array(_stride(skeleton, worlds, advance))
    _, stance = clips.get('idle') or next(iter(clips.values()))
    roots = {node.joint for node in skeleton if node.root}
    arrays['node_parents'] = np.array(parents)
    neutral = _locals(skeleton, {joint: matrices[:1] for joint, matrices in stance.items() if joint not in roots}, 1)[0][:, 0]
    jointed = [index for index, node in enumerate(skeleton) if node.joint is not None]
    neutral[jointed, :3, :3] = _orthonormal(neutral[jointed, :3, :3])
    arrays['neutral'] = neutral
    arrays['bone_nodes'] = bone_nodes
    arrays['inverse_binds'] = inverse_binds
    arrays['bind_positions'] = np.linalg.inv(inverse_binds)[:, :3, 3] * scale
    arrays['scale'] = np.array(scale)

    rest = skin.geometry.primitives[0].vertex
    bound = (np.c_[rest, np.ones(len(rest))] @ np.asarray(skin.bind_shape_matrix, dtype=np.float64).T)[:, :3] * scale
    pairs = np.asarray(skin.vertex_weight_index).reshape(-1, skin.nindices)
    influence_bone = pairs[:, skin.offsets[0]]
    influence_weight = skin.weights.data[pairs[:, skin.offsets[1]], 0]
    counts = np.asarray(skin.vcounts)
    starts = np.cumsum(counts) - counts

    materials = {material.id: material for material in dae.materials}
    parts = 0
    for primitive in skin.geometry.primitives:
        triangles = primitive if isinstance(primitive, collada.triangleset.TriangleSet) else primitive.triangleset()
        material = materials.get(primitive.material)
        rgba, image = (None, None) if material is None else _diffuse(material, actor.skin.parent)
        if material is not None and rgba is None:
            continue
        textured = image is not None and len(triangles.texcoord_indexset) > 0
        uv_index = triangles.texcoord_indexset[0] if textured else np.zeros_like(triangles.vertex_index)
        corners = np.stack([triangles.vertex_index.ravel(), uv_index.ravel()], axis=1)
        unique, inverse = np.unique(corners, axis=0, return_inverse=True)
        source = unique[:, 0]
        spans = counts[source]
        within = np.arange(spans.sum()) - np.repeat(np.cumsum(spans) - spans, spans)
        influences = np.repeat(starts[source], spans) + within
        prefix = f'part{parts}_'
        arrays[f'{prefix}vertices'] = bound[source]
        arrays[f'{prefix}faces'] = inverse.reshape(-1, 3)
        arrays[f'{prefix}rgba'] = np.array(rgba if rgba is not None else ())
        arrays[f'{prefix}influence_vertex'] = np.repeat(np.arange(len(source)), spans)
        arrays[f'{prefix}influence_bone'] = influence_bone[influences]
        arrays[f'{prefix}influence_weight'] = influence_weight[influences]
        arrays[f'{prefix}texture'] = np.array('')
        if textured:
            uv = triangles.texcoordset[0][unique[:, 1]]
            arrays[f'{prefix}uv'] = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)
            arrays[f'{prefix}texture'] = np.array(f'{stem}_{parts}.png')
            small = image.convert('RGB')
            small.thumbnail((_TEXTURE_PX, _TEXTURE_PX))
            small.save(out_dir / f'{stem}_{parts}.png')
        parts += 1
    arrays['parts'] = np.array(parts)
    np.savez(out_dir / _RIG_FILE, **arrays)


def _read_rig(out_dir: Path, z_offset: float) -> Rig:
    with np.load(out_dir / _RIG_FILE) as arrays:
        bones = tuple(str(name) for name in arrays['bones'])
        parts = []
        for index in range(int(arrays['parts'])):
            prefix = f'part{index}_'
            bone = arrays[f'{prefix}influence_bone']
            vertex = arrays[f'{prefix}influence_vertex']
            weight = arrays[f'{prefix}influence_weight']
            moved = [bone == number for number in range(len(bones))]
            rgba = arrays[f'{prefix}rgba']
            texture = str(arrays[f'{prefix}texture'])
            parts.append(
                SkinPart(
                    vertices=arrays[f'{prefix}vertices'],
                    faces=arrays[f'{prefix}faces'],
                    uv=arrays[f'{prefix}uv'] if texture else None,
                    rgba=tuple(rgba.tolist()) if len(rgba) else None,
                    texture=str(out_dir / texture) if texture else None,
                    bone_vertices=tuple(vertex[mask] for mask in moved),
                    bone_weights=tuple(weight[mask] for mask in moved),
                )
            )
        clips = {
            str(name): Clip(
                times=arrays[f'clip_{name}_times'],
                positions=arrays[f'clip_{name}_positions'],
                quaternions=arrays[f'clip_{name}_quaternions'],
                stride=float(arrays[f'clip_{name}_stride']),
            )
            for name in arrays['clips']
        }
        return Rig(
            bones=bones,
            parts=tuple(parts),
            clips=clips,
            z_offset=z_offset,
            node_parents=tuple(arrays['node_parents'].tolist()),
            neutral=arrays['neutral'],
            bone_nodes=tuple(arrays['bone_nodes'].tolist()),
            inverse_binds=arrays['inverse_binds'],
            bind_positions=arrays['bind_positions'],
            scale=float(arrays['scale']),
        )


_rigs: dict[str, Rig] = {}


def actor_rig(sdf_path: str) -> Rig:
    """Skin parts, bones and clips of an actor SDF, built once per content of its files into the mesh cache."""
    rig = _rigs.get(sdf_path)
    if rig is None:
        actor = read_actor(sdf_path)
        sources = [actor.skin, *(actor.clips[name] for name in sorted(actor.clips))]
        out_dir = cache_entry(sources, f'bones at joints:{_TEXTURE_PX}', _RIG_FILE, lambda stage, stem: _write_rig(stage, stem, actor))
        rig = _rigs[sdf_path] = _read_rig(out_dir, actor.z_offset)
    return rig


@functools.cache
def _turns(bones: tuple[str, ...]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Per bone rotation the wire drives: joint name, bone index, unit axis and gain, in BONE_MAP order."""
    rows = [(joint, bones.index(bone), axis, gain) for joint, targets in BONE_MAP.items() for bone, axis, gain in targets if bone in bones]
    axes = np.array([axis for _, _, axis, _ in rows], dtype=np.float64).reshape(-1, 3)
    return (
        tuple(joint for joint, _, _, _ in rows),
        np.array([bone for _, bone, _, _ in rows], dtype=int),
        axes / np.linalg.norm(axes, axis=1, keepdims=True),
        np.array([gain for _, _, _, gain in rows]),
    )


def _axis_rotations(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, k, 3, 3) for angles (n, k) about unit axes (k, 3)."""
    x, y, z = axes.T
    cross = np.zeros((len(axes), 3, 3))
    cross[:, 0, 1], cross[:, 0, 2], cross[:, 1, 0] = -z, y, z
    cross[:, 1, 2], cross[:, 2, 0], cross[:, 2, 1] = -x, -y, x
    sin = np.sin(angles)[..., None, None]
    cos = np.cos(angles)[..., None, None]
    return np.eye(3) + sin * cross + (1.0 - cos) * (cross @ cross)


def _quaternions(rotations: np.ndarray) -> np.ndarray:
    """wxyz quaternions of rotation matrices (..., 3, 3)."""
    m = rotations
    xx, yy, zz = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    wx, wy, wz = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    xy, xz, yz = m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1]
    squares = np.stack([1.0 + xx + yy + zz, 1.0 + xx - yy - zz, 1.0 - xx + yy - zz, 1.0 - xx - yy + zz], axis=-1)
    candidates = np.stack(
        [
            np.stack([squares[..., 0], wx, wy, wz], axis=-1),
            np.stack([wx, squares[..., 1], xy, xz], axis=-1),
            np.stack([wy, xy, squares[..., 2], yz], axis=-1),
            np.stack([wz, xz, yz, squares[..., 3]], axis=-1),
        ],
        axis=-2,
    )
    largest = np.argmax(squares, axis=-1)[..., None, None]
    quaternions = np.take_along_axis(candidates, largest, axis=-2)[..., 0, :]
    return quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)


def joint_poses(rigs: Sequence[Rig], roots: np.ndarray, angles: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """World bone positions (rigs, bones, 3) and wxyz quaternions for rigs of one skeleton layout standing at roots (rigs, 4, 4), each bone turned from its neutral stance by its wire joint angles."""
    joints, turned, axes, gains = _turns(rigs[0].bones)
    values = np.array([[pose.get(joint, 0.0) for joint in joints] for pose in angles]).reshape(len(rigs), len(joints)) * gains
    turns = _axis_rotations(axes, values)
    transforms = np.stack([rig.neutral for rig in rigs], axis=1)
    nodes = list(rigs[0].bone_nodes)
    for index, bone in enumerate(turned):
        transforms[nodes[bone], :, :3, :3] = transforms[nodes[bone], :, :3, :3] @ turns[:, index]
    worlds = _worlds(rigs[0].node_parents, transforms, roots)
    joints = np.moveaxis(worlds[nodes], 0, 1)
    skinned = joints @ np.stack([rig.inverse_binds for rig in rigs])
    scales = np.array([rig.scale for rig in rigs]).reshape(-1, 1, 1, 1)
    return joints[..., :3, 3], _quaternions(skinned[..., :3, :3] / scales)


def add_skin(spec: mujoco.MjSpec, cache: AssetCache, name: str, rig: Rig, part: SkinPart, bones: Sequence[str]) -> mujoco.MjsSkin:
    """Add a part of rig as a skin bound to the named bone bodies, each sitting at its joint with model axes at bind."""
    skin = spec.add_skin()
    skin.name = name
    skin.vert = part.vertices.ravel().tolist()
    skin.face = part.faces.ravel().tolist()
    if part.uv is not None:
        skin.texcoord = part.uv.ravel().tolist()
    used = [index for index in range(len(bones)) if len(part.bone_vertices[index])]
    skin.bodyname = [bones[index] for index in used]
    skin.bindpos = rig.bind_positions[used].ravel().tolist()
    skin.bindquat = [1.0, 0.0, 0.0, 0.0] * len(used)
    skin.vertid = [part.bone_vertices[index].tolist() for index in used]
    skin.vertweight = [part.bone_weights[index].tolist() for index in used]
    material = part_material(spec, cache, MeshPart(file='', rgba=part.rgba, texture=part.texture))
    if material is not None:
        skin.material = material
    return skin


__all__ = ['Actor', 'Clip', 'Rig', 'SkinPart', 'actor_rig', 'add_skin', 'clip_frames', 'joint_poses', 'read_actor']
