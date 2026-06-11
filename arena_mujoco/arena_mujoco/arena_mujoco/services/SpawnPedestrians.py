"""SpawnPedestrians service: kinematic pedestrians backed by the mocap pool."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from arena_people_msgs.srv import SpawnPedestrians
from geometry_msgs.msg import Pose

from arena_mujoco import context, hooks
from arena_mujoco.humans import Rig, actor_rig, add_skin, joint_poses
from arena_mujoco.scene import MOCAP_CAPSULE_SUFFIX, RejectedBodies

from .utils import Service

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arena_people_msgs.msg import Pedestrian

_POOL_SIZE = 64

_ENV_PREFIX_RE = re.compile(r'^env_(\d+)/')

_registry: dict[str, tuple[int, int]] = {}

_env_pool_allocated: set[int] = set()

_env_free_handles: dict[int, list[int]] = {}

_dressed: dict[tuple[int, int], str] = {}

_rigs: dict[tuple[int, int], Rig] = {}

_requested: dict[str, str] = {}

_PED_CENTER_Z = 0.85

_LIDAR_ONLY_GROUP = 4

_IDLE_SPEED_MPS = 0.05
_TELEPORT_SPEED_MPS = 10.0
_WALK_CLIP = 'walk'
_IDLE_CLIP = 'idle'


@dataclass
class _Gait:
    """Where one pedestrian is in which clip, and the pose and sim time that was measured at."""

    clip: str = _IDLE_CLIP
    cursor: float = 0.0
    xy: tuple[float, float] | None = None
    time: float = 0.0


_gaits: dict[str, _Gait] = {}


def _advance(gait: _Gait, rig: Rig, xy: tuple[float, float], now: float) -> None:
    """Advance the clip cursor by the distance covered while walking, by sim time while standing."""
    if gait.xy is None:
        gait.xy, gait.time = xy, now
        return
    elapsed = now - gait.time
    if elapsed <= 0.0:
        return
    moved = math.dist(xy, gait.xy)
    walk = rig.clips.get(_WALK_CLIP)
    speed = moved / elapsed
    if speed > _TELEPORT_SPEED_MPS:
        clip, step = gait.clip, 0.0
    elif walk is not None and walk.stride > 0.0 and speed >= _IDLE_SPEED_MPS:
        clip, step = _WALK_CLIP, moved * walk.duration / walk.stride
    else:
        clip, step = _IDLE_CLIP, elapsed
    gait.cursor = gait.cursor + step if clip == gait.clip else 0.0
    gait.clip, gait.xy, gait.time = clip, xy, now


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of one wxyz quaternion with a stack (n, 4)."""
    w, x, y, z = left
    rw, rx, ry, rz = right.T
    return np.stack(
        [
            w * rw - x * rx - y * ry - z * rz,
            w * rx + x * rw + y * rz - z * ry,
            w * ry - x * rz + y * rw + z * rx,
            w * rz + x * ry - y * rx + z * rw,
        ],
        axis=1,
    )


def _root(pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Position and wxyz quaternion of a wire pose, identity for an unset orientation."""
    p = pose.position
    o = pose.orientation
    quat = np.array((float(o.w), float(o.x), float(o.y), float(o.z)))
    if not quat.any():
        quat = np.array((1.0, 0.0, 0.0, 0.0))
    return np.array((float(p.x), float(p.y), float(p.z))), quat


def _place(env_id: int, handle: int, pose: Pose) -> None:
    position, quat = _root(pose)
    context.get_scene_store().set_mocap_pose(env_id, handle, (position[0], position[1], position[2] + _PED_CENTER_Z), tuple(quat))


def _play_clip(env_id: int, handle: int, pose: Pose, name: str) -> None:
    """Pose a dressed pedestrian's bones from its walk or idle clip."""
    rig = _rigs.get((env_id, handle))
    if rig is None:
        return
    store = context.get_scene_store()
    position, quat = _root(pose)
    gait = _gaits.setdefault(name, _Gait())
    _advance(gait, rig, (position[0], position[1]), store.time)
    clip = rig.clips.get(gait.clip) or next(iter(rig.clips.values()))
    positions, quaternions = clip.pose(gait.cursor)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    feet = position + (0.0, 0.0, rig.z_offset)
    store.set_bone_poses(env_id, handle, feet + positions @ rotation.reshape(3, 3).T, _quat_mul(quat, quaternions))


def _drive_joints(pedestrians: Sequence[Pedestrian]) -> None:
    """Pose dressed pedestrians' bones from the joint angles on the wire, all rigs of one skeleton layout in one pass."""
    store = context.get_scene_store()
    batches: dict[tuple, list[tuple[tuple[int, int], Rig, np.ndarray, dict[str, float]]]] = {}
    for pedestrian in pedestrians:
        entry = _registry[pedestrian.name]
        rig = _rigs[entry]
        position, quat = _root(pedestrian.pose)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        root = np.eye(4)
        root[:3, :3] = rotation.reshape(3, 3)
        root[:3, 3] = position + (0.0, 0.0, rig.z_offset)
        angles = dict(zip(pedestrian.joint_state.name, pedestrian.joint_state.position, strict=False))
        batches.setdefault(rig.layout, []).append((entry, rig, root, angles))
    for batch in batches.values():
        positions, quaternions = joint_poses([rig for _, rig, _, _ in batch], np.stack([root for _, _, root, _ in batch]), [angles for _, _, _, angles in batch])
        for (entry, _, _, _), bone_positions, bone_quaternions in zip(batch, positions, quaternions, strict=True):
            store.set_bone_poses(entry[0], entry[1], bone_positions, bone_quaternions)


def place_pedestrian(name: str, pose: Pose, teleport: bool = False) -> bool:
    """Move a spawned pedestrian to pose, heading included, a teleport restarting its clip, False when name is unknown."""
    entry = _registry.get(name)
    if entry is None:
        return False
    if teleport:
        _gaits.pop(name, None)
    _place(entry[0], entry[1], pose)
    _play_clip(entry[0], entry[1], pose, name)
    return True


def update_pedestrians(pedestrians: Sequence[Pedestrian]) -> None:
    """Move spawned pedestrians to their poses and pose their bones from the wire's joint angles (from a clip without them), first dressing each in the actor its model_uri names (read once per name)."""
    store = context.get_scene_store()
    changed: dict[int, list[int]] = {}
    for pedestrian in pedestrians:
        entry = _registry.get(pedestrian.name)
        uri = pedestrian.model_uri
        if entry is None or not uri or _requested.get(pedestrian.name) == uri:
            continue
        _requested[pedestrian.name] = uri
        rig = _actor(uri)
        worn = uri if rig is not None else ''
        if _dressed.get(entry, '') != worn:
            _dress(entry[0], entry[1], rig)
            _dressed[entry] = worn
            changed.setdefault(entry[0], []).append(entry[1])
    for env_id, handles in changed.items():
        try:
            store.recompile(env_id)
        except RejectedBodies as exc:
            hooks.get_node().get_logger().warning(f'pedestrian skins rejected, env {env_id} keeps capsules: {exc}')
            for handle in handles:
                _dress(env_id, handle, None)
                _dressed[(env_id, handle)] = ''
            store.recompile(env_id)
    jointed = []
    for pedestrian in pedestrians:
        entry = _registry.get(pedestrian.name)
        if entry is None:
            continue
        _place(entry[0], entry[1], pedestrian.pose)
        if entry in _rigs and pedestrian.joint_state.name:
            jointed.append(pedestrian)
        else:
            _play_clip(entry[0], entry[1], pedestrian.pose, pedestrian.name)
    if jointed:
        _drive_joints(jointed)


def release_pedestrians(prefix: str) -> None:
    """Park and recycle every pedestrian whose name starts with prefix, skins stay on for the next wearer."""
    store = context.get_scene_store()
    for name in [n for n in _registry if n.startswith(prefix)]:
        env_id, handle = _registry.pop(name)
        _requested.pop(name, None)
        _gaits.pop(name, None)
        store.free_mocap(env_id, handle)
        _env_free_handles.setdefault(env_id, []).append(handle)


def reset_env(env_id: int) -> None:
    """Forget every pedestrian and pool handle of env_id after its bodies were deleted."""
    for name in [n for n, (e, _h) in _registry.items() if e == env_id]:
        del _registry[name]
        _requested.pop(name, None)
        _gaits.pop(name, None)
    for key in [key for key in _dressed if key[0] == env_id]:
        del _dressed[key]
        _rigs.pop(key, None)
    _env_pool_allocated.discard(env_id)
    _env_free_handles.pop(env_id, None)


def _parse_env_id(name: str) -> int | None:
    m = _ENV_PREFIX_RE.match(name)
    if m is None:
        return None
    return int(m.group(1))


def _take_handle(env_id: int) -> int | None:
    """Return a free mocap handle for env_id, allocating the pool lazily."""
    store = context.get_scene_store()
    free = _env_free_handles.setdefault(env_id, [])
    if not free:
        if env_id in _env_pool_allocated:
            return None
        _env_pool_allocated.add(env_id)
        handles = store.alloc_mocap(env_id, _POOL_SIZE)
        free.extend(handles)
    return free.pop()


def _dress(env_id: int, handle: int, rig: Rig | None) -> None:
    """Swap the skins on a pooled body: the capsule stays for rays and contacts, cameras see it only without skins."""
    store = context.get_scene_store()
    cache = store.asset_cache(env_id)
    bones = store.set_bones(env_id, handle, len(rig.bones) if rig is not None else 0)

    def edit(spec: mujoco.MjSpec, body: mujoco.MjsBody) -> None:
        for geom in body.geoms:
            if geom.name.endswith(MOCAP_CAPSULE_SUFFIX):
                geom.group = _LIDAR_ONLY_GROUP if rig is not None else 0
        if rig is not None:
            for index, part in enumerate(rig.parts):
                cache.add(None, add_skin(spec, cache, f'{body.name}_skin{index}', rig, part, bones))

    store.edit_mocap(env_id, handle, edit)
    if rig is None:
        _rigs.pop((env_id, handle), None)
    else:
        _rigs[(env_id, handle)] = rig


def _actor(model_uri: str) -> Rig | None:
    """Rig of an actor SDF, None when it cannot be read."""
    try:
        return actor_rig(model_uri)
    except Exception as exc:  # noqa: BLE001 - an unreadable actor still gets its capsule
        hooks.get_node().get_logger().warning(f'pedestrian skin {model_uri!r} not loaded: {exc!r}')
        return None


def _spawn_pedestrians_cb(
    request: SpawnPedestrians.Request,
    response: SpawnPedestrians.Response,
) -> SpawnPedestrians.Response:
    store = context.get_scene_store()
    results: list[int] = []
    for sp in request.pedestrians:
        name: str = sp.pedestrian.name
        env_id = _parse_env_id(name)
        if env_id is None:
            results.append(SpawnPedestrians.Response.FAILED_PARSE)
            continue
        store.ensure_env(env_id)
        handle = _take_handle(env_id)
        if handle is None:
            results.append(SpawnPedestrians.Response.FAILED_CREATE)
            continue
        _registry[name] = (env_id, handle)
        results.append(SpawnPedestrians.Response.SUCCESS)
    update_pedestrians([sp.pedestrian for sp in request.pedestrians])
    response.results = results
    return response


spawn_pedestrians_service = Service(
    srv_type=SpawnPedestrians,
    srv_name='mujoco/SpawnPedestrians',
    callback=_spawn_pedestrians_cb,
)

__all__ = ['place_pedestrian', 'release_pedestrians', 'reset_env', 'spawn_pedestrians_service', 'update_pedestrians']
