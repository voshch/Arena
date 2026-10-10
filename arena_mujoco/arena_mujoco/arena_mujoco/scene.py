"""Per-env MuJoCo scene engine."""

from __future__ import annotations

import itertools
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from arena_mujoco.viewport.camera import CAPTURE_HEIGHT, CAPTURE_WIDTH

_ID_BAD = re.compile(r'[^0-9A-Za-z_]+')

_MOCAP_PARK = (0.0, 0.0, -1000.0)
_MOCAP_CAPSULE_SIZE = (0.25, 0.6)
MOCAP_CAPSULE_SUFFIX = '_capsule'

PHYSICS_DT = 0.002

_logger = logging.getLogger(__name__)

_RENDER_EXTENT_M = 20.0
_RENDER_ZNEAR_M = 0.01

_STEP_WORKERS = min(16, os.cpu_count() or 1)


def sanitize_id(s: str) -> str:
    """Map a raw frame id to a MuJoCo-safe element id (slashes and dashes folded to underscores)."""
    cleaned = _ID_BAD.sub('_', s).strip('_')
    return cleaned or 'x'


def env_prefix(env_id: int) -> str:
    """Return the MJCF id prefix that scopes a body to env_id."""
    return f'env{env_id}_'


class RejectedBodies(ValueError):
    """The bodies and assets added since the last compile made the spec uncompilable and were dropped."""


class AssetCache:
    """Names of the reusable assets in one env spec, plus the elements added since the last compile."""

    def __init__(self) -> None:
        self._names: dict[tuple, str] = {}
        self._pending: list[tuple[tuple | None, mujoco.MjsElement]] = []

    def get(self, key: tuple) -> str | None:
        return self._names.get(key)

    def add(self, key: tuple | None, element: mujoco.MjsElement) -> str:
        """Record a freshly added spec element, under key when later callers may reuse it."""
        if key is not None:
            self._names[key] = element.name
        self._pending.append((key, element))
        return element.name

    @property
    def uncommitted(self) -> bool:
        return bool(self._pending)

    def commit(self) -> None:
        self._pending.clear()

    def forget(self, names: set[str]) -> None:
        """Drop the keys of assets that left the spec."""
        self._names = {key: name for key, name in self._names.items() if name not in names}

    def rollback(self, spec: mujoco.MjSpec) -> None:
        """Delete every element added since the last commit."""
        for key, element in reversed(self._pending):
            spec.delete(element)
            if key is not None:
                del self._names[key]
        self._pending.clear()


class _EnvScene:
    """Self-contained MuJoCo state for one env, isolated so a recompile never touches a sibling."""

    def __init__(self, env_id: int) -> None:
        self.env_id = env_id
        self.spec: mujoco.MjSpec = _new_world_spec()
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.names: set[str] = set()
        self.dirty: bool = True
        self.orphans: bool = False
        self.asset_cache = AssetCache()
        self.pending: list[tuple[str, mujoco.MjsBody]] = []
        self.mocap_names: dict[int, str] = {}
        self.bones: dict[int, list[str]] = {}
        self.bone_ids: dict[int, np.ndarray] = {}
        self.mocap_free: list[int] = []
        self._mocap_next: int = 0
        self.model_edits: int = 0


def _new_world_spec() -> mujoco.MjSpec:
    """Build a fresh spec with a ground plane and default lighting."""
    spec = mujoco.MjSpec()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = PHYSICS_DT
    spec.stat.extent = _RENDER_EXTENT_M
    spec.visual.map.znear = _RENDER_ZNEAR_M / _RENDER_EXTENT_M
    spec.visual.global_.offwidth = CAPTURE_WIDTH
    spec.visual.global_.offheight = CAPTURE_HEIGHT
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.worldbody.add_geom(
        name='ground',
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, 0.0],
    )
    spec.worldbody.add_light(
        name='sun',
        pos=[0.0, 0.0, 10.0],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=False,
    )
    return spec


def _drop_unused_assets(scene: _EnvScene) -> None:
    """Delete meshes and materials no geom uses any more, then the textures no material uses."""
    spec = scene.spec
    used_meshes = {geom.meshname for geom in spec.geoms}
    used_materials = {element.material for element in (*spec.geoms, *spec.sites, *spec.skins)}
    gone: set[str] = set()
    for element in [*spec.meshes, *spec.materials]:
        used = used_meshes if isinstance(element, mujoco.MjsMesh) else used_materials
        if element.name not in used:
            gone.add(element.name)
            spec.delete(element)
    used_textures = {name for material in spec.materials for name in material.textures if name}
    for texture in list(spec.textures):
        if texture.name not in used_textures:
            gone.add(texture.name)
            spec.delete(texture)
    scene.asset_cache.forget(gone)


def _step_scene(scene: _EnvScene, n: int) -> None:
    for _ in range(n):
        mujoco.mj_step(scene.model, scene.data)


class SceneStore:
    """Per-env MuJoCo scene registry owning one MjModel/MjData per env."""

    def __init__(self) -> None:
        self._envs: dict[int, _EnvScene] = {}
        self.time = 0.0
        self._pool = ThreadPoolExecutor(max_workers=_STEP_WORKERS, thread_name_prefix='mj_step')

    def ensure_env(self, env_id: int) -> None:
        """Create and compile an empty scene for env_id if it does not exist yet."""
        if env_id in self._envs:
            return
        scene = _EnvScene(env_id)
        self._envs[env_id] = scene
        self.recompile(env_id)
        scene.data.time = self.time

    def add_body(self, env_id: int, body_builder: Callable[[mujoco.MjSpec], mujoco.MjsBody]) -> str:
        """Append a body via body_builder(spec)->body, register its name, mark dirty."""
        scene = self._require(env_id)
        body = body_builder(scene.spec)
        name = self._assign_name(scene, body)
        scene.names.add(name)
        scene.pending.append((name, body))
        scene.dirty = True
        return name

    def remove(self, env_id: int, name: str) -> bool:
        """Delete the named body from env_id's spec and mark dirty. Returns True on success."""
        scene = self._require(env_id)
        if scene.pending:
            self.recompile(env_id)
        key = sanitize_id(name)
        if key not in scene.names:
            return False
        if not self._detach_body(scene, key):
            return False
        scene.names.discard(key)
        scene.dirty = True
        scene.orphans = True
        return True

    def remove_by_prefix(self, env_id: int, prefix: str) -> int:
        """Delete every registered body whose stored id starts with sanitize_id(prefix)."""
        scene = self._require(env_id)
        if scene.pending:
            self.recompile(env_id)
        key_prefix = sanitize_id(prefix) if prefix else ''
        targets = [name for name in scene.names if name.startswith(key_prefix)]
        removed = 0
        for name in targets:
            if self._detach_body(scene, name):
                scene.names.discard(name)
                removed += 1
        if removed:
            scene.dirty = True
            scene.orphans = True
        return removed

    def compile_dirty(self) -> None:
        """Recompile every env whose spec changed since its last compile."""
        for env_id in self._envs:
            self._compile_if_dirty(env_id)

    def _compile_if_dirty(self, env_id: int) -> None:
        if not self._envs[env_id].dirty:
            return
        try:
            self.recompile(env_id)
        except RejectedBodies as exc:
            _logger.warning(f'env {env_id}: additions dropped, they did not compile: {exc}')

    def recompile(self, env_id: int) -> None:
        """Rebuild model and data for env_id, carrying joint state by name. Raises RejectedBodies after dropping additions that do not compile."""
        scene = self._require(env_id)
        old_model = scene.model
        old_data = scene.data

        if scene.orphans:
            _drop_unused_assets(scene)
            scene.orphans = False
        rejected: ValueError | None = None
        try:
            new_model = scene.spec.compile()
        except ValueError as exc:
            if not scene.pending and not scene.asset_cache.uncommitted:
                raise
            rejected = exc
            self._drop_pending(scene)
            new_model = scene.spec.compile()
        scene.pending.clear()
        scene.asset_cache.commit()
        new_data = mujoco.MjData(new_model)

        if old_model is not None and old_data is not None:
            _carry_state(old_model, old_data, new_model, new_data)
            mujoco.mj_forward(new_model, new_data)

        scene.model = new_model
        scene.data = new_data
        scene.dirty = False
        scene.bone_ids.clear()
        if rejected is not None:
            raise RejectedBodies(str(rejected)) from rejected

    def discard_pending(self, env_id: int) -> None:
        """Drop the bodies and assets added since the last compile, then recompile."""
        scene = self._require(env_id)
        self._drop_pending(scene)
        self.recompile(env_id)

    @staticmethod
    def _drop_pending(scene: _EnvScene) -> None:
        for name, body in scene.pending:
            scene.spec.delete(body)
            scene.names.discard(name)
        scene.pending.clear()
        scene.asset_cache.rollback(scene.spec)

    def find(self, env_id: int, name: str) -> bool:
        """Return True if a body named name is registered in env_id."""
        scene = self._envs.get(env_id)
        if scene is None:
            return False
        return sanitize_id(name) in scene.names

    def set_body_pose(
        self,
        env_id: int,
        name: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ) -> bool:
        """Teleport a body to pos + quat(wxyz), returns False if unknown."""
        scene = self._envs.get(env_id)
        if scene is None:
            return False
        self._compile_if_dirty(env_id)
        model = scene.model
        data = scene.data
        if model is None or data is None:
            return False
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sanitize_id(name))
        if body_id < 0:
            return False
        jnt_adr = int(model.body_jntadr[body_id])
        jnt_num = int(model.body_jntnum[body_id])
        if jnt_num == 1 and model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = int(model.jnt_qposadr[jnt_adr])
            data.qpos[qadr : qadr + 3] = pos
            data.qpos[qadr + 3 : qadr + 7] = quat
        else:
            model.body_pos[body_id] = pos
            model.body_quat[body_id] = quat
            scene.model_edits += 1
            body = scene.spec.body(sanitize_id(name))
            body.pos = list(pos)
            body.quat = list(quat)
        mujoco.mj_forward(model, data)
        return True

    def get_body_pose(
        self,
        env_id: int,
        name: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        """Return the body's current world (pos, quat wxyz) from data.xpos/xquat, None if unknown."""
        scene = self._envs.get(env_id)
        if scene is None:
            return None
        self._compile_if_dirty(env_id)
        model = scene.model
        data = scene.data
        if model is None or data is None:
            return None
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sanitize_id(name))
        if body_id < 0:
            return None
        pos = tuple(float(v) for v in data.xpos[body_id])
        quat = tuple(float(v) for v in data.xquat[body_id])
        return pos, quat

    def get_clock_time(self, env_id: int) -> float:
        """Store clock, shared by every env, stamps /clock and sensors."""
        del env_id
        return self.time

    def get_model(self, env_id: int) -> mujoco.MjModel | None:
        """Return the live MjModel for env_id, None if not compiled yet."""
        scene = self._envs.get(env_id)
        if scene is None:
            return None
        return scene.model

    def get_data(self, env_id: int) -> mujoco.MjData | None:
        """Return the live MjData for env_id, None if not compiled yet."""
        scene = self._envs.get(env_id)
        if scene is None:
            return None
        return scene.data

    def model_edits(self, env_id: int) -> int:
        """Count of in-place edits to env_id's compiled model, 0 if absent."""
        scene = self._envs.get(env_id)
        return 0 if scene is None else scene.model_edits

    def asset_cache(self, env_id: int) -> AssetCache:
        """Return the per-env asset dedup cache for env_id, raises KeyError if absent."""
        return self._envs[env_id].asset_cache

    def step(self, n: int = 1) -> None:
        """Advance every env n physics steps, envs in parallel (mj_step releases the GIL)."""
        self.time += n * PHYSICS_DT
        scenes = [scene for scene in self._envs.values() if scene.model is not None and scene.data is not None]
        if len(scenes) == 1:
            _step_scene(scenes[0], n)
        elif scenes:
            list(self._pool.map(_step_scene, scenes, itertools.repeat(n)))

    def alloc_mocap(self, env_id: int, count: int) -> list[int]:
        """Pre-create count mocap bodies (capsule geom) parked offscreen, returns their handles."""
        scene = self._require(env_id)
        handles: list[int] = []
        for _ in range(count):
            handle = scene._mocap_next
            scene._mocap_next += 1
            name = f'{env_prefix(env_id)}mocap_{handle}'
            mocap = scene.spec.worldbody.add_body(name=name, mocap=True, pos=list(_MOCAP_PARK))
            mocap.add_geom(
                name=f'{name}{MOCAP_CAPSULE_SUFFIX}',
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[_MOCAP_CAPSULE_SIZE[0], _MOCAP_CAPSULE_SIZE[1], 0.0],
            )
            scene.mocap_names[handle] = name
            scene.names.add(name)
            handles.append(handle)
        scene.dirty = True
        self.recompile(env_id)
        return handles

    def edit_mocap(self, env_id: int, handle: int, edit: Callable[[mujoco.MjSpec, mujoco.MjsBody], None]) -> None:
        """Run edit(spec, body) on a compiled mocap body and mark the env for recompile."""
        scene = self._require(env_id)
        edit(scene.spec, scene.spec.body(scene.mocap_names[handle]))
        scene.dirty = True

    def edit_body(self, env_id: int, name: str, edit: Callable[[mujoco.MjSpec, mujoco.MjsBody], None]) -> bool:
        """Run edit(spec, body) on a registered body and mark the env for recompile, False if unknown."""
        scene = self._envs.get(env_id)
        key = sanitize_id(name)
        if scene is None or key not in scene.names:
            return False
        self._compile_if_dirty(env_id)
        body = scene.spec.body(key)
        if body is None:
            return False
        edit(scene.spec, body)
        scene.dirty = True
        scene.orphans = True
        return True

    def set_bones(self, env_id: int, handle: int, count: int) -> list[str]:
        """Give a pooled body count geomless mocap bodies of its own, in place of earlier ones and the skins bound to them, returns their names."""
        scene = self._require(env_id)
        self._drop_bones(scene, handle)
        root = scene.mocap_names[handle]
        names = [f'{root}_bone{index}' for index in range(count)]
        for name in names:
            scene.asset_cache.add(None, scene.spec.worldbody.add_body(name=name, mocap=True, pos=list(_MOCAP_PARK)))
        if names:
            scene.bones[handle] = names
        scene.dirty = True
        return names

    def set_bone_poses(self, env_id: int, handle: int, pos: np.ndarray, quat: np.ndarray) -> None:
        """Set the runtime poses (bones, 3) and (bones, 4) of a pooled body's bones, nothing before they are compiled."""
        scene = self._require(env_id)
        model = scene.model
        data = scene.data
        ids = scene.bone_ids.get(handle)
        if ids is None:
            names = scene.bones.get(handle)
            if not names or model is None:
                return
            bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
            if min(bodies) < 0:
                return
            ids = scene.bone_ids[handle] = model.body_mocapid[bodies]
        data.mocap_pos[ids] = pos
        data.mocap_quat[ids] = quat

    def set_mocap_pose(
        self,
        env_id: int,
        handle: int,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ) -> None:
        """Set the runtime pose of a mocap body (no physics integration, no recompile)."""
        scene = self._require(env_id)
        model = scene.model
        data = scene.data
        if model is None or data is None:
            raise RuntimeError(f'env {env_id} not compiled')
        index = self._mocap_index(scene, model, handle)
        data.mocap_pos[index] = pos
        data.mocap_quat[index] = quat

    def free_mocap(self, env_id: int, handle: int) -> None:
        """Park a mocap body offscreen and recycle its handle (stays allocated, no recompile)."""
        scene = self._require(env_id)
        if handle not in scene.mocap_names:
            return
        model = scene.model
        data = scene.data
        if model is not None and data is not None:
            index = self._mocap_index(scene, model, handle)
            data.mocap_pos[index] = _MOCAP_PARK
            data.mocap_quat[index] = (1.0, 0.0, 0.0, 0.0)
            self.set_bone_poses(env_id, handle, np.array(_MOCAP_PARK), np.array((1.0, 0.0, 0.0, 0.0)))
        scene.mocap_free.append(handle)

    def _require(self, env_id: int) -> _EnvScene:
        scene = self._envs.get(env_id)
        if scene is None:
            raise KeyError(f'env {env_id} not initialized, call ensure_env first')
        return scene

    def _assign_name(self, scene: _EnvScene, body: mujoco.MjsBody) -> str:
        candidate = sanitize_id(body.name) if body.name else 'body'
        if candidate not in scene.names:
            body.name = candidate
            return candidate
        suffix = 0
        while f'{candidate}_{suffix}' in scene.names:
            suffix += 1
        unique = f'{candidate}_{suffix}'
        body.name = unique
        return unique

    def _detach_body(self, scene: _EnvScene, name: str) -> bool:
        body = scene.spec.body(name)
        if body is None:
            return False
        for handle in [h for h, n in scene.mocap_names.items() if n == name]:
            self._drop_bones(scene, handle)
            del scene.mocap_names[handle]
        scene.spec.delete(body)
        return True

    @staticmethod
    def _drop_bones(scene: _EnvScene, handle: int) -> None:
        names = scene.bones.pop(handle, [])
        scene.bone_ids.pop(handle, None)
        scene.orphans = scene.orphans or bool(names)
        gone = set(names)
        for skin in list(scene.spec.skins):
            if gone.intersection(skin.bodyname):
                scene.spec.delete(skin)
        for name in names:
            body = scene.spec.body(name)
            if body is not None:
                scene.spec.delete(body)

    def _mocap_index(self, scene: _EnvScene, model: mujoco.MjModel, handle: int) -> int:
        name = scene.mocap_names[handle]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sanitize_id(name))
        return int(model.body_mocapid[body_id])


def _carry_state(
    old_model: mujoco.MjModel,
    old_data: mujoco.MjData,
    new_model: mujoco.MjModel,
    new_data: mujoco.MjData,
) -> None:
    """Copy qpos/qvel by joint name for joints that survive the recompile."""
    for joint_id in range(new_model.njnt):
        name = mujoco.mj_id2name(new_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        old_id = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if old_id < 0:
            continue
        _copy_joint(old_model, old_data, old_id, new_model, new_data, joint_id)
    new_data.time = old_data.time


def _copy_joint(
    old_model: mujoco.MjModel,
    old_data: mujoco.MjData,
    old_id: int,
    new_model: mujoco.MjModel,
    new_data: mujoco.MjData,
    new_id: int,
) -> None:
    if old_model.jnt_type[old_id] != new_model.jnt_type[new_id]:
        return
    old_q = int(old_model.jnt_qposadr[old_id])
    new_q = int(new_model.jnt_qposadr[new_id])
    old_v = int(old_model.jnt_dofadr[old_id])
    new_v = int(new_model.jnt_dofadr[new_id])
    nq = _joint_nq(int(new_model.jnt_type[new_id]))
    nv = _joint_nv(int(new_model.jnt_type[new_id]))
    new_data.qpos[new_q : new_q + nq] = old_data.qpos[old_q : old_q + nq]
    new_data.qvel[new_v : new_v + nv] = old_data.qvel[old_v : old_v + nv]


def _joint_nq(jnt_type: int) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _joint_nv(jnt_type: int) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


__all__ = [
    'AssetCache',
    'RejectedBodies',
    'SceneStore',
    'env_prefix',
    'sanitize_id',
]
