"""SpawnPrims service handler: spawn box or mesh obstacle bodies into a MuJoCo env."""

from __future__ import annotations

from arena_mujoco_msgs.msg import Prim
from arena_mujoco_msgs.srv import SpawnPrims

from arena_mujoco import hooks
from arena_mujoco.context import get_scene_store
from arena_mujoco.mesh_mat import add_box_body, add_mesh_body, prepare_hulls
from arena_mujoco.scene import RejectedBodies

from .utils import Service


def _add_prim(env_id: int, prim: Prim) -> bool:
    """Add one prim to the env spec, False when its mesh cannot be read."""
    store = get_scene_store()
    cache = store.asset_cache(env_id)
    pose = prim.pose
    pos = (pose.position.x, pose.position.y, pose.position.z)
    quat = (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
    scale = (prim.scale.x, prim.scale.y, prim.scale.z)
    mesh_path = prim.mesh_path.strip()
    try:
        if mesh_path:
            store.add_body(env_id, lambda spec: add_mesh_body(spec, cache, prim.name, pos, quat, mesh_path, scale))
        else:
            store.add_body(env_id, lambda spec: add_box_body(spec, prim.name, pos, quat, scale))
    except Exception as exc:  # noqa: BLE001 - one unreadable asset must not fail the batch
        hooks.get_node().get_logger().warning(f'prim {prim.name!r} not spawned: {exc!r}')
        return False
    return True


def _compiles(env_id: int, prim: Prim) -> bool:
    try:
        get_scene_store().recompile(env_id)
    except RejectedBodies as exc:
        hooks.get_node().get_logger().warning(f'prim {prim.name!r} not spawned: {exc}')
        return False
    return True


def _spawn_prims_cb(
    request: SpawnPrims.Request,
    response: SpawnPrims.Response,
) -> SpawnPrims.Response:
    store = get_scene_store()
    env_id = int(request.env_id)
    store.ensure_env(env_id)

    prepare_hulls([path for prim in request.prims if (path := prim.mesh_path.strip())])
    results = [_add_prim(env_id, prim) for prim in request.prims]
    if any(prim.mesh_path.strip() for prim in request.prims):
        try:
            store.recompile(env_id)
        except RejectedBodies:
            results = [_add_prim(env_id, prim) and _compiles(env_id, prim) for prim in request.prims]
    response.ret = results
    return response


spawn_prims_service = Service(
    srv_type=SpawnPrims,
    srv_name='mujoco/SpawnPrims',
    callback=_spawn_prims_cb,
)

__all__ = ['spawn_prims_service']
