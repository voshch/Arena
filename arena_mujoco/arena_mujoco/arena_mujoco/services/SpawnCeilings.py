"""SpawnCeilings service handler."""

from __future__ import annotations

import mujoco
from arena_mujoco_msgs.msg import Ceiling
from arena_mujoco_msgs.srv import SpawnCeilings

from arena_mujoco import context, hooks
from arena_mujoco.mesh_mat import add_plane_body, ensure_material
from arena_mujoco.scene import RejectedBodies

from .utils import Service

CEILING_GROUP = 2


def _spawn_ceiling(env_id: int, ceiling: Ceiling) -> bool:
    store = context.get_scene_store()

    raw_name = ceiling.name
    cache = store.asset_cache(env_id)

    pos = (float(ceiling.pos.x), float(ceiling.pos.y), float(ceiling.pos.z))
    half_size = (float(ceiling.x_length) / 2.0, float(ceiling.y_length) / 2.0, 1.0)

    def builder(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        mat = ensure_material(spec, cache, ceiling.material)
        return add_plane_body(
            spec,
            name=raw_name,
            pos=pos,
            quat=(0.0, 1.0, 0.0, 0.0),
            half_size=half_size,
            material_name=mat,
            group=CEILING_GROUP,
        )

    store.add_body(env_id, builder)
    return True


def _spawn_ceilings_cb(
    request: SpawnCeilings.Request,
    response: SpawnCeilings.Response,
) -> SpawnCeilings.Response:
    store = context.get_scene_store()
    env_id = int(request.env_id)
    store.ensure_env(env_id)

    results: list[bool] = []
    for ceiling in request.ceilings:
        try:
            results.append(_spawn_ceiling(env_id, ceiling))
        except Exception:
            results.append(False)

    if request.ceilings:
        try:
            store.recompile(env_id)
        except RejectedBodies as exc:
            hooks.get_node().get_logger().error(f'ceilings not spawned: {exc}')
            results = [False] * len(results)

    response.ret = results
    return response


spawn_ceilings_service = Service(
    srv_type=SpawnCeilings,
    srv_name='mujoco/SpawnCeilings',
    callback=_spawn_ceilings_cb,
)

__all__ = ['spawn_ceilings_service', 'CEILING_GROUP']
