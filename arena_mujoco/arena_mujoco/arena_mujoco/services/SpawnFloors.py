"""SpawnFloors service handler."""

from __future__ import annotations

import mujoco
from arena_mujoco_msgs.msg import Floor
from arena_mujoco_msgs.srv import SpawnFloors

from arena_mujoco import context, hooks
from arena_mujoco.mesh_mat import add_box_body, ensure_material
from arena_mujoco.scene import RejectedBodies

from .utils import Service

_FLOOR_THICKNESS = 0.02


def _spawn_floor(env_id: int, floor: Floor) -> bool:
    store = context.get_scene_store()

    cx = float(floor.pos.x)
    cy = float(floor.pos.y)
    cz = float(floor.pos.z) + _FLOOR_THICKNESS / 2.0

    raw_name = floor.name
    cache = store.asset_cache(env_id)

    def builder(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        mat = ensure_material(spec, cache, floor.material)
        return add_box_body(
            spec,
            name=raw_name,
            pos=(cx, cy, cz),
            quat=(1.0, 0.0, 0.0, 0.0),
            size=(float(floor.x_length), float(floor.y_length), _FLOOR_THICKNESS),
            material_name=mat,
        )

    store.add_body(env_id, builder)
    return True


def _spawn_floors_cb(
    request: SpawnFloors.Request,
    response: SpawnFloors.Response,
) -> SpawnFloors.Response:
    store = context.get_scene_store()
    env_id = int(request.env_id)
    store.ensure_env(env_id)

    results: list[bool] = []
    for floor in request.floors:
        try:
            results.append(_spawn_floor(env_id, floor))
        except Exception:
            results.append(False)

    if request.floors:
        try:
            store.recompile(env_id)
        except RejectedBodies as exc:
            hooks.get_node().get_logger().error(f'floors not spawned: {exc}')
            results = [False] * len(results)

    response.ret = results
    return response


spawn_floors_service = Service(
    srv_type=SpawnFloors,
    srv_name='mujoco/SpawnFloors',
    callback=_spawn_floors_cb,
)

__all__ = ['spawn_floors_service']
