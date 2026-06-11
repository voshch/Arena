"""SpawnWalls service handler."""

from __future__ import annotations

import math

import mujoco
from arena_mujoco_msgs.msg import Wall
from arena_mujoco_msgs.srv import SpawnWalls

from arena_mujoco import context, hooks
from arena_mujoco.mesh_mat import add_box_body, ensure_material
from arena_mujoco.scene import RejectedBodies

from .utils import Service


def _spawn_wall(env_id: int, wall: Wall, cache: dict) -> bool:
    store = context.get_scene_store()

    sx, sy, sz = wall.start.x, wall.start.y, wall.start.z
    ex, ey, ez = wall.end.x, wall.end.y, wall.end.z

    length = math.hypot(ex - sx, ey - sy)
    height = abs(ez - sz)
    thickness = float(wall.thickness)

    cx = (sx + ex) / 2.0
    cy = (sy + ey) / 2.0
    cz = (sz + ez) / 2.0

    angle = math.atan2(ey - sy, ex - sx)
    quat = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))

    raw_name = wall.name

    def builder(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        mat = ensure_material(spec, cache, wall.material)
        return add_box_body(
            spec,
            name=raw_name,
            pos=(cx, cy, cz),
            quat=quat,
            size=(length, height, thickness),
            material_name=mat,
            geom_quat=(0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
        )

    store.add_body(env_id, builder)
    return True


def _spawn_walls_cb(
    request: SpawnWalls.Request,
    response: SpawnWalls.Response,
) -> SpawnWalls.Response:
    store = context.get_scene_store()
    env_id = int(request.env_id)
    store.ensure_env(env_id)
    cache = store.asset_cache(env_id)

    results: list[bool] = []
    for wall in request.walls:
        try:
            results.append(_spawn_wall(env_id, wall, cache))
        except Exception:
            results.append(False)

    if request.walls:
        try:
            store.recompile(env_id)
        except RejectedBodies as exc:
            hooks.get_node().get_logger().error(f'walls not spawned: {exc}')
            results = [False] * len(results)

    response.ret = results
    return response


spawn_walls_service = Service(
    srv_type=SpawnWalls,
    srv_name='mujoco/SpawnWalls',
    callback=_spawn_walls_cb,
)

__all__ = ['spawn_walls_service']
