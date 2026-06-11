"""EditPrims service handler: update pose and scale of existing prim bodies."""

from __future__ import annotations

from arena_mujoco_msgs.srv import EditPrims

from arena_mujoco.context import get_scene_store
from arena_mujoco.mesh_mat import scale_body

from .utils import Service


def _edit_prims_cb(
    request: EditPrims.Request,
    response: EditPrims.Response,
) -> EditPrims.Response:
    store = get_scene_store()
    env_id = int(request.env_id)

    results: list[bool] = []
    for prim in request.prims:
        ok = True

        if request.pose:
            pose = prim.pose
            pos = (pose.position.x, pose.position.y, pose.position.z)
            quat = (
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            )
            ok = store.set_body_pose(env_id, prim.name, pos, quat)

        if request.scale:
            scale = (prim.scale.x, prim.scale.y, prim.scale.z)
            ok = min(scale) > 0.0 and store.edit_body(env_id, prim.name, lambda spec, body, scale=scale: scale_body(spec, store.asset_cache(env_id), body, scale)) and ok

        results.append(ok)

    response.ret = results
    return response


edit_prims_service = Service(
    srv_type=EditPrims,
    srv_name='mujoco/EditPrims',
    callback=_edit_prims_cb,
)

__all__ = ['edit_prims_service']
