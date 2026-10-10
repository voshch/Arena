"""DeletePedestrians service: park the mocap body and recycle its handle."""

from __future__ import annotations

from arena_people_msgs.srv import DeletePedestrians

from arena_mujoco import context

from .SpawnPedestrians import _env_free_handles, _registry
from .utils import Service


def _delete_pedestrians_cb(
    request: DeletePedestrians.Request,
    response: DeletePedestrians.Response,
) -> DeletePedestrians.Response:
    store = context.get_scene_store()
    results: list[int] = []
    for name in request.names:
        entry = _registry.pop(name, None)
        if entry is not None:
            env_id, handle = entry
            store.free_mocap(env_id, handle)
            _env_free_handles.setdefault(env_id, []).append(handle)
        results.append(DeletePedestrians.Response.SUCCESS)
    response.results = results
    return response


delete_pedestrians_service = Service(
    srv_type=DeletePedestrians,
    srv_name='mujoco/DeletePedestrians',
    callback=_delete_pedestrians_cb,
)

__all__ = ['delete_pedestrians_service']
