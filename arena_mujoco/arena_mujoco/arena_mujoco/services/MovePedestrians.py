"""MovePedestrians service: explicit teleport that restarts the pedestrian's clip."""

from __future__ import annotations

from arena_people_msgs.srv import MovePedestrians

from .SpawnPedestrians import place_pedestrian
from .utils import Service


def _move_pedestrians_cb(
    request: MovePedestrians.Request,
    response: MovePedestrians.Response,
) -> MovePedestrians.Response:
    results: list[int] = []
    for ped in request.pedestrians:
        results.append(MovePedestrians.Response.SUCCESS if place_pedestrian(ped.name, ped.pose, teleport=True) else MovePedestrians.Response.NOT_FOUND)
    response.results = results
    return response


move_pedestrians_service = Service(
    srv_type=MovePedestrians,
    srv_name='mujoco/MovePedestrians',
    callback=_move_pedestrians_cb,
)

__all__ = ['move_pedestrians_service']
