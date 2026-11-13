"""Removing the world deletes every pedestrian disc: a disc no one deletes stays in the server, seen by the robots'
sensors, after its pedestrian is gone."""

from __future__ import annotations

import asyncio
import typing


async def _until(pred: typing.Callable[[], bool], what: str, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.02)


def _peds(*peds: tuple[str, float]) -> typing.Any:
    from arena_people_msgs.msg import Pedestrian, Pedestrians

    msg = Pedestrians()
    for name, x in peds:
        ped = Pedestrian(name=name)
        ped.pose.position.x = x
        ped.pose.orientation.w = 1.0
        msg.pedestrians.append(ped)
    return msg


def test_removing_the_world_deletes_every_disc_and_a_later_pedestrian_of_the_same_name_gets_a_fresh_one(on_flatsim):
    async def body(sim, world):
        standing, walking = sim._name("flow_1220"), sim._name("flow_7")
        await sim.pedestrian_update(_peds(("flow_1220", 1.0), ("flow_7", 3.0)))
        await _until(lambda: standing in world and walking in world, "both discs")
        assert await sim.remove_world()
        await _until(lambda: standing not in world and walking not in world, "no disc after the world went")
        await sim.pedestrian_update(_peds(("flow_7", 5.0)))
        await _until(lambda: walking in world, "the new pedestrian's disc")
        assert standing not in world

    on_flatsim(body)
