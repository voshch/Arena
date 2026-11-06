"""A ramp needs a simulator with floor heights, elsewhere it is plain floor for robots, pedestrians and the simulator."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

pytest.importorskip("rclpy")

from arena_simulation_setup.shared import Stairs
from arena_simulation_setup.tree.World import Level
from arena_simulation_setup.utils.geometry import Position


@pytest.mark.parametrize(("stacks", "spawned", "height"), [(True, 1, 0.5), (False, 0, 0.0)])
def test_a_ramp_is_plain_floor_where_levels_do_not_stack(stacks: bool, spawned: int, height: float) -> None:
    from arena_rclpy_mixins.spin import create_executor
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from arena_runtime.testing.env_host import env_host

    received: list[Stairs] = []

    class Simulator(DummySimulator):
        STACKS_LEVELS = stacks

        async def spawn_stairs(self, stairs: Sequence[Stairs]) -> bool:
            received.extend(stairs)
            return await super().spawn_stairs(stairs)

    async def main() -> None:
        executor = create_executor()
        node = env_host()
        executor.add_node(node)
        spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
        try:
            realizer = Realizer(level_origins={"0": (0.0, 0.0)})
            sim = Simulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
            ramp = Stairs(name="ramp", kind="ramp", position=Position(12.0, 5.0), size=[24.0, 10.0], rise=1.0)
            await env.spawn_world_obstacles(Level(stairs=[ramp]))
            assert len(received) == spawned
            assert len(human._known_stairs) == spawned
            assert env.ramp_height(Position(12.0, 5.0)) == pytest.approx(height)
        finally:
            node._heartbeat_stop.set()
            executor.shutdown()
            await spin
            node.destroy_node()

    asyncio.run(main())
