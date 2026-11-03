"""Flow regions on the human simulator: realized into the env frame, scoped to one episode."""

from __future__ import annotations

import asyncio

ORIGIN = (30.0, -12.0)
POLYGON = [(1.0, 1.0), (4.0, 1.0), (4.0, 3.0), (1.0, 3.0)]


def test_flow_regions_are_realized_into_the_env_frame():
    from arena_runtime.sim.dummy_simulator import DummySimulator

    from arena_runtime.env.constants import Constants
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.testing.env_host import env_host
    from arena_runtime.shared import Position, Region
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.humans.noop import NoopHumanSimulator

    async def main() -> None:
        node = env_host()
        try:
            realizer = Realizer(Realizer._Configuration(x=ORIGIN[0], y=ORIGIN[1], prefix="env_1"))
            sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
            region = Region(name="queue", type="source", polygon=[Position(x, y) for x, y in POLYGON])
            await env.setup_regions([region])
            config = realizer.get_config()
            engine = [(v.x - config.x, v.y - config.y) for v in human._known_regions["queue"].polygon]
            assert engine == POLYGON
            assert region.polygon == [Position(x, y) for x, y in POLYGON]
        finally:
            node._heartbeat_stop.set()
            node.destroy_node()

    asyncio.run(main())


def test_flow_regions_are_removed_before_the_next_episode():
    from arena_runtime.sim.dummy_simulator import DummySimulator

    from arena_runtime.env.constants import Constants
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.testing.env_host import env_host
    from arena_runtime.shared import Position, Region
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.humans.noop import NoopHumanSimulator

    async def main() -> None:
        node = env_host()
        try:
            realizer = Realizer(Realizer._Configuration(x=ORIGIN[0], y=ORIGIN[1], prefix="env_1"))
            sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
            polygon = [Position(x, y) for x, y in POLYGON]
            await env.setup_regions([Region(name="visitor_entrance", type="source", polygon=polygon), Region(name="exit", type="sink", polygon=polygon)])
            assert set(human._known_regions) == {"visitor_entrance", "exit"}
            await env.before_reset_episode()
            assert human._known_regions == {}
        finally:
            node._heartbeat_stop.set()
            node.destroy_node()

    asyncio.run(main())
