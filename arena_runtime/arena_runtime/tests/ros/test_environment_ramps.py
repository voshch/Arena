"""A ramp needs a simulator with floor heights, where robots and objects stand on it, elsewhere it is plain floor for robots, pedestrians and the simulator."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

pytest.importorskip("rclpy")

from arena_simulation_setup.shared import Obstacle, Stairs
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
from arena_simulation_setup.tree.World import Level
from arena_simulation_setup.utils.geometry import Pose, Position


@pytest.mark.parametrize(("stacks", "spawned", "height"), [(True, 1, 0.5), (False, 0, 0.0)])
def test_bodies_stand_on_a_ramp_only_where_levels_stack(tmp_path: Path, stacks: bool, spawned: int, height: float) -> None:
    from arena_rclpy_mixins.spin import create_executor
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from arena_runtime.testing.env_host import env_host

    bundle = tmp_path / "crate"
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.4]]}))
    received: list[Stairs] = []
    objects: list[Obstacle] = []

    class Simulator(DummySimulator):
        STACKS_LEVELS = stacks

        async def spawn_stairs(self, stairs: Sequence[Stairs]) -> bool:
            received.extend(stairs)
            return await super().spawn_stairs(stairs)

        async def obstacle_spawn(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
            objects.extend(obstacles)
            return await super().obstacle_spawn(obstacles)

    async def main() -> None:
        executor = create_executor()
        node = env_host()
        executor.add_node(node)
        spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
        try:
            realizer = Realizer(Realizer._Configuration(x=100.0, y=50.0, prefix="env_0"), level_origins={"0": (0.0, 0.0)})
            sim = Simulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
            ramp = Stairs(name="ramp", kind="ramp", position=Position(12.0, 5.0), size=[24.0, 10.0], rise=1.0)
            await env.spawn_world_obstacles(Level(stairs=[ramp]))
            assert len(received) == spawned
            assert len(human._known_stairs) == spawned
            assert env.footing(Position(12.0, 5.0)) == pytest.approx(height)
            assert env.footing(Position(-1.0, 5.0)) == 0.0
            await env.spawn_obstacles([Obstacle(name="crate", model=ObjectIdentifier.inline(ObjectView(bundle), name="crate"), pose=Pose(position=Position(12.0, 5.0, 0.1)))])
            assert [obstacle.pose.position.z for obstacle in objects] == [pytest.approx(0.1 + height)]
        finally:
            node._heartbeat_stop.set()
            executor.shutdown()
            await spin
            node.destroy_node()

    asyncio.run(main())
