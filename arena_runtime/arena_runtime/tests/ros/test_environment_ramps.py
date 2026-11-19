"""A ramp needs a simulator with floor heights, where robots and objects stand on it, elsewhere it is plain floor for robots, pedestrians and the simulator."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

pytest.importorskip("rclpy")

from arena_simulation_setup.shared import Obstacle, Stairs
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
from arena_simulation_setup.tree.World import Level
from arena_simulation_setup.utils.geometry import Pose, Position

if TYPE_CHECKING:
    from arena_runtime.env.environment_manager import EnvironmentManager


def _spawn(stacks: bool, level: Level, check: Callable[[EnvironmentManager, list[Stairs], list[Obstacle]], Awaitable[None]]) -> None:
    """Spawn a level on a stacking or planar simulator and run `check` on what it got."""
    from arena_rclpy_mixins.spin import create_executor
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from arena_runtime.testing.env_host import env_host

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
            await env.spawn_world_obstacles(level)
            await check(env, received, objects)
        finally:
            node._heartbeat_stop.set()
            executor.shutdown()
            await spin
            node.destroy_node()

    asyncio.run(main())


@pytest.mark.parametrize(("stacks", "spawned", "height"), [(True, 1, 0.5), (False, 0, 0.0)])
def test_bodies_stand_on_a_ramp_only_where_levels_stack(tmp_path: Path, stacks: bool, spawned: int, height: float) -> None:
    bundle = tmp_path / "crate"
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.4]]}))

    async def check(env: EnvironmentManager, received: list[Stairs], objects: list[Obstacle]) -> None:
        assert len(received) == spawned
        assert len(env._human_simulator._known_stairs) == spawned
        assert env.footing(Position(12.0, 5.0)) == pytest.approx(height)
        assert env.footing(Position(-1.0, 5.0)) == 0.0
        await env.spawn_obstacles([Obstacle(name="crate", model=ObjectIdentifier.inline(ObjectView(bundle), name="crate"), pose=Pose(position=Position(12.0, 5.0, 0.1)))])
        assert [obstacle.pose.position.z for obstacle in objects] == [pytest.approx(0.1 + height)]

    _spawn(stacks, Level(stairs=[Stairs(name="ramp", kind="ramp", position=Position(12.0, 5.0), size=[24.0, 10.0], rise=1.0)]), check)


@pytest.mark.parametrize(("stacks", "foot", "head"), [(True, -2.3, 1.5), (False, 0.0, 0.0)])
def test_a_stair_on_its_own_level_climbs_only_where_levels_stack(stacks: bool, foot: float, head: float) -> None:
    link = {"kind": "stairs", "size": [5.0, 1.2], "rise": 3.8, "group": "link", "same_level": True}
    bottom = Stairs(name="link_bottom", position=Position(13.5, 8.0, -2.3), mouth_side="+x", to="link_top", **link)
    top = Stairs(name="link_top", position=Position(8.5, 8.0, 1.5), mouth_side="-x", to="link_bottom", top=True, **link)

    async def check(env: EnvironmentManager, received: list[Stairs], objects: list[Obstacle]) -> None:
        assert [stair.joined for stair in received] == [True, True]
        assert [bool(stair.static_boxes()) for stair in received] == [stacks, stacks]
        assert env.footing(Position(15.9, 8.0)) == pytest.approx(foot, abs=0.2)
        assert env.footing(Position(6.1, 8.0)) == pytest.approx(head, abs=0.2)
        assert env.footing(Position(11.0, 8.0)) == pytest.approx((foot + head) / 2.0, abs=0.3)

    _spawn(stacks, Level(stairs=[bottom, top]), check)
