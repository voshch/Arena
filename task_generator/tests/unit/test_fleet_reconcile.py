from __future__ import annotations

import asyncio
import typing

import pytest


def _on_robots_manager(body: typing.Callable[[object, object], typing.Awaitable[None]]) -> None:
    pytest.importorskip("rclpy")
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.robot_manager import RobotsManager
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.sim.dummy_simulator import DummySimulator

    async def main() -> None:
        node = EpisodeNode()
        try:
            realizer = Realizer(level_origins={"0": (0.0, 0.0)})
            sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
            await body(node, RobotsManager(node=node, environment_manager=env))
        finally:
            node._heartbeat_stop.set()
            node.destroy_node()

    asyncio.run(main())


def test_unnamed_entries_are_spawned_under_platform_names_and_named_ones_keep_theirs(rclpy_context: object) -> None:
    from arena_runtime.env.task_layer import FleetEntry

    async def body(node: object, manager: object) -> None:
        manager.reconcile([FleetEntry(model="jackal"), FleetEntry(model="jackal", name="lead")], node._staged_robot)
        queued = {name: robot.model.name for name, robot in manager._diff.to_add.items()}
        assert queued == {"jackal_0": "jackal", "lead": "jackal"}

    _on_robots_manager(body)


def test_a_second_fleet_cancels_the_queued_spawns_it_does_not_list(rclpy_context: object) -> None:
    from arena_runtime.env.task_layer import FleetEntry

    async def body(node: object, manager: object) -> None:
        manager.reconcile([FleetEntry(model="jackal"), FleetEntry(model="jackal")], node._staged_robot)
        manager.reconcile([FleetEntry(model="jackal", name="jackal_1")], node._staged_robot)
        assert sorted(manager._diff.to_add) == ["jackal_1"]

    _on_robots_manager(body)


def test_a_fleet_naming_a_robot_twice_is_rejected(rclpy_context: object) -> None:
    from arena_runtime.env.task_layer import FleetEntry

    async def body(node: object, manager: object) -> None:
        with pytest.raises(ValueError, match="twice"):
            manager.reconcile([FleetEntry(model="jackal", name="a"), FleetEntry(model="jackal", name="a")], node._staged_robot)

    _on_robots_manager(body)
