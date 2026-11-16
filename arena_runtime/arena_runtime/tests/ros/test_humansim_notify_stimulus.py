"""ArenaHumanSimulator.notify_stimulus against a live arena_humansim engine."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("tf2_ros")
    pytest.importorskip("arena_humansim_msgs.msg")
    pytest.importorskip("arena_humansim.core.agent_manager")


def _with_engine(scenario: Callable[..., Awaitable[None]], engine_up: bool = True) -> None:
    import tf2_ros
    from arena_humansim.core.agent_manager import AgentManager
    from arena_humansim.params import MODE_SUBSYSTEM
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.acoustics.noop import NoopAcousticsSimulator
    from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator as Human

    class _Node(ServiceNamespace, AsyncNode):
        pass

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> None:
        node = _Node(f"notify_stimulus_{suffix}", namespace=str(ns))
        engine = None
        executor = SingleThreadedExecutor()
        try:
            node.tf_buffer = tf2_ros.Buffer()
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=""))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            acoustics = NoopAcousticsSimulator(node=node, namespace=ns)
            human = Human(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            executor.add_node(node)
            if engine_up:
                engine = AgentManager(
                    namespace=node.get_fully_qualified_name(),
                    parameter_overrides=[Parameter("mode", value=MODE_SUBSYSTEM), Parameter("publish_markers", value=0)],
                )
                executor.add_node(engine)

            if engine_up:
                for client in (human._spawn_client, human._notify_stimulus_client):
                    for _ in range(500):
                        if client.client.service_is_ready():
                            break
                        executor.spin_once(timeout_sec=0.02)

            async def drive(coro: Awaitable[object]) -> object:
                future = asyncio.ensure_future(coro)
                for _ in range(500):
                    if future.done():
                        return future.result()
                    executor.spin_once(timeout_sec=0.02)
                    await asyncio.sleep(0)
                raise AssertionError("call never completed")

            await scenario(human, engine, drive)
        finally:
            executor.shutdown()
            if engine is not None:
                engine.destroy_node()
            node.destroy_node()

    asyncio.run(main())


async def _spawn_ped(human, drive) -> int:
    from arena_runtime.shared import Pose, Position

    ped = human._runtime_obstacle(name="edge_0", pose=Pose(Position(x=2.0, y=3.0)))
    human._known_obstacles.create_or_get(name="edge_0", obstacle=ped)
    assert await drive(human._spawn_dynamic_obstacles_impl([ped])) == [ped]
    (ped_id,) = human._bridge_agent_ids
    return ped_id


def test_a_spawned_name_reaches_the_engine_under_its_agent_id(rclpy_context):
    async def scenario(human, engine, drive) -> None:
        ped_id = await _spawn_ped(human, drive)
        assert await drive(human.notify_stimulus("edge_0", "go", 0.5)) is True
        assert [(aid, stimulus, intensity) for _due, aid, stimulus, intensity in engine._pending_stimuli] == [(ped_id, "go", 0.5)]

    _with_engine(scenario)


def test_an_int_id_passes_through_to_the_engine(rclpy_context):
    async def scenario(human, engine, drive) -> None:
        ped_id = await _spawn_ped(human, drive)
        assert await drive(human.notify_stimulus(ped_id, "form", 1.0)) is True
        assert [(aid, stimulus) for _due, aid, stimulus, _intensity in engine._pending_stimuli] == [(ped_id, "form")]

    _with_engine(scenario)


def test_an_unknown_name_is_not_sent(rclpy_context):
    async def scenario(human, engine, drive) -> None:
        await _spawn_ped(human, drive)
        assert await drive(human.notify_stimulus("ghost", "go", 1.0)) is False
        assert list(engine._pending_stimuli) == []

    _with_engine(scenario)


def test_an_id_the_engine_does_not_know_is_refused(rclpy_context):
    async def scenario(human, engine, drive) -> None:
        ped_id = await _spawn_ped(human, drive)
        assert await drive(human.notify_stimulus(ped_id + 100, "go", 1.0)) is False
        assert list(engine._pending_stimuli) == []

    _with_engine(scenario)


def test_a_call_without_an_engine_times_out_as_not_taken(rclpy_context):
    async def scenario(human, engine, drive) -> None:
        del engine
        human._agent_names = {4: "edge_0"}
        human._notify_stimulus_client._timeout = 0.3
        assert await drive(human.notify_stimulus("edge_0", "go", 1.0)) is False

    _with_engine(scenario, engine_up=False)
