"""Every simulator body of a walker is deleted with its walker."""

from __future__ import annotations

import asyncio
import threading
import types
import typing
import uuid

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim_msgs.srv")

from arena_humansim_msgs.msg import AgentFrame
from arena_humansim_msgs.srv import RemoveAgents
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.realizer import Realizer
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.sim.dummy_simulator import DummySimulator
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import SetParametersAtomically


class _Simulator(DummySimulator):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.deleted: list[list[str]] = []
        self.spawned: list[list[str]] = []
        self.updated: list[list[str]] = []

    async def pedestrian_delete(self, pedestrians: typing.Sequence[typing.Any]) -> typing.Sequence[bool]:
        self.deleted.append([pedestrian.sim_path for pedestrian in pedestrians])
        return await super().pedestrian_delete(pedestrians)

    async def pedestrian_spawn(self, pedestrians: typing.Sequence[typing.Any]) -> typing.Sequence[bool]:
        self.spawned.append([pedestrian.sim_path for pedestrian in pedestrians])
        return await super().pedestrian_spawn(pedestrians)

    async def pedestrian_update(self, pedestrians: Pedestrians) -> typing.Sequence[bool]:
        self.updated.append([ped.name for ped in pedestrians.pedestrians])
        return await super().pedestrian_update(pedestrians)


class _Engine:
    def __init__(self, node: typing.Any, *, removed: bool) -> None:
        self.removed = removed
        self.requests: list[RemoveAgents.Request] = []
        node.create_service(RemoveAgents, str(node.service_namespace(ArenaHumanSimulator.SERVICE_REMOVE_AGENTS)), self._remove)
        node.create_service(SetParametersAtomically, str(node.service_namespace("arena_humansim", "set_parameters_atomically")), self._set_params)

    def _remove(self, request: RemoveAgents.Request, response: RemoveAgents.Response) -> RemoveAgents.Response:
        self.requests.append(request)
        response.success = self.removed
        response.message = "removed"
        return response

    def _set_params(self, request: SetParametersAtomically.Request, response: SetParametersAtomically.Response) -> SetParametersAtomically.Response:
        response.result = SetParametersResult(successful=True)
        return response


def _on_adapter(body: typing.Callable[[ArenaHumanSimulator, _Simulator], typing.Awaitable[None]], *, removed: bool = True) -> None:
    import rclpy.executors
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants

    class _Node(ArenaMixinNode):
        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            super().__init__(*args, **kwargs)
            self.conf = types.SimpleNamespace(General=types.SimpleNamespace(RNG=EpisodeRng()))

    async def main() -> None:
        suffix = uuid.uuid4().hex[:8]
        namespace = Namespace(f"/humansim_ped_bodies_{suffix}")
        node = _Node(f"humansim_ped_bodies_{suffix}", namespace=str(namespace))
        realizer = Realizer()
        simulator = _Simulator(node=node, namespace=namespace, realizer=realizer)
        acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=namespace)
        adapter = ArenaHumanSimulator(node=node, namespace=namespace, simulator=simulator, realizer=realizer, acoustics=acoustics)
        _Engine(node, removed=removed)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            for client in (adapter._remove_client, adapter._set_params_client):
                assert await client.ensure(timeout_sec=5.0)
            await body(adapter, simulator)
        finally:
            await simulator.stop_mechanisms()
            executor.shutdown()
            thread.join(timeout=2.0)
            node.destroy_node()

    asyncio.run(main())


def _walkers(adapter: ArenaHumanSimulator, flow: set[int], bridge: set[int]) -> None:
    adapter._flow_agent_ids = set(flow)
    adapter._bridge_agent_ids = set(bridge)
    adapter._agent_names = {aid: f"flow_{aid}" for aid in flow} | {aid: f"scenario_ped_{aid}" for aid in bridge}


def test_a_reset_deletes_the_bodies_of_the_flow_agents_it_removed() -> None:
    async def body(adapter: ArenaHumanSimulator, simulator: _Simulator) -> None:
        _walkers(adapter, flow={1220, 7, 31}, bridge={100})
        assert await adapter._remove_pedestrians_impl()
        realizer = Realizer()
        assert simulator.deleted == [[realizer.prefix(f"flow_{aid}") for aid in (7, 31, 1220)]]
        assert adapter._flow_agent_ids == set() and adapter._agent_names == {}

    _on_adapter(body)


def test_a_reset_without_flow_agents_deletes_nothing() -> None:
    async def body(adapter: ArenaHumanSimulator, simulator: _Simulator) -> None:
        _walkers(adapter, flow=set(), bridge={100})
        assert await adapter._remove_pedestrians_impl()
        assert simulator.deleted == []

    _on_adapter(body)


def test_the_bodies_go_even_when_the_engine_reports_a_failed_removal() -> None:
    async def body(adapter: ArenaHumanSimulator, simulator: _Simulator) -> None:
        _walkers(adapter, flow={4}, bridge=set())
        assert not await adapter._remove_pedestrians_impl()
        assert simulator.deleted == [[Realizer().prefix("flow_4")]]

    _on_adapter(body, removed=False)


def _ped(agent_id: int, name: str) -> Pedestrian:
    return Pedestrian(id=agent_id, name=name)


def test_the_simulator_gets_a_walker_only_under_the_name_of_its_body() -> None:
    async def body(adapter: ArenaHumanSimulator, simulator: _Simulator) -> None:
        _walkers(adapter, flow={7}, bridge={100})
        adapter._agent_names = {7: "env_0/flow_7", 100: "env_0/scenario_ped_100"}
        roster = Pedestrians()
        roster.header.frame_id = "map"
        roster.pedestrians = [_ped(7, "env_0/flow_7"), _ped(8, "8"), _ped(7, "7"), _ped(100, "env_0/scenario_ped_100")]
        named = adapter._named(roster)
        assert [ped.name for ped in named.pedestrians] == ["env_0/flow_7", "env_0/scenario_ped_100"]
        assert named.header.frame_id == "map"

    _on_adapter(body)


def test_an_update_pass_never_hands_the_simulator_a_walker_under_its_bare_id() -> None:
    async def body(adapter: ArenaHumanSimulator, simulator: _Simulator) -> None:
        _walkers(adapter, flow=set(), bridge=set())
        adapter._agent_types = {8: "adult"}
        adapter._arena_pedestrians = Pedestrians(pedestrians=[_ped(8, "8")])
        adapter._curr_agent_states = AgentFrame(agent_id=[8], x=[1.0], y=[2.0], desired_velocity=[1.2])
        task = asyncio.create_task(adapter._pedestrian_update_loop())
        try:
            async with asyncio.timeout(10.0):
                while not simulator.updated:
                    await asyncio.sleep(0.02)
        finally:
            task.cancel()
            await task
        body_name = Realizer().prefix("flow_8")
        assert simulator.spawned == [[body_name]] and adapter._agent_names == {8: body_name}
        assert simulator.updated == [[]]

    _on_adapter(body)
