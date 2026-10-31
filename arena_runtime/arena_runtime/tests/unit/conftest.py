"""A paused flatsim server behind a FlatsimSimulator, for shim and articulated link tests."""

from __future__ import annotations

import asyncio
import contextlib
import typing

import pytest

DT = 1.0 / 30.0
NAMESPACE = "/flatsim_unit/env_0"


@pytest.fixture
def on_flatsim():
    pytest.importorskip("flatsim_msgs")

    def run(body: typing.Callable[..., typing.Awaitable[None]]) -> None:
        import rclpy.parameter
        from arena_rclpy_mixins import ArenaMixinNode
        from arena_rclpy_mixins.shared import Namespace
        from arena_rclpy_mixins.spin import create_executor
        from task_generator.manager.realizer import Realizer

        from arena_runtime.sim.flatsim_simulator import FlatsimSimulator
        from flatsim.server import FlatsimServer

        async def main() -> None:
            executor = create_executor()
            server = FlatsimServer(parameter_overrides=[rclpy.parameter.Parameter("physics_dt", value=DT), rclpy.parameter.Parameter("paused", value=True)])
            node = ArenaMixinNode("flatsim_unit", namespace=NAMESPACE)
            executor.add_node(server)
            executor.add_node(node)
            spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
            sim = None
            try:
                sim = await asyncio.wait_for(FlatsimSimulator.create(node=node, namespace=Namespace(NAMESPACE), realizer=Realizer(), env_id=0), timeout=30.0)
                await body(sim, server._scene.world)
            finally:
                if sim is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(sim.shutdown(), timeout=10.0)
                executor.shutdown()
                await spin
                executor.remove_node(node)
                executor.remove_node(server)
                node.destroy_node()
                server.destroy_node()

        asyncio.run(main())

    return run
