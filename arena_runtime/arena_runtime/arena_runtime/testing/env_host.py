from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import types
import typing
from collections.abc import AsyncIterator

from arena_runtime.env.node import EnvNode
from arena_runtime.env.task_layer import ResetPlan, TaskContext, TaskModeNames

if typing.TYPE_CHECKING:
    import rclpy.task


class EnvHost(EnvNode):
    """An env node without a task layer, for tests that need a live node to host managers."""

    async def _set_up_task(self) -> None:
        pass


def env_host() -> EnvHost:
    return EnvHost("env_host", automatically_declare_parameters_from_overrides=True)


def register_as_ros_future(ros_future: rclpy.task.Future) -> concurrent.futures.Future:
    out: concurrent.futures.Future = concurrent.futures.Future()
    ros_future.add_done_callback(lambda done: out.set_result(done.result()))
    return out


class FakeTask:
    """A bound task without modes."""

    tm_robots = types.SimpleNamespace(start_poses={})

    @property
    async def done(self) -> bool:
        return False

    async def prepare_reset(self, seed: int) -> ResetPlan:
        del seed
        return ResetPlan(obstacles=[], dynamic_obstacles=[], lights=())

    def after_reset(self) -> None:
        pass

    async def teardown(self) -> None:
        pass


class FakeTaskLayer:
    """A task layer without modes that records the context it was given."""

    def __init__(self) -> None:
        self.ctx: TaskContext | None = None
        self.params = types.SimpleNamespace(robots="fake_robots", obstacles="fake_obstacles", modules=[])

    def declare_task_params(self, node: object) -> object:
        del node
        return self.params

    def declare_mode_schemas(self, node: object) -> None:
        del node

    async def create_task(self, *, node: object, ctx: TaskContext) -> FakeTask:
        del node
        self.ctx = ctx
        return FakeTask()

    def mode_names(self) -> TaskModeNames:
        return TaskModeNames(robots=["fake_robots"], obstacles=["fake_obstacles"], modules=[])

    def mode_vocabulary(self) -> TaskModeNames:
        return self.mode_names()

    def mode_namespaces(self) -> dict[str, dict[str, str]]:
        return {"robots": {"fake_robots": "task.fake_robots"}, "obstacles": {"fake_obstacles": "task.fake_obstacles"}, "modules": {}}

    def fleet_only_robots(self) -> list[str]:
        return []


class ArenaRig(typing.NamedTuple):
    executor: object
    arena: object
    probe: object
    env_id: int
    namespace: str


@contextlib.asynccontextmanager
async def arena_on_dummy() -> AsyncIterator[ArenaRig]:
    """A real ArenaNode on the dummy sim with one registered env slot, spun on one executor."""
    import pytest
    import rclpy
    from arena_rclpy_mixins.spin import create_executor
    from arena_runtime_msgs.srv import RegisterEnv

    from arena_runtime.arena_node import ArenaNode

    loop = asyncio.get_running_loop()
    executor = create_executor()
    arena = ArenaNode()
    arena.declare_parameter("physics_dt", 0.02)
    arena.declare_parameter("sim", "dummy")
    arena.declare_parameter("heartbeat_timeout_sec", 600.0)
    executor.add_node(arena)
    spin = loop.run_in_executor(None, executor.spin)
    probe = rclpy.create_node("arena_rig_probe")
    executor.add_node(probe)
    try:
        try:
            await arena.setup()
        except RuntimeError as exc:
            if "already running" in str(exc):
                pytest.skip(str(exc))
            raise
        register = probe.create_client(RegisterEnv, "/arena/register_env")
        assert await asyncio.to_thread(register.wait_for_service, 10.0)
        registered = await asyncio.wrap_future(register_as_ros_future(register.call_async(RegisterEnv.Request(caller_id="/arena_rig", env_id=0xFFFF, ns=""))))
        assert registered.success, registered.error_msg
        yield ArenaRig(executor, arena, probe, int(registered.env_id), registered.ns)
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(arena.teardown(), timeout=5.0)
        if arena._clock_task is not None:
            arena._clock_task.cancel()
            await asyncio.gather(arena._clock_task, return_exceptions=True)
        executor.shutdown()
        await spin
        probe.destroy_node()
        arena.destroy_node()
