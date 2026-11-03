from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import typing
from collections.abc import AsyncIterator, Callable, Collection, Mapping

import attrs

from arena_runtime.env.node import EnvNode
from arena_runtime.env.task_layer import EpisodeRecord, ResetPlan, TaskContext, TaskModeOverrides, TaskModes, validate_modes

if typing.TYPE_CHECKING:
    import rclpy.task
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    from arena_runtime.env.robot_manager.task_runner import PhaseState


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
    """A bound task without modes, done once `finished` is set."""

    def __init__(self) -> None:
        self.finished = False
        self.ticks: list[tuple[dict[str, PhaseState], set[str]]] = []

    @property
    async def done(self) -> bool:
        return self.finished

    async def reset_robots(self, record: EpisodeRecord) -> ResetPlan:
        del record
        return ResetPlan()

    async def reset_obstacles(self, plan: ResetPlan, record: EpisodeRecord) -> None:
        del plan, record

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        self.ticks.append((dict(ended), set(assigned)))

    def after_reset(self) -> None:
        pass

    async def teardown(self) -> None:
        pass


class FakeTaskParams:
    """A mode selection that records what select was given."""

    def __init__(self) -> None:
        self.modes: TaskModes = {"robots": ["fake_robots"], "obstacles": ["fake_obstacles"], "modules": ["fake_module"]}
        self.always_on: list[str] = []
        self.selected: list[TaskModes] = []

    def resolve(self, modes: TaskModes) -> TaskModes:
        resolved = dict(self.modes)
        for axis, names in modes.items():
            if axis == "modules":
                resolved[axis] = [*names, *(name for name in self.always_on if name not in names)]
            elif names:
                resolved[axis] = names
        return resolved

    def select(self, modes: TaskModes) -> None:
        self.selected.append(modes)
        self.modes = self.resolve(modes)


class FakeTaskLayer:
    """A task layer with a few modes that records the context it was given and the records it was asked about, `choose` deciding what the next episode changes and `gate` holding the answer back while set."""

    NAMESPACES: typing.ClassVar[dict[str, dict[str, str]]] = {
        "robots": {"fake_robots": "task.fake_robots", "fake_other": "task.fake_other"},
        "obstacles": {"fake_obstacles": "task.fake_obstacles"},
        "modules": {"fake_module": "task.fake_module"},
    }

    def __init__(self) -> None:
        self.ctx: TaskContext | None = None
        self.task = FakeTask()
        self.params = FakeTaskParams()
        self.previous: list[EpisodeRecord] = []
        self.gate: asyncio.Event | None = None
        self.choose: Callable[[EpisodeRecord, TaskModeOverrides], TaskModeOverrides] = lambda previous, queued: queued

    def declare_task_params(self, node: object) -> FakeTaskParams:
        del node
        return self.params

    def declare_mode_schema(self, node: ROSParamServer, axis: str, mode: str) -> None:
        name = f"{self.NAMESPACES[axis][mode]}.level"
        if not node.has_parameter(name):
            node.declare_parameter(name, 1)

    async def create_task(self, *, node: object, ctx: TaskContext, params: object) -> FakeTask:
        del node, params
        self.ctx = ctx
        return self.task

    def validate_modes(self, modes: TaskModes) -> TaskModes:
        return validate_modes(modes, {axis: list(names) for axis, names in self.NAMESPACES.items()}, ("modules",))

    def mode_namespaces(self) -> dict[str, dict[str, str]]:
        return self.NAMESPACES

    def fleet_only_robots(self) -> list[str]:
        return []

    async def next_episode(self, *, previous: EpisodeRecord, queued: TaskModeOverrides) -> TaskModeOverrides:
        self.previous.append(attrs.evolve(previous))
        if self.gate is not None:
            await self.gate.wait()
        return self.choose(previous, queued)


class ArenaRig(typing.NamedTuple):
    executor: object
    arena: object
    probe: object
    env_id: int
    namespace: str


@contextlib.asynccontextmanager
async def arena_on_dummy(on_taken: Callable[[str], object] | None = None) -> AsyncIterator[ArenaRig]:
    """A real ArenaRuntimeNode on the dummy sim with one registered env slot, spun on one executor, on_taken told when another runtime holds the domain."""
    import rclpy
    from arena_rclpy_mixins.spin import create_executor
    from arena_runtime_msgs.srv import RegisterEnv

    from arena_runtime.arena_runtime_node import ArenaRuntimeNode

    loop = asyncio.get_running_loop()
    executor = create_executor()
    arena = ArenaRuntimeNode()
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
            if on_taken is not None and "already running" in str(exc):
                on_taken(str(exc))
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
