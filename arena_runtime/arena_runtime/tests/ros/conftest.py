"""In-process ArenaRuntimeNode on the dummy, flatsim or gazebo host plus a probe node, for lockstep tests."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import typing
from pathlib import Path

import pytest

DT = 0.02
LATCHED_DEPTH = 1
GZ_WORLD = f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="default"><physics type="ode"><max_step_size>{DT}</max_step_size><real_time_factor>1.0</real_time_factor></physics></world></sdf>
"""


class Rig:
    def __init__(self, arena: typing.Any, probe: typing.Any, executor: typing.Any, servers: typing.Sequence[typing.Any] = ()) -> None:
        self.arena = arena
        self.servers = list(servers)
        self.probe = probe
        self.executor = executor
        self._extra: list[typing.Any] = []

    def add_node(self, node: typing.Any) -> None:
        self.executor.add_node(node)
        self._extra.append(node)

    def destroy_extra(self) -> None:
        for node in self._extra:
            self.executor.remove_node(node)
            node.destroy_node()
        self._extra.clear()

    @staticmethod
    async def until(pred: typing.Callable[[], bool], timeout: float, what: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not pred():
            if loop.time() >= deadline:
                raise AssertionError(f"timed out after {timeout}s waiting for {what}")
            await asyncio.sleep(0.02)

    async def drain(self, stable: float = 0.3, timeout: float = 20.0) -> float:
        """Wait until the sim clock is unchanged for `stable` seconds, returning its value."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last = self.probe.clock()
        since = loop.time()
        while loop.time() - since < stable:
            if loop.time() >= deadline:
                raise AssertionError(f"sim clock did not settle within {timeout}s")
            await asyncio.sleep(0.02)
            now = self.probe.clock()
            if now != last:
                last = now
                since = loop.time()
        return last

    async def frozen(self, seconds: float) -> float:
        """Assert the sim clock does not move for `seconds`, returning its value."""
        before = await self.drain()
        await asyncio.sleep(seconds)
        assert self.probe.clock() == pytest.approx(before, abs=1e-9), "sim clock moved while it should be frozen"
        return before


async def _run(scenario: typing.Callable[[Rig], typing.Awaitable[None]], sim: str) -> None:
    import rclpy.qos
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.spin import create_executor
    from arena_rclpy_mixins.Time import Time
    from arena_runtime_msgs.msg import LockstepHeartbeat, LockstepRegistration, LockstepStatus
    from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep, LifecycleUnpauseWindow, LockstepRegister, LockstepStart
    from std_msgs.msg import Bool
    from std_srvs.srv import Trigger

    from arena_runtime.arena_runtime_node import ArenaRuntimeNode

    latched = rclpy.qos.QoSProfile(depth=LATCHED_DEPTH, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)

    class Probe(AsyncNode):
        """Observes arena state and drives the lockstep services, optionally as a producer."""

        def __init__(self) -> None:
            super().__init__("lockstep_probe")
            self.status: LockstepStatus | None = None
            self.paused: Bool | None = None
            self.create_subscription(LockstepStatus, "/arena/state/lockstep", self._on_status, latched)
            self.create_subscription(Bool, "/arena/state/paused", self._on_paused, latched)
            self.beat_topic = "/probe/beat"
            self.beat = self.create_publisher(LockstepHeartbeat, self.beat_topic, 10)
            self.hold = self.create_client_wrapper(LifecycleHold, "/arena/sim_lifecycle/hold", timeout=30.0)
            self.step = self.create_client_wrapper(LifecycleStep, "/arena/sim_lifecycle/step", timeout=30.0)
            self.window = self.create_client_wrapper(LifecycleUnpauseWindow, "/arena/sim_lifecycle/unpause_window", timeout=30.0)
            self.start = self.create_client_wrapper(LockstepStart, "/arena/sim_lifecycle/lockstep/start", timeout=30.0)
            self.register = self.create_client_wrapper(LockstepRegister, "/arena/sim_lifecycle/lockstep/register", timeout=30.0)
            self.stop = self.create_client_wrapper(Trigger, "/arena/sim_lifecycle/lockstep/stop", timeout=30.0)

        def _on_status(self, msg: LockstepStatus) -> None:
            self.status = msg

        def _on_paused(self, msg: Bool) -> None:
            self.paused = msg

        def clock(self) -> float:
            return self.sim_time.to_seconds()

        def status_tick(self) -> float:
            assert self.status is not None
            return Time.from_msg(self.status.tick).to_seconds()

        def stalled_on(self, name: str) -> bool:
            return self.status is not None and self.status.active and list(self.status.waiting_on) == [name]

        def registered_callers(self) -> list[str]:
            return [] if self.status is None else [reg.caller for reg in self.status.registrations]

        def pulse(self, stamp: float) -> None:
            msg = LockstepHeartbeat()
            msg.header.stamp = Time.from_float(stamp).to_msg()
            self.beat.publish(msg)

        async def register_channels(self, channels: list, caller: str = "probe", env: str = "") -> LockstepRegister.Response:
            request = LockstepRegister.Request()
            request.registration = LockstepRegistration(caller=caller, env=env, channels=channels)
            response = await self.register.call_timeout(request)
            assert response is not None
            return response

        async def set_window(self, caller: str, acquire: bool) -> None:
            action = LifecycleUnpauseWindow.Request.ACQUIRE if acquire else LifecycleUnpauseWindow.Request.RELEASE
            response = await self.window.call_timeout(LifecycleUnpauseWindow.Request(action=action, caller_id=caller))
            assert response is not None and response.success, response.error_msg if response is not None else "timeout"

        async def start_run(self, target_rtf: float = 0.0, ungated: bool = False) -> None:
            response = await self.start.call_timeout(LockstepStart.Request(target_rtf=target_rtf, ungated=ungated))
            assert response is not None and response.success, response.error_msg if response is not None else "timeout"

        async def stop_run(self) -> None:
            response = await self.stop.call_timeout(Trigger.Request())
            assert response is not None and response.success

    executor = create_executor()
    servers: list[typing.Any] = []
    if sim == "flatsim":
        import rclpy.parameter

        from flatsim.server import FlatsimServer

        servers.append(FlatsimServer(parameter_overrides=[rclpy.parameter.Parameter("physics_dt", value=DT)]))
        executor.add_node(servers[0])
    arena = ArenaRuntimeNode()
    arena.declare_parameter("physics_dt", DT)
    arena.declare_parameter("sim", sim)
    probe = Probe()
    executor.add_node(arena)
    executor.add_node(probe)
    rig = Rig(arena, probe, executor, servers)
    loop = asyncio.get_running_loop()
    spin = loop.run_in_executor(None, executor.spin)
    teardown_hang: TimeoutError | None = None
    try:
        try:
            await arena.setup()
        except RuntimeError as e:
            if "already running" in str(e):
                pytest.skip(str(e))
            raise
        assert await probe.start.ensure(timeout_sec=30.0)
        await rig.until(lambda: probe.clock() > 0.0, 30.0, f"{sim} /clock")
        await scenario(rig)
    finally:
        for cleanup in (arena.teardown, arena.kill_launches):
            try:
                await asyncio.wait_for(cleanup(), timeout=30.0)
            except TimeoutError as e:
                teardown_hang = teardown_hang or e
            except Exception:
                pass
        if arena._clock_task is not None:
            arena._clock_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await arena._clock_task
        executor.shutdown()
        await spin
        rig.destroy_extra()
        executor.remove_node(arena)
        executor.remove_node(probe)
        arena.destroy_node()
        probe.destroy_node()
        for server in servers:
            executor.remove_node(server)
            server.destroy_node()
        if teardown_hang is not None:
            raise teardown_hang


@pytest.fixture()
def arena_rig() -> typing.Callable[[typing.Callable[[Rig], typing.Awaitable[None]]], None]:
    """Runs `scenario(rig)` on a fresh event loop against a spinning ArenaRuntimeNode(sim=dummy)."""

    def run(scenario: typing.Callable[[Rig], typing.Awaitable[None]], *, timeout: float = 90.0) -> None:
        asyncio.run(asyncio.wait_for(_run(scenario, "dummy"), timeout=timeout))

    return run


@pytest.fixture()
def flatsim_rig() -> typing.Callable[[typing.Callable[[Rig], typing.Awaitable[None]]], None]:
    """Runs `scenario(rig)` on a fresh event loop against a spinning ArenaRuntimeNode(sim=flatsim) and a flatsim server."""

    def run(scenario: typing.Callable[[Rig], typing.Awaitable[None]], *, timeout: float = 90.0) -> None:
        pytest.importorskip("flatsim_msgs")
        asyncio.run(asyncio.wait_for(_run(scenario, "flatsim"), timeout=timeout))

    return run


@pytest.fixture()
def gazebo_rig(tmp_path: Path) -> typing.Iterator[typing.Callable[[typing.Callable[[Rig], typing.Awaitable[None]]], None]]:
    """Runs `scenario(rig)` against a spinning ArenaRuntimeNode(sim=gazebo), a running headless gz server on an empty world and its /clock bridge."""
    if shutil.which("gz") is None:
        pytest.skip("gz is not on PATH")
    pytest.importorskip("ros_gz_interfaces")
    world = tmp_path / "default.sdf"
    world.write_text(GZ_WORLD)
    processes = [
        subprocess.Popen(["gz", "sim", "-s", "-r", str(world)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True),
        subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True),
    ]

    def run(scenario: typing.Callable[[Rig], typing.Awaitable[None]], *, timeout: float = 120.0) -> None:
        asyncio.run(asyncio.wait_for(_run(scenario, "gazebo"), timeout=timeout))

    try:
        yield run
    finally:
        for process in processes:
            os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
