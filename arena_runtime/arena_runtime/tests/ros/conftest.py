"""In-process ArenaNode on the dummy sim plus a probe node, for lockstep tests without gz."""

from __future__ import annotations

import asyncio
import contextlib
import typing

import pytest

DT = 0.02
LATCHED_DEPTH = 1


class Rig:
    def __init__(self, arena: typing.Any, probe: typing.Any, executor: typing.Any) -> None:
        self.arena = arena
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

    async def frozen(self, seconds: float) -> float:
        """Assert the sim clock does not move for `seconds`, returning its value."""
        before = self.probe.clock()
        await asyncio.sleep(seconds)
        assert self.probe.clock() == pytest.approx(before, abs=1e-9), "sim clock moved while it should be frozen"
        return before


async def _run(scenario: typing.Callable[[Rig], typing.Awaitable[None]]) -> None:
    import rclpy.executors
    import rclpy.qos
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.Time import Time
    from arena_runtime_msgs.msg import LockstepHeartbeat, LockstepRegistration, LockstepStatus
    from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep, LockstepRegister, LockstepStart
    from std_msgs.msg import Bool
    from std_srvs.srv import Trigger

    from arena_runtime.arena_node import ArenaNode

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
            self.hold = self.create_client_wrapper(LifecycleHold, "/arena/sim_lifecycle/hold", timeout=5.0)
            self.step = self.create_client_wrapper(LifecycleStep, "/arena/sim_lifecycle/step", timeout=5.0)
            self.start = self.create_client_wrapper(LockstepStart, "/arena/sim_lifecycle/lockstep/start", timeout=5.0)
            self.register = self.create_client_wrapper(LockstepRegister, "/arena/sim_lifecycle/lockstep/register", timeout=5.0)
            self.stop = self.create_client_wrapper(Trigger, "/arena/sim_lifecycle/lockstep/stop", timeout=5.0)

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

        async def register_channels(self, channels: list, caller: str = "probe") -> LockstepRegister.Response:
            request = LockstepRegister.Request()
            request.registration = LockstepRegistration(caller=caller, env="", channels=channels)
            response = await self.register.call_timeout(request)
            assert response is not None
            return response

        async def start_run(self, target_rtf: float = 0.0, ungated: bool = False) -> None:
            response = await self.start.call_timeout(LockstepStart.Request(target_rtf=target_rtf, ungated=ungated))
            assert response is not None and response.success, response.error_msg if response is not None else "timeout"

        async def stop_run(self) -> None:
            response = await self.stop.call_timeout(Trigger.Request())
            assert response is not None and response.success

    executor = rclpy.executors.MultiThreadedExecutor()
    arena = ArenaNode()
    arena.declare_parameter("physics_dt", DT)
    probe = Probe()
    executor.add_node(arena)
    executor.add_node(probe)
    rig = Rig(arena, probe, executor)
    loop = asyncio.get_running_loop()
    spin = loop.run_in_executor(None, executor.spin)
    try:
        try:
            await arena.setup()
        except RuntimeError as e:
            if "already running" in str(e):
                pytest.skip(str(e))
            raise
        assert await probe.start.ensure(timeout_sec=10.0)
        await rig.until(lambda: probe.clock() > 0.0, 5.0, "dummy /clock")
        await scenario(rig)
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(arena.teardown(), timeout=5.0)
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


@pytest.fixture()
def arena_rig() -> typing.Callable[[typing.Callable[[Rig], typing.Awaitable[None]]], None]:
    """Runs `scenario(rig)` on a fresh event loop against a spinning ArenaNode(sim=dummy)."""

    def run(scenario: typing.Callable[[Rig], typing.Awaitable[None]], *, timeout: float = 90.0) -> None:
        asyncio.run(asyncio.wait_for(_run(scenario), timeout=timeout))

    return run
