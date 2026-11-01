from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

OFFSET = (10.0, -2.0)
PREFIX = "env"
START_S = 100.0
TICK_S = 1.0 / 30.0
DEADLINE_S = 20.0
MARKER = "quiet_marker"


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("arena_runtime.sim.dummy_simulator")


class Stage:
    def __init__(self, node, clock, environment, bus: list, marker) -> None:
        self.node = node
        self.clock = clock
        self.environment = environment
        self.bus = bus
        self.marker = marker
        self.now = START_S

    async def advance(self, t: float) -> None:
        from builtin_interfaces.msg import Time
        from rosgraph_msgs.msg import Clock

        ticks = math.floor((t - self.now) / TICK_S + 1e-9)
        for step in range(ticks, -1, -1):
            self.now = t - step * TICK_S
            sec, nanosec = divmod(round(self.now * 1e9), 10**9)
            self.clock.publish(Clock(clock=Time(sec=sec, nanosec=nanosec)))
            await asyncio.sleep(0.002)
        deadline = time.monotonic() + DEADLINE_S
        while self.node.sim_time.to_seconds() < t - 1e-6:
            assert time.monotonic() < deadline, f"sim clock stuck at {self.node.sim_time.to_seconds()} on the way to {t}"
            self.clock.publish(Clock(clock=Time(sec=sec, nanosec=nanosec)))
            await asyncio.sleep(0.01)

    async def play(self, dynamic) -> None:
        await self.environment.respawn(lambda: self.environment.spawn_dynamic_obstacles(dynamic))

    async def roster(self, names: list[str]) -> None:
        """Advance the clock by half a second and wait for a batch of it with exactly these pedestrians."""
        seen = len(self.bus)
        await self.advance(self.now + 0.5)
        expected = sorted(f"{PREFIX}/{name}" for name in names)
        deadline = time.monotonic() + DEADLINE_S
        while not any(sorted(ped.name for ped in msg.pedestrians) == expected for msg in self.bus[seen:]):
            assert time.monotonic() < deadline, f"no batch with {expected}, last: {[ped.name for ped in self.bus[-1].pedestrians] if self.bus else None}"
            await asyncio.sleep(0.01)

    async def quiet(self) -> int:
        """Let the wall clock run with the sim clock stopped and return the batch count, which must not grow while a marker gets through."""
        from arena_people_msgs.msg import Pedestrians
        from std_msgs.msg import Header

        await asyncio.sleep(0.3)
        count = len(self.bus)
        await asyncio.sleep(0.6)
        deadline = time.monotonic() + DEADLINE_S
        while self.marker.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "the bus subscription never matched the marker"
            await asyncio.sleep(0.01)
        self.marker.publish(Pedestrians(header=Header(frame_id=MARKER)))
        while not any(msg.header.frame_id == MARKER for msg in self.bus[count:]):
            assert time.monotonic() < deadline, "the marker never got through the bus"
            await asyncio.sleep(0.01)
        assert [msg.header.frame_id for msg in self.bus[count:]] == [MARKER]
        del self.bus[count:]
        return count


def _on_stage(body: Callable[[Stage], Awaitable[None]]) -> None:
    import rclpy
    import rclpy.executors
    from arena_people_msgs.msg import Pedestrians
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.qos import best_effort
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from rosgraph_msgs.msg import Clock

    from task_generator.constants import Constants
    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.manager.realizer import Realizer
    from task_generator.simulators.acoustics import AcousticsSimulatorRegistry
    from task_generator.simulators.human import HumanSimulatorRegistry
    from task_generator.simulators.human.dummy import DummyHumanSimulator

    class _Node(ServiceNamespace, AsyncNode):
        pass

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> None:
        node = _Node(f"dummy_human_{suffix}", namespace=str(ns))
        probe = rclpy.create_node(f"dummy_human_probe_{suffix}")
        bus: list[Pedestrians] = []
        probe.create_subscription(Pedestrians, str(ns("arena_peds")), bus.append, 10)
        clock = probe.create_publisher(Clock, "/clock", best_effort(1))
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(probe)
        stop = threading.Event()

        def spin() -> None:
            while not stop.is_set():
                executor.spin_once(timeout_sec=0.02)

        thread = threading.Thread(target=spin, daemon=True)
        thread.start()
        human = None
        try:
            realizer = Realizer(Realizer._Configuration(x=OFFSET[0], y=OFFSET[1], prefix=PREFIX))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=ns)
            human = await HumanSimulatorRegistry.get(Constants.HumanSimulator.DUMMY, node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            assert type(human) is DummyHumanSimulator
            environment = EnvironmentManager(node=node, simulator=simulator, human_simulator=human, realizer=realizer)
            stage = Stage(node, clock, environment, bus, probe.create_publisher(Pedestrians, str(ns("arena_peds")), 10))
            await stage.advance(START_S)
            await body(stage)
        finally:
            if human is not None:
                human._keepalive_task.cancel()
            stop.set()
            thread.join(timeout=2.0)
            executor.remove_node(node)
            executor.remove_node(probe)
            probe.destroy_node()
            node.destroy_node()

    asyncio.run(main())


def _scenario(tmp_path: Path, name: str, peds: dict[str, tuple[float, float]]) -> list:
    import yaml
    from arena_simulation_setup.tree.World.Scenario import ScenarioView

    directory = tmp_path / "scenarios" / name
    directory.mkdir(parents=True)
    entries = [{"name": ped, "model": "arenian", "pose": [x, y, 0.0]} for ped, (x, y) in peds.items()]
    (directory / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", "humans": entries}))
    return ScenarioView(directory).load().humans


def test_roster_follows_the_sim_clock(rclpy_context, tmp_path: Path) -> None:
    first = _scenario(tmp_path, "first", {"anna": (1.0, 3.0), "ben": (2.0, 3.0)})

    async def body(stage: Stage) -> None:
        await stage.play(first)
        await stage.roster(["anna", "ben"])
        anna = next(ped for ped in stage.bus[-1].pedestrians if ped.name == f"{PREFIX}/anna")
        assert (anna.pose.position.x, anna.pose.position.y) == pytest.approx((1.0 + OFFSET[0], 3.0 + OFFSET[1]))

        before = await stage.quiet()
        await stage.advance(stage.now + 2.0)
        assert 40 <= await stage.quiet() - before <= 61

    _on_stage(body)


def test_a_stopped_sim_clock_repeats_no_roster(rclpy_context, tmp_path: Path) -> None:
    first = _scenario(tmp_path, "first", {"anna": (1.0, 3.0), "ben": (2.0, 3.0)})
    second = _scenario(tmp_path, "second", {"cleo": (7.0, 7.0)})

    async def body(stage: Stage) -> None:
        await stage.play(first)
        await stage.roster(["anna", "ben"])
        ended = await stage.quiet()

        await stage.play(second)
        assert await stage.quiet() == ended
        await stage.roster(["cleo"])
        assert all(ped.name == f"{PREFIX}/cleo" for msg in stage.bus[ended:] for ped in msg.pedestrians)

    _on_stage(body)
