"""Task._reset_episode on the dummy simulator: the order of a reset, all of it inside the paused window."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest

ROBOTS = "reset_order_robots"
OBSTACLES = "reset_order_obstacles"
WINDOW = ["set_up", "launch_pending", "before_reset", "staged_params", "staged_human_params", "module_before", "tm_robots", "tm_obstacles", "respawn", "module_after", "after_reset"]


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("arena_runtime.sim.dummy_simulator")


class _Robots:
    managers: dict = {}

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def bind_abort(self, abort: Callable[[str], None]) -> None:
        del abort

    async def set_up(self) -> None:
        self.log.append("set_up")

    async def launch_pending(self) -> None:
        self.log.append("launch_pending")


class _TmRobots:
    start_poses: dict = {}

    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def reset(self) -> None:
        self.log.append("tm_robots")


class _TmObstacles:
    lights: tuple = ()

    def __init__(self, log: list[str], error: Exception | None = None) -> None:
        self.log = log
        self.error = error

    async def reset(self, seed: int) -> tuple[list, list]:
        del seed
        self.log.append("tm_obstacles")
        if self.error is not None:
            raise self.error
        return [], []


class _Module:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def before_reset(self) -> None:
        self.log.append("module_before")

    def after_reset(self) -> None:
        self.log.append("module_after")


def _on_task(error: Exception | None, body: Callable[..., Awaitable[None]]) -> list[str]:
    import rclpy
    import rclpy.executors
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator

    from task_generator.constants import Constants
    from task_generator.constants.rng import EpisodeRng
    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.manager.realizer import Realizer
    from task_generator.simulators.acoustics import AcousticsSimulatorRegistry
    from task_generator.simulators.human import HumanSimulatorRegistry
    from task_generator.tasks.task import Task

    log: list[str] = []

    class _Node(ServiceNamespace, AsyncNode):
        conf = SimpleNamespace(
            General=SimpleNamespace(RNG=EpisodeRng()),
            TaskMode=SimpleNamespace(TM_CONFIG=SimpleNamespace(value=""), TM_ROBOTS=SimpleNamespace(value=ROBOTS), TM_OBSTACLES=SimpleNamespace(value=OBSTACLES)),
        )

        def _apply_staged_params(self) -> None:
            log.append("staged_params")

        async def _apply_staged_human_params(self) -> None:
            log.append("staged_human_params")

        def reset_timeline(self) -> None:
            pass

    class _Sim(DummySimulator):
        async def before_reset_episode(self) -> bool:
            log.append("before_reset")
            return await super().before_reset_episode()

        async def after_reset_episode(self) -> bool:
            log.append("after_reset")
            return await super().after_reset_episode()

    class _Environment(EnvironmentManager):
        async def respawn(self, callback: Callable[[], Awaitable[object]]) -> None:
            log.append("respawn")
            await super().respawn(callback)

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> None:
        node = _Node(f"reset_lifecycle_{suffix}", namespace=str(ns))
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        stop = threading.Event()

        def spin() -> None:
            while not stop.is_set():
                executor.spin_once(timeout_sec=0.02)

        thread = threading.Thread(target=spin, daemon=True)
        thread.start()
        human = None
        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env"))
            simulator = _Sim(node=node, namespace=ns, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=ns)
            human = await HumanSimulatorRegistry.get(Constants.HumanSimulator.DUMMY, node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            environment = _Environment(node=node, simulator=simulator, human_simulator=human, realizer=realizer)
            task = Task(node=node, environment_manager=environment, robots_manager=_Robots(log), world_manager=SimpleNamespace())
            task._Task__param_tm_robots = ROBOTS
            task._Task__param_tm_obstacles = OBSTACLES
            task._Task__tm_robots = _TmRobots(log)
            task._Task__tm_obstacles = _TmObstacles(log, error)
            task._Task__modules = [_Module(log)]
            await body(task)
        finally:
            if human is not None:
                human._keepalive_task.cancel()
            stop.set()
            thread.join(timeout=2.0)
            executor.remove_node(node)
            node.destroy_node()

    asyncio.run(main())
    return log


def test_a_reset_runs_every_step_in_order_inside_the_paused_window(rclpy_context) -> None:
    async def body(task) -> None:
        await task.reset(seed=3)

    assert _on_task(None, body) == WINDOW


def test_a_failing_mode_still_closes_the_paused_window(rclpy_context) -> None:
    async def body(task) -> None:
        with pytest.raises(RuntimeError, match="no obstacles today"):
            await task.reset(seed=3)

    assert _on_task(RuntimeError("no obstacles today"), body) == [*WINDOW[: WINDOW.index("tm_obstacles") + 1], "after_reset"]
