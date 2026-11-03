"""EpisodeNode._reset_episode driving the real Task on the dummy simulator: the order of a reset, all of it inside the paused window."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest

ROBOTS = "random"
OBSTACLES = "random"
WINDOW = ["set_up", "launch_pending", "before_reset", "staged_params", "staged_human_params", "module_before", "tm_robots", "module_plan", "forbid 0", "tm_obstacles", "respawn", "module_after", "after_reset"]


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("arena_runtime.sim.dummy_simulator")


class _Robots:
    managers: dict = {}

    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def set_up(self) -> None:
        self.log.append("set_up")

    async def launch_pending(self) -> None:
        self.log.append("launch_pending")


class _TmRobots:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def reset(self) -> object:
        from arena_runtime.env.task_layer import ResetPlan

        self.log.append("tm_robots")
        return ResetPlan()

    async def tick(self, ended: dict, assigned: list) -> None:
        self.log.append(f"tick {sorted(ended)} {sorted(assigned)}")


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

    def on_reset_plan(self, plan: object) -> None:
        del plan
        self.log.append("module_plan")

    def after_reset(self) -> None:
        self.log.append("module_after")


class _OverridingModule(_Module):
    def on_reset_plan(self, plan: object) -> None:
        from arena_runtime.env.task_layer import RobotReset

        super().on_reset_plan(plan)
        plan.robots["r0"] = RobotReset(start_pose="module_start")


def _on_task(error: Exception | None, body: Callable[..., Awaitable[None]], module: type[_Module] = _Module) -> list[str]:
    import rclpy
    import rclpy.executors
    from std_msgs.msg import Empty
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator

    from arena_runtime.env.constants import Constants
    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.humans import HumanSimulatorRegistry
    from arena_runtime.env.episode_node import EpisodeNode
    from task_generator.modes import Modes
    from task_generator.tasks.task import Task

    log: list[str] = []

    class _Node(ServiceNamespace, AsyncNode):
        conf = SimpleNamespace(General=SimpleNamespace(RNG=EpisodeRng()))
        episode_id = 0
        _reset_episode = EpisodeNode._reset_episode

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
            node._environment_manager = environment
            node._robots_manager = _Robots(log)
            node._world_manager = SimpleNamespace(forbid=lambda zones: log.append(f"forbid {len(zones)}"))
            task = Task(node=node, ctx=SimpleNamespace(robots={}), params=SimpleNamespace(TM_CONFIG=SimpleNamespace(value=""), TM_ROBOTS=SimpleNamespace(value=Modes.TM_Robots(ROBOTS)), TM_OBSTACLES=SimpleNamespace(value=Modes.TM_Obstacles(OBSTACLES))))
            task._Task__param_tm_robots = Modes.TM_Robots(ROBOTS)
            task._Task__param_tm_obstacles = Modes.TM_Obstacles(OBSTACLES)
            task._Task__tm_robots = _TmRobots(log)
            task._Task__tm_obstacles = _TmObstacles(log, error)
            task._Task__modules = [module(log)]
            node._task = task
            node._pub_reset_start = node.create_publisher(Empty, "reset_start", 1)
            node._pub_reset_end = node.create_publisher(Empty, "reset_end", 1)
            await body(node)
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
    from arena_runtime.env.phase_clock import PhaseClock
    from arena_runtime.env.task_layer import EpisodeRecord

    async def body(node) -> None:
        await node._reset_episode(EpisodeRecord(seed=3), PhaseClock())

    assert _on_task(None, body) == WINDOW


def test_a_failing_mode_still_closes_the_paused_window(rclpy_context) -> None:
    from arena_runtime.env.phase_clock import PhaseClock
    from arena_runtime.env.task_layer import EpisodeRecord

    async def body(node) -> None:
        with pytest.raises(RuntimeError, match="no obstacles today"):
            await node._reset_episode(EpisodeRecord(seed=3), PhaseClock())

    assert _on_task(RuntimeError("no obstacles today"), body) == [*WINDOW[: WINDOW.index("tm_obstacles") + 1], "after_reset"]


def test_a_tick_reaches_the_robot_mode_with_the_phases_that_ended_and_the_robots_assigned(rclpy_context) -> None:
    async def body(node) -> None:
        await node._task.tick({"r1": object(), "r0": object()}, ["r2"])

    assert _on_task(None, body)[-1] == "tick ['r0', 'r1'] ['r2']"


def test_a_module_changes_the_plan_after_the_robot_mode_and_before_the_env_applies_it(rclpy_context) -> None:
    from arena_runtime.env.task_layer import EpisodeRecord

    plans: list[object] = []

    async def body(node) -> None:
        plans.append(await node._task.reset_robots(EpisodeRecord()))

    log = _on_task(None, body, _OverridingModule)
    assert log[log.index("tm_robots") + 1] == "module_plan"
    assert plans[0].robots["r0"].start_pose == "module_start"
