from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import attrs
import pytest

try:
    import rclpy.logging
    from arena_simulation_setup.shared.task import TaskPhase
    from arena_simulation_setup.tree.World import WorldView
    from arena_simulation_setup.tree.World.Scenario import ScenarioView

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.env.robot_manager.task_runner import TaskRunner
    from arena_runtime.shared import Pose
    from arena_runtime.env.adapters.mobile.vla import VlaAdapter
    from arena_runtime.env.request import kind_of
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

HOSPITAL = Path(__file__).parents[4] / "arena_simulation_setup" / "worlds" / "hospital_1"
TO_PHARMACY = "Walk through the main central hallway, then take the first door on your left into the pharmacy. Walk about 4 meters into the pharmacy and stop next to the pharmacy shelf."


class _EdgeNode:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def request_reset(self, episode_id: str, initial_state: dict) -> None:
        self.payloads.append(initial_state)


def _scene() -> tuple[TaskPhase, Pose, SimpleNamespace]:
    """The landmark_pharmacy goto and start pose in an env 5 m off the map origin, plus a robot whose pose is not known yet."""
    realizer = Realizer(Realizer._Configuration(x=5.0, y=5.0, prefix="env_1"))
    robot = ScenarioView(HOSPITAL / "scenarios" / "landmark_pharmacy").read()["robots"][0]
    phase = TaskPhase.parse(robot["phases"][0])
    world = WorldView(HOSPITAL).load()
    rm = SimpleNamespace(pose=None, _environment_manager=realizer, node=SimpleNamespace(_world_manager=SimpleNamespace(world=world, world_compacted=lambda: world.levels["0"])))
    return attrs.evolve(phase, pose=realizer.realize(phase.pose)), realizer.realize(Pose.parse(robot["start"])), rm


def _adapter(rm: SimpleNamespace, phase: TaskPhase) -> VlaAdapter:
    adapter = VlaAdapter.__new__(VlaAdapter)
    adapter.rm = rm
    adapter._goal = "instruction"
    adapter._fallback_instruction = ""
    adapter._wording = "route"
    adapter._revealed = frozenset({"instruction"})
    adapter._instruction = None
    adapter._current_phase = phase
    adapter._edge_node = _EdgeNode()
    adapter._clients = {kind_of(phase): SimpleNamespace(is_done=lambda: True)}
    return adapter


def _manager(adapter: VlaAdapter, phase: TaskPhase) -> tuple[RobotManager, list[RobotManager]]:
    published: list[RobotManager] = []
    manager = RobotManager.__new__(RobotManager)
    manager._adapter_instances = [adapter]
    manager._adapters = {kind_of(phase): adapter}
    manager._runner = TaskRunner("jackal")
    manager._runner.begin_episode()
    manager._runner.submit([phase])
    manager._placed = False
    manager._NodeInterface__node = SimpleNamespace(on_task_submitted=published.append, get_logger=lambda: rclpy.logging.get_logger("test_robot_reset_instruction"))
    return manager, published


def test_reset_rewords_the_instruction_from_the_landed_pose() -> None:
    phase, start, rm = _scene()
    adapter = _adapter(rm, phase)
    manager, published = _manager(adapter, phase)
    state = manager._runner.active_state
    state.dispatched = True
    adapter._initial_state(phase)
    manager._note_goal_inputs(state)
    assert state.instruction == {"source": "target", "text": "Go to the pharmacy."}

    rm.pose = start
    assert asyncio.run(manager.reset(SimpleNamespace(start_pose=None))) == {"vla": None}
    assert state.instruction == {"source": "route", "text": TO_PHARMACY}
    assert manager._runner.goal_inputs == ("instruction",)
    assert adapter._edge_node.payloads == [{"instruction": TO_PHARMACY}]
    assert published == [manager, manager]
    assert manager._runner.serialize()["instructions"] == [{"source": "route", "text": TO_PHARMACY}]


def test_reset_before_dispatch_leaves_the_record_alone() -> None:
    phase, start, rm = _scene()
    rm.pose = start
    adapter = _adapter(rm, phase)
    manager, published = _manager(adapter, phase)
    asyncio.run(manager.reset(SimpleNamespace(start_pose=None)))
    assert manager._runner.active_state.instruction is None
    assert manager._runner.goal_inputs == ()
    assert published == []
