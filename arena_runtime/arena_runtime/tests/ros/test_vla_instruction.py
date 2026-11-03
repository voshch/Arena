from __future__ import annotations

from pathlib import Path

import attrs
import pytest

try:
    from arena_simulation_setup.kg import Scene
    from arena_simulation_setup.shared.task import TaskPhase
    from arena_simulation_setup.tree.World import Level, WorldView
    from arena_simulation_setup.tree.World.Scenario import ScenarioView

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Pose
    from arena_runtime.env.adapters.mobile.vla import localized_instruction
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

HOSPITAL = Path(__file__).parents[4] / "arena_simulation_setup" / "worlds" / "hospital_1"
TO_PHARMACY = "Walk through the main central hallway, then take the first door on your left into the pharmacy. Walk about 4 meters into the pharmacy and stop next to the pharmacy shelf."


def _world() -> tuple[Level, Scene]:
    world = WorldView(HOSPITAL).load()
    assert world.kg is not None
    return world.levels["0"], world.kg


def _env_frame(realizer: Realizer) -> tuple[TaskPhase, Pose]:
    """The landmark_pharmacy goto and start pose, realized into an env 5 m off the map origin."""
    robot = ScenarioView(HOSPITAL / "scenarios" / "landmark_pharmacy").read()["robots"][0]
    phase = TaskPhase.parse(robot["phases"][0])
    return attrs.evolve(phase, pose=realizer.realize(phase.pose)), realizer.realize(Pose.parse(robot["start"]))


@pytest.fixture()
def realizer() -> Realizer:
    return Realizer(Realizer._Configuration(x=5.0, y=5.0, prefix="env_1"))


def test_route_is_worded_in_the_world_frame_from_env_frame_poses(realizer: Realizer) -> None:
    phase, here = _env_frame(realizer)
    assert localized_instruction(phase, here, realizer.ezilear, *_world(), "route") == {"source": "route", "text": TO_PHARMACY}


def test_env_offset_left_in_place_loses_the_route(realizer: Realizer) -> None:
    phase, here = _env_frame(realizer)
    told = localized_instruction(phase, here, lambda pose: pose, *_world(), "route")
    assert told["text"].startswith("Turn left and walk through the laboratory")


def test_unknown_robot_pose_names_the_target(realizer: Realizer) -> None:
    phase, _ = _env_frame(realizer)
    assert localized_instruction(phase, None, realizer.ezilear, *_world(), "route") == {"source": "target", "text": "Go to the pharmacy."}


def test_launch_instruction_stands_in_for_missing_phase_text(realizer: Realizer) -> None:
    phase, here = _env_frame(realizer)
    told = localized_instruction(phase, here, realizer.ezilear, *_world(), "route", "Find the pharmacy.")
    assert told == {"source": "authored", "text": "Find the pharmacy."}
