from __future__ import annotations

from pathlib import Path

import attrs
import pytest
import yaml

try:
    from arena_simulation_setup.shared.task import TaskPhase
    from arena_simulation_setup.tree.World.World import LevelDescription
    from arena_simulation_setup.utils.cattrs import converter

    from task_generator.manager.realizer import Realizer
    from task_generator.shared import Orientation, Pose, Position
    from task_generator.tasks.robots.adapters.mobile.vla import localized_instruction
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

HOSPITAL = Path(__file__).parents[3] / "arena_simulation_setup" / "worlds" / "hospital_1"
TO_PHARMACY = "Walk through the main central hallway, then take the first door on your left into the pharmacy. Walk about 4 meters into the pharmacy and stop next to the pharmacy shelf."


def _level() -> LevelDescription:
    return converter.structure(yaml.safe_load((HOSPITAL / "0" / "world.yaml").read_text()), LevelDescription)


def _env_frame(realizer: Realizer) -> tuple[TaskPhase, Pose]:
    """The landmark_pharmacy goto and start pose, realized into an env 5 m off the map origin."""
    robot = yaml.safe_load((HOSPITAL / "scenarios" / "landmark_pharmacy" / "scenario.yaml").read_text())["robots"][0]
    phase = TaskPhase.parse(robot["phases"][0])
    x, y, yaw = robot["start"]
    start = Pose(position=Position(x, y), orientation=Orientation.from_yaw(yaw))
    return attrs.evolve(phase, pose=realizer.realize(phase.pose)), realizer.realize(start)


@pytest.fixture()
def realizer() -> Realizer:
    return Realizer(Realizer._Configuration(x=5.0, y=5.0, prefix="env_1"))


def test_route_is_worded_in_the_world_frame_from_env_frame_poses(realizer: Realizer) -> None:
    phase, here = _env_frame(realizer)
    assert localized_instruction(phase, here, realizer.ezilear, _level(), "route") == {"source": "route", "text": TO_PHARMACY}


def test_env_offset_left_in_place_loses_the_route(realizer: Realizer) -> None:
    phase, here = _env_frame(realizer)
    told = localized_instruction(phase, here, lambda pose: pose, _level(), "route")
    assert told["text"].startswith("Turn left and walk through the laboratory")


def test_unknown_robot_pose_names_the_target(realizer: Realizer) -> None:
    phase, _ = _env_frame(realizer)
    assert localized_instruction(phase, None, realizer.ezilear, _level(), "route") == {"source": "target", "text": "Go to the pharmacy."}


def test_launch_instruction_stands_in_for_missing_phase_text(realizer: Realizer) -> None:
    phase, here = _env_frame(realizer)
    told = localized_instruction(phase, here, realizer.ezilear, _level(), "route", "Find the pharmacy.")
    assert told == {"source": "authored", "text": "Find the pharmacy."}
