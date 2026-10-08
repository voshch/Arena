"""Agent type lookup shared by the scripted and the flow spawn paths of the arena_humansim bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from task_generator.simulators.human.arena_humansim import ArenaHumanDynamicObstacle, agent_type_def


def test_agent_type_def_resolves_builtin_names_and_yaml_paths(tmp_path: Path) -> None:
    chair = agent_type_def("wheelchair_manual")
    assert chair is not None and chair.assets == ("human::mobility::wheelchair",)
    adult = agent_type_def("adult")
    assert adult is not None and adult.assets == ()
    local = tmp_path / "scooter.yaml"
    local.write_text("name: scooter\nextends: adult\nassets: [human::mobility::scooter]\n")
    scooter = agent_type_def(str(local))
    assert scooter is not None and scooter.assets == ("human::mobility::scooter",)
    assert scooter.perception == adult.perception
    assert agent_type_def("ghost") is None
    assert agent_type_def(str(tmp_path / "missing.yaml")) is None


def test_sample_params_overrides_speed_and_radius_of_the_resolved_type() -> None:
    obstacle = ArenaHumanDynamicObstacle(agent_type="wheelchair_manual", desired_velocity_min=0.9, desired_velocity_max=0.9, agent_radius=0.325)
    params = obstacle.sample_params(np.random.default_rng(3))
    assert params is not None
    assert (params.desired_velocity, params.agent_radius) == (0.9, 0.325)
    assert params.locomotion.active and params.interaction_class == "wheelchair"
    assert ArenaHumanDynamicObstacle(agent_type="ghost").sample_params(np.random.default_rng(3)) is None
