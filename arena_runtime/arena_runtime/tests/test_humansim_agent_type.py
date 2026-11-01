"""Agent type lookup shared by the scripted and the flow spawn paths of the arena_humansim bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arena_humansim.core.agents import sample_agent_type

from arena_runtime.humans.arena_humansim.agent import ArenaHumanDynamicObstacle, agent_type_def, flow_agent_template


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


def test_sample_params_keeps_the_type_sample_without_overrides() -> None:
    definition = agent_type_def("wheelchair_manual")
    assert definition is not None
    expected = sample_agent_type(definition, np.random.default_rng(3))
    params = ArenaHumanDynamicObstacle(agent_type="wheelchair_manual").sample_params(np.random.default_rng(3))
    assert params is not None
    assert (params.desired_velocity, params.agent_radius) == (expected.desired_velocity, expected.agent_radius)


def test_agent_block_without_speed_or_radius_leaves_both_unset() -> None:
    from arena_runtime.shared import DynamicObstacle

    for agent in ("adult", {"agent_type": "adult"}):
        human = DynamicObstacle.parse({"name": "walker", "model": "arenian", "pose": [0.0, 0.0, 0.0], "agent": agent})
        parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(human)
        assert parsed is not None
        assert (parsed.desired_velocity_min, parsed.desired_velocity_max, parsed.agent_radius) == (None, None, None)


def test_scenario_human_agent_block_and_waypoint_mode_drive_the_spawn(tmp_path: Path) -> None:
    from arena_humansim_msgs.msg import Waypoints as WaypointsMsg

    from arena_runtime.shared import DynamicObstacle

    human = DynamicObstacle.parse(
        {
            "name": "nurse_1",
            "model": "arenian",
            "pose": [1.0, 2.0, 0.0],
            "waypoints": [[3.0, 4.0]],
            "waypoint_mode": "once",
            "agent": {"agent_type": "./doctor.yaml", "desired_velocity": {"min": 0.6, "max": 0.9}, "radius": 0.3},
        }
    )
    human.included_from = tmp_path
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(human)
    assert parsed is not None
    assert parsed.agent_type == str((tmp_path / "doctor.yaml").resolve())
    assert (parsed.desired_velocity_min, parsed.desired_velocity_max, parsed.agent_radius) == (0.6, 0.9, 0.3)
    assert parsed.waypoint_mode == WaypointsMsg.MODE_ONCE
    assert [(waypoint.x, waypoint.y) for waypoint in parsed._waypoints] == [(3.0, 4.0)]


def test_scenario_human_without_agent_block_is_not_an_arena_humansim_agent() -> None:
    from arena_runtime.shared import DynamicObstacle

    human = DynamicObstacle.parse({"name": "ghost", "model": "arenian", "pose": [0.0, 0.0, 0.0]})
    assert ArenaHumanDynamicObstacle.from_dynamic_obstacle(human) is None


def test_flow_agent_template_leaves_speed_and_radius_to_the_type_unless_set() -> None:
    unset = flow_agent_template({"agent_type": "adult"}, None)
    assert (unset.desired_velocity_min, unset.desired_velocity_max, unset.agent_radius) == (0.0, 0.0, 0.0)
    pinned = flow_agent_template({"agent_type": "adult", "desired_velocity": {"min": 0.9, "max": 1.2}, "agent_radius": 0.3}, None)
    assert (pinned.desired_velocity_min, pinned.desired_velocity_max, pinned.agent_radius) == (0.9, 1.2, 0.3)
    scalar = flow_agent_template({"desired_velocity": 1.1}, None)
    assert (scalar.desired_velocity_min, scalar.desired_velocity_max, scalar.agent_type) == (1.1, 1.1, "adult")
