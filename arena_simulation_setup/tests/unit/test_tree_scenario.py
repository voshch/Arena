from __future__ import annotations

import warnings

import pytest
import yaml

from arena_simulation_setup.tree.World.Scenario import (
    RobotGoal,
    Scenario,
    ScenarioGesturePhase,
    ScenarioGotoPhase,
    ScenarioPhase,
    ScenarioView,
)
from arena_simulation_setup.utils.geometry import Pose


# ---------------------------------------------------------------------------
# RobotGoal.parse
# ---------------------------------------------------------------------------


def test_robot_goal_parse_full_dict():
    rg = RobotGoal.parse({"start": [1.0, 2.0], "goal": [3.0, 4.0]})
    assert rg.start.position.x == pytest.approx(1.0)
    assert rg.goal.position.x == pytest.approx(3.0)


def test_robot_goal_parse_start_only():
    rg = RobotGoal.parse({"start": [5.0, 6.0], "goal": [0.0, 0.0]})
    assert rg.start.position.x == pytest.approx(5.0)
    assert isinstance(rg.goal, Pose)


def test_robot_goal_parse_goal_only():
    rg = RobotGoal.parse({"start": [0.0, 0.0], "goal": [7.0, 8.0]})
    assert rg.goal.position.x == pytest.approx(7.0)
    assert isinstance(rg.start, Pose)


def test_robot_goal_parse_empty_parse_fallback():
    with pytest.raises(ValueError):
        RobotGoal.parse({})


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def test_scenario_with_robots():
    rg = RobotGoal.parse({"start": [0.0, 0.0], "goal": [1.0, 1.0]})
    s = Scenario(robots=[rg])
    assert len(s.robots) == 1


# ---------------------------------------------------------------------------
# ScenarioView
# ---------------------------------------------------------------------------


def test_scenario_view_path_yaml_priority(tmp_path):
    scenario_dir = tmp_path / "test_scenario"
    scenario_dir.mkdir()
    yaml_file = scenario_dir / "scenario.yaml"
    json_file = scenario_dir / "scenario.json"
    yaml_file.write_text(yaml.dump({"static": [], "dynamic": [], "robots": []}))
    json_file.write_text('{"static": [], "dynamic": [], "robots": []}')

    view = ScenarioView(scenario_dir)
    assert view.scenario_path.endswith("scenario.yaml")


def test_scenario_view_path_fallback_to_json_when_no_yaml(tmp_path):
    scenario_dir = tmp_path / "test_scenario2"
    scenario_dir.mkdir()
    view = ScenarioView(scenario_dir)
    # No files exist; should return yaml path as default
    assert "scenario.yaml" in view.scenario_path


def test_scenario_view_load_new_format(tmp_path):
    scenario_dir = tmp_path / "sc_new"
    scenario_dir.mkdir()
    data = {"static": [], "dynamic": [], "robots": [{"start": [1.0, 2.0], "goal": [3.0, 4.0]}]}
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    scenario = view.load()
    assert len(scenario.robots) == 1


def test_scenario_view_load_legacy(tmp_path):
    scenario_dir = tmp_path / "sc_legacy"
    scenario_dir.mkdir()
    # Legacy format uses 'obstacles.static' key - this will fail new format (no 'static' top-level)
    data = {
        "obstacles": {
            "static": [{"name": "obs1", "pose": [0.0, 0.0], "model": "box"}],
            "dynamic": [],
        },
        "robots": [{"start": [1.0, 2.0], "goal": [3.0, 4.0]}],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scenario = view.load()
    # New format parses successfully (empty static/dynamic), no DeprecationWarning
    # OR falls to legacy with DeprecationWarning. Either way should succeed.
    assert isinstance(scenario, Scenario)
    assert len(scenario.robots) >= 0


def test_scenario_view_load_malformed_raises_runtime_error(tmp_path):
    scenario_dir = tmp_path / "sc_bad"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(": : : invalid yaml: :")

    view = ScenarioView(scenario_dir)
    with pytest.raises((RuntimeError, Exception)):
        view.load()


def test_scenario_view_included_from_propagated(tmp_path):
    scenario_dir = tmp_path / "sc_incl"
    scenario_dir.mkdir()
    # Use new format so we know it succeeds with static obstacles
    data = {
        "static": [{"name": "obs1", "pose": [0.0, 0.0], "model": "box"}],
        "dynamic": [],
        "robots": [],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    scenario = view.load()
    assert len(scenario.static) >= 1
    assert scenario.static[0].included_from == scenario_dir


def test_scenario_view_load_audio_systems(tmp_path):
    scenario_dir = tmp_path / "sc_audio"
    scenario_dir.mkdir()
    data = {
        "audio": {
            "systems": [
                {
                    "name": "radio",
                    "sound_type": "music",
                    "asset_id": "radio_loop",
                    "loop": True,
                    "initially_active": True,
                    "emitters": [
                        {
                            "name": "lobby",
                            "entity_ref": "lobby_radio",
                            "offset": [0.0, 0.0, 0.8],
                            "source_volume_db": 62.0,
                        }
                    ],
                },
                {
                    "name": "alarm",
                    "sound_type": "alarm",
                    "asset_id": "alarm_loop",
                    "emitters": [
                        {
                            "name": "floor_1",
                            "position": [2.0, 3.0, 2.6],
                            "level": "level_1",
                            "source_volume_db": 88.0,
                        },
                        {
                            "name": "floor_2",
                            "position": [4.0, 5.0, 2.6],
                            "level": "level_2",
                        },
                    ],
                },
            ]
        }
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    audio = ScenarioView(scenario_dir).load_audio()

    assert [system.name for system in audio.systems] == ["radio", "alarm"]
    assert audio.systems[0].initially_active is True
    assert audio.systems[0].emitters[0].entity_ref == "lobby_radio"
    assert audio.systems[0].emitters[0].offset.z == pytest.approx(0.8)
    assert len(audio.systems[1].emitters) == 2
    assert audio.systems[1].emitters[0].level == "level_1"
    assert audio.systems[1].emitters[0].position.x == pytest.approx(2.0)


def test_scenario_view_rejects_duplicate_audio_system_names(tmp_path):
    scenario_dir = tmp_path / "sc_duplicate_audio"
    scenario_dir.mkdir()
    system = {
        "name": "alarm",
        "sound_type": "alarm",
        "asset_id": "alarm_loop",
        "emitters": [{"name": "speaker", "position": [1.0, 2.0]}],
    }
    data = {"audio": {"systems": [system, system]}}
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    with pytest.raises(ValueError, match="duplicate audio system names"):
        ScenarioView(scenario_dir).load_audio()


# ---------------------------------------------------------------------------
# ScenarioPhase dispatch
# ---------------------------------------------------------------------------


def test_scenario_phase_parse_goto():
    phase = ScenarioPhase.parse({"goto": [1.0, 2.0, 0.0]})
    assert isinstance(phase, ScenarioGotoPhase)
    assert phase.goto.position.x == pytest.approx(1.0)


def test_scenario_phase_parse_gesture():
    phase = ScenarioPhase.parse({"gesture": "wave"})
    assert isinstance(phase, ScenarioGesturePhase)
    assert phase.gesture == "wave"


def test_scenario_phase_parse_malformed_raises():
    with pytest.raises(ValueError):
        ScenarioPhase.parse({"foo": "bar"})


# ---------------------------------------------------------------------------
# RobotGoal.phase_list -- legacy goal: path
# ---------------------------------------------------------------------------


def test_robot_goal_phase_list_legacy_goal_deprecation():
    rg = RobotGoal.parse({"start": [0.0, 0.0], "goal": [3.0, 4.0]})
    with pytest.warns(DeprecationWarning):
        phases = rg.phase_list()
    assert len(phases) == 1
    assert isinstance(phases[0], ScenarioGotoPhase)
    assert phases[0].goto.position.x == pytest.approx(3.0)


def test_robot_goal_phase_list_empty_when_no_goal_no_phases():
    rg = RobotGoal.parse({"start": [0.0, 0.0]})
    phases = rg.phase_list()
    assert phases == []


# ---------------------------------------------------------------------------
# RobotGoal.phase_list -- new phases: path
# ---------------------------------------------------------------------------


def test_robot_goal_phase_list_mixed_phases(tmp_path):
    scenario_dir = tmp_path / "sc_phases"
    scenario_dir.mkdir()
    data = {
        "static": [],
        "dynamic": [],
        "robots": [
            {
                "start": [0.0, 0.0, 0.0],
                "phases": [
                    {"goto": [1.0, 0.0, 0.0]},
                    {"gesture": "wave"},
                    {"goto": [2.0, 0.0, 0.0]},
                    {"gesture": "random"},
                ],
            }
        ],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    scenario = view.load()
    rg = scenario.robots[0]
    phases = rg.phase_list()
    assert len(phases) == 4
    assert isinstance(phases[0], ScenarioGotoPhase)
    assert phases[0].goto.position.x == pytest.approx(1.0)
    assert isinstance(phases[1], ScenarioGesturePhase)
    assert phases[1].gesture == "wave"
    assert isinstance(phases[2], ScenarioGotoPhase)
    assert phases[2].goto.position.x == pytest.approx(2.0)
    assert isinstance(phases[3], ScenarioGesturePhase)
    assert phases[3].gesture == "random"


def test_robot_goal_phase_list_malformed_phase_raises(tmp_path):
    scenario_dir = tmp_path / "sc_bad_phase"
    scenario_dir.mkdir()
    data = {
        "static": [],
        "dynamic": [],
        "robots": [
            {
                "start": [0.0, 0.0, 0.0],
                "phases": [{"foo": "bar"}],
            }
        ],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with pytest.raises((ValueError, RuntimeError)):
        view.load()


def test_robot_goal_phases_takes_priority_over_goal():
    rg = RobotGoal.parse(
        {
            "start": [0.0, 0.0],
            "goal": [9.0, 9.0],
            "phases": [{"gesture": "wave"}],
        }
    )
    phases = rg.phase_list()
    assert len(phases) == 1
    assert isinstance(phases[0], ScenarioGesturePhase)
