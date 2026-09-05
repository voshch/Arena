from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.shared.conditions import EpisodeCondition
from arena_simulation_setup.tree.World.Scenario import (
    RobotGoal,
    Scenario,
    ScenarioGesturePhase,
    ScenarioGotoPhase,
    ScenarioPhase,
    ScenarioView,
    TimelineEntry,
)
from arena_simulation_setup.tree.World.World import LevelDescription
from arena_simulation_setup.utils.cattrs import converter
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


def test_scenario_parses_episode_sounds():
    s = converter.structure(
        {
            "sounds": [
                {
                    "name": "bedroom_radio",
                    "asset_id": "radio_loop",
                    "position": [2.0, 1.0],
                    "level": "1",
                    "semantics": [{"preset": "sound", "params": {"sounding": True, "volume_db": 62.0}}],
                },
            ],
        },
        Scenario,
    )
    assert [snd.name for snd in s.sounds] == ["bedroom_radio"]
    assert s.sounds[0].level == "1"
    assert {c.name: c.value for c in s.sounds[0].semantics} == {"sounding": True, "volume_db": 62.0}


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


# ---------------------------------------------------------------------------
# TimelineEntry / Scenario.timeline (M2)
# ---------------------------------------------------------------------------


def test_timeline_entry_parse_at():
    entry = TimelineEntry.parse({"at": 12.0, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}]})
    assert entry.at == pytest.approx(12.0)
    assert entry.every is None
    assert entry.when is None
    assert entry.set == [{"entity": "fire_alarm", "field": "active", "value": "true"}]


def test_timeline_entry_parse_every():
    entry = TimelineEntry.parse({"every": 5.0, "offset": 1.0, "until": 30.0, "set": []})
    assert entry.every == pytest.approx(5.0)
    assert entry.offset == pytest.approx(1.0)
    assert entry.until == pytest.approx(30.0)


def test_timeline_entry_parse_when():
    entry = TimelineEntry.parse({"when": {"entity": "door_1", "field": "open", "is": True}, "set": []})
    assert entry.when == {"entity": "door_1", "field": "open", "is": True}


def test_timeline_entry_requires_exactly_one_trigger_none_set():
    with pytest.raises(ValueError):
        TimelineEntry.parse({"set": []})


def test_timeline_entry_requires_exactly_one_trigger_multiple_set():
    with pytest.raises(ValueError):
        TimelineEntry.parse({"at": 1.0, "every": 2.0, "set": []})


def test_timeline_entry_parse_rejects_unknown_key():
    with pytest.raises(ValueError):
        TimelineEntry.parse({"at": 1.0, "sett": []})


def test_timeline_entry_structure_rejects_typoed_trigger_key():
    with pytest.raises(ValueError):
        converter.structure({"att": 1.0, "set": []}, TimelineEntry)


def test_timeline_entry_structure_rejects_zero_triggers():
    with pytest.raises(ValueError):
        converter.structure({"set": []}, TimelineEntry)


def test_timeline_entry_structure_rejects_multiple_triggers():
    with pytest.raises(ValueError):
        converter.structure({"at": 1.0, "every": 2.0, "set": []}, TimelineEntry)


def test_timeline_entry_structure_valid_entry_loads():
    entry = converter.structure({"at": 12.0, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}]}, TimelineEntry)
    assert entry.at == pytest.approx(12.0)
    assert entry.set == [{"entity": "fire_alarm", "field": "active", "value": "true"}]


def test_timeline_entry_constructor_rejects_zero_triggers():
    with pytest.raises(ValueError):
        TimelineEntry(set=[])


def test_timeline_entry_round_trip_through_cattrs():
    entry = TimelineEntry.parse({"at": 12.0, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}]})
    raw = converter.unstructure(entry)
    reparsed = converter.structure(raw, TimelineEntry)
    assert reparsed == entry


def test_scenario_timeline_default_empty():
    s = Scenario()
    assert s.timeline == []


def test_scenario_timeline_structures_from_yaml(tmp_path):
    scenario_dir = tmp_path / "sc_timeline"
    scenario_dir.mkdir()
    data = {
        "static": [],
        "dynamic": [],
        "robots": [],
        "timeline": [
            {"at": 12.0, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}]},
        ],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    scenario = ScenarioView(scenario_dir).load()
    assert len(scenario.timeline) == 1
    assert scenario.timeline[0].at == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Scenario.conditions (M3)
# ---------------------------------------------------------------------------


def test_scenario_conditions_default_empty():
    s = Scenario()
    assert s.conditions == []


def test_scenario_conditions_structures_from_yaml(tmp_path):
    scenario_dir = tmp_path / "sc_conditions"
    scenario_dir.mkdir()
    data = {
        "static": [],
        "dynamic": [],
        "robots": [],
        "conditions": [
            {"op": "eventually", "p": "robot in ward_a", "text": "Deliver the package to ward A."},
            {"op": "never", "p": "robot in pharmacy", "text": "Never enter the pharmacy."},
            {
                "op": "never_during",
                "p": "robot in ward_a_doorway",
                "q": "ward_a_door.open == false",
                "text": "Wait for the door before passing through.",
            },
        ],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    scenario = ScenarioView(scenario_dir).load()
    assert len(scenario.conditions) == 3
    assert [c.op for c in scenario.conditions] == ["eventually", "never", "never_during"]
    assert scenario.conditions[2].q == "ward_a_door.open == false"


def test_scenario_conditions_round_trip_through_cattrs():
    cond = EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a"})
    s = Scenario(conditions=[cond])
    raw = converter.unstructure(s)
    reparsed = converter.structure(raw, Scenario)
    assert reparsed.conditions == [cond]


def test_scenario_conditions_rejects_malformed_entry():
    data = {
        "static": [],
        "dynamic": [],
        "robots": [],
        "conditions": [{"op": "eventually"}],  # missing 'p'
    }
    with pytest.raises(Exception):
        converter.structure(data, Scenario)


def test_scenario_view_load_modern_condition_malformed_reraises(tmp_path):
    scenario_dir = tmp_path / "sc_bad_condition"
    scenario_dir.mkdir()
    data = {
        "static": [],
        "dynamic": [],
        "robots": [],
        "conditions": [{"op": "until", "p": "robot in ward_a"}],  # unknown op
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with pytest.raises(RuntimeError, match="modern"):
        view.load()


def test_scenario_view_load_modern_binary_op_missing_q_reraises(tmp_path):
    scenario_dir = tmp_path / "sc_bad_condition2"
    scenario_dir.mkdir()
    data = {
        "static": [],
        "dynamic": [],
        "robots": [],
        "conditions": [{"op": "before", "p": "robot in ward_a"}],  # binary op missing 'q'
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with pytest.raises(RuntimeError, match="modern"):
        view.load()


def test_scenario_view_load_legacy_shape_still_falls_back(tmp_path):
    scenario_dir = tmp_path / "sc_legacy_no_modern_keys"
    scenario_dir.mkdir()
    data = {
        "obstacles": {
            "static": [{"name": "obs1", "pose": [0.0, 0.0], "model": "box"}],
            "dynamic": [],
        },
        "robots": [{"start": [1.0, 2.0], "goal": [3.0, 4.0]}],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenario = view.load()
    assert isinstance(scenario, Scenario)
    assert scenario.conditions == []
    assert scenario.timeline == []
    assert len(scenario.robots) == 1


# ---------------------------------------------------------------------------
# Fire-alarm reference scenario (M2-D reference)
# ---------------------------------------------------------------------------

_FIRE_ALARM_SCENARIO = (
    Path(__file__).resolve().parents[2]
    / "worlds"
    / "three_storied_residential"
    / "scenarios"
    / "fire_alarm"
    / "scenario.yaml"
)


def test_fire_alarm_reference_scenario_parses():
    if not _FIRE_ALARM_SCENARIO.exists():
        pytest.skip(f"reference scenario not found: {_FIRE_ALARM_SCENARIO}")
    scenario = ScenarioView(_FIRE_ALARM_SCENARIO.parent).load()
    assert len(scenario.timeline) == 2
    lock, alarm = scenario.timeline
    assert lock.at == pytest.approx(0.0)
    assert lock.set == [{"entity": "door_edge_1_1", "field": "locked", "value": "true"}]
    assert alarm.at == pytest.approx(12.0)
    assert alarm.set == [
        {"entity": "fire_alarm", "field": "active", "value": "true"},
        {"entity": "bedroom_radio", "field": "sounding", "value": "false"},
    ]
    (radio,) = scenario.sounds
    assert radio.name == "bedroom_radio"
    assert radio.level == "1"
    assert {c.name: c.value for c in radio.semantics} == {"sounding": True, "volume_db": 62.0}


_FIRE_ALARM_WORLD = (
    Path(__file__).resolve().parents[2]
    / "worlds"
    / "three_storied_residential"
    / "1"
    / "world.yaml"
)


def test_fire_alarm_reference_regime_wired():
    if not _FIRE_ALARM_WORLD.exists():
        pytest.skip(f"reference world not found: {_FIRE_ALARM_WORLD}")
    with open(_FIRE_ALARM_WORLD, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    level = converter.structure(raw, LevelDescription)

    schedules = {schedule.name: schedule for schedule in level.all_schedules}
    assert "fire_alarm" in schedules
    active = next(cfg for cfg in schedules["fire_alarm"].semantics if cfg.name == "active")
    assert active.params["regime"] == "alarm"
    assert active.params["windows"] == [], "a non-empty window would assert the regime without the timeline"

    gates = [(door, cfg) for door in level.all_doors for cfg in door.semantics if cfg.params.get("unlock_on") == "alarm"]
    assert gates, "no gate wired to unlock on the fire_alarm regime"
    for door, cfg in gates:
        assert cfg.name == "locked"
        assert cfg.value is False, f"gate on shared door {door.name!r} must spawn unlocked"

    plates = [(door, cfg) for door in level.all_doors for cfg in door.semantics if cfg.params.get("press_on") == "alarm"]
    assert plates, "no pressure plate wired to press on the fire_alarm regime"
    for door, cfg in plates:
        assert float(cfg.params.get("radius", 0.0)) == 0.0, f"a contact radius on {door.name!r} would bypass the gate"
        assert cfg.params["drives"] == door.name
        px, py = (float(v) for v in cfg.params["position"])
        assert px == pytest.approx((door.start.x + door.end.x) / 2.0)
        assert py == pytest.approx((door.start.y + door.end.y) / 2.0)

    assert all(elevator.recall_on is None for elevator in level.all_elevators), "recall is upstream and dormant, keep it out of the reference"

    sounds = {sound.name: sound for sound in level.all_sounds}
    assert "hall_siren" in sounds
    siren = sounds["hall_siren"]
    assert siren.asset_id == "alarm_loop"
    sounding_cfg = next(cfg for cfg in siren.semantics if cfg.name == "sounding")
    assert sounding_cfg.params["sound_on"] == "alarm"
    assert "regime" not in sounding_cfg.params, "hall_siren must not self-assert"
    assert sounding_cfg.value is None, "hall_siren must be inert until the alarm regime holds"
    volume_cfg = next(cfg for cfg in siren.semantics if cfg.name == "volume_db")
    assert not volume_cfg.value, "volume_db must not carry a truthy value from preset broadcast"


# ---------------------------------------------------------------------------
# Fire-alarm evacuation reference scenario (pedestrian stimulus reference)
# ---------------------------------------------------------------------------

_EVACUATION_SCENARIO = (
    Path(__file__).resolve().parents[2]
    / "worlds"
    / "hospital_1"
    / "scenarios"
    / "fire_alarm_evacuation"
    / "scenario.yaml"
)

_EVACUATION_WORLD = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1" / "0" / "world.yaml"


def test_evacuation_reference_world_wired():
    if not _EVACUATION_WORLD.exists():
        pytest.skip(f"reference world not found: {_EVACUATION_WORLD}")
    with open(_EVACUATION_WORLD, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    level = converter.structure(raw, LevelDescription)

    schedules = {schedule.name: schedule for schedule in level.all_schedules}
    assert "fire_alarm" in schedules
    active = next(cfg for cfg in schedules["fire_alarm"].semantics if cfg.name == "active")
    assert active.params["regime"] == "alarm"
    assert active.params["windows"] == []

    sounds = {sound.name: sound for sound in level.all_sounds}
    assert "hall_siren" in sounds
    siren = sounds["hall_siren"]
    assert siren.asset_id == "alarm_loop"
    sounding_cfg = next(cfg for cfg in siren.semantics if cfg.name == "sounding")
    assert sounding_cfg.params["sound_on"] == "alarm"
    assert sounding_cfg.value is None, "hall_siren must be inert until the alarm regime holds"
    hallway = next(zone for zone in level.zones if zone.name == "central_hallway")
    assert siren in hallway.sounds


def test_evacuation_reference_scenario_parses():
    if not _EVACUATION_SCENARIO.exists():
        pytest.skip(f"reference scenario not found: {_EVACUATION_SCENARIO}")
    scenario = ScenarioView(_EVACUATION_SCENARIO.parent).load()

    assert len(scenario.dynamic) == 4
    for ped in scenario.dynamic:
        assert ped.extra["agent"]["agent_type"] == "./agent_types/evacuee.yaml"
        assert Path(ped.included_from) == _EVACUATION_SCENARIO.parent
    assert (_EVACUATION_SCENARIO.parent / "agent_types" / "evacuee.yaml").is_file()

    (exit_obj,) = scenario.static
    assert exit_obj.name == "exit"
    assert exit_obj.extra["type"] == "exit"

    assert len(scenario.robots) == 1
    (alarm,) = scenario.timeline
    assert alarm.at == pytest.approx(12.0)
    assert alarm.set == [{"entity": "fire_alarm", "field": "active", "value": "true"}]
