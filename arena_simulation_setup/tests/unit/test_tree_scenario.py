from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.shared import Light
from arena_simulation_setup.shared.conditions import EpisodeCondition
from arena_simulation_setup.shared.task import GoToPhase, PlayGesturePhase, TaskPhase
from arena_simulation_setup.tree.World import WorldView, anchor_light, contents
from arena_simulation_setup.tree.World.Scenario import (
    SCENARIO_FORMAT,
    RobotGoal,
    Scenario,
    ScenarioView,
    StimulusAction,
    TimelineEntry,
    parse_scenario,
    upgrade_scenario,
)
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Pose, Position

# ---------------------------------------------------------------------------
# RobotGoal.parse
# ---------------------------------------------------------------------------


def test_robot_goal_parse_full_dict():
    rg = RobotGoal.parse({"start": [1.0, 2.0], "level": 1, "goal_level": "2", "phases": [{"goto": [3.0, 4.0, 0.0]}]})
    assert rg.start.position.x == pytest.approx(1.0)
    assert rg.phases[0].pose.position.x == pytest.approx(3.0)
    assert (rg.level, rg.goal_level) == ("1", "2")


def test_robot_goal_parse_name_and_model():
    rg = RobotGoal.parse({"start": [0.0, 0.0], "name": "lead", "model": "jackal"})
    assert (rg.name, rg.model) == ("lead", "jackal")
    assert RobotGoal.parse({"start": [0.0, 0.0]}).name == ""


def test_robot_goal_rejects_unknown_key_beside_name_and_model():
    with pytest.raises(ValueError, match="unknown keys"):
        RobotGoal.parse({"start": [0.0, 0.0], "name": "lead", "models": "jackal"})


def test_scenario_rejects_two_robots_with_one_name():
    first = RobotGoal.parse({"start": [0.0, 0.0], "name": "lead"})
    second = RobotGoal.parse({"start": [1.0, 1.0], "name": "lead"})
    with pytest.raises(ValueError, match="duplicate scenario robot"):
        Scenario(robots=[first, second])


def test_robot_goal_parse_start_only():
    rg = RobotGoal.parse({"start": [5.0, 6.0]})
    assert rg.start.position.x == pytest.approx(5.0)
    assert rg.phases == []
    assert (rg.level, rg.goal_level) == ("", "")


def test_robot_goal_legacy_goal_becomes_one_goto_phase():
    upgraded, _ = upgrade_scenario({"robots": [{"start": [0.0, 0.0], "goal": [7.0, 8.0], "start_floor": "1", "goal_floor": "2"}]})
    (rg,) = parse_scenario(upgraded).robots
    (phase,) = rg.phases
    assert isinstance(phase, GoToPhase)
    assert phase.pose.position.x == pytest.approx(7.0)
    assert isinstance(rg.start, Pose)
    assert (rg.level, rg.goal_level) == ("1", "2")


def test_robot_goal_rejects_format_1_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        RobotGoal.parse({"start": [0.0, 0.0], "goal": [7.0, 8.0]})


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


def test_scenario_parses_episode_lights():
    s = converter.structure(
        {
            "lights": [
                {"name": "floor_lamp", "fixture": "bulb", "pose": {"x": 2.0, "y": 1.0, "z": 1.5}, "lumens": 800, "light_on": "!blackout"},
                {"name": "desk_spot", "fixture": "spot", "attach": "desk", "offset": {"x": 0.2, "y": 0.0, "z": 0.8}, "lumens": 300},
            ],
        },
        Scenario,
    )
    assert [light.name for light in s.lights] == ["floor_lamp", "desk_spot"]
    assert s.lights[0].light_on == "!blackout"
    assert not s.lights[0].intrinsic


@pytest.mark.parametrize(
    ("lights", "match"),
    [
        ([Light(name="sky", fixture="dome", lux=50.0)], "belongs in the level lights"),
        ([Light(name="lamp", fixture="bulb", pose=Pose(position=Position(1.0, 1.0, 1.0)), lumens=800.0), Light(name="lamp", fixture="bulb", pose=Pose(position=Position(2.0, 1.0, 1.0)), lumens=800.0)], "duplicate scenario light 'lamp'"),
    ],
)
def test_scenario_rejects_ambient_and_duplicate_lights(lights, match):
    with pytest.raises(ValueError, match=match):
        Scenario(lights=lights)
    with pytest.raises(Exception, match="While structuring Scenario") as caught:
        converter.structure({"lights": converter.unstructure(lights)}, Scenario)
    assert caught.group_contains(ValueError, match=match)


def test_scenario_light_anchors_on_a_scenario_object():
    s = parse_scenario(
        {
            "format": SCENARIO_FORMAT,
            "objects": [{"name": "desk", "model": "desk", "pose": [4.0, 2.0]}],
            "lights": [{"name": "desk_spot", "fixture": "spot", "attach": "desk", "offset": {"x": 0.2, "y": 0.0, "z": 0.8}, "lumens": 300}],
        },
    )
    light = anchor_light(s.lights[0], s.objects)
    assert (light.owner, light.attach) == ("desk", "")
    assert (light.position.x, light.position.y, light.position.z) == pytest.approx((4.2, 2.0, 0.8))
    with pytest.raises(ValueError, match="unknown"):
        anchor_light(s.lights[0], [])


def test_scenario_with_robots():
    rg = RobotGoal.parse({"start": [0.0, 0.0], "phases": [{"goto": [1.0, 1.0, 0.0]}]})
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
    yaml_file.write_text(yaml.dump({"format": SCENARIO_FORMAT, "objects": [], "humans": [], "robots": []}))
    json_file.write_text(f'{{"format": "{SCENARIO_FORMAT}", "objects": [], "humans": [], "robots": []}}')

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
    data = {"format": SCENARIO_FORMAT, "objects": [], "humans": [], "robots": [{"start": [1.0, 2.0], "phases": [{"goto": [3.0, 4.0, 0.0]}]}]}
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        scenario = view.load()
    assert len(scenario.robots) == 1


def test_scenario_view_load_legacy(tmp_path):
    scenario_dir = tmp_path / "sc_legacy"
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
    with pytest.warns(DeprecationWarning, match="format 1"):
        scenario = view.load()
    assert [obstacle.name for obstacle in scenario.objects] == ["obs1"]
    assert scenario.humans == []
    assert len(scenario.robots) == 1


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
    data = {
        "format": SCENARIO_FORMAT,
        "objects": [{"name": "obs1", "pose": [0.0, 0.0], "model": "box"}],
        "humans": [{"name": "ped1", "pose": [1.0, 0.0], "model": "arenian"}],
        "robots": [],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    scenario = view.load()
    assert [obj.included_from for obj in (*scenario.objects, *scenario.humans)] == [scenario_dir, scenario_dir]


# ---------------------------------------------------------------------------
# TaskPhase dispatch
# ---------------------------------------------------------------------------


def test_scenario_phase_parse_goto():
    phase = TaskPhase.parse({"goto": [1.0, 2.0, 0.0]})
    assert isinstance(phase, GoToPhase)
    assert phase.pose.position.x == pytest.approx(1.0)


def test_scenario_phase_parse_gesture():
    phase = TaskPhase.parse({"gesture": "wave"})
    assert isinstance(phase, PlayGesturePhase)
    assert phase.gesture == "wave"


def test_scenario_phase_parse_malformed_raises():
    with pytest.raises(ValueError):
        TaskPhase.parse({"foo": "bar"})


# ---------------------------------------------------------------------------
# RobotGoal.phase_list -- legacy goal: path
# ---------------------------------------------------------------------------


def test_robot_goal_phase_list_legacy_goal_deprecation(tmp_path):
    (tmp_path / "scenario.yaml").write_text(yaml.dump({"robots": [{"start": [0.0, 0.0], "goal": [3.0, 4.0]}]}))
    with pytest.warns(DeprecationWarning):
        (rg,) = ScenarioView(tmp_path).load().robots
    phases = rg.phase_list()
    assert len(phases) == 1
    assert isinstance(phases[0], GoToPhase)
    assert phases[0].pose.position.x == pytest.approx(3.0)


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
        "format": SCENARIO_FORMAT,
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
    assert isinstance(phases[0], GoToPhase)
    assert phases[0].pose.position.x == pytest.approx(1.0)
    assert isinstance(phases[1], PlayGesturePhase)
    assert phases[1].gesture == "wave"
    assert isinstance(phases[2], GoToPhase)
    assert phases[2].pose.position.x == pytest.approx(2.0)
    assert isinstance(phases[3], PlayGesturePhase)
    assert phases[3].gesture is None


def test_robot_goal_phase_list_malformed_phase_raises(tmp_path):
    scenario_dir = tmp_path / "sc_bad_phase"
    scenario_dir.mkdir()
    data = {
        "format": SCENARIO_FORMAT,
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
    upgraded, _ = upgrade_scenario({"robots": [{"start": [0.0, 0.0], "goal": [9.0, 9.0], "phases": [{"gesture": "wave"}]}]})
    (rg,) = parse_scenario(upgraded).robots
    phases = rg.phase_list()
    assert len(phases) == 1
    assert isinstance(phases[0], PlayGesturePhase)


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
    entry = TimelineEntry.parse({"every": 5.0, "offset": 1.0, "until": 30.0, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}]})
    assert entry.every == pytest.approx(5.0)
    assert entry.offset == pytest.approx(1.0)
    assert entry.until == pytest.approx(30.0)


def test_timeline_entry_parse_when():
    entry = TimelineEntry.parse({"when": {"entity": "door_1", "field": "open", "is": True}, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}]})
    assert entry.when == {"entity": "door_1", "field": "open", "is": True}


def test_timeline_entry_when_reads_the_entity_form_as_an_eq_field_test():
    from arena_simulation_setup.shared.conditions import FieldTest

    entry = TimelineEntry.parse({"when": {"entity": "door_1", "field": "open", "is": True}, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}]})
    assert entry.when_node() == FieldTest(entity="door_1", field="open", op="eq", value="true")


def test_timeline_entry_when_takes_a_structured_node_and_keeps_its_form():
    from arena_simulation_setup.shared.conditions import FamilyTest, HoldNode

    raw = {"some": "fridge.openness", "ge": 0.9, "for": 2}
    entry = TimelineEntry.parse({"when": raw, "set": [{"entity": "a", "field": "x", "value": "1"}]})
    assert entry.when == raw
    assert entry.when_node() == HoldNode(node=FamilyTest(entity="fridge", family="openness", quantifier="some", op="ge", value=0.9), seconds=2.0)


@pytest.mark.parametrize(
    ("when", "message"),
    [
        ({"entity": "door_1", "field": "open", "is": True, "ge": 1}, "takes no keys \\['ge'\\]"),
        ({"entity": "door_1", "field": "open"}, "'eq' takes a string or a number"),
        ({"some": "fridge.openness", "ge": "high"}, "'ge' takes a number"),
        ("robot near kitchen", "robot near kitchen"),
    ],
)
def test_timeline_entry_when_rejects_a_bad_condition(when, message):
    with pytest.raises(ValueError, match=message):
        TimelineEntry.parse({"when": when, "set": [{"entity": "a", "field": "x", "value": "1"}]})


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


def test_timeline_entry_requires_an_action():
    with pytest.raises(ValueError, match="at least one action"):
        TimelineEntry.parse({"at": 1.0, "set": []})
    with pytest.raises(ValueError, match="at least one action"):
        TimelineEntry(at=1.0)


def test_timeline_entry_parses_stimulus_actions():
    entry = TimelineEntry.parse(
        {
            "when": "robot within 4.2 of 12.1,4.0",
            "stimulus": [{"agents": ["edge_0"], "need": "go", "intensity": 0.5}, {"agents": ["d_1", "d_2"], "need": "form"}],
        },
    )
    assert entry.set == []
    assert entry.stimulus == [
        StimulusAction(agents=("edge_0",), need="go", intensity=0.5),
        StimulusAction(agents=("d_1", "d_2"), need="form", intensity=1.0),
    ]


def test_timeline_entry_takes_set_and_stimulus_together():
    entry = converter.structure({"at": 8.0, "set": [{"entity": "fire_alarm", "field": "active", "value": "true"}], "stimulus": [{"agents": ["d_1"], "need": "form"}]}, TimelineEntry)
    assert len(entry.set) == 1
    assert entry.stimulus == [StimulusAction(agents=("d_1",), need="form")]


@pytest.mark.parametrize(
    ("stimulus", "message"),
    [
        ({"agents": "edge_0", "need": "go"}, "list of agent names"),
        ({"agents": [], "need": "go"}, "non-empty list of agent names"),
        ({"agents": ["edge_0", ""], "need": "go"}, "non-empty list of agent names"),
        ({"agents": ["edge_0"]}, "'need' must be a non-empty string"),
        ({"agents": ["edge_0"], "need": "go", "level": 1.0}, "unknown keys"),
        ({"agents": ["edge_0"], "need": "go", "intensity": "loud"}, "could not convert"),
        ("edge_0", "a stimulus is a mapping"),
    ],
)
def test_timeline_entry_rejects_malformed_stimulus(stimulus, message):
    with pytest.raises(ValueError, match=message):
        TimelineEntry.parse({"at": 1.0, "stimulus": [stimulus]})


def test_stimulus_round_trips_through_cattrs():
    entry = TimelineEntry.parse({"at": 8.0, "stimulus": [{"agents": ["d_1", "d_2"], "need": "form"}, {"agents": ["edge_0"], "need": "go", "intensity": 0.25}]})
    raw = converter.unstructure(entry)
    assert raw["stimulus"] == [{"agents": ["d_1", "d_2"], "need": "form"}, {"agents": ["edge_0"], "need": "go", "intensity": 0.25}]
    assert converter.structure(raw, TimelineEntry) == entry


def _human(name: str) -> dict:
    return {"name": name, "model": "arenian", "pose": [1.0, 2.0]}


def test_scenario_timeline_stimulus_loads_from_yaml(tmp_path):
    scenario_dir = tmp_path / "sc_stimulus"
    scenario_dir.mkdir()
    data = {
        "format": SCENARIO_FORMAT,
        "humans": [_human("edge_0")],
        "timeline": [
            {"when": "robot within 4.2 of 12.1,4.0", "stimulus": [{"agents": ["edge_0"], "need": "go"}]},
        ],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    (entry,) = ScenarioView(scenario_dir).load().timeline
    assert entry.when == "robot within 4.2 of 12.1,4.0"
    assert entry.stimulus == [StimulusAction(agents=("edge_0",), need="go")]


def test_parse_scenario_names_the_bad_timeline_entry():
    with pytest.raises(ValueError, match=r"timeline\[1\]"):
        parse_scenario(
            {
                "format": SCENARIO_FORMAT,
                "humans": [_human("edge_0")],
                "timeline": [
                    {"at": 1.0, "stimulus": [{"agents": ["edge_0"], "need": "go"}]},
                    {"at": 2.0, "stimulus": [{"agents": ["edge_0"]}]},
                ],
            },
        )


def test_scenario_rejects_a_stimulus_agent_that_is_not_a_scenario_human():
    with pytest.raises(ValueError, match=r"timeline\[1\]: stimulus 'form' names agent 'd_2', which is not a scenario human, expected one of \['d_1', 'edge_0'\]"):
        parse_scenario(
            {
                "format": SCENARIO_FORMAT,
                "humans": [_human("edge_0"), _human("d_1")],
                "timeline": [
                    {"at": 1.0, "stimulus": [{"agents": ["edge_0"], "need": "go"}]},
                    {"at": 2.0, "stimulus": [{"agents": ["d_1", "d_2"], "need": "form"}]},
                ],
            },
        )


def test_scenario_view_load_names_the_file_of_an_unknown_stimulus_agent(tmp_path):
    scenario_dir = tmp_path / "sc_typo"
    scenario_dir.mkdir()
    data = {"format": SCENARIO_FORMAT, "humans": [_human("edge_0")], "timeline": [{"at": 1.0, "stimulus": [{"agents": ["egde_0"], "need": "go"}]}]}
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))
    with pytest.raises(RuntimeError, match=r"sc_typo/scenario.yaml: timeline\[0\]: stimulus 'go' names agent 'egde_0'"):
        ScenarioView(scenario_dir).load()


def test_scenario_timeline_default_empty():
    s = Scenario()
    assert s.timeline == []


def test_scenario_timeline_structures_from_yaml(tmp_path):
    scenario_dir = tmp_path / "sc_timeline"
    scenario_dir.mkdir()
    data = {
        "format": SCENARIO_FORMAT,
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
        "format": SCENARIO_FORMAT,
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


def test_scenario_loads_structured_conditions_and_untils(tmp_path):
    scenario_dir = tmp_path / "sc_structured"
    scenario_dir.mkdir()
    until = {"all": [{"subject": "robot", "in": "ward_a"}, {"field": "wardrobe.openness.door_left", "ge": 0.8}], "for": 2}
    condition = {"op": "never_during", "p": {"subject": "robot", "in": "ward_a_doorway"}, "q": {"some": "wardrobe.openness", "lt": 0.5}}
    data = {
        "format": SCENARIO_FORMAT,
        "robots": [{"start": [1.0, 1.0, 0.0], "phases": [{"goto": [2.0, 2.0, 0.0], "until": until}], "conditions": [{"op": "eventually", "p": {"field": "wardrobe.state", "eq": "open"}}]}],
        "conditions": [condition],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    scenario = ScenarioView(scenario_dir).load()
    assert scenario.conditions[0].serialize() == condition
    assert scenario.robots[0].phases[0].until == until
    assert scenario.robots[0].phases[0].serialize()["until"] == until
    assert scenario.robots[0].conditions[0].p == {"field": "wardrobe.state", "eq": "open"}


def test_scenario_names_a_malformed_structured_condition(tmp_path):
    scenario_dir = tmp_path / "sc_structured_bad"
    scenario_dir.mkdir()
    data = {"format": SCENARIO_FORMAT, "conditions": [{"op": "eventually", "p": {"field": "wardrobe.openness.door_left", "ge": "wide"}}]}
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    with pytest.raises(RuntimeError, match=r"(?s)conditions\[0\].*'ge' takes a number"):
        ScenarioView(scenario_dir).load()


def test_scenario_conditions_round_trip_through_cattrs():
    cond = EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a"})
    s = Scenario(conditions=[cond])
    raw = converter.unstructure(s)
    reparsed = converter.structure(raw, Scenario)
    assert reparsed.conditions == [cond]


def test_scenario_conditions_rejects_malformed_entry():
    data = {
        "format": SCENARIO_FORMAT,
        "robots": [],
        "conditions": [{"op": "eventually"}],  # missing 'p'
    }
    with pytest.raises(Exception):
        converter.structure(data, Scenario)


def test_scenario_view_load_modern_condition_malformed_reraises(tmp_path):
    scenario_dir = tmp_path / "sc_bad_condition"
    scenario_dir.mkdir()
    data = {
        "format": SCENARIO_FORMAT,
        "robots": [],
        "conditions": [{"op": "until", "p": "robot in ward_a"}],  # unknown op
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with pytest.raises(RuntimeError, match=r"conditions\[0\]"):
        view.load()


def test_scenario_view_load_modern_binary_op_missing_q_reraises(tmp_path):
    scenario_dir = tmp_path / "sc_bad_condition2"
    scenario_dir.mkdir()
    data = {
        "format": SCENARIO_FORMAT,
        "robots": [],
        "conditions": [{"op": "before", "p": "robot in ward_a"}],  # binary op missing 'q'
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with pytest.raises(RuntimeError, match=r"conditions\[0\]"):
        view.load()


def test_scenario_view_modern_shape_does_not_degrade_to_empty(tmp_path):
    """A format 2 scenario that fails to parse must raise, not fall back to an empty legacy load."""
    scenario_dir = tmp_path / "sc_modern_broken"
    scenario_dir.mkdir()
    data = {
        "format": SCENARIO_FORMAT,
        "objects": [{"name": "obs1", "pose": [0.0, 0.0], "model": "box"}],
        "humans": [{"name": "ped1", "pose": [0.0, 0.0], "type": "arenian"}],
        "robots": [{"start": [1.0, 2.0], "phases": [{"goto": [3.0, 4.0, 0.0]}]}],
    }
    (scenario_dir / "scenario.yaml").write_text(yaml.dump(data))

    view = ScenarioView(scenario_dir)
    with pytest.raises(RuntimeError, match=r"humans\[0\]"):
        view.load()


def test_scenario_format_2_rejects_unknown_top_level_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        parse_scenario({"format": SCENARIO_FORMAT, "static": []})


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

_FIRE_ALARM_SCENARIO = Path(__file__).resolve().parents[2] / "worlds" / "three_storied_residential" / "scenarios" / "fire_alarm" / "scenario.yaml"


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


_FIRE_ALARM_WORLD = Path(__file__).resolve().parents[2] / "worlds" / "three_storied_residential"


def test_fire_alarm_reference_regime_wired():
    if not _FIRE_ALARM_WORLD.is_dir():
        pytest.skip(f"reference world not found: {_FIRE_ALARM_WORLD}")
    world = WorldView(_FIRE_ALARM_WORLD).load()
    level = world.levels["1"]

    schedules = {schedule.name: schedule for schedule in world.schedules}
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

_EVACUATION_SCENARIO = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1" / "scenarios" / "fire_alarm_evacuation" / "scenario.yaml"

_EVACUATION_WORLD = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"


def test_evacuation_reference_world_wired():
    if not _EVACUATION_WORLD.is_dir():
        pytest.skip(f"reference world not found: {_EVACUATION_WORLD}")
    world = WorldView(_EVACUATION_WORLD).load()
    (level,) = world.all_levels

    schedules = {schedule.name: schedule for schedule in world.schedules}
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
    assert "sound:hall_siren" in contents(world.kg, "central_hallway")


def test_evacuation_reference_scenario_parses():
    if not _EVACUATION_SCENARIO.exists():
        pytest.skip(f"reference scenario not found: {_EVACUATION_SCENARIO}")
    scenario = ScenarioView(_EVACUATION_SCENARIO.parent).load()

    assert len(scenario.humans) == 4
    for ped in scenario.humans:
        assert ped.agent["agent_type"] == "./agent_types/evacuee.yaml"
        assert Path(ped.included_from) == _EVACUATION_SCENARIO.parent
    assert (_EVACUATION_SCENARIO.parent / "agent_types" / "evacuee.yaml").is_file()

    (exit_obj,) = scenario.objects
    assert exit_obj.name == "exit"
    assert exit_obj.humansim["type"] == "exit"

    assert len(scenario.robots) == 1
    (alarm,) = scenario.timeline
    assert alarm.at == pytest.approx(12.0)
    assert alarm.set == [{"entity": "fire_alarm", "field": "active", "value": "true"}]
