from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from arena_humansim.core.agents import loader
from arena_simulation_setup.generative.scenario.formats import read_scenario
from arena_simulation_setup.generative.scenario.promptgen import prompts as P
from arena_simulation_setup.generative.scenario.promptgen.lexicon import match
from arena_simulation_setup.generative.scenario.promptgen.pipeline import RobotPlan, generate, write_result
from arena_simulation_setup.generative.scenario.promptgen.profiles import ProfileError, derive, derived_name, profile, tier_value
from arena_simulation_setup.generative.scenario.promptgen.realize import LEAD_IN_M, ROUTE_TRIGGER_M, STAND_TRIGGER_M, trigger
from arena_simulation_setup.generative.scenario.promptgen.spec import SchemaError, parse_b, parse_c, parse_d, parse_when
from arena_simulation_setup.generative.scenario.promptgen.world import WorldContext
from arena_simulation_setup.generative.scenario.route import ROBOT_SPEED_MPS
from arena_simulation_setup.shared.conditions import parse_atom

np = pytest.importorskip("numpy")


def _zone(name, x0, y0, x1, y1):
    return {"name": name, "corners": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}], "walls": [], "doors": [], "entities": {"static": []}}


ZONES = [_zone("reception", 0, 0, 8, 8), _zone("central_hallway", 8, 3, 24, 5), _zone("open_work_area", 24, 0, 32, 8)]
ROBOT = {"start": [2.0, 4.0, 0.0], "phases": [{"goto": [30.0, 4.0, 0.0]}, {"goto": [2.0, 4.0, 3.14]}]}


def _world(tmp_path):
    root = tmp_path / "worlds" / "toy"
    root.mkdir(parents=True)
    zones = [{"name": z["name"], "polygon": [[c["x"], c["y"]] for c in z["corners"]], "floor": False} for z in ZONES]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    base = root / "scenarios" / "normal_a"
    base.mkdir(parents=True)
    doc = {
        "robots": [ROBOT],
        "humans": [
            {"name": "normal_1", "model": "female_adult_medical_01", "pose": "reception", "agent": {"agent_type": "elder", "desired_velocity": 0.7}, "waypoints": ["reception", "open_work_area"], "waypoint_mode": "reverse"},
            {"name": "normal_2", "model": "male_adult_medical_01", "pose": "open_work_area", "agent": {"agent_type": "adult", "desired_velocity": 1.1}, "waypoints": ["open_work_area", "reception"], "waypoint_mode": "reverse"},
            {"name": "normal_3", "model": "female_adult_police_01", "pose": "reception", "agent": {"agent_type": "hurried", "desired_velocity": 1.6}, "waypoints": ["reception", "open_work_area"], "waypoint_mode": "reverse"},
        ],
    }
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", **doc}, sort_keys=False))
    return tmp_path / "worlds", doc, base


@pytest.fixture
def toy(tmp_path: Path) -> tuple[Path, dict, Path, WorldContext]:
    worlds, doc, base = _world(tmp_path)
    world = WorldContext.load(worlds / "toy", doc)
    return worlds, doc, base, world


def _prompt(text, targets=(), **oracle):
    return P.Prompt(id="t_001", text=text, targets=tuple(targets), oracle=dict(oracle))


def test_tiers_scale_speed_by_profile() -> None:
    assert tier_value("speed", "high", prof=profile("child")) < tier_value("speed", "high", prof=profile("hurried"))
    assert tier_value("vision_range", "low") == 2.0


def test_aliases_resolve() -> None:
    assert profile("old man").name == "elder" and profile("kid").name == "child"
    with pytest.raises(ProfileError):
        profile("dragon")


def test_derive_pins_point_masses_and_refuses_bt_params_on_plain_types() -> None:
    doc = derive({"extends": "adult"}, {"vision_range": "low", "speed": "high"}, name="x", prof=profile("adult"))
    assert doc["perception"]["vision_range"] == {"mean": 2.0, "std": 0.0, "clip_low": 2.0, "clip_high": 2.0}
    assert doc["desired_velocity"]["mean"] == 1.7
    with pytest.raises(ProfileError, match="behavior-tree"):
        derive({"extends": "adult"}, {"need_decay": "high"}, name="y")
    assert derived_name("./talker.yaml", {"repulsion": "low"}) == "talker__repulsion_low"


@pytest.mark.parametrize(
    "text, case",
    [
        ("The fire alarm goes off and everyone heads for the exit.", "evacuation"),
        ("A power cut leaves the floor in the dark.", "blackout"),
        ("The break is over and everyone surges back to work all at once.", "stampede"),
        ("It is visiting hours and the place is packed.", "busy"),
        ("Everyone is evacuated from the ward.", "evacuation"),
        ("The crowd surges towards the lifts.", "stampede"),
        ("The workers rushed back, shoving.", "stampede"),
    ],
)
def test_lexicon_matches_the_taxonomy(text, case) -> None:
    got = match(text)
    assert got is not None and got.case == case


def test_time_phrases_are_read_from_the_prompt() -> None:
    from arena_simulation_setup.generative.scenario.promptgen.lexicon import when_in

    assert when_in("Everyone is on their phone, after half a minute.").as_dict() == {"at": 30.0}
    assert when_in("The lights go out, after a while. The elderly are slow, once the robot is halfway.", exclude_cases=("blackout",)).as_dict() == {"route_fraction": 0.5}
    assert when_in("A nurse stops in front of the robot for a while.") is None
    assert match("The lights go out, twenty seconds in.").when.as_dict() == {"at": 20.0}
    assert match("An evacuation is called; everybody makes for the exit, as soon as the robot sets off.").when.as_dict() == {"route_fraction": 0.1}


def test_lexicon_ignores_the_unrelated() -> None:
    assert match("An old man walks straight at the robot.") is None


def test_busy_reads_its_multiplier() -> None:
    assert match("The floor is very busy today.").k == 3.0


def test_when_phrases_and_mappings() -> None:
    from arena_simulation_setup.generative.scenario.promptgen.spec import When

    assert parse_when("as the robot arrives", default=When(at=0.0), where="x").robot_within == 3.0
    assert parse_when({"at": 4}, default=When(), where="x").at == 4.0
    with pytest.raises(SchemaError):
        parse_when({"at": 1, "robot_within": 2, "route_fraction": 0.5}, default=When(), where="x")


def test_decisions_reject_numbers_where_tiers_are_asked() -> None:
    with pytest.raises(SchemaError, match="not one of"):
        parse_b({"case": "intrinsic", "select": {"amount": "half"}, "params": {"speed": 0.5}})
    with pytest.raises(SchemaError, match="allowed parameter"):
        parse_b({"case": "intrinsic", "select": {"amount": "half"}, "params": {"mass": "low"}})
    with pytest.raises(SchemaError, match="B.params"):
        parse_b({"case": "intrinsic", "select": {"amount": "half"}, "params": "speed: low"})
    assert parse_c({"case": "approach", "angle": 450}).angle == 90.0
    with pytest.raises(SchemaError, match="two to"):
        parse_d({"case": "formation_break", "members": ["adult"]})
    assert parse_d({"case": "formation_break", "members": ["adult", "adult"]}).when.robot_within == 3.0


def test_world_plans_the_route_and_places_on_it(toy) -> None:
    _worlds, _doc, _base, world = toy
    assert world.route[0] == (2.0, 4.0) and abs(world.route[-1][0] - 22.0) < 1.0
    from arena_simulation_setup.generative.scenario.promptgen.world import MAX_LEG_M

    assert world.route_length <= MAX_LEG_M * 1.2
    pt, heading = world.point_at(0.5)
    assert pt == pytest.approx((12.0, 4.0), abs=0.3) and heading == pytest.approx((1.0, 0.0), abs=0.05)
    width, _l, _r = world.passage_width(pt, heading)
    assert 1.6 < width < 2.4
    assert world.valid((16.0, 4.0)) is not None and world.valid((16.0, 4.0), on_route=True) is None
    assert world.valid((4.0, 7.0)) is None and world.zone_of((4.0, 7.0)) == "reception"
    assert world.exit_zone() == "central_hallway" and world.work_zone() == "open_work_area"


def test_the_shipped_prompts_round_trip_through_the_lexicon() -> None:
    prompts = P.load_sets("")
    assert len(prompts) == 13 and len({p.id for p in prompts}) == 13
    for prompt in prompts:
        wanted = [t for t in prompt.targets if t.startswith("A:")]
        if wanted:
            got = match(prompt.text)
            assert got is not None and f"A:{got.case}" == wanted[0], prompt.text
    assert {len(p.targets) for p in prompts} >= {1, 2}


def _zones_of(world, steps: dict) -> list[str]:
    return [world.zone_of((s["target_pose"]["x"], s["target_pose"]["y"])) for s in steps.values()]


def test_blackout_bakes_each_walkers_route_into_a_tree_tuned_at_the_onset(toy) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt("The lights go out and everyone slows down.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok" and result.record.steps == ["A:blackout"]
    (stimulus,) = result.effects
    assert stimulus == {"type": "stimulus", "label": "A:blackout", "scope": ["normal_1", "normal_2", "normal_3"], "need": "blackout", "when": {"at": 5.0}}
    agents = {a["name"]: a for a in result.doc["humans"]}
    assert all({k: v for k, v in agents[a["name"]].items() if k != "agent"} == {k: v for k, v in a.items() if k != "agent"} for a in doc["humans"])
    assert agents["normal_1"]["agent"]["desired_velocity"] == 0.7
    assert set(result.files.docs) == {a["agent"]["agent_type"] for a in agents.values()}
    elder = result.files.docs[agents["normal_1"]["agent"]["agent_type"]]
    assert elder["extends"] == "elder" and elder["mode"] == "behavior_tree" and elder["sequences"]["walk"]["then"] == "walk"
    assert _zones_of(world, elder["sequences"]["walk"]["steps"]) == ["reception", "open_work_area"]
    assert elder["tunes"] == [{"when": {"blackout": {"above": 50.0}}, "tune": {"perception": {"vision_range": 2.0}, "desired_velocity": {"mul": pytest.approx(0.6 / 1.1, abs=0.01)}}}]

    out = write_result(result, worlds / "toy")
    loaded = loader.load_agent_type_from_file(out / agents["normal_1"]["agent"]["agent_type"])
    assert loaded.mode == "behavior_tree" and len(loaded.sequences["walk"].steps) == 2


def test_talkers_no_step_touches_keep_their_type_file(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("A child crosses the robot's path from the right right in the robot's way.", ["C:approach"], C={"case": "approach", "profile": "child", "angle": 90, "route_fraction": 0.5})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    out = write_result(result, worlds / "toy")
    assert {a["agent"]["agent_type"] for a in result.doc["humans"] if a["name"].startswith("talker_")} == {"./talker.yaml"}
    assert (out / "talker.yaml").read_text() == (base / "talker.yaml").read_text()


def test_walk_route_bakes_the_waypoint_mode(toy) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.step_a import walk_route

    worlds, doc, base, world = toy
    walker = {"name": "w", "waypoints": ["reception", "open_work_area", [20.0, 4.0, 0.0]], "waypoint_mode": "reverse"}
    there_and_back = walk_route(walker, world, seed=0)
    assert [world.zone_of(p) for p in there_and_back] == ["reception", "open_work_area", "central_hallway", "open_work_area"]
    assert there_and_back[2] == (20.0, 4.0) and walk_route(walker, world, seed=0) == there_and_back
    shuffled = walk_route({**walker, "waypoint_mode": "random"}, world, seed=0)
    stops = [world.zone_of(p) for p in shuffled]
    assert len(stops) == 6 and all(a != b for a, b in zip(stops, stops[1:]))


def test_evacuation_switches_the_crowd_present_at_step_a(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The fire alarm goes off and everyone heads for the exit. A child crosses the robot's path from the right right in the robot's way.", ["A:evacuation", "C:approach"], C={"case": "approach", "profile": "child", "angle": 90, "route_fraction": 0.5})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    stimulus, intercept = result.effects
    assert stimulus == {"type": "stimulus", "label": "A:evacuation", "scope": ["normal_1", "normal_2", "normal_3"], "need": "alarm", "when": {"at": 4.0}}
    for agent in result.doc["humans"][:3]:
        evacuate = result.files.docs[agent["agent"]["agent_type"]]["sequences"]["evacuate"]
        assert evacuate["tune"]["max_velocity"] == 2.6 and evacuate["tune"]["desired_velocity"]["mul"] > 1.0
        assert _zones_of(world, {"leave": evacuate["steps"]["leave"]}) == ["central_hallway"]
    assert intercept["type"] == "intercept" and intercept["angle"] == 90.0 and intercept["profile"] == "child"
    assert result.record.owned["C"] == ["edge_0"]


def _with_talkers(doc: dict, base: Path) -> dict:
    from arena_simulation_setup.generative.scenario.populate import bt_type_source

    (base / "talker.yaml").write_text(bt_type_source().read_text())
    talkers = [{"name": f"talker_{i}", "model": "male_adult_medical_01", "pose": "reception", "agent": {"agent_type": "./talker.yaml", "desired_velocity": 1.1}, "waypoints": ["reception"]} for i in (1, 2)]
    return {**doc, "humans": [*doc["humans"], *talkers]}


def test_evacuation_switches_talkers_and_walkers_by_one_stimulus(toy, tmp_path: Path) -> None:
    worlds, doc, base, world = toy
    doc = _with_talkers(doc, base)
    result = generate(_prompt("The fire alarm goes off and everyone heads for the exit.", ["A:evacuation"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    (stimulus,) = result.effects
    assert stimulus == {"type": "stimulus", "label": "A:evacuation", "scope": ["normal_1", "normal_2", "normal_3", "talker_1", "talker_2"], "need": "alarm", "when": {"at": 4.0}}
    refs = {a["name"]: a["agent"]["agent_type"] for a in result.doc["humans"]}
    ref = refs["talker_1"]
    assert ref == "./types/talker__evacuate_talker_1.yaml" and refs["talker_2"] == "./types/talker__evacuate_talker_2.yaml"
    leaves = [result.files.docs[r]["sequences"]["evacuate"]["steps"]["leave"]["target_pose"] for r in refs.values()]
    assert len({(p["x"], p["y"]) for p in leaves}) == len(refs) and set(_zones_of(world, {str(i): {"target_pose": p} for i, p in enumerate(leaves)})) == {"central_hallway"}
    switched = result.files.docs[ref]
    assert switched["needs"]["alarm"] == {"initial": 0.0, "decay_rate": 0.0}
    assert all({"when": {"alarm": {"above": 50.0}}, "goto": "evacuate"} in switched["sequences"][s]["transitions"] for s in ("chat", "step_aside"))
    assert _zones_of(world, {"leave": switched["sequences"]["evacuate"]["steps"]["leave"]}) == ["central_hallway"]
    assert switched["sequences"]["evacuate"]["tune"] == {"max_velocity": 2.6, "desired_velocity": {"mul": pytest.approx(1.7 / 1.1, abs=0.01)}}
    walker = result.files.docs[result.doc["humans"][1]["agent"]["agent_type"]]
    assert {"when": {"alarm": {"above": 50.0}}, "goto": "evacuate"} in walker["sequences"]["walk"]["transitions"]

    out = write_result(result, worlds / "toy")
    assert loader.load_agent_type_from_file(out / ref).sequences["evacuate"].tune.fields["max_velocity"] == (2.6,)
    assert all(loader.load_agent_type_from_file(out / a["agent"]["agent_type"]).mode == "behavior_tree" for a in result.doc["humans"])
    assert read_scenario(out / "scenario.yaml")["timeline"] == [{"at": 4.0, "stimulus": [{"agents": ["normal_1", "normal_2", "normal_3", "talker_1", "talker_2"], "need": "alarm"}]}]


@pytest.mark.parametrize(
    ("text", "targets", "oracle", "need", "tune", "scope"),
    [
        ("The lights go out and everyone slows down.", ["A:blackout"], {}, "blackout", {"perception": {"vision_range": 2.0}, "desired_velocity": {"mul": pytest.approx(0.6 / 1.1, abs=0.01)}}, ["normal_1", "normal_2", "normal_3", "talker_1", "talker_2"]),
        ("Half the people here cannot see far.", ["B:intrinsic"], {"B": {"case": "intrinsic", "select": {"amount": "all"}, "params": {"vision_range": "low"}}}, "intrinsic", {"perception": {"vision_range": 2.0}}, ["normal_1", "normal_2", "normal_3"]),
    ],
)
def test_trees_hold_a_tune_while_the_need_is_up(toy, text: str, targets: list[str], oracle: dict, need: str, tune: dict, scope: list[str]) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt(text, targets, **oracle), world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    (stimulus,) = result.effects
    assert stimulus["scope"] == scope and stimulus["need"] == need
    ref = next(a["agent"]["agent_type"] for a in result.doc["humans"] if a["name"] == scope[-1])
    tuned = result.files.docs[ref]
    assert tuned["needs"][need] == {"initial": 0.0, "decay_rate": 0.0}
    assert tuned["tunes"] == [{"when": {need: {"above": 50.0}}, "tune": tune}]

    out = write_result(result, worlds / "toy")
    assert loader.load_agent_type_from_file(out / ref).tunes[0].tune.fields["perception.vision_range"] == (2.0,)


def test_a_stampede_switches_everyone_instead_of_rerouting(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The break is over and the whole crowd rushes back to work all at once.", ["A:stampede"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert [e["type"] for e in result.effects] == ["stimulus"]
    rush = result.files.docs["./types/talker__rush_talker_1.yaml"]["sequences"]["rush"]
    assert rush["tune"] == {"max_velocity": 2.6, "local_planner_params": {"repulsion_strength": 0.5}, "desired_velocity": {"mul": pytest.approx(1.7 / 1.1, abs=0.01)}}
    assert _zones_of(world, {"leave": rush["steps"]["leave"]}) == ["open_work_area"]


def test_busy_injects_and_stampede_runs_on_a_small_crowd(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("It is visiting hours and the place is packed.", ["A:busy"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert len(result.doc["humans"]) > 3 and result.record.owned["A"][0].startswith("busy_")
    prompt = _prompt("The break is over and the whole crowd rushes back to work all at once.", ["A:stampede"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    (stimulus,) = result.effects
    assert stimulus["need"] == "alarm" and len(stimulus["scope"]) == len(result.doc["humans"]) >= 6
    assert all(result.files.docs[a["agent"]["agent_type"]]["sequences"]["rush"]["tune"]["local_planner_params"] == {"repulsion_strength": 0.5} for a in result.doc["humans"])


def test_blocking_line_spans_the_corridor(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Two nurses stand side by side across the corridor right in the robot's way, chatting, and do not move aside.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "blocking", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    members = [a for a in result.doc["humans"] if a["name"].startswith("d_")]
    assert len(members) >= 2
    assert all(isinstance(a["pose"], str) and a["waypoint_mode"] == "reverse" for a in members)
    (form,) = result.effects
    assert form["type"] == "stimulus" and form["label"].endswith(":form") and form["need"] == "form" and form["when"] == {"at": 4.0}
    assert form["scope"] == sorted(a["name"] for a in members)
    types = [result.files.docs[a["agent"]["agent_type"]] for a in members]
    assert all(t["extends"] == profile("nurse").base and "walk" in t["sequences"] for t in types)
    spots = [t["sequences"]["form"]["steps"]["form"]["target_pose"] for t in types]
    ys = sorted(pt["y"] for pt in spots)
    assert ys[-1] - ys[0] >= 0.5 * (len(members) - 1)
    assert all(abs(pt["x"] - world.point_at(0.5)[0][0]) < 1.5 for pt in spots)
    assert all(t["sequences"]["form"]["tune"] == {"local_planner_params": {"repulsion_strength": 0.5}} for t in types)
    assert result.record.formation["variant"] == "blocking" and abs(result.record.formation["center"][0] - world.point_at(0.5)[0][0]) < 1.5


def test_a_talking_pair_stands_in_the_corridor_too_close_to_drive_between(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Two nurses stand talking right in the robot's way.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    spots = [result.files.docs[a["agent"]["agent_type"]]["sequences"]["form"]["steps"]["form"]["target_pose"] for a in result.doc["humans"] if a["name"].startswith("d_")]
    (a, b) = [(s["x"], s["y"]) for s in spots]
    assert math.dist(a, b) == pytest.approx(0.9, abs=0.05)
    assert world.route_clearance(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)) < 0.5


def test_formation_break_scatters_on_approach(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Three colleagues stand in a circle in the work area; they scatter as the robot arrives.", ["D:formation_break"], D={"case": "formation_break", "variant": "circle", "members": ["adult", "adult", "adult"], "place": {"zone": "open_work_area"}, "when": {"robot_within": 3.0}, "break": "scatter"})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    form, scatter = result.effects
    assert form["type"] == "rally" and form["when"] == {"at": 4.0} and len(form["targets"]) == 3
    assert scatter["type"] == "scatter" and scatter["when"]["robot_within"] >= 3.0 and len(scatter["scope"]) == 3
    center = tuple(scatter["when"]["of"])
    assert center == tuple(result.record.formation["center"]) and world.zone_of(center) == "open_work_area"
    members = [a for a in result.doc["humans"] if a["name"] in scatter["scope"]]
    assert all(isinstance(a["pose"], str) and a["pose"] != "open_work_area" for a in members)


def _formation_break(toy, place: dict, when: dict) -> object:
    _worlds, doc, base, world = toy
    prompt = _prompt("Three colleagues stand in a circle; they scatter as the robot arrives.", ["D:formation_break"], D={"case": "formation_break", "variant": "circle", "members": ["adult", "adult", "adult"], "place": place, "when": when, "break": "scatter"})
    return generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)


def test_a_break_the_robot_reaches_before_the_formation_forms_is_rejected(toy) -> None:
    result = _formation_break(toy, {"zone": "reception"}, {"robot_within": 3.0})
    assert result.record.d is None and not result.timeline
    assert [f.step for f in result.record.failures] == ["D"] and "before it forms" in result.record.failures[0].reason


def test_a_break_near_the_start_is_narrowed_until_the_formation_forms_first(toy) -> None:
    world = toy[3]
    result = _formation_break(toy, {"route_fraction": 0.3}, {"robot_within": 3.0})
    assert result.record.status == "ok", result.record.failures
    _form, scatter = result.effects
    reach = math.dist(world.start, tuple(scatter["when"]["of"])) - scatter["when"]["robot_within"]
    assert scatter["when"]["robot_within"] < 3.0 and reach / ROBOT_SPEED_MPS >= 7.0 - 0.05
    assert any("narrowed" in w for w in result.record.warnings)


def test_a_route_fraction_break_moves_past_the_time_the_formation_forms(toy) -> None:
    world = toy[3]
    result = _formation_break(toy, {"zone": "open_work_area"}, {"route_fraction": 0.1})
    assert result.record.status == "ok", result.record.failures
    _form, scatter = result.effects
    fraction = scatter["when"]["route_fraction"]
    assert fraction > 0.1 and (math.dist(world.start, world.point_at(fraction)[0]) - ROUTE_TRIGGER_M) / ROBOT_SPEED_MPS >= 7.0


def test_stubborn_is_a_head_on_intercept_that_does_not_yield(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("An old man walks towards the robot and refuses to give way.", ["D:stubborn"], D={"case": "stubborn", "profile": "elder", "route_fraction": 0.4})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    (effect,) = result.effects
    assert effect["type"] == "intercept" and effect["angle"] == 180.0 and "repulsion_low" in effect["profile"]
    assert effect["speed"] < 1.0


def test_b_tunes_only_unowned_agents(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The elderly people are slow to react and do not get out of the way.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"profile": "elder"}, "params": {"reaction": "high", "repulsion": "low"}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.owned["B"] == ["normal_1"]
    elder = next(a for a in result.doc["humans"] if a["name"] == "normal_1")
    (stimulus,) = result.effects
    assert stimulus == {"type": "stimulus", "label": "B:intrinsic", "scope": ["normal_1"], "need": "intrinsic", "when": {"at": 5.0}}
    tuned = result.files.docs[elder["agent"]["agent_type"]]
    assert tuned["extends"] == "elder" and tuned["tunes"] == [{"when": {"intrinsic": {"above": 50.0}}, "tune": {"local_planner_params": {"relaxation_time": 1.0, "repulsion_strength": 0.5}}}]
    assert all(a["agent"]["agent_type"] in ("adult", "hurried") for a in result.doc["humans"] if a["name"] != "normal_1")


def _one_stop(name: str, pose: list[float], stop: list[float]) -> dict:
    return {"name": name, "model": "male_adult_medical_01", "pose": pose, "agent": {"agent_type": "adult", "desired_velocity": 1.1}, "waypoints": [stop]}


def test_b_intrinsic_tunes_walkers_and_keeps_them_walking(toy) -> None:
    worlds, doc, base, world = toy
    crowd = {"robots": doc["robots"], "humans": [_one_stop("stander", [2.0, 2.0, 0.0], [2.0, 2.0, 0.0]), _one_stop("walker", [2.0, 6.0, 0.0], [6.0, 6.0, 0.0])]}
    prompt = _prompt("Everybody here is rushing.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"amount": "all"}, "params": {"speed": "high"}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=crowd, base_dir=base, oracle=True)
    assert result.record.owned["B"] == ["walker"]
    walker = next(a for a in result.doc["humans"] if a["name"] == "walker")
    steps = result.files.docs[walker["agent"]["agent_type"]]["sequences"]["walk"]["steps"]
    assert [(s["target_pose"]["x"], s["target_pose"]["y"]) for s in steps.values()] == [(6.0, 6.0), (2.0, 6.0)]


def test_b_intrinsic_on_a_crowd_that_only_stands_is_not_applicable(toy) -> None:
    worlds, doc, base, world = toy
    crowd = {"robots": doc["robots"], "humans": [_one_stop("stander", [2.0, 2.0, 0.0], [2.0, 2.0, 0.0])]}
    prompt = _prompt("Everybody here is rushing.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"amount": "all"}, "params": {"speed": "high"}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=crowd, base_dir=base, oracle=True)
    assert "B" not in result.record.owned and any("walking agents" in f.reason for f in result.record.failures)


def test_an_untunable_parameter_holds_from_the_start(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The elderly people are wide and slow to react.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"profile": "elder"}, "params": {"radius": "high", "reaction": "high"}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert [e["type"] for e in result.effects] == ["stimulus"]
    elder = next(a for a in result.doc["humans"] if a["name"] == "normal_1")
    tuned = result.files.docs[elder["agent"]["agent_type"]]
    assert tuned["agent_radius"]["mean"] == 0.4 and tuned["tunes"][0]["tune"] == {"local_planner_params": {"relaxation_time": 1.0}}


def _erratic(crowd: list[dict], world, base, *, seed: int = 0, **b) -> object:
    prompt = _prompt("Some people wander about with no settled destination.", ["B:erratic_route"], B={"case": "erratic_route", "select": {"amount": "all"}, **b})
    return generate(prompt, world=world, base_id="normal_a", base_doc={"robots": [ROBOT], "humans": crowd}, base_dir=base, oracle=True, seed=seed)


def test_erratic_points_lie_beside_the_middle_of_the_route(toy) -> None:
    worlds, doc, base, world = toy
    result = _erratic(doc["humans"], world, base, radius=1.5)
    for agent in result.doc["humans"]:
        sequences = result.files.docs[agent["agent"]["agent_type"]]["sequences"]
        points = [(s["target_pose"]["x"], s["target_pose"]["y"]) for name, seq in sequences.items() if name.startswith("erratic") for s in seq["steps"].values()]
        assert all(world.route_clearance(p) <= 2.5 and 0.1 <= world.fraction_of(p) <= 0.9 for p in points), points


def test_erratic_route_adds_wanderers_beside_a_route_too_few_are_near(toy) -> None:
    worlds, doc, base, world = toy
    result = _erratic([_one_stop("walker", [4.0, 6.0, 0.0], [6.0, 6.0, 0.0])], world, base)
    assert result.record.owned["B"] == ["walker", "wander_1", "wander_2"]
    wanderers = [a for a in result.doc["humans"] if a["name"].startswith("wander_")]
    assert all(world.route_clearance(tuple(a["pose"][:2])) >= 1.5 for a in wanderers)
    assert _erratic([_one_stop("walker", [4.0, 6.0, 0.0], [6.0, 6.0, 0.0])], world, base, around="home").record.owned["B"] == ["walker"]


def test_erratic_route_is_a_tree_sequence_for_walkers_and_trees_alike(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Everyone keeps changing where they are going.", ["B:erratic_route"], B={"case": "erratic_route", "select": {"amount": "all"}, "period": 3.0})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    (stimulus,) = result.effects
    assert stimulus["type"] == "stimulus" and stimulus["need"] == "erratic" and stimulus["scope"] == ["normal_1", "normal_2", "normal_3", "talker_1", "talker_2"]
    for agent in result.doc["humans"]:
        sequences = result.files.docs[agent["agent"]["agent_type"]]["sequences"]
        chain = [name for name in sequences if name.startswith("erratic")]
        assert len(chain) >= 2 and [sequences[n]["then"] for n in chain] == [sequences[n]["on_failure"] for n in chain] == [*chain[1:], chain[0]]
        assert all(step["patience"] == {"mean": 3.0} for n in chain for step in sequences[n]["steps"].values())

    out = write_result(result, worlds / "toy")
    written = read_scenario(out / "scenario.yaml")
    for agent in written["humans"]:
        assert loader.load_agent_type_from_file(out / agent["agent"]["agent_type"]).mode == "behavior_tree"
    radius = {a["name"]: a["agent"]["radius"] for a in written["humans"]}
    assert radius["normal_1"] == pytest.approx(0.23) and radius["normal_2"] == pytest.approx(0.25)


def test_a_reused_agent_without_a_slot_goes_back_unchanged(toy) -> None:
    worlds, doc, base, world = toy
    doc = _with_talkers(doc, base)
    reuse = ["normal_1", "normal_2", "normal_3", "talker_1"]
    prompt = _prompt("Two nurses stand side by side across the corridor right in the robot's way.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "blocking", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}, "reuse": reuse})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert sorted(result.record.owned["D"]) == ["normal_1", "normal_2", "normal_3"]
    (talker,) = [a for a in result.doc["humans"] if a["name"] == "talker_1"]
    assert talker == next(a for a in doc["humans"] if a["name"] == "talker_1")


def test_c_conflicting_with_d_moves_clear_of_the_d_placement(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt(
        "Two nurses stand talking right in the robot's way. A visitor stands still right in the robot's way, blocking it.",
        ["D:interaction_unbroken", "C:blocking"],
        D={"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}},
        C={"case": "blocking", "mode": "stand", "route_fraction": 0.5},
    )
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert result.record.c is not None and result.record.c.route_fraction != 0.5
    assert any("moved to route fraction" in w for w in result.record.warnings)


def test_written_scenario_is_self_contained(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The lights go out. A nurse stops right in front of the robot halfway along the robot's route and stares at it for a while.", ["A:blackout", "C:approach"], C={"case": "approach", "profile": "nurse", "angle": 180, "route_fraction": 0.5, "after": "stop_facing", "duration": 8})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    out = write_result(result, worlds / "toy")
    assert out is not None and (out / "scenario.yaml").is_file() and (out / "provenance.yaml").is_file() and (out / "record.json").is_file()
    assert not (out / "edge_case.yaml").exists()
    written = read_scenario(out / "scenario.yaml")
    provenance = yaml.safe_load((out / "provenance.yaml").read_text())
    assert provenance["prompt_id"] == "t_001" and provenance["steps"] == ["A:blackout", "C:approach"]
    assert provenance["decisions"]["C"]["after"] == "stop_facing"
    refs = [a["agent"]["agent_type"] for a in written["humans"]]
    assert any(ref.startswith("./types/") for ref in refs)
    for ref in refs:
        if ref.startswith("./"):
            assert (out / ref[2:]).is_file(), ref
    record = json.loads((out / "record.json").read_text())
    assert record["status"] == "ok" and record["dedup_key"] == result.record.dedup_key()


def test_robot_route_is_pinned_to_the_planned_points(toy) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    (robot,) = result.doc["robots"]
    assert robot["start"][:2] == [round(world.start[0], 2), round(world.start[1], 2)] and len(robot["start"]) == 3
    assert robot["phases"] == [{"goto": [round(world.goal[0], 2), round(world.goal[1], 2)]}]
    assert doc["robots"][0] == ROBOT and len(ROBOT["phases"]) == 2
    assert any("route pinned" in w for w in result.record.warnings)


def test_a_needs_a_crowd(toy) -> None:
    worlds, doc, base, world = toy
    empty = {"robots": doc["robots"], "humans": []}
    prompt = _prompt("The fire alarm goes off and everyone heads for the exit.", ["A:evacuation"])
    result = generate(prompt, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "denied" and result.record.failures[0].kind == "denied"
    d_slots = {"case": "interaction_unbroken", "variant": "standing", "members": ["adult", "adult", "adult"], "place": {"route_fraction": 0.5}}
    both = _prompt("The fire alarm goes off and everyone heads for the exit. Three colleagues stand talking halfway along the robot's route and will not stop.", ["A:evacuation", "D:interaction_unbroken"], D=d_slots)
    result = generate(both, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "partial" and result.record.steps == ["D:interaction_unbroken"]
    assert result.record.failures[0].kind == "denied"


def test_empty_base_is_denied_before_any_model_call(toy) -> None:
    worlds, doc, base, world = toy
    empty = {"robots": doc["robots"], "humans": []}
    prompt = _prompt("Half the people here are rushing.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"amount": "half"}, "params": {"speed": "high"}})
    result = generate(prompt, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "denied" and not result.record.steps
    assert result.record.failures[0].kind == "denied"


def test_d_generates_its_members_on_an_empty_base(toy) -> None:
    worlds, doc, base, world = toy
    empty = {"robots": doc["robots"], "humans": []}
    prompt = _prompt("Three colleagues stand talking halfway along the robot's route and will not stop.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "standing", "members": ["adult", "adult", "adult"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "ok" and len(result.record.owned["D"]) == 3
    assert len(result.doc["humans"]) == 3


def test_dedup_key_ignores_wording(toy) -> None:
    worlds, doc, base, world = toy
    a = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    b = generate(_prompt("A power cut leaves the floor in the dark.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert a.record.dedup_key() == b.record.dedup_key()
    assert math.isfinite(world.route_length)


def test_busy_quantity_beats_intensifier() -> None:
    assert match("The floor is busy today, twice as many people as usual.").k == 2.0
    assert match("The floor is very busy today, twice as many people as usual.").k == 2.0


def test_system_prompts_document_every_phrase() -> None:
    from arena_simulation_setup.generative.scenario.promptgen.spec import PHRASES

    root = Path(P.__file__).parent
    bc = (root / "system_bc.md").read_text().lower()
    d = (root / "system_d.md").read_text().lower()
    for phrase, _when in PHRASES:
        assert phrase in bc, phrase
        assert phrase in d, phrase
    assert "up to 8" in d


def test_wc_and_storage_placements_are_refused(toy) -> None:
    _worlds, doc, base, world = toy
    prompt = _prompt("Two colleagues stand talking in the washroom.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "standing", "members": ["adult", "adult"], "place": {"zone": "wc_west"}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert "D" not in result.record.owned and any("nobody is placed" in f.reason for f in result.record.failures), result.record.failures


def test_slightly_busy_reads_k_one_and_a_half() -> None:
    got = match("The floor is slightly busier than usual.")
    assert got is not None and got.case == "busy" and got.k == 1.5


def test_place_vocabulary_is_documented() -> None:
    root = Path(P.__file__).parent
    bc = (root / "system_bc.md").read_text()
    d = (root / "system_d.md").read_text()
    for phrase, _fraction in P.PLACE_ROUTE:
        assert phrase in bc, phrase
        assert phrase in d, phrase


def test_member_limits_follow_the_variant() -> None:
    assert parse_d({"case": "interaction_unbroken", "variant": "blocking", "members": ["adult"] * 6}) is not None
    with pytest.raises(SchemaError):
        parse_d({"case": "interaction_unbroken", "variant": "standing", "members": ["adult"] * 6})


def test_kb_system_prompts_are_committed_builds() -> None:
    from arena_simulation_setup.generative.scenario.promptgen import build_kb

    built = build_kb.build()
    root = Path(P.__file__).parent
    for name, text in built.items():
        assert (root / name).read_text() == text, f"{name} is stale - run python3 -m arena_simulation_setup.generative.scenario.promptgen.build_kb"


def _cache_a(cache: Path, text: str, answer: dict, model: str | None = None) -> Path:
    from arena_simulation_setup.generative.scenario.promptgen import step_a_llm
    from arena_simulation_setup.generative.scenario.promptgen.ask import system_prompt
    from arena_simulation_setup.generative.scenario.propose.client import prompt_sha256

    cache.mkdir(exist_ok=True)
    (cache / f"{prompt_sha256(model or step_a_llm.A_MODEL, system_prompt('a'), step_a_llm.user_prompt(text))}.json").write_text(json.dumps({"answer": json.dumps(answer)}))
    return cache


def test_step_a_llm_parses_and_tolerates_typos(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.spec import Record
    from arena_simulation_setup.generative.scenario.promptgen.step_a_llm import decide, who_line

    text = "An evacation is caled."
    rec = Record(prompt_id="t", prompt="", world="w", base="b")
    answer = {"case": "evacuation", "when": {"at": 10}, "who": [{"mention": "a visiter", "profile": "visitor", "count": 1}, {"mention": "three colleagues", "profile": "adult", "count": 3}, {"mention": "a dragon", "profile": "dragon", "count": 1}]}
    a, who, warnings = decide(text, rec, cache_dir=_cache_a(tmp_path / "llm_cache", text, answer))
    assert a is not None and a.case == "evacuation" and a.when.at == 10.0
    assert [w["profile"] for w in who] == ["adult", "adult"] and "dragon" in warnings[0]
    assert who_line(who).startswith("who (actors read from the prompt): 1 x adult")


_LIVE_A = """
import json, sys
from pathlib import Path
from arena_simulation_setup.generative.scenario.formats import read_scenario
from arena_simulation_setup.generative.scenario.promptgen import prompts as P
from arena_simulation_setup.generative.scenario.promptgen.pipeline import generate
from arena_simulation_setup.generative.scenario.promptgen.world import WorldContext

worlds, base, cache, text = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
model = sys.argv[5] if len(sys.argv) > 5 else None
doc = read_scenario(base / "scenario.yaml")
world = WorldContext.load(worlds / "toy", doc)
prompt = P.Prompt(id="t_001", text=text, targets=("A:evacuation",), oracle={})
record = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=False, cache_dir=cache, model=model).record
print(json.dumps({"a": record.a.case if record.a else None, "warnings": record.warnings, "who": record.provenance.get("who"), "a_models": [p["model"] for p in record.provenance.get("A", [])]}))
"""


def _live_a(worlds: Path, base: Path, cache: Path, text: str, *model: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_ENDPOINT", "GOOGLE_API_KEY", "GEMINI_API_KEY")}
    done = subprocess.run([sys.executable, "-c", _LIVE_A, str(worlds), str(base), str(cache), text, *model], env=env, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_llm_a_overrides_a_silent_lexicon(toy, tmp_path: Path) -> None:
    worlds, _doc, base, _world = toy
    text = "An evacation is caled; evrybody hurrys out."
    cache = _cache_a(tmp_path / "llm_cache", text, {"case": "evacuation", "who": [{"mention": "a visiter", "profile": "adult", "count": 1}]})
    got = _live_a(worlds, base, cache, text)
    assert got["a"] == "evacuation"
    assert any("lexicon silent" in w for w in got["warnings"])
    assert got["who"] == [{"mention": "a visiter", "profile": "adult", "count": 1}]


def test_the_model_override_reaches_step_a(toy, tmp_path: Path) -> None:
    worlds, _doc, base, _world = toy
    text = "An evacation is caled; evrybody hurrys out."
    cache = _cache_a(tmp_path / "llm_cache", text, {"case": "evacuation"}, model="override-model")
    got = _live_a(worlds, base, cache, text, "override-model")
    assert got["a"] == "evacuation" and got["a_models"] == ["override-model"]


_WITHIN = re.compile(r"robot within (\S+) of (\S+),(\S+)")


def _within(entry: dict) -> tuple[float, tuple[float, float]]:
    match = _WITHIN.fullmatch(entry["when"])
    assert match is not None, entry
    radius, x, y = (float(v) for v in match.groups())
    return radius, (x, y)


def _approach(toy, **c):
    _worlds, doc, base, world = toy
    prompt = _prompt("An adult walks straight at the robot halfway along the robot's route.", ["C:approach"], C={"case": "approach", "profile": "adult", "angle": 180, "route_fraction": 0.5, **c})
    return generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)


def _edge(result) -> tuple[dict, dict]:
    (edge,) = [a for a in result.doc["humans"] if a["name"] == "edge_0"]
    return edge, result.files.docs[edge["agent"]["agent_type"]]


def _encounter(result) -> tuple[float, float]:
    (encounter,) = result.encounters
    return tuple(encounter["point"])


def _released_m(result, world) -> float:
    """Meters of route the robot has left to the encounter when its release circle is entered."""
    radius, point = _within(result.timeline[0])
    return (world.fraction_of(_encounter(result)) - world.fraction_of(point)) * world.route_length + radius


def test_an_intercept_waits_until_a_timeline_stimulus_releases_it(toy) -> None:
    world = toy[3]
    result = _approach(toy)
    assert result.record.status == "ok", result.record.failures
    (entry,) = result.timeline
    assert set(entry) == {"when", "stimulus"} and entry["stimulus"] == [{"agents": ["edge_0"], "need": "go"}]
    parse_atom(entry["when"])
    radius, point = _within(entry)
    encounter = _encounter(result)
    edge, tree = _edge(result)
    spawn = (edge["pose"][0], edge["pose"][1])
    assert encounter == pytest.approx(world.point_at(0.5)[0], abs=0.05)
    assert math.dist(spawn, encounter) <= LEAD_IN_M + 0.01 and spawn[0] > encounter[0]
    assert radius == ROUTE_TRIGGER_M and world.route_clearance(point) < 0.05
    assert _released_m(result, world) == pytest.approx(ROBOT_SPEED_MPS * math.dist(spawn, encounter) / edge["agent"]["desired_velocity"], abs=0.1)
    assert tree["mode"] == "behavior_tree" and tree["initial_sequence"] == "wait" and tree["needs"]["go"] == {"initial": 0.0, "decay_rate": 0.0}
    assert {"when": {"go": {"above": 50.0}}, "goto": "walk"} in tree["sequences"]["wait"]["transitions"]
    assert list(tree["sequences"]["walk"]["steps"]) == ["cross", "stay"]
    assert tree["sequences"]["walk"]["steps"]["cross"]["target_pose"]["x"] < encounter[0]
    assert result.injected == ["edge_0"]
    assert result.provenance["owned"] == {"C": ["edge_0"], "D": [], "injected": ["edge_0"]}
    assert result.provenance["encounters"] == [{"agent": "edge_0", "point": [round(encounter[0], 2), round(encounter[1], 2)]}]


def test_the_release_distance_follows_the_designed_robot_speed(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("An adult walks straight at the robot.", ["C:approach"], C={"case": "approach", "profile": "adult", "angle": 180, "route_fraction": 0.5})
    slow = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    fast = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True, robot=RobotPlan(speed=2 * ROBOT_SPEED_MPS))
    assert _released_m(fast, world) == pytest.approx(2 * _released_m(slow, world), abs=0.1)


@pytest.mark.parametrize(
    ("after", "extra", "steps"),
    [
        ("stop_facing", {"duration": 8}, ["meet", "leave", "stay"]),
        ("veer", {"veer_to": "open_work_area"}, ["meet", "veer", "stay"]),
    ],
)
def test_the_after_variants_are_steps_of_the_walk(toy, after: str, extra: dict, steps: list[str]) -> None:
    world = toy[3]
    result = _approach(toy, after=after, **extra)
    assert result.record.status == "ok", result.record.failures
    _edge_agent, tree = _edge(result)
    walk = tree["sequences"]["walk"]["steps"]
    assert list(walk) == steps
    encounter = _encounter(result)
    meet = walk["meet"]["target_pose"]
    assert (meet["x"], meet["y"]) == pytest.approx(encounter, abs=0.01)
    if after == "stop_facing":
        assert walk["meet"]["duration"] == {"mean": 8.0} and abs(abs(meet["theta"]) - math.pi) < 0.01
    else:
        assert world.zone_of((walk["veer"]["target_pose"]["x"], walk["veer"]["target_pose"]["y"])) == "open_work_area"


def test_following_walks_the_robots_route_behind_it(toy) -> None:
    result = _approach(toy, after="follow", duration=6)
    _edge_agent, tree = _edge(result)
    walk = tree["sequences"]["walk"]["steps"]
    follow = [walk[k]["target_pose"]["x"] for k in walk if k.startswith("follow_")]
    encounter = _encounter(result)
    assert follow and follow == sorted(follow) and follow[0] > encounter[0]
    assert follow[-1] - encounter[0] == pytest.approx(min(6 * 1.1, 22.0 - encounter[0]), abs=0.1)


def test_a_blocker_stands_on_the_route_and_steps_on_when_the_robot_is_near(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("A visitor stands still right in the robot's way.", ["C:blocking"], C={"case": "blocking", "mode": "stand", "route_fraction": 0.5, "duration": 6})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    (entry,) = result.timeline
    radius, spot = _within(entry)
    edge, tree = _edge(result)
    assert radius == STAND_TRIGGER_M and spot == pytest.approx((edge["pose"][0], edge["pose"][1]), abs=0.01)
    assert world.route_clearance(spot) < 0.05
    go = tree["sequences"]["go"]["steps"]
    assert go["hold"]["duration"] == {"mean": 6.0} and go["release"]["target_pose"]["x"] > spot[0]


def test_a_formation_forms_and_scatters_on_needs_from_the_timeline(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt(
        "Three colleagues stand in a circle in the work area; they scatter as the robot arrives.", ["D:formation_break"], D={"case": "formation_break", "variant": "circle", "members": ["adult", "adult", "adult"], "place": {"zone": "open_work_area"}, "when": {"robot_within": 3.0}, "break": "scatter", "radius": 2.0}
    )
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    form, scatter = result.timeline
    members = sorted(result.record.owned["D"])
    assert form == {"at": 4.0, "stimulus": [{"agents": members, "need": "form"}]}
    assert scatter["stimulus"] == [{"agents": members, "need": "break"}]
    radius, center = _within(scatter)
    assert radius >= 3.0 and center == pytest.approx(tuple(result.record.formation["center"]), abs=0.01)
    assert result.record.formation["at"] == {"at": 4.0}
    for agent in (a for a in result.doc["humans"] if a["name"] in members):
        tree = result.files.docs[agent["agent"]["agent_type"]]
        assert {"walk", "form", "scatter"} <= set(tree["sequences"]) and tree["initial_sequence"] == "walk"
        assert {"when": {"break": {"above": 50.0}}, "goto": "scatter"} in tree["sequences"]["form"]["transitions"]
        spot, away = (tree["sequences"][s]["steps"]["go"]["target_pose"] for s in ("form", "scatter"))
        assert math.dist((spot["x"], spot["y"]), (away["x"], away["y"])) <= 3.0
    assert len(result.files.docs) == len(members)
    again = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert again.timeline == result.timeline and again.files.docs == result.files.docs


def test_a_robot_plan_writes_the_hold_and_the_traversal_legs(toy) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True, robot=RobotPlan(traversals=2, hold=5.0, tolerance=0.8))
    (robot,) = result.doc["robots"]
    goal = [round(world.goal[0], 2), round(world.goal[1], 2)]
    assert robot["level"] == "0"
    assert robot["phases"] == [{"hold_time": 5.0}, {"goto": goal, "tolerance_radius": 0.8}, {"goto": robot["start"], "tolerance_radius": 0.8}, {"goto": goal, "tolerance_radius": 0.8}]


def test_onsets_become_timeline_triggers(toy) -> None:
    world = toy[3]
    on_route = world.point_at(0.5)[0]
    assert trigger({"at": 5.0}, world) == {"at": 5.0}
    assert trigger({"route_fraction": 0.5}, world) == {"when": f"robot within 1.5 of {round(on_route[0], 2):g},{round(on_route[1], 2):g}"}
    assert trigger({"robot_within": 3.0, "of": [12.1, 4.0]}, world) == {"when": "robot within 3 of 12.1,4"}
    assert trigger({"robot_beyond": 2.5, "of": [-1.25, 4.0]}, world) == {"when": "not robot within 2.5 of -1.25,4"}
    assert trigger({"robot_within": 3.0}, world, reference=(1.0, 2.0)) == {"when": "robot within 3 of 1,2"}
    for when in ("robot within 3 of 12.1,4", "not robot within 2.5 of -1.25,4"):
        parse_atom(when)


def test_the_written_case_is_a_plain_scenario_the_scenario_model_reads(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt(
        "Three colleagues stand in a circle in the work area; they scatter as the robot arrives. An adult walks straight at the robot.",
        ["D:formation_break", "C:approach"],
        D={"case": "formation_break", "variant": "circle", "members": ["adult", "adult", "adult"], "place": {"zone": "open_work_area"}, "when": {"robot_within": 3.0}, "break": "scatter"},
        C={"case": "approach", "profile": "adult", "angle": 180, "route_fraction": 0.6},
    )
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    out = write_result(result, worlds / "toy")
    assert sorted(p.name for p in out.iterdir()) == ["provenance.yaml", "record.json", "scenario.yaml", "types"]
    raw = yaml.safe_load((out / "scenario.yaml").read_text())
    assert set(raw) == {"format", "humans", "robots", "timeline"}
    provenance = yaml.safe_load((out / "provenance.yaml").read_text())
    assert set(provenance) == {"id", "prompt_id", "steps", "note", "decisions", "owned", "formation", "encounters", "robot", "provenance"}
    assert set(provenance["formation"]) == {"center", "radius", "variant", "at"}
    assert "effects" not in json.dumps(provenance) + json.dumps(raw)

    from arena_simulation_setup.tree.World import WorldView
    from arena_simulation_setup.tree.World.Scenario import parse_scenario

    loaded = parse_scenario(raw, WorldView(worlds / "toy").load().compact_world({"0": (0.0, 0.0)}).zone_converter(np.random.default_rng(0)))
    assert [(e.stimulus[0].need, e.stimulus[0].agents) for e in loaded.timeline] == [("form", tuple(sorted(result.record.owned["D"]))), ("break", tuple(sorted(result.record.owned["D"]))), ("go", ("edge_0",))]
    assert {h.name for h in loaded.humans} >= {"edge_0", *result.record.owned["D"]}

    for agent in raw["humans"]:
        ref = agent["agent"]["agent_type"]
        if ref.startswith("./"):
            assert loader.load_agent_type_from_file(out / ref).mode == "behavior_tree"


def test_owned_names_follow_the_realized_intercepts_when_d_is_dropped(toy) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.agents import TypeFiles
    from arena_simulation_setup.generative.scenario.promptgen.pipeline import Result, _realized
    from arena_simulation_setup.generative.scenario.promptgen.spec import Record
    from arena_simulation_setup.generative.scenario.promptgen.step_bc import apply_c
    from arena_simulation_setup.generative.scenario.promptgen.step_d import apply as apply_d

    worlds, doc, base, world = toy
    files = TypeFiles(base_dir=base)
    d = parse_d({"case": "stubborn", "profile": "elder", "route_fraction": 0.3})
    c = parse_c({"case": "approach", "profile": "adult", "angle": 90, "route_fraction": 0.7})
    work = dict(doc)
    (stubborn,), _names, _notes, _info = apply_d(work, d, world, files, seed=0, owned=set(), robot_speed=ROBOT_SPEED_MPS)
    (approach,), _names, _notes = apply_c(c, world, files, seed=0, taken_fractions=[0.3])
    unrealizable = {**stubborn, "after": "veer", "veer_to": "nowhere"}
    record = Record(prompt_id="t_001", prompt="", world=world.name, base="normal_a", d=d, c=c, owned={"D": [], "C": []})

    result = _realized(Result(record=record, doc=work, files=files, effects=[unrealizable, approach], scenario_id="normal_a__t_001", robot=RobotPlan()), world, seed=0)

    assert result.record.d is None and "D" not in result.record.owned and [f.step for f in result.record.failures] == ["D"]
    assert result.injected == ["edge_0"] and result.record.owned["C"] == ["edge_0"]
    assert result.provenance["owned"] == {"C": ["edge_0"], "D": [], "injected": ["edge_0"]}
    assert [e["agent"] for e in result.provenance["encounters"]] == ["edge_0"]
    assert result.timeline[0]["stimulus"] == [{"agents": ["edge_0"], "need": "go"}]


def test_owned_names_number_the_realized_intercepts_in_order(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt(
        "An old man walks towards the robot and refuses to give way. A child crosses the robot's path from the right.",
        ["D:stubborn", "C:approach"],
        D={"case": "stubborn", "profile": "elder", "route_fraction": 0.3},
        C={"case": "approach", "profile": "child", "angle": 90, "route_fraction": 0.7},
    )
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert result.record.owned["D"] == ["edge_0"] and result.record.owned["C"] == ["edge_1"] and result.injected == ["edge_0", "edge_1"]


def _u_turn(tmp_path: Path) -> tuple[Path, dict, Path, WorldContext]:
    """A U-shaped hall: east along the south arm, up the east leg, west along the north arm, the arms 4 m apart."""
    root = tmp_path / "worlds" / "u_turn"
    root.mkdir(parents=True)
    rooms = {"south_arm": (0, 0, 8, 2), "east_leg": (8, 0, 10, 6), "north_arm": (0, 4, 8, 6)}
    zones = [{"name": n, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "floor": False} for n, (x0, y0, x1, y1) in rooms.items()]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    doc = {"robots": [{"start": [1.0, 1.0, 0.0], "phases": [{"goto": [1.0, 5.0, 0.0]}]}], "humans": []}
    base = root / "scenarios" / "empty"
    base.mkdir(parents=True)
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", **doc}, sort_keys=False))
    return tmp_path / "worlds", doc, base, WorldContext.load(root, doc)


def test_a_release_on_a_bent_route_fires_its_designed_arc_before_the_encounter(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.realize import entered_at

    _worlds, doc, base, world = _u_turn(tmp_path)
    assert world.route_length > 15.0 and max(x for x, _y in world.route) > 8.0
    prompt = _prompt("An old man shuffles straight at the robot.", ["C:approach"], C={"case": "approach", "profile": "elder", "speed": "low", "angle": 180, "route_fraction": 0.7})
    result = generate(prompt, world=world, base_id="empty", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    radius, point = _within(result.timeline[0])
    edge, _tree = _edge(result)
    encounter = _encounter(result)
    release = ROBOT_SPEED_MPS * math.dist((edge["pose"][0], edge["pose"][1]), encounter) / edge["agent"]["desired_velocity"]
    designed = world.fraction_of(encounter) * world.route_length - release
    assert world.zone_of(encounter) == "north_arm" and entered_at(world.route, encounter, release) < designed - 3.0
    assert radius == ROUTE_TRIGGER_M and entered_at(world.route, point, radius) == pytest.approx(designed, abs=0.2)


def test_a_proximity_onset_without_a_point_measures_to_the_route_beside_the_crowd(toy) -> None:
    worlds, doc, base, world = toy
    crowd = {"robots": doc["robots"], "humans": [_one_stop("walker", [4.0, 7.0, 0.0], [6.0, 7.0, 0.0])]}
    prompt = _prompt("Everybody here is rushing when the robot is right there.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"amount": "all"}, "params": {"speed": "high"}, "when": {"robot_within": 1.5}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=crowd, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    radius, point = _within(result.timeline[0])
    assert radius == 1.5 and point == pytest.approx((4.0, 4.0), abs=0.05) and world.route_clearance(point) < 0.05


def test_a_blocker_near_the_start_moves_until_its_trigger_does_not_hold_there(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("A visitor stands still right in the robot's way as it sets off.", ["C:blocking"], C={"case": "blocking", "mode": "stand", "route_fraction": 0.1, "duration": 6})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    radius, spot = _within(result.timeline[0])
    assert 0.1 * world.route_length < STAND_TRIGGER_M and math.dist(world.start, spot) > radius


def test_an_intercept_whose_release_holds_at_the_start_moves_along_the_route(toy) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.realize import holds_at_start

    world = toy[3]
    result = _approach(toy, profile="elder", speed="low", route_fraction=0.1)
    assert result.record.status == "ok", result.record.failures
    radius, point = _within(result.timeline[0])
    assert not holds_at_start(world, point, radius) and world.fraction_of(_encounter(result)) > 0.1


def test_timed_onsets_count_from_the_robots_departure_after_its_hold(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt(
        "Two nurses stand talking right in the robot's way after twenty seconds. The lights go out.",
        ["D:interaction_unbroken", "A:blackout"],
        D={"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}, "when": {"at": 20.0}},
    )
    plain = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    held = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True, robot=RobotPlan(hold=5.0))
    assert plain.record.status == held.record.status == "ok", held.record.failures
    assert sorted(e["at"] for e in plain.timeline) == [5.0, 20.0] and sorted(e["at"] for e in held.timeline) == [10.0, 25.0]
    assert held.record.formation["at"] == plain.record.formation["at"] == {"at": 20.0}


def test_provenance_records_the_robot_the_case_was_designed_for(toy) -> None:
    worlds, doc, base, world = toy
    plain = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert plain.provenance["robot"] == {"speed_mps": ROBOT_SPEED_MPS, "hold_s": 0.0, "traversals": 1, "tolerance_m": None, "radius_m": 0.35}
    plan = RobotPlan(speed=0.8, traversals=2, hold=3.0, tolerance=0.6)
    held = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True, robot=plan)
    out = write_result(held, worlds / "toy")
    assert yaml.safe_load((out / "provenance.yaml").read_text())["robot"] == {"speed_mps": 0.8, "hold_s": 3.0, "traversals": 2, "tolerance_m": 0.6, "radius_m": 0.35}


def test_the_robot_radius_is_the_clearance_of_its_planned_route(toy) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.world import WorldError
    from arena_simulation_setup.shared.pathing import iter_points

    worlds, doc, base, _world = toy
    wide = WorldContext.load(worlds / "toy", doc, robot_radius=0.8)
    assert all(wide.grid.clearance(p) >= 0.8 - wide.grid.res for p in iter_points(wide.route, 0.25))
    with pytest.raises(WorldError):
        WorldContext.load(worlds / "toy", doc, robot_radius=1.2)
    with pytest.raises(ValueError, match="radius"):
        generate(_prompt("The lights go out.", ["A:blackout"]), world=wide, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    result = generate(_prompt("The lights go out.", ["A:blackout"]), world=wide, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True, robot=RobotPlan(radius=0.8))
    assert result.provenance["robot"]["radius_m"] == 0.8


def _targets(result) -> list:
    return sorted((ref, seq, key, step["target_pose"]["x"], step["target_pose"]["y"]) for ref, doc in result.files.docs.items() for seq, body in doc["sequences"].items() for key, step in body["steps"].items() if "target_pose" in step)


@pytest.mark.parametrize(
    ("text", "targets", "oracle"),
    [
        ("The fire alarm goes off and everyone heads for the exit.", ["A:evacuation"], {}),
        ("An adult walks up to the robot, then veers off to the work area.", ["C:approach"], {"C": {"case": "approach", "profile": "adult", "angle": 180, "route_fraction": 0.5, "after": "veer", "veer_to": "open_work_area"}}),
        ("Everyone wanders about.", ["B:erratic_route"], {"B": {"case": "erratic_route", "select": {"amount": "all"}, "around": "home"}}),
        ("Three colleagues gather in the work area, then disperse.", ["D:formation_break"], {"D": {"case": "formation_break", "variant": "circle", "members": ["adult", "adult", "adult"], "place": {"zone": "open_work_area"}, "when": {"at": 10.0}, "break": "disperse"}}),
    ],
)
def test_the_seed_reaches_every_draw(toy, text: str, targets: list[str], oracle: dict) -> None:
    worlds, doc, base, world = toy

    def run(seed: int) -> object:
        return generate(_prompt(text, targets, **oracle), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True, seed=seed)

    first, again, other = run(1), run(1), run(2)
    assert first.record.status == "ok", first.record.failures
    assert _targets(first) == _targets(again) and _targets(first) != _targets(other)


def test_the_seed_reaches_the_wanderers_beside_the_route(toy) -> None:
    worlds, doc, base, world = toy
    walker = [_one_stop("walker", [4.0, 6.0, 0.0], [6.0, 6.0, 0.0])]
    speeds = {seed: [a["agent"]["desired_velocity"] for a in _erratic(walker, world, base, seed=seed).doc["humans"] if a["name"].startswith("wander_")] for seed in (1, 2, 3)}
    assert speeds[1] and len({tuple(v) for v in speeds.values()}) > 1


def test_a_hand_authored_scenario_is_never_written_over(toy) -> None:
    from arena_simulation_setup.generative.scenario.materialize import WouldOverwrite

    worlds, doc, base, world = toy
    result = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    mine = worlds / "toy" / "scenarios" / result.scenario_id / "scenario.yaml"
    mine.parent.mkdir(parents=True)
    mine.write_text("format: arena-scenario/2\n")
    with pytest.raises(WouldOverwrite, match="not generated"):
        write_result(result, worlds / "toy")
    assert mine.read_text() == "format: arena-scenario/2\n" and sorted(p.name for p in mine.parent.iterdir()) == ["scenario.yaml"]


def test_a_rewrite_removes_the_type_files_the_case_no_longer_has(toy) -> None:
    worlds, doc, base, world = toy
    blackout = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    out = write_result(blackout, worlds / "toy")
    before = {p.name for p in (out / "types").iterdir()}
    stale = out / "types" / "adult__gone.yaml"
    stale.write_text("name: adult__gone\n")
    again = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert write_result(again, worlds / "toy") == out
    assert not stale.exists() and {p.name for p in (out / "types").iterdir()} == before


@pytest.mark.parametrize(
    ("text", "targets", "oracle", "step", "kind", "reason"),
    [
        ("An adult walks up to the robot and veers off.", ["C:approach"], {"C": {"case": "approach", "after": "veer", "veer_to": "nowhere"}}, "C", "placement", "approach/veer"),
        ("Nobody named ghost wanders.", ["B:erratic_route"], {"B": {"case": "erratic_route", "select": {"names": ["ghost"]}}}, "B", "applicability", "ghost"),
        ("Two nurses talk in the washroom.", ["D:interaction_unbroken"], {"D": {"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"zone": "wc_west"}}}, "D", "placement", "nobody is placed"),
    ],
)
def test_a_failure_is_recorded_with_the_kind_its_step_raised(toy, text: str, targets: list[str], oracle: dict, step: str, kind: str, reason: str) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt(text, targets, **oracle), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    (failure,) = result.record.failures
    assert (failure.step, failure.kind) == (step, kind) and reason in failure.reason and not failure.reason.startswith(f"{kind}:")


def test_a_model_answer_conflicting_twice_with_d_is_a_conflict(toy, tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen import step_bc
    from arena_simulation_setup.generative.scenario.promptgen.ask import system_prompt
    from arena_simulation_setup.generative.scenario.promptgen.spec import Record, conflict_m
    from arena_simulation_setup.generative.scenario.promptgen.step_a import StepError
    from arena_simulation_setup.generative.scenario.propose.client import DEFAULT_MODEL, prompt_sha256

    _worlds, doc, _base, world = toy
    text = "A visitor stands still right in the robot's way."
    record = Record(prompt_id="t_001", prompt=text, world=world.name, base="normal_a")
    user = step_bc.user_prompt(text, doc, world, record, taken_fractions=[0.5])
    complaint = f"conflict: C.route_fraction 0.5 is within {conflict_m(world.route_length):g} m of a Type D placement at 0.50"
    cache = tmp_path / "llm_cache"
    cache.mkdir()
    for asked in (user, f"{user}\n\nYour previous answer was rejected: {complaint}. Answer again."):
        (cache / f"{prompt_sha256(DEFAULT_MODEL, system_prompt('bc'), asked)}.json").write_text(json.dumps({"answer": json.dumps({"B": "none", "C": {"case": "blocking", "route_fraction": 0.5}})}))
    with pytest.raises(StepError, match="Type D placement") as caught:
        step_bc.decide(text, doc, world, record, taken_fractions=[0.5], cache_dir=cache, model=None, oracle=None)
    assert caught.value.kind == "conflict"


def test_provenance_hashes_each_prompt_and_answer(toy) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.ask import system_prompt
    from arena_simulation_setup.generative.scenario.promptgen.spec import Record
    from arena_simulation_setup.generative.scenario.promptgen.step_d import user_prompt
    from arena_simulation_setup.generative.scenario.propose.client import prompt_sha256

    worlds, doc, base, world = toy
    prompt = _prompt("Two nurses stand talking right in the robot's way.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    (asked,) = result.provenance["provenance"]["D"]
    user = user_prompt(prompt.text, doc, world, Record(prompt_id="t_001", prompt=prompt.text, world=world.name, base="normal_a"))
    assert asked["prompt_sha256"] == prompt_sha256(system_prompt("d"), user)
    assert set(asked) == {"model", "prompt_sha256", "response_sha256", "cached", "attempt"}
