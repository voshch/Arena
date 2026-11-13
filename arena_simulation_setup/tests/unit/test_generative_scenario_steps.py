from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_humansim.core.agents import loader
from arena_simulation_setup.generative.scenario.formats import read_scenario, scenario_text
from arena_simulation_setup.generative.scenario.promptgen import prompts as P
from arena_simulation_setup.generative.scenario.promptgen.pipeline import Result, generate, write_result
from arena_simulation_setup.generative.scenario.promptgen.world import WorldContext
from arena_simulation_setup.tree.World import Scenario as scenario_format

_LOBBY = [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]
_WARD = [[6.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 4.0]]
_HUNAV = {"behavior": 2, "goal_radius": 0.4}


def _world(tmp_path: Path) -> Path:
    root = tmp_path / "world"
    root.mkdir()
    zones = [{"name": "lobby", "polygon": _LOBBY, "floor": False}, {"name": "ward", "polygon": _WARD, "floor": False}]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    return root


def _rewrite(world: Path, name: str, doc: dict) -> Path:
    out = world / "scenarios" / name
    out.mkdir(parents=True)
    (out / "scenario.yaml").write_text(scenario_text(doc, world_root=world, types_dir=out))
    return out / "scenario.yaml"


def _human(**extra: object) -> dict:
    return {"name": "walker_1", "model": "Hospital/nurse_female_caucasian_young", "pose": "lobby", "agent": {"agent_type": "adult"}, "waypoints": ["lobby", "ward"], **extra}


def test_format_2_hunav_round_trips_without_nesting(tmp_path: Path) -> None:
    world = _world(tmp_path)
    base = world / "scenarios" / "base"
    base.mkdir(parents=True)
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": scenario_format.SCENARIO_FORMAT, "humans": [_human(hunav=_HUNAV)]}))

    path = _rewrite(world, "once", read_scenario(base / "scenario.yaml"))
    again = _rewrite(world, "twice", read_scenario(path))

    assert read_scenario(again)["humans"][0]["hunav"] == _HUNAV


def test_format_1_extra_human_keys_nest_once_across_rewrites(tmp_path: Path) -> None:
    world = _world(tmp_path)
    base = world / "scenarios" / "base"
    base.mkdir(parents=True)
    (base / "scenario.yaml").write_text(yaml.safe_dump({"robots": [{"start": [1.0, 1.0, 0.0], "goal": [8.0, 2.0, 0.0]}], "dynamic": [_human(**_HUNAV)]}))

    path = _rewrite(world, "once", read_scenario(base / "scenario.yaml"))
    again = _rewrite(world, "twice", read_scenario(path))

    assert read_scenario(again)["humans"][0]["hunav"] == _HUNAV


_ROOMS = {"west": (0.0, 0.0, 8.0, 8.0), "hall": (8.0, 3.0, 24.0, 5.0), "east": (24.0, 0.0, 32.0, 8.0)}
_START = [2.0, 4.0, 0.0]
_GOAL = [30.0, 4.0, 0.0]


def _walker(name: str, home: str, away: str, agent_type: str = "adult") -> dict:
    return {"name": name, "model": "Hospital/doctor_male_caucasian_middleage", "pose": home, "agent": {"agent_type": agent_type, "desired_velocity": 1.1}, "waypoints": [home, away], "waypoint_mode": "reverse"}


def _corridor_world(tmp_path: Path, names: dict[str, str], robot: dict, humans: list[dict]) -> tuple[WorldContext, dict, Path]:
    """Two rooms joined by a 2 m corridor, each zone named by `names` (a missing key leaves that area unzoned)."""
    root = tmp_path / "worlds" / "corridor"
    root.mkdir(parents=True)
    zones = [{"name": names[key], "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "floor": False} for key, (x0, y0, x1, y1) in _ROOMS.items() if key in names]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    base = root / "scenarios" / "base"
    base.mkdir(parents=True)
    doc = {"robots": [robot], "humans": humans}
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": scenario_format.SCENARIO_FORMAT, **doc}, sort_keys=False))
    return WorldContext.load(root, doc), doc, base


def _run(world: WorldContext, doc: dict, base: Path, text: str, targets: list[str], **oracle: object) -> Result:
    prompt = P.Prompt(id="t_001", text=text, targets=tuple(targets), oracle=dict(oracle))
    return generate(prompt, world=world, base_id="base", base_doc=doc, base_dir=base, oracle=True)


def test_a_failed_stampede_leaves_no_injected_pedestrians(tmp_path: Path) -> None:
    crowd = [_walker("p_1", "reception", "open_work_area"), _walker("p_2", "open_work_area", "reception"), {**_walker("idle", "reception", "reception"), "waypoints": []}]
    world, doc, base = _corridor_world(tmp_path, {"west": "reception", "hall": "central_hallway", "east": "open_work_area"}, {"start": _START, "phases": [{"goto": _GOAL}]}, crowd)

    result = _run(world, doc, base, "The break is over and the whole crowd rushes back to work all at once.", ["A:stampede"])

    assert any(f.step == "A" and f.kind == "applicability" and "without waypoints" in f.reason for f in result.record.failures), result.record.failures
    assert result.doc["humans"] == doc["humans"] and not result.files.docs


def test_a_failed_formation_returns_the_agents_it_reused(tmp_path: Path) -> None:
    crowd = [_walker("p_1", "reception", "open_work_area"), _walker("p_2", "open_work_area", "reception")]
    world, doc, base = _corridor_world(tmp_path, {"west": "reception", "hall": "bedroom", "east": "open_work_area"}, {"start": "reception", "phases": [{"goto": "open_work_area"}]}, crowd)
    assert not world.other_zones()

    decision = {"case": "interaction_unbroken", "variant": "meet", "members": ["nurse", "nurse"], "place": {"zone": "reception"}, "reuse": ["p_1"]}
    result = _run(world, doc, base, "Two nurses meet and chat in reception.", ["D:interaction_unbroken"], D=decision)

    assert any(f.step == "D" and "walk in from" in f.reason for f in result.record.failures), result.record.failures
    assert [h["name"] for h in result.doc["humans"]] == ["p_1", "p_2"]


def test_a_failed_intrinsic_group_keeps_earlier_groups_types(tmp_path: Path) -> None:
    crowd = [_walker("plain", "west", "east"), _walker("tree", "west", "east", agent_type="./needless.yaml")]
    world, doc, base = _corridor_world(tmp_path, {"west": "west", "hall": "hall", "east": "east"}, {"start": _START, "phases": [{"goto": _GOAL}]}, crowd)
    walk = {"kind": "go_to", "target_pose": {"x": 4.0, "y": 6.0, "theta": 0.0}, "patience": {"mean": 30.0}}
    (base / "needless.yaml").write_text(yaml.safe_dump({"name": "needless", "mode": "behavior_tree", "extends": "adult", "initial_sequence": "walk", "sequences": {"walk": {"steps": {"to_1": walk}, "then": "walk"}}}))

    decision = {"case": "intrinsic", "select": {"amount": "all"}, "params": {"radius": "high", "need_decay": "high"}}
    result = _run(world, doc, base, "Everyone here is wide and restless.", ["B:intrinsic"], B=decision)

    assert any(f.step == "B" and f.kind == "applicability" and "decay_rate" in f.reason for f in result.record.failures), result.record.failures
    assert [h["agent"]["agent_type"] for h in result.doc["humans"]] == ["adult", "./needless.yaml"]
    assert not [ref for ref in result.files.docs if "radius" in ref]


_TOY = {"west": "reception", "hall": "central_hallway", "east": "open_work_area"}


def _toy(tmp_path: Path) -> tuple[WorldContext, dict, Path]:
    crowd = [_walker("p_1", "reception", "open_work_area"), _walker("p_2", "open_work_area", "reception")]
    return _corridor_world(tmp_path, _TOY, {"start": _START, "phases": [{"goto": _GOAL}]}, crowd)


def _form_sequences(result: Result) -> list[dict]:
    return [result.files.docs[a["agent"]["agent_type"]]["sequences"] for a in result.doc["humans"] if a["name"].startswith("d_")]


@pytest.mark.parametrize(("duration", "seconds"), [("low", 10.0), ("mid", 30.0)])
def test_an_interaction_with_a_duration_ends_and_walks_on(tmp_path: Path, duration: str, seconds: float) -> None:
    world, doc, base = _toy(tmp_path)
    decision = {"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}, "duration": duration}
    result = _run(world, doc, base, "Two nurses stand talking in the robot's way for a while.", ["D:interaction_unbroken"], D=decision)

    assert result.record.status == "ok", result.record.failures
    sequences = _form_sequences(result)
    assert len(sequences) == 2
    for seq in sequences:
        assert seq["form"]["steps"]["stand"]["duration"] == {"mean": seconds}
        assert seq["form"]["steps"]["stand"]["satisfies"] == {"form": -100.0} and seq["form"]["then"] == "walk"


def test_an_interaction_of_high_duration_holds_its_formation(tmp_path: Path) -> None:
    world, doc, base = _toy(tmp_path)
    decision = {"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}, "duration": "high"}
    result = _run(world, doc, base, "Two nurses stand talking in the robot's way and will not stop.", ["D:interaction_unbroken"], D=decision)

    for seq in _form_sequences(result):
        assert "satisfies" not in seq["form"]["steps"]["stand"] and seq["form"]["then"] == "form"


@pytest.mark.parametrize(("when", "entry"), [({"at": 20.0}, {"at": 20.0}), ({"robot_within": 4.0}, None)])
def test_a_meeting_starts_when_the_prompt_says(tmp_path: Path, when: dict, entry: dict | None) -> None:
    world, doc, base = _toy(tmp_path)
    decision = {"case": "interaction_unbroken", "variant": "meet", "members": ["nurse", "nurse"], "place": {"zone": "open_work_area"}, "when": when, "duration": "mid"}
    result = _run(world, doc, base, "Two nurses meet and chat for half a minute.", ["D:interaction_unbroken"], D=decision)

    assert result.record.status == "ok", result.record.failures
    members = [a for a in result.doc["humans"] if a["name"].startswith("d_")]
    assert all(len(a["waypoints"]) == 1 for a in members)
    (stimulus,) = result.timeline
    assert stimulus["stimulus"] == [{"agents": sorted(a["name"] for a in members), "need": "form"}]
    if entry is not None:
        assert {k: v for k, v in stimulus.items() if k != "stimulus"} == entry
    else:
        assert "within" in stimulus["when"] and "robot" in stimulus["when"]
    for seq in _form_sequences(result):
        assert seq["form"]["steps"]["stand"]["duration"] == {"mean": 30.0} and seq["form"]["then"] == "walk"

    out = result.files
    out.write(tmp_path / "out")
    loaded = loader.load_agent_type_from_file(tmp_path / "out" / members[0]["agent"]["agent_type"][2:])
    assert loaded.mode == "behavior_tree" and set(loaded.sequences) == {"walk", "form"}


def test_the_shipped_shift_change_prompt_applies_every_step(tmp_path: Path) -> None:
    world, doc, base = _toy(tmp_path)
    prompt = next(p for p in P.read(P.shipped_dir() / "universal.yaml") if p.id == "v_017")

    result = generate(prompt, world=world, base_id="base", base_doc=doc, base_dir=base, oracle=True)

    assert result.record.status == "ok", result.record.failures
    assert sorted(result.record.steps) == sorted(prompt.targets)


def test_an_unreached_erratic_point_gives_way_to_the_next(tmp_path: Path) -> None:
    world, doc, base = _toy(tmp_path)
    decision = {"case": "erratic_route", "select": {"amount": "all"}, "period": 3.0}
    result = _run(world, doc, base, "Everyone keeps changing where they are going.", ["B:erratic_route"], B=decision)

    assert result.record.status == "ok", result.record.failures
    agent = next(a for a in result.doc["humans"] if a["name"] == "p_1")
    sequences = result.files.docs[agent["agent"]["agent_type"]]["sequences"]
    chain = [name for name in sequences if name.startswith("erratic")]
    assert len(chain) >= 2 and chain[0] == "erratic"
    for here, there in zip(chain, [*chain[1:], chain[0]], strict=True):
        assert len(sequences[here]["steps"]) == 1
        assert sequences[here]["then"] == sequences[here]["on_failure"] == there
    assert all({"when": {"erratic": {"above": 50.0}}, "goto": "erratic"} not in sequences[n].get("transitions", []) for n in chain)

    result.files.write(tmp_path / "out")
    loaded = loader.load_agent_type_from_file(tmp_path / "out" / agent["agent"]["agent_type"][2:])
    assert all(loaded.sequences[here].on_failure == there for here, there in zip(chain, [*chain[1:], chain[0]], strict=True))


def test_a_baked_radius_replaces_the_radius_the_base_pinned(tmp_path: Path) -> None:
    crowd = [{**_walker(name, home, away), "agent": {"agent_type": "adult", "desired_velocity": 1.1, "radius": 0.25}} for name, home, away in (("p_1", "reception", "open_work_area"), ("p_2", "open_work_area", "reception"))]
    world, doc, base = _corridor_world(tmp_path, _TOY, {"start": _START, "phases": [{"goto": _GOAL}]}, crowd)
    decision = {"case": "intrinsic", "select": {"names": ["p_1"]}, "params": {"radius": "high"}}
    result = _run(world, doc, base, "One of them is unusually wide.", ["B:intrinsic"], B=decision)
    assert result.record.status == "ok", result.record.failures

    out = write_result(result, world.root)

    radius = {h["name"]: h["agent"]["radius"] for h in read_scenario(out / "scenario.yaml")["humans"]}
    assert radius == {"p_1": 0.4, "p_2": 0.25}


def test_an_erratic_route_naming_a_missing_agent_is_not_applicable(tmp_path: Path) -> None:
    world, doc, base = _toy(tmp_path)
    decision = {"case": "erratic_route", "select": {"names": ["ghost"]}}
    result = _run(world, doc, base, "Ghost wanders about with no settled destination.", ["B:erratic_route"], B=decision)

    assert any(f.step == "B" and f.kind == "applicability" and "ghost" in f.reason for f in result.record.failures), result.record.failures
    assert "B" not in result.record.owned and not [h for h in result.doc["humans"] if h["name"].startswith("wander_")]


def test_an_erratic_route_over_an_owned_crowd_adds_wanderers(tmp_path: Path) -> None:
    world, doc, base = _toy(tmp_path)
    prompt = "The break is over and the whole crowd rushes back to work all at once. Some people wander about with no settled destination."
    result = _run(world, doc, base, prompt, ["A:stampede", "B:erratic_route"], B={"case": "erratic_route", "select": {"amount": "half"}})

    assert result.record.status == "ok", result.record.failures
    assert result.record.owned["B"] and all(n.startswith("wander_") for n in result.record.owned["B"])


def test_a_scenario_the_model_rejects_leaves_no_type_files(tmp_path: Path) -> None:
    world, doc, base = _toy(tmp_path)
    result = _run(world, doc, base, "The lights go out and everyone slows down.", ["A:blackout"])
    assert result.record.status == "ok" and result.files.docs, result.record.failures
    result.doc["humans"][0]["pose"] = "kitchen"

    with pytest.raises(ValueError, match="kitchen"):
        write_result(result, world.root)

    assert not (world.root / "scenarios" / result.scenario_id).exists()
