from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.formats import first_level, read_scenario, scenario_text
from arena_simulation_setup.tree.World import legacy
from arena_simulation_setup.tree.World import Scenario as scenario_format

_LEVEL = {
    "zones": [
        {
            "name": "lobby",
            "corners": [{"x": 0.0, "y": 0.0}, {"x": 6.0, "y": 0.0}, {"x": 6.0, "y": 4.0}, {"x": 0.0, "y": 4.0}],
            "entities": {"static": [{"name": "bench", "model": "Hospital/Object/Waiting_Room_Bench", "pose": {"position": {"x": 2.0, "y": 1.0, "z": 0.0}, "orientation": {"w": math.cos(0.35), "x": 0.0, "y": 0.0, "z": math.sin(0.35)}}}]},
        },
        {
            "name": "ward",
            "corners": [{"x": 6.0, "y": 0.0}, {"x": 10.0, "y": 0.0}, {"x": 10.0, "y": 4.0}, {"x": 6.0, "y": 4.0}],
            "entities": {"static": [{"name": "bed", "model": "Hospital/Object/SM_HospitalBed_02d", "pose": {"position": {"x": 8.0, "y": 2.0, "z": 0.0}, "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}}]},
        },
    ],
}
_MAP = {"image": "map.png", "resolution": 0.05, "origin": [0.0, 0.0, 0.0], "occupied_thresh": 0.65, "free_thresh": 0.2, "negate": 0}


def _shape(level: dict) -> list[tuple]:
    out = []
    for zone in level["zones"]:
        statics = []
        for ent in zone["entities"]["static"]:
            statics.append((ent["name"], ent["model"], round(ent["pose"][0], 6), round(ent["pose"][1], 6)))
        out.append((zone["name"], [(c["x"], c["y"]) for c in zone["corners"]], statics))
    return out


def test_a_format_2_world_reads_like_the_format_1_world_it_was_migrated_from(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    (v1 / "0").mkdir(parents=True)
    (v1 / "0" / "world.yaml").write_text(yaml.safe_dump(_LEVEL))
    (v1 / "0" / "map.yaml").write_text(yaml.safe_dump(_MAP))
    world, _kg, _warnings = legacy.upgrade({"0": _LEVEL})
    v2 = tmp_path / "v2"
    (v2 / "map").mkdir(parents=True)
    (v2 / "world.yaml").write_text(yaml.safe_dump(world))
    (v2 / "map" / "0.yaml").write_text(yaml.safe_dump(_MAP))

    assert _shape(first_level(v2)) == _shape(first_level(v1))


def _world(tmp_path: Path) -> Path:
    root = tmp_path / "world"
    root.mkdir()
    zones = [{"name": z["name"], "polygon": [[c["x"], c["y"]] for c in z["corners"]], "floor": False} for z in _LEVEL["zones"]]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    return root


_DOC = {
    "robots": [{"start": [1.0, 1.0, 0.0], "phases": [{"goto": [8.0, 2.0], "tolerance_radius": 0.6}]}],
    "humans": [{"name": "walker_1", "model": "Hospital/nurse_female_caucasian_young", "pose": "lobby", "agent": {"agent_type": "adult", "desired_velocity": 1.1}, "waypoints": ["lobby", "ward"], "waypoint_mode": "reverse"}],
}


def test_a_scenario_is_written_as_format_2_the_scenario_model_reads(tmp_path: Path) -> None:
    world = _world(tmp_path)
    path = world / "scenarios" / "s" / "scenario.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(scenario_text(_DOC, world_root=world, types_dir=path.parent, header="# header\n"))

    assert path.read_text().startswith("# header\n")
    raw = yaml.safe_load(path.read_text())
    assert raw["format"] == scenario_format.SCENARIO_FORMAT and list(raw) == ["format", "humans", "robots"]
    assert raw["humans"][0]["pose"] == "lobby" and raw["humans"][0]["agent"]["radius"] > 0
    assert read_scenario(path)["robots"] == _DOC["robots"]
    loaded = scenario_format.parse_scenario(raw, _zone_converter(world))
    assert [h.name for h in loaded.humans] == ["walker_1"] and loaded.robots[0].phases[0].tolerance_radius == 0.6


def _zone_converter(world: Path) -> object:
    import numpy as np

    from arena_simulation_setup.tree.World import WorldView

    return WorldView(world).load().compact_world({"0": (0.0, 0.0)}).zone_converter(np.random.default_rng(0))


def test_a_format_1_scenario_reads_as_format_2(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump({"robots": [{"start": [1.0, 1.0, 0.0], "goal": [8.0, 2.0, 0.0]}], "dynamic": _DOC["humans"]}))
    doc = read_scenario(path)
    assert doc["robots"] == [{"start": [1.0, 1.0, 0.0], "phases": [{"goto": [8.0, 2.0, 0.0]}]}] and doc["humans"][0]["name"] == "walker_1"


def test_a_key_format_2_does_not_know_refuses_to_write(tmp_path: Path) -> None:
    world = _world(tmp_path)
    with pytest.raises(ValueError, match="does not know"):
        scenario_text({"humans": [], "knobs": {"speed": "high"}}, world_root=world, types_dir=world / "scenarios" / "s")


def test_a_scenario_the_model_rejects_is_refused(tmp_path: Path) -> None:
    world = _world(tmp_path)
    with pytest.raises(ValueError, match="kitchen"):
        scenario_text({**_DOC, "humans": [{**_DOC["humans"][0], "pose": "kitchen"}]}, world_root=world, types_dir=world / "scenarios" / "s")
