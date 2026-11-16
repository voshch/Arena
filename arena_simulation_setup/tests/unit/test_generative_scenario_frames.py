from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.formats import read_scenario
from arena_simulation_setup.generative.scenario.promptgen import prompts as P
from arena_simulation_setup.generative.scenario.promptgen.pipeline import generate, write_result
from arena_simulation_setup.generative.scenario.promptgen.world import WorldContext
from arena_simulation_setup.tree.World.view import WorldView

pytest.importorskip("numpy")

_DX, _DY = 10.0, 20.0
_ROOMS = {"reception": (0, 0, 8, 8), "central_hallway": (8, 3, 24, 5), "open_work_area": (24, 0, 32, 8)}
_WITHIN = re.compile(r"robot within (\S+) of (\S+),(\S+)")


def _zones(rooms: dict, dx: float, dy: float) -> list[dict]:
    return [{"name": name, "polygon": [[x0 + dx, y0 + dy], [x1 + dx, y0 + dy], [x1 + dx, y1 + dy], [x0 + dx, y1 + dy]], "floor": False} for name, (x0, y0, x1, y1) in rooms.items()]


def _two_levels(tmp_path: Path) -> tuple[Path, dict, Path]:
    """A two-level world whose working level 0 starts at (10, 20), a base with a robot and a walker on it."""
    root = tmp_path / "worlds" / "stacked"
    root.mkdir(parents=True)
    levels = [{"id": "0", "zones": _zones(_ROOMS, _DX, _DY)}, {"id": "1", "zones": _zones({"attic": (0, 0, 6, 6)}, 0.0, 0.0)}]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": levels}))
    origin = WorldView(root.resolve()).level_origins()["0"]
    walker = {"name": "walker", "model": "Hospital/doctor_male_caucasian_middleage", "pose": [4.0 + _DX + origin[0], 6.0 + _DY + origin[1], 0.0], "agent": {"agent_type": "adult", "desired_velocity": 1.1}, "waypoints": ["reception", "open_work_area"], "waypoint_mode": "reverse"}
    doc = {"robots": [{"start": [2.0 + _DX, 4.0 + _DY, 0.0], "level": "0", "phases": [{"goto": [30.0 + _DX, 4.0 + _DY, 0.0]}]}], "humans": [walker]}
    base = root / "scenarios" / "normal"
    base.mkdir(parents=True)
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", **doc}, sort_keys=False))
    return root, doc, base


def test_a_multi_level_case_writes_map_frame_points_and_level_frame_robot_poses(tmp_path: Path) -> None:
    root, doc, base = _two_levels(tmp_path)
    world = WorldContext.load(root, doc)
    origin = WorldView(root.resolve()).level_origins()["0"]
    assert origin == pytest.approx((-_DX, -_DY)) and world.origin == pytest.approx(origin)
    prompt = P.Prompt(id="t_001", text="An adult walks straight at the robot.", targets=("C:approach",), oracle={"C": {"case": "approach", "profile": "adult", "angle": 180, "route_fraction": 0.5}})

    result = generate(prompt, world=world, base_id="normal", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    out = write_result(result, root)
    written = read_scenario(out / "scenario.yaml")
    provenance = yaml.safe_load((out / "provenance.yaml").read_text())

    (robot,) = written["robots"]
    assert robot["level"] == robot["goal_level"] == "0"
    assert robot["start"][:2] == pytest.approx([round(world.start[0], 2), round(world.start[1], 2)])
    assert robot["phases"][-1]["goto"] == pytest.approx([round(world.goal[0], 2), round(world.goal[1], 2)]) and world.zone_of(world.goal) is not None

    humans = {h["name"]: h for h in written["humans"]}
    assert humans["walker"]["pose"] == doc["humans"][0]["pose"]
    edge = humans["edge_0"]
    spawn = (edge["pose"][0] - origin[0], edge["pose"][1] - origin[1])
    assert world.zone_of(spawn) is not None and world.valid(spawn, on_route=True) is None

    walk = yaml.safe_load((out / edge["agent"]["agent_type"]).read_text())["sequences"]["walk"]["steps"]["cross"]["target_pose"]
    assert world.zone_of((walk["x"] - origin[0], walk["y"] - origin[1])) is not None

    (entry,) = written["timeline"]
    _radius, x, y = (float(v) for v in _WITHIN.fullmatch(entry["when"]).groups())
    assert world.route_clearance((x - origin[0], y - origin[1])) < 0.05

    (encounter,) = provenance["encounters"]
    assert world.route_clearance(tuple(encounter["point"])) < 0.05 and world.zone_of(tuple(encounter["point"])) is not None
