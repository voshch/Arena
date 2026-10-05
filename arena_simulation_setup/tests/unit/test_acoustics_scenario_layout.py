from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from arena_simulation_setup.acoustics import scenario_layout

WORLDS_ROOT = Path(__file__).parents[2] / "acoustics" / "worlds"

pytestmark = pytest.mark.skipif(not WORLDS_ROOT.is_dir(), reason="generated acoustics worlds are not present")


def _matrix(world: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    root = WORLDS_ROOT / world / "scenarios"
    result = {}
    for path in root.glob("hearing__*/scenario.yaml"):
        match = scenario_layout.NAME_PATTERN.fullmatch(path.parent.name)
        assert match is not None
        key = (match.group("robot"), int(match.group("count")), match.group("direction"))
        result[key] = scenario_layout._load_yaml(path)
    return result


def test_direction_changes_humans_without_moving_robot() -> None:
    world_dir = WORLDS_ROOT / "straight_corridor_O"
    matrix = _matrix(world_dir.name)
    route = scenario_layout._canonical_route(matrix)
    frame, safe = scenario_layout._load_safe_cells(world_dir)

    a_to_b = scenario_layout._render(route, "idle", 3, "a-to-b", frame, safe)
    b_to_a = scenario_layout._render(route, "idle", 3, "b-to-a", frame, safe)
    moving_a_to_b = scenario_layout._render(route, "moving", 3, "a-to-b", frame, safe)
    moving_b_to_a = scenario_layout._render(route, "moving", 3, "b-to-a", frame, safe)

    assert a_to_b["robots"][0] == b_to_a["robots"][0]
    assert moving_a_to_b["robots"][0] == moving_b_to_a["robots"][0]
    assert math.dist(a_to_b["dynamic"][0]["pose"][:2], route[0]) < 2.0
    assert math.dist(b_to_a["dynamic"][0]["pose"][:2], route[-1]) < 1.0


def test_three_humans_are_separate_and_have_immediate_moving_targets() -> None:
    world_dir = WORLDS_ROOT / "straight_corridor_O"
    matrix = _matrix(world_dir.name)
    route = scenario_layout._canonical_route(matrix)
    frame, safe = scenario_layout._load_safe_cells(world_dir)
    scenario = scenario_layout._render(route, "moving", 3, "a-to-b", frame, safe)

    people = scenario["dynamic"]
    assert len({tuple(person["pose"][:2]) for person in people}) == 3
    assert all(math.dist(left["pose"][:2], right["pose"][:2]) >= 2 * scenario_layout.PEDESTRIAN_CLEARANCE_M for left, right in zip(people, people[1:], strict=False))
    assert all(math.dist(left["waypoints"][-1][:2], right["waypoints"][-1][:2]) >= 2 * scenario_layout.PEDESTRIAN_CLEARANCE_M for left, right in zip(people, people[1:], strict=False))
    assert all(person["velocity"] > 0.0 for person in people)
    assert all(person["agent"]["desired_velocity"] > 0.0 for person in people)
    assert all(person["model"] == "arenian" for person in people)
    assert all(person["waypoint_mode"] == "reverse" for person in people)
    assert all(math.dist(person["pose"][:2], person["waypoints"][0][:2]) >= 0.10 for person in people)


def test_rendering_an_already_normalized_matrix_is_stable() -> None:
    world_dir = WORLDS_ROOT / "straight_corridor_O"
    matrix = _matrix(world_dir.name)
    route = scenario_layout._canonical_route(matrix)
    frame, safe = scenario_layout._load_safe_cells(world_dir)
    normalized = {key: scenario_layout._render(route, key[0], key[1], key[2], frame, safe) for key in matrix}

    normalized_route = scenario_layout._canonical_route(normalized)
    rerendered = {key: scenario_layout._render(normalized_route, key[0], key[1], key[2], frame, safe) for key in normalized}
    assert normalized == rerendered


def test_all_acoustics_scenarios_use_bundled_pedestrian_model() -> None:
    for world_dir in WORLDS_ROOT.iterdir():
        if not world_dir.is_dir():
            continue
        result = scenario_layout.normalize_world_models(world_dir, write=False)
        assert result.changed == 0, world_dir.name
