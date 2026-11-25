"""A scenario lane is bounded by the walls of the level that holds its source zone."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_humansim")

from arena_simulation_setup.tree.World import Level, WallPath, Zone
from arena_simulation_setup.tree.World.lane_check import build_scenario_lanes, scenario_lane_results
from arena_simulation_setup.tree.World.model import World
from arena_simulation_setup.tree.World.parse import parse_world

REGIONS = {
    "east": {
        "type": "source",
        "ref": "start",
        "config": {
            "agent": {"sink_affinity": [{"sink": "west", "weight": 1.0}]},
            "lane": {"side": "right", "segments": [{"centerline": [[0.0, 0.0], [20.0, 0.0]]}]},
        },
    },
    "west": {"type": "sink", "ref": "goal", "config": {}},
}


def _level(level_id: str, walls: list[WallPath]) -> Level:
    zones = (
        [
            Zone(name="start", polygon=[[0, -2], [2, -2], [2, 2], [0, 2]]),
            Zone(name="goal", polygon=[[16, -2], [22, -2], [22, 2], [16, 2]]),
        ]
        if level_id == "1"
        else []
    )
    return Level(id=level_id, zones=zones, walls=walls)


CORRIDOR = [WallPath(path=[[0, -2], [20, -2]]), WallPath(path=[[0, 2], [20, 2]])]
BLOCKER = [WallPath(path=[[10, -5], [10, 5]])]


def test_scenario_lane_ignores_the_walls_of_other_levels() -> None:
    alone = World.from_levels(_level("1", CORRIDOR))
    stacked = World.from_levels(_level("0", BLOCKER), _level("1", CORRIDOR))
    assert scenario_lane_results(REGIONS, alone)["east"]["valid"] is True
    assert scenario_lane_results(REGIONS, stacked) == scenario_lane_results(REGIONS, alone)
    assert build_scenario_lanes(REGIONS, stacked)[0].keys() == {"east"}


def test_scenario_lane_is_bounded_by_the_walls_of_its_own_level() -> None:
    blocked = World.from_levels(_level("0", CORRIDOR), _level("1", BLOCKER))
    open_level = World.from_levels(_level("0", CORRIDOR), _level("1", CORRIDOR))
    assert scenario_lane_results(REGIONS, blocked) != scenario_lane_results(REGIONS, open_level)


CABIN_WORLD = {
    "format": "arena-world/2",
    "levels": [
        {
            "id": "0",
            "zones": [
                {"name": "src", "polygon": [[0, 0], [2, 0], [2, 4], [0, 4]]},
                {"name": "mid", "polygon": [[2, 0], [12, 0], [12, 4], [2, 4]]},
                {"name": "cab_zone", "polygon": [[6, 4], [8, 4], [8, 6], [6, 6]], "floor": False, "ceiling": False},
            ],
            "walls": [{"path": [[6.0, 4.0], [0.0, 4.0], [0.0, 0.0], [14.0, 0.0], [14.0, 4.0], [8.0, 4.0]]}],
            "elevators": [{"name": "cab", "position": [7.0, 5.0], "door_side": "-y"}],
        }
    ],
}
INTO_CABIN = {
    "in": {
        "type": "source",
        "ref": "src",
        "config": {
            "agent": {"sink_affinity": [{"sink": "cabin", "weight": 1.0}]},
            "lane": {"side": "right", "spawn_length": 0.5, "segments": [{"centerline": [[1.0, 2.0], [7.0, 2.0], [7.0, 5.0]]}]},
        },
    },
    "cabin": {"type": "sink", "ref": "cab_zone", "config": {}},
}


def test_a_lane_into_an_elevator_cabin_is_bounded_by_the_cabin_walls() -> None:
    without_cabin = {**CABIN_WORLD, "levels": [{**CABIN_WORLD["levels"][0], "elevators": []}]}
    refused = scenario_lane_results(INTO_CABIN, parse_world(without_cabin))["in"]
    assert refused["valid"] is False
    assert "no wall on the open side" in refused["reason"]
    assert scenario_lane_results(INTO_CABIN, parse_world(CABIN_WORLD))["in"]["valid"] is True
