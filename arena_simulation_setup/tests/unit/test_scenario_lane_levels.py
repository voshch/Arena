"""A scenario lane is bounded by the walls of the level that holds its source zone."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_humansim")

from arena_simulation_setup.tree.World import Level, WallPath, Zone
from arena_simulation_setup.tree.World.lane_check import build_scenario_lanes, scenario_lane_results
from arena_simulation_setup.tree.World.model import World

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
