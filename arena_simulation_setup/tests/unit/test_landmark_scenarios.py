from __future__ import annotations

import math
from pathlib import Path

import pytest
import shapely
import yaml

from arena_simulation_setup.shared.route import instruct, portals, zone_shapes
from arena_simulation_setup.shared.task import GoToPhase, TaskPhase
from arena_simulation_setup.tree.World.World import LevelDescription
from arena_simulation_setup.utils.cattrs import converter

WORLDS = Path(__file__).parents[2] / "worlds"
SCENARIOS = sorted(str(path.relative_to(WORLDS)) for path in WORLDS.glob("*/scenarios/landmark_*/scenario.yaml"))


def _level(world: str) -> LevelDescription:
    return converter.structure(yaml.safe_load((WORLDS / world / "0" / "world.yaml").read_text()), LevelDescription)


def _robots(scenario: str) -> list[tuple[tuple[float, float, float], list[GoToPhase]]]:
    robots = yaml.safe_load((WORLDS / scenario).read_text())["robots"]
    return [(tuple(robot["start"]), [TaskPhase.parse(dict(phase)) for phase in robot["phases"]]) for robot in robots]


def test_landmark_scenarios_ship_for_both_furnished_worlds():
    assert {scenario.split("/")[0] for scenario in SCENARIOS} == {"hospital_1", "office_1"}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_start_lies_in_a_zone(scenario):
    shapes = zone_shapes(_level(scenario.split("/")[0]))
    for start, _ in _robots(scenario):
        assert any(shape.covers(shapely.Point(start[:2])) for shape in shapes.values())


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_leg_names_a_zone_holds_a_pose_deeper_inside_than_its_tolerance_and_gets_walking_directions(scenario):
    level = _level(scenario.split("/")[0])
    shapes = zone_shapes(level)
    found = portals(level)
    for start, phases in _robots(scenario):
        here = start
        for phase in phases:
            assert phase.target in shapes
            assert shapes[phase.target].covers(shapely.Point(phase.pose.position.x, phase.pose.position.y))
            assert not shapes[phase.target].covers(shapely.Point(here[:2]))
            goal = (phase.pose.position.x, phase.pose.position.y)
            assert all(math.dist(goal, portal.mid) > phase.tolerance_radius for portal in found if phase.target in portal.zones)
            assert instruct(phase, level, here, "route", found=found)["source"] == "route"
            here = phase.pose.to_2d()
