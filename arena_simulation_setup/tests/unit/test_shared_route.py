from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.shared.route import crossings, describe_route, instruct, level_nouns, portals, zone_shapes
from arena_simulation_setup.shared.task import TaskPhase
from arena_simulation_setup.tree.World.World import LevelDescription
from arena_simulation_setup.utils.cattrs import converter

WORLDS = Path(__file__).parents[2] / "worlds"
NORTH = math.pi / 2.0
EAST = 0.0


def _point(x: float, y: float) -> dict:
    return {"x": x, "y": y, "z": 0.0}


def _walls(*segments: tuple[float, float, float, float]) -> list[dict]:
    return [{"start": _point(x0, y0), "end": _point(x1, y1)} for x0, y0, x1, y1 in segments]


def _box(x0: float, y0: float, x1: float, y1: float) -> list[dict]:
    return [_point(x0, y0), _point(x1, y0), _point(x1, y1), _point(x0, y1)]


def _level() -> LevelDescription:
    """A hall under an office (door, 5 cm above the hall) and a lounge (2 m wall opening), plus a walled-off vault."""
    return converter.structure(
        {
            "zones": [
                {"name": "hall", "corners": _box(0.0, 0.0, 12.0, 2.0), "walls": _walls((0.0, 2.0, 2.0, 2.0), (3.0, 2.0, 7.0, 2.0), (9.0, 2.0, 12.0, 2.0))},
                {
                    "name": "office",
                    "description": "Head Office",
                    "corners": _box(0.0, 2.05, 5.0, 8.0),
                    "walls": _walls((0.0, 2.05, 2.0, 2.05), (3.0, 2.05, 5.0, 2.05), (5.0, 2.05, 5.0, 8.0)),
                    "doors": [{"name": "office_door", "start": _point(2.0, 2.05), "end": _point(3.0, 2.05)}],
                    "entities": {"static": [{"name": "office_desk_1", "model": "desk", "pose": {"position": _point(2.5, 7.0)}}]},
                },
                {"name": "lounge", "corners": _box(5.0, 2.0, 12.0, 8.0)},
                {"name": "vault", "corners": _box(12.0, 0.0, 16.0, 8.0), "walls": _walls((12.0, 0.0, 12.0, 8.0))},
            ]
        },
        LevelDescription,
    )


def test_portals_are_the_door_and_the_wall_opening():
    found = {(portal.zones, portal.door): portal for portal in portals(_level())}
    assert set(found) == {(("office", "hall"), True), (("hall", "lounge"), False)}
    assert found[(("office", "hall"), True)].mid == pytest.approx((2.5, 2.05))
    opening = found[(("hall", "lounge"), False)]
    assert opening.mid == pytest.approx((8.0, 2.0))
    assert math.dist(opening.start, opening.end) == pytest.approx(2.0 - 2 * 0.15)


def test_crossings_chain_through_the_hall():
    level = _level()
    chain = crossings(portals(level), (2.5, 6.0), "office", (8.0, 6.0), "lounge")
    assert [(portal.door, entered) for portal, entered in chain] == [(True, "hall"), (False, "lounge")]
    assert crossings(portals(level), (1.0, 1.0), "hall", (11.0, 1.0), "hall") == []
    assert crossings(portals(level), (1.0, 1.0), "hall", (14.0, 4.0), "vault") is None


@pytest.mark.parametrize(
    ("start", "yaw", "goal", "text"),
    [
        ((10.0, 1.0), EAST, (2.5, 6.5), "Turn around and walk through the hall, then take the door on your right into the head office. Walk about 4 meters into the head office and stop next to the office desk."),
        ((2.5, 1.0), NORTH, (2.5, 2.6), "Go through the door ahead into the head office. Stop just inside the head office."),
        ((1.0, 1.0), EAST, (8.0, 6.0), "Walk through the hall, then turn left into the lounge. Walk about 4 meters into the lounge and stop."),
        ((1.0, 1.0), NORTH, (9.0, 1.0), "Turn right and walk about 8 meters and stop."),
        ((1.0, 1.0), EAST, (1.5, 1.0), "Stop here."),
        ((2.5, 7.5), NORTH, (8.0, 2.5), "Turn around and walk through the head office, then go through the door ahead into the hall. Turn left and walk through the hall, then turn left into the lounge. Stop just inside the lounge."),
    ],
)
def test_describe_route(start, yaw, goal, text):
    assert describe_route(_level(), start, yaw, goal) == text


def _corridor() -> LevelDescription:
    """A hall with three rooms above it and one below, each behind its own door."""
    rooms = [("archive", 0.0, 6.0), ("kitchen", 6.0, 12.0), ("studio", 12.0, 18.0)]
    zones = [{"name": "hall", "corners": _box(0.0, 0.0, 18.0, 2.0)}]
    for name, x0, x1 in rooms:
        mid = (x0 + x1) / 2.0
        zones.append({"name": name, "corners": _box(x0, 2.0, x1, 8.0), "walls": _walls((x0, 2.0, mid - 0.5, 2.0), (mid + 0.5, 2.0, x1, 2.0), (x0, 2.0, x0, 8.0), (x1, 2.0, x1, 8.0)), "doors": [{"name": f"{name}_door", "start": _point(mid - 0.5, 2.0), "end": _point(mid + 0.5, 2.0)}]})
    zones.append({"name": "garage", "corners": _box(0.0, -6.0, 18.0, 0.0), "walls": _walls((0.0, 0.0, 8.5, 0.0), (9.5, 0.0, 18.0, 0.0)), "doors": [{"name": "garage_door", "start": _point(8.5, 0.0), "end": _point(9.5, 0.0)}]})
    return converter.structure({"zones": zones}, LevelDescription)


@pytest.mark.parametrize(
    ("start", "yaw", "goal", "crossing"),
    [
        ((1.0, 1.0), EAST, (9.0, 5.0), "take the second door on your left into the kitchen"),
        ((1.0, 1.0), EAST, (15.0, 5.0), "take the third door on your left into the studio"),
        ((17.0, 1.0), math.pi, (9.0, 5.0), "take the second door on your right into the kitchen"),
        ((17.0, 1.0), math.pi, (3.0, 5.0), "take the third door on your right into the archive"),
        ((4.0, 1.0), EAST, (9.0, 5.0), "take the first door on your left into the kitchen"),
        ((10.0, 1.0), EAST, (15.0, 5.0), "take the door on your left into the studio"),
        ((1.0, 1.0), EAST, (9.0, -3.0), "take the door on your right into the garage"),
    ],
)
def test_describe_route_counts_the_doors_passed_on_that_side(start, yaw, goal, crossing):
    assert describe_route(_corridor(), start, yaw, goal).split(". ")[0] == f"Walk through the hall, then {crossing}"


def test_describe_route_takes_authored_nouns():
    text = describe_route(_level(), (1.0, 1.0), EAST, (8.0, 6.0), {"lounge": "the staff lounge", "hall": "the corridor"})
    assert text == "Walk through the corridor, then turn left into the staff lounge. Walk about 4 meters into the staff lounge and stop."


def test_describe_route_is_none_off_the_zones_and_across_walls():
    assert describe_route(_level(), (1.0, 1.0), EAST, (14.0, 4.0)) is None
    assert describe_route(_level(), (1.0, 1.0), EAST, (30.0, 30.0)) is None
    assert describe_route(_level(), (-5.0, 1.0), EAST, (1.0, 1.0)) is None


def test_level_nouns_use_descriptions_and_drop_indices():
    nouns = level_nouns(_level())
    assert nouns["office"] == "the head office"
    assert nouns["hall"] == "the hall"
    assert nouns["office_door"] == "the office door"
    assert nouns["office_desk_1"] == "the office desk"
    assert level_nouns(_level(), "office_")["office_desk_1"] == "the desk"
    assert level_nouns(_level(), "office_")["office_door"] == "the door"


HERE = (1.0, 1.0, EAST)
TO_LOUNGE = "Walk through the hall, then turn left into the lounge. Walk about 4 meters into the lounge and stop."


@pytest.mark.parametrize(
    ("phase", "here", "wording", "told"),
    [
        ({"goto": [8.0, 6.0, 0.0], "text": "Find the sofa."}, HERE, "route", {"source": "authored", "text": "Find the sofa."}),
        ({"goto": [8.0, 6.0, 0.0]}, HERE, "", {"source": "route", "text": TO_LOUNGE}),
        ({"goto": [8.0, 6.0, 0.0]}, HERE, "goal", {"source": "coordinates", "text": "Go to (8.0, 6.0)."}),
        ({"goto": [8.0, 6.0, 0.0]}, None, "", {"source": "coordinates", "text": "Go to (8.0, 6.0)."}),
        ({"goto": [14.0, 4.0, 0.0]}, HERE, "route", {"source": "coordinates", "text": "Go to (14.0, 4.0)."}),
        ({"goto": "lounge", "pose": [8.0, 6.0, 0.0]}, HERE, "", {"source": "target", "text": "Go to the lounge."}),
        ({"goto": "lounge", "pose": [8.0, 6.0, 0.0]}, HERE, "route", {"source": "route", "text": TO_LOUNGE}),
        ({"goto": "lounge"}, HERE, "route", {"source": "target", "text": "Go to the lounge."}),
    ],
)
def test_instruct_picks_the_wording_and_names_its_source(phase, here, wording, told):
    assert instruct(TaskPhase.parse(phase), _level(), here, wording) == told


@pytest.mark.parametrize("world", ["hospital_1", "office_1"])
def test_every_room_of_a_shipped_world_is_reachable(world):
    level = converter.structure(yaml.safe_load((WORLDS / world / "0" / "world.yaml").read_text()), LevelDescription)
    shapes = zone_shapes(level)
    found = portals(level)
    first = level.zones[0].name
    start = shapes[first].representative_point().coords[0]
    for name, shape in shapes.items():
        assert crossings(found, start, first, shape.representative_point().coords[0], name) is not None, name
