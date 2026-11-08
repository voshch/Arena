from __future__ import annotations

from arena_simulation_setup.shared.world import Door, clear_door_pockets
from arena_simulation_setup.tree.World import Level
from arena_simulation_setup.utils.geometry import Position


def _door(name: str, start: tuple[float, float], end: tuple[float, float], **kwargs: object) -> Door:
    return Door(name=name, start=Position(*start), end=Position(*end), **kwargs)


def _span(door: Door) -> tuple[tuple[float, float], tuple[float, float]]:
    return (door.start.x, door.start.y), (door.end.x, door.end.y)


BEDROOM = ((0.95, 4.0), (0.15, 4.0))
LIVING_ROOM = ((-0.15, 4.0), (-0.95, 4.0))


def test_leaf_that_would_rest_on_the_next_door_slides_the_other_way():
    bedroom, living = clear_door_pockets([_door("bedroom", *BEDROOM), _door("living", *LIVING_ROOM)])
    assert _span(bedroom) == (BEDROOM[1], BEDROOM[0])
    assert _span(living) == LIVING_ROOM


def test_door_clear_of_its_neighbors_keeps_its_authored_direction():
    doors = clear_door_pockets([_door("a", (0.0, 0.0), (1.0, 0.0)), _door("b", (5.0, 0.0), (6.0, 0.0)), _door("c", (2.0, 1.0), (3.0, 1.0))])
    assert [_span(door) for door in doors] == [((0.0, 0.0), (1.0, 0.0)), ((5.0, 0.0), (6.0, 0.0)), ((2.0, 1.0), (3.0, 1.0))]


def test_door_blocked_on_both_sides_teleports_open():
    door = clear_door_pockets([_door("left", (-1.0, 0.0), (-0.2, 0.0), kind="teleport"), _door("middle", (0.0, 0.0), (0.8, 0.0)), _door("right", (1.0, 0.0), (1.8, 0.0), kind="teleport")])[1]
    assert door.kind == "teleport"
    assert _span(door) == ((0.0, 0.0), (0.8, 0.0))


def test_split_door_clear_of_its_neighbors_stays_split():
    door = clear_door_pockets([_door("lift", (0.0, 0.0), (1.0, 0.0), kind="sliding_split"), _door("far", (2.0, 0.0), (3.0, 0.0))])[0]
    assert door.kind == "sliding_split"
    assert _span(door) == ((0.0, 0.0), (1.0, 0.0))


def test_split_door_with_a_taken_pocket_slides_away_from_it():
    door = clear_door_pockets([_door("near", (1.1, 0.0), (1.6, 0.0), kind="teleport"), _door("lift", (0.0, 0.0), (1.0, 0.0), kind="sliding_split")])[1]
    assert door.kind == "sliding"
    assert _span(door) == ((1.0, 0.0), (0.0, 0.0))


def test_vertical_door_never_reverses():
    door = clear_door_pockets([_door("top", *BEDROOM, kind="sliding_top"), _door("living", *LIVING_ROOM)])[0]
    assert _span(door) == BEDROOM


def test_level_doors_resolve_on_construction_and_stay_stable():
    level = Level(id="0", doors=[_door("bedroom", *BEDROOM), _door("living", *LIVING_ROOM)])
    assert _span(level.doors[0]) == (BEDROOM[1], BEDROOM[0])
    assert _span(clear_door_pockets(level.doors)[0]) == (BEDROOM[1], BEDROOM[0])
