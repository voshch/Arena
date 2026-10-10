"""Walls of a zone with a ceiling reach the ceiling, and doors lower than the ceiling get a lintel."""

from __future__ import annotations

import asyncio

import pytest

from arena_simulation_setup.shared import Door
from arena_simulation_setup.shared.walls import Wall
from arena_simulation_setup.tree.World.World import LevelDescription
from arena_simulation_setup.utils.geometry import Position


def _room(ceiling_height: float | None = 2.6, ceiling: bool = True, doors: list[Door] | None = None) -> LevelDescription:
    corners = [Position(0.0, 0.0), Position(8.0, 0.0), Position(8.0, 6.0), Position(0.0, 6.0)]
    walls = [Wall(start=corners[index], end=corners[(index + 1) % 4]) for index in range(4)]
    zone = LevelDescription.Zone(name='hall', corners=corners, walls=walls, doors=doors or [], ceiling=ceiling, ceiling_height=ceiling_height)
    return LevelDescription(zones=[zone])


def _tops(walls: list[Wall]) -> list[float]:
    result = []
    for wall in walls:
        segments, _ = asyncio.run(wall.assets())
        result.append(max(segment.start.z + segment.height for segment in segments))
    return result


def test_walls_rise_to_a_ceiling_above_them() -> None:
    plain = [wall.top for wall in _room().zones[0].walls]
    assert plain == [None, None, None, None]
    assert _tops(_room().zones[0].walls) == pytest.approx([2.0] * 4)
    assert _tops(asyncio.run(_room().closed_walls())) == pytest.approx([2.6] * 4)


def test_walls_keep_their_height_without_a_ceiling() -> None:
    walls = asyncio.run(_room(ceiling=False).closed_walls())
    assert [wall.top for wall in walls] == [None] * 4
    assert _tops(walls) == pytest.approx([2.0] * 4)


def test_walls_taller_than_the_ceiling_stay_as_they_are() -> None:
    assert _tops(asyncio.run(_room(ceiling_height=1.5).closed_walls())) == pytest.approx([2.0] * 4)


def test_derived_ceiling_adds_nothing_to_walls() -> None:
    assert _tops(asyncio.run(_room(ceiling_height=None).closed_walls())) == pytest.approx([2.0] * 4)


def test_door_below_the_ceiling_gets_a_lintel_from_its_top_to_the_ceiling() -> None:
    door = Door(name='entry', start=Position(3.0, 0.0), end=Position(4.0, 0.0), height=2.0)
    (lintel,) = asyncio.run(_room(doors=[door]).door_lintels())
    (segment,), obstacles = asyncio.run(lintel.assets())
    assert obstacles == ()
    assert (segment.start.x, segment.start.y, segment.start.z) == pytest.approx((3.0, 0.0, 2.0))
    assert (segment.end.x, segment.end.y) == pytest.approx((4.0, 0.0))
    assert segment.height == pytest.approx(0.6)


@pytest.mark.parametrize(('ceiling_height', 'ceiling'), [(2.0, True), (None, True), (2.6, False)])
def test_no_lintel_when_the_door_reaches_the_ceiling_or_there_is_none(ceiling_height: float | None, ceiling: bool) -> None:
    door = Door(name='entry', start=Position(3.0, 0.0), end=Position(4.0, 0.0), height=2.0)
    assert asyncio.run(_room(ceiling_height=ceiling_height, ceiling=ceiling, doors=[door]).door_lintels()) == []


def test_wall_serialization_leaves_out_the_derived_heights() -> None:
    wall = Wall(start=Position(0.0, 0.0), end=Position(1.0, 0.0), top=2.6, bottom=2.0)
    assert set(wall.serialize()) == {'start', 'end'}
