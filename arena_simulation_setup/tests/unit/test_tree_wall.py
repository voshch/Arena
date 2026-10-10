from __future__ import annotations

import math

import pytest

from arena_simulation_setup.tree.Wall import PlaceObstacleAsset, PositionalNumber, SubWall, TilingAsset, PlaceWallSegmentAsset, WallDescription
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Position


# ---------------------------------------------------------------------------
# PositionalNumber
# ---------------------------------------------------------------------------


def test_positional_number_absolute_positive():
    pn = PositionalNumber(absolute=3.0)
    assert pn.absolute(0.0, 10.0) == pytest.approx(3.0)


def test_positional_number_absolute_negative():
    pn = PositionalNumber(absolute=-2.0)
    # negative absolute: high + absolute
    assert pn.absolute(0.0, 10.0) == pytest.approx(8.0)


def test_positional_number_absolute_zero():
    pn = PositionalNumber(absolute=0.0)
    assert pn.absolute(0.0, 10.0) == pytest.approx(0.0)


def test_positional_number_relative_zero():
    pn = PositionalNumber(relative=0.0)
    assert pn.absolute(0.0, 10.0) == pytest.approx(0.0)


def test_positional_number_relative_half():
    pn = PositionalNumber(relative=0.5)
    assert pn.absolute(0.0, 10.0) == pytest.approx(5.0)


def test_positional_number_relative_one():
    pn = PositionalNumber(relative=1.0)
    assert pn.absolute(0.0, 10.0) == pytest.approx(10.0)


def test_positional_number_parse_percent():
    pn = PositionalNumber.parse("50%")
    assert pn.absolute(0.0, 100.0) == pytest.approx(50.0)


def test_positional_number_parse_float_string():
    pn = PositionalNumber.parse("1.5")
    assert pn.absolute(0.0, 10.0) == pytest.approx(1.5)


def test_positional_number_constructor_neither_raises():
    with pytest.raises(ValueError):
        PositionalNumber()


def test_positional_number_realize_along_x_axis():
    pn = PositionalNumber(relative=0.5)
    start = Position(0.0, 0.0, 0.0)
    end = Position(10.0, 0.0, 0.0)
    mid = pn.realize(start, end)
    assert mid.x == pytest.approx(5.0, abs=1e-5)
    assert mid.y == pytest.approx(0.0, abs=1e-5)


def test_positional_number_realize_along_y_axis():
    pn = PositionalNumber(relative=0.5)
    start = Position(0.0, 0.0, 0.0)
    end = Position(0.0, 8.0, 0.0)
    mid = pn.realize(start, end)
    assert mid.x == pytest.approx(0.0, abs=1e-5)
    assert mid.y == pytest.approx(4.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SubWall._shift
# ---------------------------------------------------------------------------


def test_subwall_shift_no_offset():
    pws = PlaceWallSegmentAsset(x=0.0, y=0.0, z=0.0)
    start = Position(0.0, 0.0, 0.0)
    end = Position(5.0, 0.0, 0.0)
    ns, ne = pws._shift(start, end)
    assert ns.x == pytest.approx(0.0, abs=1e-5)
    assert ne.x == pytest.approx(5.0, abs=1e-5)


def test_subwall_shift_y_offset_along_x_wall():
    # y-offset for a wall along x should move in y direction
    pws = PlaceWallSegmentAsset(x=0.0, y=1.0, z=0.0)
    start = Position(0.0, 0.0, 0.0)
    end = Position(5.0, 0.0, 0.0)
    ns, ne = pws._shift(start, end)
    # After shift, y should increase by ~1 (direction is x-axis, so perp rotation applies)
    assert ns.y == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# TilingAsset.realize
# ---------------------------------------------------------------------------


def test_tiling_asset_realize_zero_length_wall():
    ta = TilingAsset(
        tile=[PlaceWallSegmentAsset()],
        every=1.0,
        width=0.1,
    )
    start = Position(0.0, 0.0, 0.0)
    end = Position(0.0, 0.0, 0.0)  # zero length
    walls_iter, obs_iter = ta.realize(start, end)
    assert list(walls_iter) == []


def test_tiling_asset_realize_every_larger_than_wall():
    ta = TilingAsset(
        tile=[PlaceWallSegmentAsset()],
        every=10.0,
        width=0.0,
    )
    start = Position(0.0, 0.0, 0.0)
    end = Position(3.0, 0.0, 0.0)
    walls_iter, obs_iter = ta.realize(start, end)
    # every > wall_length → no tiles
    assert list(walls_iter) == []


def test_tiling_asset_realize_normal_tiling():
    ta = TilingAsset(
        tile=[PlaceWallSegmentAsset()],
        every=1.0,
        width=0.0,
    )
    start = Position(0.0, 0.0, 0.0)
    end = Position(5.0, 0.0, 0.0)
    walls_iter, obs_iter = ta.realize(start, end)
    walls = list(walls_iter)
    # Should produce some tiles in a 5m wall with every=1m
    assert len(walls) > 0


# ---------------------------------------------------------------------------
# PlaceWallSegmentAsset flags
# ---------------------------------------------------------------------------


def test_wall_segment_flags_default_to_true():
    (segment,), obstacles = PlaceWallSegmentAsset().realize(Position(0.0, 0.0, 0.0), Position(4.0, 0.0, 0.0))
    assert (segment.visible, segment.solid, segment.shadows) == (True, True, True)
    assert list(obstacles) == []


def test_wall_segment_flags_reach_the_realized_segment():
    asset = PlaceWallSegmentAsset(visible=False, solid=False, shadows=False)
    (segment,), _ = asset.realize(Position(0.0, 0.0, 0.0), Position(4.0, 0.0, 0.0))
    assert (segment.visible, segment.solid, segment.shadows) == (False, False, False)


def test_wall_description_structures_segment_flags_from_yaml_data():
    description = converter.structure(
        {
            'main': [
                {'material': 'Plaster_Wall', 'height': 2.5, 'visible': False, 'shadows': False},
                {'material': 'Plaster_Wall', 'height': 2.6, 'y': -0.35, 'solid': False, 'shadows': False},
                {'model': 'splat_room', 'at': '50%'},
            ]
        },
        WallDescription,
    )
    line, backdrop, splat = description.main
    assert isinstance(line, PlaceWallSegmentAsset) and isinstance(backdrop, PlaceWallSegmentAsset)
    assert isinstance(splat, PlaceObstacleAsset)
    assert (line.visible, line.solid, line.shadows) == (False, True, False)
    assert (backdrop.visible, backdrop.solid, backdrop.shadows) == (True, False, False)

    segments, obstacles = description.realize(Position(4.0, 0.0, 0.0), Position(0.0, 0.0, 0.0))
    segments = list(segments)
    assert [(s.visible, s.solid, s.shadows) for s in segments] == [(False, True, False), (True, False, False)]
    assert segments[1].start.y == pytest.approx(0.35, abs=1e-6)
    assert [o.model.name for o in obstacles] == ['splat_room']


def test_wall_segment_flags_survive_unstructure_and_structure():
    asset = PlaceWallSegmentAsset(visible=False, solid=False, shadows=False)
    data = converter.unstructure(asset)
    assert (data['visible'], data['solid'], data['shadows']) == (False, False, False)
    restored = converter.structure(data, PlaceWallSegmentAsset)
    assert (restored.visible, restored.solid, restored.shadows) == (False, False, False)
