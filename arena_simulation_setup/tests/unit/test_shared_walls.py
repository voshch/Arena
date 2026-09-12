from __future__ import annotations

import asyncio

import pytest

from arena_simulation_setup.shared.walls import Wall
from arena_simulation_setup.utils.geometry import Position


def test_wall_iter_yields_start_end():
    w = Wall(start=Position(1.0, 0.0, 0.0), end=Position(2.0, 0.0, 0.0))
    points = list(w)
    assert len(points) == 2
    assert points[0].x == pytest.approx(1.0)
    assert points[1].x == pytest.approx(2.0)


def test_wall_serialize_no_kind_no_material():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0))
    d = w.serialize()
    assert "kind" not in d
    assert "material" not in d


def test_wall_serialize_with_kind():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0), kind="brick")
    d = w.serialize()
    assert d["kind"] == "brick"
    assert "material" not in d


def test_wall_serialize_with_material_no_kind():
    from arena_simulation_setup.tree.assets.Material import MaterialIdentifier

    mat = MaterialIdentifier("Marble")
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0), material=mat)
    d = w.serialize()
    assert "kind" not in d
    assert "material" in d


def test_wall_assets_no_kind_returns_realization():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0))
    walls_iter, obs_iter = asyncio.run(w.assets())
    walls = list(walls_iter)
    # Simple wall should produce at least one WallSegment
    assert len(walls) >= 1


def test_wall_assets_exception_fallback():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0), kind="nonexistent_wall_type_xyz")
    walls_iter, obs_iter = asyncio.run(w.assets())
    walls = list(walls_iter)
    # Should fall back to a simple wall
    assert len(walls) >= 1
