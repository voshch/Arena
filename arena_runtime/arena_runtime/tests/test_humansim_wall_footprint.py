from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from arena_runtime.humans.arena_humansim.arena_humansim import _footprint
from arena_runtime.shared import Position, Wall
from arena_simulation_setup.tree.World.view import point_world_assets


@pytest.fixture
def cabinet_kind(tmp_path: Path):
    kind = tmp_path / "assets" / "Common" / "Wall" / "cabinet"
    kind.mkdir(parents=True)
    (kind / "cabinet.yaml").write_text("main:\n  - material: Plaster_Wall\n    height: 1.0\n    width: 0.25\n  - material: Plaster_Wall\n    height: 0.03\n    width: 0.4\n    z: 2.5\n")
    point_world_assets(tmp_path, [])
    yield "cabinet"
    point_world_assets(None, [])


def test_thin_wall_goes_to_the_engine_as_its_center_line() -> None:
    wall = Wall(start=Position(0.0, 0.0), end=Position(4.0, 0.0))
    assert asyncio.run(_footprint(wall)) == [((0.0, 0.0), (4.0, 0.0))]


def test_thick_wall_goes_to_the_engine_as_the_outline_of_its_body_height_layers(cabinet_kind: str) -> None:
    wall = Wall(start=Position(1.0, 0.0), end=Position(1.0, 2.0), kind=cabinet_kind)
    edges = asyncio.run(_footprint(wall))
    assert len(edges) == 4
    assert [end for _, end in edges] == [start for start, _ in edges[1:]] + [edges[0][0]]
    xs = sorted({start[0] for start, _ in edges})
    ys = sorted({start[1] for start, _ in edges})
    assert xs == pytest.approx([0.875, 1.125])
    assert ys == pytest.approx([0.0, 2.0])
