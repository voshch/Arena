from __future__ import annotations

from pathlib import Path

import pytest
from arena_simulation_setup.tree.World import WorldView
from task_generator.utils.zone_corners import write_zone_polygons

_WORLD = """\
format: arena-world/2
levels:
- id: '0'
  zones:
  - name: lobby
    polygon: [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]
    walls: outline
  - name: office
    polygon: [[6.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 4.0]]
    walls: outline
  doors:
  - {name: office_door, span: [[6.0, 1.5], [6.0, 2.5]]}
"""


@pytest.fixture
def world_root(tmp_path: Path) -> Path:
    root = tmp_path / "two_rooms"
    root.mkdir()
    (root / "world.yaml").write_text(_WORLD)
    return root


def _points(world_root: Path, zone: str) -> list[tuple[float, float]]:
    level = WorldView(world_root).load().levels["0"]
    return [(c.x, c.y) for c in next(z for z in level.zones if z.name == zone).polygon]


def test_moved_point_is_saved_and_every_other_value_kept(world_root: Path):
    before = WorldView(world_root).load()
    points = _points(world_root, "office")
    points[2] = (10.5, 4.25)

    assert write_zone_polygons(world_root, {"office": points}) == 2

    after = WorldView(world_root).load()
    office = next(z for z in after.levels["0"].zones if z.name == "office")
    assert [(c.x, c.y) for c in office.polygon] == [(6.0, 0.0), (10.0, 0.0), (10.5, 4.25), (6.0, 4.0)]
    assert after.levels["0"].zones[0] == before.levels["0"].zones[0]
    assert after.levels["0"].doors == before.levels["0"].doors
    assert (world_root / "map" / "0.yaml").is_file()


def test_untouched_zones_keep_their_polygon(world_root: Path):
    points = _points(world_root, "lobby")
    points[1] = (points[1][0] + 0.5, points[1][1])

    assert write_zone_polygons(world_root, {"lobby": points}) == 1

    assert _points(world_root, "lobby")[1] == (6.5, 0.0)
    assert _points(world_root, "office") == [(6.0, 0.0), (10.0, 0.0), (10.0, 4.0), (6.0, 4.0)]


def test_unchanged_polygons_leave_the_world_untouched(world_root: Path):
    before = (world_root / "world.yaml").read_bytes()
    polygons = {zone: _points(world_root, zone) for zone in ("lobby", "office")}

    assert write_zone_polygons(world_root, polygons) == 0
    assert (world_root / "world.yaml").read_bytes() == before
    assert sorted(path.name for path in world_root.iterdir()) == ["world.yaml"]


@pytest.mark.parametrize(("zone", "match"), [("nowhere", "no zone"), ("office", "polygon points")])
def test_bad_polygon_is_rejected(world_root: Path, zone: str, match: str):
    with pytest.raises(ValueError, match=match):
        write_zone_polygons(world_root, {zone: [(0.0, 0.0)]})
