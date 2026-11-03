from __future__ import annotations

import shutil
from pathlib import Path

import pytest

try:
    import attrs
    from arena_simulation_setup.tree.World import World, WorldView
    from task_generator.tasks.modules.zone_edit.zone_corners import write_zone_polygons
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

_LIT_ROOMS = Path(__file__).resolve().parents[1] / "worlds" / "lit_rooms"


def _load(path: Path) -> World:
    return WorldView(path).load()


@pytest.fixture
def lit_rooms(tmp_path: Path) -> Path:
    shutil.copytree(_LIT_ROOMS, tmp_path / "lit_rooms")
    return tmp_path / "lit_rooms"


def test_polygon_edit_changes_zone_lookup_after_compaction(lit_rooms: Path):
    world = _load(lit_rooms)
    zone = next(z for z in world.levels["0"].zones if z.name == "hall")
    zone.polygon[2] = attrs.evolve(zone.polygon[2], x=8.5)

    polygon = world.compact_world({"0": (0.0, 0.0)}).lookup_zone_polygon("hall")

    assert polygon is not None
    assert polygon[2].x == 8.5


def test_saved_polygon_edit_survives_reload_and_keeps_the_rest(lit_rooms: Path):
    before = _load(lit_rooms)
    world = _load(lit_rooms)
    zone = next(z for z in world.levels["0"].zones if z.name == "hall")
    zone.polygon[2] = attrs.evolve(zone.polygon[2], x=8.5, y=6.25)

    polygons = {z.name: [(c.x, c.y) for c in z.polygon] for level in world.all_levels for z in level.zones}
    assert write_zone_polygons(lit_rooms, polygons) == 2

    after = _load(lit_rooms)
    reloaded = next(z for z in after.levels["0"].zones if z.name == "hall")
    assert (reloaded.polygon[2].x, reloaded.polygon[2].y) == (8.5, 6.25)
    assert after.levels["0"] == attrs.evolve(before.levels["0"], zones=[attrs.evolve(before.levels["0"].zones[0], polygon=reloaded.polygon)])
    assert after.schedules == before.schedules
