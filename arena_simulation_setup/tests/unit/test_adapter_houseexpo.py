"""HouseExpo world importer on synthetic plans: ring to free space and walls, room boxes to zones, the archive index, and the native loader and resolver."""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import shapely
import yaml

ITEM = "00000000000000000000000000000001"
ROOMS_ITEM = "00000000000000000000000000000002"
SPIKE_ITEM = "00000000000000000000000000000003"
PINCH_ITEM = "00000000000000000000000000000004"
FLAT_ITEM = "00000000000000000000000000000005"
SHORT_ITEM = "00000000000000000000000000000006"
LEDGE_ITEM = "00000000000000000000000000000007"
CORNER_ITEM = "00000000000000000000000000000008"
ABSENT_ITEM = "ffffffffffffffffffffffffffffffff"
REAL_ITEM = "1877b5426f17c13710a92e48a8970d80"
NAME = f"houseexpo_{ITEM}"
FILES = ["ATTRIBUTION.yaml", "kg.yaml", "map/0.png", "map/0.yaml", "world.yaml"]
ARCHIVE_URL = "https://raw.githubusercontent.com/TeaganLi/HouseExpo/45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c/HouseExpo/json.tar.gz"
ARCHIVE_BYTES = 25_602_801
ARCHIVE_PLANS = 35_126
RESOLUTION = 0.05
PADDING = 5
FREE = 255
OCCUPIED = 0

RING = [[0, 0], [0, 6], [4, 6], [4, 3], [4.2, 3], [4.2, 6], [10, 6], [10, 0], [4.2, 0], [4.2, 2], [4, 2], [4, 0]]
RING_AREA_M2 = 59.0
RING_LENGTH_M = 42.0
HOUSE = {
    "verts": RING,
    "id": ITEM,
    "room_category": {
        "Living_Room": [[4.1, -0.1, 10.1, 6.1]],
        "Office": [[4.1, -0.1, 10.1, 6.1]],
        "Toilet": [[8, 4, 10.1, 6.1]],
        "Kitchen": [[-0.1, 2, 4.1, 6.1]],
        "Garage": [[20, 20, 25, 25]],
    },
    "room_num": 4,
    "bbox": {"min": [0, 0], "max": [10, 6]},
}
ZONES = ["toilet_0", "kitchen_0", "living_room_0", "area_0"]
ZONE_TYPES = ["Toilet", "Kitchen", "Living_Room, Office", ""]
ZONE_AREAS_M2 = [4.0, 16.1, 30.9, 8.0]

ROOMS = {
    "verts": [[0, 0], [0, 2], [4, 2], [4, 0]],
    "id": ROOMS_ITEM,
    "room_category": {"Bedroom": [[0, 0, 2.5, 2], [0, 0, 1, 1]], "Bathroom": [[1.5, 0, 4, 2]], "Room": [[0, 0, 1, 1]]},
    "room_num": 3,
    "bbox": {"min": [0, 0], "max": [4, 2]},
}
SHORT = {**HOUSE, "id": SHORT_ITEM, "room_category": {"Living_Room": [[4.1, -0.1, 10.1, 6.1]], "Kitchen": [[-0.1, -0.1, 3.9, 6.1]]}, "room_num": 2}
LEDGE = {"verts": [[0, 0], [0, 0.2], [3, 0.2], [3, 0]], "id": LEDGE_ITEM, "room_category": {"Balcony": [[0, 0, 3, 0.2]]}, "room_num": 1, "bbox": {"min": [0, 0], "max": [3, 0.2]}}
CORNER = {"verts": [[0, 0], [0, 1.52], [3.27, 1.52], [3.29, 1.5], [3.29, 0]], "id": CORNER_ITEM, "room_category": {"Kitchen": [[-0.1, -0.1, 2.78, 1.01]], "Hall": [[3.28, 1.51, 5.28, 3.51]]}, "room_num": 2, "bbox": {"min": [0, 0], "max": [3.29, 1.52]}}
SPIKE = {"verts": [[0, 0], [0, 4], [2, 4], [2, 6], [2, 4], [4, 4], [4, 0]], "id": SPIKE_ITEM, "room_category": {}, "room_num": 1, "bbox": {"min": [0, 0], "max": [4, 6]}}
PINCH = {"verts": [[0, 0], [0, 2], [2, 2], [2, 4], [4, 4], [4, 2], [2, 2], [2, 0]], "id": PINCH_ITEM, "room_category": {"Kitchen": [[0, 0, 2, 2]]}, "room_num": 2, "bbox": {"min": [0, 0], "max": [4, 4]}}
FLAT = {"verts": [[0, 0], [1, 0], [2, 0], [3, 0]], "id": FLAT_ITEM, "room_category": {"Kitchen": [[0, 0, 2, 2]]}, "room_num": 1, "bbox": {"min": [0, 0], "max": [3, 0]}}
PLANS = [HOUSE, ROOMS, SPIKE, PINCH, FLAT]


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")


def archive(path: Path, *plans: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        directory = tarfile.TarInfo("json")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for plan in plans:
            data = json.dumps(plan).encode()
            member = tarfile.TarInfo(f"json/{plan['id']}.json")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return path


def seed(cache: Path) -> Path:
    archive(cache / "json.tar.gz", *PLANS)
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from arena_simulation_setup.adapters.houseexpo import HouseExpo

    return HouseExpo(cache=seed(tmp_path / "cache"))


@pytest.fixture()
def imported(importer, tmp_path: Path) -> Path:
    return importer.import_item(ITEM, tmp_path / "worlds")


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _polygon(zone) -> shapely.Polygon:
    return shapely.Polygon([(corner.x, corner.y) for corner in zone.polygon])


def _built(plan: dict):
    from arena_simulation_setup.adapters.houseexpo import world

    return world(plan).levels["0"]


def _edges(points: list) -> set[frozenset]:
    return {frozenset({(round(float(a[0]), 6), round(float(a[1]), 6)), (round(float(b[0]), 6), round(float(b[1]), 6))}) for a, b in zip(points, points[1:] + points[:1], strict=True)}


def _wall_edges(walls: list) -> set[frozenset]:
    return {frozenset({(round(wall.start.x, 6), round(wall.start.y, 6)), (round(wall.end.x, 6), round(wall.end.y, 6))}) for wall in walls}


def _length(walls: list) -> float:
    return sum(math.dist((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)) for wall in walls)


def _cell(grid, bounds: tuple[float, float, float, float], x: float, y: float) -> int:
    return int(grid[int((bounds[3] - y) / RESOLUTION) + PADDING, int((x - bounds[0]) / RESOLUTION) + PADDING])


def _view(out: Path, name: str):
    from arena_simulation_setup.tree import SimplePathResolver
    from arena_simulation_setup.tree.World import WorldIdentifier

    identifier = WorldIdentifier(name)
    resolved = asyncio.run(SimplePathResolver(WorldIdentifier, path=out).resolve(identifier))
    assert resolved == out / identifier.relpath()
    return identifier.load(resolved)


def test_registry_target_is_the_class() -> None:
    from arena_simulation_setup.adapters.houseexpo import HouseExpo

    from arena_assets.importers import IMPORTERS

    assert IMPORTERS.load("world", "houseexpo") is HouseExpo
    assert HouseExpo.meta.kind == "world"
    assert HouseExpo.default_provider().url("json.tar.gz") == ARCHIVE_URL


def test_ring_is_the_free_space_and_its_outline_the_walls() -> None:
    imported = _built(HOUSE)
    free = shapely.union_all([_polygon(zone) for zone in imported.zones])
    assert free.symmetric_difference(shapely.Polygon(RING)).area == pytest.approx(0.0, abs=1e-9)
    assert free.area == pytest.approx(RING_AREA_M2)
    walls = list(imported.all_walls)
    assert _wall_edges(walls) == _edges(RING)
    assert len(walls) == len(RING)
    assert _length(walls) == pytest.approx(RING_LENGTH_M)
    assert not list(imported.all_doors)
    assert all(zone.ceiling for zone in imported.zones)


def test_room_boxes_are_zones_named_by_type_and_clipped_to_the_floor() -> None:
    from arena_simulation_setup.adapters.houseexpo import world
    from arena_simulation_setup.tree.World import describe

    built = world(HOUSE)
    imported = built.levels["0"]
    assert [zone.name for zone in imported.zones] == ZONES
    assert [describe(built.kg, zone.name) for zone in imported.zones] == ZONE_TYPES
    assert [_polygon(zone).area for zone in imported.zones] == pytest.approx(ZONE_AREAS_M2)
    assert all(shapely.Polygon(RING).covers(_polygon(zone)) for zone in imported.zones)
    assert _polygon(imported.zones[0]).bounds == (8.0, 4.0, 10.0, 6.0)
    assert _polygon(imported.zones[3]).bounds == (0.0, 0.0, 4.0, 2.0)
    assert imported.lookup_zone_polygon("living_room_0") == imported.zones[2].polygon


def test_smaller_box_owns_an_overlap_and_a_shared_box_is_one_room() -> None:
    from arena_simulation_setup.adapters.houseexpo import boxes

    regions = boxes(ROOMS)
    assert [region.name for region in regions] == ["bedroom_1", "bathroom_0", "bedroom_0"]
    assert [region.description for region in regions] == ["Bedroom, Room", "Bathroom", "Bedroom"]
    assert [region.polygon.area for region in regions] == [1.0, 5.0, 5.0]
    imported = _built(ROOMS)
    assert [zone.name for zone in imported.zones] == ["bedroom_1", "bathroom_0", "bedroom_0"]
    assert [_polygon(zone).area for zone in imported.zones] == pytest.approx([1.0, 5.0, 2.0])
    assert _polygon(imported.zones[1]).bounds == (1.5, 0.0, 4.0, 2.0)


def test_strip_between_a_box_and_its_wall_joins_the_room_beside_it() -> None:
    from arena_simulation_setup.adapters.houseexpo import boxes, floor, rooms

    free = floor(SHORT)
    regions = rooms(free, boxes(SHORT))
    assert [region.name for region in regions] == ["kitchen_0", "living_room_0"]
    assert shapely.box(3.9, 0, 4, 6).difference(regions[0].polygon).area == pytest.approx(0.0, abs=1e-9)
    imported = _built(SHORT)
    assert [zone.name for zone in imported.zones] == ["kitchen_0", "living_room_0"]
    assert [_polygon(zone).area for zone in imported.zones] == pytest.approx([24.1, 34.9])
    assert shapely.union_all([_polygon(zone) for zone in imported.zones]).symmetric_difference(shapely.Polygon(RING)).area == pytest.approx(0.0, abs=1e-9)


def test_narrow_piece_without_a_neighbor_stays_a_zone() -> None:
    imported = _built(LEDGE)
    assert [zone.name for zone in imported.zones] == ["balcony_0"]
    assert _polygon(imported.zones[0]).bounds == (0.0, 0.0, 3.0, 0.2)


def test_box_corner_touching_a_slanted_wall_makes_no_hairline_zone() -> None:
    imported = _built(CORNER)
    assert [zone.name for zone in imported.zones] == ["kitchen_0", "area_0"]
    assert [_polygon(zone).area for zone in imported.zones] == pytest.approx([2.8078, 2.1928])
    grid, _ = imported.render_grid(resolution=RESOLUTION)
    assert (grid == FREE).sum() * RESOLUTION**2 == pytest.approx(5.0006, rel=0.1)


def test_spike_in_the_ring_is_dropped() -> None:
    from arena_simulation_setup.adapters.houseexpo import floor

    assert not shapely.Polygon(SPIKE["verts"]).is_valid
    parts = floor(SPIKE)
    assert [part.bounds for part in parts] == [(0.0, 0.0, 4.0, 4.0)]
    assert parts[0].area == pytest.approx(16.0)
    imported = _built(SPIKE)
    assert [zone.name for zone in imported.zones] == ["area_0"]
    assert _length(list(imported.all_walls)) == pytest.approx(16.0)
    assert max(max(wall.start.y, wall.end.y) for wall in imported.all_walls) == 4.0


def test_ring_touching_itself_is_split_into_its_rooms() -> None:
    from arena_simulation_setup.adapters.houseexpo import floor

    assert not shapely.Polygon(PINCH["verts"]).is_valid
    assert sorted(part.bounds for part in floor(PINCH)) == [(0.0, 0.0, 2.0, 2.0), (2.0, 2.0, 4.0, 4.0)]
    imported = _built(PINCH)
    assert [zone.name for zone in imported.zones] == ["kitchen_0", "area_0"]
    assert [_polygon(zone).bounds for zone in imported.zones] == [(0.0, 0.0, 2.0, 2.0), (2.0, 2.0, 4.0, 4.0)]
    assert _length(list(imported.all_walls)) == pytest.approx(16.0)


@pytest.mark.parametrize("verts", [FLAT["verts"], [[0, 0], [1, 1]], []])
def test_ring_without_area_is_refused(verts: list) -> None:
    from arena_simulation_setup.adapters.houseexpo import floor

    with pytest.raises(ValueError, match=f"plan '{FLAT_ITEM}' has no floor"):
        floor({**FLAT, "verts": verts})


def test_refused_plan_writes_no_world(importer, tmp_path: Path) -> None:
    out = tmp_path / "worlds"
    with pytest.raises(ValueError, match="has no floor"):
        importer.import_item(FLAT_ITEM, out)
    assert not out.exists()


@pytest.mark.parametrize("item", ["", "0", "1877B5426F17C13710A92E48A8970D80", f"../{ITEM}", f"{ITEM}.json"])
def test_item_that_is_no_plan_id_is_refused(importer, item: str) -> None:
    with pytest.raises(ValueError, match="32 hex characters"):
        importer.fetch(item)
    with pytest.raises(ValueError, match="32 hex characters"):
        importer.attribution(item)


def test_plan_id_missing_from_the_archive_is_refused(importer) -> None:
    with pytest.raises(ValueError, match=f"no HouseExpo plan '{ABSENT_ITEM}' among the 5 of json.tar.gz"):
        importer.fetch(ABSENT_ITEM)
    assert not (importer.cache / "json" / f"{ABSENT_ITEM}.json").exists()


def test_items_are_the_plan_ids_in_archive_order(importer) -> None:
    assert list(importer.items()) == [plan["id"] for plan in PLANS]
    index = json.loads((importer.cache / "index.json").read_text())
    assert list(index) == [plan["id"] for plan in PLANS]
    assert all(size == len(json.dumps(plan)) for plan, (_, size) in zip(PLANS, index.values(), strict=True))


def test_archive_is_indexed_once_and_read_from_one_stream(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.houseexpo import Plans

    src = archive(tmp_path / "json.tar.gz", *PLANS)
    plans = Plans(src)
    assert json.loads(plans.read(ROOMS_ITEM)) == ROOMS
    src.unlink()
    assert json.loads(plans.read(PINCH_ITEM)) == PINCH
    assert json.loads(plans.read(ITEM)) == HOUSE
    indexed = Plans(src)
    assert list(indexed.index) == [plan["id"] for plan in PLANS]
    with pytest.raises(FileNotFoundError):
        indexed.read(ITEM)


def test_fetched_plan_is_kept_beside_the_archive(importer) -> None:
    from arena_simulation_setup.adapters.houseexpo import HouseExpo

    src = importer.fetch(ITEM)
    assert src == importer.cache / "json" / f"{ITEM}.json"
    assert json.loads(src.read_text()) == HOUSE
    (importer.cache / "json.tar.gz").unlink()
    (importer.cache / "index.json").unlink()
    assert HouseExpo(cache=importer.cache).fetch(ITEM) == src
    assert _files(importer.cache) == [f"json/{ITEM}.json"]


def test_import_writes_a_world_the_world_tree_loads(imported: Path, tmp_path: Path) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier, WorldView, describe

    assert imported == tmp_path / "worlds" / WorldIdentifier(NAME).relpath()
    assert _files(imported) == FILES
    view = WorldView(imported)
    world = view.load()
    assert list(world.levels) == ["0"]
    level = world.levels["0"]
    assert [zone.name for zone in level.zones] == ZONES
    assert [describe(world.kg, zone.name) for zone in level.zones] == ZONE_TYPES
    assert level.zone_ref_names() == ZONES
    assert _wall_edges(list(level.all_walls)) == _edges(RING)
    assert not list(level.all_static_entities)
    assert not list(view.scenario.listall())
    assert all(identifier.resolve_source_sync() for identifier in view.identifiers(strict=True))


def test_world_files_match_a_native_generator_world(imported: Path, tmp_path: Path) -> None:
    from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
    from arena_simulation_setup.tree.World import WorldView

    native = WorldView(tmp_path / "native" / "native_empty").save(WorldGenerator(WorldGeneratorType.EMPTY, {}, 0).compute())
    assert [name for name in _files(imported) if name not in ("ATTRIBUTION.yaml", "kg.yaml")] == [name for name in _files(native) if name != "kg.yaml"]
    (imported_level,), (native_level,) = (yaml.safe_load((root / "world.yaml").read_text())["levels"] for root in (imported, native))
    assert list(imported_level) == list(native_level)
    assert all(set(zone) == set(native_level["zones"][0]) for zone in imported_level["zones"])
    assert {key for wall in imported_level["walls"] for key in wall} <= {"path", "closed", "kind", "material"}
    imported_map, native_map = (yaml.safe_load((root / "map" / "0.yaml").read_text()) for root in (imported, native))
    assert {**imported_map, "origin": None} == {**native_map, "origin": None}


def test_runtime_loader_renders_rooms_wall_slit_and_door_gap(imported: Path, tmp_path: Path) -> None:
    view = _view(tmp_path / "worlds", NAME)
    level = view.load().levels["0"]
    assert view.level_origins() is None
    png, origin = level.render(resolution=RESOLUTION)
    assert png == (imported / "map" / "0.png").read_bytes()
    assert yaml.safe_load((imported / "map" / "0.yaml").read_text())["origin"] == pytest.approx([-PADDING * RESOLUTION, -PADDING * RESOLUTION, 0])
    grid, origin = level.render_grid(resolution=RESOLUTION)
    bounds = (0.0, 0.0, 10.0, 6.0)
    assert grid.shape == (math.ceil(6 / RESOLUTION) + 2 * PADDING, math.ceil(10 / RESOLUTION) + 2 * PADDING)
    assert _cell(grid, bounds, 2.02, 4.02) == FREE
    assert _cell(grid, bounds, 2.02, 1.02) == FREE
    assert _cell(grid, bounds, 9.02, 5.02) == FREE
    assert _cell(grid, bounds, 4.11, 2.52) == FREE
    assert _cell(grid, bounds, 4.11, 4.52) == OCCUPIED
    assert _cell(grid, bounds, 4.11, 1.02) == OCCUPIED
    gap = [_cell(grid, bounds, 4.11, 1.52 + step * RESOLUTION) for step in range(40)]
    assert 18 <= gap.count(FREE) <= 20
    assert gap[0] == OCCUPIED and gap[-1] == OCCUPIED
    assert (grid[:PADDING] == OCCUPIED).all()
    assert (grid == FREE).sum() * RESOLUTION**2 == pytest.approx(RING_AREA_M2, rel=0.05)


def test_reimport_replaces_the_world(importer, imported: Path, tmp_path: Path) -> None:
    (imported / "stale.txt").write_text("x")
    assert importer.import_item(ITEM, tmp_path / "worlds") == imported
    assert _files(imported) == FILES


def test_attribution_names_the_repository_license_and_the_suncg_origin(importer, imported: Path) -> None:
    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    attribution = read_attribution(imported / ATTRIBUTION_FILE)
    assert attribution == importer.attribution(ITEM)
    assert attribution.source == "houseexpo"
    assert attribution.url == ARCHIVE_URL
    assert attribution.license == "MIT (repository), floor plans built on the SUNCG dataset"
    assert attribution.authors.startswith("Tingguang Li")
    assert "Max Q.-H. Meng" in attribution.authors
    assert "arXiv:1903.09845" in attribution.citation


_RESOLVED = """
import json
from arena_simulation_setup.tree.World import WorldIdentifier

identifier = WorldIdentifier({name!r})
verdict = identifier.resolve_source_sync()
zones = [zone.name for zone in identifier.resolve_sync().load().levels["0"].zones]
print(json.dumps({{"resolver": repr(verdict.resolver), "verdict": verdict.verdict.name, "path": str(verdict.path), "zones": zones}}))
"""


def test_cli_import_lands_where_the_world_resolver_looks(tmp_path: Path) -> None:
    cache = seed(tmp_path / "cache")
    env = {key: value for key, value in os.environ.items() if key != "ARENA_WORLD_PATH"} | {"ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "imported")}
    target = tmp_path / "imported" / "worlds" / NAME
    imported = subprocess.run([sys.executable, "-m", "arena_assets", "import", "world", "houseexpo", ITEM, "--cache", str(cache)], env=env, capture_output=True, text=True, check=False)
    assert imported.returncode == 0, imported.stderr[-3000:]
    assert imported.stdout.splitlines()[-1] == str(target)
    resolved = subprocess.run([sys.executable, "-c", _RESOLVED.format(name=NAME)], env=env, capture_output=True, text=True, check=False)
    assert resolved.returncode == 0, resolved.stderr[-3000:]
    seen = json.loads(resolved.stdout.splitlines()[-1])
    assert (seen["verdict"], seen["path"], seen["zones"]) == ("HIT", str(target), ZONES)
    assert "DynamicPathResolver" in seen["resolver"]


def test_cli_refuses_an_unknown_plan_without_a_traceback(tmp_path: Path) -> None:
    cache = seed(tmp_path / "cache")
    refused = subprocess.run([sys.executable, "-m", "arena_assets", "import", "world", "houseexpo", ABSENT_ITEM, "--cache", str(cache), "--out", str(tmp_path / "out")], capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "among the 5 of json.tar.gz" in " ".join(refused.stderr.replace("\u2502", " ").split())
    assert "Traceback" not in refused.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.network
def test_live_import_of_a_real_plan(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.houseexpo import HouseExpo
    from arena_simulation_setup.tree.World import WorldView, describe

    importer = HouseExpo(cache=tmp_path / "cache")
    target = importer.import_item(REAL_ITEM, tmp_path / "worlds")
    assert (tmp_path / "cache" / "json.tar.gz").stat().st_size == ARCHIVE_BYTES
    assert len(list(importer.items())) == ARCHIVE_PLANS
    plan = json.loads(importer.fetch(REAL_ITEM).read_text())
    assert (plan["id"], plan["room_num"], plan["bbox"]) == (REAL_ITEM, 6, {"min": [0.1, 0.1], "max": [9.94, 5.21]})
    assert _files(target) == FILES
    built = WorldView(target).load()
    level = built.levels["0"]
    free = shapely.union_all([_polygon(zone) for zone in level.zones])
    assert free.bounds == pytest.approx((0.1, 0.1, 9.94, 5.21))
    assert free.symmetric_difference(shapely.Polygon(plan["verts"])).area == pytest.approx(0.0, abs=1e-6)
    assert {"toilet_0", "dining_room_0", "living_room_0"} <= {zone.name for zone in level.zones}
    assert describe(built.kg, "living_room_0") == "Living_Room, Office"
    assert len(list(level.all_walls)) == len(plan["verts"])
