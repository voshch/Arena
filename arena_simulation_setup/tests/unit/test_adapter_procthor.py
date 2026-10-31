from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PACKAGE = Path(__file__).parent.parent.parent
FIXTURES = PACKAGE / "tests" / "fixtures" / "procthor"
FIXTURE = FIXTURES / "val.jsonl"
FRAME_FIXTURE = FIXTURES / "val_161.jsonl"
HAB_SCENE = FIXTURES / "ProcTHOR-Val-0.scene_instance.json"
OBJECT_FIXTURES = FIXTURES / "objects"
FURNITURE = {"Dining_Table_203_1": "DiningTable", "Dog_Statue_4": "Statue", "BaseballBat_3": "BaseballBat"}
PLACEMENT_TOLERANCE_M = 0.005
TOUCH_WORLD = PACKAGE / "scripts" / "touch_world"
NATIVE_WORLD = PACKAGE / "worlds" / "hospital_1"
ITEM = "val/0"
NAME = "procthor_val_0"
LEVEL_FILES = ["kg.yaml", "map/0.png", "map/0.yaml", "world.yaml"]
ROOMS = ["kitchen_2", "living_room_3", "living_room_4", "bathroom_5", "bedroom_6", "bedroom_7", "bedroom_8"]
ROOM_TYPES = ["Kitchen", "LivingRoom", "LivingRoom", "Bathroom", "Bedroom", "Bedroom", "Bedroom"]
WALLS = 27
DOORS = 5
WALL_LENGTH_M = 72.0047
DOOR_OPENINGS_M = 7.0273
NATIVE_WALL_HEIGHT_M = 2.0
RESOLUTION = 0.05
DEFAULT_MATERIALS = {"Common/Material/Porcelain_Tile_4", "Common/Material/Concrete_Smooth", "Common/Material/Marble", "Common/Material/Aluminum_Anodized"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FREE = 255
OCCUPIED = 0


@pytest.fixture()
def converted(tmp_path: Path) -> tuple[Path, str]:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    importer = ProcTHOR(cache=tmp_path / "cache")
    out = tmp_path / "procthor"
    return out, importer.convert(ITEM, _bare(FIXTURE, tmp_path / "val.jsonl"), out)


@pytest.fixture(scope="module")
def blender() -> list[str]:
    from arena_assets.utils.ModelConverter import BlenderUnavailable, command

    try:
        return command()
    except BlenderUnavailable as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def furnished(blender: list[str], tmp_path_factory: pytest.TempPathFactory):
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    root = tmp_path_factory.mktemp("furnished")
    imported = root / "imported"
    previous = os.environ.get("ARENA_ASSETS_DIR_IMPORTED")
    os.environ["ARENA_ASSETS_DIR_IMPORTED"] = str(imported)
    try:
        importer = ProcTHOR(cache=root / "cache", objects=_seeded_objects(root / "ai2thor"), objects_out=imported)
        relpath = importer.convert(ITEM, _furnished_house(root / "val.jsonl"), root / "worlds")
        yield root / "worlds", relpath, imported
    finally:
        if previous is None:
            del os.environ["ARENA_ASSETS_DIR_IMPORTED"]
        else:
            os.environ["ARENA_ASSETS_DIR_IMPORTED"] = previous


def _seeded_objects(cache: Path):
    from arena_assets.importers.ai2thor import AI2THOR, TYPES

    (cache / "objects").mkdir(parents=True)
    for asset in FURNITURE:
        (cache / "objects" / f"{asset}.glb").write_bytes((OBJECT_FIXTURES / f"{asset}.glb").read_bytes())
    database: dict[str, list[dict]] = {}
    for asset, object_type in FURNITURE.items():
        database.setdefault(object_type, []).append({"assetId": asset, "objectType": object_type})
    (cache / TYPES).write_text(json.dumps(database))
    return AI2THOR(cache=cache)


def _furnished_objects() -> list[dict]:
    from arena_simulation_setup.adapters.procthor import house_objects

    return [{key: value for key, value in obj.items() if key != "children"} for obj in house_objects(_house(FIXTURE)) if obj["assetId"] in FURNITURE]


def _furnished_house(path: Path) -> Path:
    path.write_text(json.dumps({**_house(FIXTURE), "objects": _furnished_objects()}) + "\n")
    return path


def _bare(fixture: Path, path: Path) -> Path:
    path.write_text(json.dumps({**_house(fixture), "objects": []}) + "\n")
    return path


def _quaternion_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"]).reshape(4, 4).T
    matrix = np.eye(4)
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    matrix[:3, :3] = _quaternion_matrix(w, x, y, z) @ np.diag(node.get("scale", [1.0, 1.0, 1.0]))
    matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return matrix


def _glb_box(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    gltf = json.loads(data[20 : 20 + int.from_bytes(data[12:16], "little")])
    corners: list[np.ndarray] = []
    stack = [(index, np.eye(4)) for index in gltf["scenes"][gltf.get("scene", 0)]["nodes"]]
    while stack:
        index, parent = stack.pop()
        node = gltf["nodes"][index]
        matrix = parent @ _node_matrix(node)
        for primitive in gltf["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []:
            accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
            for corner in np.array(np.meshgrid(*zip(accessor["min"], accessor["max"], strict=True))).T.reshape(-1, 3):
                corners.append((matrix @ np.append(corner, 1.0))[:3])
        stack.extend((child, matrix) for child in node.get("children", []))
    points = np.array(corners)
    return points.min(axis=0), points.max(axis=0)


def _box_corners(low, high) -> np.ndarray:
    return np.array(np.meshgrid(*zip(low, high, strict=True))).T.reshape(-1, 3)


def _hab_corners(asset: str, arena_center: np.ndarray) -> np.ndarray:
    """Corners of the asset's glb box placed by the hab scene instance nearest the given centre, in Arena axes."""
    low, high = _glb_box(OBJECT_FIXTURES / f"{asset}.glb")
    placements = []
    for instance in json.loads(HAB_SCENE.read_text())["object_instances"]:
        if instance["template_name"] != f"objects/{asset}":
            continue
        hab = _box_corners(low, high) @ _quaternion_matrix(*instance["rotation"]).T + np.array(instance["translation"])
        placements.append(np.stack([-hab[:, 0], hab[:, 2], hab[:, 1]], axis=1))
    return min(placements, key=lambda corners: np.linalg.norm(corners.mean(axis=0) - arena_center))


def _native_corners(entity, root: Path) -> np.ndarray:
    from arena_simulation_setup.utils.geometry import Vector3

    (min_x, max_x), (min_y, max_y), (min_z, max_z) = yaml.safe_load((root / entity.model.relpath() / "annotation.yaml").read_text())["bounding_box"]
    placed = []
    for corner in _box_corners((min_x, min_y, min_z), (max_x, max_y, max_z)):
        rotated = entity.pose.orientation * Vector3(*corner)
        placed.append((rotated.x + entity.pose.position.x, rotated.y + entity.pose.position.y, rotated.z + entity.pose.position.z))
    return np.array(placed)


def _corner_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(np.linalg.norm(b - corner, axis=1).min() for corner in a))


def _view(out: Path, relpath: str):
    from arena_simulation_setup.tree import SimplePathResolver
    from arena_simulation_setup.tree.World import WorldIdentifier

    identifier = WorldIdentifier(relpath)
    resolved = asyncio.run(SimplePathResolver(WorldIdentifier, path=out).resolve(identifier))
    assert resolved == out / identifier.relpath()
    return identifier.load(resolved)


def _load(out: Path, relpath: str):
    return _view(out, relpath).load()


def _house(fixture: Path) -> dict:
    return json.loads(fixture.read_text())


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _length(a, b) -> float:
    return math.dist((a.x, a.y), (b.x, b.y))


def _covered(walls, x: float, y: float) -> bool:
    for wall in walls:
        cross = (wall.end.x - wall.start.x) * (y - wall.start.y) - (wall.end.y - wall.start.y) * (x - wall.start.x)
        inside_x = min(wall.start.x, wall.end.x) - 1e-6 <= x <= max(wall.start.x, wall.end.x) + 1e-6
        inside_y = min(wall.start.y, wall.end.y) - 1e-6 <= y <= max(wall.start.y, wall.end.y) + 1e-6
        if abs(cross) < 1e-6 and inside_x and inside_y:
            return True
    return False


def _patch(grid, origin: tuple[float, float], x: float, y: float):
    col = int((x - origin[0]) / RESOLUTION)
    row = grid.shape[0] - int((y - origin[1]) / RESOLUTION)
    return grid[row - 2 : row + 3, col - 2 : col + 3]


def _split_file(path: Path, *fixtures: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for fixture in fixtures:
            f.write(json.dumps({**_house(fixture), "objects": []}) + "\n")
    return path


def test_registry_target_is_the_class() -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    from arena_assets.importers import IMPORTERS

    assert IMPORTERS.load("world", "procthor") is ProcTHOR
    assert ProcTHOR.meta.kind == "world"


def test_relpath_matches_world_identifier(converted: tuple[Path, str]) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    out, relpath = converted
    assert relpath == NAME
    assert Path(relpath) == WorldIdentifier(NAME).relpath()
    assert _files(out / relpath) == LEVEL_FILES
    data = yaml.safe_load((out / relpath / "world.yaml").read_text())
    assert data["format"] == "arena-world/2"
    (level,) = data["levels"]
    assert set(level) == {"id", "zones", "walls", "doors"}
    assert len(level["zones"]) == len(ROOMS)


def test_world_files_match_a_native_generator_world(converted: tuple[Path, str], tmp_path: Path) -> None:
    from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
    from arena_simulation_setup.tree.World import WorldView

    out, relpath = converted
    native = WorldView(tmp_path / "native" / "native_empty").save(WorldGenerator(WorldGeneratorType.EMPTY, {}, 0).compute())
    assert [name for name in _files(out / relpath) if name != "kg.yaml"] == [name for name in _files(native) if name != "kg.yaml"]
    (imported_level,), (native_level,) = (yaml.safe_load((root / "world.yaml").read_text())["levels"] for root in (out / relpath, native))
    assert set(native_level) <= set(imported_level)
    assert {key for zone in imported_level["zones"] for key in zone} <= {"name", "polygon", "ceiling", "floor"}
    assert set(native_level["zones"][0]) <= {"name", "polygon", "ceiling", "floor"}
    assert {key for wall in imported_level["walls"] for key in wall} <= {"path", "closed", "kind", "material"}
    imported_map, native_map = (yaml.safe_load((root / "map" / "0.yaml").read_text()) for root in (out / relpath, native))
    assert {**imported_map, "origin": None} == {**native_map, "origin": None}


def test_door_entries_use_the_keys_of_a_shipped_world(converted: tuple[Path, str]) -> None:
    out, relpath = converted
    shipped = yaml.safe_load((NATIVE_WORLD / "world.yaml").read_text())
    shipped_keys = {key for level in shipped["levels"] for door in level.get("doors", []) for key in door}
    (level,) = yaml.safe_load((out / relpath / "world.yaml").read_text())["levels"]
    assert len(level["doors"]) == DOORS
    assert all(set(door) == {"name", "span", "kind"} and door["kind"] == "hinged" for door in level["doors"])
    assert {"name", "span"} <= shipped_keys


def test_rooms_walls_doors_round_trip(converted: tuple[Path, str]) -> None:
    from arena_simulation_setup.tree.World import describe

    world = _load(*converted)
    assert list(world.levels) == ["0"]
    level = world.levels["0"]
    assert [zone.name for zone in level.zones] == ROOMS
    assert [describe(world.kg, zone.name) for zone in level.zones] == ROOM_TYPES
    assert len(level.walls) == WALLS
    assert len(level.doors) == DOORS
    assert not list(level.all_static_entities)
    assert not list(level.all_dynamic_entities)
    assert all(zone.ceiling is not None and zone.ceiling.height is None for zone in level.zones)
    assert [ceiling.z for ceiling in asyncio.run(level.all_ceilings())] == [NATIVE_WALL_HEIGHT_M] * len(ROOMS)


def test_zone_names_spell_the_asa_room_values(converted: tuple[Path, str]) -> None:
    from arena_assets.semantic.taxonomy import load_spec

    rooms = load_spec().facets["room"].values
    level = _load(*converted).levels["0"]
    assert {zone.name.rpartition("_")[0] for zone in level.zones} <= set(rooms)
    assert sorted(level.zone_ref_names()) == sorted([*ROOMS, "door_2_6", "door_3_4", "door_4_5", "door_6_7", "door_6_8"])


def test_unity_xz_maps_to_arena_xy(converted: tuple[Path, str]) -> None:
    level = _load(*converted).levels["0"]
    kitchen = level.zones[0]
    assert (kitchen.polygon[0].x, kitchen.polygon[0].y) == (4.272, 4.272)
    assert (kitchen.polygon[1].x, kitchen.polygon[1].y) == (4.272, 10.68)
    assert all(corner.z == 0.0 for zone in level.zones for corner in zone.polygon)
    assert all(wall.start.z == 0.0 and wall.end.z == 0.0 for wall in level.all_walls)


def test_door_openings_cut_from_walls(converted: tuple[Path, str]) -> None:
    level = _load(*converted).levels["0"]
    walls = list(level.all_walls)
    doors = list(level.all_doors)
    assert sum(_length(w.start, w.end) for w in walls) == pytest.approx(WALL_LENGTH_M, abs=1e-3)
    assert sum(_length(d.start, d.end) for d in doors) == pytest.approx(DOOR_OPENINGS_M, abs=1e-3)
    assert {door.name for door in doors} == {"door_2_6", "door_3_4", "door_4_5", "door_6_7", "door_6_8"}
    assert all(door.kind == "hinged" for door in doors)
    assert all(door.height == NATIVE_WALL_HEIGHT_M for door in doors)
    for door in doors:
        assert not _covered(walls, (door.start.x + door.end.x) / 2, (door.start.y + door.end.y) / 2), f"{door.name} is covered by a wall"


def test_hole_offset_runs_from_the_first_ground_vertex_of_wall0(converted: tuple[Path, str]) -> None:
    house = _house(FIXTURE)
    upstream = next(door for door in house["doors"] if door["id"] == "door|2|6")
    wall = next(wall for wall in house["walls"] if wall["id"] == upstream["wall0"])
    assert [(p["x"], p["z"]) for p in wall["polygon"][:2]] == [(8.544, 10.68), (8.544, 4.272)]
    door = next(door for door in _load(*converted).levels["0"].all_doors if door.name == "door_2_6")
    assert (door.start.x, door.end.x) == (8.544, 8.544)
    assert sorted((door.start.y, door.end.y)) == pytest.approx([10.68 - upstream["holePolygon"][1]["x"], 10.68 - upstream["holePolygon"][0]["x"]], abs=1e-4)


def test_closed_exterior_door_stays_a_wall(converted: tuple[Path, str]) -> None:
    house = _house(FIXTURE)
    exterior = next(door for door in house["doors"] if door["openable"] is False)
    assert exterior["room0"] == exterior["room1"]
    level = _load(*converted).levels["0"]
    assert "door_1_4" not in {door.name for door in level.all_doors}
    centre = 6.408 - (exterior["holePolygon"][0]["x"] + exterior["holePolygon"][1]["x"]) / 2
    assert _covered(list(level.all_walls), centre, 2.136)


def test_doorframe_is_an_opening_without_a_door() -> None:
    from arena_simulation_setup.adapters.procthor import level

    house = _house(FRAME_FIXTURE)
    assert [(door["assetId"], door.get("openable")) for door in house["doors"]] == [("Doorframe_8", None), ("Doorway_1", False)]
    imported, _ = level(house)
    assert [zone.name for zone in imported.zones] == ["kitchen_2", "living_room_3"]
    assert not list(imported.all_doors)
    walls = list(imported.all_walls)
    frame = house["doors"][0]["holePolygon"]
    assert not _covered(walls, 1.89, 5.671 - (frame[0]["x"] + frame[1]["x"]) / 2)
    assert _covered(walls, 1.89, 5.671 - frame[1]["x"] - 0.1)
    closed = house["doors"][1]["holePolygon"]
    assert _covered(walls, 1.89 + (closed[0]["x"] + closed[1]["x"]) / 2, 5.671)
    grid, origin = imported.render_grid(resolution=RESOLUTION)
    assert (_patch(grid, origin, 1.89, 5.671 - (frame[0]["x"] + frame[1]["x"]) / 2) == FREE).all()
    assert (_patch(grid, origin, 1.89, 4.2) == OCCUPIED).any()


def test_runtime_loader_renders_the_map(converted: tuple[Path, str]) -> None:
    view = _view(*converted)
    assert not list(view.path.glob("*/world.yaml"))
    world = view.load()
    assert view.level_origins() is None
    assert not list(view.scenario.listall())
    level = world.levels["0"]
    png, origin = level.render(resolution=RESOLUTION)
    assert png[:8] == PNG_MAGIC
    assert png == (view.path / "map" / "0.png").read_bytes()
    assert yaml.safe_load((view.path / "map" / "0.yaml").read_text())["origin"] == pytest.approx([-0.25, 2.136 - 0.25, 0])
    grid, origin = level.render_grid(resolution=RESOLUTION)
    assert origin == pytest.approx((-0.25, 2.136 - 0.25))
    assert grid.shape == (math.ceil((10.68 - 2.136) / RESOLUTION) + 10, math.ceil(17.088 / RESOLUTION) + 10)
    assert (_patch(grid, origin, 6.4, 7.5) == FREE).all()
    assert (_patch(grid, origin, 8.544, (8.5862 + 9.622) / 2) == FREE).all()
    assert (_patch(grid, origin, 8.544, 7.0) == OCCUPIED).any()
    assert (_patch(grid, origin, 4.272, 7.5) == FREE).all()
    assert (grid[:3] == OCCUPIED).all()


def test_touch_world_renders_and_keeps_the_canonical_files(converted: tuple[Path, str]) -> None:
    out, relpath = converted
    world = out / relpath
    env = {key: value for key, value in os.environ.items() if key != "ARENA_DIR"}
    before = {name: (world / name).read_bytes() for name in LEVEL_FILES}
    subprocess.run([sys.executable, str(TOUCH_WORLD), relpath], cwd=out, env=env, check=True, capture_output=True)
    assert (world / "map.png").read_bytes()[:8] == PNG_MAGIC
    assert yaml.safe_load((world / "map.yaml").read_text())["image"] == "map.png"
    subprocess.run([sys.executable, str(TOUCH_WORLD), relpath, "--all"], cwd=out, env=env, check=True, capture_output=True)
    assert {name: (world / name).read_bytes() for name in LEVEL_FILES} == before


def test_references_are_the_default_materials(converted: tuple[Path, str]) -> None:
    identifiers = list(_view(*converted).identifiers(strict=True))
    assert {identifier.shortname for identifier in identifiers} == DEFAULT_MATERIALS


def test_split_is_streamed_once_for_ascending_houses(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import Houses

    src = _split_file(tmp_path / "val.jsonl.gz", FIXTURE, FRAME_FIXTURE, FIXTURE)
    houses = Houses(src)
    assert len(houses.house(0)["rooms"]) == 7
    src.unlink()
    assert len(houses.house(1)["rooms"]) == 2
    assert len(houses.house(2)["rooms"]) == 7
    with pytest.raises(FileNotFoundError):
        houses.house(1)


def test_stepping_back_and_closing_release_the_split_file(tmp_path: Path, open_handles) -> None:
    from arena_simulation_setup.adapters.procthor import Houses

    src = _split_file(tmp_path / "val.jsonl.gz", FIXTURE, FRAME_FIXTURE)
    houses = Houses(src)
    assert len(houses.house(1)["rooms"]) == 2
    assert open_handles(src) == 1
    assert len(houses.house(0)["rooms"]) == 7
    assert open_handles(src) == 1
    houses.close()
    assert open_handles(src) == 0
    with pytest.raises(IndexError):
        houses.house(5)
    assert open_handles(src) == 0


def test_importing_items_releases_the_split_file(tmp_path: Path, open_handles) -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    src = _split_file(tmp_path / "cache" / "val.jsonl.gz", FIXTURE, FRAME_FIXTURE)
    importer = ProcTHOR(cache=tmp_path / "cache")
    assert sorted(importer.import_items(["val/1", "val/0"], tmp_path / "worlds")) == ["val/0", "val/1"]
    assert open_handles(src) == 0


def test_house_past_the_end_of_a_split_is_an_index_error() -> None:
    from arena_simulation_setup.adapters.procthor import Houses

    houses = Houses(FIXTURE)
    with pytest.raises(IndexError):
        houses.house(1)
    assert len(houses.house(0)["rooms"]) == 7


def test_consecutive_items_convert_from_one_split(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    src = _split_file(tmp_path / "cache" / "val.jsonl.gz", FIXTURE, FRAME_FIXTURE)
    importer = ProcTHOR(cache=tmp_path / "cache")
    out = tmp_path / "procthor"
    assert [importer.convert(item, src, out) for item in ("val/0", "val/1", "val/0")] == ["procthor_val_0", "procthor_val_1", "procthor_val_0"]
    assert [zone.name for zone in _load(out, "procthor_val_1").levels["0"].zones] == ["kitchen_2", "living_room_3"]
    assert [zone.name for zone in _load(out, "procthor_val_0").levels["0"].zones] == ROOMS


def test_reconvert_replaces_the_world(converted: tuple[Path, str], tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    out, relpath = converted
    stale = out / relpath / "world.yaml"
    stale.write_text("zones: []\n")
    assert ProcTHOR(cache=tmp_path / "cache").convert(ITEM, _bare(FIXTURE, tmp_path / "bare.jsonl"), out) == relpath
    assert _files(out / relpath) == LEVEL_FILES


def test_import_item_from_a_cached_split(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    src = _split_file(tmp_path / "cache" / "val.jsonl.gz", FIXTURE)
    importer = ProcTHOR(cache=tmp_path / "cache")
    assert importer.fetch(ITEM) == src
    target = importer.import_item(ITEM, tmp_path / "procthor")
    assert _files(target) == sorted([*LEVEL_FILES, ATTRIBUTION_FILE])
    assert read_attribution(target / ATTRIBUTION_FILE).license == "Apache-2.0"
    assert [zone.name for zone in _load(tmp_path / "procthor", NAME).levels["0"].zones] == ROOMS


_DEFAULT_TARGET = """
import json
from pathlib import Path
from arena_simulation_setup.adapters.procthor import ProcTHOR
from arena_simulation_setup.tree.World import WorldIdentifier

target = ProcTHOR(cache=Path("{cache}")).import_item("{item}", ProcTHOR.default_out())
verdict = WorldIdentifier(target.name).resolve_source_sync()
print(json.dumps({{"target": str(target), "resolver": repr(verdict.resolver), "verdict": verdict.verdict.name, "path": str(verdict.path)}}))
"""


def test_default_target_is_read_by_the_world_resolvers(tmp_path: Path) -> None:
    _split_file(tmp_path / "cache" / "val.jsonl.gz", FIXTURE)
    env = {key: value for key, value in os.environ.items() if key not in ("ARENA_ASSETS_DIR_IMPORTED", "ARENA_WORLD_PATH")} | {"ARENA_ASSETS_DIR": str(tmp_path / "assets")}
    proc = subprocess.run([sys.executable, "-c", _DEFAULT_TARGET.format(cache=tmp_path / "cache", item=ITEM)], capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    seen = json.loads(proc.stdout.splitlines()[-1])
    target = tmp_path / "assets" / "_imported" / "worlds" / NAME
    assert (seen["target"], seen["verdict"], seen["path"]) == (str(target), "HIT", str(target))
    assert "DynamicPathResolver" in seen["resolver"]


def test_items_cover_every_split(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import SPLITS, ProcTHOR, split_item

    items = list(ProcTHOR(cache=tmp_path).items())
    assert len(items) == sum(SPLITS.values()) == 12_000
    assert items[0] == "train/0"
    assert items[-1] == "test/999"
    assert split_item("val/999") == ("val", 999)
    with pytest.raises(ValueError):
        split_item("val/1000")
    with pytest.raises(ValueError):
        split_item("houses/0")


def test_attribution(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    attribution = ProcTHOR(cache=tmp_path).attribution(ITEM)
    assert attribution.license == "Apache-2.0"
    assert attribution.source == "procthor"
    assert attribution.url == "https://media.githubusercontent.com/media/allenai/procthor-10k/main/val.jsonl.gz"
    assert "Deitke" in attribution.authors


def test_fixture_attribution_names_every_fixture_file() -> None:
    attribution = yaml.safe_load((FIXTURES / "ATTRIBUTION.yaml").read_text())
    assert attribution["license"] == "Apache-2.0"
    assert sorted(name for item in attribution["items"] for name in item) == sorted(path.relative_to(FIXTURES).as_posix() for path in FIXTURES.rglob("*") if path.is_file() and path.name != "ATTRIBUTION.yaml")


@pytest.mark.network
def test_push_preflight_finds_nothing_dangling(converted: tuple[Path, str]) -> None:
    asset = pytest.importorskip("arena_cli.asset")
    from arena_simulation_setup.tree import DynamicPaths

    view = _view(*converted)
    previous = DynamicPaths.WORLD.path
    try:
        assert asset._world_dangling(view, view.path) == []
    finally:
        DynamicPaths.WORLD.path = previous


@pytest.mark.network
def test_fetch_test_split_and_read_a_house(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR, house_objects, level

    importer = ProcTHOR(cache=tmp_path / "cache")
    src = importer.fetch("test/0")
    assert src.name == "test.jsonl.gz"
    assert src.stat().st_size > 1_000_000
    assert importer.fetch("test/1") == src
    house = importer.house(src, 0)
    description, _ = level(house)
    assert description.zones
    assert list(description.all_walls)
    assert all(obj["assetId"] for obj in house_objects(house))


def test_pose_maps_unity_rotation_and_box_centre_to_arena() -> None:
    from arena_simulation_setup.adapters.procthor import pose

    upright = pose({"position": {"x": 2.0, "y": 0.5, "z": 3.0}, "rotation": {"x": 0, "y": 0, "z": 0}}, (0.1, 0.2, 0.5))
    assert (upright.position.x, upright.position.y, upright.position.z) == pytest.approx((2.1, 3.2, 0.0))
    assert upright.orientation.to_yaw() == pytest.approx(math.pi)
    turned = pose({"position": {"x": 2.0, "y": 0.5, "z": 3.0}, "rotation": {"x": 0, "y": 90, "z": 0}}, (0.1, 0.2, 0.5))
    assert (turned.position.x, turned.position.y, turned.position.z) == pytest.approx((2.2, 2.9, 0.0))
    assert turned.orientation.to_yaw() == pytest.approx(math.pi / 2)


def test_house_objects_walk_receptacle_children_in_order() -> None:
    from arena_simulation_setup.adapters.procthor import house_objects

    placed = list(house_objects(_house(FIXTURE)))
    assert len(placed) == 142
    assert [obj["id"] for obj in placed[:3]] == ["DiningTable|2|0|0", "Ladle|surface|2|1", "Vase|surface|2|10"]


def test_house_without_objects_converts_without_the_object_importer(converted: tuple[Path, str]) -> None:
    assert not list(_load(*converted).levels["0"].all_static_entities)


def test_furniture_is_placed_as_native_static_entities(furnished) -> None:
    import shapely

    out, relpath, _ = furnished
    upstream = _furnished_objects()
    level = _load(out, relpath).levels["0"]
    entities = list(level.all_static_entities)
    assert sorted(entity.name for entity in entities) == sorted(f"{asset}_{n}" for asset in FURNITURE for n in range(sum(obj["assetId"] == asset for obj in upstream)))
    assert {entity.model.shortname for entity in entities} == {f"Common/Object/{asset}" for asset in FURNITURE}
    (raw,) = yaml.safe_load((out / relpath / "world.yaml").read_text())["levels"]
    assert all(set(entity) == {"name", "pose", "model"} for entity in raw["objects"])
    kitchen = shapely.Polygon([(point.x, point.y) for point in next(zone for zone in level.zones if zone.name == "kitchen_2").polygon])
    assert [entity.name for entity in entities if kitchen.contains(shapely.Point(entity.pose.position.x, entity.pose.position.y))] == ["Dining_Table_203_1_0"]


def test_placed_models_match_the_hab_placement_corner_for_corner(furnished) -> None:
    out, relpath, imported = furnished
    for entity in _load(out, relpath).levels["0"].all_static_entities:
        native = _native_corners(entity, imported)
        hab = _hab_corners(entity.model.name, native.mean(axis=0))
        assert _corner_gap(native, hab) < PLACEMENT_TOLERANCE_M, entity.name


def test_placed_footprints_overlay_the_hab_footprints(furnished) -> None:
    import shapely

    out, relpath, _ = furnished
    upright = [entity for entity in _load(out, relpath).levels["0"].all_static_entities if entity.model.name != "BaseballBat_3"]
    assert len(upright) == 3
    for entity in upright:
        footprint = asyncio.run(entity.footprint())
        assert footprint is not None
        hab = shapely.MultiPoint([tuple(corner[:2]) for corner in _hab_corners(entity.model.name, np.array([*footprint.centroid.coords[0], 0.0]))]).convex_hull
        assert footprint.intersection(hab).area / footprint.union(hab).area > 0.98, entity.name


def test_receptacle_children_keep_their_height(furnished) -> None:
    out, relpath, _ = furnished
    upstream = {obj["assetId"]: obj for obj in _furnished_objects() if "|surface|" in obj["id"]}
    assert set(upstream) == {"Dog_Statue_4", "BaseballBat_3"}
    entities = {entity.model.name: entity for entity in _load(out, relpath).levels["0"].all_static_entities}
    statue = entities["Dog_Statue_4"]
    assert statue.pose.position.z == pytest.approx(upstream["Dog_Statue_4"]["position"]["y"] - 0.2015 / 2, abs=PLACEMENT_TOLERANCE_M)
    assert statue.pose.position.z > 0.3
    from arena_simulation_setup.utils.geometry import Vector3

    assert (entities["BaseballBat_3"].pose.orientation * Vector3(0.0, 0.0, 1.0)).z == pytest.approx(0.0, abs=1e-3)


def test_furnished_closure_resolves_to_the_import_root(furnished) -> None:
    out, relpath, imported = furnished
    view = _view(out, relpath)
    objects = {identifier for identifier in view.identifiers(strict=True) if identifier.shortname.startswith("Common/Object/")}
    assert {identifier.shortname for identifier in objects} == {f"Common/Object/{asset}" for asset in FURNITURE}
    for identifier in objects:
        hit = identifier.resolve_source_sync()
        assert hit.path == imported / identifier.relpath()
        assert (hit.path / "ATTRIBUTION.yaml").is_file()


def test_reconvert_reuses_the_imported_objects(furnished, tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.procthor import ProcTHOR

    out, relpath, imported = furnished
    stamps = {path: path.stat().st_mtime_ns for path in imported.rglob("annotation.yaml")}
    cache = tmp_path / "ai2thor"
    importer = ProcTHOR(cache=tmp_path / "cache", objects=_seeded_objects(cache), objects_out=imported)
    assert importer.convert(ITEM, _furnished_house(tmp_path / "val.jsonl"), tmp_path / "worlds") == relpath
    assert {path: path.stat().st_mtime_ns for path in imported.rglob("annotation.yaml")} == stamps
    assert not list(cache.glob("stage-*"))
    assert (tmp_path / "worlds" / relpath / "world.yaml").read_text() == (out / relpath / "world.yaml").read_text()


@pytest.mark.network
def test_push_preflight_names_only_the_locally_imported_objects(furnished) -> None:
    asset = pytest.importorskip("arena_cli.asset")
    from arena_simulation_setup.tree import DynamicPaths

    out, relpath, imported = furnished
    view = _view(out, relpath)
    previous = DynamicPaths.WORLD.path
    try:
        dangling = asset._world_dangling(view, view.path)
    finally:
        DynamicPaths.WORLD.path = previous
    assert sorted(dangling) == sorted(f"Common/Object/{name} (only at {imported / 'Common' / 'Object' / name})" for name in FURNITURE)
