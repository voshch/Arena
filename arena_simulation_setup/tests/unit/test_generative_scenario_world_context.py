from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.formats import first_level, world_roles
from arena_simulation_setup.generative.scenario.occupancy import level_grid, level_of, load_world, world_statics
from arena_simulation_setup.generative.scenario.populate import MIN_ROUTE_CLEARANCE
from arena_simulation_setup.generative.scenario.promptgen.world import LoadedWorld, WorldContext
from arena_simulation_setup.generative.scenario.zone_screen import screen_zones

pytest.importorskip("numpy")


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _write_world(root: Path, zones: dict[str, list[list[float]]]) -> None:
    root.mkdir(parents=True)
    levels = [{"id": "0", "zones": [{"name": n, "polygon": p, "floor": False} for n, p in zones.items()]}]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": levels}))


def _robot(start: tuple[float, float], goal: tuple[float, float]) -> dict:
    return {"robots": [{"start": [start[0], start[1], 0.0], "phases": [{"goto": [goal[0], goal[1], 0.0]}]}]}


def _detour(tmp_path: Path) -> Path:
    root = tmp_path / "worlds" / "detour"
    zones = {"south": _box(0, 0, 10, 3), "east": _box(7, 3, 10, 10), "north": _box(0, 10, 10, 13), "middle": _box(0, 5, 5.5, 8)}
    _write_world(root, zones)
    return root


def test_homes_are_screened_against_the_planned_route(tmp_path: Path) -> None:
    world = WorldContext.load(_detour(tmp_path), _robot((1.5, 1.5), (1.5, 11.5)))
    assert max(x for x, _y in world.route) > 7.0
    assert "east" not in world.homes
    assert "middle" in world.homes


def test_resolve_searches_outward_for_a_valid_point(tmp_path: Path) -> None:
    root = tmp_path / "worlds" / "open"
    _write_world(root, {"room": _box(0, 0, 10, 10)})
    world = WorldContext.load(root, _robot((1.0, 1.0), (9.0, 9.0)))
    assert world.valid((5.0, 5.0)) is not None
    pt, why = world.resolve((5.0, 5.0), within=2.0)
    assert world.valid(pt) is None and why.startswith("moved")
    assert 1.5 <= math.dist(pt, (5.0, 5.0)) <= 1.65


def test_screening_on_the_loaded_grid_matches_rendering_the_world(tmp_path: Path) -> None:
    root = _detour(tmp_path)
    world = WorldContext.load(root, _robot((1.5, 1.5), (1.5, 11.5)))
    assert world.grid is not None
    rendered = screen_zones(world.zones, world_root=root, route=world.route, statics=world_statics(root), min_route_clearance=MIN_ROUTE_CLEARANCE)
    reused = screen_zones(world.zones, world_root=root, route=world.route, min_route_clearance=MIN_ROUTE_CLEARANCE, clearance_map=(world.grid.dist, world.grid.res, world.grid.ox, world.grid.oy))
    assert reused == rendered
    assert world.homes == frozenset(n for n, v in rendered.items() if v.ref_safe)


@pytest.fixture
def furnished(tmp_path: Path) -> Path:
    from arena_simulation_setup.tree import DynamicPaths

    root = tmp_path / "furnished"
    root.mkdir()
    objects = [
        {"name": "desk_a", "pose": [2.0, 2.0, 0.0], "model": "Hospital/TestDesk"},
        {"name": "desk_b", "pose": [6.0, 3.0, 1.2], "model": "Hospital/TestDesk"},
        {"name": "plant", "pose": [8.0, 5.0, 0.0], "model": "Hospital/TestPlant"},
    ]
    zone = {"name": "lobby", "polygon": _box(0, 0, 10, 6), "floor": False}
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": [zone], "objects": objects}]}))
    desk = tmp_path / "assets" / "Hospital" / "Object" / "TestDesk"
    desk.mkdir(parents=True)
    (desk / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[0.0, 2.42], [-2.76, 0.10], [0.0, 1.0]]}))
    (tmp_path / "assets" / "Hospital" / "Object" / "TestPlant").mkdir(parents=True)
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path / "assets"
    yield root
    DynamicPaths.ARENA.path = previous


def test_statics_are_the_runtime_footprint_of_each_object(furnished: Path) -> None:
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.World.view import WorldView, world_assets

    view = WorldView(furnished.resolve())
    world = view.load()
    level = level_of(world)
    assert level is not None

    async def each() -> list:
        return [await obj.footprint() for obj in level.objects]

    with world_assets(view, world):
        expected = Identifier._run_sync(each())
    statics = {s["name"]: s for s in world_statics(furnished)}
    for obj, footprint in zip(level.objects, expected, strict=True):
        got = statics[obj.name]["polygon"]
        if footprint is None:
            x, y = obj.pose.position.x, obj.pose.position.y
            assert got == [(x + 1.0, y - 1.0), (x + 1.0, y + 1.0), (x - 1.0, y + 1.0), (x - 1.0, y - 1.0)]
        else:
            assert got == list(footprint.exterior.coords)[:-1]
    assert statics["desk_a"]["polygon"] != statics["desk_b"]["polygon"]
    loaded = load_world(furnished)
    assert world_statics(furnished, loaded) == world_statics(furnished)
    with_world, fresh = level_grid(furnished, statics.values(), loaded), level_grid(furnished, statics.values())
    assert with_world is not None and fresh is not None
    assert (with_world[0] == fresh[0]).all() and with_world[1:] == fresh[1:]


_COUNT_READS = """
import json, sys
from pathlib import Path
opened, spawned = [], []

def hook(event, args):
    if event == "open" and str(args[0]).endswith("annotation.yaml"):
        opened.append(Path(str(args[0])).parent.name)
    elif event.startswith(("subprocess.", "os.posix_spawn", "os.exec", "os.spawn", "os.fork")):
        spawned.append(event)

sys.addaudithook(hook)
from arena_simulation_setup.generative.scenario.occupancy import world_statics
statics = world_statics(sys.argv[1])
print(json.dumps({"opened": sorted(opened), "spawned": spawned, "resolved": {s["name"]: s["resolved"] for s in statics}}))
"""


def test_statics_read_each_object_asset_once_and_an_unknown_asset_stays_offline(furnished: Path) -> None:
    doc = yaml.safe_load((furnished / "world.yaml").read_text())
    doc["levels"][0]["objects"].append({"name": "ghost", "pose": [1.0, 5.0, 0.0], "model": "Hospital/NoSuchAsset"})
    (furnished / "world.yaml").write_text(yaml.safe_dump(doc))
    env = {**os.environ, "ARENA_BUCKETS": ",", "ARENA_ASSETS_DIR_LOCAL": str(furnished.parent / "assets")}
    done = subprocess.run([sys.executable, "-c", _COUNT_READS, str(furnished)], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"opened": ["TestDesk", "TestDesk"], "spawned": [], "resolved": {"desk_a": True, "desk_b": True, "plant": False, "ghost": False}}


def _fields(world: WorldContext) -> tuple:
    assert world.grid is not None
    return (world.name, world.zones, world.roles, world.robot, world.start, world.goal, world.route, world.homes, world.kind, world.level, world.grid.occupied.tobytes(), world.grid.res, world.grid.ox, world.grid.oy)


def test_one_read_world_serves_every_base(tmp_path: Path) -> None:
    root = _detour(tmp_path)
    parts = LoadedWorld.read(root)
    bases = [_robot((1.5, 1.5), (1.5, 11.5)), _robot((8.5, 4.0), (2.0, 11.5))]
    shared = [WorldContext.load(root, doc, parts) for doc in bases]
    assert [_fields(w) for w in shared] == [_fields(WorldContext.load(root, doc)) for doc in bases]
    assert shared[0].grid is shared[1].grid
    shared[0].zones[0]["name"] = "renamed"
    assert parts.doc["zones"][0]["name"] != "renamed" and shared[1].zones[0]["name"] != "renamed"


def test_formats_read_the_loaded_world_as_they_read_the_root(furnished: Path) -> None:
    loaded = load_world(furnished)
    assert first_level(furnished, loaded) == first_level(furnished)
    assert world_roles(furnished, loaded) == world_roles(furnished)


def _kg_world() -> WorldContext:
    from arena_simulation_setup.generative.scenario.zones import Role

    def zone(name: str, x0: float) -> dict:
        return {"name": name, "corners": [{"x": x, "y": y} for x, y in ((x0, 0.0), (x0 + 6.0, 0.0), (x0 + 6.0, 6.0), (x0, 6.0))]}

    zones = [zone("open_work_area", 0.0), zone("studio", 10.0), zone("corridor", 20.0)]
    roles = {"open_work_area": Role.BREAK, "studio": Role.WORK, "corridor": Role.TRANSIT}
    robot = {"start": [21.0, 3.0, 0.0], "goal": [25.0, 3.0]}
    return WorldContext(name="kg", root=Path("kg"), zones=zones, roles=roles, robot=robot, grid=None, start=(21.0, 3.0), goal=(25.0, 3.0), route=[(21.0, 3.0), (25.0, 3.0)], homes=frozenset(), kind="office")


def test_the_work_zone_follows_the_kg_roles() -> None:
    assert _kg_world().work_zone() == "studio"


def test_busy_pedestrians_start_in_the_kg_work_zones() -> None:
    from arena_simulation_setup.generative.scenario.promptgen.step_a import inject_busy

    doc: dict = {"humans": []}
    names = inject_busy(doc, _kg_world(), 2.0, seed=0)
    assert names and {a["pose"] for a in doc["humans"]} == {"studio"}
