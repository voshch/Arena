"""SidewalkBench world importer: a synthetic scan converted through the real code, read back through the world tree."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
import yaml
from pxr import Gf, Usd, UsdGeom, UsdVol

SCENE = "ucla_crosswalk"
WORLD = "sidewalkbench_ucla_crosswalk"
GROUND = -1.5
ROWS, COLUMNS = 60, 200
RESOLUTION = 0.1
SIDEWALK_ROWS = (10, 30)
CROSSWALK_COLUMNS = (80, 100)
SLICE_BOXES = {"post": (4.0, 0.0, 1.0), "curb": (6.0, 0.0, 0.2), "canopy": (8.0, 2.0, 2.5)}
GAUSSIANS = 6
CENTERLINES = {
    "centerlines": [
        {"id": "centerline_0", "closed": False, "crosswalk": False, "xy": []},
        {"id": "centerline_1", "closed": False, "crosswalk": False, "xy": []},
        {"id": "centerline_2", "closed": False, "crosswalk": True, "xy": []},
    ]
}


def box(x: float, bottom: float, top: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed 0.5 m box over y 4.0 to 4.5 from GROUND + bottom to GROUND + top."""
    corners = np.array([(cx, cy, GROUND + cz) for cz in (bottom, top) for cy in (4.0, 4.5) for cx in (x, x + 0.5)], np.float32)
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return corners, np.array([tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))], np.uint32)


def mesh_parts() -> tuple[np.ndarray, np.ndarray]:
    ground = np.array([(0, 0, GROUND), (20, 0, GROUND), (20, 6, GROUND), (0, 6, GROUND)], np.float32)
    parts = [(ground, np.array([(0, 1, 2), (0, 2, 3)], np.uint32))]
    parts += [box(x, bottom, top) for x, bottom, top in SLICE_BOXES.values()]
    points = np.concatenate([part[0] for part in parts])
    offsets = np.cumsum([0, *[len(part[0]) for part in parts[:-1]]])
    return points, np.concatenate([part[1] + offset for part, offset in zip(parts, offsets, strict=True)])


def write_mesh(path: Path) -> int:
    points, triangles = mesh_parts()
    faces = np.zeros(len(triangles), [("n", "u1"), ("i", "<u4", (3,))])
    faces["n"] = 3
    faces["i"] = triangles
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(triangles)}\nproperty list uchar uint vertex_indices\nend_header\n"
    path.write_bytes(header.encode() + points.astype("<f4").tobytes() + faces.tobytes())
    return len(triangles)


def write_gaussians(path: Path) -> None:
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3", "lod"]
    table = np.zeros(GAUSSIANS, [(name, "<f4") for name in names])
    table["x"] = [5.0, 12.0, 15.0, 1.0, 1000.0, 6.0]
    table["y"] = [2.0, 2.0, 5.0, 1.0, 0.0, 2.0]
    table["z"] = [-1.4, -1.0, 0.0, -1.5, 0.0, -1.4]
    for axis in range(3):
        table[f"scale_{axis}"] = np.log(0.05)
    table["scale_0"][5] = np.log(5.0)
    table["rot_0"] = [0.6, 1, 1, 1, 1, 1]
    table["rot_2"] = [0.8, 0, 0, 0, 0, 0]
    table["opacity"] = 2.0
    header = "".join(f"property float {name}\n" for name in names)
    path.write_bytes(f"ply\nformat binary_little_endian 1.0\nelement vertex {GAUSSIANS}\n{header}end_header\n".encode() + table.tobytes())


def write_grid(path: Path) -> None:
    sidewalk = np.zeros((ROWS, COLUMNS), np.uint8)
    sidewalk[slice(*SIDEWALK_ROWS)] = 1
    crosswalk = np.zeros((ROWS, COLUMNS), np.uint8)
    crosswalk[slice(*SIDEWALK_ROWS), slice(*CROSSWALK_COLUMNS)] = 1
    np.savez(
        path,
        semantic=np.where(sidewalk == 1, 10, 11).astype(np.int16),
        sidewalk=sidewalk,
        crosswalk=crosswalk,
        origin_xy=np.array([0.0, 0.0], np.float32),
        resolution_m=np.array([RESOLUTION], np.float32),
        semantic_z=np.full((ROWS, COLUMNS), GROUND, np.float32),
        semantic_ground_like=np.ones((ROWS, COLUMNS), bool),
    )


def line(start: tuple[float, float], end: tuple[float, float], count: int) -> list[list[float]]:
    return [[start[0] + (end[0] - start[0]) * i / (count - 1), start[1] + (end[1] - start[1]) * i / (count - 1)] for i in range(count)]


def arc() -> list[list[float]]:
    angles = np.linspace(-np.pi / 2, 0.0, 12)
    return [[float(1.0 + 3.0 * np.cos(a)), float(4.0 + 3.0 * np.sin(a))] for a in angles]


def cone(name: str, xy: list[float]) -> dict:
    return {"id": name, "xy": xy, "yaw_deg": 90, "settled": {"xyz": [*xy, GROUND + 0.01], "quat_xyzw": [0, 0, 0, 1]}}


def annotation(cones: bool) -> dict:
    routes = [
        {"id": "route_1", "mode": "short_horizon", "centerline_ids": ["centerline_0"], "start": {"xy": [1.0, 2.0], "yaw_deg": 0}, "goal": {"xy": [7.0, 2.0]}, "centerline_xy": line((1, 2), (7, 2), 13), "static_pedestrians": [], "dynamic_pedestrians": [], "region_xy": []},
        {"id": "route_2", "mode": "short_horizon", "centerline_ids": ["centerline_1"], "start": {"xy": [1.0, 1.0], "yaw_deg": 0}, "goal": {"xy": [4.0, 4.0]}, "centerline_xy": arc(), "static_pedestrians": [], "dynamic_pedestrians": [], "region_xy": []},
        {"id": "route_3", "mode": "short_horizon", "centerline_ids": ["centerline_2"], "start": {"xy": [9.0, 1.0], "yaw_deg": 90}, "goal": {"xy": [9.0, 3.0]}, "centerline_xy": line((9, 1), (9, 3), 5), "static_pedestrians": [], "dynamic_pedestrians": [], "region_xy": []},
        {
            "id": "route_4",
            "mode": "long_horizon",
            "centerline_ids": ["centerline_0"],
            "start": {"xy": [1.0, 2.0], "yaw_deg": 0},
            "goal": {"xy": [19.0, 2.0]},
            "centerline_xy": line((1, 2), (19, 2), 19),
            "static_pedestrians": [{"xy": [3.0, 2.0], "yaw_deg": 90, "identity": "lab"}],
            "dynamic_pedestrians": [{"xy": [15.0, 2.0], "identity": "lab"}],
            "region_xy": [],
        },
    ]
    return {"version": 3, "scene": SCENE, "cones": [cone("C1", [5.2, 2.4]), cone("C2", [18.0, 2.5])] if cones else [], "routes": routes}


def seed(cache: Path, cones: bool = False) -> Path:
    scene = cache / SCENE
    scene.mkdir(parents=True)
    (scene / "annotation.json").write_text(json.dumps(annotation(cones)))
    (scene / "centerlines.json").write_text(json.dumps(CENTERLINES))
    write_grid(scene / "semantic_grid_0p1m.npz")
    write_mesh(scene / "mesh.ply")
    write_gaussians(scene / "gs.ply")
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from arena_simulation_setup.adapters.sidewalkbench import SidewalkBenchWorld

    return SidewalkBenchWorld(cache=seed(tmp_path / "cache"))


@pytest.fixture()
def imported(importer, tmp_path: Path) -> Path:
    return importer.import_item(SCENE, tmp_path / "worlds")


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _load(target: Path):
    from arena_simulation_setup.tree.World import WorldView

    return WorldView(target).load()


def test_registry_target_is_the_class() -> None:
    from arena_simulation_setup.adapters.sidewalkbench import SidewalkBenchWorld

    from arena_assets.importers import IMPORTERS

    assert IMPORTERS.load("world", "sidewalkbench") is SidewalkBenchWorld
    assert SidewalkBenchWorld.meta.kind == "world"


def test_items_are_the_eleven_scenes(importer) -> None:
    items = list(importer.items())
    assert len(items) == 11
    assert (items[0], items[-1]) == ("9am", "ucla_sidewalk_backward")


@pytest.mark.parametrize("item", ["ucla", "", "UCLA_crosswalk", "0"])
def test_item_outside_the_scenes_is_refused(importer, item: str) -> None:
    with pytest.raises(ValueError, match="ucla_sidewalk_backward"):
        importer.fetch(item)


def test_fetch_keeps_the_cached_scene_files(importer) -> None:
    assert importer.fetch(SCENE) == importer.cache / SCENE


def test_import_writes_the_world_the_assets_the_scenarios_and_the_attribution(imported: Path, tmp_path: Path) -> None:
    assert imported == tmp_path / "worlds" / WORLD
    bundle = f"assets/Common/Object/{WORLD}"
    assert _files(imported) == sorted(
        [
            "ATTRIBUTION.yaml",
            "kg.yaml",
            "map/0.png",
            "map/0.yaml",
            "world.yaml",
            f"{bundle}/annotation.yaml",
            f"{bundle}/{WORLD}.collider.obj",
            f"{bundle}/{WORLD}.gaussians.usdc",
            f"{bundle}/{WORLD}.sdf/{WORLD}.collider.obj",
            f"{bundle}/{WORLD}.sdf/{WORLD}.sdf",
            f"{bundle}/{WORLD}.usda",
            "scenarios/crosswalk_0/scenario.yaml",
            "scenarios/curve_0/scenario.yaml",
            "scenarios/long_0/scenario.yaml",
            "scenarios/straight_0/scenario.yaml",
        ]
    )


def test_zones_are_the_sidewalk_pieces_and_the_crosswalk_cells(imported: Path) -> None:
    zones = {zone.name: shapely.Polygon([(point.x, point.y) for point in zone.polygon]) for zone in _load(imported).levels["0"].zones}
    assert sorted(zones) == ["crosswalk", "sidewalk_0", "sidewalk_1"]
    assert zones["crosswalk"].area == pytest.approx(2.0 * 2.0)
    assert sorted(round(zones[name].area, 3) for name in ("sidewalk_0", "sidewalk_1")) == pytest.approx([16.0, 20.0])
    assert zones["crosswalk"].bounds == pytest.approx((8.0, 1.0, 10.0, 3.0))
    assert not any(zones["crosswalk"].intersection(zones[name]).area > 1e-6 for name in ("sidewalk_0", "sidewalk_1"))


def test_zones_along_a_diagonal_crosswalk_border_tile_the_walkway_without_gaps() -> None:
    from arena_simulation_setup.adapters.sidewalkbench import Grid, zones

    rows, columns = np.indices((ROWS, COLUMNS))
    sidewalk = (rows >= SIDEWALK_ROWS[0]) & (rows < SIDEWALK_ROWS[1])
    shift = columns - (rows - SIDEWALK_ROWS[0])
    crosswalk = sidewalk & (shift >= CROSSWALK_COLUMNS[0]) & (shift < CROSSWALK_COLUMNS[1])
    grid = Grid(origin=(0.0, 0.0), resolution=RESOLUTION, sidewalk=sidewalk, crosswalk=crosswalk, labeled=sidewalk, ground=np.zeros((ROWS, COLUMNS)))
    shapes = {zone.name: shapely.Polygon([(point.x, point.y) for point in zone.polygon]) for zone in zones(grid)}
    assert sorted(shapes) == ["crosswalk", "sidewalk_0", "sidewalk_1"]
    union = shapely.union_all(list(shapes.values()))
    assert isinstance(union, shapely.Polygon)
    assert not union.interiors
    assert union.area == pytest.approx(sum(shape.area for shape in shapes.values()))
    assert all(shapes["crosswalk"].intersection(shapes[name]).length > 1.0 for name in ("sidewalk_0", "sidewalk_1"))
    assert all(shape.is_valid for shape in shapes.values())


def test_outer_pieces_sit_on_the_stored_precision_and_stay_valid_once_rounded() -> None:
    from arena_simulation_setup.adapters.sidewalkbench import outer_pieces

    walkway = shapely.box(0.0, 0.0, 10.0, 3.0)
    crossing = shapely.Polygon([(4.0, -1.0), (19.0 / 3.0, 4.0 / 3.0), (38.0 / 7.0, 5.0), (11.0 / 3.0, 5.0)])
    pieces = outer_pieces(shapely.difference(walkway, crossing))
    assert len(pieces) == 2
    for piece in pieces:
        coords = list(piece.exterior.coords)
        assert all(abs(value * 1e4 - round(value * 1e4)) < 1e-6 for point in coords for value in point)
        assert shapely.Polygon([(round(x, 4), round(y, 4)) for x, y in coords]).is_valid


def test_obstacle_outline_is_the_post_alone_not_the_sidewalk_edge_curb_or_canopy(imported: Path) -> None:
    level = _load(imported).levels["0"]
    xs = [coordinate for wall in level.all_walls for point in (wall.start, wall.end) for coordinate in (point.x,)]
    ys = [coordinate for wall in level.all_walls for point in (wall.start, wall.end) for coordinate in (point.y,)]
    assert level.all_walls
    assert (min(xs), max(xs)) == pytest.approx((4.0, 4.5), abs=1.5 * RESOLUTION)
    assert (min(ys), max(ys)) == pytest.approx((4.0, 4.5), abs=1.5 * RESOLUTION)


def test_mesh_and_gaussians_ride_in_one_object_at_the_scene_center(imported: Path) -> None:
    objects = _load(imported).levels["0"].objects
    assert [obstacle.name for obstacle in objects] == [WORLD]
    assert (objects[0].pose.position.x, objects[0].pose.position.y, objects[0].pose.position.z) == pytest.approx((10.0, 3.0, 0.0))
    assert objects[0].model.name == WORLD


def test_gaussians_in_the_crop_are_kept_and_the_wrapper_registers_the_scene(imported: Path) -> None:
    bundle = imported / "assets" / "Common" / "Object" / WORLD
    gaussians = Usd.Stage.Open(str(bundle / f"{WORLD}.gaussians.usdc"))
    fields = [prim for prim in gaussians.Traverse() if prim.IsA(UsdVol.ParticleField)]
    assert len(fields) == 1
    positions = np.array(fields[0].GetAttribute("positions").Get())
    np.testing.assert_allclose(positions, [[5.0, 2.0, -1.4], [12.0, 2.0, -1.0], [15.0, 5.0, 0.0], [1.0, 1.0, -1.5]], atol=1e-5)
    first = Gf.Quatf(fields[0].GetAttribute("orientations").Get()[0])
    assert first.GetReal() == pytest.approx(0.6)
    wrapper = Usd.Stage.Open(str(bundle / f"{WORLD}.usda"))
    splat = wrapper.GetPrimAtPath(f"/{WORLD}/splat")
    shift = Gf.Matrix4d(UsdGeom.Xformable(splat).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    assert tuple(shift.ExtractTranslation()) == pytest.approx((-10.0, -3.0, 1.5))


def test_collider_holds_every_mesh_triangle_and_the_box_is_tiny(imported: Path, tmp_path: Path) -> None:
    bundle = imported / "assets" / "Common" / "Object" / WORLD
    faces = write_mesh(tmp_path / "mesh.ply")
    assert sum(1 for text in (bundle / f"{WORLD}.collider.obj").read_text().splitlines() if text.startswith("f ")) == faces
    annotation = yaml.safe_load((bundle / "annotation.yaml").read_text())
    assert annotation["collider"] == f"{WORLD}.collider.obj"
    assert float(np.abs(np.array(annotation["bounding_box"], float)).max()) < 0.1


def test_scenarios_are_named_by_kind_with_start_and_goal(imported: Path) -> None:
    from arena_simulation_setup.tree.World.Scenario import parse_scenario

    scenarios = {path.parent.name: yaml.safe_load(path.read_text()) for path in sorted((imported / "scenarios").glob("*/scenario.yaml"))}
    assert sorted(scenarios) == ["crosswalk_0", "curve_0", "long_0", "straight_0"]
    for scenario in scenarios.values():
        assert len(parse_scenario(scenario).robots) == 1
    straight = scenarios["straight_0"]["robots"][0]
    assert straight["start"] == {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.0}
    assert straight["phases"] == [{"goto": [7.0, 2.0, 0.0]}]
    assert "objects" not in scenarios["straight_0"]
    assert "humans" not in scenarios["straight_0"]


def test_long_route_pedestrians_stand_or_walk_toward_the_route_start(imported: Path) -> None:
    humans = {human["name"]: human for human in yaml.safe_load((imported / "scenarios" / "long_0" / "scenario.yaml").read_text())["humans"]}
    assert humans["ped_static_0"]["waypoints"] == [[3.0, 2.0]]
    assert humans["ped_static_0"]["pose"] == pytest.approx([3.0, 2.0, np.pi / 2], abs=1e-3)
    walker = humans["ped_walker_0"]
    assert 1.1 <= walker["velocity"] <= 1.65
    assert walker["agent"] == {"agent_type": "adult", "desired_velocity": walker["velocity"]}
    assert walker["waypoint_mode"] == "once"
    assert walker["waypoints"][0] == pytest.approx([15.0, 2.0])
    assert walker["waypoints"][-1] == pytest.approx([1.0, 2.0])
    assert walker["pose"][2] == pytest.approx(np.pi, abs=1e-3)


def test_cones_within_range_of_the_route_are_objects_at_their_settled_height(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.sidewalkbench import CONE_ASSET, Grid, frame_of, scenarios

    grid_file = seed(tmp_path / "cache", cones=True) / SCENE / "semantic_grid_0p1m.npz"
    grid = Grid.read(grid_file)
    frame = frame_of(grid, (1.0, 2.0))
    built = scenarios(annotation(cones=True), CENTERLINES, frame, grid, shapely.Polygon())
    near = built["straight_0"]["objects"]
    assert [cone["name"] for cone in near] == ["cone_C1"]
    assert near[0]["model"] == CONE_ASSET
    assert near[0]["pose"] == pytest.approx({"x": 5.2, "y": 2.4, "z": 0.01, "yaw": np.pi / 2}, abs=1e-3)
    assert [cone["name"] for cone in built["long_0"]["objects"]] == ["cone_C1", "cone_C2"]
    assert "objects" not in built["crosswalk_0"]


def test_route_end_next_to_an_obstacle_slides_along_the_centerline_until_clear() -> None:
    from arena_simulation_setup.adapters.sidewalkbench import END_CLEARANCE_M, clear_end

    centerline = shapely.LineString([(0.0, 0.0), (10.0, 0.0)])
    post = shapely.box(-0.2, 0.25, 0.2, 0.6)
    start = clear_end((0.0, 0.0), centerline, post, from_end=False)
    assert start[1] == 0.0
    assert post.distance(shapely.Point(start)) >= END_CLEARANCE_M
    assert post.distance(shapely.Point(start[0] - 0.05, 0.0)) < END_CLEARANCE_M
    goal = clear_end((10.0, 0.0), centerline, shapely.box(9.8, -0.6, 10.2, -0.25), from_end=True)
    assert goal[0] < 10.0
    assert clear_end((5.0, 0.0), centerline, post, from_end=False) == (5.0, 0.0)
    assert clear_end((0.0, 0.0), centerline, shapely.Polygon(), from_end=False) == (0.0, 0.0)


def test_route_kind_follows_mode_then_crosswalk_then_turning() -> None:
    from arena_simulation_setup.adapters.sidewalkbench import route_kind

    crosswalks = {"a": False, "b": True}
    straight = line((0, 0), (10, 0), 20)
    bend = [*line((0, 0), (5, 0), 10), *line((5, 0.5), (5, 5), 10)]
    assert route_kind({"mode": "long_horizon", "centerline_ids": ["b"], "centerline_xy": bend}, crosswalks) == "long"
    assert route_kind({"mode": "short_horizon", "centerline_ids": ["a", "b"], "centerline_xy": bend}, crosswalks) == "crosswalk"
    assert route_kind({"mode": "short_horizon", "centerline_ids": ["a"], "centerline_xy": bend}, crosswalks) == "curve"
    assert route_kind({"mode": "short_horizon", "centerline_ids": ["a"], "centerline_xy": straight}, crosswalks) == "straight"


def test_ground_fill_reaches_only_the_configured_rings() -> None:
    from arena_simulation_setup.adapters.sidewalkbench import filled

    ground = np.full((1, 12), np.nan)
    ground[0, 0] = 2.0
    out = filled(ground, cells=3)
    assert np.isfinite(out[0, :4]).all()
    assert np.isnan(out[0, 4:]).all()
    assert out[0, 3] == pytest.approx(2.0)


def test_attribution_names_the_source_license_authors_and_citation(imported: Path) -> None:
    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    attribution = read_attribution(imported / ATTRIBUTION_FILE)
    assert (attribution.source, attribution.license) == ("sidewalkbench", "MIT")
    assert attribution.url == "https://huggingface.co/datasets/UCLA-VAIL/SidewalkBench-GS-Assets"
    for surname in ("Liu", "He", "Alumootil", "Pandya", "Squicciarini", "Wu", "Zhou"):
        assert surname in attribution.authors
    assert "2606.16953" in attribution.citation


def test_reimport_replaces_the_world(importer, tmp_path: Path) -> None:
    out = tmp_path / "worlds"
    stale = importer.import_item(SCENE, out) / "stale.txt"
    stale.write_text("x")
    importer.import_item(SCENE, out)
    assert not stale.exists()


def test_cli_import_lands_where_the_world_resolver_looks(tmp_path: Path) -> None:
    cache = seed(tmp_path / "cache")
    env = {**os.environ, "ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "imported")}
    target = tmp_path / "imported" / "worlds" / WORLD
    imported = subprocess.run([sys.executable, "-m", "arena_assets", "import", "world", "sidewalkbench", SCENE, "--cache", str(cache)], env=env, capture_output=True, text=True, check=True)
    assert imported.stdout.splitlines()[-1] == str(target)
    code = "from arena_simulation_setup.tree.World import WorldIdentifier\nview = WorldIdentifier(%r).resolve_sync()\nprint(view.path)\nprint(*sorted(scenario.name for scenario in view.scenario.listall()))" % WORLD
    resolved = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert resolved.stdout.splitlines() == [str(target), "crosswalk_0 curve_0 long_0 straight_0"]


def test_cli_refuses_an_unknown_scene_without_a_traceback(tmp_path: Path) -> None:
    refused = subprocess.run([sys.executable, "-m", "arena_assets", "import", "world", "sidewalkbench", "nowhere", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out")], capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "ucla_sidewalk_backward" in " ".join(re.sub(r"\x1b\[[0-9;]*m", "", refused.stderr).replace("\u2502", " ").split())
    assert "Traceback" not in refused.stderr
