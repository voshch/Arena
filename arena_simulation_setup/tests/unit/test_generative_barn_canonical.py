"""Canonical BARN worlds against the upstream world file: cylinder footprint, lattice, bounds, start and goal."""

from __future__ import annotations

import asyncio
import math
import random
import re
import shutil
from pathlib import Path

import pytest
import shapely
import yaml

FIXTURE = Path(__file__).parent.parent / "fixtures" / "barn" / "world_0.world"
INDEX = 0
CELLS = 209
CELL_M = 0.15
RADIUS_M = 0.075
SIDES = 8
BOUNDARY_WALLS = 4
X_SHIFT_M = 4.5
UPSTREAM_INIT = (-2.25, 3.0, 1.57)
UPSTREAM_GOAL_OFFSET = (0.0, 10.0)

_STATE_POSE = re.compile(r"<model name='unit_cylinder_\d+'>\s*<pose frame=''>(\S+) (\S+) ")
_RADIUS = re.compile(r"<radius>([^<]+)</radius>")


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")


@pytest.fixture()
def generator():
    from arena_simulation_setup.generative.world.barn import WorldGeneratorBarnCanonical

    return WorldGeneratorBarnCanonical({"index": INDEX, "source": str(FIXTURE.parent)}, random.Random(0))


def _upstream_cylinders() -> list[tuple[float, float, float]]:
    text = FIXTURE.read_text()
    radii = {float(radius) for radius in _RADIUS.findall(text)}
    assert radii == {RADIUS_M}
    return [(float(x), float(y), RADIUS_M) for x, y in _STATE_POSE.findall(text)]


def _raw(level) -> list:
    return [segment for path in level.walls for segment in path.segments()]


def _segments(walls) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [((w.start.x, w.start.y), (w.end.x, w.end.y)) for w in walls]


def _length(walls) -> float:
    return sum(math.dist(a, b) for a, b in _segments(walls))


def _boxes(walls) -> list[shapely.Polygon]:
    async def realize():
        return [segment for wall in walls for segment in (await wall.assets())[0]]

    return [shapely.LineString([(s.start.x, s.start.y), (s.end.x, s.end.y)]).buffer(s.width / 2, cap_style="flat") for s in asyncio.run(realize())]


def _footprints(walls) -> list[shapely.Polygon]:
    boxes = _boxes(walls)
    return [shapely.Polygon(shapely.union_all(boxes[n : n + SIDES]).exterior) for n in range(0, len(boxes), SIDES)]


def test_fixture_world_lies_on_the_lattice():
    from arena_simulation_setup.generative.world.barn import world_cells

    cells = world_cells(FIXTURE)
    assert len(cells) == CELLS
    assert all(0 <= i < 30 and 0 <= k < 64 for i, k in cells)
    assert (29, 0) in cells
    assert (0, 63) in cells
    assert (15, 19) not in cells


def test_every_cell_is_a_cylinder_on_its_center(generator):
    from arena_simulation_setup.generative.world.barn import world_cells

    level = generator.compute().levels["0"]
    assert [zone.name for zone in level.zones] == ["barn"]
    rings = _raw(level)[BOUNDARY_WALLS:]
    assert len(rings) == SIDES * CELLS
    centers = {(round((i + 0.5) * CELL_M, 4), round((k + 0.5) * CELL_M, 4)) for i, k in world_cells(FIXTURE)}
    found = set()
    for n in range(CELLS):
        ring = shapely.MultiPoint([a for a, _ in _segments(rings[n * SIDES : (n + 1) * SIDES])] + [b for _, b in _segments(rings[n * SIDES : (n + 1) * SIDES])])
        found.add((round(ring.centroid.x, 4), round(ring.centroid.y, 4)))
    assert found == centers


def test_wall_boxes_fill_the_upstream_cylinders(generator):
    upstream = {(round(x + X_SHIFT_M, 4), round(y, 4)): radius for x, y, radius in _upstream_cylinders()}
    assert len(upstream) == CELLS
    footprints = _footprints(_raw(generator.compute().levels["0"])[BOUNDARY_WALLS:])
    assert len(footprints) == CELLS
    octagon_share = 2 * math.sqrt(2) / math.pi
    matched = set()
    for footprint in footprints:
        center = (round(footprint.centroid.x, 4), round(footprint.centroid.y, 4))
        radius = upstream[center]
        matched.add(center)
        cylinder = shapely.Point(center).buffer(radius, quad_segs=64)
        assert shapely.Point(center).buffer(radius + 1e-9, quad_segs=64).covers(footprint)
        assert footprint.covers(shapely.Point(center).buffer(radius * math.cos(math.pi / SIDES) - 1e-9, quad_segs=64))
        assert footprint.symmetric_difference(shapely.Point(center).buffer(radius, quad_segs=2)).area < 1e-12
        assert footprint.area / (math.pi * radius**2) == pytest.approx(octagon_share, abs=1e-6)
        assert footprint.hausdorff_distance(cylinder) < radius * (1 - math.cos(math.pi / SIDES)) + 1e-6
    assert matched == set(upstream)


def test_obstacle_field_spans_the_upstream_extent(generator):
    xs = [x + X_SHIFT_M for x, _, _ in _upstream_cylinders()]
    ys = [y for _, y, _ in _upstream_cylinders()]
    expected = (min(xs) - RADIUS_M, min(ys) - RADIUS_M, max(xs) + RADIUS_M, max(ys) + RADIUS_M)
    assert expected == pytest.approx((0.0, 0.0, 4.5, 9.6))
    field = shapely.union_all(_footprints(_raw(generator.compute().levels["0"])[BOUNDARY_WALLS:]))
    assert field.bounds == pytest.approx(expected, abs=1e-9)


def test_unthickened_ring_is_the_procedural_cylinder(generator):
    from arena_simulation_setup.generative.world.barn import cylinder_walls
    from arena_simulation_setup.generative.world.barn_cylinder import WorldGeneratorBarnCylinder
    from arena_simulation_setup.generative.world.utils import to_walls
    from arena_simulation_setup.tree.World import Level

    assert WorldGeneratorBarnCylinder.Configuration().cylinder_radius == RADIUS_M
    assert _segments(_raw(Level(walls=cylinder_walls(1.0, 2.0, RADIUS_M)))) == _segments(_raw(Level(walls=to_walls(shapely.Point(1.0, 2.0).buffer(RADIUS_M, quad_segs=2)))))
    procedural = _raw(WorldGeneratorBarnCylinder({}, random.Random(0)).compute().levels["0"])[BOUNDARY_WALLS : BOUNDARY_WALLS + SIDES]
    assert _length(procedural) == pytest.approx(SIDES * 2 * RADIUS_M * math.sin(math.pi / SIDES))
    canonical = _footprints(_raw(generator.compute().levels["0"])[BOUNDARY_WALLS : BOUNDARY_WALLS + SIDES])[0]
    assert canonical.exterior.length == pytest.approx(_length(procedural))


def test_boundary_walls_span_the_arena(generator):
    level = generator.compute().levels["0"]
    floor = shapely.Polygon([(c.x, c.y) for c in level.zones[0].polygon])
    assert floor.bounds == pytest.approx((0.0, 0.0, 4.5, 14.7))
    assert _length(_raw(level)[:BOUNDARY_WALLS]) == pytest.approx(2 * (4.5 + 14.7))
    assert floor.covers(shapely.MultiLineString(_segments(_raw(level))))


def test_start_and_goal_are_the_challenge_poses_and_clear_of_walls(generator):
    from arena_simulation_setup.generative.world.barn import GOAL, START

    assert START == pytest.approx((UPSTREAM_INIT[0] + X_SHIFT_M, UPSTREAM_INIT[1], UPSTREAM_INIT[2]), abs=1e-3)
    assert GOAL == pytest.approx((START[0] + UPSTREAM_GOAL_OFFSET[0], START[1] + UPSTREAM_GOAL_OFFSET[1]))
    level = generator.compute().levels["0"]
    floor = shapely.Polygon([(c.x, c.y) for c in level.zones[0].polygon])
    walls = shapely.MultiLineString(_segments(_raw(level)))
    for point in (START[:2], GOAL):
        assert floor.covers(shapely.Point(point))
        assert walls.distance(shapely.Point(point)) > 0.5


def test_scenario_file_and_episode_binding(generator):
    files = generator.files()
    assert list(files) == ["scenarios/barn/scenario.yaml"]
    content = yaml.safe_load(files["scenarios/barn/scenario.yaml"])
    assert content == {"format": "arena-scenario/2", "robots": [{"start": [2.25, 3.0, 1.571], "phases": [{"goto": [2.25, 13.0, 1.571]}]}]}
    assert generator.params() == {"tm_robots": "scenario", "robots_params": {"file": "barn"}}


def test_world_saves_and_loads_through_the_world_tree(generator, tmp_path: Path):
    from arena_simulation_setup.generative.world.world_generator import save_generated
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.tree.World import WorldView

    level = generator.compute().levels["0"]
    path = save_generated(WorldView(tmp_path / "barn_0"), generator.compute(), generator.files())
    assert sorted(f.relative_to(path).as_posix() for f in path.rglob("*") if f.is_file()) == ["kg.yaml", "map/0.png", "map/0.yaml", "scenarios/barn/scenario.yaml", "world.yaml"]
    view = WorldView(path)
    loaded = view.load()
    assert list(loaded.levels) == ["0"]
    assert [value for segment in _segments(_raw(loaded.levels["0"])) for point in segment for value in point] == pytest.approx([value for segment in _segments(_raw(level)) for point in segment for value in point], abs=1e-4)
    assert all(identifier.resolve_source_sync() for identifier in view.identifiers(strict=True))
    assert [scenario.name for scenario in view.scenario.listall()] == ["barn"]
    robot = view.scenario("barn").resolve_sync().load().robots[0]
    phases = robot.phase_list()
    assert (robot.start.position.x, robot.start.position.y) == pytest.approx((2.25, 3.0))
    assert isinstance(phases[0], GoToPhase)
    assert (phases[0].pose.position.x, phases[0].pose.position.y) == pytest.approx((2.25, 13.0))
    assert yaml.safe_load((path / "map" / "0.yaml").read_text())["resolution"] == pytest.approx(0.05)


def test_fetch_keeps_a_cached_world_and_writes_attribution_once(tmp_path: Path):
    from arena_simulation_setup.generative.world.barn import fetch_world

    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    cached = tmp_path / "worlds" / FIXTURE.name
    cached.parent.mkdir()
    shutil.copy(FIXTURE, cached)
    sidecar = tmp_path / ATTRIBUTION_FILE
    assert fetch_world(INDEX, tmp_path) == cached
    attribution = read_attribution(sidecar)
    assert attribution.license == "MIT"
    assert attribution.url == "https://github.com/Daffan/the-barn-challenge"
    assert "Perille" in attribution.authors
    assert "2008.13315" in attribution.citation
    stamp = sidecar.stat().st_mtime_ns
    assert fetch_world(INDEX, tmp_path) == cached
    assert sidecar.stat().st_mtime_ns == stamp


def test_default_cache_is_the_import_cache():
    from arena_simulation_setup.generative.world.barn import cache_dir

    assert cache_dir().parts[-2:] == (".import", "barn")


def test_index_dispatches_between_canonical_and_procedural():
    from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
    from arena_simulation_setup.generative.world.barn import WorldGeneratorBarn, WorldGeneratorBarnAny, WorldGeneratorBarnCanonical

    model = WorldGenerator.config_model(WorldGeneratorType.BARN)
    assert model is WorldGeneratorBarnAny.Configuration
    assert set(model.model_fields) == set(WorldGeneratorBarn.Configuration.model_fields) | {"index", "source"}
    defaults = model().model_dump()
    assert (defaults["index"], defaults["source"]) == (None, "")
    canonical = WorldGenerator(WorldGeneratorType.BARN, {"index": INDEX, "source": str(FIXTURE.parent)})
    assert isinstance(canonical._active, WorldGeneratorBarnCanonical)
    assert len(_raw(canonical.compute().levels["0"])) == BOUNDARY_WALLS + SIDES * CELLS
    every_field = WorldGenerator(WorldGeneratorType.BARN, {**defaults, "index": INDEX, "source": str(FIXTURE.parent)})
    assert isinstance(every_field._active, WorldGeneratorBarnCanonical)
    assert _segments(_raw(every_field.compute().levels["0"])) == _segments(_raw(canonical.compute().levels["0"]))
    for configuration in ({}, {"index": None}, defaults):
        procedural = WorldGenerator(WorldGeneratorType.BARN, configuration, seed=0)
        assert isinstance(procedural._active, WorldGeneratorBarn)
        assert _segments(_raw(procedural.compute().levels["0"])) == _segments(_raw(WorldGenerator(WorldGeneratorType.BARN, {}, seed=0).compute().levels["0"]))


def test_ros_node_declares_index_and_source_and_dispatches_on_them(rclpy_context):
    pytest.importorskip("world_generator_msgs")
    import json

    import world_generator_msgs.srv
    from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
    from arena_simulation_setup.generative.world.barn import WorldGeneratorBarn, WorldGeneratorBarnCanonical
    from arena_simulation_setup.generative.world.layout import diagnostics_of
    from arena_simulation_setup.generative.world.world_generator_ros import WorldGeneratorROS
    from rclpy.parameter import Parameter

    canonical = list(diagnostics_of(WorldGenerator(WorldGeneratorType.BARN, {"index": INDEX, "source": str(FIXTURE.parent)}).compute().all_levels[0]).extent)
    procedural = list(diagnostics_of(WorldGenerator(WorldGeneratorType.BARN, {}, seed=0).compute().all_levels[0]).extent)
    assert canonical != pytest.approx(procedural)

    def preview(config: str = "") -> world_generator_msgs.srv.GenerateWorld.Response:
        request = world_generator_msgs.srv.GenerateWorld.Request(generator="barn", config=config, preview_only=True)
        response = node._cb_generate_world(request, world_generator_msgs.srv.GenerateWorld.Response())
        assert response.success, response.message
        return response

    node = WorldGeneratorROS()
    try:
        declared = node.get_parameters_by_prefix("algorithm.barn")
        assert (declared["index"].value, declared["source"].value) == (-1, "")
        assert list(preview().extent) == pytest.approx(procedural)
        assert isinstance(node._active, WorldGeneratorBarn)
        assert list(preview(json.dumps({"index": -1})).extent) == pytest.approx(procedural)
        assert isinstance(node._active, WorldGeneratorBarn)
        assert list(preview(json.dumps({"index": INDEX, "source": str(FIXTURE.parent)})).extent) == pytest.approx(canonical)
        assert isinstance(node._active, WorldGeneratorBarnCanonical)
        node.set_parameters([Parameter("algorithm.barn.index", value=INDEX), Parameter("algorithm.barn.source", value=str(FIXTURE.parent))])
        assert list(preview().extent) == pytest.approx(canonical)
        assert isinstance(node._active, WorldGeneratorBarnCanonical)
    finally:
        node.destroy_node()


def test_ros_node_gets_and_describes_every_generator_parameter(rclpy_context):
    pytest.importorskip("world_generator_msgs")
    from arena_simulation_setup.generative.world.world_generator_ros import WorldGeneratorROS

    node = WorldGeneratorROS()
    try:
        names = node.list_parameters(["algorithm"], 10).names
        assert "algorithm.barn.index" in names
        assert len(node.get_parameters(names)) == len(node.describe_parameters(names)) == len(names)
    finally:
        node.destroy_node()


def test_index_outside_the_challenge_is_rejected():
    import pydantic
    from arena_simulation_setup.generative.world.barn import WorldGeneratorBarnCanonical

    from arena_simulation_setup.generative.world.barn import WorldGeneratorBarnAny

    for index in (-1, 300):
        with pytest.raises(pydantic.ValidationError):
            WorldGeneratorBarnCanonical({"index": index}, random.Random(0))
        with pytest.raises(pydantic.ValidationError):
            WorldGeneratorBarnAny.Configuration.model_validate({"index": index})


def test_off_lattice_cylinder_is_rejected(tmp_path: Path):
    from arena_simulation_setup.generative.world.barn import world_cells

    world = tmp_path / FIXTURE.name
    world.write_text(FIXTURE.read_text().replace("<pose frame=''>-0.075000 0.075000", "<pose frame=''>-0.100000 0.075000"))
    with pytest.raises(ValueError, match="off the 0.15 m lattice"):
        world_cells(world)


def test_other_cylinder_radius_is_rejected(tmp_path: Path):
    from arena_simulation_setup.generative.world.barn import world_cells

    world = tmp_path / FIXTURE.name
    world.write_text(FIXTURE.read_text().replace("<radius>0.075000</radius>", "<radius>0.100000</radius>", 1))
    with pytest.raises(ValueError, match="are not all 0.075 m unscaled"):
        world_cells(world)


@pytest.mark.network
def test_fetch_downloads_world_1(tmp_path: Path):
    from arena_simulation_setup.generative.world.barn import fetch_world, world_cells

    world = fetch_world(1, tmp_path)
    assert world == tmp_path / "worlds" / "world_1.world"
    cells = world_cells(world)
    assert all(0 <= i < 30 and 0 <= k < 64 for i, k in cells)
    assert len(cells) > 34
    stamp = world.stat().st_mtime_ns
    assert fetch_world(1, tmp_path) == world
    assert world.stat().st_mtime_ns == stamp
