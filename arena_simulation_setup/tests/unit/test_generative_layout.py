import pytest
import shapely
import shapely.affinity
from arena_simulation_setup.generative.world import WorldGenerator
from arena_simulation_setup.generative.world.layout import (
    Area,
    Layout,
    Region,
    Segment,
    _polygons,
    compile_layout,
    diagnostics_of,
)
from arena_simulation_setup.tree.World import describe


def corridor(a, b, width=2.0):
    return Segment(a=a, b=b, width=width)


def test_single_corridor_is_one_component():
    layout = Layout(segments=[corridor((0.0, 0.0), (10.0, 0.0))])
    world, diagnostics = compile_layout(layout)
    level = world.all_levels[0]
    assert diagnostics.components == 1
    assert diagnostics.islands == 0
    assert diagnostics.extent == (10.0, 2.0)
    assert len(list(level.all_walls)) == 4


def test_boundary_walls_are_closed_level_paths_and_zones_take_no_ceiling_by_default():
    layout = Layout(segments=[corridor((0.0, 0.0), (10.0, 0.0))], regions=[Region(name='hall', polygon=shapely.box(-1.0, -1.0, 11.0, 1.0), description='main hall', material='Marble')])
    world, _ = compile_layout(layout)
    level = world.all_levels[0]
    assert [(len(path.path), path.closed) for path in level.walls] == [(4, True)]
    assert [zone.ceiling for zone in level.zones] == [None]
    assert level.zones[0].floor.name == 'Marble'
    assert describe(world.kg, 'hall') == 'main hall'
    assert all(zone.ceiling is not None for zone in compile_layout(layout, ceiling=True)[0].all_levels[0].zones)


def test_detached_pieces_are_counted():
    layout = Layout(segments=[corridor((0.0, 0.0), (10.0, 0.0)), corridor((0.0, 20.0), (10.0, 20.0))])
    _, diagnostics = compile_layout(layout)
    assert diagnostics.components == 2


def test_ring_has_one_interior_ring():
    layout = Layout(
        segments=[
            corridor((0.0, 0.0), (30.0, 0.0)),
            corridor((30.0, 0.0), (30.0, 24.0)),
            corridor((30.0, 24.0), (0.0, 24.0)),
            corridor((0.0, 24.0), (0.0, 0.0)),
        ]
    )
    _, diagnostics = compile_layout(layout)
    assert diagnostics.components == 1
    assert diagnostics.islands == 1


def test_widths_step_at_the_join():
    layout = Layout(segments=[corridor((0.0, 0.0), (8.0, 0.0), 1.5), corridor((8.0, 0.0), (16.0, 0.0), 6.0)])
    geometry = layout.geometry()
    assert geometry.bounds == (0.0, -3.0, 16.0, 3.0)
    assert geometry.intersection(shapely.box(0.0, -3.0, 8.0, 3.0)).area == 8.0 * 1.5


def test_zones_are_named_by_region_and_never_overlap():
    layout = Layout(
        segments=[corridor((0.0, 0.0), (10.0, 0.0)), corridor((5.0, 0.0), (5.0, 10.0))],
        regions=[
            Region(name='hall', polygon=shapely.box(-1.0, -1.0, 11.0, 1.0)),
            Region(name='spur', polygon=shapely.box(4.0, -1.0, 6.0, 11.0)),
        ],
    )
    world, diagnostics = compile_layout(layout)
    level = world.all_levels[0]
    names = [zone.name for zone in level.zones]
    assert names[0] == 'hall'
    assert 'spur' in names
    assert diagnostics.zones == len(level.zones)
    polygons = [zone.shape() for zone in level.zones]
    for index, first in enumerate(polygons):
        for second in polygons[index + 1 :]:
            assert first.intersection(second).area == 0.0


def test_zones_cover_the_whole_geometry():
    layout = Layout(
        segments=[corridor((0.0, 0.0), (10.0, 0.0))],
        regions=[Region(name='half', polygon=shapely.box(-1.0, -2.0, 5.0, 2.0))],
    )
    level = compile_layout(layout)[0].all_levels[0]
    covered = shapely.union_all([zone.shape() for zone in level.zones])
    assert covered.difference(layout.geometry()).area == 0.0
    assert layout.geometry().difference(covered).area == 0.0


def test_filled_areas_merge_with_corridors():
    layout = Layout(
        segments=[corridor((0.0, 0.0), (10.0, 0.0))],
        areas=[Area(polygon=shapely.box(10.0, -5.0, 20.0, 5.0))],
    )
    _, diagnostics = compile_layout(layout)
    assert diagnostics.components == 1
    assert diagnostics.extent == (20.0, 10.0)


def test_only_polygonal_parts_survive_a_slit():
    """Cutting a hole open sheds lines and points beside the polygons, and they have no interiors.
    Areas alone would not say so: a collection reports the same total as the polygons inside it."""
    mixed = shapely.GeometryCollection([shapely.box(0.0, 0.0, 1.0, 1.0), shapely.LineString([(2.0, 0.0), (3.0, 0.0)]), shapely.Point(4.0, 4.0)])
    parts = _polygons(mixed)
    assert [type(part) for part in parts] == [shapely.Polygon]
    assert [part.area for part in parts] == [1.0]
    assert [len(part.interiors) for part in parts] == [0]


def test_edges_a_rounding_error_apart_do_not_become_a_zone():
    """Hole edges that differ in the last bits of a float are one edge. Slicing between them
    yields a hairline band: valid geometry, meaningless zone."""
    left = shapely.box(4.0, 4.0, 9.0, 9.0)
    right = shapely.box(9.0, 4.0, 14.0, 9.0)
    for angle in (51.0, -51.0):
        right = shapely.affinity.rotate(right, angle, origin=(11.5, 6.5))
    assert right.bounds[0] != 9.0
    layout = Layout(
        areas=[Area(polygon=shapely.box(0.0, 0.0, 20.0, 20.0))],
        regions=[
            Region(name='core', polygon=shapely.union_all([left, right])),
            Region(name='shell', polygon=shapely.box(0.0, 0.0, 20.0, 20.0)),
        ],
    )
    level = compile_layout(layout)[0].all_levels[0]
    assert min(zone.shape().area for zone in level.zones) > 1.0


@pytest.mark.parametrize('generator_type', sorted(WorldGenerator.available(), key=lambda t: t.value), ids=lambda t: t.value)
def test_every_generator_reports_its_size(generator_type):
    """Diagnostics come off the finished level, so a generator that builds zones by hand still
    reports them. Zero extent would also flatten the preview resolution to its floor."""
    level = WorldGenerator(generator_type, {}, 0).compute().all_levels[0]
    diagnostics = diagnostics_of(level)
    assert diagnostics.zones == len(level.zones) > 0
    assert diagnostics.components > 0
    assert min(diagnostics.extent) > 0.0
