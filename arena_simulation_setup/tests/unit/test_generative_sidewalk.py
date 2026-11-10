"""Street worlds: zone bands, free space, the robot mission and the pedestrian placement table."""

import math
import random
import re

import numpy as np
import pydantic
import pytest
import shapely
import yaml
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.sidewalk import BEHAVIORS, LAYOUTS, PLACEMENT, WorldGeneratorSidewalk
from arena_simulation_setup.tree.World import Level, World
from arena_simulation_setup.tree.World.Scenario import ScenarioView

SCENARIO_PATH = 'scenarios/default/scenario.yaml'
RESOLUTION = 0.05
JUNCTIONS = ('intersection', 'crosswalk', 'block')
BANDS = ('road', 'curb', 'verge', 'sidewalk', 'frontage')


def build(layout: str = 'straight', seed: int = 0, **config: object) -> tuple[WorldGeneratorSidewalk, World]:
    generator = WorldGeneratorSidewalk({'layout': layout, **config}, random.Random(seed))
    return generator, generator.compute()


def scenario(generator: WorldGeneratorSidewalk) -> dict:
    return yaml.safe_load(generator.files()[SCENARIO_PATH])


def level_of(world: World) -> Level:
    return world.all_levels[0]


def kind_of(name: str) -> str:
    return re.sub(r'_\d+$', '', name)


def zones_named(world: World, kind: str) -> list:
    return [zone for zone in level_of(world).zones if kind_of(zone.name) == kind]


def zone_at(world: World, x: float, y: float) -> str | None:
    point = shapely.Point(x, y)
    return next((zone.name for zone in level_of(world).zones if zone.shape().buffer(1e-6).contains(point)), None)


def sidewalk_area(world: World) -> float:
    return sum(zone.shape().area for zone in zones_named(world, 'sidewalk'))


def humans_of(generator: WorldGeneratorSidewalk, agent_type: str) -> list[dict]:
    return [human for human in scenario(generator).get('humans', []) if human['agent']['agent_type'] == agent_type]


def expected_pose(generator: WorldGeneratorSidewalk, s: float, d: float) -> tuple[float, float]:
    """Straight layouts run along +x with the route at its offset, so s is x and the left is +y."""
    return s, generator.edges[2] + generator.config.sidewalk_width / 2 + d


def test_registered_in_the_generator_registry():
    assert WorldGeneratorType.SIDEWALK in WorldGenerator.available()
    world = WorldGenerator(WorldGeneratorType.SIDEWALK, {'layout': 'corner'}, seed=3).compute()
    assert len(world.all_levels) == 1


@pytest.mark.parametrize('layout', LAYOUTS)
def test_every_layout_compiles_with_its_zones(layout: str):
    generator, world = build(layout)
    kinds = {kind_of(zone.name) for zone in level_of(world).zones}
    expected = {*BANDS, *(('crosswalk',) if layout in JUNCTIONS else ())}
    assert kinds == expected
    assert generator.diagnostics.components == 1


@pytest.mark.parametrize('layout', LAYOUTS)
def test_zone_names_are_suffixed_only_when_a_kind_has_several(layout: str):
    _, world = build(layout)
    names = [zone.name for zone in level_of(world).zones]
    assert len(set(names)) == len(names)
    for kind in {kind_of(name) for name in names}:
        group = [name for name in names if kind_of(name) == kind]
        if len(group) == 1:
            assert group == [kind]
        else:
            assert group == [f'{kind}_{index}' for index in range(len(group))]


@pytest.mark.parametrize('layout', LAYOUTS)
def test_free_space_is_one_connected_piece(layout: str):
    generator, _ = build(layout)
    assert isinstance(generator.layout().geometry(), shapely.Polygon)


def test_buildings_behind_the_frontage_are_the_walls():
    generator, world = build('straight', length=30.0)
    level = level_of(world)
    frontage_edge = generator.edges[4]
    ys = [point.y for wall in level.walls for point in wall.path]
    assert max(ys) == pytest.approx(frontage_edge, abs=0.02)
    assert min(ys) == pytest.approx(-frontage_edge, abs=0.02)


def test_the_curb_and_the_road_are_free_space_on_the_map():
    generator, world = build('straight', length=30.0)
    level = level_of(world)
    resolution = 0.1
    grid, (origin_x, origin_y) = level.render_grid(resolution)
    edges = generator.edges
    for y in (edges[0] / 2, (edges[0] + edges[1]) / 2, -(edges[0] + edges[1]) / 2):
        assert grid[grid.shape[0] - 1 - round((y - origin_y) / resolution), round((15.0 - origin_x) / resolution)] == 255


def test_the_bands_nest_from_the_axis_outward():
    generator, world = build('straight', length=30.0)
    edges = generator.edges
    names = [zone_at(world, 15.0, (inner + outer) / 2) for inner, outer in zip([0.0, *edges[:-1]], edges, strict=True)]
    assert [kind_of(name) for name in names] == list(BANDS)


@pytest.mark.parametrize('layout', LAYOUTS)
def test_robot_start_and_goal_lie_in_a_sidewalk_zone(layout: str):
    generator, world = build(layout)
    robot = scenario(generator)['robots'][0]
    start = robot['start']
    goal = robot['phases'][0]['goto']
    assert kind_of(zone_at(world, start[0], start[1])) == 'sidewalk'
    assert kind_of(zone_at(world, goal[0], goal[1])) == 'sidewalk'


@pytest.mark.parametrize('layout', LAYOUTS)
def test_the_route_stays_off_the_road(layout: str):
    generator, world = build(layout)
    route = generator.route
    kinds = {kind_of(zone_at(world, *route.interpolate(s).coords[0]) or 'none') for s in np.linspace(0.0, route.length, 60)}
    assert kinds <= {'sidewalk', 'crosswalk'}


@pytest.mark.parametrize('layout', JUNCTIONS)
def test_junction_routes_cross_a_crosswalk(layout: str):
    generator, world = build(layout)
    route = generator.route
    kinds = {kind_of(zone_at(world, *route.interpolate(s).coords[0])) for s in np.linspace(0.0, route.length, 200)}
    assert 'crosswalk' in kinds


@pytest.mark.parametrize('blocks', [1, 2, 3])
def test_a_block_route_runs_over_100_m(blocks: int):
    generator, _ = build('block', blocks=blocks)
    assert generator.route.length > 100.0


def test_a_block_grid_leaves_the_buildings_as_islands():
    generator, _ = build('block', blocks=2)
    assert generator.diagnostics.islands == 4


def test_the_narrow_passage_pinches_the_sidewalk():
    generator, world = build('narrow_passage', length=40.0)
    x, y = expected_pose(generator, 20.0, 0.0)
    zone = next(zone for zone in zones_named(world, 'sidewalk') if zone.shape().contains(shapely.Point(x, y + 0.2)) or zone.shape().contains(shapely.Point(x - 3.0, y)))
    pinch = zone.shape().intersection(shapely.box(19.0, -20.0, 21.0, 20.0))
    assert pinch.area == pytest.approx(2.0 * 1.0, rel=0.05)


def test_a_curve_bends_the_route_and_keeps_its_length():
    generator, _ = build('curve', length=40.0)
    route = generator.route
    assert route.length == pytest.approx(40.0, rel=0.05)
    start, end = route.coords[0], route.coords[-1]
    assert abs(end[1] - start[1]) > 1.0


def test_a_corner_turns_the_route_a_quarter():
    generator, _ = build('corner', length=40.0)
    coords = generator.route.coords
    assert coords[-1][0] > coords[0][0] and coords[-1][1] < coords[0][1]
    assert generator.route.length == pytest.approx(40.0, rel=0.02)


def test_the_plaza_opens_the_sidewalk_to_the_left():
    _, plain_world = build('straight')
    _, plaza_world = build('plaza')
    assert sidewalk_area(plaza_world) > sidewalk_area(plain_world) + 100.0


def test_empty_places_no_pedestrians():
    generator, _ = build('straight', scenario='empty', pedestrians=5)
    assert 'humans' not in scenario(generator)


def test_default_places_the_pedestrian_count_of_adults():
    generator, _ = build('straight', scenario='default', pedestrians=5)
    humans = scenario(generator)['humans']
    assert len(humans) == 5
    assert {human['agent']['agent_type'] for human in humans} == {'adult'}
    for human in humans:
        x, y = human['pose'][:2]
        assert generator.route.distance(shapely.Point(x, y)) <= generator.config.sidewalk_width / 2


def test_default_without_a_count_flows_a_few_adults():
    generator, _ = build('straight', scenario='default')
    assert len(humans_of(generator, 'adult')) == 4


@pytest.mark.parametrize('behavior', [name for name in BEHAVIORS if name != 'crossing'])
def test_a_behavior_places_its_group_at_the_table_offsets(behavior: str):
    generator, _ = build('straight', scenario=behavior)
    humans = humans_of(generator, f'sidewalk_{behavior}')
    assert len(humans) == len(PLACEMENT[behavior])
    assert len(scenario(generator)['humans']) == len(humans)
    for human, (s, d, _) in zip(humans, PLACEMENT[behavior], strict=True):
        x, y = expected_pose(generator, s, d)
        assert human['pose'][0] == pytest.approx(x, abs=RESOLUTION)
        assert human['pose'][1] == pytest.approx(y, abs=RESOLUTION)


def test_the_table_counts_match_the_paper_groups():
    assert {name: len(PLACEMENT[name]) for name in PLACEMENT} == {'obstructing': 1, 'conversing': 3, 'queueing': 4, 'frontal': 1, 'lateral': 1, 'overtaking': 1, 'gesturing': 1}


def test_a_frontal_pedestrian_faces_the_robot_and_an_overtaker_stands_behind_it():
    frontal, _ = build('straight', scenario='frontal')
    assert math.cos(humans_of(frontal, 'sidewalk_frontal')[0]['pose'][2]) == pytest.approx(-1.0, abs=0.01)
    overtaking, _ = build('straight', scenario='overtaking')
    start = scenario(overtaking)['robots'][0]['start']
    assert humans_of(overtaking, 'sidewalk_overtaking')[0]['pose'][0] < start[0]


def test_a_lateral_pedestrian_stands_off_the_sidewalk_edge_facing_across():
    generator, world = build('straight', scenario='lateral')
    pose = humans_of(generator, 'sidewalk_lateral')[0]['pose']
    assert kind_of(zone_at(world, pose[0], pose[1])) == 'verge'
    assert math.sin(pose[2]) == pytest.approx(1.0, abs=0.01)


def test_queueing_pedestrians_form_a_line_from_the_side_toward_the_center():
    generator, _ = build('straight', scenario='queueing')
    ys = [human['pose'][1] for human in humans_of(generator, 'sidewalk_queueing')]
    assert np.diff(ys) == pytest.approx([0.6, 0.6, 0.6], abs=RESOLUTION)


def test_crossing_pedestrians_wait_at_both_ends_of_the_crosswalk():
    generator, world = build('crosswalk', scenario='crossing')
    humans = humans_of(generator, 'sidewalk_crossing')
    assert len(humans) == 2
    xmin, _, xmax, _ = zones_named(world, 'crosswalk')[0].bounds
    near, far = humans
    assert near['pose'][0] < xmin < xmax < far['pose'][0]
    assert near['pose'][1] != far['pose'][1]


def test_crossing_needs_a_crosswalk_on_the_route():
    with pytest.raises(ValueError, match='no crosswalk'):
        build('straight', scenario='crossing')


def test_a_behavior_beyond_the_route_is_refused():
    with pytest.raises(ValueError, match='along a route'):
        build('straight', scenario='frontal', length=20.0)


def test_background_adults_join_a_behavior_group():
    generator, _ = build('straight', scenario='obstructing', pedestrians=3)
    assert len(humans_of(generator, 'adult')) == 3
    assert len(humans_of(generator, 'sidewalk_obstructing')) == 1


@pytest.mark.parametrize('layout', LAYOUTS)
def test_street_furniture_stands_in_the_verge_and_frontage_off_the_centerline(layout: str):
    generator, world = build(layout, obstacles=8)
    objects = scenario(generator)['objects']
    assert len(objects) == 8
    for obj in objects:
        x, y = obj['pose'][:2]
        assert kind_of(zone_at(world, x, y)) in {'verge', 'frontage'}
        assert generator.route.distance(shapely.Point(x, y)) >= generator.config.sidewalk_width / 2


def test_no_obstacles_means_no_objects():
    generator, _ = build('straight', obstacles=0)
    assert 'objects' not in scenario(generator)


def test_params_pin_both_task_axes_to_the_scenario():
    generator, _ = build('straight', pedestrians=4)
    assert generator.params() == {'modes': {'robots': 'scenario', 'obstacles': 'scenario'}, 'params': {'robots': {'file': 'default'}, 'obstacles': {'file': 'default'}}}


def test_a_seed_repeats_and_another_seed_moves_the_furniture():
    first, _ = build('straight', seed=7, obstacles=10)
    again, _ = build('straight', seed=7, obstacles=10)
    other, _ = build('straight', seed=8, obstacles=10)
    assert first.files() == again.files()
    assert first.files() != other.files()


def test_unknown_layouts_and_scenarios_are_refused():
    with pytest.raises(pydantic.ValidationError):
        build('boulevard')
    with pytest.raises(pydantic.ValidationError):
        build('straight', scenario='dancing')


def test_the_scenario_loads(tmp_path):
    generator, world = build('crosswalk', scenario='conversing', pedestrians=2)
    converter = world.compact_world(dict.fromkeys(world.level_ids, (0.0, 0.0))).zone_converter(np.random.default_rng(0))
    directory = tmp_path / 'default'
    directory.mkdir()
    (directory / 'scenario.yaml').write_bytes(generator.files()[SCENARIO_PATH])
    loaded = ScenarioView(str(directory)).load(converter=converter)
    assert len(loaded.robots) == 1
    assert len(loaded.humans) == 5
