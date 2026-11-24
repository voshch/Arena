from __future__ import annotations

import asyncio

import arena_simulation_setup.tree.World as world_package
import attrs
import pytest
from arena_simulation_setup.shared import Door, DynamicObstacle, Elevator, Light, Obstacle, Sound
from arena_simulation_setup.shared.walls import Wall
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import Level, WallPath, WorldMicrophone, Zone, ZoneCeiling, ZoneWalls
from arena_simulation_setup.tree.World.model import World, door_polygon
from arena_simulation_setup.utils.geometry import Pose, Position


def _square(name: str, x: float = 0.0, y: float = 0.0, size: float = 4.0, **zone: object) -> Zone:
    return Zone(name=name, polygon=[[x, y], [x + size, y], [x + size, y + size], [x, y + size]], **zone)


def _undirected(walls: list[Wall]) -> set[frozenset[tuple[float, float]]]:
    return {frozenset({(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)}) for wall in walls}


def _edge(a: tuple[float, float], b: tuple[float, float]) -> frozenset[tuple[float, float]]:
    return frozenset({a, b})


def _tops(walls: list[Wall]) -> list[float]:
    result = []
    for wall in walls:
        segments, _ = asyncio.run(wall.assets())
        result.append(max(segment.start.z + segment.height for segment in segments))
    return result


def test_package_world_is_the_model_class() -> None:
    assert world_package.World is World


def test_outline_walls_follow_the_zone_polygon() -> None:
    level = Level(zones=[_square('room', walls=ZoneWalls())])
    assert _undirected(level.all_walls) == {_edge((0, 0), (4, 0)), _edge((4, 0), (4, 4)), _edge((4, 4), (0, 4)), _edge((0, 4), (0, 0))}


def test_open_to_leaves_the_shared_edge_open_on_that_zone_only() -> None:
    level = Level(zones=[_square('hall'), _square('office', x=4, walls=ZoneWalls(open_to=['hall']))])
    assert _edge((4, 0), (4, 4)) not in _undirected(level.all_walls)
    assert len(level.all_walls) == 3
    both = Level(zones=[_square('hall', walls=ZoneWalls()), _square('office', x=4, walls=ZoneWalls(open_to=['hall']))])
    assert _edge((4, 0), (4, 4)) in _undirected(both.all_walls)


def test_open_to_on_both_sides_leaves_no_wall_between() -> None:
    level = Level(zones=[_square('hall', walls=ZoneWalls(open_to=['office'])), _square('office', x=4, walls=ZoneWalls(open_to=['hall']))])
    assert _edge((4, 0), (4, 4)) not in _undirected(level.all_walls)
    assert len(level.all_walls) == 6


def test_two_zones_sharing_an_edge_give_one_wall() -> None:
    level = Level(zones=[_square('below', walls=ZoneWalls()), _square('above', y=4, walls=ZoneWalls())])
    shared = [wall for wall in level.all_walls if {wall.start.y, wall.end.y} == {4.0}]
    assert len(shared) == 1
    assert len(level.all_walls) == 7


def test_exact_duplicate_walls_merge() -> None:
    level = Level(walls=[WallPath(path=[[0, 0], [4, 0]]), WallPath(path=[[4, 0], [0, 0]]), WallPath(path=[[0, 0], [4, 0]])])
    assert len(level.all_walls) == 1


def test_collinear_overlapping_walls_merge_into_their_span() -> None:
    level = Level(walls=[WallPath(path=[[0, 0], [4, 0]]), WallPath(path=[[2, 0], [6, 0]]), WallPath(path=[[6, 0], [3, 0]])])
    (wall,) = level.all_walls
    assert sorted((wall.start.x, wall.end.x)) == [0.0, 6.0]


def test_narrower_wall_inside_a_wider_one_merges_into_the_wider() -> None:
    level = Level(zones=[_square('wide', walls=ZoneWalls()), Zone(name='narrow', polygon=[[0, 4], [2, 4], [2, 8], [0, 8]], walls=ZoneWalls())])
    on_line = [wall for wall in level.all_walls if {wall.start.y, wall.end.y} == {4.0}]
    assert _undirected(on_line) == {_edge((4, 4), (0, 4))}


def test_touching_collinear_walls_stay_apart() -> None:
    level = Level(walls=[WallPath(path=[[0, 0], [4, 0]]), WallPath(path=[[4, 0], [8, 0]])])
    assert len(level.all_walls) == 2


def test_walls_of_different_style_do_not_merge() -> None:
    level = Level(walls=[WallPath(path=[[0, 0], [4, 0]], material='Brick'), WallPath(path=[[2, 0], [6, 0]], material='Marble')])
    assert len(level.all_walls) == 2


def test_doors_cut_every_wall_they_lie_on() -> None:
    level = Level(
        zones=[_square('room', walls=ZoneWalls())],
        walls=[WallPath(path=[[0, -1], [4, -1]])],
        doors=[Door(name='front', start=Position(1, 0), end=Position(2, 0)), Door(name='side', start=Position(4, 1), end=Position(4, 3))],
    )
    walls = level.all_walls
    bottom = sorted(tuple(sorted((wall.start.x, wall.end.x))) for wall in walls if wall.start.y == 0 and wall.end.y == 0)
    assert bottom == [(0.0, 1.0), (2.0, 4.0)]
    right = sorted(tuple(sorted((wall.start.y, wall.end.y))) for wall in walls if wall.start.x == 4 and wall.end.x == 4)
    assert right == [(0.0, 1.0), (3.0, 4.0)]
    assert _edge((0, -1), (4, -1)) in _undirected(walls)


def test_door_spanning_a_whole_wall_removes_it() -> None:
    level = Level(walls=[WallPath(path=[[0, 0], [2, 0]])], doors=[Door(name='wide', start=Position(0, 0), end=Position(2, 0))])
    assert level.all_walls == []


def test_ceiling_tops_use_an_explicit_height() -> None:
    level = Level(zones=[_square('hall', ceiling=ZoneCeiling(height=3.2)), _square('yard', x=4, ceiling=None)])
    assert asyncio.run(level.ceiling_tops()) == {'hall': 3.2}


def test_ceiling_tops_derive_from_the_tallest_wall_around_the_zone() -> None:
    level = Level(zones=[_square('hall', ceiling=ZoneCeiling())])
    walls = [Wall(start=Position(0, 0), end=Position(4, 0), top=3.5), Wall(start=Position(4, 0), end=Position(4, 4))]
    assert asyncio.run(level.ceiling_tops(walls)) == pytest.approx({'hall': 3.5})


def test_ceiling_tops_ignore_walls_away_from_the_zone() -> None:
    level = Level(zones=[_square('hall', ceiling=ZoneCeiling())])
    walls = [Wall(start=Position(10, 0), end=Position(14, 0), top=3.5)]
    assert asyncio.run(level.ceiling_tops(walls)) == {'hall': 2.0}


def test_ceiling_tops_are_at_least_two_meters() -> None:
    level = Level(zones=[_square('hall', ceiling=ZoneCeiling())], walls=[WallPath(path=[[0, 0], [4, 0], [4, 4], [0, 4]], closed=True, material='Plaster_Wall')])
    assert asyncio.run(level.ceiling_tops()) == pytest.approx({'hall': 2.0})
    assert asyncio.run(level.ceiling_z(level.zones[0])) == pytest.approx(2.0)


def _room(ceiling: ZoneCeiling | None, doors: list[Door] | None = None) -> Level:
    return Level(zones=[Zone(name='hall', polygon=[[0, 0], [8, 0], [8, 6], [0, 6]], ceiling=ceiling)], walls=[WallPath(path=[[0, 0], [8, 0], [8, 6], [0, 6]], closed=True, material='Plaster_Wall')], doors=doors or [])


def test_closed_walls_rise_to_the_ceiling_above_them() -> None:
    level = _room(ZoneCeiling(height=2.6))
    assert [wall.top for wall in level.all_walls] == [None] * 4
    closed = asyncio.run(level.closed_walls())
    assert [wall.top for wall in closed] == [2.6] * 4
    assert _tops(closed) == pytest.approx([2.6] * 4)


def test_closed_walls_keep_their_height_without_a_ceiling() -> None:
    closed = asyncio.run(_room(None).closed_walls())
    assert [wall.top for wall in closed] == [None] * 4
    assert _tops(closed) == pytest.approx([2.0] * 4)


def _railed_room(doors: list[Door] | None = None) -> Level:
    room = _room(ZoneCeiling(height=2.6), doors)
    return attrs.evolve(room, walls=[*room.walls, WallPath(path=[[0, 3], [8, 3]], kind='handrail')])


def test_closed_walls_leave_a_low_wall_kind_at_its_own_height() -> None:
    closed = asyncio.run(_railed_room().closed_walls())
    assert [wall.top for wall in closed] == [2.6] * 4 + [None]
    assert _tops(closed)[-1] == pytest.approx(0.9)


def test_door_in_a_low_wall_kind_gets_no_lintel() -> None:
    gate = Door(name='gate', start=Position(3, 3), end=Position(4, 3), height=1.0)
    assert asyncio.run(_railed_room([gate]).door_lintels()) == []


def test_door_below_the_ceiling_gets_a_lintel() -> None:
    door = Door(name='entry', start=Position(3, 0), end=Position(4, 0), height=2.0)
    (lintel,) = asyncio.run(_room(ZoneCeiling(height=2.6), [door]).door_lintels())
    assert (lintel.bottom, lintel.top) == (2.0, 2.6)
    assert lintel.material == MaterialIdentifier('Plaster_Wall')
    (segment,), obstacles = asyncio.run(lintel.assets())
    assert obstacles == ()
    assert (segment.start.z, segment.height) == pytest.approx((2.0, 0.6))


def test_a_raised_door_gets_its_lintel_above_its_own_top() -> None:
    door = Door(name='entry', start=Position(3, 0, 0.5), end=Position(4, 0, 0.5), height=1.5)
    (lintel,) = asyncio.run(_room(ZoneCeiling(height=2.6), [door]).door_lintels())
    (segment,), _ = asyncio.run(lintel.assets())
    assert (segment.start.z, segment.height) == pytest.approx((2.0, 0.6))


@pytest.mark.parametrize('ceiling', [ZoneCeiling(height=2.0), None])
def test_no_lintel_when_the_door_reaches_the_ceiling_or_there_is_none(ceiling: ZoneCeiling | None) -> None:
    door = Door(name='entry', start=Position(3, 0), end=Position(4, 0), height=2.0)
    assert asyncio.run(_room(ceiling, [door]).door_lintels()) == []


def test_elevator_destination_is_the_first_cabin() -> None:
    assert Elevator(name='e', position=Position(0, 0), destinations=[' up ', 'down']).destinations == ['up', 'down']
    assert Elevator(name='e', position=Position(0, 0), destinations=['up', 'down']).destination == 'up'
    assert Elevator(name='e', position=Position(0, 0)).destination == ''


def test_elevator_levels_maps_cabins_to_their_level() -> None:
    world = World.from_levels(
        Level(id='ground', elevators=[Elevator(name='lift_g', position=Position(0, 0), destinations=['lift_t'])]),
        Level(id='top', elevators=[Elevator(name='lift_t', position=Position(0, 0), destinations=['lift_g']), Elevator(name='service', position=Position(5, 0))]),
    )
    assert world.elevator_levels() == {'lift_g': 'ground', 'lift_t': 'top', 'service': 'top'}
    assert world.validate() == []


def test_unknown_elevator_destination_raises_unless_its_level_was_not_loaded() -> None:
    world = World.from_levels(Level(elevators=[Elevator(name='lift', position=Position(0, 0), destinations=['gone'])]))
    with pytest.raises(ValueError, match="unknown destination 'gone'"):
        world.validate()
    assert world.validate(loaded_level_ids={'0'}) == ["elevator 'lift' has destination 'gone' on a level that was not loaded, it will have no destination"]


@pytest.mark.parametrize(
    'level',
    [
        Level(objects=[Obstacle(name='desk', pose=Pose(Position(1, 1)), model='desk', level='1')]),
        Level(humans=[DynamicObstacle(name='walker', pose=Pose(Position(1, 1)), model='human', level='1')]),
        Level(lights=[Light(name='lamp', fixture='bulb', pose=Pose(position=Position(1, 1, 2)), lumens=400.0, level='1')]),
        Level(sounds=[Sound(name='siren', asset_id='alarm', position=Position(1, 2), level='1')]),
    ],
)
def test_everything_a_level_lists_sits_on_it_and_takes_no_level(level: Level) -> None:
    with pytest.raises(ValueError, match='takes no level'):
        World.from_levels(level).validate()


def _door_side(*walls: list[list[float]]) -> str | None:
    world = World.from_levels(Level(walls=[WallPath(path=path) for path in walls], elevators=[Elevator(name='lift', position=Position(0, 0))]))
    world.apply_elevator_door_sides()
    return world.levels['0'].elevators[0].door_side


@pytest.mark.parametrize(
    ('walls', 'side'),
    [
        (([[1.0, -1], [1.0, 1]], [[1.2, -1], [1.2, 1]]), '+x'),
        (([[-1.0, -1], [-1.0, 1]], [[-1.2, -1], [-1.2, 1]]), '-x'),
        (([[-1, 1.0], [1, 1.0]], [[-1, 1.2], [1, 1.2]]), '+y'),
        (([[-1, -1.0], [1, -1.0]], [[-1, -1.2], [1, -1.2]]), '-y'),
    ],
    ids=['wall to +x', 'wall to -x', 'wall to +y', 'wall to -y'],
)
def test_door_side_faces_the_wall_the_cabin_stands_against(walls: tuple[list[list[float]], ...], side: str) -> None:
    assert _door_side(*walls) == side


def test_door_side_defaults_to_plus_x_without_a_near_wall() -> None:
    assert _door_side([[5, -1], [5, 1]]) == '+x'
    assert _door_side() == '+x'


def test_door_side_inference_keeps_an_explicit_side() -> None:
    world = World.from_levels(Level(walls=[WallPath(path=[[1, -1], [1, 1]])], elevators=[Elevator(name='lift', position=Position(0, 0), door_side='-y')]))
    world.apply_elevator_door_sides()
    assert world.levels['0'].elevators[0].door_side == '-y'


def _furnished(level_id: str) -> Level:
    return Level(
        id=level_id,
        zones=[_square(f'room_{level_id}', walls=ZoneWalls())],
        walls=[WallPath(path=[[0, 0], [1, 0]])],
        doors=[Door(name=f'door_{level_id}', start=Position(1, 0), end=Position(2, 0))],
        elevators=[Elevator(name=f'lift_{level_id}', position=Position(3, 3))],
        objects=[Obstacle(name=f'desk_{level_id}', model='Desk', pose=Pose(Position(1, 1)))],
        humans=[DynamicObstacle(name=f'walker_{level_id}', model='adult', pose=Pose(Position(2, 2)), waypoints=[[3, 3]])],
        lights=[
            Light(name=f'bulb_{level_id}', fixture='bulb', lumens=10, pose=Pose(Position(1, 1, 2))),
            Light(name=f'lamp_{level_id}', fixture='bulb', lumens=10, attach=f'desk_{level_id}', offset=Pose(Position(0, 0, 1))),
            Light(name=f'sun_{level_id}', fixture='sun', lux=100, pose=Pose()),
        ],
        sounds=[Sound(name=f'siren_{level_id}', asset_id='alarm', position=Position(1, 2)), Sound(name=f'radio_{level_id}', asset_id='radio', frame='base_link', offset=Position(0, 0, 1))],
        microphones=[
            WorldMicrophone(zone=f'room_{level_id}', placement='wall', position=Position(2, 2, 1)),
            WorldMicrophone(zone=f'room_{level_id}', placement='robot', position=Position(0, 0, 1), frame='base_link'),
        ],
        pedestrian_walls=level_id == '1',
    )


def test_compact_world_carries_every_list_shifted_by_the_origin() -> None:
    world = World.from_levels(_furnished('0'), _furnished('1'))
    level = world.compact_world({'0': (0.0, 0.0), '1': (10.0, 20.0)})
    assert [zone.name for zone in level.zones] == ['room_0', 'room_1']
    assert [(point.x, point.y) for point in level.zones[1].polygon] == [(10, 20), (14, 20), (14, 24), (10, 24)]
    assert [(point.x, point.y) for point in level.walls[1].path] == [(10, 20), (11, 20)]
    assert [(door.start.x, door.start.y, door.end.x, door.end.y) for door in level.doors] == [(1, 0, 2, 0), (11, 20, 12, 20)]
    assert [(elevator.position.x, elevator.position.y) for elevator in level.elevators] == [(3, 3), (13, 23)]
    assert [(entity.pose.position.x, entity.pose.position.y) for entity in level.objects] == [(1, 1), (11, 21)]
    walker = level.humans[1]
    assert (walker.pose.position.x, walker.pose.position.y) == (12, 22)
    assert [(point.x, point.y) for point in walker.waypoints] == [(13, 23)]
    bulb, lamp, sun = level.lights[3:]
    assert bulb.pose is not None and (bulb.pose.position.x, bulb.pose.position.y, bulb.pose.position.z) == (11, 21, 2)
    assert lamp.pose is None and lamp.offset.position == Position(0, 0, 1)
    assert sun.pose == Pose()
    siren, radio = level.sounds[2:]
    assert siren.position == Position(11, 22)
    assert radio.offset == Position(0, 0, 1)
    in_map, on_robot = level.microphones[2:]
    assert in_map.position == Position(12, 22, 1)
    assert on_robot.position == Position(0, 0, 1)
    assert len(level.lights) == 6 and len(level.sounds) == 4 and len(level.microphones) == 4
    assert level.pedestrian_walls is True
    assert world.levels['1'].zones[0].polygon[0] == Position(0, 0)


def test_compact_world_requires_every_origin() -> None:
    world = World.from_levels(_furnished('0'), _furnished('1'))
    with pytest.raises(KeyError, match='origin for level 1 was not given'):
        world.compact_world({'0': (0.0, 0.0)})


def test_shifted_moves_the_level() -> None:
    moved = _furnished('0').shifted(1.0, -1.0)
    assert moved.zones[0].polygon[0] == Position(1, -1)
    assert moved.objects[0].pose.position == Position(2, 0)


def test_from_levels_numbers_levels_without_an_id() -> None:
    world = World.from_levels(Level(), Level(id='roof'), Level())
    assert list(world.levels) == ['0', 'roof', '2']
    assert [level.id for level in world.levels.values()] == ['0', 'roof', '2']


def test_level_origins_lay_levels_side_by_side() -> None:
    world = World.from_levels(Level(zones=[_square('a', x=2, y=3)]), Level(zones=[_square('b', x=-1, y=0, size=2)]))
    assert world.level_origins() == {'0': (-2.0, -3.0), '1': (10.0, 0.0)}


def test_level_origins_wrap_into_rows() -> None:
    world = World.from_levels(Level(zones=[_square('a', size=20)]), Level(zones=[_square('b', size=20)]))
    assert world.level_origins(preferred_pixel_width=100) == {'0': (0.0, 0.0), '1': (0.0, -25.0)}


def test_lookup_zone_polygon_resolves_zones_doors_and_elevators() -> None:
    level = Level(zones=[_square('room')], doors=[Door(name='front', start=Position(1, 0), end=Position(2, 0))], elevators=[Elevator(name='lift', position=Position(3, 3), size=[2.0, 2.0, 2.0])])
    assert level.zone_ref_names() == ['room', 'front', 'lift']
    assert level.lookup_zone_polygon('room') == level.zones[0].polygon
    assert level.lookup_zone_polygon('front') == door_polygon(Position(1, 0), Position(2, 0))
    assert level.lookup_zone_polygon('front') == [Position(1, -0.3), Position(2, -0.3), Position(2, 0.3), Position(1, 0.3)]
    assert level.lookup_zone_polygon('lift') == [Position(2, 2), Position(4, 2), Position(4, 4), Position(2, 4)]
    assert level.lookup_zone_polygon('nowhere') is None
