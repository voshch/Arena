from __future__ import annotations

import math

import pytest


@pytest.fixture()
def realizer():
    from task_generator.manager.realizer import Realizer

    config = Realizer._Configuration(x=1.0, y=2.0, prefix="world")
    return Realizer(config)


@pytest.fixture()
def zero_realizer():
    from task_generator.manager.realizer import Realizer

    config = Realizer._Configuration(x=0.0, y=0.0, prefix="")
    return Realizer(config)


@pytest.fixture()
def _pose():
    from arena_simulation_setup.utils.geometry import Pose

    return Pose


@pytest.fixture()
def _position():
    from arena_simulation_setup.utils.geometry import Position

    return Position


@pytest.fixture()
def _orientation():
    from arena_simulation_setup.utils.geometry import Orientation

    return Orientation


def test_realize_none_returns_prefixed_namespace(realizer):
    result = realizer.realize(None)
    assert isinstance(result, str)
    assert "world" in result


def test_realize_none_zero_prefix_returns_empty_or_slash(zero_realizer):
    result = zero_realizer.realize(None)
    assert isinstance(result, str)


def test_realize_str_adds_prefix(realizer):
    result = realizer.realize("robot1")
    assert isinstance(result, str)
    assert "robot1" in result
    assert "world" in result


def test_realize_str_zero_prefix(zero_realizer):
    result = zero_realizer.realize("robot1")
    assert "robot1" in result


def test_realize_position_translates_xy(_position, realizer):
    from arena_simulation_setup.utils.geometry import Position

    p = Position(x=3.0, y=4.0, z=5.0)
    result = realizer.realize(p)
    assert isinstance(result, Position)
    assert math.isclose(result.x, 4.0)
    assert math.isclose(result.y, 6.0)
    assert math.isclose(result.z, 5.0)


def test_realize_position_zero_offset_unchanged(zero_realizer, _position):
    from arena_simulation_setup.utils.geometry import Position

    p = Position(x=7.0, y=-3.0, z=1.0)
    result = zero_realizer.realize(p)
    assert math.isclose(result.x, 7.0)
    assert math.isclose(result.y, -3.0)


def test_realize_pose_translates_position(_pose, realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    p = Pose(Position(0.0, 0.0), Orientation.from_yaw(0.5))
    result = realizer.realize(p)
    assert math.isclose(result.position.x, 1.0)
    assert math.isclose(result.position.y, 2.0)
    assert math.isclose(result.orientation.to_yaw(), 0.5, abs_tol=1e-6)


def test_realize_pose_orientation_preserved(realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    yaw = 1.2
    p = Pose(Position(1.0, 1.0), Orientation.from_yaw(yaw))
    result = realizer.realize(p)
    assert math.isclose(result.orientation.to_yaw(), yaw, abs_tol=1e-6)


def test_ezilear_translates_pose_by_negative_offset(realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    p = Pose(Position(5.0, 7.0), Orientation.from_yaw(0.5))
    result = realizer.ezilear(p)
    assert math.isclose(result.position.x, 4.0)
    assert math.isclose(result.position.y, 5.0)
    assert math.isclose(result.orientation.to_yaw(), 0.5, abs_tol=1e-6)


def test_ezilear_round_trips_with_realize(realizer):
    from arena_simulation_setup.utils.geometry import Pose, Position, Orientation

    p = Pose(Position(3.0, -2.5), Orientation.from_yaw(1.2))
    assert math.isclose(realizer.ezilear(realizer.realize(p)).position.x, p.position.x)
    assert math.isclose(realizer.ezilear(realizer.realize(p)).position.y, p.position.y)
    assert math.isclose(realizer.realize(realizer.ezilear(p)).position.x, p.position.x)
    assert math.isclose(realizer.realize(realizer.ezilear(p)).position.y, p.position.y)


def test_realize_wall_translates_start_end(realizer):
    from arena_simulation_setup.shared import Wall
    from arena_simulation_setup.utils.geometry import Position

    w = Wall(start=Position(0, 0), end=Position(1, 0))
    result = realizer.realize(w)
    assert math.isclose(result.start.x, 1.0)
    assert math.isclose(result.start.y, 2.0)
    assert math.isclose(result.end.x, 2.0)
    assert math.isclose(result.end.y, 2.0)


def test_realize_door_translates_and_prefixes_name(realizer):
    from arena_simulation_setup.shared import Door
    from arena_simulation_setup.utils.geometry import Position

    d = Door(start=Position(0, 0), end=Position(1, 0), name="door1", extra={})
    result = realizer.realize(d)
    assert "door1" in result.name
    assert math.isclose(result.start.x, 1.0)
    assert math.isclose(result.end.x, 2.0)


def test_realize_floor_translates_and_prefixes_name(realizer):
    from arena_simulation_setup.shared import Floor
    from arena_simulation_setup.utils.geometry import Position

    f = Floor(pos=Position(0, 0), name="floor1", extra={})
    result = realizer.realize(f)
    assert "floor1" in result.name
    assert math.isclose(result.pos.x, 1.0)
    assert math.isclose(result.pos.y, 2.0)


def test_realize_door_preserves_semantics_with_prefixed_name(realizer):
    from arena_simulation_setup.shared import Door
    from arena_simulation_setup.utils.geometry import Position

    d = Door(start=Position(0, 0), end=Position(1, 0), name="door1", semantics=[{"preset": "gate"}], extra={})
    result = realizer.realize(d)
    assert "door1" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in d.semantics]


def test_realize_elevator_translates_and_prefixes(realizer):
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position

    e = Elevator(position=Position(0, 0), name="elev1", extra={})
    result = realizer.realize(e)
    assert "elev1" in result.name
    assert math.isclose(result.position.x, 1.0)


def test_realize_elevator_preserves_semantics_with_prefixed_name(realizer):
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position

    e = Elevator(position=Position(0, 0), name="elev1", semantics=[{"preset": "pressure_plate"}], extra={})
    result = realizer.realize(e)
    assert "elev1" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in e.semantics]


def test_realize_elevator_no_destination(realizer):
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position

    e = Elevator(position=Position(0, 0), name="elev2", extra={})
    result = realizer.realize(e)
    assert (result.destinations, result.destination) == ([], "")


def test_realize_elevator_destination_names_the_cabin_on_its_level():
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position

    from task_generator.manager.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(prefix="env_0"), level_origins={"0": (0.0, 0.0), "1": (30.0, 0.0)})
    realizer.reset_elevator_levels({"lift_0": "0", "lift_1": "1"})
    result = realizer.realize(Elevator(position=Position(0, 0), name="lift_0", destinations=["lift_1"]), "0")
    assert result.name == realizer.prefix("lift_0", "0")
    assert result.destinations == [realizer.prefix("lift_1", "1")]
    assert result.destination == realizer.prefix("lift_1", "1")


def test_realize_elevator_levels_reset_drops_the_prior_world():
    from arena_simulation_setup.shared import Elevator
    from arena_simulation_setup.utils.geometry import Position

    from task_generator.manager.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(prefix="env_0"))
    realizer.reset_elevator_levels({"lift_1": "1"})
    realizer.reset_elevator_levels({})
    result = realizer.realize(Elevator(position=Position(0, 0), name="lift_0", destinations=["lift_1"]))
    assert result.destinations == [realizer.prefix("lift_1")]


def test_realize_dynamic_obstacle_waypoints_follow_their_own_level():
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.shared.entities import Waypoint
    from arena_simulation_setup.utils.geometry import Pose, Position

    from task_generator.manager.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0), level_origins={"0": (0.0, 0.0), "1": (30.0, 0.0)})
    human = DynamicObstacle(name="walker", pose=Pose(Position(0, 0)), model="human", waypoints=[Waypoint(x=1.0, y=0.0, level="0"), Waypoint(x=1.0, y=0.0, level="1")])
    result = realizer.realize(human, "1")
    assert (result.pose.position.x, result.pose.position.y) == (31.0, 2.0)
    assert [(w.x, w.y) for w in result.waypoints] == [(2.0, 2.0), (32.0, 2.0)]


def test_realize_dynamic_obstacle_waypoint_without_level_stands_on_its_human_level():
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.shared.entities import Waypoint
    from arena_simulation_setup.utils.geometry import Pose, Position

    from task_generator.manager.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0), level_origins={"0": (0.0, 0.0), "1": (30.0, 0.0)})
    human = DynamicObstacle(name="walker", pose=Pose(Position(0, 0)), model="human", waypoints=[Waypoint.from_any([1.0, 0.0]), Waypoint.from_any({"x": 1.0, "y": 0.0})])
    result = realizer.realize(human, "1")
    assert [(w.x, w.y) for w in result.waypoints] == [(32.0, 2.0), (32.0, 2.0)]


def test_realize_dynamic_obstacle_zone_waypoint_stays_in_the_map_frame():
    import numpy as np
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.shared.entities import Waypoint
    from arena_simulation_setup.utils.geometry import PointResolver, Pose, Position
    from arena_simulation_setup.utils.resolution import activate_resolver

    from task_generator.manager.realizer import Realizer

    with activate_resolver(PointResolver(lookup={"lobby_1": [Position(30.0, 0.0)]}.get, rng=np.random.default_rng(0))):
        waypoint = Waypoint.from_any("lobby_1")
    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0), level_origins={"1": (30.0, 0.0)})
    human = DynamicObstacle(name="walker", pose=Pose(Position(0, 0)), model="human", waypoints=[waypoint])
    result = realizer.realize(human, "1")
    assert [(w.x, w.y) for w in result.waypoints] == [(31.0, 2.0)]


def _scenario_human(pose: object):
    import numpy as np
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.utils.cattrs import converter
    from arena_simulation_setup.utils.geometry import PointResolver, Position

    c = converter.copy()
    c.set_resolver(PointResolver(lookup={"lobby_1": [Position(32.0, 3.0)]}.get, rng=np.random.default_rng(0)))
    return c.structure({"name": "walker", "model": "human", "level": "1", "pose": pose}, DynamicObstacle)


def test_realize_level_human_placed_by_zone_name_stands_at_the_zone():
    from task_generator.manager.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0), level_origins={"0": (0.0, 0.0), "1": (30.0, 0.0)})
    human = _scenario_human("lobby_1")
    result = realizer.realize(human, human.level)
    assert (result.pose.position.x, result.pose.position.y) == (33.0, 5.0)


def test_realize_level_human_with_literal_pose_gets_its_level_origin():
    from task_generator.manager.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0), level_origins={"0": (0.0, 0.0), "1": (30.0, 0.0)})
    human = _scenario_human([2.0, 3.0])
    result = realizer.realize(human, human.level)
    assert (result.pose.position.x, result.pose.position.y) == (33.0, 5.0)
    assert human.placement_zone == ""


def test_realize_level_object_placed_by_zone_name_stands_at_the_zone():
    import numpy as np
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.utils.cattrs import converter
    from arena_simulation_setup.utils.geometry import PointResolver, Position

    from task_generator.manager.realizer import Realizer

    c = converter.copy()
    c.set_resolver(PointResolver(lookup={"lobby_1": [Position(32.0, 3.0)]}.get, rng=np.random.default_rng(0)))
    desk = c.structure({"name": "desk", "model": "desk", "level": "1", "pose": "lobby_1"}, Obstacle)
    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0), level_origins={"1": (30.0, 0.0)})
    result = realizer.realize(desk, desk.level)
    assert (result.pose.position.x, result.pose.position.y) == (33.0, 5.0)


def test_placement_level_is_none_for_a_zone_else_its_own_the_listing_then_the_sole_one():
    from task_generator.manager.realizer import Realizer

    realizer = Realizer(level_origins={"0": (0.0, 0.0), "1": (30.0, 0.0)})
    realizer.reset_levels(["0", "1"])
    assert realizer.placement_level("lobby_1", "1", "1") == ""
    assert realizer.placement_level("", "1", "0") == "1"
    assert realizer.placement_level("", "", "0") == "0"
    assert realizer.placement_level("", "", "") == ""
    realizer.reset_levels(["1"])
    assert realizer.placement_level("", "", "") == "1"


def test_realize_light_stands_on_its_own_level_or_at_its_zone():
    import numpy as np
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.cattrs import converter
    from arena_simulation_setup.utils.geometry import PointResolver, Position

    from task_generator.manager.realizer import Realizer

    c = converter.copy()
    c.set_resolver(PointResolver(lookup={"lobby_1": [Position(32.0, 3.0)]}.get, rng=np.random.default_rng(0)))
    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0), level_origins={"0": (0.0, 0.0), "1": (30.0, 0.0)})
    literal = realizer.realize(c.structure({"name": "lamp", "fixture": "bulb", "lumens": 400.0, "level": "1", "pose": {"x": 2.0, "y": 3.0, "z": 2.0}}, Light))
    zoned = realizer.realize(c.structure({"name": "lamp", "fixture": "bulb", "lumens": 400.0, "level": "1", "pose": "lobby_1"}, Light))
    assert (literal.position.x, literal.position.y) == (33.0, 5.0)
    assert (zoned.position.x, zoned.position.y) == (33.0, 5.0)


def test_realize_dynamic_obstacle_translates_pose_and_waypoints(realizer):
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.utils.geometry import Position, Pose

    obs = DynamicObstacle(
        name="human1",
        pose=Pose(Position(0, 0)),
        model="human",
        waypoints=[Position(1, 0), Position(2, 1)],
        extra={},
    )
    result = realizer.realize(obs)
    assert math.isclose(result.pose.position.x, 1.0)
    assert math.isclose(result.pose.position.y, 2.0)
    assert len(result.waypoints) == 2
    assert math.isclose(result.waypoints[0].x, 2.0)
    assert math.isclose(result.waypoints[1].x, 3.0)
    assert math.isclose(result.waypoints[1].y, 3.0)


def test_realize_dynamic_obstacle_empty_waypoints(realizer):
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.utils.geometry import Position, Pose

    obs = DynamicObstacle(
        name="human2",
        pose=Pose(Position(0, 0)),
        model="human",
        waypoints=[],
        extra={},
    )
    result = realizer.realize(obs)
    assert result.waypoints == []


def test_realize_unknown_type_raises(realizer):
    with pytest.raises(TypeError, match="realization not implemented"):
        realizer.realize(object())


def test_realize_schedule_prefixes_name(realizer):
    from arena_simulation_setup.shared import Schedule

    s = Schedule(name="fire_alarm", semantics=[{"preset": "schedule"}], extra={})
    result = realizer.realize(s)
    assert "fire_alarm" in result.name
    assert "world" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in s.semantics]


def test_realize_signal_prefixes_name(realizer):
    from arena_simulation_setup.shared import Signal

    s = Signal(name="crossing_1", semantics=[{"preset": "signal"}], extra={})
    result = realizer.realize(s)
    assert "crossing_1" in result.name
    assert "world" in result.name
    assert [(cfg.role, cfg.name) for cfg in result.semantics] == [(cfg.role, cfg.name) for cfg in s.semantics]


def test_realize_light_translates_position_and_prefixes_name(realizer):
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.geometry import Pose, Position

    light = Light(name="exit_strip", fixture="tube", pose=Pose(position=Position(7.8, 3.0, 2.2)), lumens=300.0, light_on="blackout")
    result = realizer.realize(light)
    assert "exit_strip" in result.name
    assert "world" in result.name
    assert (result.position.x, result.position.y, result.position.z) == (8.8, 5.0, 2.2)
    assert result.light_on == "blackout"


def test_realize_light_twice_from_the_same_source(realizer):
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.geometry import Pose, Position

    light = Light(name="exit_strip", fixture="tube", pose=Pose(position=Position(7.8, 3.0, 2.2)), lumens=300.0)
    first = realizer.realize(light)
    second = realizer.realize(light)
    assert (first.name, first.position) == (second.name, second.position)


def test_realize_light_rig_translates_every_fixture(realizer):
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.geometry import Position

    rig = Light(name="hall", fixture="panel", lumens=3600.0, rig=True, fixtures=[Position(1.0, 1.0, 2.58), Position(3.4, 1.0, 2.58)], fixture_ranks=[0.2, 0.7])
    result = realizer.realize(rig)
    assert [(p.x, p.y, p.z) for p in result.fixtures] == [(2.0, 3.0, 2.58), (4.4, 3.0, 2.58)]
    assert result.fixture_ranks == [0.2, 0.7]
    assert result.pose is None


def test_realize_light_prefixes_its_frame_once(realizer):
    from arena_simulation_setup.shared import Light

    bare = realizer.realize(Light(name="headlight", fixture="spot", frame="jackal/base_link", lumens=800.0))
    assert bare.frame.endswith("/jackal/base_link")
    assert "world" in bare.frame
    assert realizer.realize(Light(name="headlight", fixture="spot", frame=bare.frame, lumens=800.0)).frame == bare.frame


def test_realize_ambient_light_keeps_no_position(realizer):
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.geometry import Orientation, Pose

    result = realizer.realize(Light(name="ambient", fixture="dome", lux=5.0))
    assert result.pose is None
    assert result.fixtures == []
    aim = Pose(orientation=Orientation.from_euler((0.0, -0.5, 0.0)))
    assert realizer.realize(Light(name="sun", fixture="sun", lux=400.0, pose=aim)).pose == aim


def test_realize_polygon_translates_corners(realizer):
    from arena_simulation_setup.utils.geometry import Position

    corners = [Position(0, 0), Position(2, 0), Position(2, 2), Position(0, 2)]
    ring = realizer.realize_polygon(corners)
    assert ring == [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)]


def test_realize_polygon_zero_offset(zero_realizer):
    from arena_simulation_setup.utils.geometry import Position

    corners = [Position(1, 1), Position(3, 1), Position(3, 3)]
    ring = zero_realizer.realize_polygon(corners)
    assert ring == [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0)]


def test_entity_sim_path_set_after_realize(realizer):
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.utils.geometry import Pose, Position

    obs = Obstacle(
        name="box1",
        pose=Pose(Position(0, 0)),
        model="box",
        extra={},
    )
    result = realizer.realize(obs)
    assert "box1" in result.sim_path


def _stacked(realizer):
    realizer.reset_level_origins({"0": (0.0, 0.0), "1": (17.0, 0.0)})
    realizer.reset_placement({"0": (0.0, 0.0, 0.0), "1": (9.2, 0.0, 3.0)}, {"0": (0.0, 0.0, 14.0, 10.0), "1": (0.0, 0.0, 14.0, 10.0)})
    return realizer


def test_placement_without_stack_offsets_keeps_the_abstract_layout(realizer):
    realizer.reset_level_origins({"0": (0.0, 0.0), "1": (17.0, 0.0)})
    placement = realizer.placement()
    assert not placement.stacked
    assert placement.shift("1") == (0.0, 0.0, 0.0)


def test_place_moves_a_realized_level_1_entity_by_its_stack_shift(realizer):
    from arena_simulation_setup.shared import Ceiling, Stairs, Wall
    from arena_simulation_setup.utils.geometry import Position

    realizer = _stacked(realizer)
    wall = realizer.realize(Wall(start=Position(1.0, 1.0, 0.0), end=Position(2.0, 1.0, 0.0), top=2.6, bottom=None), "1")
    placed = realizer.place(wall, "1")
    assert (placed.start.x - wall.start.x, placed.start.y - wall.start.y, placed.start.z - wall.start.z) == pytest.approx((-7.8, 0.0, 3.0))
    assert (placed.top, placed.bottom) == (2.6, None)
    ceiling = realizer.place(realizer.realize(Ceiling(name="c", pos=Position(3.0, 3.0, 0.0), z=2.5), "1"), "1")
    assert ceiling.z == pytest.approx(5.5)
    stair = Stairs(name="up", position=Position(3.7, 2.0, 0.0), top=True, to="down")
    stair.joined = True
    realized = realizer.realize(stair, "1")
    assert realized.joined
    assert realizer.place(realized, "1").joined
    assert realizer.place(realized, "1").position.z == pytest.approx(3.0)
    assert realizer.place(wall, "0") is wall


def test_placement_finds_the_level_of_an_abstract_point_with_the_env_offset(realizer):
    placement = _stacked(realizer).placement()
    assert placement.env == (1.0, 2.0)
    assert placement.level_at(1.0 + 17.0 + 3.0, 2.0 + 4.0) == "1"
    assert placement.level_at(1.0 + 3.0, 2.0 + 4.0) == "0"
