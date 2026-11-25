"""Scenario goto poses move from level-local to the flattened map frame once."""

from __future__ import annotations

import pytest

ORIGINS = {"0": (0.0, 0.0), "1": (0.0, -40.0)}


def _realizer():
    from arena_runtime.env.task_api import Realizer

    return Realizer(level_origins=dict(ORIGINS))


def _pose(x: float, y: float):
    from arena_runtime.shared import Orientation, Pose, Position

    return Pose(position=Position(x, y, 0.0), orientation=Orientation.from_yaw(0.0))


def _entry(*phases):
    from arena_simulation_setup.tree.World.Scenario import RobotGoal

    return RobotGoal(start=_pose(0.0, 0.0), phases=list(phases))


def test_robots_without_a_scenario_entry_are_idle():
    from task_generator.tasks.robots.scenario.impl import idle_robots

    assert idle_robots([("a", "m"), ("b", "m")], []) == {"a", "b"}


def test_only_robots_with_phases_are_active():
    from arena_runtime.env.task_api import GoToPhase
    from task_generator.tasks.robots.scenario.impl import idle_robots

    assert idle_robots([("a", "m"), ("b", "m"), ("c", "m")], [_entry(GoToPhase(pose=_pose(1.0, 1.0))), _entry()]) == {"b", "c"}


def test_literal_goto_pose_gets_its_goal_level_origin():
    from arena_runtime.env.task_api import GoToPhase
    from task_generator.tasks.robots.scenario.impl import localize_phase

    phase = localize_phase(GoToPhase(pose=_pose(3.0, 4.0)), "1", _realizer())
    assert (phase.pose.position.x, phase.pose.position.y) == (3.0, -36.0)


def test_zone_dispatch_pose_stays_in_the_map_frame():
    from arena_runtime.env.task_api import GoToPhase
    from task_generator.tasks.robots.scenario.impl import localize_phase

    phase = localize_phase(GoToPhase(pose=_pose(3.0, -36.0), target="kitchen", pose_zone="lobby"), "1", _realizer())
    assert (phase.pose.position.x, phase.pose.position.y) == (3.0, -36.0)


def test_loop_phases_are_localized_and_their_goto_poses_listed():
    from arena_runtime.env.task_api import GoToPhase, LoopPhase
    from task_generator.tasks.robots.scenario.impl import goto_poses, localize_phase

    loop = localize_phase(LoopPhase(phases=[GoToPhase(pose=_pose(1.0, 1.0)), GoToPhase(pose=_pose(5.0, -35.0), target="t", pose_zone="z")]), "1", _realizer())
    assert [(p.position.x, p.position.y) for p in goto_poses(loop)] == [(1.0, -39.0), (5.0, -35.0)]


def test_start_without_a_level_stands_on_the_sole_level_the_env_runs():
    from task_generator.tasks.robots.scenario.impl import to_map_frame

    realizer = _realizer()
    realizer.reset_levels(["1"])
    pose = to_map_frame(_pose(3.0, 4.0), "", "", realizer)
    assert (pose.position.x, pose.position.y) == (3.0, -36.0)


def test_scenario_zone_refs_are_not_shifted_twice():
    from arena_runtime.env.task_api import Realizer
    from task_generator.tasks.robots.scenario.impl import to_map_frame

    realizer = Realizer(Realizer._Configuration(prefix="env0"), level_origins={"0": (0.0, 0.0), "1": (50.0, 0.0)})
    literal = to_map_frame(_pose(5.0, 5.0), "", "1", realizer)
    assert literal.position.x == pytest.approx(55.0)
    ref = to_map_frame(_pose(55.0, 5.0), "lobby", "1", realizer)
    assert ref.position.x == pytest.approx(55.0)
    assert to_map_frame(_pose(5.0, 5.0), "", "", realizer).position.x == pytest.approx(5.0)


def _declared(**kwargs):
    from arena_simulation_setup.tree.World.Scenario import RobotGoal

    return RobotGoal(start=_pose(0.0, 0.0), **kwargs)


def test_unnamed_entries_without_models_pair_in_order():
    from task_generator.tasks.robots.scenario.impl import pair_robots

    assert pair_robots([("a", "x"), ("b", "y")], [_declared(), _declared()]) == {0: 0, 1: 1}


def test_named_entries_pair_by_name_whatever_the_live_order():
    from task_generator.tasks.robots.scenario.impl import pair_robots

    entries = [_declared(name="b", model="y"), _declared(name="a", model="x")]
    assert pair_robots([("a", "x"), ("b", "y")], entries) == {0: 1, 1: 0}


def test_unnamed_entries_with_models_pair_by_model():
    from task_generator.tasks.robots.scenario.impl import pair_robots

    entries = [_declared(model="y"), _declared(model="x")]
    assert pair_robots([("a", "x"), ("b", "y")], entries) == {0: 1, 1: 0}


def test_named_entry_without_a_live_robot_stays_unpaired():
    from task_generator.tasks.robots.scenario.impl import pair_robots

    assert pair_robots([("a", "x")], [_declared(name="ghost", model="x")]) == {}


def test_robot_off_the_named_entries_is_idle():
    from arena_runtime.env.task_api import GoToPhase
    from task_generator.tasks.robots.scenario.impl import idle_robots

    entries = [_declared(name="b", model="y", phases=[GoToPhase(pose=_pose(1.0, 1.0))])]
    assert idle_robots([("a", "x"), ("b", "y")], entries) == {"a"}


def test_fleet_is_declared_when_every_entry_names_a_model():
    from arena_runtime.env.task_layer import FleetEntry
    from task_generator.tasks.robots.scenario.impl import declared_fleet

    fleet = declared_fleet([{"name": "lead", "model": "jackal", "start": [0, 0]}, {"model": "burger", "start": [1, 1]}])
    assert fleet == [FleetEntry(model="jackal", name="lead"), FleetEntry(model="burger")]


def test_fleet_is_not_declared_when_an_entry_lacks_a_model_or_there_are_none():
    from task_generator.tasks.robots.scenario.impl import declared_fleet

    assert declared_fleet([{"model": "jackal"}, {"start": [1, 1]}]) is None
    assert declared_fleet([]) is None
