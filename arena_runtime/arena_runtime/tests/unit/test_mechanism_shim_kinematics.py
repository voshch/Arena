from __future__ import annotations

import math

import pytest


@pytest.fixture
def shim():
    from arena_runtime.sim import _mechanism_shim

    return _mechanism_shim


def _door(start, end, *, kind="sliding", width=0.1, height=2.0, activation=1.5):
    from arena_runtime.shared import Door, Position

    return Door(name="d", start=Position(*start), end=Position(*end), kind=kind, width=width, height=height, activation_distance=(activation, activation))


def _runtime(shim, door, kind="sliding"):
    _, closed = shim._door_geometry(door)
    return shim._DoorRuntime(door=door, closed_pose=closed, open_pose=shim._door_open_pose(door, closed, kind), effective_kind=kind)


@pytest.mark.parametrize(("kind", "effective"), [("hinged", "hinged"), ("teleport", "teleport"), ("sliding_top", "sliding_top"), ("sliding", "sliding")])
def test_effective_kind_keeps_each_door_kind(shim, kind, effective):
    import rclpy.logging

    logger = rclpy.logging.get_logger("test_mechanism_shim_kinematics")
    assert shim._effective_kind(logger, _door((0, 0, 0), (1, 0, 0), kind=kind)) == effective


def test_door_geometry_spans_start_to_end_and_stands_on_the_floor(shim):
    size, pose = shim._door_geometry(_door((1.0, 1.0, 0.5), (1.0, 3.0, 0.5), width=0.2, height=2.4))
    assert size == pytest.approx((2.0, 0.2, 2.4))
    assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx((1.0, 2.0, 1.7))
    assert pose.orientation.to_yaw() == pytest.approx(math.pi / 2)


def test_sliding_door_opens_by_its_full_length_along_a_diagonal(shim):
    door = _door((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
    _, closed = shim._door_geometry(door)
    opened = shim._door_open_pose(door, closed, "sliding")
    assert (opened.position.x, opened.position.y, opened.position.z) == pytest.approx((4.5, 6.0, 1.0))
    assert opened.orientation == closed.orientation


def test_zero_length_sliding_door_stays_at_its_closed_pose(shim):
    door = _door((2.0, 2.0, 0.0), (2.0, 2.0, 0.0))
    _, closed = shim._door_geometry(door)
    assert shim._door_open_pose(door, closed, "sliding") is closed


def test_near_zero_length_door_measures_from_its_single_point(shim):
    door = _door((1.0, 1.0, 0.0), (1.0, 1.0, 0.0))
    assert shim._near_door_segment(door, [(5.0, 5.0), (1.5, 1.0)], 0.6)
    assert not shim._near_door_segment(door, [(5.0, 5.0), (2.0, 1.0)], 0.6)


def test_near_door_clamps_positions_before_the_start_to_the_start(shim):
    door = _door((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert shim._near_door_segment(door, [(-1.0, 0.5)], 1.2)
    assert not shim._near_door_segment(door, [(-1.0, 0.5)], 1.1)


def test_swept_slab_ignores_a_zero_length_door_and_an_empty_disc_list(shim):
    point = _runtime(shim, _door((1.0, 1.0, 0.0), (1.0, 1.0, 0.0)))
    assert not shim._swept_slab_blocked(point, 0.0, 1.0, [("r", (1.0, 1.0), 0.5)])
    span = _runtime(shim, _door((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    assert not shim._swept_slab_blocked(span, 0.0, 1.0, [])


def test_swept_slab_of_a_vertical_door_covers_only_the_slot(shim):
    slab = _runtime(shim, _door((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), kind="teleport"), kind="teleport")
    assert shim._swept_slab_blocked(slab, 0.0, 1.0, [("r", (1.9, 0.2), 0.2)])
    assert not shim._swept_slab_blocked(slab, 0.0, 1.0, [("r", (3.0, 0.0), 0.2)])


def test_sliding_entry_checks_only_the_wall_pocket_while_opening(shim):
    rt = _runtime(shim, _door((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    in_pocket = [("ped", (2.9, 0.0), 0.05)]
    assert shim._sliding_entry_blocked(rt, 0.0, 0.5, in_pocket)
    assert not shim._sliding_entry_blocked(rt, 0.0, 0.5, [("ped", (3.2, 0.0), 0.1)])
    assert not shim._sliding_entry_blocked(rt, 0.5, 0.5, in_pocket)
    assert not shim._sliding_entry_blocked(rt, 0.5, 0.0, in_pocket)
    assert not shim._sliding_entry_blocked(rt, 0.0, 0.5, [])


def test_sliding_entry_ignores_a_zero_length_door(shim):
    rt = _runtime(shim, _door((1.0, 1.0, 0.0), (1.0, 1.0, 0.0)))
    assert not shim._sliding_entry_blocked(rt, 0.0, 1.0, [("ped", (1.0, 1.0), 1.0)])


def test_elevator_without_a_door_side_opens_on_plus_x(shim):
    from arena_runtime.shared import Elevator, Position

    elevator = Elevator(name="e", position=Position(4.0, 1.0, 0.0), size=[2.0, 3.0, 2.5], door_side=None)
    start, end = shim._door_slot(elevator)
    assert start.x == end.x == pytest.approx(5.0 - shim.DOOR_INSET)
    assert sorted((start.y, end.y)) == pytest.approx([-0.5 + shim.DOOR_INSET, 2.5 - shim.DOOR_INSET])


def test_scheduled_arrival_without_riders_only_opens_the_door(shim):
    from arena_runtime.shared import Elevator, Position

    elevator = Elevator(name="b", position=Position(0.0, 0.0, 0.0), size=[2.0, 2.0, 2.5], destinations=["a"])
    runtime = shim._ElevatorRuntime(elevator=elevator, door_name="b/door", destination_name="a")
    runtime.queue_arrival(shim._Arrival(eta=5.0, source="a", occupants=()))
    door = _runtime(shim, shim._elevator_synthesized_door(elevator))
    result = shim._step_elevator(runtime, door, None, occupants=[], outside_trigger=False, now=5.0)
    assert result.teleport_job is None
    assert not result.missing_destination
    assert runtime.arriving_eta == -math.inf
    assert runtime.just_arrived == {}
    assert door.last_trigger_sim_time == 5.0
