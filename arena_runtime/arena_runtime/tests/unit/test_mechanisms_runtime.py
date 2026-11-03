"""Elevator, door and gate runtime driven through the real _tick against a duck-typed mechanism double."""

from __future__ import annotations

import asyncio
import math
import random

import pytest

pytest.importorskip("arena_runtime.shared")
pytest.importorskip("arena_runtime.sim._mechanism_shim")

from arena_runtime.sim._mechanism_shim import (  # noqa: E402
    TELEPORT_GRACE,
    _advance_state,
    _Arrival,
    _CarState,
    _check_door_conflicts,
    _door_geometry,
    _door_open_pose,
    _DoorRuntime,
    _DoorState,
    _effective_kind,
    _elevator_wall_segments,
    _ElevatorRuntime,
    _interp_pose,
    _step_elevator,
    _swept_slab_blocked,
    _tick,
    reset_mechanisms,
    shim_remove_doors,
    shim_remove_elevators,
    shim_spawn_doors,
    shim_spawn_elevators,
)
from arena_runtime.sim._semantics import GateSemantics, SemanticsManager  # noqa: E402
from arena_simulation_setup.shared.semantics import SemanticCfg  # noqa: E402
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position  # noqa: E402
from arena_runtime.shared import Door, Elevator  # noqa: E402

DT = 1.0 / 30.0


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, role, name, value=None, params=None):
        self.role = role
        self.name = name
        self.value = value
        self.params = params or {}


class _Logger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


class _Time:
    def __init__(self) -> None:
        self.seconds = 0.0

    def to_seconds(self) -> float:
        return self.seconds


class _Node:
    def __init__(self) -> None:
        self._logger = _Logger()
        self.sim_time = _Time()

    def get_logger(self) -> _Logger:
        return self._logger


class _BusyTask:
    """Stands in for a running mechanism loop so spawn helpers do not start a real one."""

    def done(self) -> bool:
        return False


class _HumanSim:
    def __init__(self, peds: dict[str, tuple[float, float]] | None = None, *, teleport_ok: bool = True) -> None:
        self.peds = dict(peds or {})
        self.wall_calls: list[tuple[str, object]] = []
        self.teleport_ok = teleport_ok

    def set_mechanism_walls(self, key, segments) -> None:
        self.wall_calls.append((key, segments))

    def pedestrian_discs(self):
        return [(name, xy, 0.25) for name, xy in self.peds.items()]

    async def pedestrian_teleport(self, destinations) -> bool:
        if self.teleport_ok:
            self.peds.update(destinations)
        return self.teleport_ok

    async def elevator_doors_changed(self, doors) -> None:
        pass


class _Mech:
    def __init__(self, human: _HumanSim | None = None) -> None:
        self.node = _Node()
        self._human_simulator = human
        self._incident_callback = None
        self._door_runtime: dict[str, _DoorRuntime] = {}
        self._elevator_runtime: dict[str, _ElevatorRuntime] = {}
        self._elevator_doors: dict[str, str] = {}
        self._door_primitives: dict[str, list[str]] = {}
        self._elevator_primitives: dict[str, list[str]] = {}
        self._stair_runtime: dict = {}
        self._stair_primitives: dict[str, list[str]] = {}
        self._mechanism_loop_task = _BusyTask()
        self._robots: dict[str, tuple[float, float]] = {}
        self._lights: dict = {}
        self._light_outputs: dict = {}
        self._light_poses: dict = {}
        self._articulated: dict = {}
        self.robot_teleport_ok = True
        self._semantics = SemanticsManager(self)
        self._semantics.set_sim("gazebo")
        self.boxes: dict[str, Pose] = {}

    def robot_discs(self):
        return [(name, xy, 0.3) for name, xy in self._robots.items()]

    def robot_pose(self, sim_path: str) -> Pose | None:
        xy = self._robots.get(sim_path)
        if xy is None:
            return None
        return Pose(position=Position(xy[0], xy[1], 0.0), orientation=Orientation.from_yaw(0.0))

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        if not self.robot_teleport_ok:
            return False
        self._robots[sim_path] = (pose.position.x, pose.position.y)
        return True

    async def spawn_box(self, name, size, pose, material=None) -> bool:
        self.boxes[name] = pose
        return True

    async def move_box(self, name, pose) -> bool:
        self.boxes[name] = pose
        return True

    async def delete_box(self, name) -> bool:
        self.boxes.pop(name, None)
        return True


def _door(name="d", start=(0.0, 0.0), end=(1.0, 0.0), *, kind="sliding", activation=1.5, hold_time=2.0, transition_time=1.0, swing_direction=1, width=0.1) -> Door:
    return Door(
        name=name,
        start=Position(start[0], start[1], 0.0),
        end=Position(end[0], end[1], 0.0),
        kind=kind,
        width=width,
        activation_distance=(activation, activation),
        transition_time=transition_time,
        hold_time=hold_time,
        swing_direction=swing_direction,
    )


def _elevator(name, position=(0.0, 0.0), destination="", *, capacity=None, travel_time=3.0, accept_outside_calls=True, call_mode="level") -> Elevator:
    return Elevator(
        name=name,
        position=Position(position[0], position[1], 0.0),
        size=[2.0, 2.0, 2.5],
        door_side="+x",
        destinations=[part.strip() for part in destination.split(",") if part.strip()],
        activation_distance=1.5,
        transition_time=1.0,
        hold_time=2.0,
        travel_time=travel_time,
        accept_outside_calls=accept_outside_calls,
        capacity=capacity,
        call_mode=call_mode,
    )


async def _spawn_elevators(mech: _Mech, *elevators: Elevator) -> None:
    assert await shim_spawn_elevators(mech, elevators)
    for elevator in elevators:
        mech._semantics.attach("elevator", elevator.name)


async def _run(mech: _Mech, seconds: float, robots=None) -> None:
    """Advance sim time by `seconds`, optionally pinning robot positions each tick (None = leave as is)."""
    for _ in range(int(round(seconds / DT))):
        mech.node.sim_time.seconds += DT
        if robots is not None:
            mech._robots.update(robots)
        await _tick(mech, DT)


def _snap(mech: _Mech, entity: str, kind: str):
    return next(s for s in mech._semantics.snapshot() if s.entity == entity and s.kind == kind)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_elevator_capacity_and_door_swing_schema():
    assert _elevator("e").capacity is None
    assert _elevator("e", capacity=2).capacity == 2
    with pytest.raises(ValueError):
        _elevator("e", capacity=0)
    assert _door().swing_direction == 1
    assert _door(swing_direction=-1).swing_direction == -1
    with pytest.raises(ValueError):
        _door(swing_direction=2)


def test_elevator_destinations_split():
    elevator = _elevator("e", destination="1.a, 2.b")
    assert _ElevatorRuntime(elevator=elevator, door_name="d", destination_name=",".join(elevator.destinations)).destinations == ("1.a", "2.b")
    assert _ElevatorRuntime(elevator=_elevator("e"), door_name="d", destination_name="").destinations == ()


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------


def test_full_cabin_ignores_outside_calls():
    a = _ElevatorRuntime(elevator=_elevator("a", capacity=1), door_name="a/door", destination_name="b")
    door = _DoorRuntime(door=_door("a/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    door.last_trigger_sim_time = -100.0
    _step_elevator(a, door, None, occupants=[("r1", (0.0, 0.0))], outside_trigger=True, now=1.0)
    assert a.full
    assert door.last_trigger_sim_time == -100.0
    assert a.departing


def test_cabin_with_room_still_answers_outside_calls():
    a = _ElevatorRuntime(elevator=_elevator("a", capacity=2), door_name="a/door", destination_name="b")
    door = _DoorRuntime(door=_door("a/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    _step_elevator(a, door, None, occupants=[("r1", (0.0, 0.0))], outside_trigger=True, now=1.0)
    assert not a.full
    assert door.last_trigger_sim_time == 1.0


def test_capacity_semantic_fields():
    async def scenario():
        mech = _Mech(_HumanSim({"p1": (0.2, 0.2)}))
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b", capacity=2), _elevator("b", (10.0, 0.0), "a"))
        await _run(mech, DT, robots={"r1": (-0.5, 0.0)})
        return _snap(mech, "a", "elevator"), _snap(mech, "b", "elevator")

    a, b = asyncio.run(scenario())
    assert a.continuous["live_occupants"] == 2.0
    assert a.continuous["capacity"] == 2.0
    assert a.predicates["full"] is True
    assert set(a.members) == {"r1", "p1"}
    assert b.continuous["capacity"] == -1.0
    assert b.predicates["full"] is False
    assert b.members == []


# ---------------------------------------------------------------------------
# multi-destination
# ---------------------------------------------------------------------------


def _multi_runtime() -> _ElevatorRuntime:
    return _ElevatorRuntime(elevator=_elevator("env/hub/0"), door_name="hub/door", destination_name="env/up/1,env/down/2")


def test_round_robin_and_target_resolution():
    rt = _multi_runtime()
    assert rt.destinations == ("env/up/1", "env/down/2")
    assert rt.current_destination() == "env/up/1"
    assert rt.resolve_target("2.down") == "env/down/2"
    assert rt.resolve_target("down") == "env/down/2"
    assert rt.resolve_target("env/up/1") == "env/up/1"
    with pytest.raises(ValueError, match="malformed value"):
        rt.resolve_target("3.sideways")


def test_round_robin_advances_one_step_per_departure():
    rt = _multi_runtime()
    up = _ElevatorRuntime(elevator=_elevator("env/up/1"), door_name="up/door", destination_name="env/hub/0")
    door = _DoorRuntime(door=_door("hub/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    rt.car = _CarState.DEPARTING
    rt.next_stop = rt.current_destination()
    _step_elevator(rt, door, up, occupants=[("r1", (0.0, 0.0))], outside_trigger=False, now=1.0)
    assert up.arrivals[0].source == "env/hub/0"
    assert rt.current_destination() == "env/down/2"


def test_target_write_overrides_round_robin_and_clears():
    mech = _Mech()
    rt = _multi_runtime()
    mech._elevator_runtime["env/hub/0"] = rt
    mech._semantics.attach("elevator", "env/hub/0")
    assert _snap(mech, "env/hub/0", "elevator").discrete["target"] == "env/up/1"
    assert mech._semantics.set_value("env/hub/0", "target", "2.down")
    assert rt.target_override == "env/down/2"
    rt.car = _CarState.DEPARTING
    rt.next_stop = rt.current_destination()
    door = _DoorRuntime(door=_door("hub/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    dest = _ElevatorRuntime(elevator=_elevator("env/down/2"), door_name="x", destination_name="")
    _step_elevator(rt, door, dest, occupants=[("r1", (0.0, 0.0))], outside_trigger=False, now=1.0)
    assert rt.current_destination() == "env/down/2"
    with pytest.raises(ValueError, match="malformed value"):
        mech._semantics.set_value("env/hub/0", "target", "nowhere")
    mech._semantics.set_value("env/hub/0", "target", "")
    assert rt.target_override is None
    reset_mechanisms(mech)
    assert rt.rr_index == 0


def test_single_destination_target_is_empty():
    mech = _Mech()
    mech._elevator_runtime["a"] = _ElevatorRuntime(elevator=_elevator("a"), door_name="a/door", destination_name="b")
    mech._semantics.attach("elevator", "a")
    assert _snap(mech, "a", "elevator").discrete["target"] == ""


def test_tick_multi_destination_rides_follow_target():
    """End to end: a hub with two destinations sends the rider to the written target."""

    async def scenario():
        mech = _Mech()
        await _spawn_elevators(
            mech,
            _elevator("hub", (0.0, 0.0), "up,down"),
            _elevator("up", (10.0, 0.0), "hub"),
            _elevator("down", (20.0, 0.0), "hub"),
        )
        mech._semantics.set_value("hub", "target", "down")
        await _run(mech, 0.2, robots={"r1": (-0.5, 0.0)})
        await _run(mech, 15.0)
        return mech

    mech = asyncio.run(scenario())
    assert mech._robots["r1"] == pytest.approx((19.5, 0.0))


# ---------------------------------------------------------------------------
# arrival queue
# ---------------------------------------------------------------------------


def test_two_cabins_sending_to_one_destination_are_served_one_after_the_other():
    """One car per group: r1's ride lands before the car fetches r2, both end in b."""

    async def scenario():
        mech = _Mech()
        await _spawn_elevators(
            mech,
            _elevator("a", (0.0, 0.0), "b"),
            _elevator("b", (10.0, 0.0), "a"),
            _elevator("c", (20.0, 0.0), "b"),
        )
        mech._robots = {"r1": (-0.5, 0.0)}
        await _run(mech, 0.5)
        mech._robots["r2"] = (19.5, 0.0)
        await _run(mech, 0.5)
        assert len(mech._elevator_runtime["b"].arrivals) == 1
        assert mech._elevator_runtime["c"].call_since < math.inf
        r1_landed = False
        for _ in range(900):
            await _run(mech, DT)
            if not r1_landed and mech._robots["r1"] == pytest.approx((9.5, 0.0)):
                r1_landed = True
                assert mech._robots["r2"] == (19.5, 0.0)
                mech._robots["r1"] = (12.5, 0.0)
        return mech

    mech = asyncio.run(scenario())
    assert mech._robots["r1"] == (12.5, 0.0)
    assert mech._robots["r2"] == pytest.approx((9.5, 0.0))
    assert mech._elevator_runtime["b"].arrivals == []
    assert mech._elevator_runtime["a"].dispatched == set()
    assert mech._elevator_runtime["c"].dispatched == set()


def test_arrival_queue_orders_by_eta():
    rt = _ElevatorRuntime(elevator=_elevator("b"), door_name="b/door", destination_name="a")
    rt.queue_arrival(_Arrival(eta=5.0, source="c", occupants=(("r2", (0.0, 0.0)),)))
    rt.queue_arrival(_Arrival(eta=3.0, source="a", occupants=(("r1", (0.0, 0.0)),)))
    assert rt.arriving_eta == 3.0
    assert rt.pending_occupants == (("r1", (0.0, 0.0)), ("r2", (0.0, 0.0)))
    door = _DoorRuntime(door=_door("b/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    first = _step_elevator(rt, door, None, occupants=[], outside_trigger=False, now=3.0)
    assert first.teleport_job == ("a", "b", [("r1", (0.0, 0.0))])
    second = _step_elevator(rt, door, None, occupants=[], outside_trigger=False, now=5.0)
    assert second.teleport_job == ("c", "b", [("r2", (0.0, 0.0))])
    assert rt.just_arrived == {"r1": False, "r2": False}


# ---------------------------------------------------------------------------
# failed teleports never seal a cabin forever
# ---------------------------------------------------------------------------


def test_refused_robot_teleport_releases_source_and_destination():
    async def scenario():
        mech = _Mech()
        mech.robot_teleport_ok = False
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "a"))
        mech._robots = {"r1": (-0.5, 0.0)}
        await _run(mech, 3.2)
        return mech

    mech = asyncio.run(scenario())
    assert mech._robots["r1"] == (-0.5, 0.0)
    assert "r1" not in mech._elevator_runtime["b"].just_arrived
    assert any("refused" in w for w in mech.node.get_logger().warnings)


def test_silent_teleport_failure_times_out_dispatched():
    a = _ElevatorRuntime(elevator=_elevator("a"), door_name="a/door", destination_name="b")
    door = _DoorRuntime(door=_door("a/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    a.dispatched = {"r1"}
    a.dispatch_deadline = 10.0
    inside = [("r1", (0.0, 0.0))]
    _step_elevator(a, door, None, occupants=inside, outside_trigger=False, now=9.0)
    assert a.dispatched == {"r1"}
    result = _step_elevator(a, door, None, occupants=inside, outside_trigger=False, now=10.5)
    assert result.dispatch_timeout == ("r1",)
    assert a.dispatched == set()
    assert a.departing


def test_unconfirmed_arrival_expires_after_grace():
    b = _ElevatorRuntime(elevator=_elevator("b"), door_name="b/door", destination_name="a")
    door = _DoorRuntime(door=_door("b/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    b.queue_arrival(_Arrival(eta=1.0, source="a", occupants=(("r1", (0.0, 0.0)),)))
    _step_elevator(b, door, None, occupants=[], outside_trigger=False, now=1.0)
    assert b.just_arrived == {"r1": False}
    _step_elevator(b, door, None, occupants=[], outside_trigger=False, now=1.0 + TELEPORT_GRACE - 0.1)
    assert b.just_arrived == {"r1": False}
    result = _step_elevator(b, door, None, occupants=[], outside_trigger=False, now=1.0 + TELEPORT_GRACE + 0.1)
    assert result.arrival_timeout == ("r1",)
    assert b.just_arrived == {}


# ---------------------------------------------------------------------------
# hinged doors
# ---------------------------------------------------------------------------


def _hinged_runtime(swing_direction=1, **kw) -> _DoorRuntime:
    door = _door("h", (0.0, 0.0), (1.0, 0.0), kind="hinged", swing_direction=swing_direction, **kw)
    kind = _effective_kind(_Logger(), door)
    _size, closed = _door_geometry(door)
    return _DoorRuntime(door=door, closed_pose=closed, open_pose=_door_open_pose(door, closed, kind), effective_kind=kind)


def test_hinged_is_implemented_not_teleport():
    rt = _hinged_runtime()
    assert rt.effective_kind == "hinged"


@pytest.mark.parametrize("swing_direction, side", [(1, 1.0), (-1, -1.0)])
def test_hinged_open_pose_rotates_about_start(swing_direction, side):
    rt = _hinged_runtime(swing_direction=swing_direction)
    assert rt.open_pose.position.x == pytest.approx(0.0, abs=1e-9)
    assert rt.open_pose.position.y == pytest.approx(0.5 * side)
    rt.progress = 0.5
    mid = _interp_pose(rt)
    assert math.hypot(mid.position.x, mid.position.y) == pytest.approx(0.5)
    assert mid.position.y * side > 0.0
    rt.progress = 0.0
    closed = _interp_pose(rt)
    assert (closed.position.x, closed.position.y) == pytest.approx((0.5, 0.0))


def test_hinged_swept_arc_blocks_open_on_swing_side_only():
    rt = _hinged_runtime(swing_direction=1)
    in_arc = [("r", (0.5, 0.6), 0.2)]
    other_side = [("r", (0.5, -0.6), 0.2)]
    assert _swept_slab_blocked(rt, 0.0, 1.0, in_arc)
    assert not _swept_slab_blocked(rt, 0.0, 1.0, other_side)
    assert not _swept_slab_blocked(rt, 0.0, 1.0, [("r", (3.0, 3.0), 0.2)])


def test_hinged_opening_holds_at_contact_and_close_reverses():
    rt = _hinged_runtime(swing_direction=1)
    rt.last_trigger_sim_time = 10.0
    blocker = [("r", (0.6, 0.12), 0.05)]
    assert not _swept_slab_blocked(rt, 0.0, 0.0, blocker)
    _advance_state(rt, DT, 10.0, blocker)
    assert rt.progress == 0.0
    rt.progress = 1.0
    rt.state = _DoorState.OPEN
    rt.last_trigger_sim_time = 0.0
    _advance_state(rt, DT, 10.0, blocker)
    assert rt.progress == 1.0
    assert rt.close_blocked
    assert rt.last_trigger_sim_time == 10.0
    _advance_state(rt, DT, 10.1, [])
    assert not rt.close_blocked
    _advance_state(rt, DT, 12.2, [])
    assert rt.state == _DoorState.CLOSING


def test_hinged_door_ticks_move_box_with_yaw():
    async def scenario():
        mech = _Mech()
        assert await shim_spawn_doors(mech, [_door("h", (0.0, 0.0), (1.0, 0.0), kind="hinged")])
        await _run(mech, 1.5, robots={"r": (0.5, -1.0)})
        return mech

    mech = asyncio.run(scenario())
    pose = mech.boxes["h"]
    assert mech._door_runtime["h"].progress == 1.0
    assert pose.position.y == pytest.approx(0.5)
    assert pose.orientation == Orientation.from_yaw(math.pi / 2.0)


# ---------------------------------------------------------------------------
# door semantics: members and blocked_close
# ---------------------------------------------------------------------------


def test_door_members_are_agents_in_activation_radius():
    async def scenario():
        mech = _Mech(_HumanSim({"p_near": (0.5, 1.0), "p_far": (0.5, 5.0)}))
        await shim_spawn_doors(mech, [_door("d", activation=1.5)])
        mech._semantics.attach("door", "d")
        await _run(mech, DT, robots={"r_near": (0.0, -1.0), "r_far": (9.0, 0.0)})
        return _snap(mech, "d", "door")

    snap = asyncio.run(scenario())
    assert set(snap.members) == {"p_near", "r_near"}
    assert snap.predicates["blocked_close"] is False


# ---------------------------------------------------------------------------
# human-sim wall hooks
# ---------------------------------------------------------------------------


def test_elevator_spawn_and_remove_push_cabin_walls():
    async def scenario():
        hs = _HumanSim()
        mech = _Mech(hs)
        elevator = _elevator("a", (0.0, 0.0), "b")
        await shim_spawn_elevators(mech, [elevator])
        await shim_remove_elevators(mech, ["a"])
        return hs, elevator

    hs, elevator = asyncio.run(scenario())
    assert hs.wall_calls[0] == ("elevator:a", _elevator_wall_segments(elevator))
    assert hs.wall_calls[-1] == ("elevator:a", None)


def test_elevator_wall_segments_leave_door_side_open():
    back, side_pos, side_neg = _elevator_wall_segments(_elevator("a"))
    assert {p[0] for p in back} == {-1.0}
    assert {p[1] for p in side_pos} == {1.0}
    assert {p[1] for p in side_neg} == {-1.0}


def test_door_without_activation_is_a_ped_wall_until_a_plate_drives_it():
    async def scenario():
        hs = _HumanSim()
        mech = _Mech(hs)
        await shim_spawn_doors(mech, [_door("d", activation=0.0), _door("open_door", (5.0, 0.0), (6.0, 0.0))])
        mech._semantics.attach("pressure_plate", "d", [_Cfg("predicate", "pressed", params={"position": [3.0, 3.0], "radius": 0.5, "drives": "d"})])
        await _run(mech, 0.2)
        calls_idle = list(hs.wall_calls)
        await _run(mech, 0.2, robots={"r": (3.0, 3.0)})
        calls_pressed = list(hs.wall_calls)
        await _run(mech, 0.2, robots={"r": (9.0, 9.0)})
        return calls_idle, calls_pressed, list(hs.wall_calls)

    idle, pressed, released = asyncio.run(scenario())
    assert idle == [("door:d", [((0.0, 0.0), (1.0, 0.0))])]
    assert pressed[-1] == ("door:d", None)
    assert len(pressed) == 2
    assert released[-1] == ("door:d", [((0.0, 0.0), (1.0, 0.0))])
    assert len(released) == 3


def test_locked_gate_walls_peds_until_authorized_robot_arrives():
    async def scenario():
        hs = _HumanSim()
        mech = _Mech(hs)
        await shim_spawn_doors(mech, [_door("g", activation=1.5)])
        mech._semantics.attach("gate", "g", [_Cfg("predicate", "locked", value=True, params={"authorized": ["robot1"]}), _Cfg("predicate", "blocked")])
        await _run(mech, 0.2)
        locked_calls = list(hs.wall_calls)
        await _run(mech, 0.2, robots={"env/robot1": (0.5, 1.0)})
        await shim_remove_doors(mech, ["g"])
        return locked_calls, list(hs.wall_calls)

    locked, after = asyncio.run(scenario())
    assert locked == [("door:g", [((0.0, 0.0), (1.0, 0.0))])]
    assert after[1] == ("door:g", None)
    assert len(after) == 2


def test_wall_hooks_are_skipped_without_a_human_sim():
    async def scenario():
        mech = _Mech(None)
        await shim_spawn_elevators(mech, [_elevator("a", (0.0, 0.0), "b")])
        await shim_spawn_doors(mech, [_door("d", (5.0, 0.0), (6.0, 0.0), activation=0.0)])
        await _run(mech, 0.1)
        return mech

    mech = asyncio.run(scenario())
    assert mech._door_runtime["d"].ped_wall is True
    assert not mech.node.get_logger().warnings


# ---------------------------------------------------------------------------
# gate default, plate drive lookup
# ---------------------------------------------------------------------------


def test_gate_without_locked_logs_default_once():
    mech = _Mech()
    mech._door_runtime["g_default_log"] = _DoorRuntime(door=_door("g_default_log"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    inst = GateSemantics.attach(mech, "g_default_log", [_Cfg("predicate", "locked"), _Cfg("predicate", "blocked")])
    GateSemantics.attach(mech, "g_default_log", [_Cfg("predicate", "locked"), _Cfg("predicate", "blocked")])
    assert inst._locked is True
    assert sum("authored without `locked`" in m for m in mech.node.get_logger().infos) == 1
    explicit = GateSemantics.attach(mech, "g_default_log", [_Cfg("predicate", "locked", value=False)])
    assert explicit._locked is False


def test_plate_driving_unknown_door_warns_once():
    mech = _Mech()
    mech._semantics.attach("pressure_plate", "p", [_Cfg("predicate", "pressed", value=None, params={"position": [0.0, 0.0], "radius": 1.0, "drives": "ghost"})])
    mech._semantics.set_value("p", "pressed", "true")
    mech._semantics.step(1.0)
    mech._semantics.apply_plate_drives(1.0)
    mech._semantics.apply_plate_drives(1.1)
    assert sum("ghost" in w for w in mech.node.get_logger().warnings) == 1


# ---------------------------------------------------------------------------
# call_mode: edge (crowded lobbies)
# ---------------------------------------------------------------------------


def test_call_mode_schema():
    assert _elevator("e").call_mode == "level"
    assert _elevator("e", call_mode="edge").call_mode == "edge"
    with pytest.raises(ValueError):
        _elevator("e", call_mode="pulse")


def _open_cabin(call_mode: str) -> tuple[_ElevatorRuntime, _DoorRuntime]:
    rt = _ElevatorRuntime(elevator=_elevator("a", call_mode=call_mode), door_name="a/door", destination_name="b")
    door = _DoorRuntime(door=_door("a/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    door.state = _DoorState.OPEN
    door.progress = 1.0
    door.last_trigger_sim_time = 0.0
    return rt, door


def test_edge_waiting_caller_does_not_refresh_hold_of_a_boarded_cabin():
    rt, door = _open_cabin("edge")
    waiting = frozenset({"p1"})
    rider = [("r1", (0.0, 0.0))]
    _step_elevator(rt, door, None, occupants=rider, outside_trigger=True, now=0.0, callers=waiting)
    assert door.last_trigger_sim_time == 0.0
    _step_elevator(rt, door, None, occupants=rider, outside_trigger=True, now=1.0, callers=waiting)
    assert door.last_trigger_sim_time == 0.0
    _step_elevator(rt, door, None, occupants=rider, outside_trigger=True, now=1.5, callers=frozenset({"p1", "p2"}))
    assert door.last_trigger_sim_time == 1.5


def test_edge_waiting_caller_holds_an_empty_open_cabin():
    rt, door = _open_cabin("edge")
    for now in (0.5, 1.0, 1.5):
        _step_elevator(rt, door, None, occupants=[], outside_trigger=True, now=now, callers=frozenset({"p1"}))
        assert door.last_trigger_sim_time == now


def test_edge_waiting_caller_keeps_the_empty_car_open_without_a_reopen():
    async def scenario():
        mech = _Mech(_HumanSim({"p1": (1.4, 0.6)}))
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b", call_mode="edge"), _elevator("b", (10.0, 0.0), "a"))
        states = []
        for _ in range(int(round(12.0 / DT))):
            await _run(mech, DT)
            states.append(mech._door_runtime["a/door"].state)
        return states

    states = asyncio.run(scenario())
    first_open = states.index(_DoorState.OPEN)
    assert all(state == _DoorState.OPEN for state in states[first_open:])


def test_level_waiting_caller_refreshes_every_tick():
    rt, door = _open_cabin("level")
    for now in (0.5, 1.0, 1.5):
        _step_elevator(rt, door, None, occupants=[], outside_trigger=True, now=now, callers=frozenset({"p1"}))
        assert door.last_trigger_sim_time == now


def test_edge_closed_idle_cabin_answers_waiting_caller():
    rt, door = _open_cabin("edge")
    rt.hall_callers = frozenset({"p1"})
    door.state = _DoorState.CLOSED
    door.progress = 0.0
    _step_elevator(rt, door, None, occupants=[], outside_trigger=True, now=5.0, callers=frozenset({"p1"}))
    assert door.last_trigger_sim_time == 5.0


def test_edge_closed_cabin_with_rider_departs_despite_waiting_caller():
    rt, door = _open_cabin("edge")
    rt.hall_callers = frozenset({"p1"})
    door.state = _DoorState.CLOSED
    door.progress = 0.0
    door.last_trigger_sim_time = -100.0
    _step_elevator(rt, door, None, occupants=[("r1", (0.0, 0.0))], outside_trigger=True, now=5.0, callers=frozenset({"p1"}))
    assert rt.departing
    assert door.last_trigger_sim_time == -100.0


@pytest.mark.parametrize("call_mode", ["level", "edge"])
def test_crowded_lobby_tick(call_mode):
    """Peds idling at the hall door hold a loaded cabin at most hold_time, then it departs."""

    async def scenario():
        mech = _Mech(_HumanSim({"p1": (1.4, 0.6), "p2": (1.4, -0.6)}))
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b", call_mode=call_mode), _elevator("b", (10.0, 0.0), "a"))
        await _run(mech, 1.0)
        mech._robots = {"r1": (-0.5, 0.0)}
        await _run(mech, 10.0)
        return mech

    mech = asyncio.run(scenario())
    assert mech._robots["r1"] == pytest.approx((9.5, 0.0))


# ---------------------------------------------------------------------------
# one car per group
# ---------------------------------------------------------------------------


def _open(mech: _Mech, name: str) -> bool:
    return mech._door_runtime[f"{name}/door"].state != _DoorState.CLOSED


def test_a_call_at_the_cabin_without_the_car_brings_it_there_empty():
    async def scenario():
        mech = _Mech()
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "a"))
        cars = (mech._elevator_runtime["a"].car, mech._elevator_runtime["b"].car)
        mech._robots = {"r1": (11.8, 0.0)}
        opened_without_car = False
        for _ in range(int(round(8.0 / DT))):
            await _run(mech, DT)
            opened_without_car |= mech._elevator_runtime["b"].car != _CarState.PRESENT and _open(mech, "b")
        return mech, cars, opened_without_car

    mech, cars, opened_without_car = asyncio.run(scenario())
    assert cars == (_CarState.PRESENT, _CarState.ABSENT)
    assert not opened_without_car
    assert mech._elevator_runtime["b"].car == _CarState.PRESENT
    assert mech._elevator_runtime["a"].car == _CarState.ABSENT
    assert _open(mech, "b")
    assert mech._robots["r1"] == (11.8, 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_never_two_open_doors_in_a_group_under_random_calls(seed):
    async def scenario():
        mech = _Mech(_HumanSim())
        incidents: list[str] = []
        mech._incident_callback = incidents.append
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "c"), _elevator("c", (20.0, 0.0), "a"))
        rng = random.Random(seed)
        spots = [(x + dx, 0.0) for x in (0.0, 10.0, 20.0) for dx in (-0.5, 1.8, 4.0)]
        worst = 0
        for _ in range(60):
            mech._robots = {f"r{i}": rng.choice(spots) for i in range(3)}
            mech._human_simulator.peds = {f"p{i}": rng.choice(spots) for i in range(3)}
            for _ in range(int(round(1.0 / DT))):
                await _run(mech, DT)
                worst = max(worst, sum(_open(mech, name) for name in "abc"))
        return worst, incidents

    worst, incidents = asyncio.run(scenario())
    assert worst == 1
    assert incidents == []


def test_an_idle_car_answers_the_oldest_call_first():
    a = _ElevatorRuntime(elevator=_elevator("a", destination="b"), door_name="a/door", destination_name="b")
    b = _ElevatorRuntime(elevator=_elevator("b", destination="c"), door_name="b/door", destination_name="c", car=_CarState.ABSENT, call_since=5.0)
    c = _ElevatorRuntime(elevator=_elevator("c", destination="a"), door_name="c/door", destination_name="a", car=_CarState.ABSENT, call_since=3.0)
    door = _DoorRuntime(door=_door("a/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    _step_elevator(a, door, b, occupants=[], outside_trigger=False, now=6.0, group={"b": b, "c": c})
    assert a.departing
    assert a.next_stop == "c"
    _step_elevator(a, door, b, occupants=[], outside_trigger=False, now=6.1, group={"b": b, "c": c})
    assert a.car == _CarState.ABSENT
    assert c.car == _CarState.ARRIVING
    assert c.arrivals[0].occupants == ()


def test_an_empty_car_held_open_by_a_passerby_still_answers_a_call_elsewhere():
    async def scenario():
        mech = _Mech(_HumanSim({"p1": (1.8, 0.4)}))
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "a"))
        await _run(mech, 1.0)
        held = _open(mech, "a")
        mech._robots = {"r1": (11.8, 0.0)}
        await _run(mech, 9.0)
        return mech, held

    mech, held = asyncio.run(scenario())
    assert held
    assert mech._elevator_runtime["b"].car == _CarState.PRESENT
    assert not _open(mech, "a")


def test_an_arriving_empty_car_answers_its_callers_for_hold_time_after_its_door_opened():
    a = _ElevatorRuntime(elevator=_elevator("a", destination="b"), door_name="a/door", destination_name="b", car=_CarState.ABSENT)
    b = _ElevatorRuntime(elevator=_elevator("b", destination="a"), door_name="b/door", destination_name="a", car=_CarState.ABSENT, call_since=1.0)
    a.queue_arrival(_Arrival(eta=10.0, source="b", occupants=()))
    door = _DoorRuntime(door=_door("a/door"), closed_pose=Pose(), open_pose=Pose(), effective_kind="sliding")
    _step_elevator(a, door, b, occupants=[], outside_trigger=True, now=10.0, group={"b": b})
    assert a.car == _CarState.PRESENT
    for now in (11.0, 12.5, 12.9):
        _step_elevator(a, door, b, occupants=[], outside_trigger=True, now=now, group={"b": b})
        assert door.last_trigger_sim_time == now
    _step_elevator(a, door, b, occupants=[], outside_trigger=True, now=13.1, group={"b": b})
    assert door.last_trigger_sim_time == 12.9


def test_the_arrival_hold_ends_at_teleport_grace_when_a_call_waits():
    async def scenario():
        mech = _Mech()
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "a"))
        mech._robots = {"r1": (-0.5, 0.0)}
        await _run(mech, 4.0)
        landed = mech._robots["r1"]
        mech._robots["r2"] = (1.8, 0.0)
        await _run(mech, 8.0)
        held = _open(mech, "b") and mech._elevator_runtime["b"].car == _CarState.PRESENT
        await _run(mech, 10.0)
        return mech, landed, held

    mech, landed, held = asyncio.run(scenario())
    assert landed == pytest.approx((9.5, 0.0))
    assert held
    assert mech._robots["r1"] == pytest.approx((-0.5, 0.0))
    assert mech._elevator_runtime["a"].car == _CarState.PRESENT


def test_a_recall_brings_the_car_and_keeps_it_there_with_the_door_open():
    async def scenario():
        mech = _Mech()
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "a"))
        cfgs = [SemanticCfg(role="state", name="state", params={"windows": [{"start": 0.0, "end": 100.0, "value": "on"}], "regime": "alarm"})]
        assert mech._semantics.attach("schedule", "fire", cfgs)
        mech._semantics.set_recall("b", "alarm")
        await _run(mech, 8.0)
        arrived = (mech._elevator_runtime["b"].car, _open(mech, "b"), _open(mech, "a"))
        mech._robots = {"r1": (1.8, 0.0)}
        await _run(mech, 20.0)
        return mech, arrived

    mech, arrived = asyncio.run(scenario())
    assert arrived == (_CarState.PRESENT, True, False)
    assert mech._elevator_runtime["b"].car == _CarState.PRESENT
    assert _open(mech, "b")
    assert not _open(mech, "a")


def test_a_failed_teleport_brings_the_car_back_to_its_riders():
    async def scenario():
        mech = _Mech()
        mech.robot_teleport_ok = False
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "a"))
        mech._robots = {"r1": (-0.5, 0.0)}
        left = returned = False
        for _ in range(int(round(15.0 / DT))):
            await _run(mech, DT)
            left |= mech._elevator_runtime["b"].car == _CarState.PRESENT
            returned |= left and mech._elevator_runtime["a"].car == _CarState.PRESENT and _open(mech, "a")
        return mech, left, returned

    mech, left, returned = asyncio.run(scenario())
    assert left
    assert returned
    assert mech._robots["r1"] == (-0.5, 0.0)


def test_reset_puts_the_car_back_at_the_first_cabin():
    async def scenario():
        mech = _Mech()
        await _spawn_elevators(mech, _elevator("b", (10.0, 0.0), "a"), _elevator("a", (0.0, 0.0), "b"))
        mech._robots = {"r1": (11.8, 0.0)}
        await _run(mech, 6.0)
        moved = mech._elevator_runtime["b"].car
        reset_mechanisms(mech)
        return mech, moved

    mech, moved = asyncio.run(scenario())
    assert moved == _CarState.PRESENT
    assert mech._elevator_runtime["a"].car == _CarState.PRESENT
    assert mech._elevator_runtime["b"].car == _CarState.ABSENT
    assert mech._elevator_runtime["b"].call_since == math.inf


def test_two_open_doors_in_a_group_are_reported_once():
    async def scenario():
        mech = _Mech()
        incidents: list[str] = []
        mech._incident_callback = incidents.append
        await _spawn_elevators(mech, _elevator("a", (0.0, 0.0), "b"), _elevator("b", (10.0, 0.0), "a"))
        for name in "ab":
            mech._door_runtime[f"{name}/door"].state = _DoorState.OPEN
        _check_door_conflicts(mech)
        _check_door_conflicts(mech)
        return incidents

    incidents = asyncio.run(scenario())
    assert len(incidents) == 1
    assert "'a'" in incidents[0] and "'b'" in incidents[0]
