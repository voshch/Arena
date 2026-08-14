"""Pure-logic tests for the runtime semantic kinds and manager.

Skipped if task_generator / arena_simulation_setup aren't importable (no
sourced overlay), since the door/elevator runtime types live there.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("task_generator.shared")
pytest.importorskip("arena_runtime.sim._mechanism_shim")

from arena_runtime.sim._mechanism_shim import (  # noqa: E402
    _DoorRuntime,
    _DoorState,
    _ElevatorRuntime,
)
from arena_runtime.sim._semantics import (  # noqa: E402
    DoorSemantics,
    ElevatorSemantics,
    SemanticChange,
    SemanticsManager,
)
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position  # noqa: E402
from task_generator.shared import Door, Elevator  # noqa: E402

# ---------------------------------------------------------------------------
# hand-built fixtures
# ---------------------------------------------------------------------------


class _Cfg:
    """Stand-in for SemanticCfg: only .role and .name are read by the kinds."""

    def __init__(self, role: str, name: str) -> None:
        self.role = role
        self.name = name


_DOOR_PRESET = [
    _Cfg("state", "state"),
    _Cfg("state", "progress"),
    _Cfg("predicate", "open"),
    _Cfg("predicate", "in_transit"),
    _Cfg("predicate", "triggered"),
]

_ELEVATOR_PRESET = [
    _Cfg("state", "arriving_eta"),
    _Cfg("state", "occupants"),
    _Cfg("predicate", "departing"),
    _Cfg("predicate", "in_transit"),
    _Cfg("predicate", "dispatched"),
    _Cfg("predicate", "just_arrived"),
]


class _Logger:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def get_child(self, _name: str) -> "_Logger":
        return self


class _Node:
    def __init__(self) -> None:
        self._logger = _Logger()

    def get_logger(self) -> _Logger:
        return self._logger


class _Mech:
    def __init__(self) -> None:
        self.node = _Node()
        self._door_runtime: dict[str, _DoorRuntime] = {}
        self._elevator_runtime: dict[str, _ElevatorRuntime] = {}


def _door(name: str = "d", hold_time: float = 2.0) -> Door:
    return Door(
        name=name,
        start=Position(0.0, 0.0, 0.0),
        end=Position(1.0, 0.0, 0.0),
        kind="sliding",
        activation_distance=(1.5, 1.5),
        transition_time=1.0,
        hold_time=hold_time,
    )


def _door_runtime(name: str = "d", hold_time: float = 2.0) -> _DoorRuntime:
    closed = Pose(position=Position(0.0, 0.0, 1.0), orientation=Orientation.from_yaw(0.0))
    open_p = Pose(position=Position(0.0, 1.0, 1.0), orientation=Orientation.from_yaw(0.0))
    return _DoorRuntime(door=_door(name, hold_time), closed_pose=closed, open_pose=open_p, effective_kind="sliding")


def _elevator(name: str = "e") -> Elevator:
    return Elevator(
        name=name,
        position=Position(0.0, 0.0, 0.0),
        size=[2.0, 2.0, 2.5],
        door_side="+x",
        destination="other",
    )


def _elevator_runtime(name: str = "e") -> _ElevatorRuntime:
    return _ElevatorRuntime(elevator=_elevator(name), door_name=f"{name}/door", destination_name="other")


def _attach_door(rt: _DoorRuntime, entity: str = "d") -> DoorSemantics:
    mech = _Mech()
    mech._door_runtime[entity] = rt
    return DoorSemantics.attach(mech, entity, _DOOR_PRESET)


def _attach_elevator(rt: _ElevatorRuntime, entity: str = "e") -> ElevatorSemantics:
    mech = _Mech()
    mech._elevator_runtime[entity] = rt
    return ElevatorSemantics.attach(mech, entity, _ELEVATOR_PRESET)


# ---------------------------------------------------------------------------
# door snapshots against representative FSM states (C1)
# ---------------------------------------------------------------------------


def test_door_snapshot_closed():
    rt = _door_runtime()
    rt.state = _DoorState.CLOSED
    rt.progress = 0.0
    rt.last_trigger_sim_time = -math.inf
    inst = _attach_door(rt)
    inst.step(10.0)
    snap = inst.snapshot()
    assert snap.kind == "door"
    assert snap.discrete == {"state": "closed"}
    assert snap.continuous == {"progress": 0.0}
    assert snap.predicates == {"open": False, "in_transit": False, "triggered": False}


def test_door_snapshot_opening_and_triggered():
    rt = _door_runtime(hold_time=2.0)
    rt.state = _DoorState.OPENING
    rt.progress = 0.4
    rt.last_trigger_sim_time = 10.0
    inst = _attach_door(rt)
    inst.step(10.0)
    snap = inst.snapshot()
    assert snap.discrete["state"] == "opening"
    assert snap.continuous["progress"] == pytest.approx(0.4)
    assert snap.predicates["open"] is False
    assert snap.predicates["in_transit"] is True
    assert snap.predicates["triggered"] is True


def test_door_snapshot_open():
    rt = _door_runtime()
    rt.state = _DoorState.OPEN
    rt.progress = 1.0
    inst = _attach_door(rt)
    inst.step(0.0)
    snap = inst.snapshot()
    assert snap.predicates["open"] is True
    assert snap.predicates["in_transit"] is False


def test_door_snapshot_closing_is_in_transit():
    rt = _door_runtime()
    rt.state = _DoorState.CLOSING
    rt.progress = 0.6
    inst = _attach_door(rt)
    inst.step(0.0)
    assert inst.snapshot().predicates["in_transit"] is True


def test_door_triggered_stale_is_false():
    rt = _door_runtime(hold_time=2.0)
    rt.last_trigger_sim_time = 5.0
    inst = _attach_door(rt)
    inst.step(10.0)  # 5s old, hold_time 2 -> stale
    assert inst.snapshot().predicates["triggered"] is False


def test_door_attach_ignores_cfgs_publishes_full_vocab():
    """Mechanism semantics are intrinsic: a restricted (or bogus) cfg list is ignored,
    attach always binds the kind's full DISCRETE/CONTINUOUS/PREDICATES vocabulary."""
    rt = _door_runtime()
    mech = _Mech()
    mech._door_runtime["d"] = rt
    inst = DoorSemantics.attach(mech, "d", [_Cfg("predicate", "sideways")])
    inst.step(0.0)
    snap = inst.snapshot()
    assert snap.discrete == {"state": "closed"}
    assert snap.continuous == {"progress": 0.0}
    assert snap.predicates == {"open": False, "in_transit": False, "triggered": False}


def test_door_attach_with_no_cfgs_publishes_full_vocab():
    """A bare, unannotated door still attaches: state publishes because the door exists."""
    rt = _door_runtime()
    mech = _Mech()
    mech._door_runtime["d"] = rt
    inst = DoorSemantics.attach(mech, "d", [])
    inst.step(0.0)
    snap = inst.snapshot()
    assert snap.discrete == {"state": "closed"}
    assert snap.continuous == {"progress": 0.0}
    assert snap.predicates == {"open": False, "in_transit": False, "triggered": False}


# ---------------------------------------------------------------------------
# elevator snapshots against representative FSM states (C1)
# ---------------------------------------------------------------------------


def test_elevator_snapshot_idle():
    rt = _elevator_runtime()
    inst = _attach_elevator(rt)
    inst.step(3.0)
    snap = inst.snapshot()
    assert snap.kind == "elevator"
    assert snap.continuous["arriving_eta"] == pytest.approx(-1.0)
    assert snap.continuous["occupants"] == pytest.approx(0.0)
    assert snap.predicates == {
        "departing": False,
        "in_transit": False,
        "dispatched": False,
        "just_arrived": False,
    }


def test_elevator_snapshot_scheduled_arrival():
    rt = _elevator_runtime()
    rt.arriving_eta = 8.0
    inst = _attach_elevator(rt)
    inst.step(3.0)
    snap = inst.snapshot()
    assert snap.continuous["arriving_eta"] == pytest.approx(5.0)
    assert snap.predicates["in_transit"] is True


def test_elevator_arriving_eta_clamped_to_zero():
    rt = _elevator_runtime()
    rt.arriving_eta = 2.0
    inst = _attach_elevator(rt)
    inst.step(5.0)  # already past the eta
    assert inst.snapshot().continuous["arriving_eta"] == pytest.approx(0.0)


def test_elevator_occupants_counts_all_committed_sets():
    rt = _elevator_runtime()
    rt.dispatched = {"r1"}
    rt.pending_occupants = (("r2", (0.0, 0.0)),)
    rt.just_arrived = {"r3": True, "r4": False}
    inst = _attach_elevator(rt)
    inst.step(0.0)
    snap = inst.snapshot()
    assert snap.continuous["occupants"] == pytest.approx(4.0)
    assert snap.predicates["dispatched"] is True
    assert snap.predicates["just_arrived"] is True


def test_elevator_departing_predicate():
    rt = _elevator_runtime()
    rt.departing = True
    inst = _attach_elevator(rt)
    inst.step(0.0)
    assert inst.snapshot().predicates["departing"] is True


# ---------------------------------------------------------------------------
# reset mutates every FSM field the spec lists
# ---------------------------------------------------------------------------


def test_door_reset_restores_all_fields():
    rt = _door_runtime()
    rt.state = _DoorState.OPEN
    rt.progress = 1.0
    rt.last_trigger_sim_time = 5.0
    rt.last_applied_progress = 0.7
    inst = _attach_door(rt)
    inst.reset()
    assert rt.state == _DoorState.CLOSED
    assert rt.progress == 0.0
    assert rt.last_trigger_sim_time == -math.inf
    assert rt.last_applied_progress == -1.0


def test_elevator_reset_restores_all_fields():
    rt = _elevator_runtime()
    rt.arriving_eta = 8.0
    rt.pending_occupants = (("r", (1.0, 2.0)),)
    rt.just_arrived = {"r": True}
    rt.departing = True
    rt.dispatched = {"r"}
    inst = _attach_elevator(rt)
    inst.reset()
    assert rt.arriving_eta == -math.inf
    assert rt.pending_occupants == ()
    assert rt.just_arrived == {}
    assert rt.departing is False
    assert rt.dispatched == set()


def test_door_reset_defers_geometry_via_last_applied_progress():
    """reset sets progress==0.0 and last_applied_progress==-1.0 so they differ, forcing one move_box."""
    rt = _door_runtime()
    inst = _attach_door(rt)
    inst.reset()
    assert rt.progress != rt.last_applied_progress


# ---------------------------------------------------------------------------
# SemanticsManager: step, snapshot, change emission
# ---------------------------------------------------------------------------


def _manager_with_door(rt: _DoorRuntime, entity: str = "d") -> tuple[SemanticsManager, list[list[SemanticChange]]]:
    mech = _Mech()
    mech._door_runtime[entity] = rt
    mgr = SemanticsManager(mech)
    mgr.set_sim("gazebo")
    batches: list[list[SemanticChange]] = []
    mgr.set_change_callback(lambda batch: batches.append(list(batch)))
    mgr.attach("door", entity, _DOOR_PRESET)
    return mgr, batches


def test_manager_snapshot_lists_attached_instances():
    rt = _door_runtime()
    mgr, _ = _manager_with_door(rt)
    snaps = mgr.snapshot()
    assert len(snaps) == 1
    assert snaps[0].entity == "d"
    assert snaps[0].kind == "door"


def test_manager_first_step_emits_all_fields():
    rt = _door_runtime()
    rt.state = _DoorState.CLOSED
    mgr, batches = _manager_with_door(rt)
    mgr.step(1.0)
    assert len(batches) == 1
    fields = {c.field for c in batches[0]}
    assert fields == {"state", "progress", "open", "in_transit", "triggered"}


def test_manager_no_change_no_callback():
    rt = _door_runtime()
    mgr, batches = _manager_with_door(rt)
    mgr.step(1.0)
    batches.clear()
    mgr.step(1.0)  # identical FSM state
    assert batches == []


def test_manager_emits_change_on_field_transition():
    rt = _door_runtime()
    rt.state = _DoorState.CLOSED
    rt.progress = 0.0
    mgr, batches = _manager_with_door(rt)
    mgr.step(1.0)
    batches.clear()
    rt.state = _DoorState.OPENING
    rt.progress = 0.5
    mgr.step(1.1)
    assert len(batches) == 1
    by_field = {c.field: c for c in batches[0]}
    assert by_field["state"].previous == "closed"
    assert by_field["state"].current == "opening"
    assert by_field["state"].field_kind == "state"
    assert by_field["progress"].current == "0.5"
    assert by_field["in_transit"].field_kind == "predicate"
    assert by_field["in_transit"].current == "true"
    assert "open" not in by_field


def test_manager_predicate_change_stringified_lowercase():
    rt = _door_runtime()
    rt.state = _DoorState.OPENING
    mgr, batches = _manager_with_door(rt)
    mgr.step(1.0)
    batches.clear()
    rt.state = _DoorState.OPEN
    rt.progress = 1.0
    mgr.step(1.1)
    by_field = {c.field: c for c in batches[0]}
    assert by_field["open"].previous == "false"
    assert by_field["open"].current == "true"


def test_manager_reset_clears_snapshot_cache_and_restores():
    rt = _door_runtime()
    rt.state = _DoorState.OPEN
    rt.progress = 1.0
    mgr, batches = _manager_with_door(rt)
    mgr.step(1.0)
    batches.clear()
    mgr.reset()
    assert rt.state == _DoorState.CLOSED
    mgr.step(2.0)
    # cache cleared: the post-reset step re-emits every field from empty.
    assert len(batches) == 1
    assert {c.field for c in batches[0]} == {"state", "progress", "open", "in_transit", "triggered"}


def test_manager_detach_removes_instance_and_cache():
    rt = _door_runtime()
    mgr, batches = _manager_with_door(rt)
    mgr.step(1.0)
    mgr.detach("d")
    assert mgr.snapshot() == []
    batches.clear()
    mgr.step(2.0)
    assert batches == []


# ---------------------------------------------------------------------------
# support matrix (C5)
# ---------------------------------------------------------------------------


def test_support_matrix_values():
    assert DoorSemantics.SUPPORTED_SIMS == frozenset({"gazebo", "isaac"})
    assert ElevatorSemantics.SUPPORTED_SIMS == frozenset({"gazebo", "isaac"})


def test_manager_skips_unsupported_sim():
    rt = _door_runtime()
    mech = _Mech()
    mech._door_runtime["d"] = rt
    mgr = SemanticsManager(mech)
    mgr.set_sim("dummy")
    mgr.attach("door", "d", _DOOR_PRESET)
    assert mgr.snapshot() == []
    assert any("unsupported" in msg for msg in mech.node.get_logger().infos)


def test_manager_attaches_on_supported_sim():
    rt = _door_runtime()
    mech = _Mech()
    mech._door_runtime["d"] = rt
    mgr = SemanticsManager(mech)
    mgr.set_sim("isaac")
    mgr.attach("door", "d", _DOOR_PRESET)
    assert len(mgr.snapshot()) == 1


def test_manager_attach_with_no_cfgs_publishes_full_door_vocab():
    """A spawned door with no annotation still snapshots the full door vocabulary (intrinsic)."""
    rt = _door_runtime()
    mech = _Mech()
    mech._door_runtime["d"] = rt
    mgr = SemanticsManager(mech)
    mgr.set_sim("gazebo")
    mgr.attach("door", "d")
    snaps = mgr.snapshot()
    assert len(snaps) == 1
    assert snaps[0].discrete == {"state": "closed"}
    assert snaps[0].continuous == {"progress": 0.0}
    assert snaps[0].predicates == {"open": False, "in_transit": False, "triggered": False}


def test_manager_attach_with_no_cfgs_publishes_full_elevator_vocab():
    """A spawned elevator with no annotation snapshots the full vocabulary, including the
    cabin door fields."""
    rt = _elevator_runtime()
    cabin = _door_runtime("e/door")
    mech = _Mech()
    mech._elevator_runtime["e"] = rt
    mech._door_runtime["e/door"] = cabin
    mgr = SemanticsManager(mech)
    mgr.set_sim("gazebo")
    mgr.attach("elevator", "e")
    snaps = mgr.snapshot()
    assert len(snaps) == 1
    assert snaps[0].discrete == {"cabin_door": "closed"}
    assert snaps[0].continuous == {
        "arriving_eta": -1.0,
        "occupants": 0.0,
        "cabin_door_progress": 0.0,
    }


def test_manager_skips_unknown_kind():
    mech = _Mech()
    mgr = SemanticsManager(mech)
    mgr.set_sim("gazebo")
    mgr.attach("teleporter", "x", _DOOR_PRESET)
    assert mgr.snapshot() == []
    assert any("unknown kind" in msg for msg in mech.node.get_logger().infos)
