"""Pure-logic tests for the M2 semantic kinds, write path, regime bus, quantization and hooks.

Skipped if task_generator / arena_simulation_setup aren't importable (no sourced overlay),
since the door/elevator runtime types live there.
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
    GateSemantics,
    OccupancyCapSemantics,
    PressurePlateSemantics,
    ScheduleSemantics,
    SemanticEntitySnapshot,
    SemanticsManager,
    SignalSemantics,
    _diff,
    _in_polygon,
)
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position  # noqa: E402
from task_generator.shared import Door, Elevator  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class _MCfg:
    """Stand-in for SemanticCfg with M2 params/value fields."""

    def __init__(self, role, name, value=None, params=None):
        self.role = role
        self.name = name
        self.value = value
        self.params = params or {}


class _Logger:
    def __init__(self):
        self.infos = []

    def info(self, msg):
        self.infos.append(msg)

    def get_child(self, _name):
        return self


class _Node:
    def __init__(self):
        self._logger = _Logger()

    def get_logger(self):
        return self._logger


class _HumanSim:
    def __init__(self, peds):
        self._peds = list(peds)

    def pedestrian_discs(self):
        return [(name, xy, 0.3) for name, xy in self._peds]


class _Mech:
    def __init__(self, robots=(), peds=None):
        self.node = _Node()
        self._door_runtime = {}
        self._elevator_runtime = {}
        self._robots = list(robots)
        self._human_simulator = _HumanSim(peds) if peds is not None else None
        self._semantics = None

    def robot_discs(self):
        return [(name, xy, 0.3) for name, xy in self._robots]


def _manager(mech, sim="gazebo"):
    mgr = SemanticsManager(mech)
    mgr.set_sim(sim)
    mech._semantics = mgr
    return mgr


def _door(name="d", start=(0.0, 0.0), end=(1.0, 0.0), hold_time=2.0):
    return Door(
        name=name,
        start=Position(start[0], start[1], 0.0),
        end=Position(end[0], end[1], 0.0),
        kind="sliding",
        activation_distance=(1.5, 1.5),
        transition_time=1.0,
        hold_time=hold_time,
    )


def _door_runtime(name="d", **kw):
    closed = Pose(position=Position(0.0, 0.0, 1.0), orientation=Orientation.from_yaw(0.0))
    open_p = Pose(position=Position(0.0, 1.0, 1.0), orientation=Orientation.from_yaw(0.0))
    return _DoorRuntime(door=_door(name, **kw), closed_pose=closed, open_pose=open_p, effective_kind="sliding")


def _elevator(name="e"):
    return Elevator(name=name, position=Position(0.0, 0.0, 0.0), size=[2.0, 2.0, 2.5], door_side="+x", destination="other")


def _elevator_runtime(name="e"):
    return _ElevatorRuntime(elevator=_elevator(name), door_name=f"{name}/door", destination_name="other")


def _signal_cfgs(phases, stop_phases=None, regime=None):
    params = {"phases": phases}
    if stop_phases is not None:
        params["stop_phases"] = stop_phases
    if regime is not None:
        params["regime"] = regime
    return [
        _MCfg("state", "state", params=params),
        _MCfg("state", "phase_remaining"),
        _MCfg("predicate", "stop"),
    ]


def _schedule_cfgs(windows, default="", regime=None):
    params = {"windows": windows, "default": default}
    if regime is not None:
        params["regime"] = regime
    return [
        _MCfg("state", "state", params=params),
        _MCfg("predicate", "active"),
        _MCfg("state", "window_remaining"),
    ]


def _gate_cfgs(authorized=(), locked=None, unlock_on=None):
    params = {"authorized": list(authorized)}
    if unlock_on is not None:
        params["unlock_on"] = unlock_on
    return [
        _MCfg("predicate", "locked", value=locked, params=params),
        _MCfg("predicate", "blocked"),
    ]


def _plate_cfgs(position, radius, drives=None, latch=0.0, press_on=None, regime=None):
    params = {"position": list(position), "radius": radius, "latch": latch}
    if drives is not None:
        params["drives"] = drives
    if press_on is not None:
        params["press_on"] = press_on
    if regime is not None:
        params["regime"] = regime
    return [_MCfg("predicate", "pressed", params=params)]


def _occupancy_cfgs(cap):
    return [
        _MCfg("state", "occupancy", params={"cap": cap}),
        _MCfg("state", "cap"),
        _MCfg("predicate", "over_cap"),
    ]


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------


def _attach_signal(mech, cfgs, entity="sig"):
    return SignalSemantics.attach(mech, entity, cfgs)


def test_signal_cycles_phases_from_sim_time():
    inst = _attach_signal(_Mech(), _signal_cfgs([{"name": "green", "duration": 5.0}, {"name": "red", "duration": 3.0}]))
    inst.step(0.0)
    snap = inst.snapshot()
    assert snap.kind == "signal"
    assert snap.discrete["state"] == "green"
    assert snap.continuous["phase_remaining"] == pytest.approx(5.0)
    assert snap.predicates["stop"] is False
    inst.step(6.0)  # 6 - t0(0) = 6, into the red phase (5..8)
    snap = inst.snapshot()
    assert snap.discrete["state"] == "red"
    assert snap.continuous["phase_remaining"] == pytest.approx(2.0)
    assert snap.predicates["stop"] is True  # red is the default stop phase (last)


def test_signal_t0_defers_to_first_step():
    inst = _attach_signal(_Mech(), _signal_cfgs([{"name": "a", "duration": 4.0}, {"name": "b", "duration": 4.0}]))
    inst.step(100.0)  # t0 homes here
    assert inst.snapshot().discrete["state"] == "a"
    inst.step(105.0)  # 5s in -> phase b
    assert inst.snapshot().discrete["state"] == "b"


def test_signal_reset_restores_t0():
    inst = _attach_signal(_Mech(), _signal_cfgs([{"name": "a", "duration": 4.0}, {"name": "b", "duration": 4.0}]))
    inst.step(0.0)
    inst.step(5.0)
    assert inst.snapshot().discrete["state"] == "b"
    inst.reset()
    inst.step(50.0)  # new t0
    assert inst.snapshot().discrete["state"] == "a"


def test_signal_writable_state_snaps_t0():
    inst = _attach_signal(_Mech(), _signal_cfgs([{"name": "green", "duration": 5.0}, {"name": "red", "duration": 3.0}]))
    inst.step(0.0)
    inst.set_value("state", "red")
    snap = inst.snapshot()
    assert snap.discrete["state"] == "red"
    assert snap.continuous["phase_remaining"] == pytest.approx(3.0)


def test_signal_writable_rejects_unknown_phase():
    inst = _attach_signal(_Mech(), _signal_cfgs([{"name": "green", "duration": 5.0}]))
    inst.step(0.0)
    with pytest.raises(ValueError, match="malformed value"):
        inst.set_value("state", "purple")


def test_signal_writable_rejects_unwritable_field():
    inst = _attach_signal(_Mech(), _signal_cfgs([{"name": "green", "duration": 5.0}]))
    with pytest.raises(ValueError, match="field not writable"):
        inst.set_value("phase_remaining", "3.0")


def test_signal_explicit_stop_phases_and_regime():
    inst = _attach_signal(
        _Mech(),
        _signal_cfgs([{"name": "go", "duration": 5.0}, {"name": "caution", "duration": 2.0}], stop_phases=["caution"], regime="halt"),
    )
    inst.step(0.0)
    assert inst.asserts_regime("halt") is False
    inst.step(6.0)  # caution phase
    assert inst.snapshot().predicates["stop"] is True
    assert inst.asserts_regime("halt") is True
    assert inst.asserts_regime("other") is False


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def _attach_schedule(mech, cfgs, entity="sched"):
    return ScheduleSemantics.attach(mech, entity, cfgs)


def test_schedule_windows_flip_state_and_active():
    inst = _attach_schedule(_Mech(), _schedule_cfgs([{"start": 10.0, "end": 20.0, "value": "on"}], default="off"))
    inst.step(0.0)  # t0
    snap = inst.snapshot()
    assert snap.discrete["state"] == "off"
    assert snap.predicates["active"] is False
    assert snap.continuous["window_remaining"] == pytest.approx(-1.0)
    inst.step(15.0)
    snap = inst.snapshot()
    assert snap.discrete["state"] == "on"
    assert snap.predicates["active"] is True
    assert snap.continuous["window_remaining"] == pytest.approx(5.0)
    inst.step(25.0)
    assert inst.snapshot().predicates["active"] is False


def test_schedule_regime_while_active():
    inst = _attach_schedule(_Mech(), _schedule_cfgs([{"start": 5.0, "end": 10.0, "value": "x"}], regime="alarm"))
    inst.step(0.0)
    assert inst.asserts_regime("alarm") is False
    inst.step(7.0)
    assert inst.asserts_regime("alarm") is True


def test_schedule_override_persists_until_boundary():
    inst = _attach_schedule(_Mech(), _schedule_cfgs([{"start": 10.0, "end": 20.0, "value": "on"}], default="off"))
    inst.step(0.0)
    inst.set_value("active", "true")
    assert inst.snapshot().predicates["active"] is True
    inst.step(2.0)  # natural still false, no boundary crossed: override holds
    assert inst.snapshot().predicates["active"] is True
    inst.step(15.0)  # natural becomes true: boundary crossed, override cleared
    assert inst.snapshot().predicates["active"] is True  # natural now true anyway
    inst.step(25.0)  # natural false again
    assert inst.snapshot().predicates["active"] is False


def test_schedule_reset_clears_override_and_t0():
    inst = _attach_schedule(_Mech(), _schedule_cfgs([{"start": 10.0, "end": 20.0, "value": "on"}], default="off"))
    inst.step(0.0)
    inst.set_value("active", "true")
    inst.reset()
    inst.step(0.0)
    assert inst.snapshot().predicates["active"] is False


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _attach_gate(mech, cfgs, entity="d"):
    mech._door_runtime[entity] = _door_runtime(entity)
    return GateSemantics.attach(mech, entity, cfgs)


def test_gate_default_locked_blocks_unauthorized():
    mech = _Mech(robots=[("intruder", (0.5, 0.5))])
    mgr = _manager(mech)
    gate = _attach_gate(mech, _gate_cfgs(authorized=["jackal"]))
    mgr._instances["d"] = [gate]
    snap = gate.snapshot()
    assert snap.predicates["locked"] is True
    assert snap.predicates["blocked"] is True
    assert snap.members == ["intruder"]
    assert gate.trigger_allowed(0.0) is False
    assert mgr.trigger_allowed("d", 0.0) is False


def test_gate_authorized_agent_allows_trigger():
    mech = _Mech(robots=[("jackal", (0.5, 0.5))])
    _manager(mech)
    gate = _attach_gate(mech, _gate_cfgs(authorized=["jackal"]))
    assert gate.trigger_allowed(0.0) is True
    snap = gate.snapshot()
    assert snap.predicates["blocked"] is False  # authorized agent is not an unauthorized blocker
    assert snap.members == []


def test_gate_authorized_matches_bare_name_of_sim_path():
    mech = _Mech(robots=[("env_0/jackal", (0.5, 0.5))])
    _manager(mech)
    gate = _attach_gate(mech, _gate_cfgs(authorized=["jackal"]))
    assert gate.trigger_allowed(0.0) is True


def test_gate_initial_locked_from_value():
    mech = _Mech(robots=[("x", (0.5, 0.5))])
    _manager(mech)
    gate = _attach_gate(mech, _gate_cfgs(authorized=[], locked=False))
    assert gate.snapshot().predicates["locked"] is False
    assert gate.trigger_allowed(0.0) is True


def test_gate_unlock_on_regime_forces_unlocked():
    mech = _Mech(robots=[("x", (0.5, 0.5))])
    mgr = _manager(mech)
    mech._door_runtime["d"] = _door_runtime("d")
    sched = _attach_schedule(mech, _schedule_cfgs([{"start": 0.0, "end": 100.0, "value": "v"}], regime="alarm"))
    mgr._instances["sched"] = [sched]
    mgr._instances["d"] = [GateSemantics.attach(mech, "d", _gate_cfgs(authorized=[], unlock_on="alarm"))]
    mgr.step(5.0)
    assert mgr.regime("alarm") is True
    assert mgr.trigger_allowed("d", 5.0) is True
    gate = mgr._instances["d"][0]
    assert gate.snapshot().predicates["locked"] is False


def test_gate_writable_locked():
    mech = _Mech(robots=[("x", (5.0, 5.0))])
    _manager(mech)
    gate = _attach_gate(mech, _gate_cfgs(authorized=[]))
    assert gate.snapshot().predicates["locked"] is True
    gate.set_value("locked", "false")
    assert gate.snapshot().predicates["locked"] is False
    with pytest.raises(ValueError, match="field not writable"):
        gate.set_value("blocked", "true")


def test_gate_reset_restores_initial_locked():
    mech = _Mech(robots=[])
    _manager(mech)
    gate = _attach_gate(mech, _gate_cfgs(authorized=[], locked=True))
    gate.set_value("locked", "false")
    gate.reset()
    assert gate.snapshot().predicates["locked"] is True


# ---------------------------------------------------------------------------
# pressure_plate
# ---------------------------------------------------------------------------


def test_plate_pressed_within_radius():
    mech = _Mech(robots=[("r", (1.0, 1.0))])
    _manager(mech)
    plate = PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(1.0, 1.0), radius=0.5))
    plate.step(0.0)
    assert plate.snapshot().predicates["pressed"] is True


def test_plate_momentary_releases_when_agent_leaves():
    mech = _Mech(robots=[("r", (1.0, 1.0))])
    _manager(mech)
    plate = PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(1.0, 1.0), radius=0.5, latch=0.0))
    plate.step(0.0)
    assert plate.snapshot().predicates["pressed"] is True
    mech._robots = [("r", (9.0, 9.0))]
    plate.step(1.0)
    assert plate.snapshot().predicates["pressed"] is False


def test_plate_latch_holds_after_contact():
    mech = _Mech(robots=[("r", (1.0, 1.0))])
    _manager(mech)
    plate = PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(1.0, 1.0), radius=0.5, latch=3.0))
    plate.step(0.0)
    mech._robots = [("r", (9.0, 9.0))]
    plate.step(2.0)  # within latch of last contact (0.0)
    assert plate.snapshot().predicates["pressed"] is True
    plate.step(5.0)  # 5s past last contact, latch 3 -> released
    assert plate.snapshot().predicates["pressed"] is False


def test_plate_press_on_regime():
    mech = _Mech(robots=[])
    mgr = _manager(mech)
    sched = _attach_schedule(mech, _schedule_cfgs([{"start": 0.0, "end": 100.0, "value": "v"}], regime="alarm"))
    plate = PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(0.0, 0.0), radius=0.1, press_on="alarm"))
    mgr._instances["sched"] = [sched]
    mgr._instances["p"] = [plate]
    mgr.step(5.0)
    assert plate.snapshot().predicates["pressed"] is True


def test_plate_drives_refreshes_door_via_apply_plate_drives():
    mech = _Mech(robots=[("r", (1.0, 1.0))])
    mgr = _manager(mech)
    mech._door_runtime["target_door"] = _door_runtime("target_door")
    plate = PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(1.0, 1.0), radius=0.5, drives="target_door"))
    mgr._instances["p"] = [plate]
    mgr.step(4.0)  # plate pressed
    mech._door_runtime["target_door"].last_trigger_sim_time = -math.inf
    mgr.apply_plate_drives(9.0)
    assert mech._door_runtime["target_door"].last_trigger_sim_time == pytest.approx(9.0)


def test_plate_writable_forced_hold():
    mech = _Mech(robots=[])
    _manager(mech)
    plate = PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(0.0, 0.0), radius=0.1))
    plate.step(0.0)
    assert plate.snapshot().predicates["pressed"] is False
    plate.set_value("pressed", "true")
    assert plate.snapshot().predicates["pressed"] is True


def test_plate_asserts_regime_while_pressed():
    mech = _Mech(robots=[("r", (0.0, 0.0))])
    _manager(mech)
    plate = PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(0.0, 0.0), radius=0.5, regime="zone_hot"))
    plate.step(0.0)
    assert plate.asserts_regime("zone_hot") is True
    assert plate.asserts_regime("nope") is False


# ---------------------------------------------------------------------------
# occupancy_cap
# ---------------------------------------------------------------------------


_SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]


def test_in_polygon_basic():
    assert _in_polygon(2.0, 2.0, _SQUARE) is True
    assert _in_polygon(5.0, 2.0, _SQUARE) is False


def test_occupancy_counts_agents_inside_polygon():
    mech = _Mech(robots=[("a", (1.0, 1.0)), ("b", (2.0, 2.0)), ("c", (9.0, 9.0))], peds=[("p1", (3.0, 3.0))])
    _manager(mech)
    occ = OccupancyCapSemantics.attach(mech, "lobby", _occupancy_cfgs(cap=2), polygon=_SQUARE)
    snap = occ.snapshot()
    assert snap.continuous["occupancy"] == pytest.approx(3.0)
    assert snap.continuous["cap"] == pytest.approx(2.0)
    assert snap.predicates["over_cap"] is True
    assert set(snap.members) == {"a", "b", "p1"}


def test_occupancy_not_over_cap():
    mech = _Mech(robots=[("a", (1.0, 1.0))])
    _manager(mech)
    occ = OccupancyCapSemantics.attach(mech, "lobby", _occupancy_cfgs(cap=5), polygon=_SQUARE)
    assert occ.snapshot().predicates["over_cap"] is False


def test_occupancy_cap_writable_and_reset():
    mech = _Mech(robots=[("a", (1.0, 1.0)), ("b", (2.0, 2.0))])
    _manager(mech)
    occ = OccupancyCapSemantics.attach(mech, "lobby", _occupancy_cfgs(cap=5), polygon=_SQUARE)
    assert occ.snapshot().predicates["over_cap"] is False
    occ.set_value("cap", "1")
    assert occ.snapshot().continuous["cap"] == pytest.approx(1.0)
    assert occ.snapshot().predicates["over_cap"] is True
    occ.reset()
    assert occ.snapshot().continuous["cap"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# elevator cabin_door outputs and members (M2.C0)
# ---------------------------------------------------------------------------


def _elevator_full_cfgs():
    return [
        _MCfg("state", "arriving_eta"),
        _MCfg("state", "occupants"),
        _MCfg("predicate", "departing"),
        _MCfg("predicate", "in_transit"),
        _MCfg("predicate", "dispatched"),
        _MCfg("predicate", "just_arrived"),
        _MCfg("state", "cabin_door"),
        _MCfg("state", "cabin_door_progress"),
    ]


def test_elevator_cabin_door_reads_synthesized_runtime():
    mech = _Mech()
    mech._elevator_runtime["e"] = _elevator_runtime("e")
    cabin = _door_runtime("e/door")
    cabin.state = _DoorState.OPENING
    cabin.progress = 0.6
    mech._door_runtime["e/door"] = cabin
    inst = ElevatorSemantics.attach(mech, "e", _elevator_full_cfgs())
    inst.step(0.0)
    snap = inst.snapshot()
    assert snap.discrete["cabin_door"] == "opening"
    assert snap.continuous["cabin_door_progress"] == pytest.approx(0.6)


def test_elevator_cabin_door_reset_restores():
    mech = _Mech()
    mech._elevator_runtime["e"] = _elevator_runtime("e")
    cabin = _door_runtime("e/door")
    cabin.state = _DoorState.OPEN
    cabin.progress = 1.0
    cabin.last_applied_progress = 0.9
    mech._door_runtime["e/door"] = cabin
    inst = ElevatorSemantics.attach(mech, "e", _elevator_full_cfgs())
    inst.reset()
    assert cabin.state == _DoorState.CLOSED
    assert cabin.progress == 0.0
    assert cabin.last_applied_progress == -1.0


def test_elevator_members_populated():
    mech = _Mech()
    rt = _elevator_runtime("e")
    rt.dispatched = {"r1"}
    rt.pending_occupants = (("r2", (0.0, 0.0)),)
    rt.just_arrived = {"r3": True}
    mech._elevator_runtime["e"] = rt
    inst = ElevatorSemantics.attach(mech, "e", _elevator_full_cfgs())
    inst.step(0.0)
    assert set(inst.snapshot().members) == {"r1", "r2", "r3"}


def test_door_members_empty():
    mech = _Mech()
    mech._door_runtime["d"] = _door_runtime("d")
    inst = DoorSemantics.attach(mech, "d", [_MCfg("predicate", "open")])
    inst.step(0.0)
    assert inst.snapshot().members == []


# ---------------------------------------------------------------------------
# F2 quantization in _diff
# ---------------------------------------------------------------------------


def _snap(continuous):
    return SemanticEntitySnapshot("e", "k", {}, continuous, {})


def test_diff_continuous_no_emit_within_quantum():
    prev = _snap({"progress": 0.40})
    cur = _snap({"progress": 0.44})
    assert _diff(prev, cur, lambda _f: 0.1) == []


def test_diff_continuous_emits_on_quantum_crossing():
    prev = _snap({"progress": 0.40})
    cur = _snap({"progress": 0.46})
    changes = _diff(prev, cur, lambda _f: 0.1)
    assert len(changes) == 1
    assert changes[0].field == "progress"
    assert changes[0].current == "0.46"


def test_diff_continuous_first_observation_emits():
    cur = _snap({"progress": 0.03})
    changes = _diff(None, cur, lambda _f: 0.1)
    assert len(changes) == 1


def test_diff_quantum_1_0_default_for_occupancy():
    prev = _snap({"occupancy": 2.0})
    cur = _snap({"occupancy": 2.4})
    assert _diff(prev, cur, lambda _f: 1.0) == []
    cur2 = _snap({"occupancy": 2.6})
    assert len(_diff(prev, cur2, lambda _f: 1.0)) == 1


def test_kind_quantum_lookup():
    mech = _Mech()
    mech._door_runtime["d"] = _door_runtime("d")
    door = DoorSemantics.attach(mech, "d", [_MCfg("state", "progress")])
    assert door.quantum("progress") == pytest.approx(0.1)
    assert door.quantum("something_else") == pytest.approx(1.0)


def test_manager_progress_quantization_suppresses_small_moves():
    mech = _Mech()
    rt = _door_runtime("d")
    mech._door_runtime["d"] = rt
    mgr = _manager(mech)
    batches = []
    mgr.set_change_callback(lambda b: batches.append(list(b)))
    mgr.attach("door", "d", [_MCfg("state", "progress")])
    rt.progress = 0.40
    mgr.step(1.0)
    batches.clear()
    rt.progress = 0.44  # below quantum
    mgr.step(2.0)
    assert batches == []
    rt.progress = 0.46  # crosses quantum boundary
    mgr.step(3.0)
    assert len(batches) == 1
    assert batches[0][0].field == "progress"


# ---------------------------------------------------------------------------
# write path validation via SemanticsManager.set_value
# ---------------------------------------------------------------------------


def test_manager_set_value_unknown_entity_returns_false():
    mech = _Mech()
    mgr = _manager(mech)
    assert mgr.set_value("nope", "state", "x") is False


def test_manager_set_value_door_is_read_only():
    mech = _Mech()
    mech._door_runtime["d"] = _door_runtime("d")
    mgr = _manager(mech)
    mgr.attach("door", "d", [_MCfg("state", "state")])
    with pytest.raises(ValueError, match="read-only"):
        mgr.set_value("d", "state", "open")


def test_manager_set_value_elevator_is_read_only():
    mech = _Mech()
    mech._elevator_runtime["e"] = _elevator_runtime("e")
    mech._door_runtime["e/door"] = _door_runtime("e/door")
    mgr = _manager(mech)
    mgr.attach("elevator", "e", [_MCfg("state", "occupants")])
    with pytest.raises(ValueError, match="read-only"):
        mgr.set_value("e", "occupants", "3")


def test_manager_set_value_field_not_writable():
    mech = _Mech(robots=[])
    mgr = _manager(mech)
    mgr._instances["p"] = [PressurePlateSemantics.attach(mech, "p", _plate_cfgs(position=(0.0, 0.0), radius=0.1))]
    with pytest.raises(ValueError, match="field not writable"):
        mgr.set_value("p", "nonexistent", "true")


def test_manager_set_value_malformed():
    mech = _Mech(robots=[])
    mgr = _manager(mech)
    mgr._instances["lobby"] = [OccupancyCapSemantics.attach(mech, "lobby", _occupancy_cfgs(cap=1), polygon=_SQUARE)]
    with pytest.raises(ValueError, match="malformed value"):
        mgr.set_value("lobby", "cap", "not_a_number")


def test_manager_set_value_applies_and_fires_callback():
    mech = _Mech(robots=[])
    mgr = _manager(mech)
    batches = []
    mgr.set_change_callback(lambda b: batches.append(list(b)))
    mech._door_runtime["d0"] = _door_runtime("d0")
    gate = GateSemantics.attach(mech, "d0", _gate_cfgs(authorized=[]))
    mgr._instances["d0"] = [gate]
    ok = mgr.set_value("d0", "locked", "false")
    assert ok is True
    assert gate.snapshot().predicates["locked"] is False
    assert any(c.field == "locked" and c.current == "false" for batch in batches for c in batch)


# ---------------------------------------------------------------------------
# regime bus + hooks
# ---------------------------------------------------------------------------


def test_regime_bus_or_of_scripted_instances():
    mech = _Mech()
    mgr = _manager(mech)
    s1 = _attach_schedule(mech, _schedule_cfgs([{"start": 0.0, "end": 100.0, "value": "v"}], regime="alarm"), entity="s1")
    s2 = _attach_schedule(mech, _schedule_cfgs([{"start": 50.0, "end": 60.0, "value": "v"}], regime="alarm"), entity="s2")
    mgr._instances["s1"] = [s1]
    mgr._instances["s2"] = [s2]
    mgr.step(5.0)  # s1 active, s2 not
    assert mgr.regime("alarm") is True
    assert mgr.regime("other") is False


def test_trigger_allowed_true_with_no_gate():
    mech = _Mech()
    mgr = _manager(mech)
    assert mgr.trigger_allowed("any_door", 0.0) is True


def test_elevator_recalled_registers_and_consults_regime():
    """set_recall (the Elevator.recall_on plumbing) drives elevator_recalled."""
    mech = _Mech()
    mech._elevator_runtime["e"] = _elevator_runtime("e")
    mech._door_runtime["e/door"] = _door_runtime("e/door")
    mgr = _manager(mech)
    mgr.attach("elevator", "e")
    mgr.set_recall("e", "alarm")
    sched = _attach_schedule(mech, _schedule_cfgs([{"start": 0.0, "end": 100.0, "value": "v"}], regime="alarm"), entity="fire")
    mgr._instances["fire"] = [sched]
    assert mgr.elevator_recalled("e", 5.0) is False  # regime not yet stepped true
    mgr.step(5.0)
    assert mgr.regime("alarm") is True
    assert mgr.elevator_recalled("e", 5.0) is True
    assert mgr.elevator_recalled("other", 5.0) is False


def test_elevator_recall_not_registered_without_set_recall():
    mech = _Mech()
    mech._elevator_runtime["e"] = _elevator_runtime("e")
    mech._door_runtime["e/door"] = _door_runtime("e/door")
    mgr = _manager(mech)
    mgr.attach("elevator", "e")
    assert mgr.elevator_recalled("e", 5.0) is False


def test_set_recall_none_clears_registration():
    mech = _Mech()
    mech._elevator_runtime["e"] = _elevator_runtime("e")
    mech._door_runtime["e/door"] = _door_runtime("e/door")
    mgr = _manager(mech)
    mgr.attach("elevator", "e")
    mgr.set_recall("e", "alarm")
    mgr.set_recall("e", None)
    sched = _attach_schedule(mech, _schedule_cfgs([{"start": 0.0, "end": 100.0, "value": "v"}], regime="alarm"), entity="fire")
    mgr._instances["fire"] = [sched]
    mgr.step(5.0)
    assert mgr.elevator_recalled("e", 5.0) is False


def test_fire_alarm_cascade_through_regime_bus():
    """Schedule asserts alarm -> gate unlocks and plate drives the door, all via regime consults."""
    mech = _Mech(robots=[("robot", (0.5, 0.5))])
    mgr = _manager(mech)
    mech._door_runtime["fire_door"] = _door_runtime("fire_door")
    fire = _attach_schedule(mech, _schedule_cfgs([{"start": 12.0, "end": 999.0, "value": "on"}], regime="alarm"), entity="fire_alarm")
    gate = GateSemantics.attach(mech, "fire_door", _gate_cfgs(authorized=[], unlock_on="alarm"))
    plate = PressurePlateSemantics.attach(mech, "fire_door", _plate_cfgs(position=(0.5, 0.5), radius=1.0, drives="fire_door", press_on="alarm"))
    mgr._instances["fire_alarm"] = [fire]
    mgr._instances["fire_door"] = [gate, plate]

    mgr.step(0.0)  # reset t0 to episode-start
    mgr.step(5.0)  # before alarm
    assert mgr.regime("alarm") is False
    assert mgr.trigger_allowed("fire_door", 5.0) is False  # gate locked, robot unauthorized

    mgr.step(15.0)  # alarm active
    assert mgr.regime("alarm") is True
    assert mgr.trigger_allowed("fire_door", 15.0) is True  # unlock_on forces unlocked
    mech._door_runtime["fire_door"].last_trigger_sim_time = -math.inf
    mgr.apply_plate_drives(15.0)
    assert mech._door_runtime["fire_door"].last_trigger_sim_time == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# support matrix (M2 kind tables)
# ---------------------------------------------------------------------------


def test_support_matrix_new_kinds():
    assert SignalSemantics.SUPPORTED_SIMS == frozenset({"gazebo", "isaac", "dummy"})
    assert ScheduleSemantics.SUPPORTED_SIMS == frozenset({"gazebo", "isaac", "dummy"})
    assert GateSemantics.SUPPORTED_SIMS == frozenset({"gazebo", "isaac"})
    assert PressurePlateSemantics.SUPPORTED_SIMS == frozenset({"gazebo", "isaac"})
    assert OccupancyCapSemantics.SUPPORTED_SIMS == frozenset({"gazebo", "isaac"})


def test_signal_schedule_attach_on_dummy():
    mech = _Mech()
    mgr = _manager(mech, sim="dummy")
    mgr.attach("signal", "sig", _signal_cfgs([{"name": "a", "duration": 1.0}]))
    mgr.attach("schedule", "sc", _schedule_cfgs([]))
    assert len(mgr.snapshot()) == 2


def test_position_kinds_skip_dummy():
    mech = _Mech()
    mgr = _manager(mech, sim="dummy")
    mgr.attach("occupancy_cap", "lobby", _occupancy_cfgs(cap=1), polygon=_SQUARE)
    assert mgr.snapshot() == []
    assert any("unsupported" in msg for msg in mech.node.get_logger().infos)


def test_writable_sets_per_kind():
    assert SignalSemantics.WRITABLE == frozenset({"state"})
    assert ScheduleSemantics.WRITABLE == frozenset({"state", "active"})
    assert GateSemantics.WRITABLE == frozenset({"locked"})
    assert PressurePlateSemantics.WRITABLE == frozenset({"pressed"})
    assert OccupancyCapSemantics.WRITABLE == frozenset({"cap"})
    assert DoorSemantics.WRITABLE == frozenset()
    assert ElevatorSemantics.WRITABLE == frozenset()
