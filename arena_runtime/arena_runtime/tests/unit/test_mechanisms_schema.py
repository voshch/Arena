"""Mechanism schema parsing and realization."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_runtime.env.realizer")

from arena_simulation_setup.shared import Door, Elevator, Signal  # noqa: E402
from arena_simulation_setup.shared.semantics import parse_semantics  # noqa: E402
from arena_simulation_setup.tree.World.legacy import upgrade  # noqa: E402
from arena_simulation_setup.utils.geometry import Position  # noqa: E402
from arena_runtime.env.environment_manager import _attachments, _route_semantics  # noqa: E402
from arena_runtime.env.realizer import Realizer  # noqa: E402


@pytest.fixture()
def realizer() -> Realizer:
    r = Realizer(Realizer._Configuration(x=100.0, y=50.0, prefix="env3"))
    r.reset_level_origins({"0": (0.0, 0.0), "1": (0.0, 20.0)})
    r.reset_elevator_levels({"up": "1", "side": "0"})
    return r


def _elevator(destinations: list[str], name: str = "hub") -> Elevator:
    return Elevator(name=name, position=Position(0.0, 0.0), destinations=destinations)


# ---------------------------------------------------------------------------
# multi-destination realization
# ---------------------------------------------------------------------------


def test_single_destination_unchanged(realizer):
    assert realizer.realize(_elevator(["up"]), "0").destinations == ["env3/up/1"]


def test_multi_destination_realized_per_target(realizer):
    realized = realizer.realize(_elevator(["up", "side"]), "0")
    assert realized.destinations == ["env3/up/1", "env3/side/0"]


def test_destination_takes_the_level_of_its_cabin(realizer):
    assert realizer.realize(_elevator(["side"]), "1").destinations == ["env3/side/0"]
    assert Realizer().realize(_elevator(["side"])).destinations == ["side"]


def test_empty_destination_stays_empty(realizer):
    assert realizer.realize(_elevator([]), "0").destinations == []


# ---------------------------------------------------------------------------
# pressure plate params follow the realizer prefix / origin
# ---------------------------------------------------------------------------


def _plate_door(drives: str) -> Door:
    return Door(
        name="host",
        start=Position(0.0, 0.0),
        end=Position(1.0, 0.0),
        semantics=[{"preset": "pressure_plate", "params": {"position": [2.0, 3.0], "radius": 0.5, "drives": drives}}],
    )


def _attached_plate(realizer, host, host_kind, level_id):
    realized = realizer.realize(host, level_id)
    ((kind, (cfg,)),) = [(k, c) for k, c in _attachments(realizer, _route_semantics(realized.semantics, host_kind), level_id) if k == "pressure_plate"]
    return realized, cfg


def test_plate_drives_and_position_are_realized(realizer):
    door = _plate_door("gate_a")
    realized, cfg = _attached_plate(realizer, door, "door", "1")
    assert cfg.params["drives"] == "env3/gate_a/1"
    assert cfg.params["position"] == [102.0, 73.0]
    assert realized.name == "env3/host/1"
    assert door.semantics[0].params["drives"] == "gate_a"
    assert _attached_plate(realizer, door, "door", "1")[1].params["drives"] == "env3/gate_a/1"


def test_plate_drives_level_qualified(realizer):
    _realized, cfg = _attached_plate(realizer, _plate_door("0.gate_b"), "door", "1")
    assert cfg.params["drives"] == "env3/gate_b/0"


def test_plate_drives_on_elevator_host(realizer):
    elevator = Elevator(
        name="lift",
        position=Position(0.0, 0.0),
        semantics=[{"preset": "pressure_plate", "params": {"position": [0.0, 0.0], "radius": 0.5, "drives": "gate_c"}}],
    )
    _realized, cfg = _attached_plate(realizer, elevator, "elevator", "0")
    assert cfg.params["drives"] == "env3/gate_c/0"


def test_gate_params_pass_through(realizer):
    door = Door(name="g", start=Position(0.0, 0.0), end=Position(1.0, 0.0), semantics=[{"preset": "gate", "params": {"authorized": ["robot1"]}}])
    realized = realizer.realize(door, "0")
    assert [cfg.params for cfg in realized.semantics] == [cfg.params for cfg in door.semantics]


# ---------------------------------------------------------------------------
# signal stop-line params
# ---------------------------------------------------------------------------


def test_signal_position_radius_params():
    signal = Signal(name="light", semantics=parse_semantics([{"preset": "signal", "params": {"phases": [{"name": "go", "duration": 5.0}], "position": [1.0, 2.0], "radius": 1.5}}]))
    assert signal.stop_position == (1.0, 2.0)
    assert signal.stop_radius == 1.5
    plain = Signal(name="light2", semantics=parse_semantics([{"preset": "signal", "params": {"phases": []}}]))
    assert plain.stop_position is None
    assert plain.stop_radius is None


@pytest.mark.parametrize("params", [{"position": [1.0]}, {"position": "here"}, {"radius": 0.0}])
def test_signal_param_validation(params):
    with pytest.raises(ValueError):
        Signal(name="bad", semantics=parse_semantics([{"preset": "signal", "params": {"phases": [], **params}}]))


# ---------------------------------------------------------------------------
# format 1 destinations: several on the same level, bare names
# ---------------------------------------------------------------------------


def _legacy_hub(destination: str) -> list[str]:
    zone = {"name": "ground", "corners": [[0, 0], [6, 0], [6, 6], [0, 6]], "elevators": [{"name": "hub", "position": [2, 2], "destination": destination}]}
    data, _, _ = upgrade({"0": {"zones": [zone]}})
    return data["levels"][0]["elevators"][0]["destinations"]


def test_legacy_destinations_allow_several_per_level():
    assert _legacy_hub("1.up_a,1.up_b,0.down") == ["up_a", "up_b", "down"]


def test_legacy_destination_level_prefix_is_dropped_and_bare_names_kept():
    assert _legacy_hub("1.up") == ["up"]
    assert _legacy_hub("side") == ["side"]


# ---------------------------------------------------------------------------
# door activation_distance: scalar or pair through the yaml/cattrs path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [(2.0, (2.0, 2.0)), (2, (2.0, 2.0)), ([1.0, 2.5], (1.0, 2.5))])
def test_door_activation_distance_structures_scalar_and_pair(value, expected):
    from arena_simulation_setup.utils.cattrs import converter

    door = converter.structure({"name": "d", "span": [[0.0, 0.0], [1.0, 0.0]], "activation_distance": value}, Door)
    assert door.activation_distance == expected


def test_elevator_call_mode_structures():
    from arena_simulation_setup.utils.cattrs import converter

    assert converter.structure({"name": "e", "position": [0.0, 0.0]}, Elevator).call_mode == "level"
    assert converter.structure({"name": "e", "position": [0.0, 0.0], "call_mode": "edge"}, Elevator).call_mode == "edge"
