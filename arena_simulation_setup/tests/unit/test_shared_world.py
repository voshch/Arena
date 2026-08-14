from __future__ import annotations

import pytest
from arena_simulation_setup.shared.semantics import SemanticCfg
from arena_simulation_setup.shared.world import Door, Elevator, Floor
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Position

# ---------------------------------------------------------------------------
# Elevator
# ---------------------------------------------------------------------------


def _elev_pos() -> Position:
    return Position(0.0, 0.0, 0.0)


def test_elevator_material_converter():
    from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
    e = Elevator(name="elev", position=_elev_pos())
    assert isinstance(e.material, MaterialIdentifier)


def test_elevator_position_converter():
    e = Elevator(name="elev", position=Position(1.0, 2.0, 3.0))
    assert e.position.x == pytest.approx(1.0)


def test_elevator_cabin_corners_centered():
    e = Elevator(name="elev", position=Position(5.0, -3.0, 0.0), size=[4.0, 2.0, 2.5])
    corners = e.cabin_corners()
    assert len(corners) == 4
    xs = sorted({c.x for c in corners})
    ys = sorted({c.y for c in corners})
    assert xs == pytest.approx([3.0, 7.0])  # cx +- hw
    assert ys == pytest.approx([-4.0, -2.0])  # cy +- hh


# ---------------------------------------------------------------------------
# Door
# ---------------------------------------------------------------------------


def test_door_corners_count():
    d = Door(
        name="door",
        start=Position(0.0, 0.0, 0.0),
        end=Position(1.0, 0.0, 0.0),
        width=0.2,
    )
    assert len(d.corners) == 4


def test_door_corners_perpendicular_offset():
    d = Door(
        name="door",
        start=Position(0.0, 0.0, 0.0),
        end=Position(1.0, 0.0, 0.0),
        width=0.2,
    )
    corners = d.corners
    # y coordinates should be +/- width/2 = +/-0.1
    ys = sorted({c.y for c in corners})
    assert ys[0] == pytest.approx(-0.1, abs=1e-6)
    assert ys[1] == pytest.approx(0.1, abs=1e-6)


def test_door_corners_non_axis_aligned():
    d = Door(
        name="door",
        start=Position(0.0, 0.0, 0.0),
        end=Position(0.0, 1.0, 0.0),
        width=0.2,
    )
    corners = d.corners
    assert len(corners) == 4
    # perpendicular to (0,1,0) is (-1,0,0) so x should be +/-0.1
    xs = sorted({c.x for c in corners})
    assert xs[0] == pytest.approx(-0.1, abs=1e-6)
    assert xs[1] == pytest.approx(0.1, abs=1e-6)


def test_door_zero_width_corners():
    d = Door(
        name="door",
        start=Position(0.0, 0.0, 0.0),
        end=Position(1.0, 0.0, 0.0),
        width=0.0,
    )
    corners = d.corners
    assert len(corners) == 4


def test_door_activation_distance_scalar_broadcast():
    d = Door(
        name="door",
        start=Position(0.0, 0.0, 0.0),
        end=Position(1.0, 0.0, 0.0),
        activation_distance=1.5,
    )
    assert d.activation_distance == (1.5, 1.5)


def test_door_activation_distance_tuple():
    d = Door(
        name="door",
        start=Position(0.0, 0.0, 0.0),
        end=Position(1.0, 0.0, 0.0),
        activation_distance=(1.5, 0.0),
    )
    assert d.activation_distance == (1.5, 0.0)


# ---------------------------------------------------------------------------
# Door/Elevator semantics
# ---------------------------------------------------------------------------


def test_door_semantics_default_empty():
    d = Door(name="door", start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0))
    assert d.semantics == []


def test_door_semantics_empty_omitted_on_serialize():
    d = Door(name="door", start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0))
    assert 'semantics' not in d.serialize()


def test_door_semantics_preset_round_trip():
    raw = {
        'name': 'door_edge_1_1',
        'start': {'x': 4.0, 'y': 1.55, 'z': 0.0},
        'end': {'x': 4.0, 'y': 2.45, 'z': 0.0},
        'semantics': [{'preset': 'gate', 'distance': 2.0}],
    }
    d = converter.structure(raw, Door)
    assert [c.name for c in d.semantics] == ['locked', 'blocked']
    unstructured = converter.unstructure(d)
    d2 = converter.structure(unstructured, Door)
    assert d2.semantics == d.semantics


def test_door_semantics_binding_joint_raises():
    raw = {
        'name': 'door',
        'start': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'end': {'x': 1.0, 'y': 0.0, 'z': 0.0},
        'semantics': [{'state': 'state', 'binding': 'joint'}],
    }
    with pytest.raises(Exception):
        converter.structure(raw, Door)


def test_door_semantics_trigger_contact_raises():
    raw = {
        'name': 'door',
        'start': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'end': {'x': 1.0, 'y': 0.0, 'z': 0.0},
        'semantics': [{'predicate': 'open', 'trigger': 'contact'}],
    }
    with pytest.raises(Exception):
        converter.structure(raw, Door)


def test_elevator_semantics_default_empty():
    e = Elevator(name="elev", position=_elev_pos())
    assert e.semantics == []


def test_elevator_semantics_empty_omitted_on_serialize():
    e = Elevator(name="elev", position=_elev_pos())
    assert 'semantics' not in e.serialize()


def test_elevator_semantics_preset_round_trip():
    raw = {
        'name': '1_elevator',
        'position': {'x': 5.0, 'y': 0.0, 'z': 0.0},
        'destination': '2.2_elevator',
        'semantics': [{'preset': 'pressure_plate', 'params': {'position': [5.0, 0.0]}}],
    }
    e = converter.structure(raw, Elevator)
    assert [c.name for c in e.semantics] == ['pressed']
    unstructured = converter.unstructure(e)
    e2 = converter.structure(unstructured, Elevator)
    assert e2.semantics == e.semantics


def test_elevator_recall_on_default_none():
    e = Elevator(name="elev", position=_elev_pos())
    assert e.recall_on is None


def test_elevator_recall_on_first_class_field():
    e = Elevator(name="elev", position=_elev_pos(), recall_on="alarm")
    assert e.recall_on == "alarm"


def test_door_semantics_evolve_preserves_semantics():
    import attrs

    d = Door(
        name="door",
        start=Position(0.0, 0.0, 0.0),
        end=Position(1.0, 0.0, 0.0),
        semantics=[SemanticCfg.parse({'state': 'state'})],
    )
    d2 = attrs.evolve(d, width=0.5)
    assert d2.semantics == d.semantics


# ---------------------------------------------------------------------------
# Floor
# ---------------------------------------------------------------------------


def test_floor_position_converter():
    f = Floor(name="floor", pos=Position(5.0, 6.0, 0.0))
    assert f.pos.x == pytest.approx(5.0)
    assert f.pos.y == pytest.approx(6.0)


def test_floor_material_is_identifier():
    from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
    f = Floor(name="floor", pos=Position(0.0, 0.0, 0.0))
    assert isinstance(f.material, MaterialIdentifier)
