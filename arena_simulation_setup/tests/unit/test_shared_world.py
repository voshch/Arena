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


def test_elevator_boarding_radius_fits_the_narrow_side_behind_the_door():
    e = Elevator(name="elev", position=Position(5.0, -3.0, 0.0), size=[4.0, 2.0, 2.5])
    assert e.boarding_radius(0.3) == pytest.approx(1.0 - 0.05 - 0.025 - 0.3)
    assert e.boarding_radius(0.5) < e.boarding_radius(0.3)


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


@pytest.mark.parametrize(
    "entity",
    [
        Door(name="door", start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0)),
        Elevator(name="elev", position=Position(0.0, 0.0, 0.0)),
    ],
    ids=["door", "elevator"],
)
def test_semantics_default_empty_and_omitted_on_serialize(entity):
    assert entity.semantics == []
    assert 'semantics' not in entity.serialize()


def test_door_semantics_preset_round_trip():
    raw = {
        'name': 'door_edge_1_1',
        'span': [[4.0, 1.55], [4.0, 2.45]],
        'semantics': [{'preset': 'gate', 'distance': 2.0}],
    }
    d = converter.structure(raw, Door)
    assert [c.name for c in d.semantics] == ['locked', 'blocked']
    unstructured = converter.unstructure(d)
    d2 = converter.structure(unstructured, Door)
    assert d2.semantics == d.semantics


@pytest.mark.parametrize(
    "semantic",
    [{'state': 'state', 'binding': 'joint'}, {'predicate': 'open', 'trigger': 'contact'}],
    ids=["binding_joint", "trigger_contact"],
)
def test_door_semantics_reject_what_a_door_cannot_bind(semantic):
    with pytest.raises(Exception):
        converter.structure({'name': 'door', 'span': [[0.0, 0.0], [1.0, 0.0]], 'semantics': [semantic]}, Door)


def test_elevator_semantics_preset_round_trip():
    raw = {
        'name': '1_elevator',
        'position': {'x': 5.0, 'y': 0.0, 'z': 0.0},
        'destinations': ['2_elevator'],
        'semantics': [{'preset': 'pressure_plate', 'params': {'position': [5.0, 0.0]}}],
    }
    e = converter.structure(raw, Elevator)
    assert [c.name for c in e.semantics] == ['pressed']
    unstructured = converter.unstructure(e)
    e2 = converter.structure(unstructured, Elevator)
    assert e2.semantics == e.semantics
    assert (e2.destinations, e2.destination) == (['2_elevator'], '2_elevator')


def test_elevator_recall_on_is_optional():
    assert Elevator(name="elev", position=_elev_pos()).recall_on is None
    assert Elevator(name="elev", position=_elev_pos(), recall_on="alarm").recall_on == "alarm"


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
