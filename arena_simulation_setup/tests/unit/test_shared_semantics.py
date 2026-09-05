from __future__ import annotations

import pytest

from arena_simulation_setup.shared.semantics import SemanticCfg, parse_semantics
from arena_simulation_setup.utils.cattrs import converter

# ---------------------------------------------------------------------------
# SemanticCfg.parse / serialize
# ---------------------------------------------------------------------------


def test_parse_state_primitive():
    cfg = SemanticCfg.parse({'state': 'progress', 'value': 0.4})
    assert cfg.role == 'state'
    assert cfg.name == 'progress'
    assert cfg.value == 0.4


def test_parse_predicate_primitive():
    cfg = SemanticCfg.parse({'predicate': 'quiet', 'value': True})
    assert cfg.role == 'predicate'
    assert cfg.name == 'quiet'
    assert cfg.value is True


def test_parse_defaults():
    cfg = SemanticCfg.parse({'state': 'state'})
    assert cfg.value is None
    assert cfg.binding == 'kinematic'
    assert cfg.trigger == 'proximity'
    assert cfg.distance == -1.0


def test_parse_requires_state_or_predicate_key():
    with pytest.raises(ValueError):
        SemanticCfg.parse({'value': 1.0})


def test_parse_rejects_unknown_keys():
    with pytest.raises(ValueError):
        SemanticCfg.parse({'state': 'state', 'bogus': 1})


def test_parse_binding_joint_rejected():
    with pytest.raises(ValueError):
        SemanticCfg.parse({'state': 'state', 'binding': 'joint'})


def test_parse_trigger_contact_rejected():
    with pytest.raises(ValueError):
        SemanticCfg.parse({'predicate': 'open', 'trigger': 'contact'})


def test_serialize_omits_defaults():
    cfg = SemanticCfg.parse({'state': 'state'})
    assert cfg.serialize() == {'state': 'state'}


def test_serialize_includes_non_default_distance():
    cfg = SemanticCfg.parse({'predicate': 'quiet', 'value': True, 'distance': 2.0})
    assert cfg.serialize() == {'predicate': 'quiet', 'value': True, 'distance': 2.0}


# ---------------------------------------------------------------------------
# parse_semantics: presets and passthrough
# ---------------------------------------------------------------------------


def test_parse_semantics_door_and_elevator_presets_removed():
    with pytest.raises(ValueError):
        parse_semantics([{'preset': 'door'}])
    with pytest.raises(ValueError):
        parse_semantics([{'preset': 'elevator'}])


def test_parse_semantics_unknown_preset_rejected():
    with pytest.raises(ValueError):
        parse_semantics([{'preset': 'bogus'}])


def test_parse_semantics_primitive_passthrough():
    cfgs = parse_semantics([{'state': 'max_speed', 'value': 1.5}, {'predicate': 'quiet', 'value': True}])
    assert len(cfgs) == 2
    assert cfgs[0].name == 'max_speed'
    assert cfgs[1].name == 'quiet'


def test_parse_semantics_idempotent_on_existing_cfgs():
    cfgs = parse_semantics([{'state': 'state'}])
    assert parse_semantics(cfgs) == cfgs


def test_parse_semantics_empty_list():
    assert parse_semantics([]) == []


# ---------------------------------------------------------------------------
# cattrs structure/unstructure of list[SemanticCfg] (as used on Door/Elevator/Zone)
# ---------------------------------------------------------------------------


def test_structure_list_of_semanticcfg_expands_preset():
    cfgs = converter.structure([{'preset': 'gate'}], list[SemanticCfg])
    assert len(cfgs) == 2


def test_structure_list_of_semanticcfg_rejects_joint_binding():
    with pytest.raises(Exception):
        converter.structure([{'state': 'state', 'binding': 'joint'}], list[SemanticCfg])


def test_structure_field_form_rejects_joint_binding():
    with pytest.raises(ValueError):
        converter.structure({'role': 'state', 'name': 'state', 'binding': 'joint'}, SemanticCfg)


def test_structure_field_form_rejects_contact_trigger():
    with pytest.raises(ValueError):
        converter.structure({'role': 'predicate', 'name': 'open', 'trigger': 'contact'}, SemanticCfg)


def test_unstructure_semanticcfg_round_trip():
    cfgs = parse_semantics([{'preset': 'gate', 'distance': 2.0}])
    raw = converter.unstructure(cfgs)
    restructured = converter.structure(raw, list[SemanticCfg])
    assert restructured == cfgs


# ---------------------------------------------------------------------------
# M2: params field
# ---------------------------------------------------------------------------


def test_parse_params_defaults_empty():
    cfg = SemanticCfg.parse({'state': 'state'})
    assert cfg.params == {}


def test_parse_params_carried_through():
    cfg = SemanticCfg.parse({'state': 'state', 'params': {'phases': [{'name': 'go', 'duration': 5.0}]}})
    assert cfg.params == {'phases': [{'name': 'go', 'duration': 5.0}]}


def test_serialize_omits_empty_params():
    cfg = SemanticCfg.parse({'state': 'state'})
    assert 'params' not in cfg.serialize()


def test_serialize_includes_non_empty_params():
    cfg = SemanticCfg.parse({'predicate': 'stop', 'params': {'regime': 'alarm'}})
    assert cfg.serialize() == {'predicate': 'stop', 'params': {'regime': 'alarm'}}


def test_params_round_trip_through_cattrs():
    cfgs = parse_semantics([{'predicate': 'pressed', 'params': {'radius': 1.5, 'drives': 'north_fire_door'}}])
    raw = converter.unstructure(cfgs)
    restructured = converter.structure(raw, list[SemanticCfg])
    assert restructured == cfgs


# ---------------------------------------------------------------------------
# M2: new presets
# ---------------------------------------------------------------------------


def test_parse_semantics_signal_preset_expands():
    cfgs = parse_semantics([{'preset': 'signal'}])
    assert [(c.role, c.name) for c in cfgs] == [
        ('state', 'state'),
        ('state', 'phase_remaining'),
        ('predicate', 'stop'),
    ]


def test_parse_semantics_schedule_preset_expands():
    cfgs = parse_semantics([{'preset': 'schedule'}])
    assert [(c.role, c.name) for c in cfgs] == [
        ('state', 'state'),
        ('predicate', 'active'),
        ('state', 'window_remaining'),
    ]


def test_parse_semantics_gate_preset_expands():
    cfgs = parse_semantics([{'preset': 'gate'}])
    assert [(c.role, c.name) for c in cfgs] == [
        ('predicate', 'locked'),
        ('predicate', 'blocked'),
    ]


def test_parse_semantics_pressure_plate_preset_expands():
    cfgs = parse_semantics([{'preset': 'pressure_plate'}])
    assert [(c.role, c.name) for c in cfgs] == [('predicate', 'pressed')]


def test_parse_semantics_occupancy_cap_preset_expands():
    cfgs = parse_semantics([{'preset': 'occupancy_cap'}])
    assert [(c.role, c.name) for c in cfgs] == [
        ('state', 'occupancy'),
        ('state', 'cap'),
        ('predicate', 'over_cap'),
    ]


def test_parse_semantics_sound_preset_expands():
    cfgs = parse_semantics([{'preset': 'sound'}])
    assert [(c.role, c.name) for c in cfgs] == [
        ('predicate', 'sounding'),
        ('state', 'volume_db'),
    ]


def test_parse_semantics_elevator_full_preset_removed():
    with pytest.raises(ValueError):
        parse_semantics([{'preset': 'elevator_full'}])


def test_parse_semantics_preset_params_applied_to_every_primitive():
    cfgs = parse_semantics([{'preset': 'schedule', 'params': {'windows': [], 'regime': 'alarm'}}])
    assert all(c.params == {'windows': [], 'regime': 'alarm'} for c in cfgs)


def test_parse_semantics_preset_param_named_after_primitive_becomes_its_value():
    cfgs = parse_semantics([{'preset': 'sound', 'params': {'sound_on': 'alarm', 'volume_db': 88.0}}])
    by_name = {c.name: c for c in cfgs}
    assert by_name['volume_db'].value == 88.0
    assert by_name['sounding'].value is None
    assert all(c.params == {'sound_on': 'alarm'} for c in cfgs)


def test_parse_semantics_gate_locked_via_params():
    cfgs = parse_semantics([{'preset': 'gate', 'params': {'authorized': [], 'locked': False}}])
    by_name = {c.name: c for c in cfgs}
    assert by_name['locked'].value is False
    assert by_name['blocked'].value is None
    assert by_name['blocked'].params == {'authorized': []}


def test_parse_semantics_preset_value_rejected_on_multi_primitive_preset():
    with pytest.raises(ValueError):
        parse_semantics([{'preset': 'sound', 'value': 88.0}])


def test_parse_semantics_preset_value_allowed_on_single_primitive_preset():
    cfgs = parse_semantics([{'preset': 'pressure_plate', 'value': True, 'params': {'position': [0.0, 0.0]}}])
    assert len(cfgs) == 1
    assert cfgs[0].value is True


def test_parse_semantics_pressure_plate_params_and_distance_override():
    cfgs = parse_semantics([
        {'preset': 'pressure_plate', 'distance': 1.0, 'params': {'position': [4.0, 2.0], 'press_on': 'alarm'}},
    ])
    assert len(cfgs) == 1
    assert cfgs[0].distance == 1.0
    assert cfgs[0].params == {'position': [4.0, 2.0], 'press_on': 'alarm'}
