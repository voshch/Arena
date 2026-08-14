from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.tree.World.World import LevelDescription
from arena_simulation_setup.utils.cattrs import converter

_WORLD_YAML = Path(__file__).resolve().parents[2] / 'worlds' / 'three_storied_residential' / '1' / 'world.yaml'


@pytest.fixture(scope='module')
def _raw_world_data() -> dict:
    if not _WORLD_YAML.exists():
        pytest.skip(f'fixture world not found: {_WORLD_YAML}')
    with open(_WORLD_YAML, encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_existing_world_loads_unchanged(_raw_world_data):
    """Zones stay unannotated; doors/elevators carry whatever `semantics:` the fixture declares."""
    level_desc = converter.structure(_raw_world_data, LevelDescription)
    for raw_zone, zone in zip(_raw_world_data['zones'], level_desc.zones):
        assert zone.semantics == []
        for raw_door, door in zip(raw_zone.get('doors', []), zone.doors):
            if 'semantics' in raw_door:
                assert door.semantics != []
            else:
                assert door.semantics == []
        for raw_elevator, elevator in zip(raw_zone.get('elevators', []), zone.elevators):
            if 'semantics' in raw_elevator:
                assert elevator.semantics != []
            else:
                assert elevator.semantics == []


def test_existing_world_semantics_omitted_on_unstructure(_raw_world_data):
    """Empty `semantics:` is omitted for doors/elevators, present when the fixture declares it."""
    level_desc = converter.structure(_raw_world_data, LevelDescription)
    unstructured = converter.unstructure(level_desc)
    for raw_zone, zone in zip(_raw_world_data['zones'], unstructured['zones']):
        assert zone['semantics'] == []
        for raw_door, door in zip(raw_zone.get('doors', []), zone.get('doors', [])):
            if 'semantics' in raw_door:
                assert door['semantics']
            else:
                assert 'semantics' not in door
        for raw_elevator, elevator in zip(raw_zone.get('elevators', []), zone.get('elevators', [])):
            if 'semantics' in raw_elevator:
                assert elevator['semantics']
            else:
                assert 'semantics' not in elevator


def test_existing_world_reparse_is_stable(_raw_world_data):
    level_desc = converter.structure(_raw_world_data, LevelDescription)
    unstructured = converter.unstructure(level_desc)
    reparsed = converter.structure(unstructured, LevelDescription)
    assert [zone.name for zone in reparsed.zones] == [zone.name for zone in level_desc.zones]
    assert [door.name for door in reparsed.all_doors] == [door.name for door in level_desc.all_doors]
    assert [elevator.name for elevator in reparsed.all_elevators] == [elevator.name for elevator in level_desc.all_elevators]
    assert [door.semantics for door in reparsed.all_doors] == [door.semantics for door in level_desc.all_doors]
    assert [elevator.semantics for elevator in reparsed.all_elevators] == [elevator.semantics for elevator in level_desc.all_elevators]
    for zone in reparsed.zones:
        assert zone.semantics == []
