"""Every object and pedestrian model a shipped world or its scenarios use resolves on a clean machine, from the world's
own assets or the default asset store source. An object model the store no longer has under that name is dropped at
runtime, so the crowd walks through it; a pedestrian model falls back to the default one. Checked offline against the
cached bucket index, skipped without one. Names the store lacks and whose successor is not proven are listed with the
reason and stay as authored."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest
from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.World import WorldView, world_assets

from arena_assets.impl.fetch import DEFAULT_SOURCE, assets_root

WORLDS = Path(__file__).resolve().parents[2] / 'worlds'

_CLUTTER = 'PaperTray PencilCup RingBinder Smartphone StandForPapers'
KNOWN_UNRESOLVED = {
    'Office/Object/SM_ChairOffice': "NVIDIA office prop, the store's Office/Chair_Office is the different prop SM_ChairOffice_A",
    'Office/Object/SM_TableWorkingDouble': "NVIDIA office prop, the store's Office/Table_Work_Security is the different prop SM_TableWorkSecurity",
    'Office/Object/SM_MarkerBoard': "NVIDIA office prop, the store's Office/SM_Board is a different prop",
    **{f'Office/Object/SM_{name}': 'NVIDIA desk prop with no copy in the store' for name in _CLUTTER.split()},
    'Common/Object/vending_machine': "legacy Gazebo model, the store's drinks machines are different assets",
    'Common/Object/shelf': 'legacy Gazebo model, no store asset has its data',
    'Common/Object/chair': 'legacy Gazebo model, no store asset has its data',
    'Common/Human/female_adult_business_': 'truncated pedestrian name, the intended model is not recorded',
    'Common/Human/male_adult_medical_': 'truncated pedestrian name, the intended model is not recorded',
}


@pytest.fixture(scope='module')
def stored() -> set[str]:
    """Every directory of the default source that holds a stored file, as the cached bucket index lists them."""
    bucket, _, folder = DEFAULT_SOURCE.partition('/')
    root = assets_root()
    index = root / '.index' / f'{bucket}.json' if root is not None else None
    if index is None or not index.is_file():
        pytest.skip(f'no cached {bucket} bucket index, fetch one with `arena asset ls object --network`')
    prefix = f'{folder}/'
    names = (entry['name'].removeprefix(prefix) for entry in json.loads(index.read_bytes())['objects'] if entry['name'].startswith(prefix))
    return {str(parent) for name in names for parent in PurePosixPath(name).parents}


@pytest.mark.parametrize('world', sorted(p.name for p in WORLDS.iterdir() if p.is_dir() and not p.name.startswith('.')))
def test_every_object_and_pedestrian_model_of_a_shipped_world_resolves(world: str, stored: set[str]) -> None:
    view = WorldView(WORLDS / world)
    missing = []
    with world_assets(view):
        for identifier in view.identifiers(strict=True):
            if not isinstance(identifier, (ObjectIdentifier, HumanIdentifier)) or str(identifier.relpath()) in stored or identifier.shortname in KNOWN_UNRESOLVED:
                continue
            try:
                identifier.resolve_source_sync()
            except FileNotFoundError:
                missing.append(identifier.shortname)
    assert not missing, f'{world} uses models the asset store does not have: {", ".join(sorted(missing))}'


@pytest.mark.parametrize('name', sorted(KNOWN_UNRESOLVED))
def test_a_known_unresolved_name_is_still_missing_from_the_store(name: str, stored: set[str]) -> None:
    assert name not in stored, f'{name} is in the asset store again (listed as: {KNOWN_UNRESOLVED[name]}), drop it from KNOWN_UNRESOLVED'
