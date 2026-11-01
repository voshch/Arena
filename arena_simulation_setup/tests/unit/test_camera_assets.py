from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.tree import DynamicPaths
from arena_simulation_setup.tree.assets import Camera
from arena_simulation_setup.tree.assets.Camera import CameraIdentifier, candidates, collect, find_spec, local_spec, pick_camera, step_names
from arena_simulation_setup.tree.World import point_scenario_assets
from arena_simulation_setup.tree.World.view import point_world_assets


def _spec(tag: str, *steps: str) -> dict:
    return {'params': {'tag': tag}, 'timeline': [{step: {}} for step in steps]}


def _asset(root: Path, name: str, spec: dict) -> Path:
    path = root / 'Common' / 'Camera' / name / f'{name}.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(spec), encoding='utf-8')
    return path


def _scenario(directory: Path, **specs: dict) -> Path:
    (directory / 'cameras').mkdir(parents=True)
    for name, spec in specs.items():
        (directory / 'cameras' / f'{name}.yaml').write_text(yaml.safe_dump(spec), encoding='utf-8')
    return directory


def _tag(name: str) -> str:
    spec = asyncio.run(local_spec(name))
    assert spec is not None
    return spec['params']['tag']


def _pick(request: str) -> tuple[str | None, str | None]:
    identifier, warning = asyncio.run(pick_camera(request))
    return None if identifier is None else identifier.name, warning


@pytest.fixture(autouse=True)
def tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Empty local and imported roots, no world, no scenario and no buckets."""
    monkeypatch.setattr(Camera, '_NETWORK', [])
    monkeypatch.setattr(Camera, '_LISTED', {})
    monkeypatch.setattr(DynamicPaths.ARENA, '_assigned', tmp_path / 'local')
    monkeypatch.setattr(DynamicPaths.IMPORTED, '_assigned', tmp_path / 'imported')
    previous = DynamicPaths.WORLD.path, list(DynamicPaths.LINKED.paths)
    point_world_assets(None, [])
    yield tmp_path
    point_world_assets(*previous)
    point_scenario_assets(None)


def test_tiers_resolve_scenario_then_world_then_linked_then_local(tiers: Path) -> None:
    world, linked = tiers / 'world', tiers / 'linked'
    _asset(tiers / 'local', 'wide', _spec('local'))
    assert _tag('wide') == 'local'
    _asset(linked / 'assets', 'wide', _spec('linked'))
    point_world_assets(world, [linked])
    assert _tag('wide') == 'linked'
    _asset(world / 'assets', 'wide', _spec('world'))
    point_world_assets(world, [linked])
    assert _tag('wide') == 'world'
    point_scenario_assets(_scenario(tiers / 'scenario', wide=_spec('scenario')))
    assert _tag('wide') == 'scenario'


def test_local_shadows_the_bundled_asset(tiers: Path) -> None:
    bundled = asyncio.run(local_spec('establishing'))
    assert bundled is not None
    assert 'tag' not in bundled.get('params', {})
    _asset(tiers / 'local', 'establishing', _spec('local'))
    assert _tag('establishing') == 'local'


def test_a_scenario_swap_changes_the_answer_at_once(tiers: Path) -> None:
    first = _scenario(tiers / 'first', wide=_spec('first'))
    second = _scenario(tiers / 'second', wide=_spec('second'))
    point_scenario_assets(first)
    assert _tag('wide') == 'first'
    point_scenario_assets(second)
    assert _tag('wide') == 'second'
    point_scenario_assets(None)
    assert asyncio.run(local_spec('wide')) is None


def test_a_scenario_asset_loads_through_the_identifier(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', wide=_spec('scenario')))
    assert CameraIdentifier.parse('wide').resolve_sync().spec['params']['tag'] == 'scenario'


def test_a_domain_qualified_name_skips_the_scenario_tier(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', wide=_spec('scenario')))
    assert asyncio.run(local_spec('Other/wide')) is None


def test_a_spec_that_is_no_mapping_is_refused(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', wide=['look']))
    with pytest.raises(ValueError, match='expected a mapping'):
        assert CameraIdentifier.parse('wide').resolve_sync().spec


def test_an_exact_name_wins_over_an_index(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', **{'1': _spec('one'), 'a': _spec('a'), 'b': _spec('b')}))
    assert _pick('1') == ('1', None)
    assert _pick('2') == ('b', None)


def test_a_non_negative_index_reads_the_most_specific_tier(tiers: Path) -> None:
    _asset(tiers / 'local', 'zoom_in', _spec('local'))
    point_scenario_assets(_scenario(tiers / 'scenario', b=_spec('b'), a=_spec('a')))
    assert _pick('0') == ('a', None)
    assert _pick('1') == ('b', None)


def test_an_index_falls_to_the_world_tier_without_scenario_assets(tiers: Path) -> None:
    world = tiers / 'world'
    _asset(world / 'assets', 'w', _spec('w'))
    point_world_assets(world, [])
    assert _pick('0') == ('w', None)


def test_an_index_reads_the_global_tier_without_a_world() -> None:
    assert _pick('0') == ('establishing', None)


def test_an_out_of_range_index_falls_back_to_the_first_with_a_warning(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', a=_spec('a')))
    name, warning = _pick('7')
    assert name == 'a'
    assert warning is not None
    assert "'7'" in warning
    assert "'a'" in warning


def test_an_unknown_name_falls_back_to_the_first_with_a_warning(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', a=_spec('a'), b=_spec('b')))
    name, warning = _pick('typo')
    assert name == 'a'
    assert warning is not None
    assert "'typo'" in warning


def test_no_candidate_at_all_picks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Camera, '_TIERS', ())
    name, warning = _pick('typo')
    assert name is None
    assert warning is not None


def test_a_number_never_reaches_the_buckets(tiers: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Bucket:
        async def listall_async(self, **kwargs: bool) -> list[CameraIdentifier]:
            raise AssertionError('the bucket was listed')

    monkeypatch.setattr(Camera, '_NETWORK', [Bucket()])
    assert _pick('0') == ('establishing', None)


def test_a_name_only_a_bucket_lists_is_picked(monkeypatch: pytest.MonkeyPatch) -> None:
    class Bucket:
        async def listall_async(self, **kwargs: bool) -> list[CameraIdentifier]:
            return [CameraIdentifier('remote')]

    monkeypatch.setattr(Camera, '_NETWORK', [Bucket()])
    assert _pick('remote') == ('remote', None)


def test_a_name_no_bucket_lists_is_asked_once(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[bool] = []

    class Bucket:
        async def listall_async(self, **kwargs: bool) -> list[CameraIdentifier]:
            asked.append(True)
            return []

    monkeypatch.setattr(Camera, '_NETWORK', [Bucket()])
    _pick('typo')
    _pick('typo')
    assert len(asked) == 1


def test_find_spec_answers_none_for_an_unknown_name() -> None:
    assert asyncio.run(find_spec('ghost')) is None


def test_candidates_sort_by_name(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', b=_spec('b'), a=_spec('a')))
    assert [identifier.name for identifier in asyncio.run(candidates())] == ['a', 'b']


def test_step_names_walk_the_timeline_and_every_camera() -> None:
    spec = {'cameras': {'a': {'timeline': [{'look': {}}, {'x': {}}]}, 'b': {'mount': {'frame': 'f'}}, 'c': {'timeline': [{'y': {}}]}}}
    assert list(step_names(spec)) == ['look', 'x', 'y']
    assert list(step_names(_spec('t', 'z'))) == ['z']


def test_collect_follows_nested_assets_across_tiers_and_skips_verbs(tiers: Path) -> None:
    _asset(tiers / 'local', 'inner', _spec('inner', 'hold'))
    middle = _spec('middle', 'inner', 'look')
    point_scenario_assets(_scenario(tiers / 'scenario', middle=middle))
    root = {'cameras': {'a': {'timeline': [{'middle': {}}, {'orbit': {}}]}}}
    bundle = asyncio.run(collect('root', root))
    assert sorted(bundle) == ['inner', 'middle', 'root']
    assert bundle['root'] is root


def test_collect_survives_a_cycle(tiers: Path) -> None:
    point_scenario_assets(_scenario(tiers / 'scenario', a=_spec('a', 'b'), b=_spec('b', 'a')))
    assert sorted(asyncio.run(collect('a', _spec('a', 'b')))) == ['a', 'b']


def test_collect_never_asks_the_buckets_for_a_step(tiers: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Bucket:
        async def listall_async(self, **kwargs: bool) -> list[CameraIdentifier]:
            raise AssertionError('the bucket was listed')

    monkeypatch.setattr(Camera, '_NETWORK', [Bucket()])
    assert sorted(asyncio.run(collect('root', _spec('root', 'look', 'ghost')))) == ['root']
