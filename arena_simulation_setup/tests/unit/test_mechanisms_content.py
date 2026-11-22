"""Load every mechanisms world, scenario, agent type and suite, and cross-check the names they use."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import shapely
import yaml

from arena_simulation_setup.shared.conditions import EntityAtom, MembershipAtom, parse_atom
from arena_simulation_setup.shared.task import GoToPhase
from arena_simulation_setup.tree.World.Scenario import ScenarioView
from arena_simulation_setup.tree.World import WorldView

_PKG = Path(__file__).resolve().parents[2]
_WORLDS = _PKG / 'worlds'
_SUITES = _PKG.parent / 'arena_evaluation' / 'arena_evaluation' / 'configs' / 'suites'

WORLD_NAMES = sorted(p.name for p in _WORLDS.glob('mechanisms_*') if p.is_dir())

PENDING: dict[str, str] = {}

_ENTITY_FIELDS = {
    'door': {'state', 'progress', 'open', 'in_transit', 'triggered', 'blocked_close', 'locked', 'blocked', 'pressed'},
    'elevator': {'cabin_door', 'target', 'arriving_eta', 'occupants', 'cabin_door_progress', 'live_occupants', 'capacity', 'departing', 'in_transit', 'dispatched', 'just_arrived', 'full', 'pressed'},
    'schedule': {'state', 'window_remaining', 'active'},
    'signal': {'state', 'phase_remaining', 'stop'},
    'sound': {'sounding', 'volume_db'},
    'zone': {'occupancy', 'cap', 'over_cap', 'max_speed', 'quiet', 'restricted', 'egress'},
}


def _view(world: str) -> WorldView:
    return WorldView(_WORLDS / world)


def _entities(world: str) -> tuple[dict[str, str], dict[str, str]]:
    """(realized-bare name, `<name>/<level>` or `<name>` at world scope, -> kind, authored name -> kind) for every semantic entity."""
    desc = _view(world).load()
    realized: dict[str, str] = {}
    authored: dict[str, str] = {}
    for level_id, level in desc.levels.items():
        items = [(z.name, 'zone') for z in level.zones]
        items += [(d.name, 'door') for d in level.doors]
        items += [(e.name, 'elevator') for e in level.elevators]
        items += [(s.name, 'sound') for s in level.sounds]
        for name, kind in items:
            realized[f'{name}/{level_id}'] = kind
            authored[name] = kind
    for name, kind in [*((s.name, 'schedule') for s in desc.schedules), *((s.name, 'signal') for s in desc.signals)]:
        realized[name] = kind
        authored[name] = kind
    return realized, authored


def _zone_shapes(world: str) -> dict[str, dict[str, shapely.Polygon]]:
    """level id -> zone name -> polygon (level frame)."""
    desc = _view(world).load()
    return {lid: {z.name: shapely.Polygon([(c.x, c.y) for c in z.polygon]) for z in level.zones} for lid, level in desc.levels.items()}


def _scenario_ids() -> list[tuple[str, str]]:
    out = []
    for world in WORLD_NAMES:
        for path in sorted((_WORLDS / world / 'scenarios').iterdir()):
            if path.is_dir():
                out.append((world, path.name))
    return out


def _params() -> list:
    params = []
    for world, sid in _scenario_ids():
        marks = [pytest.mark.xfail(reason=PENDING[sid], strict=False)] if sid in PENDING else []
        params.append(pytest.param(world, sid, id=sid, marks=marks))
    return params


def _load(world: str, sid: str):
    desc = _view(world).load()
    compacted = desc.compact_world(dict.fromkeys(desc.level_ids, (0.0, 0.0)))
    converter = compacted.zone_converter(np.random.default_rng(0))
    return ScenarioView(_WORLDS / world / 'scenarios' / sid).load(converter=converter)


def test_mechanisms_worlds_present():
    assert {'mechanisms_doors', 'mechanisms_doorway', 'mechanisms_elevator', 'mechanisms_regimes'} <= set(WORLD_NAMES)


@pytest.mark.parametrize('world', WORLD_NAMES)
def test_world_loads_and_validates(world):
    desc = _view(world).load(validate=True)
    assert desc.levels
    names = [z.name for level in desc.levels.values() for z in level.zones]
    assert len(names) == len(set(names)), 'zone names must be unique across levels (conditions address them bare)'
    for level in desc.levels.values():
        for zone in level.zones:
            assert shapely.Polygon([(c.x, c.y) for c in zone.polygon]).is_valid, zone.name


@pytest.mark.parametrize('world', WORLD_NAMES)
def test_world_zones_do_not_overlap(world):
    for level_id, shapes in _zone_shapes(world).items():
        items = list(shapes.items())
        for i, (a, pa) in enumerate(items):
            for b, pb in items[i + 1 :]:
                assert pa.intersection(pb).area < 1e-6, f'level {level_id}: zones {a} and {b} overlap'


@pytest.mark.parametrize(('world', 'sid'), _params())
def test_scenario_loads(world, sid):
    scenario = _load(world, sid)
    assert scenario.robots, 'every mechanisms scenario drives one robot'
    assert scenario.conditions, 'every mechanisms scenario encodes its norm in conditions:'


@pytest.mark.parametrize(('world', 'sid'), _params())
def test_scenario_names_resolve(world, sid):
    """Atoms, timeline entities and via elevators reference things that exist in the world."""
    scenario = _load(world, sid)
    realized, authored = _entities(world)
    zones = {name for shapes in _zone_shapes(world).values() for name in shapes}
    peds = {d.name for d in scenario.humans}

    for clause in scenario.conditions:
        for raw in filter(None, (clause.p, clause.q)):
            atom = parse_atom(raw)
            if isinstance(atom, MembershipAtom):
                assert atom.zone in zones, f'{raw}: unknown zone'
                assert atom.subject == 'robot' or atom.subject in peds, f'{raw}: unknown subject'
            else:
                assert isinstance(atom, EntityAtom)
                assert atom.entity in realized, f'{raw}: entity must be `<authored name>/<level id>` or a world-scope name, known: {sorted(realized)}'
                assert atom.field in _ENTITY_FIELDS[realized[atom.entity]], f'{raw}: unknown field for {realized[atom.entity]}'

    for entry in scenario.timeline:
        for action in entry.set:
            assert action['entity'] in authored, f"timeline entity {action['entity']!r} is not an authored name"
            assert action['field'] in _ENTITY_FIELDS[authored[action['entity']]]

    for goal in scenario.robots:
        for phase in goal.phase_list():
            if isinstance(phase, GoToPhase):
                for via in phase.via:
                    assert via.split('>')[0] in authored and authored[via.split('>')[0]] == 'elevator', via


@pytest.mark.parametrize(('world', 'sid'), _params())
def test_scenario_poses_inside_zones(world, sid):
    """Robot starts/goals and ped spawns/waypoints sit inside a zone of their level."""
    scenario = _load(world, sid)
    shapes = _zone_shapes(world)
    authored_kinds = _entities(world)[1]
    single = next(iter(shapes)) if len(shapes) == 1 else None

    def inside(x: float, y: float, level: str | None) -> bool:
        level = level or single
        assert level is not None, f'({x}, {y}) needs a level in a multi-level world'
        return any(p.covers(shapely.Point(x, y)) for p in shapes[level].values())

    for goal in scenario.robots:
        p = goal.start.position
        assert inside(p.x, p.y, goal.level), f'robot start ({p.x}, {p.y})'
        for phase in goal.phase_list():
            if isinstance(phase, GoToPhase) and phase.pose is not None:
                q = phase.pose.position
                assert inside(q.x, q.y, goal.goal_level or goal.level), f'robot goto ({q.x}, {q.y})'
    for d in scenario.humans:
        p = d.pose.position
        assert inside(p.x, p.y, d.level or None), f'{d.name} spawn ({p.x}, {p.y})'
        for w in d.waypoints:
            if w.elevator:
                assert authored_kinds.get(w.elevator) == 'elevator', f'{d.name} rides unknown elevator {w.elevator!r}'
                continue
            assert inside(w.x, w.y, w.level), f'{d.name} waypoint ({w.x}, {w.y})'


def _agent_type_files() -> list[Path]:
    return sorted(p for world in WORLD_NAMES for p in (_WORLDS / world / 'scenarios').glob('*/*.yaml') if p.name != 'scenario.yaml')


@pytest.mark.parametrize('path', _agent_type_files(), ids=lambda p: f'{p.parent.name}/{p.name}')
def test_agent_types_load(path):
    loader = pytest.importorskip('arena_humansim.core.agents.loader')
    agent_type = loader.load_agent_type_from_file(str(path))
    assert agent_type is not None


def _suites() -> list[Path]:
    return sorted(_SUITES.glob('mechanisms_*/suite.yaml')) if _SUITES.is_dir() else []


def test_mechanisms_suites_present():
    if not _SUITES.is_dir():
        pytest.skip('arena_evaluation submodule not checked out')
    names = {p.parent.name for p in _suites()}
    assert {'mechanisms_doors', 'mechanisms_doors_crowd', 'mechanisms_elevator', 'mechanisms_regimes', 'mechanisms_faults', 'mechanisms_smoke'} <= names


@pytest.mark.parametrize('path', _suites(), ids=lambda p: p.parent.name)
def test_suite_stages_resolve(path):
    suite = yaml.safe_load(path.read_text())
    assert suite['name'] == path.parent.name
    assert suite['stages']
    seen = set()
    for stage in suite['stages']:
        assert stage['name'] not in seen
        seen.add(stage['name'])
        assert stage['tm_robots'] == 'scenario' and stage['tm_obstacles'] == 'scenario'
        assert re.fullmatch(r'\d+s', stage['timeout'])
        assert (_WORLDS / stage['map'] / 'scenarios' / stage['config']['scenario']['file'] / 'scenario.yaml').is_file(), stage['name']


def test_every_scenario_is_in_a_suite():
    if not _SUITES.is_dir():
        pytest.skip('arena_evaluation submodule not checked out')
    staged = {(s['map'], s['config']['scenario']['file']) for p in _suites() if p.parent.name != 'mechanisms_smoke' for s in yaml.safe_load(p.read_text())['stages']}
    assert staged == set(_scenario_ids())
