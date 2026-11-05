"""Read a format 2 world mapping into a World, defaults applied, every error prefixed with its YAML path."""

from __future__ import annotations

import logging
import typing
from collections.abc import Callable, Mapping

import attrs

from arena_simulation_setup.shared import Door, DynamicObstacle, Elevator, Light, Obstacle, Schedule, Signal, Sound, Stairs
from arena_simulation_setup.shared.world import STAIRS_HANDRAIL, STAIRS_HANDRAIL_INSET, STAIRS_MODEL_SURFACES, STAIRS_RAIL, STAIRS_SURFACES, STOREY, StairsModel, StairsRail, StairsSide
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.utils.cattrs import ArenaConverter, converter
from arena_simulation_setup.utils.geometry import Position

from ..Stairs import STAIRS_KIND_KEYS, StairsIdentifier
from .model import FORMAT, Level, WallPath, World, WorldMicrophone, Zone, ZoneCeiling, ZoneWalls

T = typing.TypeVar('T')

logger = logging.getLogger(__name__)

WORLD_KEYS = frozenset({'format', 'layout', 'defaults', 'schedules', 'signals', 'levels'})
DEFAULT_KEYS = frozenset({'floor', 'ceiling', 'wall', 'door'})
LEVEL_KEYS = frozenset({'id', 'elevation', 'zones', 'walls', 'doors', 'elevators', 'stairs', 'escalators', 'ramps', 'objects', 'humans', 'lights', 'sounds', 'microphones', 'pedestrian_walls'})
RAMP_KEYS = frozenset({'name', 'mouth', 'mouth_side', 'size', 'base', 'rise', 'floor', 'sides'})
STAIRS_KEYS = frozenset({'name', 'kind', 'mouth', 'mouth_side', 'size', 'to', 'exit', 'style', 'speed', 'direction', 'tread', 'landing_length', 'plank', 'headroom', 'material', 'surfaces', 'sides', 'rails', 'models'})
STAIRS_SURFACE_KEYS = frozenset({'material', 'model'})
STAIRS_MODEL_KEYS = frozenset({'model', 'at', 'yaw'})
STAIRS_SIDE_KEYS = frozenset({'kind', 'material', 'height'})
STAIRS_RAIL_KEYS = frozenset({'across', 'kind', 'material', 'height', 'solid'})
ZONE_KEYS = frozenset({'name', 'polygon', 'floor', 'ceiling', 'walls', 'semantics'})
CEILING_KEYS = frozenset({'height', 'material', 'cast_shadows', 'lights'})
ZONE_WALL_KEYS = frozenset({'outline', 'open_to', 'kind', 'material'})
WALL_KEYS = frozenset({'path', 'closed', 'kind', 'material'})
WALL_STYLE_KEYS = frozenset({'kind', 'material'})
MICROPHONE_KEYS = frozenset({'zone', 'placement', 'position', 'frame', 'index'})
LINK_KEYS = frozenset({'id', 'from', 'prefix'})
STAIRS_LISTS = (('stairs', 'stairs'), ('escalators', 'escalator'))
_SIDES = {'+x': (1.0, 0.0), '-x': (-1.0, 0.0), '+y': (0.0, 1.0), '-y': (0.0, -1.0)}


class WorldFormatError(ValueError):
    """A world file that does not follow the format, with the YAML path of the offending entry."""


def _at(path: str, fn: Callable[[], T]) -> T:
    try:
        return fn()
    except WorldFormatError:
        raise
    except Exception as exc:
        raise WorldFormatError(f'{path}: {exc}') from exc


def _keys(path: str, value: object, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> Mapping:
    if not isinstance(value, Mapping):
        raise WorldFormatError(f'{path}: expected a mapping, got {value!r}')
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise WorldFormatError(f'{path}: unknown keys {unknown}, expected some of {sorted(allowed)}')
    missing = sorted(required - set(value))
    if missing:
        raise WorldFormatError(f'{path}: missing keys {missing}')
    return value


def _items(path: str, value: object) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise WorldFormatError(f'{path}: expected a list, got {value!r}')
    return value


def _structure(c: ArenaConverter, value: object, cls: type[T]) -> T:
    return typing.cast(T, c.structure(value, cls))


def _wall_style(path: str, value: object) -> dict:
    style = dict(_keys(path, value, WALL_STYLE_KEYS))
    if 'kind' in style and 'material' in style:
        raise WorldFormatError(f'{path}: takes kind or material, not both')
    return style


def _ceiling(path: str, value: object, default: object) -> ZoneCeiling | None:
    if value is False or value is None:
        return None
    base = dict(default) if isinstance(default, Mapping) else {}
    if value is True:
        value = {}
    merged = {**base, **_keys(path, value, CEILING_KEYS)}
    return _at(path, lambda: ZoneCeiling(**merged))


def _zone(path: str, raw: object, defaults: Mapping) -> Zone:
    raw = _keys(path, raw, ZONE_KEYS, frozenset({'name'}))
    fields: dict = {'name': str(raw['name']), 'polygon': raw.get('polygon') or [], 'semantics': raw.get('semantics') or []}
    floor = raw['floor'] if 'floor' in raw else defaults.get('floor', ...)
    if floor is not ...:
        fields['floor'] = None if floor in (False, None, '') else floor
    if 'ceiling' in raw:
        fields['ceiling'] = _ceiling(f'{path}.ceiling', raw['ceiling'], defaults.get('ceiling'))
    elif 'ceiling' in defaults:
        fields['ceiling'] = _ceiling(f'{path}.ceiling', defaults['ceiling'] if defaults['ceiling'] is not False else False, None)
    walls = raw.get('walls')
    if walls is not None and walls is not False:
        style = _wall_style('defaults.wall', defaults.get('wall') or {})
        if walls == 'outline' or walls is True:
            fields['walls'] = ZoneWalls(**style)
        else:
            walls = dict(_keys(f'{path}.walls', walls, ZONE_WALL_KEYS))
            if walls.pop('outline', True) is not True:
                raise WorldFormatError(f'{path}.walls: outline must be true, drop walls to draw none')
            own = _wall_style(f'{path}.walls', {key: walls.pop(key) for key in ('kind', 'material') if key in walls})
            fields['walls'] = _at(f'{path}.walls', lambda: ZoneWalls(**{**(own or style), **walls}))
    return _at(path, lambda: Zone(**fields))


def _wall(path: str, raw: object, defaults: Mapping) -> WallPath:
    raw = dict(_keys(path, raw, WALL_KEYS, frozenset({'path'})))
    if 'kind' in raw and 'material' in raw:
        raise WorldFormatError(f'{path}: takes kind or material, not both')
    if 'kind' not in raw and 'material' not in raw:
        raw.update(_wall_style('defaults.wall', defaults.get('wall') or {}))
    return _at(path, lambda: WallPath(**raw))


def _named(path: str, raw: object, cls: type[T], c: ArenaConverter, default: Mapping | None = None) -> T:
    if not isinstance(raw, Mapping):
        raise WorldFormatError(f'{path}: expected a mapping, got {raw!r}')
    merged = {**(default or {}), **raw}
    return _at(path, lambda: _structure(c, merged, cls))


def _stair_entries(path: str, raw: Mapping) -> list[tuple[str, str, Mapping]]:
    """The raw stair and escalator entries of a level as (YAML path, kind, mapping)."""
    entries: list[tuple[str, str, Mapping]] = []
    for key, kind in STAIRS_LISTS:
        for i, item in enumerate(_items(f'{path}.{key}', raw.get(key))):
            item_path = f'{path}.{key}[{i}]'
            entries.append((item_path, kind, _keys(item_path, item, STAIRS_KEYS, frozenset({'name', 'mouth', 'to'}))))
    return entries


def _stair_halves(path: str, raw: Mapping, kind: str, level_id: str, layout: str, elevations: Mapping[str, float], c: ArenaConverter, stair_kind: Callable[[str], Mapping]) -> tuple[Stairs, Stairs]:
    """The bottom half on `level_id` and the top half on the `to` level of one authored stair, over the stair kind it names."""
    preset = str(raw['kind']) if 'kind' in raw else ''
    authored = _authored(raw) if preset else frozenset()
    if preset:
        raw = _over_kind(path, raw, stair_kind)
    name = str(raw['name'])
    to = str(raw['to'])
    if to not in elevations:
        raise WorldFormatError(f'{path}.to: no level {to!r} in this world')
    rise = elevations[to] - elevations[level_id]
    if rise <= 0.0:
        raise WorldFormatError(f'{path}.to: level {to!r} lies at elevation {elevations[to]}, not above level {level_id!r} at {elevations[level_id]}')
    size = raw.get('size', [5.2, 1.4])
    if not isinstance(size, list) or len(size) != 2 or min(float(v) for v in size) <= 0.0:
        raise WorldFormatError(f'{path}.size: expected a positive [length, width], got {size!r}')
    length, width = float(size[0]) / 2.0, float(size[1])
    style = str(raw.get('style', 'steps'))
    if style == 'belt' and kind != 'escalator':
        raise WorldFormatError(f'{path}.style: belt is an escalator style, stairs take steps')
    direction = str(raw.get('direction', 'up'))
    if 'direction' in raw and kind != 'escalator':
        raise WorldFormatError(f'{path}.direction: stairs run both ways, only escalators take a direction')
    if direction not in ('up', 'down'):
        raise WorldFormatError(f'{path}.direction: expected up or down, got {direction!r}')
    if layout == 'stacked' and 'exit' in raw:
        raise WorldFormatError(f'{path}.exit: a stacked world places the top half itself, right after the bottom half')
    if layout == 'spread' and 'exit' not in raw:
        raise WorldFormatError(f'{path}.exit: a spread world takes the mouth of the top half on level {to!r}, in its frame')
    shared = {key: raw[key] for key in ('style', 'speed', 'tread', 'landing_length', 'plank', 'headroom', 'material') if key in raw}
    probe = _named(path, {'name': name, 'position': [0.0, 0.0], 'kind': kind, 'size': [length, width], 'rise': rise, **shared}, Stairs, c)
    for key in ('speed', 'tread', 'plank', 'headroom'):
        if getattr(probe, key) <= 0.0:
            raise WorldFormatError(f'{path}.{key}: must be positive, got {getattr(probe, key)!r}')
    if not 0.0 <= probe.landing_length < length:
        raise WorldFormatError(f'{path}.landing_length: must lie in [0, {length}), half the length of the stair, got {probe.landing_length!r}')
    mouth_side = str(raw.get('mouth_side', '-x'))
    if mouth_side not in _SIDES:
        raise WorldFormatError(f'{path}.mouth_side: expected one of {sorted(_SIDES)}, got {mouth_side!r}')
    ax, ay = (-v for v in _SIDES[mouth_side])
    mouth = _at(f'{path}.mouth', lambda: Position.parse(raw['mouth']))
    exit_point = Position(mouth.x + ax * 2.0 * length, mouth.y + ay * 2.0 * length) if layout == 'stacked' else _at(f'{path}.exit', lambda: Position.parse(raw['exit']))
    far_side = next(side for side, vector in _SIDES.items() if vector == (ax, ay))
    bottom_name, top_name = f'{name}_bottom', f'{name}_top'
    boards_bottom = kind == 'stairs' or direction == 'up'
    boards_top = kind == 'stairs' or direction == 'down'
    common = {'kind': kind, 'size': [length, width], 'rise': rise, 'group': name, 'stacked': layout == 'stacked', **shared}
    bottom = _named(path, {**common, 'name': bottom_name, 'position': [mouth.x + ax * length / 2.0, mouth.y + ay * length / 2.0], 'mouth_side': mouth_side, 'to': top_name if boards_bottom else ''}, Stairs, c)
    top = _named(path, {**common, 'name': top_name, 'position': [exit_point.x - ax * length / 2.0, exit_point.y - ay * length / 2.0], 'mouth_side': far_side, 'top': True, 'to': bottom_name if boards_top else ''}, Stairs, c)
    surfaces, sides, rails = _stair_dressing(path, raw, width)
    surface_models, models = _stair_models(path, raw)
    bottom = attrs.evolve(bottom, surfaces=surfaces, sides=sides, rails=rails, surface_models=surface_models, models=models)
    top = attrs.evolve(top, surfaces=dict(surfaces), sides={'left': sides['right'], 'right': sides['left']}, rails=[attrs.evolve(rail, across=-rail.across) for rail in rails], surface_models=dict(surface_models), models=list(models))
    bottom.preset = top.preset = preset
    bottom.authored = top.authored = authored
    if layout == 'stacked':
        top.overhang = bottom.overhang = _overhang(bottom)
    return bottom, top


def _ramp(path: str, raw: object, defaults: Mapping, c: ArenaConverter) -> Stairs:
    """A level's ramp from `base` at its mouth up `rise` over its length, a slab in the `floor` material unless the floor is off, a side wall only where one is asked for."""
    raw = _keys(path, raw, RAMP_KEYS, frozenset({'name', 'mouth', 'size'}))
    size = raw['size']
    if not isinstance(size, list) or len(size) != 2 or min(float(v) for v in size) <= 0.0:
        raise WorldFormatError(f'{path}.size: expected a positive [length, width], got {size!r}')
    length, width = float(size[0]), float(size[1])
    rise = _at(f'{path}.rise', lambda: float(raw.get('rise', 0.0)))
    if rise < 0.0:
        raise WorldFormatError(f'{path}.rise: the mouth is the low end, expected 0 or more, got {rise!r}')
    base = _at(f'{path}.base', lambda: float(raw.get('base', 0.0)))
    mouth_side = str(raw.get('mouth_side', '-x'))
    if mouth_side not in _SIDES:
        raise WorldFormatError(f'{path}.mouth_side: expected one of {sorted(_SIDES)}, got {mouth_side!r}')
    ax, ay = (-v for v in _SIDES[mouth_side])
    mouth = _at(f'{path}.mouth', lambda: Position.parse(raw['mouth']))
    floor = raw['floor'] if 'floor' in raw else defaults.get('floor', True)
    material = {'material': floor} if isinstance(floor, str) and floor else {}
    position = [mouth.x + ax * length / 2.0, mouth.y + ay * length / 2.0, base]
    ramp = _named(path, {'name': str(raw['name']), 'kind': 'ramp', 'position': position, 'mouth_side': mouth_side, 'size': [length, width], 'rise': rise, 'floor': floor not in (False, None, ''), **material}, Stairs, c)
    sides = _keys(f'{path}.sides', raw.get('sides') or {}, frozenset({'left', 'right'}))
    return attrs.evolve(ramp, sides=_stair_dressing(path, {'sides': {name: {'height': 0.0, **(sides.get(name) or {})} for name in ('left', 'right')}}, width)[1])


def _resolve_stair_kind(name: str) -> dict:
    return StairsIdentifier(name).resolve_sync()


def _authored(raw: Mapping) -> frozenset[str]:
    """The keys an entry sets itself, `surfaces.<surface>` and `sides.<side>.<key>` one by one."""
    keys = {key for key in raw if key not in ('surfaces', 'sides')}
    surfaces, sides = raw.get('surfaces'), raw.get('sides')
    if isinstance(surfaces, Mapping):
        keys.update(f'surfaces.{key}' for key in surfaces)
    if isinstance(sides, Mapping):
        keys.update(f'sides.{side}.{key}' for side, fields in sides.items() if isinstance(fields, Mapping) for key in fields)
    return frozenset(keys)


def _stair_models(path: str, raw: Mapping) -> tuple[dict[str, ObjectIdentifier], list[StairsModel]]:
    """Objects dressing an authored stair: a surface's `model` on each of its boxes, and the `models` placed along it."""
    surface_models: dict[str, ObjectIdentifier] = {}
    for key, value in (raw.get('surfaces') or {}).items():
        if isinstance(value, Mapping) and 'model' in value:
            if key not in STAIRS_MODEL_SURFACES:
                raise WorldFormatError(f'{path}.surfaces.{key}.model: only {", ".join(STAIRS_MODEL_SURFACES)} take a model, {key} takes a material')
            surface_models[key] = _at(f'{path}.surfaces.{key}.model', lambda value=value: ObjectIdentifier.converter(value['model']))
    models = [_at(f'{path}.models[{i}]', lambda item=item, i=i: StairsModel(**_keys(f'{path}.models[{i}]', item, STAIRS_MODEL_KEYS, frozenset({'model'})))) for i, item in enumerate(_items(f'{path}.models', raw.get('models')))]
    return surface_models, models


def _over_kind(path: str, raw: Mapping, stair_kind: Callable[[str], Mapping]) -> dict:
    """The entry over the stair kind it names: its own keys win, surfaces and each side merge key by key, a side's own kind or material replaces the kind's."""
    name = str(raw['kind'])
    preset = _at(f'{path}.kind', lambda: stair_kind(name))
    preset = _keys(f'{path}.kind: stair kind {name!r}', preset, STAIRS_KIND_KEYS)
    merged = {**preset, **raw}
    if 'surfaces' in preset or 'surfaces' in raw:
        merged['surfaces'] = {**(preset.get('surfaces') or {}), **(raw.get('surfaces') or {})}
    if 'sides' in preset or 'sides' in raw:
        sides: dict = {}
        for side in {*(preset.get('sides') or {}), *(raw.get('sides') or {})}:
            own = dict((raw.get('sides') or {}).get(side) or {})
            base = {key: value for key, value in ((preset.get('sides') or {}).get(side) or {}).items() if not ({'kind', 'material'} & set(own) and key in ('kind', 'material'))}
            sides[side] = {**base, **own}
        merged['sides'] = sides
    return merged


def _style_of(path: str, raw: Mapping) -> dict:
    if 'kind' in raw and 'material' in raw:
        raise WorldFormatError(f'{path}: takes kind or material, not both')
    return {'kind': str(raw.get('kind', '')), 'material': _at(f'{path}.material', lambda: MaterialIdentifier.converter(raw['material'])) if 'material' in raw else None}


def _stair_dressing(path: str, raw: Mapping, width: float) -> tuple[dict[str, MaterialIdentifier], dict[str, StairsSide], list[StairsRail]]:
    """Surface materials, side walls and rails of an authored stair, left and right seen walking up from its mouth."""
    surfaces = _keys(f'{path}.surfaces', raw.get('surfaces') or {}, frozenset(STAIRS_SURFACES))
    materials: dict[str, MaterialIdentifier] = {}
    for key, value in surfaces.items():
        if isinstance(value, Mapping):
            value = _keys(f'{path}.surfaces.{key}', value, STAIRS_SURFACE_KEYS).get('material')
        if value is not None:
            materials[key] = _at(f'{path}.surfaces.{key}', lambda value=value: MaterialIdentifier.converter(value))
    sides_raw = _keys(f'{path}.sides', raw.get('sides') or {}, frozenset({'left', 'right'}))
    sides: dict[str, StairsSide] = {}
    for name in ('left', 'right'):
        side_path = f'{path}.sides.{name}'
        side = _keys(side_path, sides_raw.get(name) or {}, STAIRS_SIDE_KEYS)
        height = _at(f'{side_path}.height', lambda side=side: float(side.get('height', STAIRS_RAIL)))
        if height < 0.0:
            raise WorldFormatError(f'{side_path}.height: must be 0 for no side wall or positive, got {height!r}')
        sides[name] = StairsSide(height=height, **_style_of(side_path, side))
    rails: list[StairsRail] = []
    inner = width / 2.0 - STAIRS_HANDRAIL_INSET
    for i, item in enumerate(_items(f'{path}.rails', raw.get('rails'))):
        rail_path = f'{path}.rails[{i}]'
        rail = _keys(rail_path, item, STAIRS_RAIL_KEYS, frozenset({'across'}))
        across = {'left': inner, 'right': -inner}.get(rail['across']) if isinstance(rail['across'], str) else _at(f'{rail_path}.across', lambda rail=rail: float(rail['across']))
        if across is None or abs(across) > width / 2.0:
            raise WorldFormatError(f'{rail_path}.across: expected left, right or meters left of the axis within {width / 2.0}, got {rail["across"]!r}')
        height = _at(f'{rail_path}.height', lambda rail=rail: float(rail.get('height', STAIRS_HANDRAIL)))
        if height <= 0.0:
            raise WorldFormatError(f'{rail_path}.height: must be positive, got {height!r}')
        solid = rail.get('solid', False)
        if not isinstance(solid, bool):
            raise WorldFormatError(f'{rail_path}.solid: expected true or false, got {solid!r}')
        rails.append(StairsRail(across=across, height=height, solid=solid, **_style_of(rail_path, rail)))
    return materials, sides, rails


def _overhang(bottom: Stairs) -> float:
    """How far before the seam the bottom half's walking surface comes within `headroom` of the floor above."""
    limit = bottom.rise - bottom.headroom
    start, end = bottom.ramp
    if limit <= 0.0:
        return bottom.length
    if limit >= bottom.rise / 2.0:
        return 0.0
    return bottom.length - (start + limit / (bottom.rise / 2.0) * (end - start))


def _drop_lost_partners(levels: dict[str, Level]) -> dict[str, Level]:
    """The levels with every `to` of a linked level's half dropped whose partner is not in the world, warned once each."""
    names = {stair.name for level in levels.values() for stair in level.stairs}
    result: dict[str, Level] = {}
    for level_id, level in levels.items():
        lost = [stair for stair in level.stairs if stair.to and stair.to not in names]
        for stair in lost:
            logger.warning('level %s from %s: stair %s loses %r, it is not in this world, it does not board', level_id, level.source, stair.name, stair.to)
        result[level_id] = attrs.evolve(level, stairs=[attrs.evolve(stair, to='') if stair in lost else stair for stair in level.stairs]) if lost else level
    return result


def parse_level(path: str, raw: object, defaults: Mapping, c: ArenaConverter = converter) -> Level:
    raw = _keys(path, raw, LEVEL_KEYS, frozenset({'id'}))
    level_id = str(raw['id'])
    door_default = dict(defaults.get('door') or {})
    return Level(
        id=level_id,
        elevation=None if raw.get('elevation') is None else float(raw['elevation']),
        zones=[_zone(f'{path}.zones[{i}]', item, defaults) for i, item in enumerate(_items(f'{path}.zones', raw.get('zones')))],
        walls=[_wall(f'{path}.walls[{i}]', item, defaults) for i, item in enumerate(_items(f'{path}.walls', raw.get('walls')))],
        doors=[_named(f'{path}.doors[{i}]', item, Door, c, door_default) for i, item in enumerate(_items(f'{path}.doors', raw.get('doors')))],
        elevators=[_named(f'{path}.elevators[{i}]', item, Elevator, c) for i, item in enumerate(_items(f'{path}.elevators', raw.get('elevators')))],
        stairs=[_ramp(f'{path}.ramps[{i}]', item, defaults, c) for i, item in enumerate(_items(f'{path}.ramps', raw.get('ramps')))],
        objects=[_named(f'{path}.objects[{i}]', item, Obstacle, c) for i, item in enumerate(_items(f'{path}.objects', raw.get('objects')))],
        humans=[_named(f'{path}.humans[{i}]', item, DynamicObstacle, c) for i, item in enumerate(_items(f'{path}.humans', raw.get('humans')))],
        lights=[_named(f'{path}.lights[{i}]', item, Light, c) for i, item in enumerate(_items(f'{path}.lights', raw.get('lights')))],
        sounds=[_named(f'{path}.sounds[{i}]', item, Sound, c) for i, item in enumerate(_items(f'{path}.sounds', raw.get('sounds')))],
        microphones=[_at(f'{path}.microphones[{i}]', lambda item=item, i=i: WorldMicrophone(**_keys(f'{path}.microphones[{i}]', item, MICROPHONE_KEYS, frozenset({'zone', 'placement', 'position'})))) for i, item in enumerate(_items(f'{path}.microphones', raw.get('microphones')))],
        pedestrian_walls=bool(raw.get('pedestrian_walls', True)),
    )


def parse_defaults(raw: object) -> dict:
    defaults = dict(_keys('defaults', raw or {}, DEFAULT_KEYS))
    if 'wall' in defaults:
        _wall_style('defaults.wall', defaults['wall'])
    if 'ceiling' in defaults and defaults['ceiling'] is not False:
        _ceiling('defaults.ceiling', defaults['ceiling'], None)
    if 'door' in defaults:
        defaults['door'] = dict(_keys('defaults.door', defaults['door'], _door_default_keys()))
    return defaults


def _door_default_keys() -> frozenset[str]:
    import attrs

    return frozenset(field.name for field in attrs.fields(Door) if field.name not in ('name', 'start', 'end', 'extra', 'semantics'))


def _linked(path: str, raw: Mapping, link: Callable[[str], Level] | None) -> Level:
    raw = _keys(path, raw, LINK_KEYS, frozenset({'from'}))
    source = str(raw['from'])
    if link is None:
        raise WorldFormatError(f'{path}.from: a linked level is read through WorldView, which knows where {source!r} lives')
    try:
        level = link(source)
    except (OSError, LookupError, ValueError) as exc:
        raise WorldFormatError(f'{path}.from: {exc}') from exc
    prefix = str(raw.get('prefix', ''))
    return attrs.evolve(level.prefixed(prefix) if prefix else level, id=str(raw.get('id', level.id)), source=source, prefix=prefix)


def parse_world(raw: object, c: ArenaConverter = converter, link: Callable[[str], Level] | None = None, stair_kind: Callable[[str], Mapping] | None = None) -> World:
    """A World from the mapping of a format 2 world.yaml, its `from:` levels read through `link`, its stair kinds through `stair_kind`."""
    raw = _keys('', raw, WORLD_KEYS, frozenset({'format', 'levels'}))
    if raw['format'] != FORMAT:
        raise WorldFormatError(f'format: expected {FORMAT!r}, got {raw["format"]!r}')
    defaults = parse_defaults(raw.get('defaults'))
    layout = str(raw.get('layout', 'spread'))
    if layout not in ('stacked', 'spread'):
        raise WorldFormatError(f'layout: expected stacked or spread, got {layout!r}')
    levels: dict[str, Level] = {}
    entries: list[tuple[str, str, str, Mapping]] = []
    for index, item in enumerate(_items('levels', raw['levels'])):
        path = f'levels[{index}]'
        linked = isinstance(item, Mapping) and 'from' in item
        level = _linked(path, item, link) if linked else parse_level(path, item, defaults, c)
        if level.id in levels:
            raise WorldFormatError(f'{path}: level id {level.id!r} appears twice')
        levels[level.id] = level
        if not linked and isinstance(item, Mapping):
            entries.extend((level.id, entry_path, kind, entry) for entry_path, kind, entry in _stair_entries(path, item))
    elevations = {level_id: level.elevation if level.elevation is not None else index * STOREY for index, (level_id, level) in enumerate(levels.items())}
    named: dict[str, str] = {}
    halves: dict[str, list[Stairs]] = {level_id: [] for level_id in levels}
    for level_id, path, kind, entry in entries:
        name = str(entry['name'])
        if name in named:
            raise WorldFormatError(f'{path}.name: stair {name!r} is already named at {named[name]}')
        named[name] = path
        bottom, top = _stair_halves(path, entry, kind, level_id, layout, elevations, c, stair_kind or _resolve_stair_kind)
        halves[level_id].append(bottom)
        halves[str(entry['to'])].append(top)
    levels = {level_id: attrs.evolve(level, stairs=[*level.stairs, *halves[level_id]]) if halves[level_id] else level for level_id, level in levels.items()}
    return World(
        levels=_drop_lost_partners(levels),
        layout=layout,
        defaults=defaults,
        schedules=[_named(f'schedules[{i}]', item, Schedule, c) for i, item in enumerate(_items('schedules', raw.get('schedules')))],
        signals=[_named(f'signals[{i}]', item, Signal, c) for i, item in enumerate(_items('signals', raw.get('signals')))],
    )
