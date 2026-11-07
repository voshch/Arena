"""Write a World as canonical format 2: short forms, rounded numbers, nothing the parser would fill back in."""

from __future__ import annotations

import math
from collections.abc import Mapping

import attrs
import yaml

from arena_simulation_setup import DOMAIN_DEFAULT
from arena_simulation_setup.shared import Door, Stairs
from arena_simulation_setup.shared.entities import Waypoint
from arena_simulation_setup.shared.semantics import authored_semantics
from arena_simulation_setup.shared.world import STAIRS_HANDRAIL_INSET, StairsRail, StairsSide
from arena_simulation_setup.tree import DomainAssetIdentifier, ModifiersDomainAssetIdentifier
from arena_simulation_setup.utils.cattrs import Serializable
from arena_simulation_setup.utils.geometry import Pose, Position, Scale

from .model import FORMAT, Level, WallPath, World, Zone, ZoneCeiling, ZoneWalls

POSITION_DECIMALS = 4
ANGLE_DECIMALS = 6
NUMBER_DECIMALS = 6
_SKIP = frozenset({'extra', 'included_from', 'level'})


def _number(value: float, decimals: int = NUMBER_DECIMALS) -> float | int:
    rounded = round(float(value), decimals)
    return 0.0 if rounded == 0.0 else rounded


def short_position(position: Position) -> list:
    values = [_number(position.x, POSITION_DECIMALS), _number(position.y, POSITION_DECIMALS)]
    if _number(position.z, POSITION_DECIMALS) != 0.0:
        values.append(_number(position.z, POSITION_DECIMALS))
    return values


def short_pose(pose: Pose) -> list | dict:
    roll, pitch, yaw = (_number(angle, ANGLE_DECIMALS) for angle in pose.orientation.to_euler())
    x, y, z = (_number(component, POSITION_DECIMALS) for component in pose.position)
    if roll == 0.0 and pitch == 0.0 and z == 0.0:
        return [x, y] if yaw == 0.0 else [x, y, yaw]
    result: dict = {'x': x, 'y': y}
    for key, value in (('z', z), ('roll', roll), ('pitch', pitch), ('yaw', yaw)):
        if value != 0.0:
            result[key] = value
    return result


def short_identifier(identifier: DomainAssetIdentifier) -> str | list:
    name = identifier.name if identifier.domain == DOMAIN_DEFAULT else f'{identifier.domain}/{identifier.name}'
    if isinstance(identifier, ModifiersDomainAssetIdentifier) and identifier.modifiers:
        return [name, short(identifier.modifiers)]
    return name


def short(value: object) -> object:
    """The shortest form of a value that the format 2 parser reads back to an equal value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, Waypoint):
        if value.level:
            result: dict[str, object] = dict(zip(('x', 'y', 'z'), short_position(value), strict=False))
            result['level'] = value.level
            return result
        return short_position(value)
    if isinstance(value, Position):
        return short_position(value)
    if isinstance(value, Pose):
        return short_pose(value)
    if isinstance(value, Scale):
        return [_number(value.x), _number(value.y), _number(value.z)]
    if isinstance(value, DomainAssetIdentifier):
        return short_identifier(value)
    if isinstance(value, tuple) and len(value) == 2 and value[0] == value[1]:
        return short(value[0])
    if isinstance(value, Mapping):
        return {str(key): short(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [short(item) for item in value]
    if isinstance(value, Serializable):
        return short(value.serialize())
    if attrs.has(type(value)):
        return {field.name: short(getattr(value, field.name)) for field in attrs.fields(type(value))}
    raise TypeError(f'cannot write {value!r}')


def _default(field: attrs.Attribute) -> object:
    if field.default is attrs.NOTHING:
        return attrs.NOTHING
    if isinstance(field.default, attrs.Factory):  # type: ignore[arg-type]
        return field.default.factory()  # type: ignore[union-attr]
    return field.default


def _fields(obj: object, baseline: Mapping[str, object] | None = None, keep: frozenset[str] = frozenset({'name'}), skip: frozenset[str] = _SKIP) -> dict:
    """The fields of obj in short form, dropping those equal to the baseline or the field default."""
    result: dict = {}
    for field in attrs.fields(type(obj)):
        if field.name in skip or not field.init:
            continue
        value = getattr(obj, field.name)
        written = authored_semantics(value) if field.name == 'semantics' else short(value)
        if field.name not in keep:
            if baseline is not None and field.name in baseline:
                reference = baseline[field.name]
            else:
                reference = _default(field)
                if reference is attrs.NOTHING:
                    result[field.name] = written
                    continue
            if field.name == 'semantics':
                assert isinstance(reference, list)
                reference = authored_semantics(reference)
            else:
                reference = short(reference)
            if written == reference:
                continue
        result[field.name] = written
    return result


def _baseline(obj: object, raw_defaults: Mapping) -> dict:
    """Field values the parser gives an item from the defaults alone."""
    cls = type(obj)
    return {field.name: getattr(obj, field.name) for field in attrs.fields(cls) if field.name in raw_defaults}


def dump_door(door: Door, defaults: Mapping) -> dict:
    door_defaults = dict(defaults.get('door') or {})
    baseline = {}
    if door_defaults:
        reference = Door.parse({'name': door.name, 'span': [[0, 0], [1, 0]], **door_defaults})
        baseline = _baseline(reference, door_defaults)
    result = _fields(door, baseline, skip=_SKIP | {'start', 'end'})
    return {'name': result.pop('name'), 'span': [short_position(door.start), short_position(door.end)], **result}


def dump_zone(zone: Zone, defaults: Mapping) -> dict:
    result: dict = {'name': zone.name, 'polygon': [short_position(point) for point in zone.polygon]}
    floor = short(zone.floor) if zone.floor is not None else False
    floor_default = defaults.get('floor', ...)
    reference_floor = short(Zone(name='').floor) if floor_default is ... else (False if floor_default in (False, None, '') else short(Zone(name='', floor=floor_default).floor))
    if floor != reference_floor:
        result['floor'] = floor
    ceiling = _dump_ceiling(zone.ceiling)
    ceiling_default = defaults.get('ceiling', ...)
    reference_ceiling = _dump_ceiling(ZoneCeiling()) if ceiling_default is ... else (False if ceiling_default is False else _dump_ceiling(ZoneCeiling(**ceiling_default)))
    if ceiling != reference_ceiling:
        if isinstance(ceiling, dict) and isinstance(reference_ceiling, dict):
            ceiling = {key: value for key, value in ceiling.items() if reference_ceiling.get(key, ...) != value}
        result['ceiling'] = ceiling
    if zone.walls is not None:
        result['walls'] = _dump_zone_walls(zone.walls, defaults)
    semantics = authored_semantics(zone.semantics)
    if semantics:
        result['semantics'] = semantics
    return result


def _dump_ceiling(ceiling: ZoneCeiling | None) -> dict | bool:
    if ceiling is None:
        return False
    return _fields(ceiling, keep=frozenset())


def _style(kind: str, material: object, defaults: Mapping) -> dict:
    own: dict = {'kind': kind} if kind else ({'material': short(material)} if material is not None else {})
    default = dict(defaults.get('wall') or {})
    if {key: short(value) if key == 'material' else value for key, value in default.items()} == own:
        return {}
    return own


def _dump_zone_walls(walls: ZoneWalls, defaults: Mapping) -> str | dict:
    result = _style(walls.kind, walls.material, defaults)
    if walls.open_to:
        result = {'outline': True, 'open_to': list(walls.open_to), **result}
    return result or 'outline'


def dump_wall(wall: WallPath, defaults: Mapping) -> dict:
    result: dict = {'path': [short_position(point) for point in wall.path]}
    if wall.closed:
        result['closed'] = True
    result.update(_style(wall.kind, wall.material, defaults))
    return result


_STAIRS_DEFAULTS = Stairs(name='', position=Position(0.0, 0.0))


def dump_stair(bottom: Stairs, top: Stairs, to: str, layout: str) -> dict:
    """The authored entry of the stair a bottom and a top half make, over a stair kind only the keys it sets itself."""
    result: dict = {'name': bottom.group, **({'kind': bottom.preset} if bottom.preset else {}), 'mouth': short_position(bottom.mouth)}
    if bottom.mouth_side != _STAIRS_DEFAULTS.mouth_side:
        result['mouth_side'] = bottom.mouth_side
    result['size'] = [_number(2.0 * bottom.length), _number(bottom.width)]
    result['to'] = to
    if layout == 'spread':
        result['exit'] = short_position(top.mouth)
    if bottom.kind == 'escalator' and not bottom.boards:
        result['direction'] = 'down'
    own = (lambda key: key in bottom.authored) if bottom.preset else None
    for key in ('style', 'speed', 'tread', 'landing_length', 'plank', 'headroom', 'material'):
        value = getattr(bottom, key)
        if own(key) if own else value != getattr(_STAIRS_DEFAULTS, key):
            result[key] = short(value)
    surfaces = {key: _surface(bottom, key) for key in dict.fromkeys([*bottom.surfaces, *bottom.surface_models]) if not own or own(f'surfaces.{key}')}
    if surfaces:
        result['surfaces'] = surfaces
    sides = {name: _dressing(side, StairsSide(), keep=frozenset(key.rsplit('.', 1)[1] for key in bottom.authored if key.startswith(f'sides.{name}.')) if own else None) for name, side in bottom.sides.items()}
    if any(sides.values()):
        result['sides'] = {name: side for name, side in sides.items() if side}
    if bottom.models and (not own or own('models')):
        result['models'] = [{'model': short(item.model), **({'at': [_number(v) for v in item.at]} if any(item.at) else {}), **({'yaw': _number(item.yaw)} if item.yaw else {})} for item in bottom.models]
    inner = bottom.width / 2.0 - STAIRS_HANDRAIL_INSET
    if bottom.rails and (not own or own('rails')):
        result['rails'] = [{'across': 'left' if math.isclose(rail.across, inner) else 'right' if math.isclose(rail.across, -inner) else _number(rail.across), **_dressing(rail, StairsRail(), skip=frozenset({'across'}))} for rail in bottom.rails]
    return result


def _surface(stair: Stairs, key: str) -> object:
    """A surface as authored: its material, or a mapping of material and model when it takes a model."""
    if key not in stair.surface_models:
        return short(stair.surfaces[key])
    return {**({'material': short(stair.surfaces[key])} if key in stair.surfaces else {}), 'model': short(stair.surface_models[key])}


def _dressing(value: StairsSide | StairsRail, default: StairsSide | StairsRail, skip: frozenset[str] = frozenset(), keep: frozenset[str] | None = None) -> dict:
    """The fields of a stair side or rail that differ from its default, or exactly the `keep` fields."""
    result: dict = {}
    for field in attrs.fields(type(value)):
        mine = getattr(value, field.name)
        if field.name in skip or mine is None or (field.name not in keep if keep is not None else mine == getattr(default, field.name)):
            continue
        result[field.name] = short(mine)
    return result


def dump_ramp(ramp: Stairs, defaults: Mapping) -> dict:
    """The authored entry of a ramp, its floor only where it differs from the world's default."""
    result: dict = {'name': ramp.name, 'mouth': [_number(ramp.mouth.x, POSITION_DECIMALS), _number(ramp.mouth.y, POSITION_DECIMALS)]}
    if ramp.mouth_side != _STAIRS_DEFAULTS.mouth_side:
        result['mouth_side'] = ramp.mouth_side
    result['size'] = [_number(ramp.length), _number(ramp.width)]
    if ramp.position.z:
        result['base'] = _number(ramp.position.z)
    if ramp.rise:
        result['rise'] = _number(ramp.rise)
    floor = False if not ramp.floor else short(ramp.material) if ramp.material != _STAIRS_DEFAULTS.material else True
    if floor != defaults.get('floor', True):
        result['floor'] = floor
    sides = {name: _dressing(side, StairsSide(height=0.0)) for name, side in ramp.sides.items()}
    if any(sides.values()):
        result['sides'] = {name: side for name, side in sides.items() if side}
    return result


def _stair_entries(world: World, level: Level) -> dict[str, list[dict]]:
    """The stair, escalator and ramp entries a level authors, one per stair whose bottom half it holds."""
    tops = {stair.group: (level_id, stair) for level_id, other in world.levels.items() for stair in other.stairs if stair.top}
    entries: dict[str, list[dict]] = {'stairs': [], 'escalators': [], 'ramps': []}
    for stair in level.stairs:
        if stair.kind == 'ramp':
            entries['ramps'].append(dump_ramp(stair, world.defaults))
            continue
        if stair.top or stair.group not in tops:
            continue
        to, top = tops[stair.group]
        entries['stairs' if stair.kind == 'stairs' else 'escalators'].append(dump_stair(stair, top, to, world.layout))
    return entries


def dump_level(level: Level, defaults: Mapping, stairs: Mapping[str, list[dict]] | None = None) -> dict:
    result: dict = {'id': level.id}
    if level.elevation is not None:
        result['elevation'] = _number(level.elevation)
    if level.source:
        result['from'] = level.source
        if level.prefix:
            result['prefix'] = level.prefix
        return result
    sections: list[tuple[str, list]] = [
        ('zones', [dump_zone(zone, defaults) for zone in level.zones]),
        ('walls', [dump_wall(wall, defaults) for wall in level.walls]),
        ('doors', [dump_door(door, defaults) for door in level.doors]),
        ('elevators', [_fields(elevator) for elevator in level.elevators]),
        ('stairs', list((stairs or {}).get('stairs', []))),
        ('escalators', list((stairs or {}).get('escalators', []))),
        ('ramps', list((stairs or {}).get('ramps', []))),
        ('objects', [_fields(entity, keep=frozenset({'name', 'model', 'pose'})) for entity in level.objects]),
        ('humans', [_fields(human, keep=frozenset({'name', 'model', 'pose'})) for human in level.humans]),
        ('lights', [_fields(light, skip=frozenset({'extra', 'owner', 'rig', 'fixtures', 'fixture_ranks'})) for light in level.lights]),
        ('sounds', [_fields(sound, skip=frozenset({'extra'})) for sound in level.sounds]),
        ('microphones', [_fields(microphone, keep=frozenset({'zone', 'placement', 'position'})) for microphone in level.microphones]),
    ]
    for key, items in sections:
        if items:
            result[key] = items
    if not level.pedestrian_walls:
        result['pedestrian_walls'] = False
    return result


def dump_world(world: World) -> dict:
    """The mapping of a canonical format 2 world.yaml."""
    result: dict = {'format': FORMAT}
    if world.layout != 'spread':
        result['layout'] = world.layout
    if world.defaults:
        result['defaults'] = short(world.defaults)
    if world.schedules:
        result['schedules'] = [_fields(item) for item in world.schedules]
    if world.signals:
        result['signals'] = [_fields(item) for item in world.signals]
    result['levels'] = [dump_level(level, world.defaults, _stair_entries(world, level)) for level in world.levels.values()]
    return result


class _Dumper(yaml.SafeDumper):
    pass


def _flat(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _inline(value: object, depth: int = 0) -> bool:
    if _flat(value):
        return True
    if depth > 2:
        return False
    if isinstance(value, list):
        return all(_inline(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(_inline(item, depth + 1) for item in value.values())
    return False


def _represent_list(dumper: yaml.SafeDumper, value: list) -> yaml.Node:
    flow = all(_flat(item) or (isinstance(item, list) and all(_flat(x) for x in item)) for item in value) and len(repr(value)) <= 400
    return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=flow)


def _represent_dict(dumper: yaml.SafeDumper, value: dict) -> yaml.Node:
    flow = _inline(value) and len(repr(value)) <= 140
    return dumper.represent_mapping('tag:yaml.org,2002:map', value.items(), flow_style=flow)


_Dumper.add_representer(list, _represent_list)
_Dumper.add_representer(dict, _represent_dict)


def _represent_float(dumper: yaml.SafeDumper, value: float) -> yaml.Node:
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return dumper.represent_scalar('tag:yaml.org,2002:float', f'{int(value)}.0')
    return dumper.represent_scalar('tag:yaml.org,2002:float', repr(value))


_Dumper.add_representer(float, _represent_float)


def to_yaml(data: object) -> str:
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=False, width=1000)


def write_world(world: World) -> str:
    """Canonical format 2 world.yaml text of a world."""
    return to_yaml(dump_world(world))
