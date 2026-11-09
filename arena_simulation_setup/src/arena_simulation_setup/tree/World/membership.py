"""Zone membership and connectivity of a world: stated in its KG, derived from geometry where the KG is silent."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from arena_simulation_setup.kg.scene import DERIVED, Scene, Value
from arena_simulation_setup.params import DERIVE_MODES

from .model import Level, World, anchor_light

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.geometry import Position

_DOOR_MARGIN_M = 0.3


def _inside(level: Level, point: Position) -> list[str]:
    import shapely

    spot = shapely.Point(point.x, point.y)
    return [zone.name for zone in level.zones if len(zone.polygon) >= 3 and zone.shape().covers(spot)]


def _beside(level: Level, start: Position, end: Position) -> list[str]:
    import shapely

    middle = shapely.Point((start.x + end.x) / 2, (start.y + end.y) / 2)
    near = sorted((zone.shape().boundary.distance(middle), zone.name) for zone in level.zones if len(zone.polygon) >= 3)
    return [name for distance, name in near if distance <= _DOOR_MARGIN_M]


def _geometric_refs(world: World) -> set[str]:
    refs: set[str] = set()
    for level in world.levels.values():
        refs.update(f'room:{zone.name}' for zone in level.zones)
        refs.update(f'door:{door.name}' for door in level.doors)
        refs.update(f'obj:{entity.name}' for entity in level.objects)
        refs.update(f'light:{light.name}' for light in level.lights)
        refs.update(f'sound:{sound.name}' for sound in level.sounds)
        refs.update(f'elevator:{elevator.name}' for elevator in level.elevators)
        refs.update(f'stair:{stair.name}' for stair in level.stairs)
    return refs


def _drop_stated_geometry(world: World, scene: Scene) -> None:
    refs = _geometric_refs(world)
    for fact in scene.stated():
        if fact.verb == 'contains' and str(fact.object) in refs and fact.subject.split(':', 1)[0] in ('room', 'floor'):
            scene.remove_fact(fact.subject, fact.verb, fact.object, fact.params)
        elif fact.verb == 'connects' and str(fact.params.get('door')) in refs:
            scene.remove_fact(fact.subject, fact.verb, fact.object, fact.params)


def derive(world: World, mode: str = 'auto') -> Scene:
    """The world's kg with derived facts recomputed: contains for every placed thing and connects for every door that no stated fact covers, or that no fact covers under `all`."""
    if mode not in DERIVE_MODES:
        raise ValueError(f'kg.derive {mode!r} is not one of {", ".join(DERIVE_MODES)}')
    scene = world.kg if world.kg is not None else Scene()
    scene.drop_derived()
    if mode == 'off':
        return scene
    if mode == 'all':
        _drop_stated_geometry(world, scene)
    stated_members = {str(fact.object) for fact in scene.stated() if fact.verb == 'contains'}
    stated_doors = {str(fact.params.get('door')) for fact in scene.stated() if fact.verb == 'connects' and fact.params.get('door')}

    def add(subject: str, verb: str, obj: str, params: dict | None = None) -> None:
        for ref in (subject, obj):
            if scene.get_entity(ref) is None:
                scene.add_entity(ref)
        if scene.get_fact(subject, verb, obj, params) is None:
            scene.add_fact(subject, verb, obj, params, source=DERIVED)

    def contain(rooms: list[str], ref: str) -> None:
        if ref in stated_members:
            return
        for room in rooms:
            add(f'room:{room}', 'contains', ref)

    for level_id, level in world.levels.items():
        floor = f'floor:{level_id}'
        for zone in level.zones:
            if f'room:{zone.name}' not in stated_members:
                add(floor, 'contains', f'room:{zone.name}')
        for door in level.doors:
            rooms = _beside(level, door.start, door.end)
            contain(rooms, f'door:{door.name}')
            if len(rooms) >= 2 and f'door:{door.name}' not in stated_doors:
                add(f'room:{rooms[0]}', 'connects', f'room:{rooms[1]}', {'door': f'door:{door.name}'})
        for entity in level.objects:
            contain(_inside(level, entity.pose.position), f'obj:{entity.name}')
        for light in level.lights:
            placed = anchor_light(light, level.objects)
            if placed.pose is not None and not placed.spec.ambient:
                contain(_inside(level, placed.position), f'light:{light.name}')
        objects = {entity.name: entity for entity in level.objects}
        for sound in level.sounds:
            where = sound.position if sound.position is not None else objects[sound.attach].pose.position if sound.attach in objects else None
            if where is not None:
                contain(_inside(level, where), f'sound:{sound.name}')
        for elevator in level.elevators:
            if f'elevator:{elevator.name}' not in stated_members:
                add(floor, 'contains', f'elevator:{elevator.name}')
            contain(_inside(level, elevator.position), f'elevator:{elevator.name}')
            for destination in elevator.destinations:
                add(f'elevator:{elevator.name}', 'connects_floor', f'elevator:{destination}')
        for stair in level.stairs:
            if f'stair:{stair.name}' not in stated_members:
                add(floor, 'contains', f'stair:{stair.name}')
            contain(_inside(level, stair.position), f'stair:{stair.name}')
            if stair.to:
                add(f'stair:{stair.name}', 'connects_floor', f'stair:{stair.to}')
    return scene


def prune(world: World) -> Scene | None:
    """The world's kg without the stated memberships geometry derives anyway, None when nothing else is left."""
    if world.kg is None:
        return None
    stated = world.kg
    geometric = derive(type(world)(levels=world.levels), 'auto')
    rooms: dict[str, set[str]] = {}
    for fact in stated.stated():
        if fact.verb == 'contains' and fact.subject.startswith('room:'):
            rooms.setdefault(str(fact.object), set()).add(fact.subject)
    for ref, subjects in rooms.items():
        derived = {fact.subject for fact in geometric.facts if fact.verb == 'contains' and fact.object == ref and fact.subject.startswith('room:')}
        if derived == subjects:
            for subject in subjects:
                stated.remove_fact(subject, 'contains', ref)
    referenced = {fact.subject for fact in stated.facts} | {str(fact.object) for fact in stated.facts} | {str(value) for fact in stated.facts for value in fact.params.values()} | {ref for constraint in stated.constraints.values() for ref in constraint.scope}
    for entity in list(stated.entities.values()):
        if not entity.props and entity.id not in referenced:
            stated.remove_entity(entity.id)
    if not stated.entities and not stated.facts and not stated.constraints:
        return None
    return stated


def zones_of(scene: Scene, ref: str) -> list[str]:
    """Names of the zones that contain `ref` (`<kind>:<name>`)."""
    return [fact.subject.split(':', 1)[1] for fact in scene.facts if fact.verb == 'contains' and fact.object == ref and fact.subject.startswith('room:')]


def contents(scene: Scene, zone: str) -> list[str]:
    """Refs (`<kind>:<name>`) a zone contains."""
    return [str(fact.object) for fact in scene.facts if fact.verb == 'contains' and fact.subject == f'room:{zone}']


def connections(scene: Scene) -> list[tuple[str, str, str]]:
    """Every (zone, zone, door name) a door joins."""
    return [(fact.subject.split(':', 1)[1], str(fact.object).split(':', 1)[1], str(fact.params.get('door', '')).split(':', 1)[-1]) for fact in scene.facts if fact.verb == 'connects' and fact.subject.startswith('room:')]


def describe(scene: Scene, zone: str) -> str:
    """The description of a zone from the KG, empty without one."""
    entity = scene.get_entity(f'room:{zone}')
    return str(entity.props.get('description', '')) if entity is not None else ''


_WORLD_KINDS = {'room': 'zone', 'door': 'door', 'elevator': 'elevator', 'stair': 'stair', 'obj': 'obj', 'light': 'light', 'sound': 'sound'}


def _union_uses(*scenes: Scene) -> list[str]:
    best: dict[str, str] = {}
    for scene in scenes:
        for use in scene.uses:
            name, _, version = use.partition('@')
            if name not in best or int(version or 0) > int(best[name].partition('@')[2] or 0):
                best[name] = use
    return list(best.values())


def merge_linked(scene: Scene | None, linked: Scene, world: World, renames: Mapping[str, str]) -> Scene:
    """`scene` plus the stated KG of a world this one links levels from, its ids renamed by `renames` (id there to id here, floors of levels not linked dropped), kept where they name what this world has, `scene` winning every conflict."""
    from arena_simulation_setup.kg import dump, load

    merged = Scene(_union_uses(linked)) if scene is None else load({**dump(scene), 'uses': _union_uses(scene, linked)})
    names = world.names()
    present = {kind: set(names[key]) for kind, key in _WORLD_KINDS.items()} | {'floor': set(world.level_ids)}

    @typing.overload
    def rename(value: str) -> str | None: ...

    @typing.overload
    def rename(value: Value | None) -> Value | None: ...

    def rename(value: Value | None) -> Value | None:
        if not isinstance(value, str) or not linked.vocab.is_entity_ref(value):
            return value
        if value in renames:
            return renames[value]
        return None if value.startswith('floor:') else value

    def kept(value: Value | None) -> bool:
        if value is None:
            return False
        if not isinstance(value, str) or not linked.vocab.is_entity_ref(value):
            return True
        kind, _, name = value.partition(':')
        entity = linked.get_entity(value)
        if kind == 'obj' and entity is not None and entity.props.get('placed') is False:
            return True
        return kind not in present or name in present[kind]

    def declare(*refs: Value | None) -> None:
        for ref in refs:
            if isinstance(ref, str) and linked.vocab.is_entity_ref(ref) and merged.get_entity(ref) is None:
                merged.add_entity(ref)

    for entity in linked.entities.values():
        ref = rename(entity.id)
        if ref is None or not kept(ref):
            continue
        existing = merged.get_entity(ref)
        merged.add_entity(ref, {key: value for key, value in entity.props.items() if existing is None or key not in existing.props})
    for fact in linked.stated():
        subject, obj = rename(fact.subject), rename(fact.object)
        params = {key: rename(value) for key, value in fact.params.items()}
        if subject is None or obj is None or not (kept(subject) and kept(obj) and all(kept(value) for value in params.values())):
            continue
        kept_params = {key: value for key, value in params.items() if value is not None}
        if merged.get_fact(subject, fact.verb, obj, kept_params) is None:
            declare(subject, obj, *kept_params.values())
            merged.add_fact(subject, fact.verb, obj, kept_params)
    for constraint in linked.constraints.values():
        scope = [rename(ref) for ref in constraint.scope]
        params = {key: rename(value) for key, value in constraint.params.items()}
        if constraint.id not in merged.constraints and all(kept(value) for value in [*scope, *params.values()]):
            declare(*scope, *params.values())
            merged.add_constraint(constraint.id, constraint.kind, [ref for ref in scope if ref is not None], {key: value for key, value in params.items() if value is not None}, constraint.hard)
    return merged


def own_kg(scene: Scene, linked: Scene | None) -> Scene:
    """`scene` without what `linked` contributes unchanged, the part a world writes to its own KG."""
    from arena_simulation_setup.kg import dump, load

    if linked is None:
        return scene
    own = load(dump(scene))
    for fact in linked.facts:
        mine = own.get_fact(fact.subject, fact.verb, fact.object, fact.params)
        if mine is not None and mine.params == fact.params and mine.source == fact.source:
            own.remove_fact(fact.subject, fact.verb, fact.object, fact.params)
    for constraint in linked.constraints.values():
        if own.get_constraint(constraint.id) == constraint:
            own.remove_constraint(constraint.id)
    referenced = {fact.subject for fact in own.facts} | {str(fact.object) for fact in own.facts} | {str(value) for fact in own.facts for value in fact.params.values()} | {ref for constraint in own.constraints.values() for ref in constraint.scope}
    for entity in linked.entities.values():
        mine = own.get_entity(entity.id)
        if mine is None:
            continue
        for key, value in entity.props.items():
            if mine.props.get(key) == value:
                del mine.props[key]
        if not mine.props and entity.id not in referenced:
            own.remove_entity(entity.id)
    return own
