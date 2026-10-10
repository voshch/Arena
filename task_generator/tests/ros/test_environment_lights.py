"""Lights resolve static entities by their authored names after the world manager prefixed them."""

import asyncio

import pytest
from arena_simulation_setup.shared import Light, Obstacle
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.World import LevelDescription
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

from task_generator.manager.environment_manager import authored_statics, stage_ambient, stage_ambient_conflict, world_lights
from task_generator.manager.world_manager.world_manager import WORLD_ENTITY_PREFIX


def _level(static_name: str) -> LevelDescription:
    desk = Obstacle(name=static_name, pose=Pose(Position(4.0, 2.0, 0.0), Orientation.from_yaw(0.0)), model=ObjectIdentifier('test_model'))
    lamp = Light(name='desk_lamp', fixture='downlight', entity_ref='desk', offset=Position(0.5, 0.0, 0.8), lumens=400.0)
    zone = LevelDescription.Zone(name='office', corners=[Position(0, 0), Position(8, 0), Position(8, 6), Position(0, 6)], lights=[lamp], entities=LevelDescription.Zone.WorldEntities(static=[desk]))
    return LevelDescription(zones=[zone])


def test_entity_ref_light_finds_a_prefixed_static_entity() -> None:
    level = _level(f'{WORLD_ENTITY_PREFIX}desk')
    with pytest.raises(ValueError, match='unknown'):
        asyncio.run(level.all_lights())
    (light,) = asyncio.run(authored_statics(level).all_lights())
    assert (light.position.x, light.position.y, light.position.z) == pytest.approx((4.5, 2.0, 0.8))


def test_authored_statics_leaves_the_level_untouched() -> None:
    level = _level(f'{WORLD_ENTITY_PREFIX}desk')
    authored = authored_statics(level)
    assert [entity.name for entity in authored.all_static_entities] == ['desk']
    assert [entity.name for entity in level.all_static_entities] == [f'{WORLD_ENTITY_PREFIX}desk']


def test_world_lights_name_their_owner_as_it_spawns() -> None:
    (light,) = asyncio.run(world_lights(_level(f'{WORLD_ENTITY_PREFIX}desk')))
    assert (light.name, light.owner) == ('desk_lamp', f'{WORLD_ENTITY_PREFIX}desk')


def _dome(lux: float, name: str = 'ambient') -> Light:
    return Light(name=name, fixture='dome', lux=lux)


def test_stage_ambient_keeps_dome_and_sun_without_their_names() -> None:
    lamp = Light(name='lamp', fixture='bulb', position=Position(1.0, 1.0, 1.0), lumens=800.0)
    assert stage_ambient([lamp, _dome(400.0, 'env_0/ambient/0')]) == stage_ambient([_dome(400.0, 'env_1/ambient/0')])
    assert [entry['fixture'] for entry in stage_ambient([lamp, _dome(400.0)])] == ['dome']


def test_stage_ambient_conflict_names_both_sides() -> None:
    assert stage_ambient_conflict('env_1', stage_ambient([_dome(400.0)]), stage_ambient([_dome(400.0)])) is None
    message = stage_ambient_conflict('env_1', stage_ambient([_dome(200.0)]), stage_ambient([_dome(400.0)]))
    assert message == 'dome and sun light the whole stage and env_0 sets them: env_1 renders under dome 400 lux, its world declares dome 200 lux'
    message = stage_ambient_conflict('env_2', [], stage_ambient([_dome(400.0)]))
    assert message.endswith('its world declares no dome or sun (default lighting)')
