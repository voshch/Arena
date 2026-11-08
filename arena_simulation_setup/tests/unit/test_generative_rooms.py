"""Rooms generator: a grid of rooms joined by doors along a spanning tree from the entrance."""

import random

import pydantic
import pytest
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.rooms import ENTRANCE, WorldGeneratorRooms
from arena_simulation_setup.tree.World import World, connections
from arena_simulation_setup.tree.World.membership import derive

SEEDS = range(8)


def build(seed: int, **config: object) -> tuple[WorldGeneratorRooms, World]:
    generator = WorldGeneratorRooms({"rows": 3, "cols": 4, **config}, random.Random(seed))
    return generator, generator.compute()


def door_depths(world: World) -> dict[str, int]:
    """Doors from the entrance to every zone, walking the door graph the world's semantics derive."""
    graph: dict[str, set[str]] = {}
    for a, b, _ in connections(derive(world)):
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    depths = {ENTRANCE: 0}
    frontier = [ENTRANCE]
    while frontier:
        nxt: list[str] = []
        for zone in frontier:
            for other in graph.get(zone, ()):
                if other not in depths:
                    depths[other] = depths[zone] + 1
                    nxt.append(other)
        frontier = nxt
    return depths


def test_the_generator_is_registered_and_runs_on_its_defaults():
    assert WorldGeneratorType.ROOMS in WorldGenerator.available()
    generator = WorldGenerator(WorldGeneratorType.ROOMS, {}, 0)
    assert len(generator.compute().all_levels[0].zones) == 12
    assert generator.layout() is not None


@pytest.mark.parametrize("seed", SEEDS)
def test_a_rooms_world_is_one_component(seed: int):
    generator, _ = build(seed)
    assert generator.diagnostics is not None
    assert generator.diagnostics.components == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_every_room_is_reachable_from_the_entrance_through_doors(seed: int):
    _, world = build(seed)
    assert set(door_depths(world)) == {zone.name for zone in world.all_levels[0].zones}


@pytest.mark.parametrize("seed", SEEDS)
def test_no_room_lies_more_doors_deep_than_max_depth(seed: int):
    _, world = build(seed, max_depth=5)
    assert max(door_depths(world).values()) <= 5


def test_a_deep_tree_grows_longer_chains_than_a_shallow_one():
    deep = sum(max(build(seed, deep=1.0)[0].depths.values()) for seed in SEEDS)
    shallow = sum(max(build(seed, deep=0.0)[0].depths.values()) for seed in SEEDS)
    assert deep > shallow


@pytest.mark.parametrize("loops", [0, 1, 3])
def test_each_loop_adds_one_door_beyond_the_tree(loops: int):
    _, world = build(0, loops=loops)
    assert len(world.all_levels[0].doors) == 12 - 1 + loops


def test_a_seed_repeats():
    first, second = build(3, loops=2)[1].all_levels[0], build(3, loops=2)[1].all_levels[0]
    assert [(door.name, door.start, door.end) for door in first.doors] == [(door.name, door.start, door.end) for door in second.doors]
    assert [zone.polygon for zone in first.zones] == [zone.polygon for zone in second.zones]


def test_rooms_too_deep_for_max_depth_fail_the_world():
    with pytest.raises(ValueError, match="doors of an entrance"):
        build(0, max_depth=1)


def test_a_door_wider_than_a_room_side_is_refused():
    with pytest.raises(pydantic.ValidationError, match="does not fit"):
        WorldGeneratorRooms({"room_min": 2.0, "room_max": 3.0, "door_width": 2.0}, random.Random(0))
