from __future__ import annotations

import math

import pytest

PREFIX = "env_0/"


def _door(name: str, kind: str = "sliding"):
    from task_generator.shared import Door, Position

    return Door(name=name, start=Position(0.0, 0.0, 0.0), end=Position(2.0, 0.0, 0.0), kind=kind, width=0.1, height=2.0)


def _elevator(name: str = "lift"):
    from task_generator.shared import Elevator, Position

    return Elevator(name=name, position=Position(5.0, 5.0, 0.0), size=[2.0, 2.0, 2.5], door_side="+x", destinations=["other"])


def _ours(world) -> set[str]:
    return {name.removeprefix(PREFIX) for name in world.names if name.startswith(PREFIX)}


def test_spawned_doors_become_boxes_with_runtimes_and_a_running_loop(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_doors([_door("slide"), _door("swing", kind="hinged")])
        assert _ours(world) == {"slide", "swing"}
        assert sim._box_sizes == {"slide": (2.0, 0.1, 2.0), "swing": (2.0, 0.1, 2.0)}
        assert sim._door_primitives == {"slide": ["slide"], "swing": ["swing"]}
        assert {name: rt.effective_kind for name, rt in sim._door_runtime.items()} == {"slide": "sliding", "swing": "hinged"}
        swing = sim._door_runtime["swing"]
        assert (swing.open_pose.position.x, swing.open_pose.position.y, swing.open_pose.position.z) == pytest.approx((0.0, 1.0, swing.closed_pose.position.z))
        assert sim._mechanism_loop_task is not None and not sim._mechanism_loop_task.done()

    on_flatsim(body)


def test_spawning_no_doors_starts_no_loop(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_doors([])
        assert await sim.spawn_elevators([])
        assert sim._mechanism_loop_task is None
        assert _ours(world) == set()

    on_flatsim(body)


def test_removed_doors_leave_the_world_and_their_runtimes(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_doors([_door("slide"), _door("keep")])
        assert await sim.remove_doors(["slide", "never_spawned"])
        assert _ours(world) == {"keep"}
        assert set(sim._door_runtime) == {"keep"}
        assert set(sim._door_primitives) == {"keep"}

    on_flatsim(body)


def test_removing_a_door_whose_box_is_already_gone_reports_failure_and_forgets_it(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_doors([_door("slide")])
        assert await sim.delete_box("slide")
        assert not await sim.remove_doors(["slide"])
        assert sim._door_runtime == {}
        assert sim._door_primitives == {}

    on_flatsim(body)


def test_spawned_elevator_builds_walls_floor_and_a_sliding_cabin_door(on_flatsim):
    from arena_runtime.sim._mechanism_shim import _elevator_synthesized_door, _elevator_wall_geometries

    async def body(sim, world):
        elevator = _elevator()
        assert await sim.spawn_elevators([elevator])
        walls = {f"lift/{suffix}": size for suffix, size, _ in _elevator_wall_geometries(elevator)}
        assert set(walls) == {"lift/back", "lift/side_pos", "lift/side_neg", "lift/floor"}
        assert _ours(world) == {*walls, "lift/door"}
        assert {name: sim._box_sizes[name] for name in walls} == walls
        door = _elevator_synthesized_door(elevator)
        assert sim._box_sizes["lift/door"] == pytest.approx((math.dist((door.start.x, door.start.y), (door.end.x, door.end.y)), door.width, 2.5))
        assert sim._elevator_primitives == {"lift": ["lift/back", "lift/side_pos", "lift/side_neg", "lift/floor"]}
        assert sim._elevator_doors == {"lift": "lift/door"}
        runtime = sim._elevator_runtime["lift"]
        assert (runtime.door_name, runtime.destination_name) == ("lift/door", "other")
        assert sim._door_runtime["lift/door"].effective_kind == "sliding"

    on_flatsim(body)


def test_removed_elevator_takes_its_walls_and_cabin_door_along(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_elevators([_elevator("lift"), _elevator("stay")])
        assert await sim.remove_elevators(["lift", "never_spawned"])
        assert _ours(world) == {"stay/back", "stay/side_pos", "stay/side_neg", "stay/floor", "stay/door"}
        assert set(sim._elevator_runtime) == set(sim._elevator_primitives) == set(sim._elevator_doors) == {"stay"}
        assert set(sim._door_runtime) == {"stay/door"}

    on_flatsim(body)


def test_removing_an_elevator_whose_walls_are_gone_reports_failure(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_elevators([_elevator("lift")])
        assert await sim.delete_box("lift/back")
        assert not await sim.remove_elevators(["lift"])
        assert _ours(world) == set()
        assert sim._elevator_runtime == {}
        assert sim._door_runtime == {}

    on_flatsim(body)


def test_removing_an_elevator_whose_cabin_door_is_gone_reports_failure(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_elevators([_elevator("lift")])
        assert await sim.delete_box("lift/door")
        assert not await sim.remove_elevators(["lift"])
        assert _ours(world) == set()

    on_flatsim(body)


def test_remove_mechanisms_clears_doors_and_elevators(on_flatsim):
    async def body(sim, world):
        assert await sim.spawn_doors([_door("slide")])
        assert await sim.spawn_elevators([_elevator("lift")])
        assert await sim.remove_mechanisms()
        assert _ours(world) == set()
        assert sim._door_runtime == {} and sim._elevator_runtime == {}

    on_flatsim(body)


def _stairs():
    from task_generator.shared import Position

    from arena_simulation_setup.shared import Stairs

    return [
        Stairs(name="esc_low", kind="escalator", position=Position(10.3, 5.0, 0.0), size=[2.6, 1.0], mouth_side="-x", to="esc_high"),
        Stairs(name="esc_high", kind="escalator", position=Position(23.7, 5.0, 0.0), size=[2.6, 1.0], mouth_side="+x", top=True),
    ]


def test_stairs_become_boxes_in_the_plane_with_runtimes_and_semantics(on_flatsim):
    from arena_runtime.sim._mechanism_shim import stair_boxes

    async def body(sim, world):
        stairs = _stairs()
        assert await sim.spawn_stairs(stairs)
        names = {prim.name for stair in stairs for prim in stair_boxes(stair)}
        assert _ours(world) == names
        assert set(sim._box_sizes) == names
        assert set(sim._stair_runtime) == {"esc_low", "esc_high"}
        assert sim._stair_runtime["esc_low"].partner == "esc_high"
        assert {snap.entity: snap.kind for snap in sim.semantics_snapshot()} == {"esc_low": "escalator", "esc_high": "escalator"}
        assert sim._mechanism_loop_task is not None and not sim._mechanism_loop_task.done()
        assert await sim.remove_mechanisms()
        assert _ours(world) == set()
        assert sim._stair_runtime == {} and sim._stair_primitives == {}
        assert sim.semantics_snapshot() == []

    on_flatsim(body)
