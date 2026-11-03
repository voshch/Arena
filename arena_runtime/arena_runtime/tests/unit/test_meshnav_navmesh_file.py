from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from arena_simulation_setup.shared.navmesh import NavLevel, build_navmesh, navmesh_levels
from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.utils.geometry import Position

from arena_runtime.env.world_manager.mechanism_map import static_shut_doors
from arena_runtime.env.adapters.mobile.meshnav import navmesh_file, separated

WORLDS = Path(get_package_share_directory("arena_simulation_setup")) / "worlds"


def _stairs_demo() -> tuple[NavLevel, ...]:
    world = WorldView(WORLDS / "stairs_demo").load()
    return tuple(navmesh_levels(world, world.stack_offsets()))


def test_the_same_world_reuses_one_mesh_file(tmp_path):
    levels = _stairs_demo()
    first = navmesh_file(build_navmesh(levels, 0.1), tmp_path)
    written = first.stat().st_mtime_ns
    assert navmesh_file(build_navmesh(levels, 0.1), tmp_path) == first
    assert first.stat().st_mtime_ns == written
    assert [path.name for path in tmp_path.iterdir()] == [first.name]
    assert first.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")


def test_a_finer_resolution_gets_its_own_mesh_file(tmp_path):
    levels = _stairs_demo()
    assert navmesh_file(build_navmesh(levels, 0.1), tmp_path) != navmesh_file(build_navmesh(levels, 0.05), tmp_path)
    assert len(list(tmp_path.iterdir())) == 2


def test_each_robot_step_limit_gets_its_own_mesh_file(tmp_path):
    levels = _stairs_demo()
    flat = navmesh_file(build_navmesh(levels, 0.1, climb=0.1), tmp_path)
    climber = navmesh_file(build_navmesh(levels, 0.1, climb=0.2), tmp_path)
    assert flat != climber
    assert navmesh_file(build_navmesh(levels, 0.1, climb=0.2), tmp_path) == climber
    assert len(list(tmp_path.iterdir())) == 2


def test_points_on_levels_the_mesh_does_not_join_are_separated():
    levels = _stairs_demo()
    lobby, gallery = Position(15.0, 8.0, 0.0), Position(15.0, 8.0, 3.0)
    joined = build_navmesh(levels, 0.1)
    assert not separated(joined, joined.components(), lobby, gallery)
    floors_only = build_navmesh(tuple(NavLevel(level=level.level, floors=level.floors, walls=level.walls, stairs=(), doors=level.doors) for level in levels), 0.1)
    assert separated(floors_only, floors_only.components(), lobby, gallery)


def test_a_point_off_the_mesh_is_never_separated():
    levels = tuple(NavLevel(level=level.level, floors=level.floors, walls=level.walls, stairs=(), doors=level.doors) for level in _stairs_demo())
    mesh = build_navmesh(levels, 0.1)
    assert not separated(mesh, mesh.components(), Position(15.0, 8.0, 0.0), Position(15.0, 8.0, 6.0))


def test_static_shut_doors_are_those_nothing_opens_for_the_robot():
    world = WorldView(WORLDS / "mechanisms_doors").load()
    doors = world.levels["0"].all_doors
    assert static_shut_doors(doors, ["jackal"]) == frozenset({"door_dz"})


def test_a_locked_gate_is_shut_only_for_robots_it_does_not_authorize():
    from arena_simulation_setup.shared.world import Door

    gate = Door.parse({"name": "vault", "span": [[0.0, 0.0], [1.0, 0.0]], "semantics": [{"preset": "gate", "params": {"authorized": ["jackal"], "locked": True}}]})
    unlocking = Door.parse({"name": "lobby", "span": [[0.0, 0.0], [1.0, 0.0]], "semantics": [{"preset": "gate", "params": {"authorized": [], "locked": True, "unlock_on": "alarm"}}]})
    assert static_shut_doors([gate, unlocking], ["burger", "env0/burger"]) == frozenset({"vault"})
    assert static_shut_doors([gate, unlocking], ["jackal"]) == frozenset()
