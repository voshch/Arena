from __future__ import annotations

import asyncio
import threading
import uuid

import pytest
import shapely


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("interactive_markers")
    pytest.importorskip("task_generator_msgs.srv")


def _spin_in_background(rclpy, node, stop: threading.Event) -> threading.Thread:
    def _spin() -> None:
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread


def test_known_static_obstacle_respawn_moves_instead_of_spawning(rclpy_context):
    import attrs
    import rclpy
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator

    from task_generator.manager.realizer import Realizer
    from task_generator.shared import Obstacle, Orientation, Pose, Position
    from task_generator.simulators.acoustics.noop import NoopAcousticsSimulator
    from task_generator.simulators.human.noop import NoopHumanSimulator

    class _Node(ServiceNamespace, AsyncNode):
        pass

    class _RecordingSim(DummySimulator):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.spawned: list[str] = []
            self.moved: list[str] = []

        async def obstacle_spawn(self, obstacles):
            self.spawned.extend(o.name for o in obstacles)
            return await super().obstacle_spawn(obstacles)

        async def obstacle_move(self, obstacles):
            self.moved.extend(o.name for o in obstacles)
            return await super().obstacle_move(obstacles)

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> tuple[list[str], list[str], float]:
        node = _Node(f"interactive_move_{suffix}", namespace=str(ns))
        stop = threading.Event()
        thread = _spin_in_background(rclpy, node, stop)
        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=""))
            simulator = _RecordingSim(node=node, namespace=ns, realizer=realizer)
            acoustics = NoopAcousticsSimulator(node=node, namespace=ns)
            human = NoopHumanSimulator(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            crate = Obstacle(name="crate", model="box", pose=Pose(Position(1.0, 1.0), orientation=Orientation.from_yaw(0.0)))
            await human.spawn_obstacles([crate])
            await human.spawn_obstacles([attrs.evolve(crate, pose=Pose(Position(2.0, 3.0), orientation=Orientation.from_yaw(0.0)))])
            return simulator.spawned, simulator.moved, human._known_obstacles.get("crate").obstacle.pose.position.x
        finally:
            stop.set()
            thread.join(timeout=1.0)
            node.destroy_node()

    spawned, moved, x = asyncio.run(main())
    assert spawned.count("crate") <= 1
    assert moved == ["crate"]
    assert x == 2.0


def test_restamped_static_footprint_leaves_no_old_cells_after_sync():
    from task_generator.manager.collision_grid import Cell, CollisionGrid
    from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from task_generator.shared import Position

    import numpy as np
    from arena_rclpy_mixins.Time import Time

    grid_cells = np.full((6, 6), WorldOccupancy.EMPTY, dtype=np.uint8)
    world_map = WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid_cells)), origin=Position(x=0.0, y=0.0), resolution=1.0, time=Time(-1, 0))
    grid = CollisionGrid.build(world_map, origin=Position(x=0.0, y=0.0), walls=())

    grid.stamp("crate", shapely.box(0.0, 0.0, 1.0, 1.0))
    grid.stamp("crate", shapely.box(4.0, 4.0, 5.0, 5.0))
    assert grid.hit(shapely.box(0.1, 0.1, 0.9, 0.9))[0] is Cell.STATIC
    grid.sync(["crate"])
    assert grid.hit(shapely.box(0.1, 0.1, 0.9, 0.9)) is None
    assert grid.hit(shapely.box(4.1, 4.1, 4.9, 4.9)) == (Cell.STATIC, "crate")
