from __future__ import annotations

import asyncio
import math
import threading
import uuid

import pytest
import yaml

_spin_in_background = pytest.importorskip("arena_rclpy_mixins.testing").spin_in_background


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("task_generator_msgs.srv")


@pytest.fixture
def lid_box(tmp_path):
    """A box whose lid hinges up about the back edge, one link bundle per link."""
    from arena_simulation_setup.tree import DynamicPaths

    name = f"LidBox{uuid.uuid4().hex[:8]}"
    bundle = tmp_path / "Storage" / "Object" / name
    for link, box in (("base", [[-0.3, 0.3], [-0.2, 0.2], [0.0, 0.4]]), ("lid", [[-0.3, 0.3], [-0.2, 0.2], [0.4, 0.42]])):
        (bundle / "links" / link).mkdir(parents=True)
        (bundle / "links" / link / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    articulation = {
        "links": {"base": ["base"], "lid": ["lid"]},
        "joints": [{"name": "lid", "type": "revolute", "parent": "base", "child": "lid", "origin": [0.0, 0.2, 0.42], "axis": [-1, 0, 0], "limit": [0.0, 1.5]}],
        "states": {"closed": {"lid": 0.0}, "open": {"lid": 1.5}},
    }
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.3, 0.3], [-0.2, 0.2], [0.0, 0.42]], "articulation": articulation}))
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield f"Storage/{name}"
    DynamicPaths.ARENA.path = previous


def test_the_human_simulator_spawns_replaces_and_removes_an_articulated_obstacle_by_its_links(rclpy_context, lid_box):
    import attrs
    import rclpy
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator

    from task_generator.manager.realizer import Realizer
    from task_generator.shared import Obstacle, Pose, Position
    from task_generator.simulators.acoustics.noop import NoopAcousticsSimulator
    from task_generator.simulators.human import ObstacleLayer
    from task_generator.simulators.human.noop import NoopHumanSimulator

    class _Node(ServiceNamespace, AsyncNode):
        pass

    class _RecordingSim(DummySimulator):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.calls: list[tuple[str, list[Obstacle]]] = []

        async def obstacle_spawn(self, obstacles):
            self.calls.append(("spawn", list(obstacles)))
            return await super().obstacle_spawn(obstacles)

        async def obstacle_move(self, obstacles):
            self.calls.append(("move", list(obstacles)))
            return await super().obstacle_move(obstacles)

        async def obstacle_delete(self, obstacles):
            self.calls.append(("delete", list(obstacles)))
            return await super().obstacle_delete(obstacles)

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> list[tuple[str, list[Obstacle]]]:
        node = _Node(f"articulated_funnel_{suffix}", namespace=str(ns))
        stop = threading.Event()
        thread = _spin_in_background(rclpy, node, stop)
        simulator = None
        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))
            simulator = _RecordingSim(node=node, namespace=ns, realizer=realizer)
            acoustics = NoopAcousticsSimulator(node=node, namespace=ns)
            human = NoopHumanSimulator(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            box = realizer.realize(Obstacle(name="chest", model=lid_box, pose=Pose(Position(1.0, 1.0))))
            await human.spawn_obstacles([box])
            await human.spawn_obstacles([attrs.evolve(box, pose=Pose(Position(2.0, 3.0)), joints="open")])
            await human.remove_obstacles(ObstacleLayer.INUSE)
            return simulator.calls
        finally:
            if simulator is not None:
                await simulator.stop_mechanisms()
            stop.set()
            thread.join(timeout=1.0)
            node.destroy_node()

    calls = asyncio.run(main())
    assert [(verb, [obstacle.name for obstacle in obstacles]) for verb, obstacles in calls] == [
        ("spawn", ["chest__base", "chest__lid"]),
        ("move", ["chest__base", "chest__lid"]),
        ("delete", ["chest__base", "chest__lid"]),
    ]
    lid = calls[1][1][1]
    hinge_y, hinge_z = 0.2, 0.42
    swung_y = hinge_y * math.cos(1.5) + hinge_z * math.sin(1.5)
    swung_z = hinge_z * math.cos(1.5) - hinge_y * math.sin(1.5)
    assert (lid.pose.position.x, lid.pose.position.y, lid.pose.position.z) == pytest.approx((2.0, 3.0 + hinge_y - swung_y, hinge_z - swung_z))
    assert lid.sim_path == "env_0/chest__lid"
