"""A MuJoCo obstacle spawned as its annotated box keeps the box center offset when it moves."""

from __future__ import annotations

import asyncio
import contextlib
import math
from pathlib import Path

import pytest
import yaml


def test_moved_box_obstacle_keeps_its_bbox_center_offset(tmp_path: Path):
    pytest.importorskip("arena_mujoco")
    import rclpy.node
    from arena_mujoco import context, hooks
    from arena_mujoco.scene import SceneStore
    from arena_mujoco.services.EditPrims import edit_prims_service
    from arena_mujoco.services.SpawnPrims import spawn_prims_service
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.spin import create_executor
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Obstacle, Orientation, Pose, Position

    from arena_runtime.sim._interface import _BOX_FLOOR_CLEARANCE, offset_pose
    from arena_runtime.sim.mujoco_simulator import MujocoSimulator

    bundle = tmp_path / "crate"
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[0.2, 1.0], [-0.1, 0.3], [0.0, 0.6]]}))
    crate = Obstacle(name="crate", model=ObjectIdentifier.inline(ObjectView(bundle), name="crate"), pose=Pose(position=Position(2.0, 1.0, 0.0), orientation=Orientation.from_yaw(math.pi / 2)))
    moved = Obstacle(name="crate", model=crate.model, pose=Pose(position=Position(-1.0, 4.0, 0.0), orientation=Orientation.from_yaw(math.pi)))
    store = SceneStore()
    context.set_scene_store(store)

    def body_pose() -> tuple[tuple[float, ...], float]:
        found = store.get_body_pose(0, "env_0/Obstacles/crate")
        assert found is not None
        (x, y, z), (w, qx, qy, qz) = found
        return (x, y, z), Orientation(w=w, x=qx, y=qy, z=qz).to_yaw()

    def expected(pose: Pose) -> tuple[tuple[float, ...], float]:
        placed = offset_pose(pose, (0.6, 0.1, 0.3 + _BOX_FLOOR_CLEARANCE))
        return (placed.position.x, placed.position.y, placed.position.z), pose.orientation.to_yaw()

    async def main() -> None:
        executor = create_executor()
        server = rclpy.node.Node("mujoco_box_obstacles")
        hooks.set_node(server)
        spawn_prims_service.create(server)
        edit_prims_service.create(server)
        node = ArenaMixinNode("mujoco_box_client", namespace="/mujoco_box_client")
        executor.add_node(server)
        executor.add_node(node)
        spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
        try:
            sim = MujocoSimulator(node=node, namespace=Namespace("/mujoco_box_client"), realizer=Realizer(), env_id=0)
            assert list(await asyncio.wait_for(sim.obstacle_spawn([crate]), timeout=30.0)) == [True]
            position, yaw = body_pose()
            want_position, want_yaw = expected(crate.pose)
            assert position == pytest.approx(want_position)
            assert abs(math.remainder(yaw - want_yaw, 2 * math.pi)) < 1e-6
            assert list(await asyncio.wait_for(sim.obstacle_move([moved]), timeout=30.0)) == [True]
            position, yaw = body_pose()
            want_position, want_yaw = expected(moved.pose)
            assert position == pytest.approx(want_position)
            assert abs(math.remainder(yaw - want_yaw, 2 * math.pi)) < 1e-6
        finally:
            executor.shutdown()
            with contextlib.suppress(Exception):
                await spin
            executor.remove_node(node)
            executor.remove_node(server)
            node.destroy_node()
            server.destroy_node()

    asyncio.run(main())
