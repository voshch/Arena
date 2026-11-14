"""MuJoCo welds the world object a closed gripper reaches to the hand until the next open, the robot holds it meanwhile, and an episode reset puts back every object carried since the last one."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_HAND = "env_0/tiago/gripper_grasping_frame"
_HAND_AT = (2.6, 1.1, 0.65)
_TIP_BELOW = 0.02
_FAR_AT = (9.0, 9.0, 0.75)
_CRATE_AT = (2.0, 1.0, 0.0)
_CRATE_BOX = [[0.2, 1.0], [-0.1, 0.3], [0.0, 0.6]]


def test_a_gripper_carries_the_object_it_closed_on_until_the_episode_resets(tmp_path: Path):
    pytest.importorskip("arena_mujoco")
    import mujoco
    import rclpy.node
    from arena_mujoco import context, control, hooks
    from arena_mujoco.scene import SceneStore
    from arena_mujoco.services.EditPrims import edit_prims_service
    from arena_mujoco.services.Grasp import attach_prim_service, detach_prim_service
    from arena_mujoco.services.SpawnPrims import spawn_prims_service
    import tf2_ros
    from arena_rclpy_mixins import ArenaMixinNode
    from geometry_msgs.msg import TransformStamped
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.spin import create_executor
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Obstacle, Orientation, Pose, Position
    from arena_runtime.sim.mujoco_simulator import MujocoSimulator

    bundle = tmp_path / "crate"
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": _CRATE_BOX}))
    crate = Obstacle(name="crate", model=ObjectIdentifier.inline(ObjectView(bundle), name="crate"), pose=Pose(position=Position(*_CRATE_AT), orientation=Orientation.identity()))
    tiago = SimpleNamespace(name="tiago", sim_path="env_0/Robots/tiago")

    def build_robot(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        base = spec.worldbody.add_body(name="env0_base", pos=list(_FAR_AT))
        base.add_freejoint().name = "env0_base_free"
        base.gravcomp = 1.0
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], mass=1.0)
        hand = base.add_body(name="env0_hand", pos=[0.0, 0.0, 0.0])
        hand.gravcomp = 1.0
        hand.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.1)
        finger = hand.add_body(name="env0_finger", pos=[0.0, 0.0, 0.0])
        finger.gravcomp = 1.0
        finger.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01], mass=0.1)
        finger.add_joint(name="env0_finger_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 1.0, 0.0])
        return base

    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    store.add_body(0, build_robot)
    store.recompile(0)
    wiring = control._Robot(0, ("finger_joint",), {}, "env0_base")
    wiring.mjcf_joints = {"finger_joint": "env0_finger_joint"}
    control._robots[(0, "tiago")] = wiring

    def crate_z() -> float:
        found = store.get_body_pose(0, "env_0/Obstacles/crate")
        assert found is not None
        return found[0][2]

    def crate_xy() -> tuple[float, float]:
        found = store.get_body_pose(0, "env_0/Obstacles/crate")
        assert found is not None
        return found[0][0], found[0][1]

    tf_buffer = tf2_ros.Buffer()

    def put_base(at: tuple[float, float, float]) -> None:
        tip = TransformStamped()
        tip.header.frame_id = "map"
        tip.child_frame_id = _HAND
        tip.transform.translation.x, tip.transform.translation.y, tip.transform.translation.z = at[0], at[1], at[2] - _TIP_BELOW
        tip.transform.rotation.w = 1.0
        tf_buffer.set_transform_static(tip, "test")
        model, data = store.get_model(0), store.get_data(0)
        assert model is not None and data is not None
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "env0_base_free")]
        data.qpos[adr : adr + 3] = at
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    async def main() -> None:
        executor = create_executor()
        server = rclpy.node.Node("mujoco_grasp_server")
        hooks.set_node(server)
        for service in (spawn_prims_service, edit_prims_service, attach_prim_service, detach_prim_service):
            service.create(server)
        node = ArenaMixinNode("mujoco_grasp_client", namespace="/mujoco_grasp_client")
        executor.add_node(server)
        executor.add_node(node)
        spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
        try:
            sim = MujocoSimulator(node=node, namespace=Namespace("/mujoco_grasp_client"), realizer=Realizer(), env_id=0)
            sim._robot_prims[tiago.sim_path] = "env0_base"
            sim._mechanism_tf_buffer = tf_buffer
            assert list(await asyncio.wait_for(sim.spawn_objects([crate]), timeout=30.0)) == [True]
            spawned_xy, spawned_z = crate_xy(), crate_z()
            assert sim.held_objects() == {}

            put_base(_FAR_AT)
            assert await asyncio.wait_for(sim.grasp_attach(tiago, "finger_joint", "crate", _HAND), timeout=30.0) is False
            assert sim.held_objects() == {}

            put_base(_HAND_AT)
            assert await asyncio.wait_for(sim.grasp_attach(tiago, "finger_joint", "crate", _HAND), timeout=30.0) is True
            assert sim.held_objects() == {"tiago": frozenset({"crate"})}

            put_base((0.0, 5.0, 1.5))
            store.step(5)
            assert crate_xy() == pytest.approx((spawned_xy[0] - _HAND_AT[0], spawned_xy[1] - _HAND_AT[1] + 5.0), abs=1e-3)
            assert crate_z() == pytest.approx(spawned_z - _HAND_AT[2] + 1.5, abs=1e-3)

            await asyncio.wait_for(sim.grasp_detach(tiago, "finger_joint"), timeout=30.0)
            assert sim.held_objects() == {}
            put_base(_FAR_AT)
            store.step(5)
            assert crate_xy() == pytest.approx((spawned_xy[0] - _HAND_AT[0], spawned_xy[1] - _HAND_AT[1] + 5.0), abs=1e-3)
            assert crate_z() == pytest.approx(spawned_z, abs=1e-3)

            put_base((0.0, 5.0, _HAND_AT[2]))
            assert await asyncio.wait_for(sim.grasp_attach(tiago, "finger_joint", "crate", _HAND), timeout=30.0) is True
            put_base((4.0, 4.0, 1.5))
            store.step(5)
            assert crate_xy() != pytest.approx(spawned_xy, abs=1e-3)
            assert await asyncio.wait_for(sim.before_reset_episode(), timeout=30.0) is True
            assert sim.held_objects() == {}
            assert crate_xy() == pytest.approx(spawned_xy, abs=1e-3)
            assert crate_z() == pytest.approx(spawned_z, abs=1e-3)

            put_base(_HAND_AT)
            assert await asyncio.wait_for(sim.grasp_attach(tiago, "finger_joint", "crate", _HAND), timeout=30.0) is True
            put_base((4.0, 4.0, 1.5))
            store.step(5)
            await asyncio.wait_for(sim.grasp_detach(tiago, "finger_joint"), timeout=30.0)
            put_base(_FAR_AT)
            store.step(5)
            assert crate_xy() != pytest.approx(spawned_xy, abs=1e-3)
            assert await asyncio.wait_for(sim.before_reset_episode(), timeout=30.0) is True
            assert crate_xy() == pytest.approx(spawned_xy, abs=1e-3)
            assert crate_z() == pytest.approx(spawned_z, abs=1e-3)
        finally:
            control._robots.pop((0, "tiago"), None)
            executor.shutdown()
            with contextlib.suppress(Exception):
                await spin
            executor.remove_node(node)
            executor.remove_node(server)
            node.destroy_node()
            server.destroy_node()

    asyncio.run(main())
