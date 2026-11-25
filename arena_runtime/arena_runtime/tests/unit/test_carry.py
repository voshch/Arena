"""A closed gripper carries the object within reach of its hand frame on simulators without a native attach, and the episode reset puts it back."""

from __future__ import annotations

import asyncio
import contextlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

HAND = "env_0/tiago/arm_tool_link"
CRATE_AT = (2.0, 1.0, 0.0)
CRATE_BOX = [[-0.2, 0.2], [-0.15, 0.15], [0.0, 0.4]]
TIAGO = SimpleNamespace(name="tiago", sim_path="env_0/Robots/tiago")


def _crate(root: Path):
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView

    from arena_runtime.shared import Obstacle, Orientation, Pose, Position

    bundle = root / "crate"
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": CRATE_BOX}))
    return Obstacle(name="world_crate", model=ObjectIdentifier.inline(ObjectView(bundle), name="crate"), pose=Pose(Position(*CRATE_AT), Orientation.identity()))


def _tf(sim):
    import tf2_ros

    sim._mechanism_tf_buffer = tf2_ros.Buffer()
    return sim._mechanism_tf_buffer


def _hand_at(buffer, x: float, y: float, z: float, yaw: float = 0.0) -> None:
    from geometry_msgs.msg import TransformStamped

    t = TransformStamped()
    t.header.frame_id = "map"
    t.child_frame_id = HAND
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
    t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    buffer.set_transform_static(t, "test")


def _fingertip(buffer, ahead: float) -> None:
    from geometry_msgs.msg import TransformStamped

    t = TransformStamped()
    t.header.frame_id = HAND
    t.child_frame_id = "env_0/tiago/gripper_grasping_frame"
    t.transform.translation.x = ahead
    t.transform.rotation.w = 1.0
    buffer.set_transform_static(t, "test")


def test_box_gap_measures_to_the_oriented_box_surface():
    from arena_runtime.shared import Orientation, Pose, Position
    from arena_runtime.sim._carry import box_gap

    pose = Pose(Position(1.0, 0.0, 0.0), Orientation.from_yaw(math.pi / 2))
    box = ((0.4, 0.2, 0.4), (0.0, 0.0, 0.2))
    assert box_gap(Position(1.0, 0.0, 0.2), pose, box) == 0.0
    assert box_gap(Position(1.0, 0.5, 0.2), pose, box) == pytest.approx(0.3)
    assert box_gap(Position(1.5, 0.0, 0.2), pose, box) == pytest.approx(0.4)


def test_a_hand_stopped_short_of_the_object_grasps_nothing(tmp_path: Path):
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.sim.dummy_simulator import DummySimulator

    crate = _crate(tmp_path)

    async def main() -> None:
        node = ArenaMixinNode("carry_short_unit", namespace="/carry_short_unit")
        sim = DummySimulator(node=node, namespace=Namespace("/carry_short_unit"), realizer=Realizer(), env_id=0)
        try:
            buffer = _tf(sim)
            assert list(await sim.spawn_objects([crate])) == [True]
            _hand_at(buffer, 2.3, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is False
            assert sim.held_objects() == {}
            _hand_at(buffer, 2.24, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
        finally:
            await sim.stop_mechanisms()
            node.destroy_node()

    asyncio.run(main())


def test_a_carried_object_follows_the_hand_until_released_and_goes_home_at_reset(tmp_path: Path):
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.spin import create_executor

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.sim._carry import follow
    from arena_runtime.sim.dummy_simulator import DummySimulator

    class Placing(DummySimulator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.placed: dict[str, tuple[tuple[float, float, float], float]] = {}

        async def obstacle_move(self, obstacles):
            for obstacle in obstacles:
                self.placed[obstacle.name] = (tuple(obstacle.pose.position), obstacle.pose.orientation.to_yaw())
            return await super().obstacle_move(obstacles)

    crate = _crate(tmp_path)

    async def main() -> None:
        executor = create_executor()
        node = ArenaMixinNode("carry_unit", namespace="/carry_unit")
        executor.add_node(node)
        spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
        sim = Placing(node=node, namespace=Namespace("/carry_unit"), realizer=Realizer(), env_id=0)
        try:
            buffer = _tf(sim)
            assert list(await sim.spawn_objects([crate])) == [True]
            assert sim.held_objects() == {}

            _hand_at(buffer, 3.0, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is False
            assert await sim.grasp_attach(TIAGO, "finger", "crate", "") is False
            _hand_at(buffer, 2.15, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            assert sim.held_objects() == {"tiago": frozenset({"crate"})}

            _hand_at(buffer, 0.0, 5.0, 1.2, math.pi / 2)
            await follow(sim)
            position, yaw = sim.placed["world_crate"]
            assert position == pytest.approx((0.0, 4.85, 1.0), abs=1e-6)
            assert yaw == pytest.approx(math.pi / 2)

            await sim.grasp_detach(TIAGO, "finger")
            assert sim.held_objects() == {}
            assert sim.placed["world_crate"][0] == pytest.approx((0.0, 4.85, 0.0), abs=1e-6)
            _hand_at(buffer, 3.0, 3.0, 1.2)
            await follow(sim)
            assert sim.placed["world_crate"][0] == pytest.approx((0.0, 4.85, 0.0), abs=1e-6)

            _hand_at(buffer, 0.0, 5.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            assert await sim.before_reset_episode() is True
            assert sim.held_objects() == {}
            position, yaw = sim.placed["world_crate"]
            assert position == pytest.approx(CRATE_AT)
            assert yaw == pytest.approx(0.0)
        finally:
            await sim.stop_mechanisms()
            executor.shutdown()
            with contextlib.suppress(Exception):
                await spin
            executor.remove_node(node)
            node.destroy_node()

    asyncio.run(main())


def test_a_welded_object_is_neither_lifted_nor_moved_by_the_tick_and_drops_to_the_floor_below_the_hand(tmp_path: Path):
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Orientation, Pose, Position
    from arena_runtime.sim._carry import follow
    from arena_runtime.sim.dummy_simulator import DummySimulator

    released_at = Pose(Position(4.0, 4.0, 0.9), Orientation.from_yaw(1.0))

    class Welding(DummySimulator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.lifted: list[str] = []
            self.welds: list[str] = []
            self.placed: dict[str, tuple[float, float, float]] = {}

        async def _carry_weld(self, carry) -> bool:
            self.welds.append(carry.obstacle.name)
            return True

        async def _carry_unweld(self, carry) -> Pose | None:
            self.welds.remove(carry.obstacle.name)
            return released_at

        async def _carry_lift(self, obstacles) -> None:
            self.lifted.extend(obstacle.name for obstacle in obstacles)

        async def obstacle_move(self, obstacles):
            for obstacle in obstacles:
                self.placed[obstacle.name] = tuple(obstacle.pose.position)
            return await super().obstacle_move(obstacles)

    crate = _crate(tmp_path)

    async def main() -> None:
        node = ArenaMixinNode("carry_weld_unit", namespace="/carry_weld_unit")
        sim = Welding(node=node, namespace=Namespace("/carry_weld_unit"), realizer=Realizer(), env_id=0)
        try:
            buffer = _tf(sim)
            assert list(await sim.spawn_objects([crate])) == [True]
            _hand_at(buffer, 2.15, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            assert sim.welds == ["world_crate"]
            assert sim.lifted == []

            _hand_at(buffer, 0.0, 5.0, 1.2)
            await follow(sim)
            assert "world_crate" not in sim.placed

            await sim.grasp_detach(TIAGO, "finger")
            assert sim.welds == []
            assert sim.placed["world_crate"] == pytest.approx((-0.15, 5.0, 0.0))
            assert sim.held_objects() == {}

            _hand_at(buffer, 4.0, 4.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is False
            _hand_at(buffer, 0.0, 5.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            assert await sim.before_reset_episode() is True
            assert sim.welds == []
            assert sim.placed["world_crate"] == pytest.approx(CRATE_AT)
        finally:
            await sim.stop_mechanisms()
            node.destroy_node()

    asyncio.run(main())


def test_a_finger_frame_below_the_hand_reaches_the_object_the_hand_frame_stops_short_of(tmp_path: Path):
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.sim.dummy_simulator import DummySimulator

    crate = _crate(tmp_path)

    async def main() -> None:
        node = ArenaMixinNode("carry_finger_unit", namespace="/carry_finger_unit")
        sim = DummySimulator(node=node, namespace=Namespace("/carry_finger_unit"), realizer=Realizer(), env_id=0)
        try:
            buffer = _tf(sim)
            assert list(await sim.spawn_objects([crate])) == [True]
            _hand_at(buffer, 1.5, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is False
            _fingertip(buffer, 0.3)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            assert sim.held_objects() == {"tiago": frozenset({"crate"})}
        finally:
            await sim.stop_mechanisms()
            node.destroy_node()

    asyncio.run(main())


def test_flatsim_takes_a_carried_outline_out_of_the_world_until_it_is_released(on_flatsim, tmp_path: Path):
    crate = _crate(tmp_path)
    name = "env_0/world_crate"

    async def body(sim, world):
        buffer = _tf(sim)
        try:
            assert list(await sim.spawn_objects([crate])) == [True]
            assert name in world.names

            _hand_at(buffer, 2.15, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            assert name not in world.names
            assert sim.held_objects() == {"tiago": frozenset({"crate"})}

            await sim.grasp_detach(TIAGO, "finger")
            assert name in world.names
            assert sim.held_objects() == {}

            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            assert name not in world.names
            assert await sim.before_reset_episode() is True
            assert name in world.names
            assert sim.held_objects() == {}
        finally:
            await sim.stop_mechanisms()

    on_flatsim(body)


def _table(root: Path, at: tuple[float, float, float], yaw: float = 0.0):
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView

    from arena_runtime.shared import Obstacle, Orientation, Pose, Position

    bundle = root / "table"
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.1, 1.1], [-0.4, 0.4], [0.0, 0.75]]}))
    return Obstacle(name="world_table", model=ObjectIdentifier.inline(ObjectView(bundle), name="table"), pose=Pose(Position(*at), Orientation.from_yaw(yaw)))


@pytest.mark.parametrize(("released_z", "rests_at"), [(0.9, 0.75), (0.76, 0.75), (0.5, 0.0)], ids=["above_the_top", "within_slack_of_the_top", "below_the_top"])
def test_a_released_object_rests_on_the_top_under_its_center_it_is_not_below(tmp_path: Path, released_z: float, rests_at: float):
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Pose
    from arena_runtime.sim.dummy_simulator import DummySimulator

    class Welding(DummySimulator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.placed: dict[str, Pose] = {}

        async def _carry_weld(self, carry) -> bool:
            return True

        async def obstacle_move(self, obstacles):
            for obstacle in obstacles:
                self.placed[obstacle.name] = obstacle.pose
            return await super().obstacle_move(obstacles)

    crate = _crate(tmp_path)
    table = _table(tmp_path, (4.3, 3.5, 0.0), yaw=math.pi / 2)

    async def main() -> None:
        node = ArenaMixinNode("carry_settle_unit", namespace="/carry_settle_unit")
        sim = Welding(node=node, namespace=Namespace("/carry_settle_unit"), realizer=Realizer(), env_id=0)
        try:
            buffer = _tf(sim)
            assert list(await sim.spawn_objects([crate, table])) == [True, True]
            _hand_at(buffer, 2.15, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            _hand_at(buffer, 4.0 + 0.15 * math.cos(0.3), 4.0 + 0.15 * math.sin(0.3), released_z + 0.2, 0.3)
            await sim.grasp_detach(TIAGO, "finger")
            pose = sim.placed["world_crate"]
            assert tuple(pose.position) == pytest.approx((4.0, 4.0, rests_at))
            assert pose.orientation.to_yaw() == pytest.approx(0.3)
        finally:
            await sim.stop_mechanisms()
            node.destroy_node()

    asyncio.run(main())


def test_a_released_object_beside_a_top_rests_on_the_robot_floor(tmp_path: Path):
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from geometry_msgs.msg import TransformStamped

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Pose
    from arena_runtime.sim.dummy_simulator import DummySimulator

    class Welding(DummySimulator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.placed: dict[str, Pose] = {}

        async def _carry_weld(self, carry) -> bool:
            return True

        async def obstacle_move(self, obstacles):
            for obstacle in obstacles:
                self.placed[obstacle.name] = obstacle.pose
            return await super().obstacle_move(obstacles)

    crate = _crate(tmp_path)
    table = _table(tmp_path, (3.5, 4.2, 3.0))

    async def main() -> None:
        node = ArenaMixinNode("carry_floor_unit", namespace="/carry_floor_unit")
        sim = Welding(node=node, namespace=Namespace("/carry_floor_unit"), realizer=Realizer(), env_id=0)
        try:
            buffer = _tf(sim)
            for parent, child, z in (("map", "env_0/tiago/odom", 3.0), ("env_0/tiago/odom", "env_0/tiago/base_link", 0.1)):
                t = TransformStamped()
                t.header.frame_id, t.child_frame_id = parent, child
                t.transform.translation.z = z
                t.transform.rotation.w = 1.0
                buffer.set_transform_static(t, "test")
            t = TransformStamped()
            t.header.frame_id, t.child_frame_id = "env_0/tiago/base_link", HAND
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = 2.15, 1.0, -2.9
            t.transform.rotation.w = 1.0
            buffer.set_transform_static(t, "test")
            assert list(await sim.spawn_objects([crate, table])) == [True, True]
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = 4.15, 2.5, 1.0
            buffer.set_transform_static(t, "test")
            await sim.grasp_detach(TIAGO, "finger")
            assert tuple(sim.placed["world_crate"].position) == pytest.approx((4.0, 2.5, 3.0))
        finally:
            await sim.stop_mechanisms()
            node.destroy_node()

    asyncio.run(main())


def test_a_welded_object_without_a_hand_frame_at_the_release_drops_below_where_the_simulator_reports_it(tmp_path: Path):
    import tf2_ros
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Orientation, Pose, Position
    from arena_runtime.sim.dummy_simulator import DummySimulator

    class Welding(DummySimulator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.placed: dict[str, Pose] = {}

        async def _carry_weld(self, carry) -> bool:
            return True

        async def _carry_unweld(self, carry) -> Pose | None:
            return Pose(Position(6.0, 7.0, 0.9), Orientation.identity())

        async def obstacle_move(self, obstacles):
            for obstacle in obstacles:
                self.placed[obstacle.name] = obstacle.pose
            return await super().obstacle_move(obstacles)

    crate = _crate(tmp_path)

    async def main() -> None:
        node = ArenaMixinNode("carry_reported_unit", namespace="/carry_reported_unit")
        sim = Welding(node=node, namespace=Namespace("/carry_reported_unit"), realizer=Realizer(), env_id=0)
        try:
            buffer = _tf(sim)
            assert list(await sim.spawn_objects([crate])) == [True]
            _hand_at(buffer, 2.15, 1.0, 0.2)
            assert await sim.grasp_attach(TIAGO, "finger", "crate", HAND) is True
            sim._mechanism_tf_buffer = tf2_ros.Buffer()
            await sim.grasp_detach(TIAGO, "finger")
            assert tuple(sim.placed["world_crate"].position) == pytest.approx((6.0, 7.0, 0.0))
        finally:
            await sim.stop_mechanisms()
            node.destroy_node()

    asyncio.run(main())
