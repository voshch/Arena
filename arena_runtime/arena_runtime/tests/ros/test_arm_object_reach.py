from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

try:
    import rclpy.time
    import tf2_ros
    from arena_robots_msgs.action import ReachPose
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
    from geometry_msgs.msg import TransformStamped

    from arena_runtime.env.adapters.arm.moveit import MoveItArmAdapter, _carried, _locate
    from arena_runtime.env.adapters.arm.scene import StaticBox, object_id
    from arena_runtime.env.request import ReachPhase
    from arena_runtime.humans.utils import ObstacleLayer
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")


def _robot() -> SimpleNamespace:
    buffer = tf2_ros.Buffer()
    tf = TransformStamped()
    tf.header.frame_id = "map"
    tf.child_frame_id = "tiago/base_link"
    tf.transform.translation.x = 1.0
    tf.transform.rotation.w = 1.0
    buffer.set_transform_static(tf, "test")
    return SimpleNamespace(node=SimpleNamespace(tf_buffer=buffer, sim_time=rclpy.time.Time()), base_frame="tiago/base_link")


@pytest.mark.parametrize("motion", ["free", "linear"])
def test_a_reach_in_an_object_frame_may_touch_that_object(motion: str) -> None:
    box = StaticBox(name="box", pose=Pose(position=Position(3.0, 0.0, 0.5), orientation=Orientation.identity()), scale=None, extent=((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1)))
    phase = ReachPhase(target=Pose(position=Position(-0.3, 0.0, 0.05), orientation=Orientation.identity()), frame="box", motion=motion)
    goal = ReachPose.Goal()
    MoveItArmAdapter._place_in_entity(MoveItArmAdapter.__new__(MoveItArmAdapter), goal, phase, _robot(), box, None)
    assert list(goal.contact) == [object_id("box")]
    assert goal.target.header.frame_id == "tiago/base_link"
    assert (goal.target.pose.position.x, goal.target.pose.position.y, goal.target.pose.position.z) == pytest.approx((1.7, 0.0, 0.55))


def test_a_world_object_is_located_by_its_bare_name() -> None:
    box = StaticBox(name="world_box", pose=Pose(position=Position(3.0, 0.0, 0.5), orientation=Orientation.identity()), scale=None, extent=((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1)))
    robot = SimpleNamespace(node=SimpleNamespace(_resolve_semantic_entity=lambda name: f"env_0/{name}"))
    assert _locate("box", {"env_0/world_box": box}, robot) == (box, None)
    assert _locate("crate", {"env_0/world_box": box}, robot) is None


def test_a_carried_object_is_posed_in_the_hand_frame_of_its_carry_record() -> None:
    box = StaticBox(name="world_box", pose=Pose(position=Position(3.0, 0.0, 0.5), orientation=Orientation.identity()), scale=None, extent=((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1)))
    crate = StaticBox(name="crate", pose=Pose(position=Position(5.0, 0.0, 0.5), orientation=Orientation.identity()), scale=None, extent=((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1)))
    entities = {"env_0/world_box": box, "env_0/crate": crate}
    offset = Pose(position=Position(0.0, 0.0, 0.36), orientation=Orientation.identity())

    def robot(carries: dict[str, dict[str, tuple[str, Pose]]]) -> SimpleNamespace:
        return SimpleNamespace(
            robot=SimpleNamespace(name="tiago"),
            environment_manager=SimpleNamespace(carry_offsets=lambda name: carries.get(name, {})),
            node=SimpleNamespace(_resolve_semantic_entity=lambda name: f"env_0/{name}"),
        )

    assert _carried(robot({"tiago": {"box": ("tiago/arm_tool_link", offset)}}), "tiago/arm_tool_link", entities) == [(box, offset)]
    assert _carried(robot({"tiago": {"box": ("tiago/other_arm_tool_link", offset)}}), "tiago/arm_tool_link", entities) == []
    assert _carried(robot({"jackal": {"crate": ("jackal/arm_tool_link", offset)}}), "tiago/arm_tool_link", entities) == []
    assert _carried(robot({"tiago": {"ghost": ("tiago/arm_tool_link", offset)}}), "tiago/arm_tool_link", entities) == []
    assert _carried(robot({}), "tiago/arm_tool_link", entities) == []


def test_a_static_object_the_simulator_moved_is_in_the_scene_where_it_now_rests() -> None:
    view = SimpleNamespace(extent=((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1)))

    async def resolve() -> SimpleNamespace:
        return view

    def obstacle(name: str, x: float) -> SimpleNamespace:
        return SimpleNamespace(name=name, pose=Pose(position=Position(x, 0.0, 0.5), orientation=Orientation.identity()), scale=None, model=SimpleNamespace(resolve=resolve))

    known = {name: SimpleNamespace(obstacle=obstacle(name, x), spawned=True, layer=ObstacleLayer.WORLD) for name, x in (("world_box", 3.0), ("crate", 5.0))}
    simulator = SimpleNamespace(_articulated={}, _carry_objects={"world_box": obstacle("world_box", 9.0)})
    node = SimpleNamespace(_simulator=simulator, _human_simulator=SimpleNamespace(_known_obstacles=known), _realizer=SimpleNamespace(prefix=lambda name: f"env_0/{name}"))
    entities = asyncio.run(MoveItArmAdapter._episode_entities(MoveItArmAdapter.__new__(MoveItArmAdapter), SimpleNamespace(node=node)))
    assert {name: entity.pose.position.x for name, entity in entities.items()} == {"env_0/world_box": 9.0, "env_0/crate": 5.0}
