from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("tf2_ros")
    pytest.importorskip("arena_people_msgs.msg")


def _base_at(frame: str, x: float, y: float, stamp_ms: int):
    from geometry_msgs.msg import TransformStamped

    t = TransformStamped()
    t.header.frame_id = "map"
    t.header.stamp.sec = stamp_ms // 1000
    t.header.stamp.nanosec = stamp_ms % 1000 * 1_000_000
    t.child_frame_id = frame
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.w = 1.0
    return t


def _robot(x: float, y: float, name: str = "rob"):
    from arena_robots.Robot import RobotIdentifier
    from task_generator.shared import Pose, Position, Robot

    return Robot(name=name, pose=Pose(Position(x=x, y=y)), model=RobotIdentifier.parse("jackal"), adapters={"mobile": "nav2"}, extra={})


def _with_human(scenario: Callable[..., Awaitable[None]], backend: str = "noop", offset: tuple[float, float] = (0.0, 0.0)) -> None:
    import tf2_ros
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from task_generator.manager.realizer import Realizer
    from task_generator.simulators.acoustics.noop import NoopAcousticsSimulator
    from task_generator.simulators.human.noop import NoopHumanSimulator

    if backend == "arena":
        pytest.importorskip("arena_humansim_msgs.msg")
        from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator

        class Human(ArenaHumanSimulator):
            @classmethod
            def _register_task_modes(cls) -> None:
                pass

    else:
        Human = NoopHumanSimulator

    class _Node(ServiceNamespace, AsyncNode):
        pass

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> None:
        node = _Node(f"robot_tracking_{suffix}", namespace=str(ns))
        try:
            node.tf_buffer = tf2_ros.Buffer()
            realizer = Realizer(Realizer._Configuration(x=offset[0], y=offset[1], prefix=""))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            acoustics = NoopAcousticsSimulator(node=node, namespace=ns)
            human = Human(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            await scenario(human, node.tf_buffer)
        finally:
            node.destroy_node()

    asyncio.run(main())


def test_spawned_robot_follows_its_base_frame(rclpy_context):
    async def scenario(human, tf_buffer) -> None:
        assert await human.spawn_robot((_robot(1.0, 2.0),)) == (True,)
        (tracked,) = human.tracked_robots()
        assert tracked.tf_frame is not None
        assert (tracked.pose.position.x, tracked.pose.position.y) == (1.0, 2.0)
        assert tracked.velocity == (0.0, 0.0)

        tf_buffer.set_transform(_base_at(tracked.tf_frame, 1.5, 2.0, 1000), "test")
        (tracked,) = human.tracked_robots()
        assert (tracked.pose.position.x, tracked.pose.position.y) == (1.5, 2.0)
        assert tracked.velocity == (0.0, 0.0)

        tf_buffer.set_transform(_base_at(tracked.tf_frame, 1.55, 1.98, 1050), "test")
        (tracked,) = human.tracked_robots()
        assert (tracked.pose.position.x, tracked.pose.position.y) == pytest.approx((1.55, 1.98))
        assert tracked.velocity == pytest.approx((1.0, -0.4))

        (tracked,) = human.tracked_robots()
        assert tracked.velocity == pytest.approx((1.0, -0.4))

    _with_human(scenario)


def test_moved_robot_rests_at_its_new_pose(rclpy_context):
    async def scenario(human, tf_buffer) -> None:
        robot = _robot(0.0, 0.0)
        await human.spawn_robot((robot,))
        frame = human.tracked_robots()[0].tf_frame
        tf_buffer.set_transform(_base_at(frame, 0.0, 0.0, 1000), "test")
        human.tracked_robots()
        tf_buffer.set_transform(_base_at(frame, 0.05, 0.0, 1050), "test")
        assert human.tracked_robots()[0].velocity == pytest.approx((1.0, 0.0))

        robot.pose.position.x = 7.0
        assert await human.move_robot((robot,)) == (True,)
        robot.pose.position.x = 99.0
        tracked = human._robots["rob"]
        assert tracked.pose.position.x == 7.0
        assert tracked.velocity == (0.0, 0.0)

        tf_buffer.set_transform(_base_at(frame, 7.0, 0.0, 1100), "test")
        (tracked,) = human.tracked_robots()
        assert tracked.pose.position.x == 7.0
        assert tracked.velocity == (0.0, 0.0)

        tf_buffer.set_transform(_base_at(frame, 7.05, 0.0, 1150), "test")
        assert human.tracked_robots()[0].velocity == pytest.approx((1.0, 0.0))

    _with_human(scenario)


def test_removed_robot_is_no_longer_tracked(rclpy_context):
    async def scenario(human, tf_buffer) -> None:
        del tf_buffer
        robot = _robot(0.0, 0.0)
        await human.spawn_robot((robot,))
        assert await human.remove_robot((robot,)) == (True,)
        assert human.tracked_robots() == []

    _with_human(scenario)


def test_arena_adapter_streams_robot_pose_and_velocity_in_the_engine_frame(rclpy_context):
    async def scenario(human, tf_buffer) -> None:
        import rclpy
        from arena_humansim_msgs.msg import AgentStates
        from task_generator.constants.rng import stable_int

        received: list[AgentStates] = []
        human.node.create_subscription(AgentStates, str(human.node.service_namespace("world_state")), received.append, 10)

        def published() -> AgentStates | None:
            for _ in range(10):
                rclpy.spin_once(human.node, timeout_sec=0.02)
            before = len(received)
            human._publish_world_state()
            for _ in range(50):
                rclpy.spin_once(human.node, timeout_sec=0.02)
                if len(received) > before:
                    return received[-1]
            return None

        robot = _robot(11.0, -3.0)
        await human.spawn_robot((robot,))
        frame = human._robots["rob"].tf_frame
        tf_buffer.set_transform(_base_at(frame, 11.0, -3.0, 1000), "test")
        published()
        tf_buffer.set_transform(_base_at(frame, 11.05, -3.02, 1050), "test")

        msg = published()
        assert msg is not None
        agent, alias = msg.agents
        assert agent.name == "rob"
        assert agent.agent_id == stable_int("rob") & 0x7FFFFFFF
        assert (agent.pose.x, agent.pose.y) == pytest.approx((1.05, 1.98))
        assert (agent.velocity.x, agent.velocity.y) == pytest.approx((1.0, -0.4))
        assert alias.name == "robot_0"
        assert alias.agent_id == agent.agent_id
        assert (alias.pose, alias.velocity, alias.radius) == (agent.pose, agent.velocity, agent.radius)

        msg = published()
        assert msg is not None
        assert (msg.agents[0].velocity.x, msg.agents[0].velocity.y) == pytest.approx((1.0, -0.4))

        await human.remove_robot((robot,))
        assert published() is None

    _with_human(scenario, backend="arena", offset=(10.0, -5.0))


def test_arena_adapter_skips_an_index_alias_that_is_a_robot_name(rclpy_context):
    async def scenario(human, tf_buffer) -> None:
        del tf_buffer
        import rclpy
        from arena_humansim_msgs.msg import AgentStates
        from task_generator.constants.rng import stable_int

        received: list[AgentStates] = []
        human.node.create_subscription(AgentStates, str(human.node.service_namespace("world_state")), received.append, 10)
        await human.spawn_robot((_robot(0.0, 0.0, name="robot_1"), _robot(3.0, 0.0)))
        for _ in range(10):
            rclpy.spin_once(human.node, timeout_sec=0.02)
        received.clear()
        human._publish_world_state()
        for _ in range(50):
            rclpy.spin_once(human.node, timeout_sec=0.02)
            if received:
                break

        first, second = (stable_int(name) & 0x7FFFFFFF for name in ("robot_1", "rob"))
        assert [(a.name, a.agent_id) for a in received[-1].agents] == [("robot_1", first), ("rob", second), ("robot_0", first)]

    _with_human(scenario, backend="arena")


def test_engine_binds_the_index_alias_to_the_robot(rclpy_context):
    pytest.importorskip("arena_humansim.core.agent_manager")

    async def scenario(human, tf_buffer) -> None:
        del tf_buffer
        from arena_humansim.core.agent_manager import AgentManager
        from arena_humansim.core.pool import KIND_ROBOT
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from task_generator.constants.rng import stable_int

        engine = AgentManager(
            namespace=human.node.get_fully_qualified_name(),
            parameter_overrides=[Parameter("mode", value=AgentManager.MODE_SUBSYSTEM), Parameter("publish_markers", value=0)],
        )
        executor = SingleThreadedExecutor()
        executor.add_node(human.node)
        executor.add_node(engine)

        def feed_robots() -> None:
            for _ in range(200):
                human._publish_world_state()
                executor.spin_once(timeout_sec=0.02)
                if engine._latest_world_state is not None:
                    engine.tick()
                    return
            raise AssertionError("engine never received world_state")

        try:
            await human.spawn_robot((_robot(1.0, 2.0), _robot(4.0, 2.0, name="rob_1")))
            feed_robots()
            feed_robots()
            first, second = (engine._external_entities[stable_int(name) & 0x7FFFFFFF].agent_id for name in ("rob", "rob_1"))
            assert first != second
            assert engine._lookup_agent_name("rob", KIND_ROBOT) == first
            assert engine._lookup_agent_name("robot_0", KIND_ROBOT) == first
            assert engine._lookup_agent_name("rob_1", KIND_ROBOT) == second
            assert engine._lookup_agent_name("robot_1", KIND_ROBOT) == second
            pool = engine._pool
            assert sorted(int(aid) for aid, kind in zip(pool.agent_ids[: pool.n], pool.kind[: pool.n], strict=True) if int(kind) == KIND_ROBOT) == sorted((first, second))
            assert (engine._agents[first].state.pose.x, engine._agents[second].state.pose.x) == pytest.approx((1.0, 4.0))
        finally:
            executor.shutdown()
            engine.destroy_node()

    _with_human(scenario, backend="arena")


def test_pedestrian_spawned_after_the_robot_gets_its_own_engine_agent(rclpy_context):
    pytest.importorskip("arena_humansim.core.agent_manager")

    async def scenario(human, tf_buffer) -> None:
        del tf_buffer
        from arena_humansim.core.agent_manager import AgentManager
        from arena_humansim.core.pool import KIND_ROBOT
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from task_generator.constants.rng import stable_int
        from task_generator.shared import Pose, Position

        engine = AgentManager(
            namespace=human.node.get_fully_qualified_name(),
            parameter_overrides=[Parameter("mode", value=AgentManager.MODE_SUBSYSTEM), Parameter("publish_markers", value=0)],
        )
        executor = SingleThreadedExecutor()
        executor.add_node(human.node)
        executor.add_node(engine)

        def feed_robot() -> None:
            for _ in range(200):
                human._publish_world_state()
                executor.spin_once(timeout_sec=0.02)
                if engine._latest_world_state is not None:
                    engine.tick()
                    return
            raise AssertionError("engine never received world_state")

        try:
            await human.spawn_robot((_robot(1.0, 2.0),))
            feed_robot()
            robot_id = engine._external_entities[stable_int("rob") & 0x7FFFFFFF].agent_id

            for _ in range(200):
                if human._spawn_client.client.service_is_ready():
                    break
                executor.spin_once(timeout_sec=0.02)
            ped = human._runtime_obstacle(name="ped", pose=Pose(Position(x=20.0, y=20.0)))
            spawn = asyncio.ensure_future(human._spawn_dynamic_obstacles_impl([ped]))
            for _ in range(500):
                if spawn.done():
                    break
                executor.spin_once(timeout_sec=0.02)
                await asyncio.sleep(0)
            assert spawn.result() == [ped]

            feed_robot()
            (ped_id,) = human._bridge_agent_ids
            assert human._agent_names == {ped_id: ped.sim_path}
            assert ped_id != robot_id
            pedestrian = engine._agents[ped_id].state
            assert (pedestrian.pose.x, pedestrian.pose.y) == pytest.approx((20.0, 20.0), abs=0.2)
            robot = engine._agents[robot_id].state
            assert robot.kind == KIND_ROBOT
            assert (robot.pose.x, robot.pose.y) == pytest.approx((1.0, 2.0))
        finally:
            executor.shutdown()
            engine.destroy_node()

    _with_human(scenario, backend="arena")
