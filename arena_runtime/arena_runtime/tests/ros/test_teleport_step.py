"""Robot teleport against an in-process ArenaRuntimeNode on the dummy sim, with a body node that follows set-pose on the next sim tick."""

from __future__ import annotations

import asyncio
import contextlib
import types
import typing

import pytest

DT = 0.02
START = (0.0, 0.0)
TARGET = (3.0, 4.0)


async def _until(pred: typing.Callable[[], bool], timeout: float, what: str) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.02)


async def _run(scenario: typing.Callable[..., typing.Awaitable[None]], *, follows: bool) -> None:
    import arena_runtime_msgs.srv
    import geometry_msgs.msg
    import rclpy.qos
    import rosgraph_msgs.msg
    import tf2_ros
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import FrameNamespace
    from arena_rclpy_mixins.spin import create_executor
    from arena_rclpy_mixins.Time import Time
    from arena_runtime._node import NodeInterface
    from arena_runtime.arena_runtime_node import ArenaRuntimeNode

    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.env.episode_node import EpisodeNode

    frame = FrameNamespace("probe")
    base = frame("base_link").raw()

    class Body(ArenaMixinNode):
        """Stands in for the sim's robot body: a commanded pose applies on the next sim tick, TF is stamped per tick."""

        def __init__(self) -> None:
            super().__init__("teleport_body")
            self.pose = START
            self.pending: tuple[float, float] | None = None
            self._last: Time | None = None
            self._tf = tf2_ros.TransformBroadcaster(self)
            self.create_subscription(rosgraph_msgs.msg.Clock, "/clock", self._on_clock, rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT))

        def _on_clock(self, msg: rosgraph_msgs.msg.Clock) -> None:
            stamp = Time.from_msg(msg.clock)
            if self._last is not None and stamp <= self._last:
                return
            if self._last is not None and self.pending is not None and follows:
                self.pose, self.pending = self.pending, None
            self._last = stamp
            tf = geometry_msgs.msg.TransformStamped()
            tf.header.stamp = msg.clock
            tf.header.frame_id = "map"
            tf.child_frame_id = base
            tf.transform.translation.x, tf.transform.translation.y = self.pose
            tf.transform.rotation.w = 1.0
            self._tf.sendTransform(tf)

    class Bench(ArenaMixinNode):
        """The task_generator side: the real hold, step and unpause window calls against arena_runtime_node."""

        hold = EpisodeNode.hold
        release = EpisodeNode.release
        step_tick = EpisodeNode.step_tick
        unpause_window = EpisodeNode.unpause_window

        def __init__(self) -> None:
            super().__init__("teleport_bench")
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self._arena_hold_client = self.create_client_wrapper(arena_runtime_msgs.srv.LifecycleHold, "/arena/sim_lifecycle/hold", timeout=30.0)
            self._arena_unpause_window_client = self.create_client_wrapper(arena_runtime_msgs.srv.LifecycleUnpauseWindow, "/arena/sim_lifecycle/unpause_window", timeout=30.0)
            self._arena_step_client = self.create_client_wrapper(arena_runtime_msgs.srv.LifecycleStep, "/arena/sim_lifecycle/step", timeout=30.0)

    class Environment:
        def __init__(self, body: Body) -> None:
            self._body = body

        def realize(self, pose: object) -> object:
            return pose

        async def move_robot(self, robots: typing.Sequence[types.SimpleNamespace]) -> tuple[bool, ...]:
            self._body.pending = (robots[0].pose.position.x, robots[0].pose.position.y)
            return (True,)

    class Robot(RobotManager):
        def __init__(self, node: Bench, environment: Environment) -> None:
            NodeInterface.__init__(self, node=node)
            self._robot = types.SimpleNamespace(name="probe", frame=frame, pose=None)
            self._config = types.SimpleNamespace(model_params=types.SimpleNamespace(z_offset=0.0, base_frame="base_link"))
            self._environment_manager = environment
            self._adapter_instances = []
            self._stop_pub = node.create_publisher(geometry_msgs.msg.Twist, "/probe/stop", 1)

    executor = create_executor()
    arena = ArenaRuntimeNode()
    arena.declare_parameter("physics_dt", DT)
    arena.declare_parameter("sim", "dummy")
    body = Body()
    bench = Bench()
    nodes = (arena, body, bench)
    for node in nodes:
        executor.add_node(node)
    loop = asyncio.get_running_loop()
    spin = loop.run_in_executor(None, executor.spin)
    try:
        try:
            await arena.setup()
        except RuntimeError as e:
            if "already running" in str(e):
                pytest.skip(str(e))
            raise
        assert await bench._arena_step_client.ensure(timeout_sec=30.0)
        await _until(lambda: bench.sim_time.to_seconds() > 0.0, 30.0, "dummy /clock")
        await _until(lambda: bench.tf_buffer.can_transform("map", base, Time()), 30.0, "body TF")
        await scenario(arena, body, bench, Robot(bench, Environment(body)))
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(arena.teardown(), timeout=30.0)
        if arena._clock_task is not None:
            arena._clock_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await arena._clock_task
        executor.shutdown()
        await spin
        for node in nodes:
            executor.remove_node(node)
            node.destroy_node()


def _target() -> object:
    from arena_runtime.shared import Pose, Position

    return Pose(Position(*TARGET))


def _ticks(advance: float) -> int:
    ticks = round(advance / DT)
    assert advance == pytest.approx(ticks * DT, abs=1e-6), f"sim advanced {advance}s, not a whole number of {DT}s steps"
    return ticks


async def _frozen(bench: typing.Any, seconds: float) -> float:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 20.0
    before = bench.sim_time.to_seconds()
    since = loop.time()
    while loop.time() - since < 0.3:
        assert loop.time() < deadline, "sim clock did not settle"
        await asyncio.sleep(0.02)
        now = bench.sim_time.to_seconds()
        if now != before:
            before = now
            since = loop.time()
    await asyncio.sleep(seconds)
    assert bench.sim_time.to_seconds() == pytest.approx(before, abs=1e-9), "sim clock moved while held"
    return before


def test_held_teleport_steps_until_landed() -> None:
    from arena_runtime.env.robot_manager import robot_manager

    async def scenario(arena: typing.Any, body: typing.Any, bench: typing.Any, robot: typing.Any) -> None:
        await bench.hold("reset")
        before = await _frozen(bench, 0.2)
        await robot._apply_pose(_target())
        assert arena._windows.is_empty()
        after = await _frozen(bench, 0.2)
        assert 1 <= _ticks(after - before) <= robot_manager._TELEPORT_TICK_CAP
        assert body.pose == TARGET
        error, _ = robot._pose_error(_target())
        assert error <= robot_manager._TELEPORT_TOLERANCE
        await bench.release("reset")

    asyncio.run(asyncio.wait_for(_run(scenario, follows=True), timeout=90.0))


def test_held_teleport_that_never_lands_stops_at_the_cap() -> None:
    from arena_runtime.env.robot_manager import robot_manager

    async def scenario(arena: typing.Any, body: typing.Any, bench: typing.Any, robot: typing.Any) -> None:
        await bench.hold("reset")
        before = await _frozen(bench, 0.2)
        with pytest.raises(RuntimeError, match="never landed"):
            await robot._apply_pose(_target())
        after = await _frozen(bench, 0.2)
        assert _ticks(after - before) == robot_manager._TELEPORT_ATTEMPTS * robot_manager._TELEPORT_TICK_CAP
        assert body.pose == START
        await bench.release("reset")

    asyncio.run(asyncio.wait_for(_run(scenario, follows=False), timeout=120.0))


def test_unheld_teleport_falls_back_to_the_unpause_window() -> None:
    async def scenario(arena: typing.Any, body: typing.Any, bench: typing.Any, robot: typing.Any) -> None:
        assert await bench.step_tick() is None
        await robot._apply_pose(_target())
        assert arena._windows.is_empty()
        assert body.pose == TARGET

    asyncio.run(asyncio.wait_for(_run(scenario, follows=True), timeout=90.0))
