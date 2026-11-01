"""FlatsimSimulator and FlatsimHost against a flatsim server in the same process: a jackal driven by cmd_vel, its scan, cloud and imu."""

from __future__ import annotations

import asyncio
import contextlib
import math
import typing

import pytest

DT = 1.0 / 30.0
WALL_X = 3.0


async def _until(pred: typing.Callable[[], bool], timeout: float, what: str) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.01)


async def _run(dynamics: str, scenario: typing.Callable[..., typing.Awaitable[None]]) -> None:
    pytest.importorskip("flatsim_msgs")
    import geometry_msgs.msg
    import nav_msgs.msg
    import rclpy.parameter
    import sensor_msgs.msg
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.spin import create_executor
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Pose, Robot

    from arena_runtime.sim.flatsim_simulator.flatsim_simulator import FlatsimHost, FlatsimSimulator
    from flatsim.server import FlatsimServer

    class Bench(ArenaMixinNode):
        """Plays the runtime (steps the server) and the planner (publishes cmd_vel), records what the server publishes."""

        def __init__(self) -> None:
            super().__init__("flatsim_bench", namespace="/flatsim_test/env_0")
            self.host = FlatsimHost(DT, self)
            self.odom: list[nav_msgs.msg.Odometry] = []
            self.scans: list[sensor_msgs.msg.LaserScan] = []
            self.clouds: list[sensor_msgs.msg.PointCloud2] = []
            self.imu: list[sensor_msgs.msg.Imu] = []

        @property
        def now(self) -> float:
            return self.sim_time.to_seconds()

        def stamp(self) -> float:
            return _seconds(self.odom[-1]) if self.odom else -math.inf

        async def step(self, steps: int = 1) -> None:
            assert await self.host.step_seconds(steps * DT) == pytest.approx(steps * DT)
            await _until(lambda: self.stamp() >= self.now - 1e-6, 5.0, f"odom at {self.now:.3f}")

        async def drive(self, steps: int, command: geometry_msgs.msg.Twist, publisher: typing.Any) -> None:
            for _ in range(steps):
                publisher.publish(command)
                await asyncio.sleep(0.02)
                await self.step()

        async def seen(self, sim: FlatsimSimulator, pred: typing.Callable[[], bool], what: str) -> None:
            """Republish until `pred` holds, for inputs that reach the server on a topic."""
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while True:
                assert await sim.step()
                await asyncio.sleep(0.05)
                if pred():
                    return
                assert loop.time() < deadline, f"timed out waiting for {what}"

    executor = create_executor()
    server = FlatsimServer(parameter_overrides=[rclpy.parameter.Parameter("physics_dt", value=DT), rclpy.parameter.Parameter("paused", value=True), rclpy.parameter.Parameter("dynamics", value=dynamics)])
    bench = Bench()
    executor.add_node(server)
    executor.add_node(bench)
    loop = asyncio.get_running_loop()
    spin = loop.run_in_executor(None, executor.spin)
    sim: FlatsimSimulator | None = None
    robot = Robot.parse({"name": "jackal", "model": "jackal"}, node=bench)
    try:
        await asyncio.wait_for(bench.host.ensure_ready(), timeout=10.0)
        sim = await FlatsimSimulator.create(node=bench, namespace=Namespace("/flatsim_test/env_0"), realizer=Realizer(), env_id=0)
        ns = bench.service_namespace(robot.name)
        bench.create_subscription(nav_msgs.msg.Odometry, str(ns("odom")), bench.odom.append, 50)
        bench.create_subscription(sensor_msgs.msg.LaserScan, str(ns("lidar")), bench.scans.append, 50)
        bench.create_subscription(sensor_msgs.msg.PointCloud2, str(ns("lidar", "points")), bench.clouds.append, 50)
        bench.create_subscription(sensor_msgs.msg.Imu, str(ns("imu", "data")), bench.imu.append, 50)
        cmd_vel = bench.create_publisher(geometry_msgs.msg.Twist, str(ns("cmd_vel")), 10)
        assert await sim.spawn_box("wall", (0.1, 6.0, 2.0), Pose.parse({"x": WALL_X, "y": 0.0, "z": 1.0}))
        assert list(await sim.robot_spawn([robot])) == [True]
        await _until(lambda: cmd_vel.get_subscription_count() == 1, 5.0, "cmd_vel subscription")
        await bench.seen(sim, lambda: bool(bench.odom) and bool(bench.scans) and bool(bench.clouds) and bool(bench.imu), "first odom, scan, cloud and imu")
        await scenario(bench, sim, robot, cmd_vel)
    finally:
        if sim is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sim.shutdown(), timeout=10.0)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(bench.kill_launches(), timeout=10.0)
        executor.shutdown()
        await spin
        executor.remove_node(bench)
        executor.remove_node(server)
        bench.destroy_node()
        server.destroy_node()


def _seconds(msg: typing.Any) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _ahead(scan: typing.Any) -> float:
    return scan.ranges[round(-scan.angle_min / scan.angle_increment)]


def _mobile() -> typing.Any:
    from arena_robots.Robot import RobotIdentifier

    return RobotIdentifier.parse("jackal").resolve_sync().model_params


async def _drives_and_scans(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    import geometry_msgs.msg
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_runtime.humans import PED_RADIUS

    params = _mobile()
    mobile = params.caps.mobile
    top, accel = 0.5, mobile.planar.traction

    first = bench.odom[-1]
    assert first.header.frame_id.endswith("odom") and first.child_frame_id.endswith(params.base_frame)
    assert first.pose.pose.position.x == pytest.approx(0.0, abs=1e-6)
    scan = bench.scans[-1]
    assert len(scan.ranges) == mobile.laser.num_beams
    assert scan.range_max == pytest.approx(mobile.laser.range)
    gap = _ahead(scan)
    assert 1.5 < gap < WALL_X

    walker = Pedestrian(name="walker")
    walker.pose.position.x = 1.5
    walker.pose.orientation.w = 1.0
    assert list(await sim.pedestrian_update(Pedestrians(pedestrians=[walker]))) == [True]
    await bench.seen(sim, lambda: _ahead(bench.scans[-1]) == pytest.approx(gap - (WALL_X - 0.05) + 1.5 - PED_RADIUS, abs=1e-3), "scan hitting the pedestrian disc")
    walker.pose.position.y = 4.0
    await sim.pedestrian_update(Pedestrians(pedestrians=[walker]))
    await bench.seen(sim, lambda: _ahead(bench.scans[-1]) == pytest.approx(gap, abs=1e-3), "scan back on the wall")

    speed, travelled = 0.0, 0.0
    steps = 45
    start = bench.now
    driving = len(bench.scans)
    await bench.drive(steps, geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=top)), cmd_vel)
    for _ in range(steps):
        speed = min(top, speed + accel * DT)
        travelled += speed * DT
    last = bench.odom[-1]
    assert last.twist.twist.linear.x == pytest.approx(top, abs=1e-4)
    assert last.pose.pose.position.x == pytest.approx(travelled, abs=2e-3)
    assert bench.now == pytest.approx(start + steps * DT, abs=1e-6)
    assert bench.stamp() == pytest.approx(bench.now, abs=1e-6)
    await _until(lambda: _ahead(bench.scans[-1]) < gap - travelled + 0.05, 2.0, "scan following the robot")
    stamps = [_seconds(scan) for scan in bench.scans[driving:]]
    assert len(stamps) >= 3
    assert all(later - earlier == pytest.approx(round(1.0 / mobile.laser.update_rate / DT) * DT, abs=1e-6) for earlier, later in zip(stamps, stamps[1:], strict=False))
    await _until(lambda: (pose := sim.robot_pose(robot.sim_path)) is not None and pose.position.x == pytest.approx(travelled, abs=2e-3), 5.0, "tf of the driven robot")

    frozen = len(bench.odom)
    await asyncio.sleep(0.3)
    assert len(bench.odom) == frozen

    assert await sim.delete_box("wall")
    assert not await sim.move_box("wall", robot.pose)
    await bench.seen(sim, lambda: math.isinf(_ahead(bench.scans[-1])), "scan after the wall is gone")


async def _reset_publishes_the_new_pose(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    from arena_runtime.shared import Pose

    await asyncio.sleep(0.3)
    gap = _ahead(bench.scans[-1])
    before = len(bench.scans)
    assert await sim.set_robot_pose(robot.sim_path, Pose.parse({"x": 1.0, "y": 0.0, "z": 0.0}))
    assert not await sim.set_robot_pose("nobody", Pose.parse({"x": 1.0, "y": 0.0, "z": 0.0}))
    await asyncio.sleep(0.3)
    assert len(bench.scans) == before
    assert await sim.after_reset_episode()
    await _until(lambda: len(bench.scans) > before, 2.0, "scan after the reset")
    assert _ahead(bench.scans[-1]) == pytest.approx(gap - 1.0, abs=1e-3)
    assert bench.odom[-1].pose.pose.position.x == pytest.approx(1.0, abs=1e-6)


async def _shutdown_takes_the_env_out_of_the_world(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    odom = str(bench.service_namespace(robot.name)("odom"))
    assert bench.count_publishers(odom) == 1
    await sim.shutdown()
    await _until(lambda: bench.count_publishers(odom) == 0, 5.0, "robot topics to close")
    assert await bench.host.cleanup_namespace(bench.host.env_prefix(0)) == 0


async def _host_removes_the_entities_of_an_env(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    odom = str(bench.service_namespace(robot.name)("odom"))
    assert await bench.host.cleanup_namespace(bench.host.env_prefix(1)) == 0
    assert await bench.host.cleanup_namespace(bench.host.env_prefix(0)) == 2
    await _until(lambda: bench.count_publishers(odom) == 0, 5.0, "robot topics to close")
    assert await bench.host.unpause()
    await _until(lambda: bench.now > 0.2, 5.0, "free-running clock")
    assert await bench.host.pause()
    paused = bench.now
    await asyncio.sleep(0.3)
    assert bench.now == pytest.approx(paused, abs=DT + 1e-6)


async def _two_envs_share_the_world(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.shared import Pose

    from arena_runtime.sim.flatsim_simulator import FlatsimSimulator

    gap = _ahead(bench.scans[-1])
    other = await FlatsimSimulator.create(node=bench, namespace=Namespace("/flatsim_test/env_1"), realizer=Realizer(), env_id=1)
    try:
        assert await other.spawn_box("wall", (0.1, 6.0, 2.0), Pose.parse({"x": WALL_X, "y": 40.0, "z": 1.0}))
        assert await other.spawn_box("near", (0.1, 6.0, 2.0), Pose.parse({"x": 1.0, "y": 40.0, "z": 1.0}))
        assert await bench.host.cleanup_namespace(bench.host.env_prefix(1)) == 2
        assert not await other.delete_box("wall")
        await bench.seen(sim, lambda: _ahead(bench.scans[-1]) == pytest.approx(gap, abs=1e-3), "env_0 still seeing its own wall")
    finally:
        await other.shutdown()


async def _stops_at_wall(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    import geometry_msgs.msg

    await bench.drive(240, geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=10.0)), cmd_vel)
    x = bench.odom[-1].pose.pose.position.x
    assert 2.4 < x < WALL_X - 0.05 - 0.2
    assert abs(bench.odom[-1].twist.twist.linear.x) < 0.05


async def _passes_wall(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    import geometry_msgs.msg

    await bench.drive(240, geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=10.0)), cmd_vel)
    assert bench.odom[-1].pose.pose.position.x > WALL_X + 0.5


def _points(cloud: typing.Any) -> typing.Any:
    import numpy as np

    layout = np.dtype({"names": ["x", "y", "z", "intensity", "ring"], "formats": ["<f4", "<f4", "<f4", "<f4", "<u2"], "offsets": [0, 4, 8, 16, 24], "itemsize": 32})
    return np.frombuffer(bytes(cloud.data), dtype=layout)


async def _lidar_points(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    import numpy as np
    import sensor_msgs.msg
    from arena_people_msgs.msg import Pedestrian, Pedestrians

    cloud = bench.clouds[-1]
    await _until(lambda: any(scan.header.stamp == cloud.header.stamp for scan in bench.scans), 2.0, "scan of the cloud's cast")
    scan = next(scan for scan in bench.scans if scan.header.stamp == cloud.header.stamp)
    assert cloud.header.frame_id == scan.header.frame_id and cloud.header.frame_id.endswith("lidar_link")
    assert (cloud.height, cloud.point_step, cloud.is_bigendian) == (1, 32, False)
    assert cloud.row_step == 32 * cloud.width == len(cloud.data)
    float32, uint16 = sensor_msgs.msg.PointField.FLOAT32, sensor_msgs.msg.PointField.UINT16
    assert [(field.name, field.offset, field.datatype, field.count) for field in cloud.fields] == [("x", 0, float32, 1), ("y", 4, float32, 1), ("z", 8, float32, 1), ("intensity", 16, float32, 1), ("ring", 24, uint16, 1)]

    points = _points(cloud)
    assert len(points) == cloud.width > 0
    assert set(points["ring"].tolist()) <= set(range(16))
    assert not points["intensity"].any()
    ahead = points[(points["ring"] == 8) & (np.abs(points["y"]) < 0.02)]
    assert len(ahead) > 0
    assert ahead["x"] == pytest.approx(np.full(len(ahead), _ahead(scan)), abs=1e-3)
    assert ahead["z"] == pytest.approx(ahead["x"] * math.tan(0.261799 / 15), abs=1e-3)

    walker = Pedestrian(name="walker")
    walker.pose.position.x = 0.65
    walker.pose.orientation.w = 1.0
    assert list(await sim.pedestrian_update(Pedestrians(pedestrians=[walker]))) == [True]

    def on_walker() -> set[int]:
        latest = _points(bench.clouds[-1])
        return set(latest["ring"][latest["x"] < 0.4].tolist())

    await bench.seen(sim, lambda: on_walker() == set(range(16)), "all sixteen rings on the pedestrian")
    points = _points(bench.clouds[-1])
    slant = np.sqrt(points["x"].astype(float) ** 2 + points["y"].astype(float) ** 2 + points["z"].astype(float) ** 2)
    assert slant.min() >= scan.range_min - 1e-6 and slant.max() <= scan.range_max + 1e-6
    assert slant.min() == pytest.approx((0.65 - 0.3) / math.cos(0.261799 / 15), abs=5e-3)
    assert np.isfinite(slant).all()


async def _imu_follows_the_drive(bench: typing.Any, sim: typing.Any, robot: typing.Any, cmd_vel: typing.Any) -> None:
    import geometry_msgs.msg

    planar = _mobile().caps.mobile.planar
    top, accel = planar.velocity["linear"].max, planar.traction
    turn = planar.velocity["angular"].max * planar.gain["angular"]

    async def synced() -> typing.Any:
        await _until(lambda: bool(bench.imu) and _seconds(bench.imu[-1]) >= bench.stamp() - 1e-6, 2.0, "imu of the last tick")
        return bench.imu[-1]

    rest = await synced()
    assert rest.header.frame_id.endswith("imu_link")
    assert rest.linear_acceleration.z == pytest.approx(9.80665)
    assert (rest.linear_acceleration.x, rest.linear_acceleration.y, rest.angular_velocity.z) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert rest.orientation.w == pytest.approx(1.0)
    assert not any(rest.orientation_covariance) and not any(rest.angular_velocity_covariance) and not any(rest.linear_acceleration_covariance)

    await asyncio.sleep(0.3)
    driving = len(bench.imu)
    await bench.drive(3, geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=10.0)), cmd_vel)
    speeding = await synced()
    assert speeding.linear_acceleration.x == pytest.approx(accel, abs=1e-3)
    assert speeding.linear_acceleration.z == pytest.approx(9.80665)

    await bench.drive(60, geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=10.0), angular=geometry_msgs.msg.Vector3(z=10.0)), cmd_vel)
    turning = await synced()
    odom = bench.odom[-1]
    assert turning.angular_velocity.z == pytest.approx(turn, abs=1e-4)
    assert (odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z) == pytest.approx((top, 0.0, planar.velocity["angular"].max), abs=1e-4)
    assert (turning.orientation.z, turning.orientation.w) == pytest.approx((odom.pose.pose.orientation.z, odom.pose.pose.orientation.w), abs=1e-6)
    assert turning.linear_acceleration.x == pytest.approx(planar.pivot * turn**2, abs=0.1)
    assert turning.linear_acceleration.y == pytest.approx(top * turn, abs=0.1)
    assert turning.linear_acceleration.z == pytest.approx(9.80665)
    stamps = [_seconds(imu) for imu in bench.imu[driving:]]
    assert len(stamps) == 63
    assert all(later - earlier == pytest.approx(DT, abs=1e-6) for earlier, later in zip(stamps, stamps[1:], strict=False))


def test_cmd_vel_and_host_steps_drive_odom_and_scans() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _drives_and_scans), timeout=120.0))


def test_rigid_robot_stops_at_a_box() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _stops_at_wall), timeout=120.0))


def test_kinematic_robot_passes_a_box() -> None:
    asyncio.run(asyncio.wait_for(_run("kinematic", _passes_wall), timeout=120.0))


def test_jackal_publishes_a_sixteen_ring_lidar_cloud() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _lidar_points), timeout=120.0))


def test_jackal_imu_follows_the_commanded_turn() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _imu_follows_the_drive), timeout=120.0))


def test_reset_publishes_the_scan_of_the_new_pose() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _reset_publishes_the_new_pose), timeout=120.0))


def test_shutdown_takes_the_env_out_of_the_world() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _shutdown_takes_the_env_out_of_the_world), timeout=120.0))


def test_host_removes_the_entities_of_an_env_and_pauses_the_clock() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _host_removes_the_entities_of_an_env), timeout=120.0))


def test_two_envs_share_the_world_and_leave_it_by_prefix() -> None:
    asyncio.run(asyncio.wait_for(_run("rigid", _two_envs_share_the_world), timeout=120.0))


def test_overrides_take_the_parameters_of_the_simulator_interface() -> None:
    pytest.importorskip("flatsim_msgs")
    import inspect

    from arena_runtime.sim import BaseSim
    from arena_runtime.sim.flatsim_simulator import FlatsimSimulator

    own = vars(FlatsimSimulator)
    for name, base in inspect.getmembers(BaseSim, inspect.isfunction):
        if name in own and not name.startswith("_"):
            assert list(inspect.signature(own[name]).parameters) == list(inspect.signature(base).parameters), name
