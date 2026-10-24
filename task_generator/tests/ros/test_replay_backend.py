from __future__ import annotations

import asyncio
import itertools
import math
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest

ITEM = "eth"
WALKERS = {1: ((0.0, 0.0), (8.0, 0.0)), 2: ((8.0, 2.0), (0.0, 2.0)), 3: ((0.0, 5.0), (6.0, 1.0))}
PEDS = [f"ped_{ident}" for ident in WALKERS]
OFFSET = (10.0, -2.0)
PREFIX = "env"
START_S = 100.0
TICK_S = 1.0 / 30.0
DEADLINE_S = 20.0


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("arena_assets.importers")
    pytest.importorskip("arena_simulation_setup.shared.track")


def _stamp_s(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Stage:
    def __init__(self, node, probe, clock, realizer, simulator, human, environment, bus: list) -> None:
        self.node = node
        self.probe = probe
        self.clock = clock
        self.realizer = realizer
        self.simulator = simulator
        self.human = human
        self.environment = environment
        self.bus = bus
        self.now = START_S

    async def advance(self, t: float) -> None:
        from builtin_interfaces.msg import Time
        from rosgraph_msgs.msg import Clock

        ticks = math.floor((t - self.now) / TICK_S + 1e-9)
        for step in range(ticks, -1, -1):
            self.now = t - step * TICK_S
            sec, nanosec = divmod(round(self.now * 1e9), 10**9)
            self.clock.publish(Clock(clock=Time(sec=sec, nanosec=nanosec)))
            await asyncio.sleep(0.002)
        deadline = time.monotonic() + DEADLINE_S
        while self.node.sim_time.to_seconds() < t - 1e-6:
            assert time.monotonic() < deadline, f"sim clock stuck at {self.node.sim_time.to_seconds()} on the way to {t}"
            self.clock.publish(Clock(clock=Time(sec=sec, nanosec=nanosec)))
            await asyncio.sleep(0.01)

    async def play(self, dynamic) -> None:
        await self.environment.respawn(lambda: self.environment.spawn_dynamic_obstacles(dynamic))
        self.start = self.now

    async def at(self, t: float, expected: dict[str, tuple[float, float] | None]):
        seen = len(self.bus)
        await self.advance(self.start + t)
        names = sorted(self.realizer.prefix(name) for name in expected)
        deadline = time.monotonic() + DEADLINE_S
        while True:
            upto = len(self.bus)
            for msg in self.bus[seen:upto]:
                peds = {ped.name: ped for ped in msg.pedestrians}
                if _stamp_s(msg) > self.start + t - 1e-6 and sorted(peds) == names and all(xy is None or math.hypot(peds[self.realizer.prefix(name)].pose.position.x - xy[0], peds[self.realizer.prefix(name)].pose.position.y - xy[1]) < 2e-3 for name, xy in expected.items()):
                    return {name: peds[self.realizer.prefix(name)] for name in expected}
            seen = upto
            assert time.monotonic() < deadline, f"no bus batch with {expected} at t={t}, last: {[(ped.name, ped.pose.position.x, ped.pose.position.y) for ped in self.bus[-1].pedestrians] if self.bus else None}"
            await asyncio.sleep(0.01)


def _on_stage(body: Callable[[Stage], Awaitable[None]]) -> None:
    import rclpy
    import rclpy.executors
    from arena_people_msgs.msg import Pedestrians
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.qos import best_effort
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from rosgraph_msgs.msg import Clock

    from task_generator.constants import Constants
    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.manager.realizer import Realizer
    from task_generator.simulators.acoustics import AcousticsSimulatorRegistry
    from task_generator.simulators.human import HumanSimulatorRegistry
    from task_generator.simulators.human.replay import ReplayHumanSimulator

    class _Node(ServiceNamespace, AsyncNode):
        pass

    class _Sim(DummySimulator):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.spawned: list[dict[str, tuple[float, float]]] = []
            self.deleted: list[list[str]] = []
            self.updated: list[dict[str, tuple[float, float]]] = []

        async def pedestrian_spawn(self, pedestrians):
            self.spawned.append({ped.sim_path: (ped.pose.position.x, ped.pose.position.y) for ped in pedestrians})
            return await super().pedestrian_spawn(pedestrians)

        async def pedestrian_delete(self, pedestrians):
            self.deleted.append(sorted(ped.sim_path for ped in pedestrians))
            return await super().pedestrian_delete(pedestrians)

        async def pedestrian_update(self, pedestrians):
            self.updated.append({ped.name: (ped.pose.position.x, ped.pose.position.y) for ped in pedestrians.pedestrians})
            return await super().pedestrian_update(pedestrians)

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> None:
        node = _Node(f"replay_{suffix}", namespace=str(ns))
        probe = rclpy.create_node(f"replay_probe_{suffix}")
        bus: list[Pedestrians] = []
        probe.create_subscription(Pedestrians, str(ns("arena_peds")), bus.append, 10)
        clock = probe.create_publisher(Clock, "/clock", best_effort(1))
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(probe)
        stop = threading.Event()

        def spin() -> None:
            while not stop.is_set():
                executor.spin_once(timeout_sec=0.02)

        thread = threading.Thread(target=spin, daemon=True)
        thread.start()
        human = None
        try:
            realizer = Realizer(Realizer._Configuration(x=OFFSET[0], y=OFFSET[1], prefix=PREFIX))
            simulator = _Sim(node=node, namespace=ns, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=ns)
            human = await HumanSimulatorRegistry.get(Constants.HumanSimulator.REPLAY, node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            assert type(human) is ReplayHumanSimulator
            environment = EnvironmentManager(node=node, simulator=simulator, human_simulator=human, realizer=realizer)
            stage = Stage(node, probe, clock, realizer, simulator, human, environment, bus)
            await stage.advance(START_S)
            await body(stage)
        finally:
            if human is not None:
                human._replay_task.cancel()
                human._keepalive_task.cancel()
            stop.set()
            thread.join(timeout=2.0)
            executor.remove_node(node)
            executor.remove_node(probe)
            probe.destroy_node()
            node.destroy_node()

    asyncio.run(main())


def test_imported_recording_plays_on_the_bus_and_in_the_sim(rclpy_context, tmp_path: Path) -> None:
    from arena_people_msgs.msg import Pedestrian
    from arena_simulation_setup.shared.track import Track
    from arena_simulation_setup.tree import SimplePathResolver
    from arena_simulation_setup.tree.World import WorldIdentifier

    from task_generator.adapters.eth_ucy import ETHUCY, SEQUENCES
    from task_generator.simulators.human.gait import GaitGenerator
    from task_generator.simulators.human.replay import WALK_SPEED
    from task_generator.tasks.registry import identifier_to_available

    entry = SEQUENCES[ITEM]
    frames = range(round(10.0 * entry.fps) + 1)
    columns = [(frame, ident, y0 + frame / frames[-1] * (y1 - y0), x0 + frame / frames[-1] * (x1 - x0)) for frame in frames for ident, ((x0, y0), (x1, y1)) in WALKERS.items()]
    src = tmp_path / "cache" / entry.file
    src.parent.mkdir(parents=True)
    src.write_text("\n".join(",".join(str(value) for value in row) for row in zip(*columns, strict=True)) + "\n")
    out = tmp_path / "eth_ucy"
    world = Path(ETHUCY(cache=tmp_path / "cache").convert(ITEM, src, out)).parts[0]
    tracks = {name: Track.load(out / world / "scenarios" / ITEM / f"{name}.csv") for name in PEDS}

    def expected(t: float) -> dict[str, tuple[float, float]]:
        return {name: (track.state(t).x + OFFSET[0], track.state(t).y + OFFSET[1]) for name, track in tracks.items()}

    async def body(stage: Stage) -> None:
        identifier = WorldIdentifier(world)
        view = identifier.load(await SimplePathResolver(WorldIdentifier, out).resolve(identifier))
        assert list(identifier_to_available(view.scenario)) == [ITEM]
        scenario = view.scenario(ITEM).resolve_sync().load()
        names = sorted(stage.realizer.prefix(name) for name in PEDS)
        assert names == sorted(f"{PREFIX}/{name}" for name in PEDS)

        await stage.play(scenario.humans)
        assert sorted(stage.simulator.spawned[0]) == names
        for name, xy in stage.simulator.spawned[0].items():
            assert xy == pytest.approx(expected(0.0)[name.removeprefix(f"{PREFIX}/")], abs=1e-3)

        for t in (1.035, 5.075, 9.045):
            peds = await stage.at(t, expected(t))
            for name, ped in peds.items():
                state = tracks[name].state(t)
                assert ped.pose.position.z == 0.0
                assert (ped.twist.linear.x, ped.twist.linear.y) == pytest.approx((state.vx, state.vy), abs=1e-6)
                assert ped.animation_state == (Pedestrian.WALKING if math.hypot(state.vx, state.vy) >= WALK_SPEED else Pedestrian.IDLE)
                assert set(ped.joint_state.name) <= set(GaitGenerator.JOINT_NAMES)
                assert len(ped.joint_state.name) == len(ped.joint_state.position) > 0
            assert sorted(stage.simulator.updated[-1]) == names
            for name, xy in stage.simulator.updated[-1].items():
                assert xy == pytest.approx(expected(t)[name.removeprefix(f"{PREFIX}/")], abs=2e-3)
        assert not any(stage.simulator.deleted)
        assert len(stage.simulator.spawned) == 1

        await stage.at(12.0, {})
        assert sorted(itertools.chain.from_iterable(stage.simulator.deleted)) == names
        stamps = [_stamp_s(msg) for msg in stage.bus]
        assert all(later >= earlier for earlier, later in itertools.pairwise(stamps))
        walked = [msg for msg in stage.bus if len(msg.pedestrians) == len(PEDS)]
        first = {ped.name: ped for ped in walked[0].pedestrians}
        last = {ped.name: ped for ped in walked[-1].pedestrians}
        assert max(math.hypot(last[name].pose.position.x - first[name].pose.position.x, last[name].pose.position.y - first[name].pose.position.y) for name in names) > 3.0

    _on_stage(body)


def test_pedestrians_enter_and_leave_with_their_track(rclpy_context, tmp_path: Path) -> None:
    import yaml
    from arena_people_msgs.msg import Pedestrian
    from arena_simulation_setup.shared.track import Track
    from arena_simulation_setup.tree.World.Scenario import ScenarioView

    directory = tmp_path / "scenarios" / "gap"
    Track.from_segments(
        [
            (np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([3.0, 3.0]), np.array([0.0, 0.0])),
            (np.array([4.0, 5.0]), np.array([5.0, 5.0]), np.array([3.0, 1.0]), np.array([-math.pi / 2, -math.pi / 2])),
        ]
    ).save(directory / "late.csv")
    scenario = {
        "format": "arena-scenario/2",
        "humans": [
            {"name": "late", "model": "arenian", "pose": [0.0, 3.0, 0.0], "track": "./late.csv"},
            {"name": "idle", "model": "arenian", "pose": [7.0, 7.0, 0.0]},
        ],
    }
    (directory / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    dynamic = ScenarioView(directory).load().humans
    idle = (7.0 + OFFSET[0], 7.0 + OFFSET[1])

    def late(x: float, y: float) -> tuple[float, float]:
        return (x + OFFSET[0], y + OFFSET[1])

    async def body(stage: Stage) -> None:
        both = sorted(f"{PREFIX}/{name}" for name in ("late", "idle"))
        await stage.play(dynamic)
        assert sorted(stage.simulator.spawned[0]) == both

        await stage.at(0.5, {"idle": idle})
        assert [f"{PREFIX}/late"] in stage.simulator.deleted
        assert len(stage.simulator.spawned) == 1

        peds = await stage.at(1.5, {"idle": idle, "late": late(0.5, 3.0)})
        assert (peds["late"].twist.linear.x, peds["late"].twist.linear.y) == pytest.approx((1.0, 0.0))
        assert peds["late"].animation_state == Pedestrian.WALKING
        assert peds["idle"].animation_state == Pedestrian.IDLE
        assert len(stage.simulator.spawned) == 2
        ((name, entry),) = stage.simulator.spawned[1].items()
        assert name == f"{PREFIX}/late"
        assert entry[1] == pytest.approx(3.0 + OFFSET[1])
        assert OFFSET[0] <= entry[0] < OFFSET[0] + 0.5
        assert stage.simulator.updated[-1] == {f"{PREFIX}/late": pytest.approx(late(0.5, 3.0), abs=1e-6)}

        deletions = stage.simulator.deleted.count([f"{PREFIX}/late"])
        await stage.at(3.0, {"idle": idle})
        assert stage.simulator.deleted.count([f"{PREFIX}/late"]) == deletions + 1

        peds = await stage.at(4.5, {"idle": idle, "late": late(5.0, 2.0)})
        assert peds["late"].pose.orientation.z == pytest.approx(math.sin(-math.pi / 4), abs=1e-4)
        assert len(stage.simulator.spawned) == 3

        await stage.at(6.0, {"idle": idle})

        await stage.play(dynamic)
        await stage.at(0.5, {"idle": idle})
        await stage.at(1.5, {"idle": idle, "late": late(0.5, 3.0)})

    _on_stage(body)


def test_a_stopped_sim_clock_repeats_no_roster(rclpy_context, tmp_path: Path) -> None:
    import yaml
    from arena_simulation_setup.shared.track import Track
    from arena_simulation_setup.tree.World.Scenario import ScenarioView

    def scenario(name: str, entries: list[dict]) -> list:
        directory = tmp_path / "scenarios" / name
        Track.from_segments([(np.array([0.0, 4.0]), np.array([0.0, 4.0]), np.array([3.0, 3.0]), np.array([0.0, 0.0]))]).save(directory / "walker.csv")
        (directory / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", "humans": entries}))
        return ScenarioView(directory).load().humans

    walked = scenario("walked", [{"name": "walker", "model": "arenian", "pose": [0.0, 3.0, 0.0], "track": "./walker.csv"}])
    stood = scenario("stood", [{"name": "idle", "model": "arenian", "pose": [7.0, 7.0, 0.0]}])
    idle = (7.0 + OFFSET[0], 7.0 + OFFSET[1])

    async def quiet(stage: Stage) -> int:
        await asyncio.sleep(0.3)
        count = len(stage.bus)
        await asyncio.sleep(0.6)
        assert len(stage.bus) == count
        return count

    async def body(stage: Stage) -> None:
        await stage.play(walked)
        await stage.at(1.5, {"walker": (1.5 + OFFSET[0], 3.0 + OFFSET[1])})
        ended = await quiet(stage)

        await stage.play(stood)
        assert await quiet(stage) == ended
        await stage.at(0.5, {"idle": idle})
        assert all(ped.name == f"{PREFIX}/idle" for msg in stage.bus[ended:] for ped in msg.pedestrians)
        await quiet(stage)

    _on_stage(body)


def test_prespawned_pedestrian_stands_at_its_first_pose_until_its_track_begins(rclpy_context, tmp_path: Path) -> None:
    import yaml
    from arena_people_msgs.msg import Pedestrian
    from arena_simulation_setup.shared.track import Track
    from arena_simulation_setup.tree.World.Scenario import ScenarioView

    directory = tmp_path / "scenarios" / "early"
    Track.from_segments(
        [
            (np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([3.0, 3.0]), np.array([0.0, 0.0])),
            (np.array([4.0, 5.0]), np.array([5.0, 5.0]), np.array([3.0, 1.0]), np.array([-math.pi / 2, -math.pi / 2])),
        ]
    ).save(directory / "early.csv")
    scenario = {
        "format": "arena-scenario/2",
        "humans": [
            {"name": "early", "model": "arenian", "pose": [0.0, 3.0, 0.0], "track": "./early.csv", "prespawn": True},
            {"name": "idle", "model": "arenian", "pose": [7.0, 7.0, 0.0]},
        ],
    }
    (directory / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    dynamic = ScenarioView(directory).load().humans
    first = (0.0 + OFFSET[0], 3.0 + OFFSET[1])
    idle = (7.0 + OFFSET[0], 7.0 + OFFSET[1])

    async def body(stage: Stage) -> None:
        name = f"{PREFIX}/early"
        await stage.play(dynamic)
        assert sorted(stage.simulator.spawned[0]) == sorted([name, f"{PREFIX}/idle"])

        peds = await stage.at(0.5, {"idle": idle, "early": first})
        assert (peds["early"].twist.linear.x, peds["early"].twist.linear.y) == (0.0, 0.0)
        assert peds["early"].animation_state == Pedestrian.IDLE
        assert [name] not in stage.simulator.deleted
        assert len(stage.simulator.spawned) == 1

        peds = await stage.at(1.5, {"idle": idle, "early": (0.5 + OFFSET[0], 3.0 + OFFSET[1])})
        assert peds["early"].animation_state == Pedestrian.WALKING
        assert len(stage.simulator.spawned) == 1

        await stage.at(3.0, {"idle": idle})
        assert [name] in stage.simulator.deleted

    _on_stage(body)


def test_human_launch_takes_replay_and_starts_nothing(rclpy_context) -> None:
    import os

    import launch
    from ament_index_python.packages import get_package_share_directory
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import Node

    path = os.path.join(get_package_share_directory("task_generator"), "launch", "human", "human.launch.py")

    def started(simulator: str) -> tuple[list[str], list[type]]:
        ctx = launch.LaunchContext()
        ctx.launch_configurations.update({"simulator": simulator, "namespace": "env7/task_generator_node"})
        declared: list[str] = []
        actions: list[type] = []

        def walk(entity) -> None:
            if isinstance(entity, launch.actions.DeclareLaunchArgument):
                declared.append(entity.name)
            if isinstance(entity, Node | launch.actions.IncludeLaunchDescription):
                actions.append(type(entity))
            elif isinstance(entity, launch.LaunchDescription):
                for child in entity.entities:
                    walk(child)
            elif isinstance(entity, launch.Action):
                for child in entity.execute(ctx) or []:
                    walk(child)

        walk(PythonLaunchDescriptionSource(path).get_launch_description(ctx))
        return declared, actions

    declared, actions = started("replay")
    assert sorted(declared) == ["namespace", "simulator"]
    assert actions == []
    assert started("arena")[1] == [launch.actions.IncludeLaunchDescription]
    with pytest.raises(RuntimeError, match="replay"):
        started("recorded")
