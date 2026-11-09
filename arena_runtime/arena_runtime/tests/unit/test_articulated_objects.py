from __future__ import annotations

import asyncio
import math
import typing
import uuid
from pathlib import Path

import attrs
import pytest
import yaml

DT = 0.25
CABINET_LINKS = {
    "body": [[-0.5, 0.5], [-0.3, 0.3], [0.0, 1.0]],
    "door": [[-0.5, 0.5], [-0.32, -0.3], [0.0, 0.5]],
    "drawer": [[-0.45, 0.45], [-0.3, 0.25], [0.6, 0.8]],
}
CABINET = {
    "links": {"body": ["body"], "door": ["door"], "drawer": ["drawer"], "knob": []},
    "joints": [
        {"name": "door", "type": "revolute", "parent": "body", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.6]},
        {"name": "drawer", "type": "prismatic", "parent": "body", "child": "drawer", "axis": [0, -1, 0], "limit": [0.0, 0.4], "velocity": 0.2},
        {"name": "knob", "type": "continuous", "parent": "door", "child": "knob", "origin": [0.0, -0.32, 0.25], "axis": [0, 1, 0]},
    ],
    "states": {"closed": {"door": 0.0, "drawer": 0.0}, "open": {"door": 1.5, "drawer": 0.4}},
    "swept": [[-0.5, 0.52], [-1.32, 0.3], [0.0, 1.0]],
}
CONSOLE = {
    "links": {"base": ["base"], "door": [], "drawer": [], "knob": [], "wheel": [], "dial": []},
    "joints": [
        {"name": "door", "type": "revolute", "parent": "base", "child": "door", "limit": [0.0, 1.6]},
        {"name": "drawer", "type": "prismatic", "parent": "base", "child": "drawer", "axis": [0, -1, 0], "limit": [0.0, 0.4]},
        {"name": "knob", "type": "revolute", "parent": "base", "child": "knob", "limit": [0.0, 2.0]},
        {"name": "wheel", "type": "continuous", "parent": "base", "child": "wheel"},
        {"name": "dial", "type": "revolute", "parent": "base", "child": "dial", "limit": [0.0, 3.0]},
    ],
    "states": {"closed": {"door": 0.0, "drawer": 0.0}, "open": {"door": 1.5, "drawer": 0.4}},
    "detents": {"dial": {"off": 0.0, "low": 1.0, "high": 2.0}},
}
DOOR_LINKS = {
    "frame": [[-0.5, 0.5], [-0.05, 0.05], [0.0, 2.1]],
    "leaf": [[-0.45, 0.45], [-0.02, 0.02], [0.0, 2.0]],
    "lever": [[-0.4, -0.28], [0.02, 0.06], [0.98, 1.02]],
}
DOOR = {
    "links": {"frame": ["frame"], "leaf": ["leaf"], "lever": ["lever"]},
    "joints": [
        {"name": "hinge", "type": "revolute", "parent": "frame", "child": "leaf", "origin": [0.45, 0.0, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.6], "role": "door", "latch": {"by": "handle", "release": 0.6}},
        {"name": "handle", "type": "revolute", "parent": "leaf", "child": "lever", "origin": [-0.35, 0.04, 1.0], "axis": [0, 1, 0], "limit": [0.0, 0.8], "role": "handle"},
    ],
    "states": {"closed": {"hinge": 0.0}, "open": {"hinge": 1.5}},
}


@pytest.fixture
def assets(tmp_path: Path):
    """Asset root with an articulated cabinet and console and one rigid crate, the model names unique per test."""
    from arena_simulation_setup.tree import DynamicPaths

    suffix = uuid.uuid4().hex[:8]
    cabinet = tmp_path / "Kitchen" / "Object" / f"Cabinet{suffix}"
    for link, box in CABINET_LINKS.items():
        (cabinet / "links" / link).mkdir(parents=True)
        (cabinet / "links" / link / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    (cabinet / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": CABINET_LINKS["body"], "articulation": CABINET}))
    console = tmp_path / "Kitchen" / "Object" / f"Console{suffix}"
    (console / "links" / "base").mkdir(parents=True)
    (console / "links" / "base" / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": CABINET_LINKS["body"]}))
    (console / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": CABINET_LINKS["body"], "articulation": CONSOLE}))
    door = tmp_path / "Kitchen" / "Object" / f"Door{suffix}"
    for link, box in DOOR_LINKS.items():
        (door / "links" / link).mkdir(parents=True)
        (door / "links" / link / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    (door / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": DOOR_LINKS["frame"], "articulation": DOOR}))
    crate = tmp_path / "Kitchen" / "Object" / f"Crate{suffix}"
    crate.mkdir(parents=True)
    (crate / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.4]]}))
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield {"cabinet": f"Kitchen/Cabinet{suffix}", "console": f"Kitchen/Console{suffix}", "door": f"Kitchen/Door{suffix}", "crate": f"Kitchen/Crate{suffix}"}
    DynamicPaths.ARENA.path = previous


def _realizer():
    from arena_runtime.env.realizer import Realizer

    return Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))


def _obstacle(name: str, model: str, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, **kwargs: typing.Any):
    from arena_runtime.shared import Obstacle, Orientation, Pose, Position

    return _realizer().realize(Obstacle(name=name, model=model, pose=Pose(position=Position(x, y, 0.0), orientation=Orientation.from_yaw(yaw)), **kwargs))


@pytest.fixture
def on_dummy_sim():

    def run(body: typing.Callable[..., typing.Awaitable[None]]) -> None:
        from arena_rclpy_mixins import ArenaMixinNode
        from arena_rclpy_mixins.shared import Namespace

        from arena_runtime.sim.dummy_simulator import DummySimulator

        class RecordingSimulator(DummySimulator):
            def __init__(self, **kwargs: typing.Any) -> None:
                super().__init__(**kwargs)
                self.calls: list[tuple] = []

            async def obstacle_spawn(self, obstacles):
                self.calls.append(("spawn", list(obstacles)))
                return await super().obstacle_spawn(obstacles)

            async def obstacle_move(self, obstacles):
                self.calls.append(("move", list(obstacles)))
                return await super().obstacle_move(obstacles)

            async def obstacle_delete(self, obstacles):
                self.calls.append(("delete", list(obstacles)))
                return await super().obstacle_delete(obstacles)

            async def _move_light(self, light, position, yaw):
                self.calls.append(("light", light.name, position, yaw))
                return await super()._move_light(light, position, yaw)

        async def main() -> None:
            suffix = uuid.uuid4().hex[:8]
            node = ArenaMixinNode(f"articulated_{suffix}", namespace=f"/articulated_{suffix}")
            sim = RecordingSimulator(node=node, namespace=Namespace(f"/articulated_{suffix}"), realizer=_realizer())
            try:
                await body(sim)
            finally:
                await sim.stop_mechanisms()
                node.destroy_node()

        asyncio.run(main())

    return run


def _names(obstacles) -> list[str]:
    return [obstacle.name for obstacle in obstacles]


def _yaw_gap(a: float, b: float) -> float:
    return abs(math.remainder(a - b, 2 * math.pi))


def _snapshot(sim, entity: str = "env_0/cabinet"):
    return next(snap for snap in sim.semantics_snapshot() if snap.entity == entity)


def _register_jackal(sim) -> tuple[str, float]:
    from arena_robots.Robot import RobotIdentifier
    from arena_runtime.shared import Pose, Robot

    from arena_runtime.sim._interface import robot_footprint_radius

    model = RobotIdentifier.parse("jackal")
    robot = Robot(name="jackal", pose=Pose(), model=model, adapters={}, extra={})
    params = model.resolve_sync().model_params
    sim._register_agent_robot(robot, params)
    return robot.frame.tf(params.base_frame), robot_footprint_radius(params)


def _place(sim, frame: str, x: float, y: float, stamp: int) -> None:
    import geometry_msgs.msg

    t = geometry_msgs.msg.TransformStamped()
    t.header.frame_id = "map"
    t.header.stamp.sec = stamp
    t.child_frame_id = frame
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.w = 1.0
    sim._mechanism_tf_buffer.set_transform(t, "test")


def test_link_obstacles_carry_the_link_names_scale_and_poses(assets):
    from arena_simulation_setup.utils.geometry import Scale

    from arena_runtime.sim._articulation import resolve_articulated

    cabinet = _obstacle("cabinet", assets["cabinet"], x=1.0, y=2.0, yaw=math.pi / 2, scale=Scale(2.0, 2.0, 1.0), joints={"door": math.pi / 2, "drawer": 0.2})
    cabinet.level = "1"
    runtime = asyncio.run(resolve_articulated(cabinet, "env_0/cabinet"))
    assert runtime is not None
    assert runtime.initial == pytest.approx({"door": math.pi / 2, "drawer": 0.2, "knob": 0.0})
    links = {part.name: part for part in runtime.link_obstacles()}
    assert list(links) == ["cabinet__body", "cabinet__door", "cabinet__drawer"]
    expected = {"cabinet__body": (1.0, 2.0, math.pi / 2), "cabinet__door": (2.6, 2.4, math.pi), "cabinet__drawer": (1.4, 2.0, math.pi / 2)}
    for name, (x, y, yaw) in expected.items():
        part = links[name]
        assert (part.pose.position.x, part.pose.position.y, part.pose.position.z) == pytest.approx((x, y, 0.0))
        assert _yaw_gap(part.pose.orientation.to_yaw(), yaw) == pytest.approx(0.0, abs=1e-9)
        assert part.sim_path == f"env_0/{name}"
        assert part.scale == Scale(2.0, 2.0, 1.0)
        assert part.level == "1"
        assert part.light is None
        assert part.humansim == {}
        assert asyncio.run(part.model.resolve()).path.parts[-2:] == ("links", name.removeprefix("cabinet__"))


def test_a_rigid_or_unresolvable_object_is_not_articulated(assets):
    from arena_runtime.sim._articulation import resolve_articulated

    assert asyncio.run(resolve_articulated(_obstacle("crate", assets["crate"]), "env_0/crate")) is None
    assert asyncio.run(resolve_articulated(_obstacle("ghost", "Kitchen/NoSuchCabinet"), "env_0/ghost")) is None


def test_plain_and_articulated_obstacles_spawn_move_and_delete_in_one_batch_each(assets, on_dummy_sim):
    import attrs
    from arena_runtime.shared import Pose, Position

    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"])
        crate = _obstacle("crate", assets["crate"], x=3.0)
        assert list(await sim.spawn_objects([cabinet, crate])) == [True, True]
        assert [(verb, _names(obstacles)) for verb, obstacles in sim.calls] == [("spawn", ["cabinet__body", "cabinet__door", "cabinet__drawer", "crate"])]
        assert set(sim._articulated) == {"cabinet"}
        assert _snapshot(sim).kind == "articulated"
        assert sim._mechanism_loop_task is not None

        sim.calls.clear()
        moved = attrs.evolve(cabinet, pose=Pose(position=Position(5.0, 0.0, 0.0)), joints="open")
        assert list(await sim.move_objects([moved, crate])) == [True, True]
        ((verb, obstacles),) = sim.calls
        assert (verb, _names(obstacles)) == ("move", ["cabinet__body", "cabinet__door", "cabinet__drawer", "crate"])
        assert obstacles[2].pose.position.x == pytest.approx(5.0)
        assert obstacles[2].pose.position.y == pytest.approx(-0.4)
        runtime = sim._articulated["cabinet"]
        assert runtime.current == runtime.target == runtime.initial == pytest.approx({"door": 1.5, "drawer": 0.4, "knob": 0.0})

        sim.calls.clear()
        assert list(await sim.delete_objects([moved, crate])) == [True, True]
        assert [(verb, _names(obstacles)) for verb, obstacles in sim.calls] == [("delete", ["cabinet__body", "cabinet__door", "cabinet__drawer", "crate"])]
        assert sim._articulated == {}
        assert sim.semantics_snapshot() == []

    on_dummy_sim(body)


def test_a_level_object_keeps_its_level_sim_path_and_takes_the_env_prefixed_semantic_name(assets, on_dummy_sim):
    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"])
        cabinet.sim_path = "env_0/cabinet/1"
        await sim.spawn_objects([cabinet])
        ((_verb, obstacles),) = sim.calls
        assert [obstacle.sim_path for obstacle in obstacles] == ["env_0/cabinet/1__body", "env_0/cabinet/1__door", "env_0/cabinet/1__drawer"]
        assert [snap.entity for snap in sim.semantics_snapshot()] == ["env_0/cabinet"]

    on_dummy_sim(body)


def test_an_unknown_initial_state_fails_only_its_obstacle(assets, on_dummy_sim):
    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"], joints="ajar")
        crate = _obstacle("crate", assets["crate"])
        assert list(await sim.spawn_objects([cabinet, crate])) == [False, True]
        assert [(verb, _names(obstacles)) for verb, obstacles in sim.calls] == [("spawn", ["crate"])]
        assert sim._articulated == {}

    on_dummy_sim(body)


def test_joints_reach_a_written_target_state_at_their_speed(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        sim.calls.clear()
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        runtime = sim._articulated["cabinet"]
        await shim._tick(sim, DT)
        assert runtime.current == pytest.approx({"door": 0.25, "drawer": 0.05, "knob": 0.0})
        ((verb, obstacles),) = sim.calls
        assert (verb, _names(obstacles)) == ("move", ["cabinet__door", "cabinet__drawer"])
        assert _snapshot(sim).predicates["moving"]
        for _ in range(5):
            await shim._tick(sim, DT)
        assert runtime.current["door"] == pytest.approx(1.5)
        assert _names(sim.calls[-1][1]) == ["cabinet__door", "cabinet__drawer"]
        await shim._tick(sim, DT)
        assert _names(sim.calls[-1][1]) == ["cabinet__drawer"]
        await shim._tick(sim, DT)
        assert runtime.current == pytest.approx({"door": 1.5, "drawer": 0.4, "knob": 0.0})
        calls = len(sim.calls)
        await shim._tick(sim, DT)
        assert len(sim.calls) == calls
        snapshot = _snapshot(sim)
        assert snapshot.discrete == {"state": "open", "target": "open"}
        assert snapshot.predicates == {"closed": False, "open": True, "moving": False, "blocked": False}

    on_dummy_sim(body)


def _door_light(link: str, x: float, y: float, yaw: float):
    from arena_simulation_setup.shared import Light
    from arena_runtime.shared import Orientation, Pose, Position

    origin = Pose(position=Position(x, y, 0.0), orientation=Orientation.from_yaw(yaw))
    return Light(name=f"cabinet_{link or 'free'}_light", fixture="downlight", lumens=100.0, owner="cabinet", link=link, pose=origin * Pose(position=Position(0.0, -0.31, 0.4), orientation=Orientation.from_euler((0.3, 0.0, 0.0))))


def _light_moves(sim) -> list[tuple]:
    return [call for call in sim.calls if call[0] == "light"]


def test_a_light_on_the_door_link_swings_with_the_door(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    x, y, yaw = 1.0, 2.0, 0.7

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"], x=x, y=y, yaw=yaw)])
        light = _door_light("door", x, y, yaw)
        assert await sim.spawn_lights([light])
        sim.calls.clear()
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        runtime = sim._articulated["cabinet"]
        for _ in range(8):
            await shim._tick(sim, DT)
        assert runtime.current["door"] == pytest.approx(1.5)
        angle = 1.5
        px, py = 0.0 - 0.5, -0.31 + 0.3
        qx, qy = 0.5 + math.cos(angle) * px - math.sin(angle) * py, -0.3 + math.sin(angle) * px + math.cos(angle) * py
        expected = (x + math.cos(yaw) * qx - math.sin(yaw) * qy, y + math.sin(yaw) * qx + math.cos(yaw) * qy, 0.4)
        _verb, name, position, turn = _light_moves(sim)[-1]
        assert name == light.name
        assert (position.x, position.y, position.z) == pytest.approx(expected)
        assert _yaw_gap(turn, angle) == pytest.approx(0.0, abs=1e-9)

    on_dummy_sim(body)


def test_lights_off_the_moving_links_stay_put_while_the_door_swings(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"], x=1.0, y=2.0, yaw=0.7)])
        assert await sim.spawn_lights([_door_light("", 1.0, 2.0, 0.7), _door_light("body", 1.0, 2.0, 0.7)])
        sim.calls.clear()
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        for _ in range(8):
            await shim._tick(sim, DT)
        assert sim._articulated["cabinet"].current["door"] == pytest.approx(1.5)
        assert _light_moves(sim) == []

    on_dummy_sim(body)


def test_an_agent_in_the_swing_holds_the_object_until_it_leaves(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        frame, radius = _register_jackal(sim)
        assert 0.2 < radius < 0.45
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        sim.calls.clear()
        _place(sim, frame, 0.2, -1.2, stamp=1)
        assert sim.set_semantic_value("env_0/cabinet", "q.door", "1.5")
        runtime = sim._articulated["cabinet"]
        for _ in range(8):
            await shim._tick(sim, DT)
        held = runtime.current["door"]
        assert 0.0 < held < 1.5
        assert runtime.blocked
        assert _snapshot(sim).predicates["blocked"]
        assert {name for _verb, obstacles in sim.calls for name in _names(obstacles)} == {"cabinet__door"}
        await shim._tick(sim, DT)
        assert runtime.current["door"] == held
        _place(sim, frame, 10.0, 10.0, stamp=2)
        for _ in range(8):
            await shim._tick(sim, DT)
        assert runtime.current["door"] == pytest.approx(1.5)
        assert not runtime.blocked
        assert runtime.current["drawer"] == 0.0

    on_dummy_sim(body)


def test_semantic_writes_take_state_names_and_clamped_joint_values(assets, on_dummy_sim):
    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        snapshot = _snapshot(sim)
        assert snapshot.discrete == {"state": "closed", "target": "closed"}
        assert snapshot.continuous == {"openness.door": 0.0, "openness.drawer": 0.0, "angle.knob": 0.0, "q.door": 0.0, "q.drawer": 0.0, "q.knob": 0.0}
        assert snapshot.predicates == {"closed": True, "open": False, "moving": False, "blocked": False}
        runtime = sim._articulated["cabinet"]
        assert sim.set_semantic_value("env_0/cabinet", "q.drawer", "9")
        assert runtime.target == {"door": 0.0, "drawer": 0.4, "knob": 0.0}
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        assert runtime.target == {"door": 1.5, "drawer": 0.4, "knob": 0.0}
        assert sim.set_semantic_value("env_0/cabinet", "q.knob", str(3 * math.pi / 2))
        assert runtime.target["knob"] == pytest.approx(-math.pi / 2)
        assert _snapshot(sim).discrete == {"state": "closed", "target": "open"}
        for field, value, reason in (
            ("target", "ajar", "malformed value"),
            ("q.drawer", "wide", "malformed value"),
            ("q.drawer", "nan", "malformed value"),
            ("door", "1.0", "field not writable"),
            ("q.lid", "1.0", "field not writable"),
            ("angle.knob", "1.0", "field not writable"),
            ("moving", "true", "field not writable"),
        ):
            with pytest.raises(ValueError, match=reason):
                sim.set_semantic_value("env_0/cabinet", field, value)
        assert runtime.target == {"door": 1.5, "drawer": 0.4, "knob": pytest.approx(-math.pi / 2)}

    on_dummy_sim(body)


def test_reset_returns_the_joints_to_the_initial_values_and_places_the_links_again(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"], joints={"drawer": "open"})])
        runtime = sim._articulated["cabinet"]
        assert sim.set_semantic_value("env_0/cabinet", "target", "closed")
        await shim._tick(sim, DT)
        assert runtime.current["drawer"] == pytest.approx(0.35)
        sim.reset_semantics()
        assert runtime.current == runtime.target == {"door": 0.0, "drawer": 0.4, "knob": 0.0}
        sim.calls.clear()
        await shim._tick(sim, DT)
        ((verb, obstacles),) = sim.calls
        assert (verb, _names(obstacles)) == ("move", ["cabinet__body", "cabinet__door", "cabinet__drawer"])
        assert obstacles[2].pose.position.y == pytest.approx(-0.4)

    on_dummy_sim(body)


def test_an_episode_reset_move_starts_from_the_new_joints(assets, on_dummy_sim):
    import attrs

    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"])
        await sim.spawn_objects([cabinet])
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        await sim.move_objects([attrs.evolve(cabinet, joints={"door": "open"})])
        runtime = sim._articulated["cabinet"]
        assert runtime.current == runtime.target == runtime.initial == {"door": 1.5, "drawer": 0.0, "knob": 0.0}
        snapshot = _snapshot(sim)
        assert snapshot.discrete == {"state": "", "target": ""}
        assert snapshot.predicates == {"closed": False, "open": False, "moving": False, "blocked": False}
        sim.reset_semantics()
        assert runtime.current == {"door": 1.5, "drawer": 0.0, "knob": 0.0}
        await sim.move_objects([cabinet])
        assert runtime.current == {"door": 0.0, "drawer": 0.0, "knob": 0.0}

    on_dummy_sim(body)


def _console_semantics(model: str):
    from arena_runtime.sim._articulation import resolve_articulated
    from arena_runtime.sim._semantics import ArticulatedSemantics

    runtime = asyncio.run(resolve_articulated(_obstacle("console", model), "env_0/console"))
    assert runtime is not None
    return runtime, ArticulatedSemantics("env_0/console", runtime)


def test_the_snapshot_reads_each_joint_in_its_family(assets):
    runtime, semantics = _console_semantics(assets["console"])
    snapshot = semantics.snapshot()
    assert snapshot.discrete == {"state": "closed", "target": "closed", "setting.dial": "off"}
    assert snapshot.continuous == {
        "openness.door": 0.0,
        "openness.drawer": 0.0,
        "pos.dial": 0.0,
        "pos.knob": 0.0,
        "angle.wheel": 0.0,
        "q.dial": 0.0,
        "q.door": 0.0,
        "q.drawer": 0.0,
        "q.knob": 0.0,
        "q.wheel": 0.0,
    }
    assert snapshot.predicates == {"closed": True, "open": False, "moving": False, "blocked": False}
    runtime.current = {"door": 0.75, "drawer": 0.1, "knob": 1.5, "wheel": -math.pi / 2, "dial": 1.0}
    snapshot = semantics.snapshot()
    assert snapshot.discrete == {"state": "", "target": "closed", "setting.dial": "low"}
    assert snapshot.continuous == pytest.approx(
        {
            "openness.door": 0.5,
            "openness.drawer": 0.25,
            "pos.dial": 1.0 / 3.0,
            "pos.knob": 0.75,
            "angle.wheel": 3 * math.pi / 2,
            "q.dial": 1.0,
            "q.door": 0.75,
            "q.drawer": 0.1,
            "q.knob": 1.5,
            "q.wheel": -math.pi / 2,
        }
    )
    assert all(semantics.has_field(field) for field in (*snapshot.discrete, *snapshot.continuous, *snapshot.predicates, "openness"))
    assert not semantics.has_field("door")
    assert not semantics.is_continuous("setting.dial") and not semantics.is_continuous("state") and not semantics.is_continuous("target")
    assert all(semantics.is_continuous(field) for field in snapshot.continuous)


def test_changes_fire_at_the_family_quanta(assets):
    from arena_runtime.sim._semantics import _diff

    runtime, semantics = _console_semantics(assets["console"])
    previous = semantics.snapshot()
    runtime.current = {"door": 0.006, "drawer": 0.0008, "knob": 0.0, "wheel": 0.004, "dial": 0.0}
    snapshot = semantics.snapshot()
    assert {change.field for change in _diff(previous, snapshot, semantics.quantum)} == {"q.door", "q.drawer", "state", "closed", "moving"}
    previous = snapshot
    runtime.current = {"door": 0.03, "drawer": 0.0008, "knob": 0.0, "wheel": 0.007, "dial": 0.012}
    changes = {change.field: change for change in _diff(previous, semantics.snapshot(), semantics.quantum)}
    assert set(changes) == {"openness.door", "q.door", "angle.wheel", "q.wheel", "q.dial", "setting.dial"}
    assert (changes["setting.dial"].previous, changes["setting.dial"].current) == ("off", "")


def test_family_writes_drive_the_target_and_a_detent_snaps(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("console", assets["console"])])
        runtime = sim._articulated["console"]
        assert sim.set_semantic_value("env_0/console", "openness", "0.5")
        assert runtime.target == pytest.approx({"door": 0.75, "drawer": 0.2, "knob": 0.0, "wheel": 0.0, "dial": 0.0})
        assert sim.set_semantic_value("env_0/console", "setting.dial", "high")
        assert sim.set_semantic_value("env_0/console", "pos.knob", "0.25")
        assert runtime.target == pytest.approx({"door": 0.75, "drawer": 0.2, "knob": 0.5, "wheel": 0.0, "dial": 2.0})
        for _ in range(8):
            await shim._tick(sim, DT)
        snapshot = _snapshot(sim, "env_0/console")
        assert snapshot.discrete["setting.dial"] == "high"
        assert (snapshot.continuous["openness.door"], snapshot.continuous["openness.drawer"], snapshot.continuous["pos.knob"]) == pytest.approx((0.5, 0.5, 0.25))
        assert not snapshot.predicates["moving"]
        for field, value, reason in (
            ("openness", "1.5", "malformed value"),
            ("openness.door", "half", "malformed value"),
            ("setting.dial", "max", "malformed value"),
            ("pos.knob", "nan", "malformed value"),
            ("setting.knob", "off", "field not writable"),
            ("openness.knob", "0.5", "field not writable"),
            ("angle.wheel", "1.0", "field not writable"),
        ):
            with pytest.raises(ValueError, match=reason):
                sim.set_semantic_value("env_0/console", field, value)
        assert runtime.target == pytest.approx({"door": 0.75, "drawer": 0.2, "knob": 0.5, "wheel": 0.0, "dial": 2.0})

    on_dummy_sim(body)


def test_disc_hits_a_convex_hull_inside_and_within_its_radius():
    from arena_runtime.sim._articulation import disc_hits, hull

    square = hull([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)])
    assert sorted(square) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    assert disc_hits(square, (0.5, 0.5), 0.01)
    assert disc_hits(square, (1.2, 0.5), 0.3)
    assert not disc_hits(square, (1.2, 0.5), 0.1)
    assert not disc_hits([], (0.0, 0.0), 1.0)


PHYSICS_DT = 0.05


@pytest.fixture
def on_physics_sim():
    """A dummy simulator with articulated physics: drives reach their reference one call later, pushes and holds set from the test."""

    def run(body: typing.Callable[..., typing.Awaitable[None]]) -> None:
        from arena_rclpy_mixins import ArenaMixinNode
        from arena_rclpy_mixins.shared import Namespace

        from arena_runtime.sim.dummy_simulator import DummySimulator

        class PhysicsSimulator(DummySimulator):
            ARTICULATED_PHYSICS = True

            def __init__(self, **kwargs: typing.Any) -> None:
                super().__init__(**kwargs)
                self.calls: list[tuple] = []
                self.joints: dict[str, dict[str, float]] = {}
                self.ctrl: dict[str, dict[str, float]] = {}
                self.held: dict[tuple[str, str], float] = {}

            async def obstacle_spawn(self, obstacles):
                self.calls.append(("spawn", list(obstacles)))
                return await super().obstacle_spawn(obstacles)

            async def obstacle_move(self, obstacles):
                self.calls.append(("move", list(obstacles)))
                return await super().obstacle_move(obstacles)

            async def obstacle_delete(self, obstacles):
                self.calls.append(("delete", list(obstacles)))
                return await super().obstacle_delete(obstacles)

            async def _articulated_spawn(self, objects):
                self.calls.append(("articulated", [runtime.name for runtime in objects]))
                for runtime in objects:
                    self.joints[runtime.name] = dict(runtime.current)
                    self.ctrl[runtime.name] = {}
                return [True] * len(objects)

            async def _articulated_drive(self, commands):
                self.calls.append(("drive", list(commands)))
                readings = []
                for command in commands:
                    name = command.runtime.name
                    joints = self.joints.get(name)
                    if joints is None:
                        readings.append(None)
                        continue
                    for joint, value in self.ctrl[name].items():
                        joints[joint] = self.held.get((name, joint), value)
                    for joint in command.place | command.hold:
                        joints[joint] = command.reference[joint]
                    self.ctrl[name] = {joint: command.reference[joint] for joint in command.drive}
                    readings.append(dict(joints))
                return readings

            def push(self, name: str, joint: str, value: float) -> None:
                self.joints[name][joint] = value

        async def main() -> None:
            suffix = uuid.uuid4().hex[:8]
            node = ArenaMixinNode(f"articulated_physics_{suffix}", namespace=f"/articulated_physics_{suffix}")
            sim = PhysicsSimulator(node=node, namespace=Namespace(f"/articulated_physics_{suffix}"), realizer=_realizer())
            try:
                await body(sim)
            finally:
                await sim.stop_mechanisms()
                node.destroy_node()

        asyncio.run(main())

    return run


async def _settle(sim, runtime, ticks: int = 200) -> int:
    from arena_runtime.sim import _mechanism_shim as shim

    for tick in range(ticks):
        await shim._tick(sim, PHYSICS_DT)
        if not runtime.moving:
            return tick
    return ticks


def test_a_physics_simulator_spawns_the_whole_object_with_derived_inertias_and_dynamics(assets, on_physics_sim):
    from arena_runtime.sim._articulation import ANGULAR, LINEAR, MASS_RANGE

    async def body(sim):
        assert list(await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"]), _obstacle("crate", assets["crate"])])) == [True, True]
        assert [(verb, _names(items) if verb == "spawn" else items) for verb, items in sim.calls] == [("spawn", ["crate"]), ("articulated", ["cabinet"])]
        runtime = sim._articulated["cabinet"]
        assert runtime.physical
        assert runtime.inertias["body"].mass == MASS_RANGE[1]
        assert runtime.inertias["door"].mass == pytest.approx(150.0 * 1.0 * 0.02 * 0.5)
        assert runtime.inertias["door"].center == pytest.approx((0.0, -0.31, 0.25))
        assert runtime.inertias["knob"].center == pytest.approx((0.0, -0.32, 0.25))
        assert set(runtime.dynamics) == {"door", "drawer", "knob"}
        assert runtime.dynamics["door"].friction == ANGULAR.friction
        assert runtime.dynamics["drawer"].friction == LINEAR.friction
        assert runtime.dynamics["door"].effort == ANGULAR.effort
        assert runtime.dynamics["door"].stiffness == pytest.approx(ANGULAR.effort / ANGULAR.saturation)
        assert _snapshot(sim).discrete == {"state": "closed", "target": "closed"}

    on_physics_sim(body)


def test_gravity_raises_the_friction_that_holds_a_lid_and_the_effort_that_lifts_it():
    from arena_simulation_setup.utils.articulation import structure_articulation

    from arena_runtime.sim._articulation import ANGULAR, EFFORT_PER_FRICTION, GRAVITY, HOLD_MARGIN, LinkInertia, joint_dynamics

    articulation = structure_articulation(
        {
            "links": {"box": [], "lid": [], "flap": []},
            "joints": [
                {"name": "lid", "type": "revolute", "parent": "box", "child": "lid", "origin": [0.0, 0.2, 0.5], "axis": [1, 0, 0], "limit": [0.0, 1.5]},
                {"name": "flap", "type": "revolute", "parent": "box", "child": "flap", "origin": [0.3, 0.0, 0.5], "axis": [0, 0, 1], "limit": [0.0, 1.5]},
            ],
        }
    )
    assert articulation is not None
    inertias = {
        "box": LinkInertia(5.0, (0.0, 0.0, 0.25), (0.1, 0.1, 0.1)),
        "lid": LinkInertia(2.0, (0.0, 0.0, 0.5), (0.01, 0.01, 0.01)),
        "flap": LinkInertia(1.0, (0.3, -0.3, 0.5), (0.01, 0.01, 0.01)),
    }
    dynamics = joint_dynamics(articulation, inertias, None)
    assert dynamics["lid"].friction == pytest.approx(HOLD_MARGIN * 2.0 * GRAVITY * 0.2)
    assert dynamics["lid"].effort == pytest.approx(EFFORT_PER_FRICTION * (dynamics["lid"].friction + ANGULAR.damping * articulation.joint_map["lid"].speed))
    assert dynamics["flap"].friction == ANGULAR.friction
    assert dynamics["flap"].drive_damping == pytest.approx(2.0 * math.sqrt(dynamics["flap"].stiffness * (1.0 * 0.09 + 0.01 + ANGULAR.armature)))


def test_authored_dynamics_win_over_role_defaults_which_win_over_kind_defaults():
    from arena_simulation_setup.utils.articulation import structure_articulation

    from arena_runtime.sim._articulation import EFFORT_PER_FRICTION, GRAVITY, LINEAR, LINEAR_ROLES, LinkInertia, joint_dynamics

    articulation = structure_articulation(
        {
            "links": {"body": [], "drawer": [], "rail": [], "door": [], "flap": []},
            "joints": [
                {"name": "drawer", "type": "prismatic", "parent": "body", "child": "drawer", "axis": [0, -1, 0], "limit": [0.0, 0.4], "role": "drawer"},
                {"name": "rail", "type": "prismatic", "parent": "body", "child": "rail", "axis": [1, 0, 0], "limit": [0.0, 0.2]},
                {
                    "name": "door",
                    "type": "revolute",
                    "parent": "body",
                    "child": "door",
                    "axis": [0, 0, 1],
                    "limit": [0.0, 1.5],
                    "role": "door",
                    "dynamics": {"friction": 0.7, "damping": 3.0, "armature": 0.05, "spring": {"stiffness": 4.0, "rest": 0.0}},
                },
                {"name": "flap", "type": "prismatic", "parent": "body", "child": "flap", "axis": [0, 1, 0], "limit": [0.0, 0.1], "role": "drawer", "dynamics": {"friction_coefficient": 0.5, "damping": 1.0}},
            ],
        }
    )
    assert articulation is not None
    inertias = {link: LinkInertia(2.0, (0.0, 0.0, 0.0), (0.01, 0.01, 0.01)) for link in articulation.links}
    dynamics = joint_dynamics(articulation, inertias, None)
    assert (dynamics["drawer"].friction, dynamics["drawer"].damping) == LINEAR_ROLES["drawer"]
    assert (dynamics["rail"].friction, dynamics["rail"].damping, dynamics["rail"].armature) == (LINEAR.friction, LINEAR.damping, LINEAR.armature)
    door = dynamics["door"]
    assert (door.friction, door.damping, door.armature, door.spring_stiffness, door.spring_rest) == (0.7, 3.0, 0.05, 4.0, 0.0)
    assert door.effort == pytest.approx(EFFORT_PER_FRICTION * (0.7 + 4.0 * 1.5 + 3.0 * articulation.joint_map["door"].speed))
    assert (dynamics["flap"].friction, dynamics["flap"].damping) == (pytest.approx(0.5 * 2.0 * GRAVITY), 1.0)


def test_a_link_material_fills_an_unset_friction_from_the_other_and_restitution_with_zero():
    from arena_simulation_setup.utils.articulation import Material

    from arena_runtime.sim._articulation import SURFACE_FRICTION, surface

    assert surface(Material(static_friction=0.6, dynamic_friction=0.4, restitution=0.2)) == (0.6, 0.4, 0.2)
    assert surface(Material(dynamic_friction=0.3)) == (0.3, 0.3, 0.0)
    assert surface(Material(restitution=0.5)) == (SURFACE_FRICTION, SURFACE_FRICTION, 0.5)


def test_a_physical_drive_ramps_to_the_target_releases_and_snaps_onto_it(assets, on_physics_sim):
    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        runtime = sim._articulated["cabinet"]
        sim.calls.clear()
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        assert runtime.driving == {"door", "drawer"}
        ticks = await _settle(sim, runtime)
        assert 0.4 / 0.2 / PHYSICS_DT <= ticks < 0.4 / 0.2 / PHYSICS_DT + 5
        assert runtime.current == {"door": 1.5, "drawer": 0.4, "knob": 0.0}
        assert runtime.driving == set()
        commands = [command for verb, items in sim.calls if verb == "drive" for command in items]
        assert all(abs(command.reference["door"] - previous.reference["door"]) <= PHYSICS_DT + 1e-9 for previous, command in zip(commands, commands[1:], strict=False))
        assert any(command.place == frozenset({"door"}) for command in commands)
        assert {verb for verb, _items in sim.calls} == {"drive"}
        snapshot = _snapshot(sim)
        assert snapshot.discrete == {"state": "open", "target": "open"}
        assert snapshot.predicates == {"closed": False, "open": True, "moving": False, "blocked": False}

    on_physics_sim(body)


def test_a_pushed_joint_moves_its_target_with_it_and_no_drive_pulls_it_back(assets, on_physics_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        runtime = sim._articulated["cabinet"]
        sim.push("cabinet", "door", 0.7)
        await shim._tick(sim, PHYSICS_DT)
        assert runtime.current["door"] == pytest.approx(0.7)
        assert runtime.target["door"] == pytest.approx(0.7)
        assert runtime.moving
        await shim._tick(sim, PHYSICS_DT)
        assert not runtime.moving
        assert sim.calls[-1][1][0].drive == frozenset()
        assert _snapshot(sim).continuous["openness.door"] == pytest.approx(0.7 / 1.5)

    on_physics_sim(body)


def test_a_held_joint_reads_blocked_until_it_is_let_go(assets, on_physics_sim):
    async def body(sim):
        from arena_runtime.sim import _mechanism_shim as shim

        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        runtime = sim._articulated["cabinet"]
        sim.held[("cabinet", "door")] = 0.3
        assert sim.set_semantic_value("env_0/cabinet", "q.door", "1.5")
        for _ in range(20):
            await shim._tick(sim, PHYSICS_DT)
        assert runtime.current["door"] == pytest.approx(0.3)
        assert runtime.blocked
        assert _snapshot(sim).predicates["blocked"]
        assert sim.calls[-1][1][0].reference["door"] == pytest.approx(0.3 + 0.25)
        del sim.held[("cabinet", "door")]
        await _settle(sim, runtime)
        assert runtime.current["door"] == 1.5
        assert not runtime.blocked

    on_physics_sim(body)


def test_a_latched_door_holds_closed_against_a_push_until_its_handle_is_pressed(assets, on_physics_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("door", assets["door"])])
        runtime = sim._articulated["door"]
        await shim._tick(sim, PHYSICS_DT)
        sim.push("door", "hinge", 0.03)
        await shim._tick(sim, PHYSICS_DT)
        held = sim.calls[-1][1][0]
        assert ("hinge" in held.drive, "hinge" in held.hold, "hinge" in held.place, held.reference["hinge"]) == (True, True, False, 0.0)
        assert runtime.current["hinge"] == 0.0
        sim.push("door", "handle", 0.5)
        await shim._tick(sim, PHYSICS_DT)
        await shim._tick(sim, PHYSICS_DT)
        assert sim.calls[-1][1][0].drive == frozenset()
        sim.push("door", "hinge", 0.03)
        await shim._tick(sim, PHYSICS_DT)
        assert runtime.current["hinge"] == pytest.approx(0.03)
        sim.push("door", "hinge", 0.7)
        sim.push("door", "handle", 0.0)
        for _ in range(3):
            await shim._tick(sim, PHYSICS_DT)
        assert runtime.current["hinge"] == pytest.approx(0.7)
        assert sim.calls[-1][1][0].drive == frozenset()
        sim.push("door", "hinge", 0.02)
        await shim._tick(sim, PHYSICS_DT)
        await shim._tick(sim, PHYSICS_DT)
        assert runtime.current["hinge"] == pytest.approx(0.02)
        sim.push("door", "hinge", 0.008)
        await shim._tick(sim, PHYSICS_DT)
        await shim._tick(sim, PHYSICS_DT)
        assert runtime.current["hinge"] == 0.0
        assert _snapshot(sim, "env_0/door").discrete["state"] == "closed"

    on_physics_sim(body)


def test_a_semantic_write_opens_a_latched_door_without_its_handle(assets, on_physics_sim):
    async def body(sim):
        await sim.spawn_objects([_obstacle("door", assets["door"])])
        runtime = sim._articulated["door"]
        assert sim.set_semantic_value("env_0/door", "target", "open")
        await _settle(sim, runtime)
        assert runtime.current["hinge"] == 1.5
        assert runtime.current["handle"] == 0.0

    on_physics_sim(body)


def test_a_physical_object_moves_and_deletes_as_one_and_resets_its_joints_in_place(assets, on_physics_sim):
    import attrs
    from arena_runtime.shared import Pose, Position

    async def body(sim):
        from arena_runtime.sim import _mechanism_shim as shim

        cabinet = _obstacle("cabinet", assets["cabinet"])
        await sim.spawn_objects([cabinet])
        runtime = sim._articulated["cabinet"]
        sim.push("cabinet", "drawer", 0.3)
        await shim._tick(sim, PHYSICS_DT)
        sim.calls.clear()
        moved = attrs.evolve(cabinet, pose=Pose(position=Position(5.0, 0.0, 0.0)), joints={"door": "open"})
        assert list(await sim.move_objects([moved])) == [True]
        (move, drive) = sim.calls
        assert (move[0], _names(move[1])) == ("move", ["cabinet"])
        assert drive[0] == "drive"
        assert drive[1][0].place == frozenset({"door", "drawer", "knob"})
        assert sim.joints["cabinet"] == runtime.current == {"door": 1.5, "drawer": 0.0, "knob": 0.0}
        sim.push("cabinet", "door", 0.2)
        await shim._tick(sim, PHYSICS_DT)
        sim.reset_semantics()
        await shim._tick(sim, PHYSICS_DT)
        assert sim.calls[-1][1][0].place == frozenset({"door", "drawer", "knob"})
        assert sim.joints["cabinet"]["door"] == 1.5
        sim.calls.clear()
        assert list(await sim.delete_objects([moved])) == [True]
        assert [(verb, _names(items)) for verb, items in sim.calls] == [("delete", ["cabinet"])]
        assert sim._articulated == {}

    on_physics_sim(body)


def test_a_continuous_joint_reads_back_wrapped(assets, on_physics_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        runtime = sim._articulated["cabinet"]
        sim.push("cabinet", "knob", 3 * math.pi)
        await shim._tick(sim, PHYSICS_DT)
        assert abs(runtime.current["knob"]) == pytest.approx(math.pi)
        assert _snapshot(sim).continuous["angle.knob"] == pytest.approx(math.pi)

    on_physics_sim(body)


def test_a_placement_the_simulator_missed_is_sent_again(assets, on_physics_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"], joints={"door": "open"})])
        runtime = sim._articulated["cabinet"]
        sim.reset_semantics()
        joints = sim.joints.pop("cabinet")
        await shim._tick(sim, PHYSICS_DT)
        assert not runtime.placed
        joints["door"] = 0.0
        sim.joints["cabinet"] = joints
        await shim._tick(sim, PHYSICS_DT)
        assert runtime.placed
        assert sim.calls[-1][1][0].place == frozenset({"door", "drawer", "knob"})
        assert joints["door"] == 1.5

    on_physics_sim(body)


def test_the_gazebo_model_pins_the_root_and_drives_the_movable_joints(assets, tmp_path):
    import xml.etree.ElementTree as ET

    from arena_simulation_setup.utils.geometry import Scale

    from arena_runtime.sim._articulation import resolve_articulated
    from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _articulated_sdf

    (tmp_path / "door.sdf").write_text('<sdf version="1.7"><model name="door"><static>true</static><link name="link"><visual name="visual"><geometry><mesh><uri>door.dae</uri></mesh></geometry></visual><collision name="collision"><geometry><mesh><uri>door.dae</uri></mesh></geometry></collision></link></model></sdf>')
    runtime = asyncio.run(resolve_articulated(_obstacle("cabinet", assets["cabinet"], scale=Scale(2.0, 1.0, 1.0), joints={"door": 0.5}), "env_0/cabinet"))
    assert runtime is not None
    runtime.make_physical()
    links = [(link, joint, tmp_path / "door.sdf" if link == "door" else None) for link, joint in runtime.tree()]
    model = ET.fromstring(_articulated_sdf("env_0/cabinet", runtime, links)).find("model")
    assert model is not None
    assert model.get("name") == "env_0/cabinet"
    joints = {joint.get("name"): joint for joint in model.findall("joint")}
    assert (joints["arena_world_pin"].get("type"), joints["arena_world_pin"].findtext("parent"), joints["arena_world_pin"].findtext("child")) == ("fixed", "world", "body")
    door = joints["door"]
    assert (door.get("type"), door.findtext("parent"), door.findtext("child")) == ("revolute", "body", "door")
    assert [float(value) for value in door.findtext("pose").split()] == pytest.approx([1.0, -0.3, 0.0, 0.0, 0.0, 0.0])
    assert float(door.findtext("axis/limit/upper")) == pytest.approx(1.6)
    assert float(door.findtext("axis/dynamics/friction")) == pytest.approx(runtime.dynamics["door"].friction)
    assert joints["knob"].get("type") == "revolute"
    assert float(joints["knob"].findtext("axis/limit/upper")) > 1e15
    links_by_name = {link.get("name"): link for link in model.findall("link")}
    assert set(links_by_name) == {"body", "door", "drawer", "knob"}
    assert float(links_by_name["door"].findtext("inertial/mass")) == pytest.approx(runtime.inertias["door"].mass)
    assert links_by_name["door"].findtext("visual/geometry/mesh/uri") == str(tmp_path / "door.dae")
    assert links_by_name["door"].findtext("visual/geometry/mesh/scale") == "2.0 1.0 1.0"
    assert links_by_name["door"].findtext("collision/surface/contact/collide_bitmask") == "0xfffe"
    assert links_by_name["body"].find("collision") is None
    drives = {joint.get("name"): joint for joint in model.findall("plugin/joint")}
    assert set(drives) == {"door", "drawer", "knob"}
    assert (float(drives["door"].get("initial")), drives["knob"].get("wraps")) == (0.5, "true")
    assert float(drives["door"].get("stiffness")) == pytest.approx(runtime.dynamics["door"].stiffness)


def test_springs_and_link_materials_reach_the_gazebo_model_and_every_simulators_link_message(assets, tmp_path):
    import xml.etree.ElementTree as ET

    from arena_mujoco_msgs.msg import ArticulatedLink as MujocoLink
    from arena_mujoco_msgs.msg import JointDynamics as MujocoDynamics
    from arena_simulation_setup.utils.articulation import Material
    from isaacsim_msgs.msg import ArticulatedLink as IsaacLink
    from isaacsim_msgs.msg import JointDynamics as IsaacDynamics

    from arena_runtime.sim._articulation import link_message, resolve_articulated
    from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _articulated_sdf

    (tmp_path / "door.sdf").write_text('<sdf version="1.7"><model name="door"><static>true</static><link name="link"><collision name="collision"><geometry><mesh><uri>door.dae</uri></mesh></geometry></collision></link></model></sdf>')
    runtime = asyncio.run(resolve_articulated(_obstacle("cabinet", assets["cabinet"]), "env_0/cabinet"))
    assert runtime is not None
    runtime.make_physical()
    runtime.articulation.materials = {"door": Material(static_friction=0.9, dynamic_friction=0.6, restitution=0.2)}
    runtime.dynamics["door"] = attrs.evolve(runtime.dynamics["door"], spring_stiffness=3.0, spring_rest=0.2)
    links = [(link, joint, tmp_path / "door.sdf" if link == "door" else None) for link, joint in runtime.tree()]
    model = ET.fromstring(_articulated_sdf("env_0/cabinet", runtime, links)).find("model")
    assert model is not None
    door = next(joint for joint in model.findall("joint") if joint.get("name") == "door")
    assert (float(door.findtext("axis/dynamics/spring_stiffness")), float(door.findtext("axis/dynamics/spring_reference"))) == (3.0, 0.2)
    surface = next(link for link in model.findall("link") if link.get("name") == "door").find("collision/surface")
    assert surface is not None
    assert (float(surface.findtext("friction/ode/mu")), float(surface.findtext("bounce/restitution_coefficient")), surface.findtext("contact/collide_bitmask")) == (0.6, 0.2, "0xfffe")
    joint = runtime.articulation.joint_map["door"]
    for link_type, dynamics_type in ((MujocoLink, MujocoDynamics), (IsaacLink, IsaacDynamics)):
        message = link_message(link_type(), runtime, "door", joint, dynamics_type)
        assert (message.has_material, message.static_friction, message.dynamic_friction, message.restitution) == (True, 0.9, 0.6, 0.2)
        assert (message.dynamics.spring_stiffness, message.dynamics.spring_rest) == (3.0, 0.2)
        assert not link_message(link_type(), runtime, "body", None, dynamics_type).has_material


def test_the_joint_codes_match_every_simulators_link_message():
    from arena_mujoco_msgs.msg import ArticulatedLink as MujocoLink
    from isaacsim_msgs.msg import ArticulatedLink as IsaacLink

    from arena_runtime.sim._articulation import JOINT_CODES

    for message in (MujocoLink, IsaacLink):
        assert {kind.name: code for kind, code in JOINT_CODES.items()} == {name: getattr(message, name) for name in ("REVOLUTE", "PRISMATIC", "CONTINUOUS", "FIXED")}
