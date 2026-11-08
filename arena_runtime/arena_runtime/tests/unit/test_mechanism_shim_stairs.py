"""Stair and escalator halves driven through the real _tick against a duck-typed mechanism double."""

from __future__ import annotations

import asyncio
import math

import attrs

import pytest

pytest.importorskip("arena_runtime.shared")
pytest.importorskip("arena_runtime.sim._mechanism_shim")

from arena_runtime.sim._mechanism_shim import (  # noqa: E402
    _tick,
    stair_boxes,
    reset_mechanisms,
    shim_remove_stairs,
    shim_spawn_stairs,
)
from arena_runtime.sim._semantics import SemanticsManager  # noqa: E402
from arena_simulation_setup.shared import Stairs  # noqa: E402
from arena_simulation_setup.shared.world import STAIRS_GAP, StairsModel  # noqa: E402
from arena_simulation_setup.tree.assets.Material import Material  # noqa: E402
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier  # noqa: E402
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position  # noqa: E402

DT = 1.0 / 30.0


class _Logger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


class _Time:
    def __init__(self) -> None:
        self.seconds = 0.0

    def to_seconds(self) -> float:
        return self.seconds


class _Node:
    def __init__(self) -> None:
        self._logger = _Logger()
        self.sim_time = _Time()

    def get_logger(self) -> _Logger:
        return self._logger


class _BusyTask:
    def done(self) -> bool:
        return False


class _HumanSim:
    def __init__(self) -> None:
        self.peds: dict[str, tuple[float, float]] = {}
        self.stair_calls: list[dict[str, bool]] = []

    def pedestrian_discs(self):
        return [(name, xy, 0.25) for name, xy in self.peds.items()]

    def set_mechanism_walls(self, key, segments) -> None:
        del key, segments

    async def elevator_doors_changed(self, doors) -> None:
        del doors

    async def stairs_changed(self, stairs) -> None:
        self.stair_calls.append(dict(stairs))


class _Mech:
    def __init__(self, human: _HumanSim | None = None) -> None:
        self.node = _Node()
        self._human_simulator = human
        self._incident_callback = None
        self._door_runtime: dict = {}
        self._elevator_runtime: dict = {}
        self._elevator_doors: dict[str, str] = {}
        self._stair_runtime: dict = {}
        self._stair_primitives: dict[str, list[str]] = {}
        self._stair_models: dict = {}
        self._mechanism_loop_task = _BusyTask()
        self._lights: dict = {}
        self._light_outputs: dict = {}
        self._light_poses: dict = {}
        self._articulated: dict = {}
        self._semantics = SemanticsManager(self)
        self._semantics.set_sim("gazebo")
        self.robots: dict[str, Pose] = {}
        self.robot_teleport_ok = True
        self.teleports: list[tuple[str, Pose]] = []
        self.boxes: dict[str, tuple[tuple[float, float, float], Pose, tuple[float, float, float] | None]] = {}
        self.moved: dict[str, Pose] = {}
        self.move_calls = 0
        self.velocity_calls: list[dict[str, tuple[float, float, float]]] = []
        self.level_changes: list[tuple[str, str]] = []
        self.objects: dict[str, Pose] = {}
        self.object_moves = 0

    def robot_discs(self):
        return [(name, (pose.position.x, pose.position.y), 0.3) for name, pose in self.robots.items()]

    def robot_pose(self, sim_path: str) -> Pose | None:
        return self.robots.get(sim_path)

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        self.teleports.append((sim_path, pose))
        if self.robot_teleport_ok:
            self.robots[sim_path] = pose
        return self.robot_teleport_ok

    async def spawn_box(self, name, size, pose, material=None, *, velocity=None, comb=False) -> bool:
        del material, comb
        self.boxes[name] = (size, pose, velocity)
        return True

    async def set_box_velocities(self, velocities) -> bool:
        self.velocity_calls.append(dict(velocities))
        return True

    async def robot_changed_level(self, sim_path, level) -> None:
        self.level_changes.append((sim_path, level))

    async def move_boxes(self, poses) -> bool:
        self.move_calls += 1
        self.moved.update(poses)
        return True

    async def delete_box(self, name) -> bool:
        self.boxes.pop(name, None)
        return True

    async def obstacle_spawn(self, obstacles) -> list[bool]:
        self.objects.update((obstacle.name, obstacle.pose) for obstacle in obstacles)
        return [True] * len(obstacles)

    async def obstacle_move(self, obstacles) -> list[bool]:
        self.object_moves += 1
        self.objects.update((obstacle.name, obstacle.pose) for obstacle in obstacles)
        return [True] * len(obstacles)

    async def obstacle_delete(self, obstacles) -> list[bool]:
        for obstacle in obstacles:
            self.objects.pop(obstacle.name)
        return [True] * len(obstacles)


def _stairs() -> tuple[Stairs, Stairs]:
    low = Stairs(name="stairs_low", position=Position(10.3, 2.0, 0.0), mouth_side="-x", to="stairs_high")
    high = Stairs(name="stairs_high", position=Position(23.7, 2.0, 0.0), mouth_side="+x", to="stairs_low", top=True)
    return low, high


def _escalators(*, down: bool = False) -> tuple[Stairs, Stairs]:
    low = Stairs(name="esc_low", kind="escalator", position=Position(10.3, 5.0, 0.0), size=[2.6, 1.0], mouth_side="-x", to="" if down else "esc_high")
    high = Stairs(name="esc_high", kind="escalator", position=Position(23.7, 5.0, 0.0), size=[2.6, 1.0], mouth_side="+x", to="esc_low" if down else "", top=True)
    return low, high


async def _spawn(mech: _Mech, *stairs: Stairs) -> None:
    assert await shim_spawn_stairs(mech, stairs)
    for stair in stairs:
        assert mech._semantics.attach(stair.kind, stair.name)


async def _run(mech: _Mech, ticks: int) -> None:
    for _ in range(ticks):
        mech.node.sim_time.seconds += DT
        await _tick(mech, DT)


def _snap(mech: _Mech, entity: str):
    return next(s for s in mech._semantics.snapshot() if s.entity == entity)


def _xyyaw(pose: Pose) -> tuple[float, float, float]:
    return pose.position.x, pose.position.y, pose.orientation.to_yaw()


def _at(x: float, y: float, yaw: float = 0.0, *, z: float = 0.0, roll: float = 0.0, pitch: float = 0.0) -> Pose:
    return Pose(position=Position(x, y, z), orientation=Orientation.from_euler((roll, pitch, yaw)))


def test_stair_boxes_are_static_steps_or_a_static_base_under_moving_treads():
    stairs_low, stairs_high = _stairs()
    esc_low, esc_high = _escalators()
    kinds = {stair.name: [(prim.name.rsplit("/", 1)[1].rstrip("0123456789"), prim.velocity) for prim in stair_boxes(stair)] for stair in (stairs_low, stairs_high, esc_low, esc_high)}
    assert kinds["stairs_low"] == [("step_", None)] * stairs_low.tread_count
    assert kinds["stairs_high"] == [("step_", None)] * stairs_high.tread_count
    for stair in (esc_low, esc_high):
        assert kinds[stair.name] == [("base_", None)] * len(stair.base()) + [("plate_", None)] + [("tread_", velocity) for velocity in stair.tread_velocities(0.0)]
    assert [prim.box for prim in stair_boxes(esc_low)] == [*esc_low.base(), *esc_low.plates(), *esc_low.treads(0.0)]
    assert [prim.box for prim in stair_boxes(stairs_high)] == stairs_high.steps()
    assert [prim.name.rsplit("/", 1)[1] for prim in stair_boxes(esc_low) if prim.comb] == ["plate_0"]
    assert all(prim.material is esc_low.material for prim in stair_boxes(esc_low))


def _ramp(**fields: object) -> Stairs:
    return Stairs(name="ramp", kind="ramp", position=Position(22.0, 5.0, -1.0), size=[24.0, 8.0], rise=1.0, **fields)


def test_a_ramp_is_one_static_slab_or_nothing_when_the_world_has_the_floor():
    assert [(prim.name, prim.velocity, prim.comb) for prim in stair_boxes(_ramp())] == [("ramp/base_0", None, False)]
    assert stair_boxes(_ramp(floor=False)) == []


def test_a_ramp_never_teleports_and_counts_who_stands_on_it():
    async def scenario():
        mech = _Mech()
        await _spawn(mech, _ramp(floor=False))
        mech.robots["robot"] = _at(33.9, 5.0)
        await _run(mech, 3)
        return mech

    mech = asyncio.run(scenario())
    assert mech.boxes == {}
    assert mech.teleports == []
    assert mech.node.get_logger().infos == []
    assert mech._stair_runtime["ramp"].members == ("robot",)
    assert mech._stair_runtime["ramp"].treads == {}


def test_spawned_stairs_track_their_primitives_until_removed():
    async def scenario():
        mech = _Mech()
        await _spawn(mech, *_stairs(), *_escalators())
        spawned = dict(mech._stair_primitives)
        boxes = dict(mech.boxes)
        partners = {name: runtime.partner for name, runtime in mech._stair_runtime.items()}
        treads = sorted(mech._stair_runtime["esc_low"].treads.values())
        assert await shim_remove_stairs(mech, list(mech._stair_runtime))
        return spawned, boxes, partners, treads, mech

    spawned, boxes, partners, treads, mech = asyncio.run(scenario())
    assert set(boxes) == {name for names in spawned.values() for name in names}
    assert boxes["esc_low/tread_0"][2] == _escalators()[0].tread_velocities(0.0)[0]
    assert boxes["stairs_high/step_0"][2] is None
    assert partners == {"stairs_low": "stairs_high", "stairs_high": "stairs_low", "esc_low": "esc_high", "esc_high": "esc_low"}
    assert treads == sorted(f"esc_low/tread_{k}" for k in range(9))
    assert mech.boxes == {}
    assert mech._stair_primitives == {}
    assert mech._stair_runtime == {}


def test_robot_crossing_the_seam_lands_past_the_partner_seam_heading_out():
    async def scenario():
        mech = _Mech()
        low, high = _stairs()
        await _spawn(mech, low, high)
        mech.robots["r"] = _at(11.3, 2.2, 0.1, z=0.05, roll=0.02, pitch=-0.01)
        await _run(mech, 1)
        return mech, low, high

    mech, low, high = asyncio.run(scenario())
    [(sim_path, pose)] = mech.teleports
    assert sim_path == "r"
    x, y, yaw, dz = low.landing(high, 11.3, 2.2, 0.1)
    assert _xyyaw(pose) == pytest.approx((x, y, yaw))
    assert dz == pytest.approx(high.surface(high.along(x, y)) - low.surface(low.along(11.3, 2.2)))
    roll, pitch, _yaw = pose.orientation.to_euler()
    assert (pose.position.z, roll, pitch) == pytest.approx((0.05 + dz, 0.02, -0.01))
    assert high.contains(pose.position.x, pose.position.y)
    assert not high.crossed(pose.position.x, pose.position.y)


def test_robot_short_of_the_trigger_line_stays_put():
    async def scenario():
        mech = _Mech()
        await _spawn(mech, *_stairs())
        mech.robots["r"] = _at(10.6, 2.0)
        await _run(mech, 3)
        return mech.teleports

    assert asyncio.run(scenario()) == []


def test_landed_robot_is_not_sent_back_while_its_pose_is_stale_or_at_the_landing():
    async def scenario():
        mech = _Mech()
        await _spawn(mech, *_stairs())
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 1)
        landed = mech.robots["r"]
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 3)
        mech.robots["r"] = landed
        await _run(mech, 5)
        return mech.teleports

    teleports = asyncio.run(scenario())
    assert len(teleports) == 1


def test_robot_walking_back_through_the_upper_seam_returns_to_the_lower_half():
    async def scenario():
        mech = _Mech()
        low, high = _stairs()
        await _spawn(mech, low, high)
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 2)
        mech.robots["r"] = _at(22.8, 2.0, math.pi)
        await _run(mech, 1)
        return mech.teleports, low, high

    teleports, low, high = asyncio.run(scenario())
    assert len(teleports) == 2
    assert _xyyaw(teleports[1][1]) == pytest.approx(high.landing(low, 22.8, 2.0, math.pi)[:3])
    assert low.contains(teleports[1][1].position.x, teleports[1][1].position.y)


def test_failed_teleport_is_one_incident_until_the_robot_leaves_the_trigger_band():
    async def scenario():
        mech = _Mech()
        incidents: list[str] = []
        mech._incident_callback = incidents.append
        mech.robot_teleport_ok = False
        await _spawn(mech, *_stairs())
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 5)
        first = list(incidents)
        mech.robots["r"] = _at(10.0, 2.0)
        await _run(mech, 1)
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 1)
        return first, incidents, _snap(mech, "stairs_low")

    first, incidents, snap = asyncio.run(scenario())
    assert first == ["r: stair teleport 'stairs_low' -> 'stairs_high' failed: the simulator refused the move"]
    assert len(incidents) == 2
    assert snap.predicates["just_crossed"] is False


def test_exit_half_never_teleports():
    async def scenario():
        mech = _Mech()
        low, high = _escalators()
        await _spawn(mech, low, high)
        mech.robots["r"] = _at(22.8, 5.0, math.pi)
        await _run(mech, 3)
        return mech.teleports

    assert asyncio.run(scenario()) == []


def test_half_whose_partner_is_not_spawned_keeps_the_robot_without_incidents():
    async def scenario():
        mech = _Mech()
        incidents: list[str] = []
        mech._incident_callback = incidents.append
        lonely = Stairs(name="lonely", position=Position(10.3, 2.0, 0.0), mouth_side="-x", to="ghost")
        await _spawn(mech, lonely)
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 5)
        return mech, incidents

    mech, incidents = asyncio.run(scenario())
    assert mech.teleports == []
    assert incidents == []
    assert mech.node.get_logger().warnings == []
    assert "Stairs 'lonely': partner half 'ghost' not spawned, its seam lands nowhere" in mech.node.get_logger().infos


def test_escalator_half_without_a_partner_is_a_dead_end_whose_belt_still_runs():
    async def scenario():
        mech = _Mech()
        orphan = Stairs(name="orphan", kind="escalator", position=Position(10.3, 5.0, 0.0), size=[2.6, 1.0], mouth_side="-x")
        await _spawn(mech, orphan)
        mech.robots["r"] = _at(11.3, 5.0)
        await _run(mech, 30)
        return mech, orphan

    mech, orphan = asyncio.run(scenario())
    assert mech.teleports == []
    assert orphan.belt == -0.5
    assert mech._stair_runtime["orphan"].travel == pytest.approx(-0.5)
    assert _snap(mech, "orphan").discrete == {"running": "true", "direction": ""}


def test_belts_carry_the_treads_towards_the_upper_level_and_wrap():
    async def scenario():
        mech = _Mech()
        low, high = _escalators()
        await _spawn(mech, low, high)
        await _run(mech, 180)
        return mech, low, high

    mech, low, high = asyncio.run(scenario())
    assert low.belt == 0.5
    assert high.belt == -0.5
    assert mech._stair_runtime["esc_low"].travel == pytest.approx(3.0)
    assert mech._stair_runtime["esc_high"].travel == pytest.approx(-3.0)
    assert mech.move_calls == 180
    sent: dict = {}
    for call in mech.velocity_calls:
        sent.update(call)
    for stair in (low, high):
        wanted = stair.tread_velocities(mech._stair_runtime[stair.name].travel)
        assert {name: sent.get(name, velocity) for name, velocity in mech._stair_runtime[stair.name].velocities.items()} == {f"{stair.name}/tread_{k}": velocity for k, velocity in enumerate(wanted)}
    for stair in (low, high):
        expected = stair.treads(mech._stair_runtime[stair.name].travel)
        for index, (_size, pose) in enumerate(expected):
            moved = mech.moved[f"{stair.name}/tread_{index}"]
            assert (moved.position.x, moved.position.y, moved.position.z) == pytest.approx((pose.position.x, pose.position.y, pose.position.z))


def test_stairs_have_no_belt_to_move():
    async def scenario():
        mech = _Mech()
        await _spawn(mech, *_stairs())
        await _run(mech, 10)
        return mech.move_calls

    assert asyncio.run(scenario()) == 0


def test_stopped_escalator_freezes_its_treads_and_tells_the_human_simulator():
    async def scenario():
        human = _HumanSim()
        mech = _Mech(human)
        await _spawn(mech, *_escalators())
        await _run(mech, 3)
        before = len(mech.velocity_calls)
        assert mech._semantics.set_value("esc_high", "running", "false")
        await _run(mech, 1)
        stop_call = mech.velocity_calls[before]
        frozen = (dict(mech.moved), {name: rt.travel for name, rt in mech._stair_runtime.items()}, mech.move_calls)
        await _run(mech, 10)
        stopped = (dict(mech.moved), {name: rt.travel for name, rt in mech._stair_runtime.items()}, mech.move_calls)
        snaps = (_snap(mech, "esc_low").discrete, _snap(mech, "esc_high").discrete)
        quiet = len(mech.velocity_calls)
        assert mech._semantics.set_value("esc_low", "running", "true")
        await _run(mech, 1)
        return human.stair_calls, frozen, stopped, snaps, mech, stop_call, quiet

    calls, frozen, stopped, snaps, mech, stop_call, quiet = asyncio.run(scenario())
    assert calls == [{"esc_low": False, "esc_high": False}, {"esc_low": True, "esc_high": True}]
    assert frozen[:2] == stopped[:2]
    assert (frozen[2], stopped[2], mech.move_calls) == (4, 14, 15)
    assert snaps == ({"running": "false", "direction": "up"}, {"running": "false", "direction": "up"})
    low, high = _escalators()
    treads = {stair.name: [f"{stair.name}/tread_{k}" for k in range(stair.tread_count)] for stair in (low, high)}
    assert stop_call == {name: (0.0, 0.0, 0.0) for names in treads.values() for name in names}
    assert len(mech.velocity_calls) == quiet + 1
    assert mech.velocity_calls[-1] == {name: velocity for stair in (low, high) for name, velocity in zip(treads[stair.name], stair.tread_velocities(mech._stair_runtime[stair.name].travel), strict=True)}


def test_running_takes_only_true_or_false():
    async def scenario():
        mech = _Mech()
        await _spawn(mech, *_escalators(), *_stairs())
        with pytest.raises(ValueError, match="malformed value"):
            mech._semantics.set_value("esc_low", "running", "maybe")
        with pytest.raises(ValueError, match="field not writable"):
            mech._semantics.set_value("esc_low", "direction", "down")
        with pytest.raises(ValueError, match="read-only"):
            mech._semantics.set_value("stairs_low", "occupants", "3")

    asyncio.run(scenario())


def test_escalator_direction_follows_the_boarding_half():
    async def scenario():
        up, down = _Mech(), _Mech()
        await _spawn(up, *_escalators())
        await _spawn(down, *_escalators(down=True))
        return [_snap(mech, name).discrete for mech in (up, down) for name in ("esc_low", "esc_high")]

    assert asyncio.run(scenario()) == [
        {"running": "true", "direction": "up"},
        {"running": "true", "direction": "up"},
        {"running": "true", "direction": "down"},
        {"running": "true", "direction": "down"},
    ]


def test_occupants_count_robots_and_pedestrians_on_the_half():
    async def scenario():
        human = _HumanSim()
        mech = _Mech(human)
        await _spawn(mech, *_stairs())
        mech.robots["r"] = _at(9.5, 2.0)
        mech.robots["far"] = _at(0.0, 0.0)
        human.peds = {"p": (10.0, 1.8), "q": (24.0, 2.3)}
        await _run(mech, 1)
        return _snap(mech, "stairs_low"), _snap(mech, "stairs_high")

    low, high = asyncio.run(scenario())
    assert low.kind == "stairs"
    assert (low.continuous, low.members) == ({"occupants": 2.0}, ["r", "p"])
    assert (high.continuous, high.members) == ({"occupants": 1.0}, ["q"])
    assert low.discrete == {}


def test_just_crossed_holds_for_one_tick_after_a_robot_teleport():
    async def scenario():
        mech = _Mech()
        changes: list = []
        mech._semantics.set_change_callback(changes.extend)
        await _spawn(mech, *_stairs())
        mech.robots["r"] = _at(9.5, 2.0)
        await _run(mech, 1)
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 1)
        crossed = (_snap(mech, "stairs_low").predicates, _snap(mech, "stairs_high").predicates)
        await _run(mech, 1)
        return changes, crossed, _snap(mech, "stairs_low").predicates

    changes, crossed, after = asyncio.run(scenario())
    assert crossed == ({"just_crossed": True}, {"just_crossed": False})
    assert after == {"just_crossed": False}
    flips = [(c.entity, c.current) for c in changes if c.field == "just_crossed" and c.previous]
    assert flips == [("stairs_low", "true"), ("stairs_low", "false")]


def test_just_crossed_marks_a_pedestrian_jump_to_the_partner():
    async def scenario():
        human = _HumanSim()
        mech = _Mech(human)
        await _spawn(mech, *_escalators())
        human.peds = {"p": (11.4, 5.0)}
        await _run(mech, 1)
        human.peds = {"p": (22.95, 5.0)}
        await _run(mech, 1)
        crossed = (_snap(mech, "esc_low").predicates, _snap(mech, "esc_high").predicates)
        await _run(mech, 1)
        return crossed, _snap(mech, "esc_low").predicates

    crossed, after = asyncio.run(scenario())
    assert crossed == ({"just_crossed": True}, {"just_crossed": False})
    assert after == {"just_crossed": False}


def test_reset_restarts_belts_and_clears_latches():
    async def scenario():
        mech = _Mech()
        await _spawn(mech, *_escalators(), *_stairs())
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 5)
        mech._semantics.set_value("esc_low", "running", "false")
        reset_mechanisms(mech)
        return mech._stair_runtime

    runtimes = asyncio.run(scenario())
    assert {name: (rt.travel, rt.running, rt.latched, rt.members) for name, rt in runtimes.items()} == {name: (0.0, True, set(), ()) for name in runtimes}


def _joined_stairs() -> tuple[Stairs, Stairs]:
    low = Stairs(name="env_0/stairs_low/0", position=Position(10.3, 2.0, 0.0), mouth_side="-x", to="env_0/stairs_high/1")
    high = Stairs(name="env_0/stairs_high/1", position=Position(23.7, 2.0, 0.0), mouth_side="+x", to="env_0/stairs_low/0", top=True)
    low.joined = high.joined = True
    return low, high


def test_robot_walking_past_a_joined_seam_changes_level_without_a_teleport():
    async def scenario():
        mech = _Mech()
        low, high = _joined_stairs()
        await _spawn(mech, low, high)
        mech.robots["r"] = _at(11.3, 2.0)
        await _run(mech, 2)
        before = list(mech.level_changes)
        mech.robots["r"] = _at(11.7, 2.0)
        await _run(mech, 1)
        up = list(mech.level_changes)
        mech.robots["r"] = _at(21.9, 2.0, math.pi)
        await _run(mech, 1)
        return mech, before, up, _snap(mech, "env_0/stairs_low/0")

    mech, before, up, snap = asyncio.run(scenario())
    assert mech.teleports == []
    assert before == []
    assert up == [("r", "1")]
    assert mech.level_changes == [("r", "1"), ("r", "0")]
    assert snap.predicates["just_crossed"] is False


def test_joined_halves_spawn_without_a_seam_wall_and_treads_to_the_seam():
    low, high = _escalators()
    low.joined = high.joined = True
    for stair in (low, high):
        starts = [start for start in stair.tread_starts(0.0) if start is not None]
        assert max(starts) + stair.pitch == pytest.approx(stair.length - STAIRS_GAP)


def test_a_joined_pair_spawns_one_whole_body_on_its_bottom_half():
    low, high = _escalators()
    low.joined = high.joined = True
    body = low.whole_stair()
    assert stair_boxes(high) == []
    names = [prim.name for prim in stair_boxes(low)]
    assert names == [*(f"esc_low/base_{i}" for i in range(len(body.base()))), "esc_low/plate_0", "esc_low/plate_1", *(f"esc_low/tread_{k}" for k in range(body.tread_count))]
    assert [prim.box for prim in stair_boxes(low)] == [*body.base(), *body.plates(), *body.treads(0.0)]

    async def scenario():
        mech = _Mech()
        await _spawn(mech, low, high)
        await _run(mech, 30)
        return mech

    mech = asyncio.run(scenario())
    assert len(mech._stair_runtime["esc_low"].treads) == body.tread_count
    assert mech._stair_runtime["esc_high"].treads == {}
    moved = mech.moved["esc_low/tread_0"]
    expected = body.treads(mech._stair_runtime["esc_low"].travel)[0][1]
    assert (moved.position.x, moved.position.z) == pytest.approx((expected.position.x, expected.position.z))


def test_stair_boxes_take_each_surfaces_material():
    steel, granite, rubber = Material.default("wall"), Material.default("floor"), Material.default("elevator")
    low, _high = _escalators()
    low = attrs.evolve(low, surfaces={"fixture": steel, "landings": granite, "steps": rubber})
    materials = {prim.name.rsplit("/", 1)[1].split("_")[0]: prim.material for prim in stair_boxes(low)}
    assert materials == {"base": steel, "plate": granite, "tread": rubber}


def test_models_dress_the_static_boxes_and_ride_the_treads_until_removed():
    low, high = _escalators()
    capture = ObjectIdentifier("step_capture")
    low = attrs.evolve(low, surface_models={"steps": capture, "landings": capture}, models=[StairsModel(model="sign", at=(0.5, 0.0, 2.0))])
    high = attrs.evolve(high, surface_models={"steps": capture, "landings": capture})

    async def scenario():
        mech = _Mech()
        assert await shim_spawn_stairs(mech, [low, high])
        plates = [name for name in mech.objects if "/model_plate_" in name]
        treads = sorted(name for name in mech.objects if name.startswith("esc_low/model_tread_"))
        assert len(plates) == 2 and len(treads) == low.tread_count
        assert "esc_low/model_0" in mech.objects
        before = mech.objects["esc_low/model_tread_0"]
        mech._stair_runtime["esc_low"].running = True
        await _tick(mech, 0.5)
        assert mech.object_moves == 1
        assert mech.objects["esc_low/model_tread_0"] != before
        assert mech.objects["esc_low/model_tread_0"] == low.tread_models(mech._stair_runtime["esc_low"].travel)[0]
        assert await shim_remove_stairs(mech, ["esc_low", "esc_high"])
        assert mech.objects == {}

    asyncio.run(scenario())
