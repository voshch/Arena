"""Arena humansim adapter, world-mechanism side: teleports, mechanism walls, semantic stimuli, delayed spawns, itineraries."""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import types
import typing

import pytest

pytest.importorskip("rclpy")

from arena_humansim_msgs.msg import AgentFrame
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from task_generator.manager.realizer import Realizer
from task_generator.shared import Door, DynamicObstacle, Obstacle, Pose, Position
from task_generator.simulators.human import BaseHumanSimulator
from task_generator.simulators.human.arena_humansim.arena_humansim import _DEFERRED_ID_BASE, ArenaHumanSimulator
from task_generator.simulators.human.drivers import DrivenJoints, JoystickDriver
from task_generator.simulators.human.itinerary import ItineraryDriver, Phase, Stop
from task_generator.simulators.human.utils import KnownObstacles, ObstacleLayer

ENV_X, ENV_Y = 100.0, 50.0


class _Client:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self._ids = itertools.count(1)

    async def call_timeout(self, request: object) -> object:
        self.requests.append(request)
        return types.SimpleNamespace(success=True, message="", spawned_ids=[a.agent_id or next(self._ids) for a in request.agents] if isinstance(request, SpawnAgents.Request) else [])


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def get_child(self, _name: str) -> _Logger:
        return self

    def __getattr__(self, level: str) -> typing.Callable[..., None]:
        return lambda msg, *a, **k: self.lines.append(f"{level}: {msg}")


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()
        self.semantic_writes: list[tuple[str, str, str]] = []
        self.conf = None
        self.sim_time = types.SimpleNamespace(sec=2, nanosec=0)

    def get_logger(self) -> _Logger:
        return self.logger

    def _set_semantic(self, entity: str, field: str, value: str) -> bool:
        self.semantic_writes.append((entity, field, value))
        return True


CLIENTS = ("_spawn_client", "_update_robot_client", "_teleport_agents_client", "_set_waypoints_client", "_notify_stimulus_client", "_add_walls_client", "_remove_walls_client", "_add_world_objects_client", "_remove_world_objects_client")


def _adapter(level_origins: dict[str, tuple[float, float]] | None = None) -> ArenaHumanSimulator:
    sim = ArenaHumanSimulator.__new__(ArenaHumanSimulator)
    sim._NodeInterface__node = _Node()
    sim._realizer = Realizer(Realizer._Configuration(x=ENV_X, y=ENV_Y, prefix="env_0"), dict(level_origins or {}))
    for name in CLIENTS:
        setattr(sim, name, _Client())
    sim._now_s = lambda: 0.0
    sim._spawns_in_flight = 0
    sim._next_deferred_id = _DEFERRED_ID_BASE
    sim._bridge_agent_ids = set()
    sim._flow_agent_ids = set()
    sim._agent_names = {}
    sim._prev_agent_states = None
    sim._curr_agent_states = None
    sim._ped_positions_xy = {}
    sim._ped_orientations = {}
    sim._teleport_stamps = {}
    sim._deferred_spawns = []
    sim._itineraries = {}
    sim._semantic_aliases = {}
    sim._semantic_values = {}
    sim._semantic_backlog = __import__("collections").deque()
    sim._semantic_draining = False
    sim._mech_walls_desired = {}
    sim._mech_walls_applied = {}
    sim._mech_objects = {}
    sim._mech_walls_syncing = False
    return sim


def _run[T](coro: typing.Coroutine[typing.Any, typing.Any, T]) -> T:
    return asyncio.run(coro)


def _states(*agents: tuple[int, float, float]) -> AgentFrame:
    msg = AgentFrame()
    msg.agent_id = [aid for aid, _, _ in agents]
    msg.x = [x for _, x, _ in agents]
    msg.y = [y for _, _, y in agents]
    msg.theta = [0.7] * len(agents)
    msg.vx = [0.0] * len(agents)
    msg.vy = [0.0] * len(agents)
    msg.radius = [0.27] * len(agents)
    return msg


def _stimuli(sim: ArenaHumanSimulator) -> list[tuple[int, str, float]]:
    return [(r.agent_id, r.stimulus, round(r.intensity, 3)) for r in sim._notify_stimulus_client.requests]


# teleport


def test_teleport_moves_the_engine_agents_and_pulses_teleported() -> None:
    sim = _adapter()
    sim._agent_names = {3: "env_0/rider", 40: "env_0/flow_40"}
    sim._bridge_agent_ids = {3}

    assert _run(sim.pedestrian_teleport({"env_0/rider": (130.0, 55.0), "env_0/flow_40": (131.0, 55.0)}))
    (request,) = sim._teleport_agents_client.requests
    assert list(request.agent_id) == [3, 40]
    assert [(p.x, p.y) for p in request.pose] == pytest.approx([(30.0, 5.0), (31.0, 5.0)])
    assert sim._teleport_stamps == {3: 2_000_000_000, 40: 2_000_000_000}
    assert _stimuli(sim) == [(3, "teleported", 1.0), (40, "teleported", 1.0)]


def test_teleport_reports_unknown_peds() -> None:
    sim = _adapter()
    assert not _run(sim.pedestrian_teleport({"ghost": (1.0, 1.0)}))
    assert sim._teleport_agents_client.requests == []
    assert _stimuli(sim) == []


def test_teleport_hands_an_itinerary_ped_its_next_leg() -> None:
    sim = _adapter()
    sim._agent_names = {3: "env_0/rider"}
    sim._bridge_agent_ids = {3}
    driver = ItineraryDriver([Stop(5.0, 0.0, ride=True), Stop(30.0, 0.0)], "once")
    driver.start()
    sim._itineraries[3] = driver
    _run(sim.pedestrian_teleport({"env_0/rider": (121.0, 50.0)}))
    (leg,) = sim._set_waypoints_client.requests
    assert leg.agent_id == 3
    assert leg.waypoints.mode == WaypointsMsg.MODE_ONCE
    assert [(p.pose.x, p.pose.y) for p in leg.waypoints.points] == [(30.0, 0.0)]


# itinerary stepping and delayed spawns


def _ped(name: str, waypoints: list[tuple[float, float]], itinerary: list[dict] | None = None, **fields: typing.Any) -> DynamicObstacle:
    extra = {"itinerary": itinerary} if itinerary is not None else {}
    obs = DynamicObstacle(name=name, model="arenian", pose=Pose(Position(ENV_X, ENV_Y, 0.0)), waypoints=[Position(x + ENV_X, y + ENV_Y, 0.0) for x, y in waypoints], extra=extra, **fields)
    obs.sim_path = f"env_0/{name}"
    return obs


def test_spawn_sends_the_first_leg_and_defers_delayed_peds() -> None:
    sim = _adapter()
    itinerary = [{"dwell": 3.0}, {}]
    plain = _ped("plain", [(1.0, 0.0), (2.0, 0.0)])
    held = _ped("held", [(1.0, 0.0), (2.0, 0.0)], itinerary=itinerary, waypoint_mode="repeat")
    late = _ped("late", [(4.0, 0.0)], spawn_delay=5.0, semantic_alias={"elevator": "elev_a"})
    _run(sim._spawn_dynamic_obstacles_impl([plain, held, late]))

    (request,) = sim._spawn_client.requests
    sent = {a.name: a for a in request.agents}
    assert set(sent) == {"env_0/plain", "env_0/held"}
    assert len(sent["env_0/plain"].waypoints.points) == 2
    assert sent["env_0/plain"].waypoints.mode == WaypointsMsg.MODE_REPEAT
    assert [(p.pose.x, p.pose.y) for p in sent["env_0/held"].waypoints.points] == [(1.0, 0.0)]
    assert sent["env_0/held"].waypoints.mode == WaypointsMsg.MODE_ONCE
    assert [a.agent_id for a in request.agents] == [0, 0]
    assert sim._agent_names[2] == "env_0/held" and list(sim._itineraries) == [2]
    ((due, msg, alias),) = sim._deferred_spawns
    assert due == 5.0 and msg.agent_id == _DEFERRED_ID_BASE and alias == {"elevator": "elev_a"}
    assert msg.agent_id in sim._bridge_agent_ids


def test_delayed_ped_appears_when_due_and_gets_its_aliases() -> None:
    sim = _adapter()
    sim._semantic_values = {("elev_a", "cabin_door"): "open", ("elev_b", "cabin_door"): "closed"}
    _run(sim._spawn_dynamic_obstacles_impl([_ped("late", [(4.0, 0.0)], spawn_delay=5.0, semantic_alias={"elevator": "elev_a"})]))
    assert sim._spawn_client.requests == []
    _run(sim._step_peds(4.9))
    assert sim._spawn_client.requests == []
    _run(sim._step_peds(5.0))
    (request,) = sim._spawn_client.requests
    assert [a.agent_id for a in request.agents] == [_DEFERRED_ID_BASE]
    assert sim._deferred_spawns == []
    assert sim._semantic_aliases == {_DEFERRED_ID_BASE: {"elevator": "elev_a"}}
    assert _stimuli(sim) == [(_DEFERRED_ID_BASE, "elevator.cabin_door==open", 1.0)]


def test_step_advances_itineraries_and_presses_the_elevator_button() -> None:
    sim = _adapter()
    driver = ItineraryDriver([Stop(0.0, 0.0, dwell=1.0), Stop(5.0, 0.0, ride=True, elevator="elev_a", target="2.elev_c"), Stop(9.0, 0.0)], "once")
    driver.start()
    sim._itineraries[7] = driver
    sim._curr_agent_states = _states((7, 0.1, 0.0))
    _run(sim._step_peds(0.0))
    assert driver.phase is Phase.DWELL
    _run(sim._step_peds(1.0))
    (leg,) = sim._set_waypoints_client.requests
    assert [(p.pose.x, p.pose.y) for p in leg.waypoints.points] == [(5.0, 0.0)]
    sim._curr_agent_states = _states((7, 5.0, 0.2))
    _run(sim._step_peds(2.0))
    assert driver.phase is Phase.RIDE
    assert sim.node.semantic_writes == [("elev_a", "target", "2.elev_c")]


# semantic stimuli


def test_semantic_change_broadcasts_old_off_new_on_and_aliases() -> None:
    sim = _adapter()
    sim._semantic_aliases = {9: {"elevator": "elev_a"}, 10: {"elevator": "elev_b"}}

    async def go() -> None:
        sim.notify_semantic_change("elev_a", "cabin_door", "opening")
        sim.notify_semantic_change("elev_a", "cabin_door", "open")
        await asyncio.sleep(0)
        while sim._semantic_draining:
            await asyncio.sleep(0)

    _run(go())
    assert _stimuli(sim) == [
        (-1, "elev_a.cabin_door==opening", 1.0),
        (9, "elevator.cabin_door==opening", 1.0),
        (-1, "elev_a.cabin_door==opening", 0.0),
        (-1, "elev_a.cabin_door==open", 1.0),
        (9, "elevator.cabin_door==opening", 0.0),
        (9, "elevator.cabin_door==open", 1.0),
    ]
    assert sim._semantic_values == {("elev_a", "cabin_door"): "open"}


def test_base_semantic_change_is_inert_without_stimuli() -> None:
    sent: list[tuple[int, str, float]] = []

    class _Fake:
        _semantic_values: dict = {}
        _semantic_backlog = __import__("collections").deque()
        _semantic_draining = False
        _logger = logging.getLogger("t")

        async def notify_stimulus(self, agent_id: int, stimulus: str, intensity: float) -> None:
            sent.append((agent_id, stimulus, intensity))

        _in_background = BaseHumanSimulator._in_background
        _drain_semantic_backlog = BaseHumanSimulator._drain_semantic_backlog
        _apply_semantic_change = BaseHumanSimulator._apply_semantic_change
        notify_semantic_change = BaseHumanSimulator.notify_semantic_change

    fake = _Fake()

    async def go() -> None:
        fake.notify_semantic_change("door_a", "state", "closed")
        fake.notify_semantic_change("door_a", "state", "closed")
        while fake._semantic_draining:
            await asyncio.sleep(0)

    _run(go())
    assert sent == [(-1, "door_a.state==closed", 1.0), (-1, "door_a.state==closed", 1.0)]


def test_authored_name_strips_env_and_level() -> None:
    sim = _adapter(level_origins={"1": (0.0, 0.0), "2": (20.0, 0.0)})
    assert sim.authored_name("env_0/door_a") == "door_a"
    assert sim.authored_name("env_0/elev_a/2") == "elev_a"
    assert sim.authored_name("env_0/zone/sub") == "zone/sub"
    assert sim.authored_name("other") == "other"


# mechanism walls and world objects


def _segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    # cabin [100,102]x[50,52] in the map frame, door on +x: back wall at x=100, sides at y=50 / y=52
    return [((100.0, 50.0), (100.0, 52.0)), ((100.0, 52.0), (102.0, 52.0)), ((100.0, 50.0), (102.0, 50.0))]


def _sync(sim: ArenaHumanSimulator, key: str, segments: list | None) -> None:
    async def go() -> None:
        sim.set_mechanism_walls(key, segments)
        while sim._mech_walls_syncing:
            await asyncio.sleep(0)

    _run(go())


def test_elevator_walls_and_objects() -> None:
    sim = _adapter()
    _sync(sim, "elevator:env_0/elev_a", _segments())
    (walls,) = sim._add_walls_client.requests
    assert list(walls.names) == ["__mech/elevator:env_0/elev_a/0", "__mech/elevator:env_0/elev_a/1", "__mech/elevator:env_0/elev_a/2"]
    assert (walls.starts[0].x, walls.starts[0].y) == pytest.approx((0.0, 0.0))
    (objects,) = sim._add_world_objects_client.requests
    cabin, queue = objects.objects
    assert (cabin.object_id, cabin.type) == ("elev_a", "elevator")
    assert (cabin.pose.x, cabin.pose.y, cabin.pose.theta) == pytest.approx((1.0, 1.0, 0.0))
    assert (queue.object_id, queue.type, queue.formation_type) == ("elev_a/queue", "elevator_queue", "line")
    assert (queue.pose.x, queue.pose.y) == pytest.approx((2.0, 1.0))
    assert math.cos(queue.pose.theta) == pytest.approx(-1.0)

    _sync(sim, "elevator:env_0/elev_a", None)
    (removed,) = sim._remove_walls_client.requests
    assert list(removed.names) == list(walls.names)
    (gone,) = sim._remove_world_objects_client.requests
    assert list(gone.object_ids) == ["elev_a", "elev_a/queue"]
    assert sim._mech_walls_applied == {} and sim._mech_objects == {}


def test_door_wall_toggles_replace_the_named_set() -> None:
    sim = _adapter()
    door = [((104.0, 50.0), (104.0, 51.0))]
    _sync(sim, "door:env_0/d1", door)
    _sync(sim, "door:env_0/d1", door)
    _sync(sim, "door:env_0/d1", None)
    assert len(sim._add_walls_client.requests) == 2
    assert [list(r.names) for r in sim._remove_walls_client.requests] == [["__mech/door:env_0/d1/0"], ["__mech/door:env_0/d1/0"]]
    assert sim._add_world_objects_client.requests == []


def test_latest_wall_state_wins_and_world_teardown_clears() -> None:
    sim = _adapter()

    async def go() -> None:
        sim.set_mechanism_walls("door:env_0/d1", [((0.0, 0.0), (1.0, 0.0))])
        sim.set_mechanism_walls("door:env_0/d1", None)
        sim.set_mechanism_walls("elevator:env_0/e", _segments())
        while sim._mech_walls_syncing:
            await asyncio.sleep(0)
        await sim._clear_mechanism_walls()

    _run(go())
    assert [list(r.names) for r in sim._add_walls_client.requests] == [[f"__mech/elevator:env_0/e/{i}" for i in range(3)]]
    assert list(sim._remove_walls_client.requests[-1].names) == [f"__mech/elevator:env_0/e/{i}" for i in range(3)]
    assert list(sim._remove_world_objects_client.requests[-1].object_ids) == ["e", "e/queue"]
    assert sim._mech_walls_applied == {}


def test_doors_become_world_objects() -> None:
    sim = _adapter()
    door = Door(name="env_0/door_a", start=Position(104.0, 50.0, 0.0), end=Position(104.0, 52.0, 0.0))
    _run(sim._spawn_doors_impl({door.name: door}))
    (request,) = sim._add_world_objects_client.requests
    (obj,) = request.objects
    assert (obj.object_id, obj.type) == ("door_a", "door")
    assert (obj.pose.x, obj.pose.y) == pytest.approx((4.0, 1.0))
    assert len(sim._remove_world_objects_client.requests) == 1
    _run(sim._remove_doors_impl([door.name]))
    assert list(sim._remove_world_objects_client.requests[1].object_ids) == ["door_a"]
    _run(sim._remove_doors_impl([door.name]))
    assert len(sim._remove_world_objects_client.requests) == 2, "an empty id list would remove every world object"


class _Simulator:
    async def delete_objects(self, obstacles: typing.Sequence[object]) -> list[bool]:
        return [True] * len(obstacles)

    async def pedestrian_delete(self, pedestrians: typing.Sequence[object]) -> list[bool]:
        return [True] * len(pedestrians)


def _removal_adapter(*obstacles: Obstacle) -> ArenaHumanSimulator:
    sim = _adapter()
    sim._remove_obstacles_client = _Client()
    sim._known_obstacles = KnownObstacles()
    sim._known_walls = KnownObstacles()
    sim._known_doors = KnownObstacles()
    sim._ped_bus_index = {}
    sim._simulator = _Simulator()
    for obstacle in obstacles:
        sim._known_obstacles.create_or_get(name=obstacle.name, obstacle=obstacle, layer=ObstacleLayer.UNUSED)
    return sim


def _chair(name: str) -> Obstacle:
    chair = Obstacle(name=name, model="chair", pose=Pose(Position(1.0, 1.0)))
    chair.sim_path = f"env_0/{name}"
    return chair


def test_removed_static_obstacles_leave_the_engine_by_sim_path() -> None:
    sim = _removal_adapter(_chair("chair_1"), _chair("chair_2"))
    _run(sim.remove_obstacles(ObstacleLayer.UNUSED))
    (removed,) = sim._remove_obstacles_client.requests
    assert list(removed.names) == ["env_0/chair_1", "env_0/chair_2"]
    (gone,) = sim._remove_world_objects_client.requests
    assert list(gone.object_ids) == ["env_0/chair_1", "env_0/chair_2"]
    assert dict(sim._known_obstacles.items()) == {}


def test_unused_static_obstacles_leave_the_engine_by_sim_path() -> None:
    sim = _removal_adapter(_chair("chair_1"))
    sim._hearing = None
    sim._driven = DrivenJoints()
    sim._joystick = JoystickDriver()
    sim._possession = types.SimpleNamespace(clear=lambda: None)
    sim._warned_unknown = set()
    sim._warned_bad_joints = set()

    async def no_pedestrians() -> bool:
        return True

    sim._remove_pedestrians_impl = no_pedestrians
    _run(sim.unuse_obstacles())
    (removed,) = sim._remove_obstacles_client.requests
    assert list(removed.names) == ["env_0/chair_1"]
    (gone,) = sim._remove_world_objects_client.requests
    assert list(gone.object_ids) == ["env_0/chair_1"]


def test_removing_no_static_obstacle_sends_no_filter() -> None:
    sim = _removal_adapter()
    _run(sim.remove_obstacles(ObstacleLayer.UNUSED))
    assert sim._remove_obstacles_client.requests == []
    assert sim._remove_world_objects_client.requests == []
