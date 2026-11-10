"""The shipped mechanisms agent types (configs/agent_types/mechanisms) run in the real humansim engine."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim")

AGENT_TYPES = Path(__file__).resolve().parents[3] / "arena_simulation_setup" / "configs" / "agent_types" / "mechanisms"

ELEVATOR = (8.0, 3.0)  # cabin centre, door facing -x
QUEUE = (7.0, 3.0)  # doorway
EXIT = (2.0, 5.0)
DOOR = (5.0, 1.0)
HOLDER_EXIT = (1.0, 1.0)


@pytest.fixture
def manager(rclpy_context: object, request: pytest.FixtureRequest) -> Iterator[object]:
    from arena_humansim.core.agent_manager import AgentManager
    from arena_humansim.params import MODE_SUBSYSTEM
    from rclpy.node import Node
    from rclpy.parameter import Parameter

    overrides = [
        Parameter("seed", Parameter.Type.INTEGER, 5),
        Parameter("dt", Parameter.Type.DOUBLE, 0.05),
        Parameter("bt_tick_interval", Parameter.Type.INTEGER, 1),
        Parameter("global_planner", Parameter.Type.STRING, "navmesh"),
        Parameter("mode", Parameter.Type.STRING, MODE_SUBSYSTEM),
        Parameter("publish_markers", Parameter.Type.INTEGER, 0),
    ]
    orig_init = Node.__init__

    def patched(self, _name: str, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        kwargs.setdefault("parameter_overrides", overrides)
        orig_init(self, f"mb_types_{request.node.name}"[:50], *args, **kwargs)

    Node.__init__ = patched
    try:
        mgr = AgentManager()
    finally:
        Node.__init__ = orig_init
    _world(mgr)
    yield mgr
    mgr.destroy_node()


def _world(mgr: object) -> None:
    from arena_humansim_msgs.msg import WorldObjectInfo
    from arena_humansim_msgs.srv import AddWalls, AddWorldObjects
    from geometry_msgs.msg import Point32, Pose2D

    walls = [((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (10.0, 6.0)), ((10.0, 6.0), (0.0, 6.0)), ((0.0, 6.0), (0.0, 0.0))]
    req = AddWalls.Request()
    for i, ((sx, sy), (ex, ey)) in enumerate(walls):
        req.names.append(f"w{i}")
        req.starts.append(Point32(x=sx, y=sy))
        req.ends.append(Point32(x=ex, y=ey))
    mgr._add_walls_callback(req, AddWalls.Response())

    def obj(object_id: str, kind: str, xy: tuple[float, float], theta: float = 0.0, line: bool = False) -> WorldObjectInfo:
        info = WorldObjectInfo(object_id=object_id, type=kind, pose=Pose2D(x=xy[0], y=xy[1], theta=theta), capacity=0)
        if line:
            info.formation_type = "line"
            info.formation_param_keys = ["front_offset", "base_step"]
            info.formation_param_values = [0.7, 0.8]
        return info

    objects = [
        obj("elev_a", "elevator", ELEVATOR, math.pi),
        obj("elev_a/queue", "elevator_queue", QUEUE, 0.0, line=True),
        obj("exit", "elevator_exit", EXIT),
        obj("door_a", "door", DOOR),
        obj("holder_exit", "door_holder_exit", HOLDER_EXIT),
    ]
    mgr._add_world_objects_callback(AddWorldObjects.Request(objects=objects), AddWorldObjects.Response())


def _spawn(mgr: object, agent_type: str, xy: tuple[float, float], name: str) -> int:
    from arena_humansim_msgs.msg import AgentState
    from arena_humansim_msgs.srv import SpawnAgents
    from geometry_msgs.msg import Pose2D

    msg = AgentState(agent_id=0, name=name, pose=Pose2D(x=xy[0], y=xy[1]), desired_velocity=1.2, agent_type=str(AGENT_TYPES / f"{agent_type}.yaml"))
    resp = mgr._spawn_agents_callback(SpawnAgents.Request(agents=[msg]), SpawnAgents.Response())
    assert resp.success, resp.message
    return int(resp.spawned_ids[0])


def _sequence(mgr: object, aid: int) -> str:
    from arena_humansim.core.behavior.nodes import SequenceStateMachine

    root = mgr._behavior_trees[aid].root
    nodes = [root, *root.children]
    return next(n for n in nodes if isinstance(n, SequenceStateMachine))._current_name


def _pos(mgr: object, aid: int) -> tuple[float, float]:
    idx = mgr._pool.idx(aid)
    return float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1])


def _run(mgr: object, seconds: float) -> None:
    for _ in range(int(round(seconds / mgr._dt))):
        mgr.tick()


def _notify(mgr: object, aid: int, stimulus: str, intensity: float = 1.0) -> None:
    from arena_humansim_msgs.srv import NotifyStimulus

    assert mgr._notify_stimulus_callback(NotifyStimulus.Request(agent_id=aid, stimulus=stimulus, intensity=intensity), NotifyStimulus.Response()).success


def test_shipped_types_load() -> None:
    from arena_humansim.core.agents.loader import load_agent_type_from_file

    for name in ("elevator_rider", "door_holder"):
        agent_type = load_agent_type_from_file(AGENT_TYPES / f"{name}.yaml")
        assert agent_type.mode == "behavior_tree"
        assert agent_type.name == name


def test_elevator_rider_queues_boards_on_open_and_leaves_after_the_ride(manager: object) -> None:
    aid = _spawn(manager, "elevator_rider", (3.0, 3.0), "env_0/rider")
    _run(manager, 6.0)
    assert _sequence(manager, aid) == "queue"
    x, _ = _pos(manager, aid)
    assert x > 4.5, "the rider walks up to the line in front of the cabin"
    _run(manager, 4.0)
    assert math.dist(_pos(manager, aid), ELEVATOR) > 1.0, "it waits outside while the door is closed"

    _notify(manager, aid, "elevator.cabin_door==open")
    _run(manager, 8.0)
    assert _sequence(manager, aid) == "board"
    assert math.dist(_pos(manager, aid), ELEVATOR) < 0.8, "it rides in the cabin"
    _run(manager, 5.0)
    assert math.dist(_pos(manager, aid), ELEVATOR) < 0.8, "and stays there until teleported"

    from arena_humansim_msgs.srv import UpdateRobot
    from geometry_msgs.msg import Pose2D

    manager._update_robot_callback(UpdateRobot.Request(name="env_0/rider", pose=Pose2D(x=8.0, y=4.5), radius=0.25), UpdateRobot.Response())
    _notify(manager, aid, "teleported")
    _run(manager, 12.0)
    assert _sequence(manager, aid) == "leave"
    assert math.dist(_pos(manager, aid), EXIT) < 1.0


def test_rider_sees_a_door_already_open_at_spawn(manager: object) -> None:
    _notify(manager, -1, "elevator.cabin_door==open")
    aid = _spawn(manager, "elevator_rider", (3.0, 3.0), "env_0/late_rider")
    manager.tick()
    manager.tick()
    assert _sequence(manager, aid) == "board"


def test_door_holder_stands_in_the_doorway_then_leaves(manager: object) -> None:
    aid = _spawn(manager, "door_holder", (3.0, 3.0), "env_0/holder")
    _run(manager, 4.0)
    assert math.dist(_pos(manager, aid), DOOR) < 0.8
    assert _sequence(manager, aid) == "hold"
    _run(manager, 18.0)
    assert _sequence(manager, aid) == "leave"
    assert math.dist(_pos(manager, aid), HOLDER_EXIT) < 1.0
