from __future__ import annotations

import asyncio
import math
import time

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rclpy")

from arena_mcp.core import ToolError

from arena_runtime.env import entities_mcp as entities

from arena_runtime.testing.env_peer import ENV, PLAIN_ENV, ROBOT


def test_spawn_static_sends_a_pose_only_when_both_coordinates_are_given(run, peers):
    placed = run(entities.entities_spawn_static("shelf", env_id=ENV, x=1.5, y=-2.0, yaw=math.pi / 2))
    assert placed["id"] == "env_7/ext_shelf"
    kind, request = peers.humansim.spawns[-1]
    assert kind == "static"
    assert request.use_pose is True
    assert (request.pose.pose.position.x, request.pose.pose.position.y) == (1.5, -2.0)
    assert request.pose.pose.orientation.z == pytest.approx(math.sin(math.pi / 4))
    run(entities.entities_spawn_dynamic("adult", env_id=ENV))
    kind, request = peers.humansim.spawns[-1]
    assert kind == "dynamic"
    assert request.use_pose is False
    calls = len(peers.humansim.spawns)
    with pytest.raises(ToolError, match="both x and y"):
        run(entities.entities_spawn_static("shelf", env_id=ENV, x=1.0))
    assert len(peers.humansim.spawns) == calls
    with pytest.raises(ToolError, match="not resolvable"):
        run(entities.entities_spawn_static("missing", env_id=ENV))


def test_spawn_robot_forwards_args_and_despawn_and_move_reach_the_node(run, peers):
    spawned = run(entities.entities_spawn_robot("burger", env_id=ENV, immediate=True, args={"mobile": "nav2", "robot.train": False, "radius": 0.2}))
    assert spawned["name"] == "burger_1"
    _, request = peers.humansim.spawns[-1]
    assert request.immediate is True
    assert {pair.key: pair.value for pair in request.args} == {"mobile": "nav2", "robot.train": "false", "radius": "0.2"}
    run(entities.entities_despawn_robot("burger_1", env_id=ENV))
    assert peers.humansim.despawns[-1] == "burger_1"
    moved = run(entities.entities_move(ROBOT, 4.0, 5.0, env_id=ENV, yaw=1.0))
    assert moved["name"] == ROBOT
    request = peers.humansim.moves[-1]
    assert request.pose.header.frame_id == "map"
    assert request.pose.pose.position.x == 4.0
    with pytest.raises(ToolError, match="unknown entity"):
        run(entities.entities_move("ghost", 0.0, 0.0, env_id=ENV))
    assert run(entities.entities_require_map(env_id=ENV))["map_server"] is True


def test_semantics_snapshot_pairs_names_with_values_and_set_stringifies(run, peers):
    snapshot = run(entities.entities_semantics(env_id=ENV))
    door = snapshot["entities"][0]
    assert door == {"entity": "env_7/door_1", "kind": "door", "discrete": {"state": "closed"}, "continuous": {}, "predicates": {"locked": False}, "members": ["adult_1"]}
    run(entities.entities_set_semantic("env_7/door_1", "locked", True, env_id=ENV))
    assert peers.humansim.semantic_writes[-1] == ("env_7/door_1", "locked", "true")
    run(entities.entities_set_semantic("env_7/door_1", "state", "open", env_id=ENV))
    assert peers.humansim.semantic_writes[-1][2] == "open"
    with pytest.raises(ToolError, match="not writable"):
        run(entities.entities_set_semantic("env_7/door_1", "width", 2.0, env_id=ENV))


def test_spawn_sound_customizes_playback_only_with_a_volume(run, peers):
    spawned = run(entities.entities_spawn_sound("music", 1.0, 2.0, env_id=ENV, frame="env_7/jackal_0/base_link", attach_to_frame=True))
    assert spawned["entity"] == "runtime_music_1"
    request = peers.humansim.sounds[-1]
    assert request.customize_playback is False
    assert request.attach_to_frame is True
    assert request.pose.header.frame_id == "env_7/jackal_0/base_link"
    run(entities.entities_spawn_sound("music", 1.0, 2.0, env_id=ENV, loop=False, source_volume_db=65.0))
    request = peers.humansim.sounds[-1]
    assert request.customize_playback is True
    assert request.loop is False
    assert request.source_volume_db == pytest.approx(65.0)
    with pytest.raises(ToolError, match="source_volume_db"):
        run(entities.entities_spawn_sound("music", 1.0, 2.0, env_id=ENV, asset_id="radio"))
    with pytest.raises(ToolError, match="unknown environment sound kind"):
        run(entities.entities_spawn_sound("thunder", 1.0, 2.0, env_id=ENV))
    run(entities.entities_remove_sound("runtime_music_1", env_id=ENV))
    with pytest.raises(ToolError, match="unknown sound"):
        run(entities.entities_remove_sound("music_1", env_id=ENV))


def test_pedestrians_read_move_and_stream(run, peers):
    peds = run(entities.entities_pedestrians(env_id=ENV))
    assert peds["frame"] == "map"
    adult = peds["pedestrians"][0]
    assert (adult["name"], adult["x"], adult["y"], adult["animation_state"]) == ("adult_1", 2.0, 3.0, "walking")
    assert "joint_state" in run(entities.entities_pedestrians(env_id=ENV, full=True))["pedestrians"][0]
    moved = run(entities.entities_move_pedestrians([entities.PedestrianPose(name="adult_1", x=5.0, y=6.0), entities.PedestrianPose(name="ghost", x=0.0, y=0.0)], env_id=ENV))
    assert moved["results"] == {"adult_1": "success", "ghost": "not_found"}
    assert peers.humansim.ped_moves[-1].pedestrians[0].pose.position.x == 5.0
    with pytest.raises(ToolError, match="no pedestrian moved"):
        run(entities.entities_move_pedestrians([entities.PedestrianPose(name="ghost", x=0.0, y=0.0)], env_id=ENV))
    count = len(peers.humansim.streamed)
    streamed = run(entities.entities_stream_pedestrians([entities.PedestrianState(name="adult_1", x=1.0, y=1.0, vx=0.5, animation_state="running")], env_id=ENV))
    assert streamed["published"] == ["adult_1"]
    deadline = time.monotonic() + 5
    while len(peers.humansim.streamed) == count and time.monotonic() < deadline:
        run(asyncio.sleep(0.05))
    message = peers.humansim.streamed[-1]
    assert message.pedestrians[0].animation_state == 2
    assert message.pedestrians[0].twist.linear.x == 0.5
    with pytest.raises(ToolError, match="unknown animation states"):
        run(entities.entities_stream_pedestrians([entities.PedestrianState(name="adult_1", x=1.0, y=1.0, animation_state="moonwalk")], env_id=ENV))


def test_humansim_calls_reach_its_services_and_resolve_agent_names(run, peers):
    agents = run(entities.entities_ped_agents(env_id=ENV))
    assert agents["agents"][0] == {"agent_id": 4, "name": "anna", "agent_type": "adult", "handedness": "r"}
    assert agents["policies"] == ["social_force"]
    run(entities.entities_ped_waypoints("anna", [entities.Waypoint(x=1.0, y=2.0), entities.Waypoint(x=3.0, y=4.0, radius=0.5)], env_id=ENV, mode="reverse"))
    name, request = peers.humansim.humansim[-1]
    assert name == "set_waypoints"
    assert (request.agent_id, request.name) == (0, "anna")
    assert request.waypoints.mode == request.waypoints.MODE_REVERSE
    assert request.waypoints.points[1].radius == 0.5
    run(entities.entities_ped_waypoints(9, [], env_id=ENV))
    assert peers.humansim.humansim[-1][1].agent_id == 9
    stimulus = run(entities.entities_ped_stimulus("curiosity", 0.4, env_id=ENV, ped="anna"))
    assert stimulus["agent_id"] == 4
    assert run(entities.entities_ped_stimulus("panic", 1.0, env_id=ENV))["agent_id"] == -1
    with pytest.raises(ToolError, match="no humansim agent named"):
        run(entities.entities_ped_stimulus("panic", 1.0, env_id=ENV, ped="bob"))
    with pytest.raises(ToolError, match="intensity"):
        run(entities.entities_ped_stimulus("panic", 2.0, env_id=ENV))
    flow = run(entities.entities_ped_flow([{"name": "door", "pose": {"x": 1.0, "y": 2.0, "theta": 0.0}, "max_concurrent": 5}], [{"name": "exit", "capacity": 3}], env_id=ENV))
    assert flow["sources"] == ["door"] and flow["sinks"] == ["exit"]
    assert peers.humansim.humansim[-1][1].flow.sources[0].max_concurrent == 5
    with pytest.raises(ToolError, match="cannot build"):
        run(entities.entities_ped_flow([{"name": "door", "wobble": 1}], [], env_id=ENV))
    assert run(entities.entities_ped_add_source({"name": "gate"}, env_id=ENV))["name"] == "add_source_1"
    assert run(entities.entities_ped_add_sink({"name": "hole", "absorption_radius": 0.5}, env_id=ENV))["name"] == "add_sink_1"
    run(entities.entities_ped_remove_source("gate", env_id=ENV))
    run(entities.entities_ped_remove_sink("hole", env_id=ENV))
    assert [entry[0] for entry in peers.humansim.humansim[-4:]] == ["add_source", "add_sink", "remove_source", "remove_sink"]
    profile = run(entities.entities_ped_profile(env_id=ENV, reset=True))
    assert profile["phases"] == {"social_force": {"mean_ms": 0.5, "p95_ms": 0.9}}
    assert peers.humansim.humansim[-1][1].reset is True


def test_humansim_calls_fail_fast_on_another_human_simulator(run, peers):
    started = time.monotonic()
    with pytest.raises(ToolError, match="needs humansim|need humansim") as failure:
        run(entities.entities_ped_waypoints("anna", [], env_id=PLAIN_ENV))
    assert "human:=arena" in str(failure.value)
    assert time.monotonic() - started < 3
    assert peers.plain.humansim == []
