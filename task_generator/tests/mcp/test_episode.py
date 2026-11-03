from __future__ import annotations

import asyncio
import json
import time

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rclpy")

from arena_mcp.core import JOBS, ToolError
from arena_runtime_msgs.msg import EpisodeRecord

from arena_runtime.env.constants.runtime import EPISODE_PARAMS
from arena_runtime.env import entities_mcp as entities
from arena_runtime.env import episode_mcp as episode

from arena_runtime.testing.env_peer import ENV, MISSING_WORLD, PLAIN_ENV, ROBOT, SCENARIOS, WORLDS


def test_reset_blocks_until_the_next_episode_runs_and_returns_its_record(run, peers):
    before = peers.humansim.episode_id
    started = time.monotonic()
    reset = run(episode.episode_reset(env_id=ENV, world="small_warehouse", seed=42))
    assert time.monotonic() - started >= 0.3
    assert peers.humansim.resets[-1] == ("small_warehouse", 42)
    assert reset["episode_id"] == before + 1
    assert reset["outcome"] == "running"
    assert reset["seed"] == 42
    assert reset["world"] == "small_warehouse"
    assert reset["path_length"] == pytest.approx(3.5)
    assert reset["phases"] == {"jackal_0": {"phases": [{"goto": [1.0, 2.0, 0.0]}]}}
    assert reset["conditions"][0]["text"] == "no collisions"
    assert reset["params"] == {"task.random.n": 3}
    assert reset["start_time"] == pytest.approx(12.0)

    queued = run(episode.episode_reset(env_id=ENV, wait=False))
    assert queued == {"env": "/arena/env_7/arena_env_node", "requested": True, "previous_episode_id": before + 1}
    run(episode.episode_wait(env_id=ENV, until="running", timeout_s=10))


def test_reset_onto_a_world_that_fails_to_load_raises_the_node_error(run, peers):
    with pytest.raises(ToolError, match="FileNotFoundError") as failed:
        run(episode.episode_reset(env_id=ENV, world=MISSING_WORLD))
    assert MISSING_WORLD in failed.value.hint


def test_env_id_is_required_when_several_envs_run(run):
    with pytest.raises(ToolError, match="2 envs are up"):
        run(episode.episode_current())


def test_run_job_returns_the_outcome_with_the_seed_it_was_given(run, peers):
    job = run(episode.episode_run(env_id=ENV, seed=5))
    assert job["kind"] == "episode"
    done = run(JOBS.wait(job["id"], 20)).as_dict()
    assert done["state"] == "succeeded", done
    result = done["result"]
    assert result["state"] == "success"
    assert result["record"]["outcome"] == "success"
    assert result["record"]["seed"] == 5
    assert result["record"]["episode_id"] == result["episode_id"]


def test_run_rejects_a_second_goal_and_cancel_cancels_the_goal(run, peers):
    job = run(episode.episode_run(env_id=ENV, world="forever"))
    run(asyncio.sleep(0.3))
    started = time.monotonic()
    with pytest.raises(ToolError, match="rejected") as rejected:
        run(episode.episode_run(env_id=ENV))
    assert "single-flight" in str(rejected.value)
    assert time.monotonic() - started < 5
    cancelled = run(JOBS.cancel(job["id"])).as_dict()
    assert cancelled["state"] == "cancelled"
    deadline = time.monotonic() + 10
    while not peers.humansim.cancelled and time.monotonic() < deadline:
        run(asyncio.sleep(0.05))
    assert peers.humansim.cancelled
    while peers.humansim.running_goal and time.monotonic() < deadline:
        run(asyncio.sleep(0.05))


def test_run_fails_the_job_when_the_node_answers_busy(run):
    job = run(episode.episode_run(env_id=ENV, world="busy_result"))
    done = run(JOBS.wait(job["id"], 20)).as_dict()
    assert done["state"] == "failed"
    assert "already running" in done["error"]
    assert "auto_reset" in done["error"]


def test_history_keeps_terminal_outcomes_through_auto_reset_bursts(run, peers):
    run(episode.episode_history(env_id=ENV))
    first = peers.humansim.episode_id
    peers.humansim.burst([EpisodeRecord.SUCCESS, EpisodeRecord.FAILED, EpisodeRecord.FATAL])
    run(asyncio.sleep(0.5))
    history = run(episode.episode_history(env_id=ENV, limit=50))
    outcomes = {entry["episode_id"]: entry["outcome"] for entry in history}
    assert outcomes[first] == "success"
    assert outcomes[first + 1] == "failed"
    assert outcomes[first + 2] == "fatal"
    assert outcomes[first + 3] == "running"
    assert [entry["episode_id"] for entry in history] == sorted(outcomes, reverse=True)
    resource = json.loads(run(episode.history_resource(str(ENV))))
    assert resource[0]["episode_id"] == first + 3


def test_wait_terminal_returns_the_end_of_the_episode_running_at_call_time(run, peers):
    current = peers.humansim.episode_id

    async def scenario():
        waiter = asyncio.ensure_future(episode.episode_wait(env_id=ENV, until="terminal", timeout_s=10))
        await asyncio.sleep(0.3)
        assert not waiter.done()
        peers.humansim.burst([EpisodeRecord.SUCCESS])
        return await waiter

    matched = run(scenario())
    assert matched["record"]["episode_id"] == current
    assert matched["record"]["outcome"] == "success"
    with pytest.raises(ToolError, match="no terminal episode within 1s"):
        run(episode.episode_wait(env_id=ENV, until="terminal", timeout_s=1))
    with pytest.raises(ToolError, match="timeout_s"):
        run(episode.episode_wait(env_id=ENV, timeout_s=301))


def test_current_reports_current_queued_and_resetting(run):
    current = run(episode.episode_current(env_id=ENV))
    assert current["resetting"] is False
    assert current["current"]["outcome"] in ("running", "success", "skipped")
    assert current["queued"]["outcome"] == "queued"
    resource = json.loads(run(episode.current_resource(str(ENV))))
    assert resource["env"] == current["env"]


def test_queue_without_modules_keeps_them_and_types_params_by_declaration(run, peers):
    modes = run(episode.episode_task_modes(env_id=ENV, source="live"))
    doubles = [(mode, entry) for mode, entries in modes["robots"].items() for entry in entries if entry["type"] == "double"]
    pairs = [(mode, entry) for mode, entries in modes["obstacles"].items() for entry in entries if entry["type"] == "integer_array"]
    robots_mode, double = doubles[0]
    obstacles_mode, pair = pairs[0]
    from arena_runtime.env.task_layer import decode_modes

    from task_generator.tasks import layer

    spaces = layer.mode_namespaces()
    robots_name = f"{spaces['robots'][robots_mode]}.{double['name']}"
    obstacles_name = f"{spaces['obstacles'][obstacles_mode]}.{pair['name']}"
    queued = run(episode.episode_queue(env_id=ENV, modes={"robots": robots_mode, "obstacles": obstacles_mode}, params={robots_name: 3, obstacles_name: [1, 2]}))
    request = peers.humansim.queued[-1]
    assert decode_modes(request.modes) == {"robots": [robots_mode], "obstacles": [obstacles_mode]}
    sent = {param.name: param.value for param in request.params}
    assert (sent[robots_name].type, sent[robots_name].double_value) == (3, 3.0)
    assert (sent[obstacles_name].type, list(sent[obstacles_name].integer_array_value)) == (7, [1, 2])
    assert queued["queued"]["modes"]["robots"] == [robots_mode]
    assert queued["queued"]["params"][robots_name] == 3.0

    run(episode.episode_queue(env_id=ENV, modes={"modules": []}))
    assert decode_modes(peers.humansim.queued[-1].modes)["modules"] == []

    with pytest.raises(ToolError, match="are not parameters of the selected task modes") as unknown:
        run(episode.episode_queue(env_id=ENV, modes={"robots": robots_mode}, params={"not_a_leaf": 1}))
    assert double["name"] in str(unknown.value)
    with pytest.raises(ToolError, match="unknown modules mode"):
        run(episode.episode_queue(env_id=ENV, modes={"modules": ["not_a_module"]}))
    with pytest.raises(ToolError, match="unknown robots mode"):
        run(episode.episode_queue(env_id=ENV, modes={"robots": "not_a_mode"}))


def test_task_modes_live_and_registry_agree_and_list_the_fleet_only_null(run):
    import importlib

    from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES

    for package in ("robots", "obstacles", "modules"):
        importlib.import_module(f"task_generator.tasks.{package}")
    live = run(episode.episode_task_modes(env_id=ENV, source="live"))
    offline = run(episode.episode_task_modes(source="registry"))
    assert live["source"] == "live" and offline["source"] == "registry"
    for axis, registry in (("robots", ROBOTS_MODES), ("obstacles", OBSTACLES_MODES), ("modules", MODULE_MODES)):
        assert sorted(live[axis]) == sorted(key.value for key in registry.keys())
        assert sorted(offline[axis]) == sorted(live[axis])
        for mode in live[axis]:
            assert [entry["name"] for entry in live[axis][mode]] == [entry["name"] for entry in offline[axis][mode]]
    assert "null" in live["fleet_only_robots"]
    assert any(entry["description"] for entries in offline["obstacles"].values() for entry in entries)
    active = run(episode.episode_task_modes_active(env_id=ENV))
    assert active["modes"]["robots"][0] in live["robots"]


def test_params_read_the_episode_catalog_and_set_coerces_to_declared_types(run, peers):
    params = run(episode.episode_params(env_id=ENV))
    assert sorted(params) == sorted(EPISODE_PARAMS)
    assert params["task.episode.count"]["description"] == EPISODE_PARAMS["task.episode.count"]
    declared = peers.humansim.get_parameter("task.episode.timeout").type_
    written = run(episode.episode_set_params({"task.episode.timeout": 30.0, "task.episode.auto_reset": "false"}, env_id=ENV))
    assert written["task.episode.auto_reset"] is False
    assert written["task.episode.timeout"] == 30
    assert peers.humansim.get_parameter("task.episode.timeout").type_ == declared
    with pytest.raises(ToolError, match="not episode parameters"):
        run(episode.episode_set_params({"robot.train": True}, env_id=ENV))
    with pytest.raises(ToolError, match="not episode parameters"):
        run(episode.episode_params(env_id=ENV, names=["sim"]))
    run(episode.episode_set_params({"task.episode.auto_reset": True}, env_id=ENV))


def test_config_reads_every_configuration_param_that_is_not_an_episode_param(run):
    config = run(episode.episode_config(env_id=ENV))
    assert config["human"] == "arena"
    assert config["world"] == "map_empty"
    for name in ("sim", "acoustics", "world.lighting", "kg.derive", "robot.mobile_adapter", "robot.hearing", "human.robots.visible", "task.config", "task.robots"):
        assert name in config
    assert not set(config) & set(EPISODE_PARAMS)
    assert run(episode.episode_config(env_id=PLAIN_ENV))["human"] == "dummy"


def test_catalog_lists_each_kind_from_its_query_service(run):
    assert run(episode.episode_catalog("worlds", env_id=ENV)) == sorted(WORLDS)
    assert run(episode.episode_catalog("scenarios", env_id=ENV, world="small_warehouse")) == SCENARIOS["small_warehouse"]
    assert run(episode.episode_catalog("scenarios", env_id=ENV)) == SCENARIOS["map_empty"]
    assert run(episode.episode_catalog("robots", env_id=ENV)) == ["burger", "jackal"]
    assert run(episode.episode_catalog("environments", env_id=ENV)) == []
    world = run(episode.episode_wait_for_world(env_id=ENV))
    assert world["world"] == "map_empty"


def test_submit_task_returns_phase_indices_and_passes_the_node_error_through(run, peers):
    task = "- goto: [1.0, 2.0, 0.0]\n- goto: [3.0, 2.0, 1.5]\n  tolerance_radius: 0.5\n"
    first = run(episode.episode_submit_task(task, env_id=ENV))
    assert first["robot"] == ROBOT
    second = run(episode.episode_submit_task("phases:\n  - goto: [0.0, 0.0, 0.0]\n", env_id=ENV, robot=ROBOT))
    assert second["phase_indices"] == [first["phase_indices"][-1] + 1]
    calls = len(peers.humansim.submitted)
    with pytest.raises(ToolError, match="task rejected: .*wobble"):
        run(episode.episode_submit_task("- goto: [1, 2, 0]\n  wobble: 3\n", env_id=ENV))
    assert len(peers.humansim.submitted) == calls + 1
    with pytest.raises(ToolError, match="no robot 'ghost'"):
        run(episode.episode_submit_task(task, env_id=ENV, robot="ghost"))
    assert len(peers.humansim.submitted) == calls + 1


def test_submit_task_with_an_empty_phase_list_stops_the_robot(run, peers):
    stops = peers.humansim.stops
    for task_yaml in ("[]", ""):
        stopped = run(episode.episode_submit_task(task_yaml, env_id=ENV))
        assert stopped["phase_indices"] == []
        assert peers.humansim.submitted[-1] == (ROBOT, task_yaml)
    assert peers.humansim.stops == stops + 2


def test_both_domains_install_into_one_server_and_dispatch_by_name(run):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("task_generator_domains")
    episode.domain.install(server)
    entities.domain.install(server)
    listed = {tool.name for tool in run(server.list_tools())}
    assert listed == {spec.name for spec in (*episode.domain.tools, *entities.domain.tools)}
    result = run(server.call_tool("episode_catalog", {"kind": "worlds", "env_id": ENV}))
    assert not result.is_error
    assert "small_warehouse" in json.dumps(result.model_dump(mode="json"))
    templates = {template.uri_template for template in run(server.list_resource_templates())}
    assert templates == {"arena://episode/{env_id}/current", "arena://episode/{env_id}/history"}
