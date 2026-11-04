"""EpisodeNode's next-episode queue against a fake task layer: what the next record is built from, when the layer is asked, and what survives a failure."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.msg")

TIMEOUT = 60.0


async def _until(predicate: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.05)


async def _on_running_env(body: Callable[[object, object], Awaitable[None]]) -> None:
    import rclpy.parameter

    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.testing.env_host import arena_on_dummy

    async with arena_on_dummy(on_taken=pytest.skip) as rig:
        overrides = [
            rclpy.parameter.Parameter("env_id", value=rig.env_id),
            rclpy.parameter.Parameter("world", value="map_empty"),
            rclpy.parameter.Parameter("sim", value="dummy"),
            rclpy.parameter.Parameter("human", value="dummy"),
            rclpy.parameter.Parameter("acoustics", value="none"),
            rclpy.parameter.Parameter("task.episode.auto_reset", value=False),
        ]
        env = EpisodeNode(namespace=f"/{rig.namespace}", parameter_overrides=overrides)
        rig.executor.add_node(env)
        try:
            await asyncio.wait_for(env.setup(), TIMEOUT)
            env.trigger_activate()
            try:
                await body(env, rig)
            finally:
                await asyncio.wait_for(env.teardown(), TIMEOUT)
        finally:
            rig.executor.remove_node(env)
            env.destroy_node()


def _watch_queue(env: object, rig: object) -> list:
    import rclpy.qos
    from arena_runtime_msgs.msg import EpisodeRecord

    received: list = []
    qos = rclpy.qos.QoSProfile(depth=50, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
    rig.probe.create_subscription(EpisodeRecord, f"{env.get_fully_qualified_name()}/state/queue", received.append, qos)
    return received


async def _queue(env: object, **fields: object) -> None:
    import arena_runtime_msgs.srv

    request = arena_runtime_msgs.srv.QueueEpisode.Request(action=arena_runtime_msgs.srv.QueueEpisode.Request.MERGE)
    for name, value in fields.items():
        setattr(request, name, value)
    response = await env._cb_queue_episode(request, arena_runtime_msgs.srv.QueueEpisode.Response())
    assert response.success, response.error_msg


def _param(name: str, value: int) -> object:
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

    return Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value))


def _key_values(**axes: str) -> list:
    from diagnostic_msgs.msg import KeyValue

    return [KeyValue(key=axis, value=value) for axis, value in axes.items()]


def test_the_layer_is_asked_with_the_previous_record_and_its_answer_reaches_the_next_record(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime_msgs.action import RunEpisode

    from arena_runtime.env.task_layer import TaskModeOverrides

    answers = iter([TaskModeOverrides(modes={"robots": ["fake_other"]}, seed=41), TaskModeOverrides(modes={"robots": ["fake_robots"]}, world="map_empty", seed=42), TaskModeOverrides()])
    fake_task_layer.choose = lambda previous, queued: next(answers)
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        first = env._episodes.current
        assert (first.episode_id, first.seed, first.modes["robots"]) == (1, 41, ["fake_other"])
        assert fake_task_layer.params.selected[-1]["robots"] == ["fake_other"]
        assert [(record.episode_id, record.outcome_state) for record in fake_task_layer.previous] == [(0, 0), (1, RunEpisode.Result.SUCCESS)]

        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        second = env._episodes.current
        assert (second.episode_id, second.seed, second.world, second.modes["robots"]) == (2, 42, "map_empty", ["fake_robots"])
        assert [(record.episode_id, record.outcome_state) for record in fake_task_layer.previous][2:] == [(2, RunEpisode.Result.SUCCESS)]

    asyncio.run(_on_running_env(body))


def test_the_queue_preview_published_before_an_episode_equals_the_record_it_builds(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime.env.task_layer import TaskModeOverrides, decode_modes

    fake_task_layer.choose = lambda previous, queued: TaskModeOverrides(modes={"robots": ["fake_other"]}, seed=5) if previous.episode_id == 1 else queued
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        queue = _watch_queue(env, rig)
        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        await _until(lambda: queue and queue[-1].episode_id == env._episodes.current.episode_id + 1 and queue[-1].seed == 5, "the preview of the next episode")
        preview = queue[-1]

        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        built = env._episodes.current
        assert (preview.episode_id, preview.world, preview.seed, decode_modes(preview.modes)) == (built.episode_id, built.world, built.seed, built.modes)
        assert preview.seed == 5

    asyncio.run(_on_running_env(body))


def test_a_queued_modes_and_seed_show_in_the_preview_and_params_overlay_the_live_values(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime.env.task_layer import decode_modes

    async def body(env: object, rig: object) -> None:
        queue = _watch_queue(env, rig)
        await _queue(env, modes=_key_values(robots="fake_other"), params=[_param("task.fake_other.level", 9)])
        await _until(lambda: queue and decode_modes(queue[-1].modes).get("robots") == ["fake_other"], "the preview of the queued robots mode")
        preview = queue[-1]
        assert {p.name: p.value.integer_value for p in preview.params}["task.fake_other.level"] == 9
        assert env.get_parameter("task.fake_other.level").value == 1

    asyncio.run(_on_running_env(body))


def test_an_explicit_seed_wins_over_the_one_the_layer_chose(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime.env.task_layer import TaskModeOverrides

    fake_task_layer.choose = lambda previous, queued: TaskModeOverrides(seed=41)
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        await asyncio.wait_for(env._run_episode(seed=7), TIMEOUT)
        assert env._episodes.current.seed == 7
        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        assert env._episodes.current.seed == 41

    asyncio.run(_on_running_env(body))


def test_reset_episode_hands_its_world_and_seed_to_the_episode_it_starts(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv
    from arena_runtime_msgs.action import RunEpisode

    async def body(env: object, rig: object) -> None:
        first = asyncio.create_task(env._run_episode())
        await _until(lambda: env._episodes.current.outcome_state == RunEpisode.Result.RUNNING, "episode 1 running")
        response = await env._cb_reset_episode(arena_runtime_msgs.srv.ResetEpisode.Request(world="map_empty", seed=9), arena_runtime_msgs.srv.ResetEpisode.Response())
        assert response.success
        result = await asyncio.wait_for(first, TIMEOUT)
        assert (result.state, result.info) == (RunEpisode.Result.SKIPPED, "reset")
        await _until(lambda: env._episodes.current.episode_id == 2 and env._episodes.current.outcome_state == RunEpisode.Result.RUNNING, "episode 2 running")
        assert (env._episodes.current.world, env._episodes.current.seed) == ("map_empty", 9)
        fake_task_layer.task.finished = True
        await _until(lambda: env._episodes.current.outcome_state == RunEpisode.Result.SUCCESS, "episode 2 finished")

    asyncio.run(_on_running_env(body))


def test_a_queued_empty_module_list_replaces_the_modules_of_the_next_episode(rclpy_context: object, fake_task_layer: object) -> None:
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        await _queue(env, modes=_key_values(modules=""))
        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        assert env._episodes.current.modes == {"robots": ["fake_robots"], "obstacles": ["fake_obstacles"], "modules": []}
        assert fake_task_layer.params.selected[-1]["modules"] == []

    asyncio.run(_on_running_env(body))


def test_a_layer_that_fails_when_the_next_episode_builds_ends_it_fatal_twice_and_the_third_build_uses_the_queue(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime_msgs.action import RunEpisode

    calls: list[int] = []

    def choose(previous: object, queued: object) -> object:
        calls.append(previous.episode_id)
        if len(calls) <= 2:
            raise RuntimeError("no answer")
        return queued

    fake_task_layer.choose = choose
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        await _queue(env, modes=_key_values(robots="fake_other"), params=[_param("task.fake_other.level", 4)])
        for _ in range(2):
            result = await asyncio.wait_for(env._run_episode(), TIMEOUT)
            assert result.state == RunEpisode.Result.FATAL
            assert "no answer" in result.info
            assert env._episodes.current.episode_id == 0
            assert env._episodes.pending.modes == {"robots": ["fake_other"]}
            assert not env._episodes.decided

        result = await asyncio.wait_for(env._run_episode(), TIMEOUT)
        assert result.state == RunEpisode.Result.SUCCESS
        assert env._episodes.current.modes["robots"] == ["fake_other"]
        assert env.get_parameter("task.fake_other.level").value == 4

    asyncio.run(_on_running_env(body))


def test_a_layer_that_fails_when_an_episode_ends_is_asked_again_when_the_next_one_builds(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime_msgs.action import RunEpisode

    from arena_runtime.env.task_layer import TaskModeOverrides

    calls: list[int] = []

    def choose(previous: object, queued: object) -> object:
        calls.append(previous.episode_id)
        if len(calls) == 2:
            raise RuntimeError("no answer")
        return TaskModeOverrides(modes={"robots": ["fake_other"]}) if len(calls) == 3 else queued

    fake_task_layer.choose = choose
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        result = await asyncio.wait_for(env._run_episode(), TIMEOUT)
        assert result.state == RunEpisode.Result.SUCCESS
        assert not env._episodes.decided

        result = await asyncio.wait_for(env._run_episode(), TIMEOUT)
        assert result.state == RunEpisode.Result.SUCCESS
        assert calls == [0, 1, 1, 2]
        assert env._episodes.current.modes["robots"] == ["fake_other"]

    asyncio.run(_on_running_env(body))


def test_staged_params_are_applied_at_the_reset_and_an_undeclared_name_is_dropped(rclpy_context: object, fake_task_layer: object) -> None:
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        await _queue(env, modes=_key_values(robots="fake_other"), params=[_param("task.fake_other.level", 3), _param("task.fake_module.level", 6), _param("task.nowhere.level", 1)])
        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        assert env.get_parameter("task.fake_other.level").value == 3
        assert env.get_parameter("task.fake_module.level").value == 6
        assert env.get_parameter("task.fake_robots.level").value == 1
        assert not env.has_parameter("task.nowhere.level")
        assert env._staged_params == {}

    asyncio.run(_on_running_env(body))


def test_an_empty_value_on_a_single_mode_axis_is_refused_and_leaves_the_queue_as_it_was(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv

    async def body(env: object, rig: object) -> None:
        await _queue(env, modes=_key_values(robots="fake_other"))
        request = arena_runtime_msgs.srv.QueueEpisode.Request(action=arena_runtime_msgs.srv.QueueEpisode.Request.MERGE, modes=_key_values(robots=""))
        response = await env._cb_queue_episode(request, arena_runtime_msgs.srv.QueueEpisode.Response())
        assert not response.success
        assert response.error_msg == "task axis 'robots' takes exactly one mode, got [], allowed: fake_robots, fake_other"
        assert env._episodes.pending.modes == {"robots": ["fake_other"]}

    asyncio.run(_on_running_env(body))


def test_staged_params_of_an_always_on_module_survive_a_queue_that_clears_the_modules(rclpy_context: object, fake_task_layer: object) -> None:
    fake_task_layer.params.always_on = ["fake_module"]
    fake_task_layer.task.finished = True

    async def body(env: object, rig: object) -> None:
        await _queue(env, params=[_param("task.fake_module.level", 6)])
        await _queue(env, modes=_key_values(modules=""))
        assert set(env._staged_params) == {"task.fake_module.level"}

        await asyncio.wait_for(env._run_episode(), TIMEOUT)
        assert env._episodes.current.modes["modules"] == ["fake_module"]
        assert env.get_parameter("task.fake_module.level").value == 6

    asyncio.run(_on_running_env(body))


def test_staged_params_of_a_namespace_two_axes_share_survive_until_both_axes_drop_it(rclpy_context: object, fake_task_layer: object) -> None:
    fake_task_layer.NAMESPACES = {
        "robots": {"fake_shared": "task.fake_shared", "fake_robots": "task.fake_robots"},
        "obstacles": {"fake_shared": "task.fake_shared", "fake_obstacles": "task.fake_obstacles"},
        "modules": {"fake_module": "task.fake_module"},
    }
    fake_task_layer.params.modes = {"robots": ["fake_shared"], "obstacles": ["fake_shared"], "modules": ["fake_module"]}

    async def body(env: object, rig: object) -> None:
        await _queue(env, params=[_param("task.fake_shared.level", 5)])
        await _queue(env, modes=_key_values(obstacles="fake_obstacles"))
        assert {name: value.integer_value for name, value in env._staged_params.items()} == {"task.fake_shared.level": 5}

        await _queue(env, modes=_key_values(robots="fake_robots"))
        assert env._staged_params == {}

    asyncio.run(_on_running_env(body))


def test_staged_params_of_a_switched_off_mode_are_dropped_with_it(rclpy_context: object, fake_task_layer: object) -> None:
    async def body(env: object, rig: object) -> None:
        await _queue(env, params=[_param("task.fake_module.level", 6)])
        await _queue(env, modes=_key_values(modules=""))
        assert env._staged_params == {}

    asyncio.run(_on_running_env(body))


def test_a_queue_merged_while_the_layer_decides_is_kept_and_the_decision_does_not_overwrite_it(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime_msgs.action import RunEpisode

    from arena_runtime.env.task_layer import TaskModeOverrides

    fake_task_layer.choose = lambda previous, queued: TaskModeOverrides(seed=5)

    async def body(env: object, rig: object) -> None:
        first = asyncio.create_task(env._run_episode())
        await _until(lambda: env._episodes.current.outcome_state == RunEpisode.Result.RUNNING, "episode 1 running")
        fake_task_layer.gate = asyncio.Event()
        fake_task_layer.task.finished = True
        await _until(lambda: len(fake_task_layer.previous) == 2, "the layer being asked at the episode end")

        queued = asyncio.create_task(_queue(env, modes=_key_values(robots="fake_other")))
        await asyncio.sleep(0.2)
        assert not queued.done()
        fake_task_layer.gate.set()
        await asyncio.wait_for(asyncio.gather(first, queued), TIMEOUT)

        assert env._episodes.pending.modes == {"robots": ["fake_other"]}
        assert env._episodes.pending.seed == 5

    asyncio.run(_on_running_env(body))


def test_a_reset_episode_arriving_while_the_layer_decides_starts_the_next_episode_with_its_seed(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv
    from arena_runtime_msgs.action import RunEpisode

    async def body(env: object, rig: object) -> None:
        first = asyncio.create_task(env._run_episode())
        await _until(lambda: env._episodes.current.outcome_state == RunEpisode.Result.RUNNING, "episode 1 running")
        fake_task_layer.gate = asyncio.Event()
        fake_task_layer.task.finished = True
        await _until(lambda: len(fake_task_layer.previous) == 2, "the layer being asked at the episode end")
        fake_task_layer.task.finished = False

        reset = asyncio.create_task(env._cb_reset_episode(arena_runtime_msgs.srv.ResetEpisode.Request(seed=9), arena_runtime_msgs.srv.ResetEpisode.Response()))
        await asyncio.sleep(0.2)
        assert not reset.done()
        fake_task_layer.gate.set()
        assert (await asyncio.wait_for(reset, TIMEOUT)).success
        await asyncio.wait_for(first, TIMEOUT)

        await _until(lambda: env._episodes.current.episode_id == 2 and env._episodes.current.outcome_state == RunEpisode.Result.RUNNING, "episode 2 running")
        assert env._episodes.current.seed == 9

    asyncio.run(_on_running_env(body))


def test_a_reset_episode_arriving_after_the_outcome_is_set_respawns_through_the_request(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv
    from arena_runtime_msgs.action import RunEpisode

    async def body(env: object, rig: object) -> None:
        first = asyncio.create_task(env._run_episode())
        await _until(lambda: env._episodes.current.outcome_state == RunEpisode.Result.RUNNING, "episode 1 running")
        env._episodes.pending_outcomes[1].set_result((RunEpisode.Result.SUCCESS, "done"))
        response = await env._cb_reset_episode(arena_runtime_msgs.srv.ResetEpisode.Request(seed=9), arena_runtime_msgs.srv.ResetEpisode.Response())
        assert response.success
        assert env._episodes.respawn_requested
        result = await asyncio.wait_for(first, TIMEOUT)
        assert (result.state, result.info) == (RunEpisode.Result.SUCCESS, "done")

        await _until(lambda: env._episodes.current.episode_id == 2 and env._episodes.current.outcome_state == RunEpisode.Result.RUNNING, "episode 2 running")
        assert env._episodes.current.seed == 9
        assert not env._episodes.respawn_requested

    asyncio.run(_on_running_env(body))
