"""Tests for Agent B scope: seed derivation, EpisodeRuntime invariants,
task-mode enum validation, and ROSParamT.destroy idempotency.

These tests are pure-Python and do not require a live ROS graph.
"""

from __future__ import annotations

import inspect

import pytest

# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def _derive_seed(run_seed: str, world: str, episode_id: int) -> int:
    from arena_runtime.env.episode_node import _derive_seed as derive

    return derive(run_seed, world, episode_id)


def test_seed_derivation_stable() -> None:
    s1 = _derive_seed("abc", "map_empty", 1)
    s2 = _derive_seed("abc", "map_empty", 1)
    assert s1 == s2


def test_seed_derivation_sensitive_to_run_seed() -> None:
    assert _derive_seed("aaa", "world", 1) != _derive_seed("bbb", "world", 1)


def test_seed_derivation_sensitive_to_world() -> None:
    assert _derive_seed("seed", "world_a", 1) != _derive_seed("seed", "world_b", 1)


def test_seed_derivation_sensitive_to_episode_id() -> None:
    assert _derive_seed("seed", "world", 1) != _derive_seed("seed", "world", 2)


def test_seed_derivation_nonnegative() -> None:
    v = _derive_seed("x", "y", 3)
    assert v >= 0


# ---------------------------------------------------------------------------
# ROSParamT.destroy idempotency (structural, no live node needed)
# ---------------------------------------------------------------------------


def test_roparam_destroy_is_abstract_method() -> None:
    from arena_rclpy_mixins.ROSParamServer import ROSParamT

    assert "destroy" in {m for m in dir(ROSParamT) if not m.startswith("__")}
    assert inspect.isfunction(ROSParamT.destroy)


def test_ros_param_impl_destroy_callable() -> None:
    from arena_rclpy_mixins.ROSParamServer import _ROSParam

    assert callable(_ROSParam.destroy)


# ---------------------------------------------------------------------------
# _cb_queue_episode: per-axis merge contract
# (an axis the request leaves out keeps whatever is already pending; a listed
# axis replaces its selection.)
# ---------------------------------------------------------------------------


def _queue_stub() -> object:
    import asyncio

    from arena_runtime.env.episode_node import EpisodeNode, EpisodeRuntime
    from arena_runtime.testing.env_host import FakeTaskLayer

    layer = FakeTaskLayer()
    stub = type("Stub", (), {})()
    stub._task_layer = layer
    stub.task_params = layer.params
    stub._episodes = EpisodeRuntime()
    stub._staged_params = {}
    stub._staged_human_params = {}
    stub._pending_lock = asyncio.Lock()
    stub._mode_namespaces = EpisodeNode._mode_namespaces.__get__(stub)
    stub._publish_queue_state = lambda: None
    return stub


def _request(**fields: object) -> object:
    import arena_runtime_msgs.srv

    request = arena_runtime_msgs.srv.QueueEpisode.Request(action=arena_runtime_msgs.srv.QueueEpisode.Request.MERGE)
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def _modes(**axes: str) -> list:
    from diagnostic_msgs.msg import KeyValue

    return [KeyValue(key=axis, value=value) for axis, value in axes.items()]


def _queue(stub: object, request: object) -> object:
    import asyncio

    import arena_runtime_msgs.srv
    from arena_runtime.env.episode_node import EpisodeNode

    response = arena_runtime_msgs.srv.QueueEpisode.Response()
    asyncio.run(EpisodeNode._cb_queue_episode(stub, request, response))
    return response


def _invoke_queue_episode(request: object) -> tuple[object, object]:
    stub = _queue_stub()
    return stub, _queue(stub, request)


def _param(name: str, value: int) -> object:
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

    return Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value))


def _staged(stub: object) -> dict[str, int]:
    return {name: value.integer_value for name, value in stub._staged_params.items()}


def _human_params_request(name: str) -> object:
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

    return _request(human_params=[Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.1))])


def test_queue_episode_stages_namespaced_human_param() -> None:
    stub, response = _invoke_queue_episode(_human_params_request("humansim.global_planner.resolution"))
    assert response.success
    assert stub._staged_human_params["humansim.global_planner.resolution"].double_value == 0.1


@pytest.mark.parametrize("name", ["human.speed", "humansm.global_planner.resolution", "resolution", "humansim"])
def test_queue_episode_rejects_human_param_outside_known_namespaces(name: str) -> None:
    from arena_runtime.env.task_layer import TaskModeOverrides

    stub, response = _invoke_queue_episode(_human_params_request(name))
    assert not response.success
    assert name in response.error_msg
    assert stub._staged_human_params == {}
    assert stub._episodes.pending == TaskModeOverrides()


@pytest.mark.parametrize(
    ("axes", "error"),
    [
        ({"robots": "__bad__"}, "unknown robots mode '__bad__', allowed: fake_robots, fake_other"),
        ({"obstacles": "__bad__"}, "unknown obstacles mode '__bad__', allowed: fake_obstacles"),
        ({"modules": "fake_module,__bad__"}, "unknown modules mode '__bad__', allowed: fake_module"),
        ({"terrain": "flat"}, "unknown task axis 'terrain', allowed: robots, obstacles, modules"),
        ({"robots": ""}, "task axis 'robots' takes exactly one mode, got [], allowed: fake_robots, fake_other"),
        ({"robots": "fake_robots,fake_other"}, "task axis 'robots' takes exactly one mode, got ['fake_robots', 'fake_other'], allowed: fake_robots, fake_other"),
    ],
)
def test_queue_episode_refuses_what_the_layer_validator_refuses_with_its_message(axes: dict, error: str) -> None:
    from arena_runtime.env.task_layer import TaskModeOverrides, decode_modes

    request = _request(modes=_modes(**axes))
    stub = _queue_stub()
    with pytest.raises(ValueError) as raised:
        stub._task_layer.validate_modes(decode_modes(request.modes))
    response = _queue(stub, request)
    assert not response.success
    assert response.error_msg == error == str(raised.value)
    assert stub._episodes.pending == TaskModeOverrides()


def test_queue_episode_combined_call_populates_both_modes() -> None:
    stub, response = _invoke_queue_episode(_request(modes=_modes(robots="fake_other", obstacles="fake_obstacles")))
    assert response.success
    assert stub._episodes.pending.modes == {"robots": ["fake_other"], "obstacles": ["fake_obstacles"]}


def test_queue_episode_modes_replace_an_axis_and_an_empty_value_clears_a_multi_mode_axis() -> None:
    stub, response = _invoke_queue_episode(_request(modes=_modes(robots="fake_other", modules="")))
    assert response.success
    assert stub._episodes.pending.modes == {"robots": ["fake_other"], "modules": []}


def test_queue_episode_without_modes_stages_no_mode() -> None:
    stub, response = _invoke_queue_episode(_request())
    assert response.success
    assert stub._episodes.pending.modes == {}


def test_queue_episode_empty_field_preserves_prior_stage() -> None:
    stub = _queue_stub()
    _queue(stub, _request(modes=_modes(obstacles="fake_obstacles")))
    _queue(stub, _request(modes=_modes(robots="fake_other")))
    assert stub._episodes.pending.modes == {"robots": ["fake_other"], "obstacles": ["fake_obstacles"]}


def test_queue_episode_non_empty_field_overwrites() -> None:
    stub = _queue_stub()
    _queue(stub, _request(modes=_modes(robots="fake_robots")))
    _queue(stub, _request(modes=_modes(robots="fake_other")))
    assert stub._episodes.pending.modes == {"robots": ["fake_other"]}


# ---------------------------------------------------------------------------
# _cb_queue_episode: world field merge
# ---------------------------------------------------------------------------


def test_queue_episode_world_empty_keeps_prior() -> None:
    stub = _queue_stub()
    _queue(stub, _request(world="map_a"))
    _queue(stub, _request(world=""))
    assert stub._episodes.pending.world == "map_a"


def test_queue_episode_world_non_empty_overwrites() -> None:
    stub = _queue_stub()
    _queue(stub, _request(world="map_a"))
    _queue(stub, _request(world="map_b"))
    assert stub._episodes.pending.world == "map_b"


# ---------------------------------------------------------------------------
# _cb_queue_episode: staged params by full name
# ---------------------------------------------------------------------------


def test_queue_episode_full_name_params_upsert() -> None:
    stub = _queue_stub()
    _queue(stub, _request(params=[_param("task.fake_robots.level", 1), _param("task.fake_module.level", 2)]))
    _queue(stub, _request(params=[_param("task.fake_robots.level", 5)]))
    assert _staged(stub) == {"task.fake_robots.level": 5, "task.fake_module.level": 2}


def test_queue_episode_a_mode_switch_drops_the_staged_names_of_the_old_mode() -> None:
    stub = _queue_stub()
    _queue(stub, _request(params=[_param("task.fake_robots.level", 3), _param("task.fake_module.level", 2)]))
    _queue(stub, _request(modes=_modes(robots="fake_other")))
    assert _staged(stub) == {"task.fake_module.level": 2}


def test_queue_episode_a_mode_switch_keeps_the_params_the_same_request_stages_for_the_new_mode() -> None:
    stub = _queue_stub()
    _queue(stub, _request(params=[_param("task.fake_robots.level", 3)]))
    _queue(stub, _request(modes=_modes(robots="fake_other"), params=[_param("task.fake_other.level", 7)]))
    assert _staged(stub) == {"task.fake_other.level": 7}


def test_queue_episode_a_mode_switch_keeps_names_a_mode_of_another_axis_still_uses() -> None:
    stub = _queue_stub()
    stub._task_layer.NAMESPACES = {**stub._task_layer.NAMESPACES, "obstacles": {"fake_obstacles": "task.fake_robots"}}
    _queue(stub, _request(params=[_param("task.fake_robots.level", 3)]))
    _queue(stub, _request(modes=_modes(robots="fake_other")))
    assert _staged(stub) == {"task.fake_robots.level": 3}


def test_queue_episode_resending_the_same_mode_keeps_the_staged_names() -> None:
    stub = _queue_stub()
    _queue(stub, _request(modes=_modes(robots="fake_robots"), params=[_param("task.fake_robots.level", 3)]))
    _queue(stub, _request(modes=_modes(robots="fake_robots")))
    assert _staged(stub) == {"task.fake_robots.level": 3}


# ---------------------------------------------------------------------------
# _cb_queue_episode: action != MERGE rejected
# ---------------------------------------------------------------------------


def test_queue_episode_unknown_action_rejected() -> None:
    from arena_runtime.env.task_layer import TaskModeOverrides

    stub = _queue_stub()
    response = _queue(stub, _request(action=99))

    assert not response.success
    assert response.error_msg != ""
    assert stub._episodes.pending == TaskModeOverrides()


# ---------------------------------------------------------------------------
# declare forces the STRING_ARRAY type for an empty list default.
# ---------------------------------------------------------------------------


def test_declare_forces_string_array_for_empty_list() -> None:
    import rclpy
    from arena_rclpy_mixins.params import Catalog, Param, declare
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    node = ROSParamServer("declare_test_node")
    try:
        declare(node, Param[list[str]]("models", [], "Object models of the test.", type_=rclpy.Parameter.Type.STRING_ARRAY, constraint=Catalog("objects")))
        param = node.get_parameter("models")
        assert param.type_ == rclpy.Parameter.Type.STRING_ARRAY
        assert list(param.value) == []
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# Curriculum _split_task_params: 3-part `task.<mode>.<leaf>` keys route to the
# mode params under the namespace the env reports for <mode>.
# ---------------------------------------------------------------------------


def _curriculum_stub():
    pytest.importorskip("rosnav_rl.utils.curriculum.curriculum_base")
    from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase

    stub = type("Stub", (), {})()
    stub.verbose = 0
    stub._param_to_rcl_param = CurriculumBase._param_to_rcl_param.__get__(stub)
    return CurriculumBase, stub


def test_curriculum_split_prefixes_task_keys_with_the_env_namespace_of_the_mode() -> None:
    curriculum, stub = _curriculum_stub()

    task, plain = curriculum._split_task_params(
        stub,
        {
            "task.random.static.n": [1, 5],
            "task.scenario.file": "default.yaml",
            "lr": 1e-4,
        },
        {"random": "task.random", "scenario": "task.scn"},
    )

    assert {p.name for p in task} == {"task.random.static.n", "task.scn.file"}
    assert plain == {"lr": 1e-4}


def test_curriculum_split_skips_empty_lists() -> None:
    curriculum, stub = _curriculum_stub()

    task, plain = curriculum._split_task_params(stub, {"task.random.static.models": []}, {"random": "task.random"})

    assert task == []
    assert plain == {}


# ---------------------------------------------------------------------------
# _cb_reset_episode resolves an in-flight episode with SKIPPED rather than
# letting the termination watcher close it as SUCCESS.
# ---------------------------------------------------------------------------


def test_reset_episode_in_flight_resolves_skipped() -> None:
    import asyncio

    import arena_runtime_msgs.action
    import arena_runtime_msgs.srv
    from arena_runtime.env.episode_node import EpisodeNode, EpisodeRuntime
    from arena_runtime.env.task_layer import EpisodeRecord

    async def run() -> None:
        stub = object.__new__(EpisodeNode)
        stub._force_reset = False
        stub._episodes = EpisodeRuntime(
            current=EpisodeRecord(episode_id=7),
        )
        stub._episodes.action_in_flight = True
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        stub._episodes.pending_outcomes[7] = fut
        stub._pending_lock = asyncio.Lock()
        stub._task = object()

        request = arena_runtime_msgs.srv.ResetEpisode.Request()
        request.world = "map_empty"
        request.seed = -1
        response = arena_runtime_msgs.srv.ResetEpisode.Response()

        await EpisodeNode._cb_reset_episode(stub, request, response)

        assert response.success
        assert stub._episodes.pending.world == "map_empty"
        assert stub._episodes.pending.seed == -1
        assert fut.done()
        state, reason = fut.result()
        assert state == arena_runtime_msgs.action.RunEpisode.Result.SKIPPED
        assert reason == "reset"
        assert stub._force_reset is False

    asyncio.run(run())


def test_a_tearing_down_node_refuses_new_episodes() -> None:
    import asyncio
    import threading

    import arena_runtime_msgs.action
    from rclpy.action import GoalResponse
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = object.__new__(EpisodeNode)
    stub._episodes = EpisodeRuntime()
    stub._heartbeat_stop = threading.Event()
    stub._tick_loop_task = stub._check_status_task = stub._node_watch_task = stub._episode_task = None
    stub._task = stub._robots_manager = stub._simulator = None
    asyncio.run(stub.teardown())

    assert EpisodeNode._goal_callback(stub, None) == GoalResponse.REJECT
    result = asyncio.run(EpisodeNode._run_episode(stub))
    assert (result.state, result.info) == (arena_runtime_msgs.action.RunEpisode.Result.SKIPPED, "shutting down")
