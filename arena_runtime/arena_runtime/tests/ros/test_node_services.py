"""Tests for Agent B scope: seed derivation, EpisodeRuntime invariants,
task-mode enum validation, and ROSParamT.destroy idempotency.

These tests are pure-Python and do not require a live ROS graph.
"""

from __future__ import annotations

import inspect
import types

import pytest
from arena_runtime.env.task_layer import TaskModeNames

_LAYER = types.SimpleNamespace(mode_vocabulary=lambda: TaskModeNames(robots=["random", "scenario"], obstacles=["random", "scenario"], modules=["sounds"]))

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
# _cb_queue_episode enum validation (pure-Python path)
# ---------------------------------------------------------------------------


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
# _cb_queue_episode: per-field merge contract
# (empty tm_robots/tm_obstacles preserves whatever's already staged in
# pending_overrides; only non-empty fields overwrite.)
# ---------------------------------------------------------------------------


def _invoke_queue_episode(request: object) -> tuple[object, object]:
    import asyncio

    import task_generator_msgs.srv
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = type("Stub", (), {})()
    stub._task_layer = _LAYER
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}
    stub._staged_human_params = {}

    def _noop_publish_queue_state() -> None:
        pass

    stub._publish_queue_state = _noop_publish_queue_state

    response = task_generator_msgs.srv.QueueEpisode.Response()
    asyncio.run(EpisodeNode._cb_queue_episode(stub, request, response))
    return stub, response


def _human_params_request(name: str) -> object:
    import task_generator_msgs.srv
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

    request = task_generator_msgs.srv.QueueEpisode.Request()
    request.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    request.keep_modules = True
    request.human_params = [Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.1))]
    return request


def test_queue_episode_stages_namespaced_human_param() -> None:
    stub, response = _invoke_queue_episode(_human_params_request("humansim.global_planner.resolution"))
    assert response.success
    assert stub._staged_human_params["humansim.global_planner.resolution"].double_value == 0.1


@pytest.mark.parametrize("name", ["human.speed", "humansm.global_planner.resolution", "resolution", "humansim"])
def test_queue_episode_rejects_human_param_outside_known_namespaces(name: str) -> None:
    stub, response = _invoke_queue_episode(_human_params_request(name))
    assert not response.success
    assert name in response.error_msg
    assert stub._staged_human_params == {}
    assert stub._episodes.pending_overrides is None


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("tm_robots", "__bad__", "tm_robots: unknown value '__bad__'. Allowed: random, scenario"),
        ("tm_obstacles", "__bad__", "tm_obstacles: unknown value '__bad__'. Allowed: random, scenario"),
    ],
)
def test_queue_episode_refuses_a_mode_outside_the_vocabulary(field: str, value: str, error: str) -> None:
    import task_generator_msgs.srv

    request = task_generator_msgs.srv.QueueEpisode.Request()
    request.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    request.keep_modules = True
    setattr(request, field, value)

    stub, response = _invoke_queue_episode(request)
    assert not response.success
    assert response.error_msg == error
    assert stub._episodes.pending_overrides is None


def test_queue_episode_refuses_a_module_outside_the_vocabulary() -> None:
    import task_generator_msgs.srv

    request = task_generator_msgs.srv.QueueEpisode.Request()
    request.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    request.keep_modules = False
    request.tm_modules = ["sounds", "__bad__"]

    stub, response = _invoke_queue_episode(request)
    assert not response.success
    assert response.error_msg == "tm_modules: unknown value '__bad__'. Allowed: sounds"
    assert stub._episodes.pending_overrides is None


def test_queue_episode_combined_call_populates_both_modes() -> None:
    import task_generator_msgs.srv

    request = task_generator_msgs.srv.QueueEpisode.Request()
    request.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    request.tm_robots = "random"
    request.tm_obstacles = "random"
    request.keep_modules = True

    stub, response = _invoke_queue_episode(request)
    assert response.success
    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.tm_robots == "random"
    assert stub._episodes.pending_overrides.tm_obstacles == "random"


def test_queue_episode_empty_field_preserves_prior_stage() -> None:
    import asyncio

    import task_generator_msgs.srv
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = type("Stub", (), {})()
    stub._task_layer = _LAYER
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}

    def _noop() -> None:
        pass

    stub._publish_queue_state = _noop

    first = task_generator_msgs.srv.QueueEpisode.Request()
    first.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    first.tm_obstacles = "random"
    first.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.tm_robots = "random"
    second.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.tm_robots == "random"
    assert stub._episodes.pending_overrides.tm_obstacles == "random"


def test_queue_episode_non_empty_field_overwrites() -> None:
    import asyncio

    import task_generator_msgs.srv
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = type("Stub", (), {})()
    stub._task_layer = _LAYER
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}

    def _noop() -> None:
        pass

    stub._publish_queue_state = _noop

    first = task_generator_msgs.srv.QueueEpisode.Request()
    first.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    first.tm_obstacles = "random"
    first.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.tm_obstacles = "scenario"
    second.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.tm_obstacles == "scenario"


# ---------------------------------------------------------------------------
# _cb_queue_episode: world field merge
# ---------------------------------------------------------------------------


def test_queue_episode_world_empty_keeps_prior() -> None:
    import asyncio

    import task_generator_msgs.srv
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = type("Stub", (), {})()
    stub._task_layer = _LAYER
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}

    def _noop() -> None:
        pass

    stub._publish_queue_state = _noop

    first = task_generator_msgs.srv.QueueEpisode.Request()
    first.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    first.world = "map_a"
    first.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.world = ""
    second.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.world == "map_a"


def test_queue_episode_world_non_empty_overwrites() -> None:
    import asyncio

    import task_generator_msgs.srv
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = type("Stub", (), {})()
    stub._task_layer = _LAYER
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}

    def _noop() -> None:
        pass

    stub._publish_queue_state = _noop

    first = task_generator_msgs.srv.QueueEpisode.Request()
    first.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    first.world = "map_a"
    first.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.world = "map_b"
    second.keep_modules = True
    asyncio.run(EpisodeNode._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.world == "map_b"


# ---------------------------------------------------------------------------
# _cb_queue_episode: action != MERGE rejected
# ---------------------------------------------------------------------------


def test_queue_episode_unknown_action_rejected() -> None:
    import asyncio

    import task_generator_msgs.srv
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = type("Stub", (), {})()
    stub._task_layer = _LAYER
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}

    request = task_generator_msgs.srv.QueueEpisode.Request()
    request.action = 99
    request.keep_modules = True

    response = task_generator_msgs.srv.QueueEpisode.Response()
    asyncio.run(EpisodeNode._cb_queue_episode(stub, request, response))

    assert not response.success
    assert response.error_msg != ""
    assert stub._episodes.pending_overrides is None


# ---------------------------------------------------------------------------
# declare_safe forces descriptor.type when value is type-ambiguous (e.g. []).
# ---------------------------------------------------------------------------


def test_declare_safe_forces_string_array_for_empty_list() -> None:
    import rclpy
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from rcl_interfaces.msg import ParameterDescriptor

    node = ROSParamServer("declare_safe_test_node")
    try:
        node.rosparam.declare_safe(
            "models",
            [],
            descriptor=ParameterDescriptor(
                type=rclpy.Parameter.Type.STRING_ARRAY.value,
                additional_constraints="catalog:objects",
                description="",
            ),
        )
        param = node.get_parameter("models")
        assert param.type_ == rclpy.Parameter.Type.STRING_ARRAY
        assert list(param.value) == []
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# Curriculum _split_task_mode_params: 3-part `task.<mode>.<leaf>` keys route
# to obstacles_params with the leaf stripped of the "task." prefix only.
# ---------------------------------------------------------------------------


def test_curriculum_split_routes_task_keys_to_obstacles() -> None:
    pytest.importorskip("rosnav_rl.utils.curriculum.curriculum_base")
    from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase

    stub = type("Stub", (), {})()
    stub.verbose = 0
    stub._param_to_rcl_param = CurriculumBase._param_to_rcl_param.__get__(stub)

    obstacles, robots, plain = CurriculumBase._split_task_mode_params(
        stub,
        {
            "task.random.static.n": [1, 5],
            "task.scenario.file": "default.yaml",
            "lr": 1e-4,
        },
    )

    assert {p.name for p in obstacles} == {"static.n", "file"}
    assert robots == []
    assert plain == {"lr": 1e-4}


def test_curriculum_split_skips_empty_lists() -> None:
    pytest.importorskip("rosnav_rl.utils.curriculum.curriculum_base")
    from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase

    stub = type("Stub", (), {})()
    stub.verbose = 0
    stub._param_to_rcl_param = CurriculumBase._param_to_rcl_param.__get__(stub)

    obstacles, robots, plain = CurriculumBase._split_task_mode_params(
        stub,
        {"task.random.static.models": []},
    )

    assert obstacles == []
    assert robots == []
    assert plain == {}


# ---------------------------------------------------------------------------
# _cb_reset_episode resolves an in-flight episode with SKIPPED rather than
# letting the termination watcher close it as SUCCESS.
# ---------------------------------------------------------------------------


def test_reset_episode_in_flight_resolves_skipped() -> None:
    import asyncio

    import task_generator_msgs.action
    import task_generator_msgs.srv
    from arena_runtime.env.episode_node import EpisodeRecord, EpisodeRuntime, EpisodeNode

    async def run() -> None:
        stub = object.__new__(EpisodeNode)
        stub._force_reset = False
        stub._episodes = EpisodeRuntime(
            current=EpisodeRecord(episode_id=7),
        )
        stub._episodes.action_in_flight = True
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        stub._episodes.pending_outcomes[7] = fut
        stub._task = object()

        request = task_generator_msgs.srv.ResetEpisode.Request()
        request.world = "map_empty"
        request.seed = -1
        response = task_generator_msgs.srv.ResetEpisode.Response()

        await EpisodeNode._cb_reset_episode(stub, request, response)

        assert response.success
        assert stub._episodes.pending_world == "map_empty"
        assert stub._episodes.pending_seed == -1
        assert fut.done()
        state, reason = fut.result()
        assert state == task_generator_msgs.action.RunEpisode.Result.SKIPPED
        assert reason == "reset"
        assert stub._force_reset is False

    asyncio.run(run())


def test_a_tearing_down_node_refuses_new_episodes() -> None:
    import asyncio
    import threading

    import task_generator_msgs.action
    from rclpy.action import GoalResponse
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = object.__new__(EpisodeNode)
    stub._episodes = EpisodeRuntime()
    stub._heartbeat_stop = threading.Event()
    stub._tick_loop_task = stub._check_status_task = stub._episode_task = None
    stub._task = stub._robots_manager = stub._simulator = None
    asyncio.run(stub.teardown())

    assert EpisodeNode._goal_callback(stub, None) == GoalResponse.REJECT
    result = asyncio.run(EpisodeNode._run_episode(stub))
    assert (result.state, result.info) == (task_generator_msgs.action.RunEpisode.Result.SKIPPED, "shutting down")
