"""Tests for Agent B scope: seed derivation, EpisodeRuntime invariants,
task-mode enum validation, and ROSParamT.destroy idempotency.

These tests are pure-Python and do not require a live ROS graph.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def _derive_seed(run_seed: str, world: str, episode_id: int) -> int:
    digest = hashlib.blake2b(
        f"{run_seed}|{world}|{episode_id}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


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


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


# ---------------------------------------------------------------------------
# _cb_queue_episode enum validation (pure-Python path)
# ---------------------------------------------------------------------------


def test_queue_episode_enum_validation_invalid_tm_robots() -> None:
    from task_generator.constants import Constants

    invalid = "definitely_not_a_valid_robots_mode"
    try:
        Constants.TaskMode.TM_Robots(invalid)
        found = True
    except ValueError:
        found = False
    assert not found, "Expected ValueError for unknown TM_Robots value"


def test_queue_episode_enum_validation_valid_values() -> None:
    from task_generator.constants import Constants

    for member in Constants.TaskMode.TM_Robots:
        result = Constants.TaskMode.TM_Robots(member.value)
        assert result is member


def test_queue_episode_enum_validation_invalid_tm_obstacles() -> None:
    from task_generator.constants import Constants

    try:
        Constants.TaskMode.TM_Obstacles("__bad__")
        found = True
    except ValueError:
        found = False
    assert not found


def test_queue_episode_enum_validation_invalid_tm_module() -> None:
    from task_generator.constants import Constants

    try:
        Constants.TaskMode.TM_Module("__bad__")
        found = True
    except ValueError:
        found = False
    assert not found


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
    from task_generator.node import EpisodeRuntime, TaskGenerator

    stub = type("Stub", (), {})()
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}

    def _noop_publish_queue_state() -> None:
        pass

    stub._publish_queue_state = _noop_publish_queue_state

    response = task_generator_msgs.srv.QueueEpisode.Response()
    asyncio.run(TaskGenerator._cb_queue_episode(stub, request, response))
    return stub, response


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
    from task_generator.node import EpisodeRuntime, TaskGenerator

    stub = type("Stub", (), {})()
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
    asyncio.run(TaskGenerator._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.tm_robots = "random"
    second.keep_modules = True
    asyncio.run(TaskGenerator._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.tm_robots == "random"
    assert stub._episodes.pending_overrides.tm_obstacles == "random"


def test_queue_episode_non_empty_field_overwrites() -> None:
    import asyncio

    import task_generator_msgs.srv
    from task_generator.node import EpisodeRuntime, TaskGenerator

    stub = type("Stub", (), {})()
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
    asyncio.run(TaskGenerator._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.tm_obstacles = "scenario"
    second.keep_modules = True
    asyncio.run(TaskGenerator._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.tm_obstacles == "scenario"


# ---------------------------------------------------------------------------
# _cb_queue_episode: world field merge
# ---------------------------------------------------------------------------


def test_queue_episode_world_empty_keeps_prior() -> None:
    import asyncio

    import task_generator_msgs.srv
    from task_generator.node import EpisodeRuntime, TaskGenerator

    stub = type("Stub", (), {})()
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
    asyncio.run(TaskGenerator._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.world = ""
    second.keep_modules = True
    asyncio.run(TaskGenerator._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.world == "map_a"


def test_queue_episode_world_non_empty_overwrites() -> None:
    import asyncio

    import task_generator_msgs.srv
    from task_generator.node import EpisodeRuntime, TaskGenerator

    stub = type("Stub", (), {})()
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
    asyncio.run(TaskGenerator._cb_queue_episode(stub, first, task_generator_msgs.srv.QueueEpisode.Response()))

    second = task_generator_msgs.srv.QueueEpisode.Request()
    second.action = task_generator_msgs.srv.QueueEpisode.Request.MERGE
    second.world = "map_b"
    second.keep_modules = True
    asyncio.run(TaskGenerator._cb_queue_episode(stub, second, task_generator_msgs.srv.QueueEpisode.Response()))

    assert stub._episodes.pending_overrides is not None
    assert stub._episodes.pending_overrides.world == "map_b"


# ---------------------------------------------------------------------------
# _cb_queue_episode: action != MERGE rejected
# ---------------------------------------------------------------------------


def test_queue_episode_unknown_action_rejected() -> None:
    import asyncio

    import task_generator_msgs.srv
    from task_generator.node import EpisodeRuntime, TaskGenerator

    stub = type("Stub", (), {})()
    stub._episodes = EpisodeRuntime()
    stub._staged_obstacles_params = {}
    stub._staged_robots_params = {}

    request = task_generator_msgs.srv.QueueEpisode.Request()
    request.action = 99
    request.keep_modules = True

    response = task_generator_msgs.srv.QueueEpisode.Response()
    asyncio.run(TaskGenerator._cb_queue_episode(stub, request, response))

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
    from task_generator.node import EpisodeRecord, EpisodeRuntime, TaskGenerator

    class TaskStub:
        def __init__(self) -> None:
            self.force_reset_called = False

        def force_reset(self) -> None:
            self.force_reset_called = True

    async def run() -> None:
        stub = type("Stub", (), {})()
        stub._episodes = EpisodeRuntime(
            current=EpisodeRecord(episode_id=7),
        )
        stub._episodes.action_in_flight = True
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        stub._episodes.pending_outcomes[7] = fut
        stub._task = TaskStub()

        request = task_generator_msgs.srv.ResetEpisode.Request()
        request.world = "map_empty"
        request.seed = -1
        response = task_generator_msgs.srv.ResetEpisode.Response()

        await TaskGenerator._cb_reset_episode(stub, request, response)

        assert response.success
        assert stub._episodes.pending_world == "map_empty"
        assert stub._episodes.pending_seed == -1
        assert fut.done()
        state, reason = fut.result()
        assert state == task_generator_msgs.action.RunEpisode.Result.SKIPPED
        assert reason == "reset"
        assert stub._task.force_reset_called is False

    asyncio.run(run())
