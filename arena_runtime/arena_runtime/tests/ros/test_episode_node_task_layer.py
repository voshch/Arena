"""EpisodeNode against a fake task layer: the env finds it through the entry point and hands it the env parts."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("task_generator_msgs.msg")


async def _on_running_env(body: object) -> None:
    import rclpy.parameter

    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.testing.env_host import arena_on_dummy

    async with arena_on_dummy() as rig:
        overrides = [
            rclpy.parameter.Parameter("env_id", value=rig.env_id),
            rclpy.parameter.Parameter("world", value="map_empty"),
            rclpy.parameter.Parameter("sim", value="dummy"),
            rclpy.parameter.Parameter("human", value="dummy"),
            rclpy.parameter.Parameter("acoustics", value="none"),
        ]
        env = EpisodeNode(namespace=f"/{rig.namespace}", parameter_overrides=overrides)
        rig.executor.add_node(env)
        try:
            await asyncio.wait_for(env.setup(), 60.0)
            await body(env)
            await env.teardown()
        finally:
            rig.executor.remove_node(env)
            env.destroy_node()


def test_setup_hands_the_layer_the_env_parts(rclpy_context: object, fake_task_layer: object) -> None:
    async def body(env: object) -> None:
        ctx = fake_task_layer.ctx
        assert ctx.environment_manager is env._environment_manager
        assert ctx.robots_manager is env._robots_manager
        assert ctx.world_manager is env._world_manager
        assert ctx.realizer is env._realizer
        assert ctx.simulator is env._simulator
        assert ctx.acoustics_simulator is env._acoustics_simulator
        assert ctx.reset_lock is env._reset_lock
        ctx.abort_episode("from a mode")
        assert env._abort_reason == "from a mode"

    asyncio.run(_on_running_env(body))


def test_setup_binds_the_collision_callback_of_the_robots_manager(rclpy_context: object, fake_task_layer: object) -> None:
    async def body(env: object) -> None:
        assert env._robots_manager._on_collision == env._on_collision

    asyncio.run(_on_running_env(body))


def test_query_task_modes_answers_from_the_layer(rclpy_context: object, fake_task_layer: object) -> None:
    import task_generator_msgs.srv

    async def body(env: object) -> None:
        response = await env._cb_query_task_modes(task_generator_msgs.srv.QueryTaskModes.Request(), task_generator_msgs.srv.QueryTaskModes.Response())
        assert (response.robots, response.obstacles, response.modules) == (["fake_robots"], ["fake_obstacles"], [])

    asyncio.run(_on_running_env(body))


def test_a_hand_spawned_static_obstacle_with_an_unknown_model_is_refused_and_compromises_nothing(rclpy_context: object, fake_task_layer: object) -> None:
    import task_generator_msgs.srv

    async def body(env: object) -> None:
        request = task_generator_msgs.srv.SpawnStatic.Request(model="no_such_model", use_pose=False)
        response = await env._cb_spawn_static(request, task_generator_msgs.srv.SpawnStatic.Response())
        assert not response.success
        assert "no_such_model" in response.error_msg
        assert env._episodes.current.integrity == []

    asyncio.run(_on_running_env(body))


def test_a_hand_spawned_dynamic_obstacle_gets_a_pose_and_waypoints(rclpy_context: object, fake_task_layer: object) -> None:
    import task_generator_msgs.srv

    async def body(env: object) -> None:
        request = task_generator_msgs.srv.SpawnDynamic.Request(model="arenian", use_pose=False)
        response = await env._cb_spawn_dynamic(request, task_generator_msgs.srv.SpawnDynamic.Response())
        assert response.success, response.error_msg
        assert env._episodes.current.integrity == ["dynamic arenian spawned by hand"]

    asyncio.run(_on_running_env(body))


def test_a_hand_spawned_robot_is_queued_for_the_next_reset(rclpy_context: object, fake_task_layer: object) -> None:
    import task_generator_msgs.srv

    async def body(env: object) -> None:
        request = task_generator_msgs.srv.SpawnRobot.Request(model="jackal", name="", use_pose=False, immediate=False)
        response = await env._cb_spawn_robot(request, task_generator_msgs.srv.SpawnRobot.Response())
        assert response.success, response.error_msg
        assert response.name in env._robots_manager._diff.to_add
        assert env._episodes.current.integrity == [f"robot {response.name} spawned by hand"]

    asyncio.run(_on_running_env(body))
