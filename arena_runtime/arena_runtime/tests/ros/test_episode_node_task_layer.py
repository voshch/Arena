"""EpisodeNode against a fake task layer: the env finds it through the entry point and hands it the env parts."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.msg")


async def _on_running_env(body: object) -> None:
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
        assert ctx.conf.RNG is env.conf.General.RNG
        assert ctx.conf.TIMEOUT is env.conf.Robot.TIMEOUT
        assert ctx.interactive.markers is env.markers
        assert ctx.interactive.robot_colors is env.robot_colors
        assert ctx.interactive.tf_buffer is env.tf_buffer
        ctx.episode.compromise("from a mode")
        assert "from a mode" in env._episodes.current.integrity
        ctx.episode.abort_episode("from a mode")
        assert env._abort_reason == "from a mode"

    asyncio.run(_on_running_env(body))


def test_setup_binds_the_collision_callback_of_the_robots_manager(rclpy_context: object, fake_task_layer: object) -> None:
    async def body(env: object) -> None:
        assert env._robots_manager._on_collision == env._on_collision

    asyncio.run(_on_running_env(body))


def test_query_task_modes_answers_from_the_layer(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv

    from arena_runtime.env.task_layer import decode_modes, decode_namespaces

    async def body(env: object) -> None:
        response = await env._cb_query_task_modes(arena_runtime_msgs.srv.QueryTaskModes.Request(), arena_runtime_msgs.srv.QueryTaskModes.Response())
        assert decode_modes(response.modes) == {"robots": ["fake_robots", "fake_other"], "obstacles": ["fake_obstacles"], "modules": ["fake_module"]}
        assert decode_namespaces(response.namespaces) == fake_task_layer.NAMESPACES

    asyncio.run(_on_running_env(body))


def test_get_task_modes_answers_the_live_selection_per_axis(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv

    from arena_runtime.env.task_layer import decode_modes

    async def body(env: object) -> None:
        response = await env._cb_get_task_modes(arena_runtime_msgs.srv.GetTaskModes.Request(), arena_runtime_msgs.srv.GetTaskModes.Response())
        assert decode_modes(response.modes) == {"robots": ["fake_robots"], "obstacles": ["fake_obstacles"], "modules": ["fake_module"]}

    asyncio.run(_on_running_env(body))


def test_the_task_is_ticked_only_while_an_episode_runs_with_the_phases_that_ended(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime_msgs.action import RunEpisode

    async def body(env: object) -> None:
        await asyncio.sleep(1.0)
        assert fake_task_layer.task.ticks == []

        env.trigger_activate()
        env._spawn_episode()
        for _ in range(200):
            if fake_task_layer.task.ticks:
                break
            await asyncio.sleep(0.1)
        assert fake_task_layer.task.ticks
        assert env._episodes.current.outcome_state == RunEpisode.Result.RUNNING
        assert all(ended == {} and assigned == set() for ended, assigned in fake_task_layer.task.ticks)

        fake_task_layer.task.finished = True
        for _ in range(200):
            if env._episodes.current.outcome_state != RunEpisode.Result.RUNNING:
                break
            await asyncio.sleep(0.1)
        settled = len(fake_task_layer.task.ticks)
        await asyncio.sleep(1.0)
        assert len(fake_task_layer.task.ticks) == settled

    asyncio.run(_on_running_env(body))


def test_the_task_is_not_ticked_while_the_reset_lock_is_held(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime_msgs.action import RunEpisode

    async def body(env: object) -> None:
        env.trigger_activate()
        env._spawn_episode()
        for _ in range(200):
            if env._episodes.current.outcome_state == RunEpisode.Result.RUNNING:
                break
            await asyncio.sleep(0.1)
        async with env._reset_lock:
            await asyncio.sleep(0.2)
            settled = len(fake_task_layer.task.ticks)
            await asyncio.sleep(1.0)
            assert len(fake_task_layer.task.ticks) == settled

    asyncio.run(_on_running_env(body))


def test_robots_assigned_from_outside_reach_the_next_tick_once_and_a_reset_clears_them(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime_msgs.action import RunEpisode

    async def body(env: object) -> None:
        env._assigned.add("ghost")
        env.trigger_activate()
        env._spawn_episode()
        for _ in range(200):
            if env._episodes.current.outcome_state == RunEpisode.Result.RUNNING:
                break
            await asyncio.sleep(0.1)
        assert env._assigned == set()
        await asyncio.sleep(0.5)
        assert all("ghost" not in assigned for _, assigned in fake_task_layer.task.ticks)

        env._assigned.add("ghost")
        for _ in range(200):
            if any("ghost" in assigned for _, assigned in fake_task_layer.task.ticks):
                break
            await asyncio.sleep(0.1)
        assert [assigned for _, assigned in fake_task_layer.task.ticks if "ghost" in assigned] == [{"ghost"}]
        assert env._assigned == set()

    asyncio.run(_on_running_env(body))


def test_the_node_reports_the_task_layer_it_resolved(rclpy_context: object, fake_task_layer: object) -> None:
    from arena_runtime.env.task_layer import DEFAULT_TASK_LAYER

    async def body(env: object) -> None:
        assert env.rosparam[str].get("task_layer", "") == DEFAULT_TASK_LAYER

    asyncio.run(_on_running_env(body))


def test_a_hand_spawned_static_obstacle_with_an_unknown_model_is_refused_and_compromises_nothing(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv

    async def body(env: object) -> None:
        request = arena_runtime_msgs.srv.SpawnStatic.Request(model="no_such_model", use_pose=False)
        response = await env._cb_spawn_static(request, arena_runtime_msgs.srv.SpawnStatic.Response())
        assert not response.success
        assert "no_such_model" in response.error_msg
        assert env._episodes.current.integrity == []

    asyncio.run(_on_running_env(body))


def test_a_hand_spawned_dynamic_obstacle_gets_a_pose_and_waypoints(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv

    from arena_simulation_setup.tree import local_root

    bundle = local_root() / "Common" / "Human" / "seam_walker"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "seam_walker.sdf").write_text("<sdf/>")

    async def body(env: object) -> None:
        request = arena_runtime_msgs.srv.SpawnDynamic.Request(model="seam_walker", use_pose=False)
        response = await env._cb_spawn_dynamic(request, arena_runtime_msgs.srv.SpawnDynamic.Response())
        assert response.success, response.error_msg
        assert env._episodes.current.integrity == ["dynamic seam_walker spawned by hand"]

    asyncio.run(_on_running_env(body))


def test_a_hand_spawned_robot_is_queued_for_the_next_reset(rclpy_context: object, fake_task_layer: object) -> None:
    import arena_runtime_msgs.srv

    async def body(env: object) -> None:
        request = arena_runtime_msgs.srv.SpawnRobot.Request(model="jackal", name="", use_pose=False, immediate=False)
        response = await env._cb_spawn_robot(request, arena_runtime_msgs.srv.SpawnRobot.Response())
        assert response.success, response.error_msg
        assert response.name in env._robots_manager._diff.to_add
        assert env._episodes.current.integrity == [f"robot {response.name} spawned by hand"]

    asyncio.run(_on_running_env(body))
