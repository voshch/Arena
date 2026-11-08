from __future__ import annotations

import importlib.util
import os
import sys

import pytest


@pytest.fixture(scope="module")
def env_launch():
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")
    from ament_index_python.packages import get_package_share_directory

    spec = importlib.util.spec_from_file_location("env_launch", os.path.join(get_package_share_directory("arena_runtime"), "launch", "env.launch.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(env_launch, cmd: list[str], *, shutdown_after: float | None = None) -> int:
    import launch
    import launch.actions
    from launch.event_handlers import OnProcessExit

    node = launch.actions.ExecuteProcess(cmd=cmd, sigterm_timeout="2", sigkill_timeout="2")
    entities: list[launch.LaunchDescriptionEntity] = [
        node,
        launch.actions.RegisterEventHandler(OnProcessExit(target_action=node, on_exit=env_launch._on_env_node_exit)),
    ]
    if shutdown_after is not None:
        entities.append(launch.actions.TimerAction(period=shutdown_after, actions=[launch.actions.Shutdown(reason="test")]))
    service = launch.LaunchService()
    service.include_launch_description(launch.LaunchDescription(entities))
    return service.run()


def test_env_launch_fails_when_the_env_node_exits_non_zero(env_launch) -> None:
    assert _run(env_launch, [sys.executable, "-c", "raise SystemExit(3)"]) == 1


def test_env_launch_succeeds_when_the_env_node_exits_cleanly(env_launch) -> None:
    assert _run(env_launch, [sys.executable, "-c", "raise SystemExit(0)"]) == 0


def test_env_launch_succeeds_when_the_env_node_dies_during_shutdown(env_launch) -> None:
    assert _run(env_launch, ["sleep", "30"], shutdown_after=1.0) == 0
