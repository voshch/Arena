from __future__ import annotations

import os

import pytest
import yaml


@pytest.fixture(scope="module", autouse=True)
def _require_ament() -> None:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")


def _env_node_parameters(**configs: str) -> dict[str, object]:
    import launch
    from ament_index_python.packages import get_package_share_directory
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import Node

    ctx = launch.LaunchContext()
    ctx.launch_configurations.update({"env.managed": "true", "sim": "dummy", "env.id": "7", "env.ns": "/arena/env_7/arena_env_node", **configs})
    path = os.path.join(get_package_share_directory("arena_runtime"), "launch", "env.launch.py")
    params: dict[str, object] = {}

    def walk(entity) -> None:
        if isinstance(entity, Node):
            if entity.node_executable == "arena_env_node":
                entity._perform_substitutions(ctx)
                for argument, is_file in entity._Node__expanded_parameter_arguments:
                    if is_file:
                        with open(argument) as f:
                            for section in yaml.load(f, Loader=yaml.FullLoader).values():
                                params.update(section["ros__parameters"])
        elif isinstance(entity, launch.actions.ExecuteProcess):
            return
        elif isinstance(entity, launch.LaunchDescription):
            for child in entity.entities:
                walk(child)
        elif isinstance(entity, launch.Action):
            for child in entity.execute(ctx) or []:
                walk(child)

    walk(PythonLaunchDescriptionSource(path).get_launch_description(ctx))
    return params


def test_optim_tokens_reach_the_env_node_as_its_param() -> None:
    assert _env_node_parameters(optim="no_camera")["optim"] == "no_camera"


def test_optim_defaults_to_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_OPTIM", "no_lidar")
    assert _env_node_parameters()["optim"] == "no_lidar"
