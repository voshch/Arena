from __future__ import annotations

import os

import pytest
import yaml


@pytest.fixture(scope="module", autouse=True)
def _require_ament() -> None:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")


def _launch_path(*parts: str) -> str:
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory("task_generator"), "launch", *parts)


def _declared(path: str) -> set[str]:
    import launch
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    ctx = launch.LaunchContext()
    return {entity.name for entity in PythonLaunchDescriptionSource(path).get_launch_description(ctx).entities if isinstance(entity, launch.actions.DeclareLaunchArgument)}


def _arena_nodes() -> dict:
    import launch
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import Node

    ctx = launch.LaunchContext()
    ctx.launch_configurations.update({"namespace": "env7/task_generator_node", "environment_namespace": "env7"})
    nodes: dict = {}

    def walk(entity) -> None:
        if isinstance(entity, Node):
            entity._perform_substitutions(ctx)
            nodes[entity.node_name.rsplit("/", 1)[-1]] = entity
        elif isinstance(entity, launch.LaunchDescription):
            for child in entity.entities:
                walk(child)
        elif isinstance(entity, launch.Action):
            for child in entity.execute(ctx) or []:
                walk(child)

    walk(PythonLaunchDescriptionSource(_launch_path("acoustics", "arena", "arena.launch.py")).get_launch_description(ctx))
    return nodes


def _parameters(node) -> dict:
    params: dict = {}
    for argument, is_file in node._Node__expanded_parameter_arguments:
        if is_file:
            with open(argument) as f:
                for section in yaml.load(f, Loader=yaml.FullLoader).values():
                    params.update(section["ros__parameters"])
    return params


@pytest.mark.usefixtures("requires_auditory")
def test_pedestrian_hearing_defaults_off() -> None:
    from arena_auditory.params import all_params

    assert "pedestrian_listeners.enabled" not in _parameters(_arena_nodes()["sound_propagation_node"])
    assert all_params()["pedestrian_listeners.enabled"].default is False
    assert "auditory.pedestrian_listeners.enabled" not in _declared(_launch_path("task_generator.launch.py"))
