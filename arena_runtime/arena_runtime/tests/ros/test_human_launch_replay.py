from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")


def test_human_launch_takes_replay_and_starts_nothing(rclpy_context) -> None:
    import os

    import launch
    from ament_index_python.packages import get_package_share_directory
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import Node

    path = os.path.join(get_package_share_directory("arena_runtime"), "launch", "human", "human.launch.py")

    def started(simulator: str) -> tuple[list[str], list[type]]:
        ctx = launch.LaunchContext()
        ctx.launch_configurations.update({"simulator": simulator, "namespace": "env7/arena_env_node"})
        declared: list[str] = []
        actions: list[type] = []

        def walk(entity) -> None:
            if isinstance(entity, launch.actions.DeclareLaunchArgument):
                declared.append(entity.name)
            if isinstance(entity, Node | launch.actions.IncludeLaunchDescription):
                actions.append(type(entity))
            elif isinstance(entity, launch.LaunchDescription):
                for child in entity.entities:
                    walk(child)
            elif isinstance(entity, launch.Action):
                for child in entity.execute(ctx) or []:
                    walk(child)

        walk(PythonLaunchDescriptionSource(path).get_launch_description(ctx))
        return declared, actions

    declared, actions = started("replay")
    assert sorted(declared) == ["namespace", "simulator"]
    assert actions == []
    assert started("arena")[1] == [launch.actions.IncludeLaunchDescription]
    with pytest.raises(RuntimeError, match="replay"):
        started("recorded")
