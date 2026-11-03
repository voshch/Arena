"""Per-node log levels a robot launch hands to its nav2 nodes."""

from __future__ import annotations

import pytest

ROBOT_NS = "/arena/env_0/arena_env_node/jackal"


def _level(spec: str, name: str) -> str | None:
    import launch
    from arena_bringup.extensions.NodeLogLevelExtension import NodeLogLevelExtension
    from launch_ros.actions import Node

    from arena_runtime.env.robot_manager.robot_manager import log_level_actions

    context = launch.LaunchContext()
    for action in log_level_actions(spec):
        action.execute(context)
    node = Node(package="nav2_planner", executable=name, name=name, namespace=ROBOT_NS)
    node._perform_substitutions(context)
    args, _ = NodeLogLevelExtension().prepare_for_execute(context, {}, node)
    return args[1][0].text if args else None


SUITE_SPEC = "{**/planner_server:info, **/lifecycle_manager_navigation:info, warn}"


@pytest.mark.parametrize(
    ("spec", "name", "level"),
    [
        (SUITE_SPEC, "planner_server", "info"),
        (SUITE_SPEC, "lifecycle_manager_navigation", "info"),
        (SUITE_SPEC, "bt_navigator", "error"),
        (SUITE_SPEC, "goal_to_plan_bridge", "warn"),
        ("warn", "planner_server", "error"),
        ("info", "goal_to_plan_bridge", "info"),
        ("{**/nav2*/**:fatal, debug}", "planner_server", "error"),
    ],
)
def test_robot_launch_level(spec: str, name: str, level: str) -> None:
    assert _level(spec, name) == level
