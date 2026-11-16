from __future__ import annotations

EXPECTED = {
    "task.episode.count",
    "task.episode.auto_reset",
    "task.episode.fail_on_collision",
    "task.episode.timeout",
    "task.episode.timeout.robot_ready",
    "task.episode.reset.max_fails",
    "task.episode.spawn.robot_clearance",
    "task.episode.spawn.obstacle_clearance",
    "task.episode.spawn.obstacle_max_radius",
    "task.episode.goto_pose.tolerance.radius",
    "task.episode.goto_pose.tolerance.angle",
    "task.episode.goto_pose.timeout.no_progress",
    "task.episode.goto_pose.hold_time",
    "task.episode.goto_pose.signal",
    "task.episode.node_ping.period",
    "task.episode.node_ping.misses",
}


def _groups() -> tuple[type, ...]:
    from arena_runtime.env.constants.runtime import ArenaParams, EpisodeParams, GeneralParams, HumanParams, ObstaclesParams, RobotParams

    return ArenaParams, GeneralParams, ObstaclesParams, RobotParams, HumanParams, EpisodeParams


def test_episode_params_lists_the_task_episode_parameters() -> None:
    from arena_runtime.env.constants.runtime import EPISODE_PARAMS

    assert set(EPISODE_PARAMS) == EXPECTED
    assert all(param.description for param in EPISODE_PARAMS.values())


def test_episode_params_are_the_declared_task_episode_parameters() -> None:
    from arena_runtime.env.constants.runtime import EPISODE_PARAMS

    declared = {name: param for group in _groups() for name, param in group.by_name().items() if name.startswith("task.episode.")}
    assert declared == EPISODE_PARAMS


def test_no_parameter_name_is_declared_by_two_groups() -> None:
    names = [name for group in _groups() for name in group.by_name()]
    assert len(names) == len(set(names))


def test_world_is_required_and_the_queue_depth_is_bounded() -> None:
    from arena_runtime.env.constants.runtime import ArenaParams, EpisodeParams

    assert ArenaParams.WORLD.required
    assert EpisodeParams.QUEUE_DEPTH.constraint is not None
