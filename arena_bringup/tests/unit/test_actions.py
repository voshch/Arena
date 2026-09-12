from __future__ import annotations

import pytest

_SKIP_REASON = "launch package not available (source ROS)"


def _launch_available() -> bool:
    try:
        import launch.actions  # noqa: F401
        import launch.launch_description_source  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


_skip = pytest.mark.skipif(not _launch_available(), reason=_SKIP_REASON)


def _resolve_args(ctx: object, include: object) -> dict[str, str]:
    from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions

    return {name: perform_substitutions(ctx, normalize_to_list_of_substitutions(value)) for name, value in include.launch_arguments}


@_skip
def test_forwards_parent_configurations() -> None:
    import launch
    import launch.launch_description_source as lds
    from arena_bringup.actions import IncludeLaunchDescriptionForward

    ctx = launch.LaunchContext()
    ctx.launch_configurations["sim"] = "gazebo"
    ctx.launch_configurations["world"] = "map_empty"

    action = IncludeLaunchDescriptionForward(lds.LaunchDescriptionSource())
    [include] = action.execute(ctx)

    assert _resolve_args(ctx, include) == {"sim": "gazebo", "world": "map_empty"}


@_skip
def test_overrides_win_over_forwarded() -> None:
    import launch
    import launch.launch_description_source as lds
    from arena_bringup.actions import IncludeLaunchDescriptionForward

    ctx = launch.LaunchContext()
    ctx.launch_configurations["env_n"] = "5"
    ctx.launch_configurations["sim"] = "gazebo"

    action = IncludeLaunchDescriptionForward(
        lds.LaunchDescriptionSource(),
        overrides={"env_n": "0"},
    )
    [include] = action.execute(ctx)

    args = _resolve_args(ctx, include)
    assert args["env_n"] == "0"
    assert args["sim"] == "gazebo"


@_skip
def test_overrides_added_when_absent_from_parent() -> None:
    import launch
    import launch.launch_description_source as lds
    from arena_bringup.actions import IncludeLaunchDescriptionForward

    ctx = launch.LaunchContext()

    action = IncludeLaunchDescriptionForward(
        lds.LaunchDescriptionSource(),
        overrides={"x": "1"},
    )
    [include] = action.execute(ctx)

    assert _resolve_args(ctx, include) == {"x": "1"}
