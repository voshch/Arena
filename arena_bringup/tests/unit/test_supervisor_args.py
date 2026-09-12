from __future__ import annotations

import pytest

_SKIP_REASON = "arena_bringup.supervisor needs rclpy (source ROS)"


def _supervisor_available() -> bool:
    try:
        from arena_bringup import supervisor  # noqa: F401
    except ImportError:
        return False
    return True


_skip = pytest.mark.skipif(not _supervisor_available(), reason=_SKIP_REASON)


@_skip
def test_steering_defaults_to_auto() -> None:
    from arena_bringup.supervisor import parse_args

    assert parse_args(["sim:=gazebo"]).steering == "auto"


@_skip
def test_steering_parses_and_normalizes() -> None:
    from arena_bringup.supervisor import parse_args

    assert parse_args(["human.steering:=true"]).steering == "true"
    assert parse_args(["human.steering:=1"]).steering == "true"
    assert parse_args(["human.steering:=false"]).steering == "false"
    assert parse_args(["human.steering:=0"]).steering == "false"
    assert parse_args(["human.steering:=auto"]).steering == "auto"


@_skip
def test_steering_is_not_forwarded() -> None:
    from arena_bringup.supervisor import parse_args

    args = parse_args(["human.steering:=true", "sim:=gazebo"])
    assert "human.steering:=true" not in args.runtime_args
    assert "human.steering:=true" not in args.env_args
    assert "sim:=gazebo" in args.runtime_args


@_skip
def test_steering_rejects_unknown_value() -> None:
    from arena_bringup.supervisor import parse_args

    with pytest.raises(SystemExit):
        parse_args(["human.steering:=yes"])
