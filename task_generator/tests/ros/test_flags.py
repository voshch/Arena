from __future__ import annotations

from types import SimpleNamespace


def _ctx(configs: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(launch_configurations=dict(configs))


def _coerce(value: str) -> object:
    return int(value) if value.lstrip("-").isdigit() else value


def test_truthy_falsy():
    from task_generator.utils.flags import truthy

    for value in ("", " ", "0", "false", "FALSE", " no ", "off", 0, False, None):
        assert truthy(value) is False


def test_truthy_enabled():
    from task_generator.utils.flags import truthy

    for value in ("1", "true", "yes", "on", "no_obstacles", 1, True, 2):
        assert truthy(value) is True


def test_expand_shorthand_is_one():
    from task_generator.utils.flags import expand_flag_namespace

    assert expand_flag_namespace(_ctx({"optim": "a,b"}), "optim", _coerce) == {
        "optim.a": 1,
        "optim.b": 1,
    }


def test_expand_strips_blank_tokens():
    from task_generator.utils.flags import expand_flag_namespace

    assert expand_flag_namespace(_ctx({"optim": " a , ,b "}), "optim", _coerce) == {
        "optim.a": 1,
        "optim.b": 1,
    }


def test_expand_explicit_is_coerced():
    from task_generator.utils.flags import expand_flag_namespace

    assert expand_flag_namespace(_ctx({"optim.lod": "5"}), "optim", _coerce) == {"optim.lod": 5}


def test_expand_explicit_wins_over_shorthand():
    from task_generator.utils.flags import expand_flag_namespace

    result = expand_flag_namespace(
        _ctx({"optim": "no_obstacles", "optim.no_obstacles": "false"}),
        "optim",
        _coerce,
    )
    assert result == {"optim.no_obstacles": "false"}


def test_expand_ignores_other_namespaces():
    from task_generator.utils.flags import expand_flag_namespace

    assert expand_flag_namespace(_ctx({"task.foo": "1", "debug.x": "1"}), "optim", _coerce) == {}


def test_expand_absent_is_empty():
    from task_generator.utils.flags import expand_flag_namespace

    assert expand_flag_namespace(_ctx({}), "optim", _coerce) == {}


def _node(params: dict[str, object]) -> SimpleNamespace:
    from rclpy.parameter import Parameter

    by_prefix = {k: Parameter(k, value=v) for k, v in params.items()}
    return SimpleNamespace(get_parameters_by_prefix=lambda _ns: by_prefix)


def test_obstacles_optim_coerce():
    from task_generator.utils.flags import ObstaclesOptim

    cases = {
        # ints (shorthand `optim:=obstacles` lands as 1) and clamping
        0: ObstaclesOptim.FULL,
        1: ObstaclesOptim.BBOX,
        2: ObstaclesOptim.NONE,
        5: ObstaclesOptim.NONE,
        -3: ObstaclesOptim.FULL,
        # alias names, case-insensitive, plus numeric strings
        "full": ObstaclesOptim.FULL,
        "bbox": ObstaclesOptim.BBOX,
        "BBOX": ObstaclesOptim.BBOX,
        "none": ObstaclesOptim.NONE,
        "no_obstacles": ObstaclesOptim.NONE,
        "2": ObstaclesOptim.NONE,
        "": ObstaclesOptim.FULL,
        "garbage": ObstaclesOptim.FULL,
    }
    for value, expected in cases.items():
        assert ObstaclesOptim.coerce(value) is expected


def test_obstacles_optim_level_reads_param():
    from task_generator.utils.flags import ObstaclesOptim, obstacles_optim_level

    assert obstacles_optim_level(_node({})) is ObstaclesOptim.FULL
    assert obstacles_optim_level(_node({"obstacles": "bbox"})) is ObstaclesOptim.BBOX
    assert obstacles_optim_level(_node({"obstacles": 1})) is ObstaclesOptim.BBOX
    # legacy optim.no_obstacles still maps to NONE when obstacles is unset
    assert obstacles_optim_level(_node({"no_obstacles": "1"})) is ObstaclesOptim.NONE
    # explicit obstacles wins over the legacy flag
    assert obstacles_optim_level(_node({"obstacles": "full", "no_obstacles": "1"})) is ObstaclesOptim.FULL
