from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arena_runtime.env.task_layer import (
    DEFAULT_TASK_LAYER,
    TASK_LAYER_ENV,
    TASK_LAYER_GROUP,
    TaskModeOverrides,
    declare_mode_schemas,
    legacy_mode_params,
    mode_param,
    registered_modes,
    selection_args,
    task_layer,
    task_layer_entry_point,
    task_layer_name,
    validate_modes,
)
from arena_runtime.testing.task_layer import register_task_layer


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.delenv(TASK_LAYER_ENV, raising=False)
    return tmp_path


def test_the_default_entry_point_loads_with_no_name(root: Path) -> None:
    register_task_layer(root, DEFAULT_TASK_LAYER, "MARK = 1\n")
    sys.modules.pop(DEFAULT_TASK_LAYER, None)
    assert task_layer().MARK == 1
    sys.modules.pop(DEFAULT_TASK_LAYER, None)


def test_no_registered_entry_point_names_the_group(root: Path) -> None:
    with pytest.raises(LookupError, match=f"{TASK_LAYER_GROUP}.*none"):
        task_layer_entry_point()


def test_a_name_selects_among_many_and_the_default_stays_the_default(root: Path) -> None:
    register_task_layer(root, DEFAULT_TASK_LAYER, "")
    register_task_layer(root, "second_runtime", "")
    assert task_layer_entry_point().name == DEFAULT_TASK_LAYER
    assert task_layer_entry_point("second_runtime").name == "second_runtime"


def test_the_environment_selects_when_no_name_is_given(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    register_task_layer(root, DEFAULT_TASK_LAYER, "")
    register_task_layer(root, "second_runtime", "")
    monkeypatch.setenv(TASK_LAYER_ENV, "second_runtime")
    assert task_layer_entry_point().name == "second_runtime"
    assert task_layer_entry_point(DEFAULT_TASK_LAYER).name == DEFAULT_TASK_LAYER


def test_an_unknown_name_lists_what_is_registered(root: Path) -> None:
    register_task_layer(root, "first", "")
    register_task_layer(root, "second", "")
    with pytest.raises(LookupError, match="nope.*first.*second"):
        task_layer_entry_point("nope")


def test_the_runtime_name_is_the_argument_then_the_environment_then_the_default(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert task_layer_name() == DEFAULT_TASK_LAYER
    monkeypatch.setenv(TASK_LAYER_ENV, "second_runtime")
    assert task_layer_name() == "second_runtime"
    assert task_layer_name("third_runtime") == "third_runtime"


class _Layer:
    def __init__(self) -> None:
        self.declared: list[tuple[str, str]] = []

    def mode_namespaces(self) -> dict[str, dict[str, str]]:
        return {"patrol": {"loop": "runtime.loop", "sweep": "runtime.sweep"}, "extras": {"trace": "runtime.trace"}}

    def declare_mode_schema(self, node: object, axis: str, mode: str) -> None:
        self.declared.append((axis, mode))


def test_registered_modes_are_the_names_of_the_layer_namespaces_per_axis() -> None:
    assert registered_modes(_Layer()) == {"patrol": ["loop", "sweep"], "extras": ["trace"]}


def test_every_registered_mode_gets_its_schema_declared() -> None:
    layer = _Layer()
    declare_mode_schemas(layer, object())
    assert layer.declared == [("patrol", "loop"), ("patrol", "sweep"), ("extras", "trace")]


def test_modes_round_trip_the_wire_encoding_with_an_empty_axis() -> None:
    pytest.importorskip("diagnostic_msgs.msg")
    from arena_runtime.env.task_layer import decode_modes, encode_modes

    modes = {"patrol": ["loop", "sweep"], "extras": []}
    pairs = encode_modes(modes)
    assert [(pair.key, pair.value) for pair in pairs] == [("patrol", "loop,sweep"), ("extras", "")]
    assert decode_modes(pairs) == modes


VOCABULARY = {"patrol": ["loop", "sweep"], "extras": ["trace"]}


def test_the_validator_canonicalizes_dedupes_and_leaves_unlisted_axes_out() -> None:
    assert validate_modes({}, VOCABULARY, ("extras",)) == {}
    assert validate_modes({"patrol": ["Loop"], "extras": ["TRACE", "trace"]}, VOCABULARY, ("extras",)) == {"patrol": ["loop"], "extras": ["trace"]}
    assert validate_modes({"extras": []}, VOCABULARY, ("extras",)) == {"extras": []}


def test_the_validator_names_the_first_axis_or_mode_that_would_not_run() -> None:
    with pytest.raises(ValueError, match="unknown task axis 'planner'.*patrol, extras"):
        validate_modes({"planner": ["dwb"]}, VOCABULARY, ("extras",))
    with pytest.raises(ValueError, match="unknown patrol mode 'nope'.*loop, sweep"):
        validate_modes({"patrol": ["nope"]}, VOCABULARY, ("extras",))
    with pytest.raises(ValueError, match="takes exactly one mode.*\\['loop', 'sweep'\\]"):
        validate_modes({"patrol": ["loop", "sweep"]}, VOCABULARY, ("extras",))


def test_the_selection_spelling_comes_from_the_axis_names() -> None:
    assert mode_param("robots") == "task.robots"
    assert selection_args({"robots": ["random"], "modules": []}) == ["task.robots:=random"]
    assert legacy_mode_params() == {"tm_robots": "task.robots", "tm_obstacles": "task.obstacles", "tm_modules": "task.modules"}


def test_the_deprecated_launch_args_still_spell_the_mode_params_the_seam_names() -> None:
    from arena_bringup.substitutions import DEPRECATED_LAUNCH_ARGS

    assert {name: DEPRECATED_LAUNCH_ARGS[name] for name in legacy_mode_params()} == legacy_mode_params()
    assert DEPRECATED_LAUNCH_ARGS["task_config"] == mode_param("config")


def test_overrides_merge_replaces_listed_axes_and_keeps_the_rest() -> None:
    queued = TaskModeOverrides(modes={"robots": ["a"], "modules": ["m"]}, world="w", seed=3)
    queued.merge(TaskModeOverrides(modes={"robots": ["b"]}))
    assert queued == TaskModeOverrides(modes={"robots": ["b"], "modules": ["m"]}, world="w", seed=3)
    queued.merge(TaskModeOverrides(modes={"modules": []}, world="x", seed=0))
    assert queued == TaskModeOverrides(modes={"robots": ["b"], "modules": []}, world="x", seed=0)


def test_the_fake_layer_names_its_multi_axes_and_validates_by_them() -> None:
    from arena_runtime.testing.env_host import FakeTaskLayer

    layer = FakeTaskLayer()
    two_modules = list(layer.NAMESPACES["modules"])[:2]

    assert set(layer.multi_axes()) == {"modules"}
    assert layer.validate_modes({"modules": two_modules}) == {"modules": two_modules}
    with pytest.raises(ValueError, match="exactly one mode"):
        layer.validate_modes({"robots": list(layer.NAMESPACES["robots"])[:2]})
