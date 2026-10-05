from __future__ import annotations

import pytest

_SKIP_REASON = "arena_bringup.substitutions needs launch (source ROS)"


def _substitutions_available() -> bool:
    try:
        from arena_bringup import substitutions  # noqa: F401
    except ImportError:
        return False
    return True


_skip = pytest.mark.skipif(not _substitutions_available(), reason=_SKIP_REASON)


def _resolve(configs: dict[str, str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    from arena_bringup.substitutions import resolve_deprecated_args

    warned: list[tuple[str, str]] = []
    resolve_deprecated_args(configs, lambda old, new: warned.append((old, new)))
    return configs, warned


@_skip
@pytest.mark.parametrize("old", ["task.scenario", "scenario_file"])
def test_scenario_aliases_land_on_scenario_file(old: str) -> None:
    configs, warned = _resolve({old: "corridor"})

    assert configs == {"task.scenario.file": "corridor"}
    assert warned == [(old, "task.scenario.file")]


@_skip
def test_scenario_alias_leaves_sibling_scenario_keys_alone() -> None:
    configs, warned = _resolve({"task.scenario": "corridor", "task.scenario.linger_after_completion": "true"})

    assert configs == {"task.scenario.linger_after_completion": "true", "task.scenario.file": "corridor"}
    assert warned == [("task.scenario", "task.scenario.file")]


@_skip
def test_explicit_scenario_file_wins_over_alias() -> None:
    configs, _ = _resolve({"task.scenario": "old", "task.scenario.file": "new"})

    assert configs == {"task.scenario.file": "new"}
