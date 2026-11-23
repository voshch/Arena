"""Tests for `arena robot paths` and the `robot:=` entry completion."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

pytest.importorskip("arena_robots.override_paths")

from arena_cli import cli, complete, robot
from arena_cli.complete import Context, _expand

ARENA = Path(__file__).resolve().parents[3]
MANIFEST = {
    "values": {"robot": ["husky", "jackal"]},
    "launch": {"arena_runtime/env.launch.py": {"robot": {"desc": "robots of the env", "default": ""}}},
}


def _rows(capsys: pytest.CaptureFixture[str]) -> list[list[str]]:
    return [line.split("\t") for line in capsys.readouterr().out.splitlines()]


def _launch_lines(cur: str) -> list[str]:
    return complete.render(_expand(cli.ENV_ARGS, [], cur, Context("", MANIFEST)), cur)


def _launch_candidates(cur: str) -> list[str]:
    return [line for line in _launch_lines(cur) if not line.startswith("@")]


def test_paths_verb_prints_a_row_per_path_with_the_effective_value(capsys: pytest.CaptureFixture[str]) -> None:
    authored = yaml.safe_load((ARENA / "arena_robots" / "arena_robots" / "robots" / "jackal" / "caps.yaml").read_text())["mobile"]
    assert robot._main(["paths", "jackal"]) == 0
    rows = _rows(capsys)
    assert ["mobile.velocity_limits.linear", "list", json.dumps(authored["velocity_limits"]["linear"])] in rows
    assert any(row[1] == "unset" and row[2] == "-" for row in rows)


def test_paths_verb_applies_the_entry_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    assert robot._main(["paths", "jackal[mobile.velocity_limits.linear=[-0.5,1.0]]"]) == 0
    assert ["mobile.velocity_limits.linear", "list", "[-0.5, 1.0]"] in _rows(capsys)


def test_paths_verb_names_the_brackets_of_a_placed_part(capsys: pytest.CaptureFixture[str], require_installed: Callable[..., None]) -> None:
    require_installed("arm/ur")
    assert robot._main(["paths", "jackal[top=arm/ur5e]"]) == 0
    assert ["arm.top.stall_time", "unset", "-", "brackets of top: stall_time"] in _rows(capsys)


def test_paths_verb_lists_the_models_for_an_unknown_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert robot._main(["paths", "no_such_robot"]) == 1
    err = capsys.readouterr().err
    assert "models:" in err and "jackal" in err


def test_paths_verb_states_the_entry_form_for_an_invalid_override(capsys: pytest.CaptureFixture[str]) -> None:
    assert robot._main(["paths", "jackal[mobile.velocty_limits.linear=[0,1]]"]) == 1
    err = capsys.readouterr().err
    assert "unknown field 'velocty_limits'" in err and "arena robot paths --help" in err


def test_bare_paths_verb_lists_the_models_and_an_example(capsys: pytest.CaptureFixture[str]) -> None:
    assert robot._main(["paths"]) == 2
    err = capsys.readouterr().err
    assert "models:" in err and "jackal" in err and "example:" in err


def test_paths_help_explains_the_rows(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        robot._main(["paths", "--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "unset" in out and "brackets of" in out


def test_launch_robot_value_completes_override_paths_without_a_space() -> None:
    lines = _launch_lines("robot:=jackal[mobile.vel")
    assert "@nospace" in lines
    assert _launch_candidates("robot:=jackal[mobile.vel") == ["jackal[mobile.velocity_limits."]


def test_launch_robot_value_completes_inside_part_brackets(require_installed: Callable[..., None]) -> None:
    require_installed("arm/ur")
    assert "jackal[top=arm/ur5e[stall_time=" in _launch_candidates("robot:=jackal[top=arm/ur5e[stall_t")


def test_launch_robot_value_keeps_a_leading_quote() -> None:
    assert _launch_candidates("robot:='jackal[mobile.rad") == ["'jackal[mobile.radius="]


def test_launch_robot_value_without_a_bracket_completes_models() -> None:
    assert _launch_candidates("robot:=jac") == ["jackal"]


def test_launch_robot_value_of_an_unknown_model_completes_nothing() -> None:
    assert _launch_candidates("robot:=no_such_robot[mobile.") == []


def test_paths_verb_completes_the_entry() -> None:
    result = _expand(robot.SPEC, ["paths"], "jackal[mobile.rad", Context("", MANIFEST))
    assert [value for value, _desc, _group in result.items] == ["jackal[mobile.radius="]
