"""The census flags undescribed or raw declarations and finds none in the Arena tree."""

from __future__ import annotations

import pathlib

import pytest

from arena_rclpy_mixins.census import offenders


def _arena_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "_meta").is_dir():
            return parent
    raise RuntimeError("the Arena tree (a parent holding _meta/) was not found above this test")


def test_every_parameter_and_launch_argument_is_declared_and_described() -> None:
    found = offenders(_arena_root())
    assert not found, f"{len(found)} declarations outside params.py or without a description:\n" + "\n".join(found)


@pytest.mark.parametrize(
    ("source", "what"),
    [
        ('Param[int]("x", 1, "")', "Param without a description"),
        ('Param[int]("x", 1, "   ")', "Param without a description"),
        ('Param[int]("x", 1, description="")', "Param without a description"),
        ('LaunchArgument("x", default_value="1", description="")', "LaunchArgument without a description"),
        ('DeclareLaunchArgument("x", default_value="1")', "DeclareLaunchArgument without a description"),
        ('node.declare_parameter("x", 1)', "declare_parameter() outside params.py"),
        ('node.rosparam[str].get("x", "")', "rosparam.get(name, default) declares without a description"),
        ("ParameterDescriptor(description='x')", "hand-built ParameterDescriptor, declare a Param"),
    ],
)
def test_undescribed_or_raw_declarations_are_flagged(tmp_path: pathlib.Path, source: str, what: str) -> None:
    (tmp_path / "module.py").write_text(source + "\n")
    assert offenders(tmp_path) == [f"module.py:1 {what}"]


def test_described_declarations_pass(tmp_path: pathlib.Path) -> None:
    (tmp_path / "module.py").write_text('Param[int]("x", 1, "Count.")\nLaunchArgument("x", default_value="1", description="Count.")\n')
    assert offenders(tmp_path) == []
