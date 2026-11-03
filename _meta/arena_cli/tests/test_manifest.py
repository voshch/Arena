"""Tests for the installed-only completion values in arena_cli.manifest."""

from pathlib import Path

from arena_cli import manifest

ROBOT_GITMODULES = """\
[submodule "arena_robots/robots/mpo700/meshes"]
\tpath = arena_robots/robots/mpo700/meshes
\turl = https://example.invalid/mpo700.git
\trobot = mpo700
[submodule "arena_robots/robots/jackal/meshes"]
\tpath = arena_robots/robots/jackal/meshes
\turl = https://example.invalid/jackal.git
\trobot = jackal
"""

PLANNER_GITMODULES = """\
[submodule "planners/drlvo"]
\tpath = planners/drlvo
\turl = https://example.invalid/drlvo.git
\tplanner = drlvo
[submodule "planners/navid"]
\tpath = planners/navid
\turl = https://example.invalid/navid.git
\tplanner = navid
"""


def _arena(tmp_path: Path) -> Path:
    for sdk, body in (("arena_robots", ROBOT_GITMODULES), ("arena_planners", PLANNER_GITMODULES)):
        (tmp_path / sdk).mkdir()
        (tmp_path / sdk / ".gitmodules").write_text(body)
    for path in ("arena_robots/arena_robots/robots/jackal/meshes", "arena_planners/planners/drlvo"):
        (tmp_path / path).mkdir(parents=True)
        (tmp_path / path / ".git").write_text("gitdir: elsewhere\n")
    return tmp_path


def test_uninstalled_robots_are_the_ones_not_checked_out(tmp_path):
    assert manifest._uninstalled_robots(_arena(tmp_path)) == {"mpo700"}


def test_installed_planners_are_only_the_checked_out_ones(tmp_path):
    assert manifest._installed_planners(_arena(tmp_path)) == ["drlvo"]


LAYER = """
def mode_namespaces():
    return {AXES}
"""


def test_task_layer_values_union_the_modes_of_every_registered_layer(tmp_path, monkeypatch):
    from arena_runtime.testing.task_layer import register_task_layer

    register_task_layer(tmp_path, "patrol_layer", LAYER.replace("{AXES}", repr({"robots": {"patrol": "patrol"}, "fleet": {"convoy": "convoy"}})))
    register_task_layer(tmp_path, "grazing_layer", LAYER.replace("{AXES}", repr({"robots": {"graze": "graze"}})))
    monkeypatch.syspath_prepend(str(tmp_path))
    values, deps = manifest._task_layers()
    assert {"patrol_layer", "grazing_layer"} <= set(values["task_layer"])
    assert {"patrol", "graze"} <= set(values["task.robots"])
    assert values["task.fleet"] == ["convoy"]
    assert str((tmp_path / "patrol_layer.py").resolve()) in deps


def test_task_layer_with_a_broken_entry_point_is_skipped_beside_a_working_one(tmp_path, monkeypatch, capsys):
    from arena_runtime.testing.task_layer import register_task_layer

    register_task_layer(tmp_path, "sound_layer", LAYER.replace("{AXES}", repr({"robots": {"sound": "sound"}})))
    register_task_layer(tmp_path, "broken_layer", "raise ImportError('missing dependency')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    values, _ = manifest._task_layers()
    assert "sound_layer" in values["task_layer"]
    assert "broken_layer" not in values["task_layer"]
    assert "sound" in values["task.robots"]
    err = capsys.readouterr().err
    assert "broken_layer" in err and "missing dependency" in err


def test_axis_launch_args_are_the_task_keys_of_the_manifest_values():
    args = manifest._task_axis_args({"sim": ["gazebo"], "task_layer": ["task_generator"], "task.robots": ["explore"], "task.fleet": ["convoy"]})
    assert set(args) == {"task.robots", "task.fleet"}
    assert args["task.fleet"]["desc"] == "fleet mode of the task layer"
