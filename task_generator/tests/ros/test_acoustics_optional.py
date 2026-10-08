from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_DRIVER = r"""
import json
import os
import sys

import launch
import launch.actions
import yaml
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

ctx = launch.LaunchContext()
ctx.launch_configurations.update(json.loads(sys.argv[1]))
path = os.path.join(get_package_share_directory("task_generator"), "launch", "task_generator.launch.py")
env_actions = []
for entity in PythonLaunchDescriptionSource(path).get_launch_description(ctx).entities:
    if isinstance(entity, launch.actions.DeclareLaunchArgument):
        entity.execute(ctx)
    elif isinstance(entity, launch.actions.OpaqueFunction):
        env_actions = entity.execute(ctx) or []

found = {"includes": {}, "params": {}}
expanded = ("acoustics.launch.py", "hearing.launch.py", "arena.launch.py", "arena_auditory.launch.py")

def perform(value):
    return launch.utilities.perform_substitutions(ctx, launch.utilities.normalize_to_list_of_substitutions(value))

def walk(entity):
    if isinstance(entity, Node):
        if entity.node_package in ("arena_auditory", "task_generator"):
            entity._perform_substitutions(ctx)
            params = {}
            for argument, is_file in entity._Node__expanded_parameter_arguments:
                if is_file:
                    with open(argument) as f:
                        for section in yaml.safe_load(f).values():
                            params.update(section["ros__parameters"])
            found["params"][entity.node_name.rsplit("/", 1)[-1]] = params
    elif isinstance(entity, launch.actions.IncludeLaunchDescription):
        entity.launch_description_source.get_launch_description(ctx)
        location = entity.launch_description_source.location
        name = os.path.basename(location)
        found["includes"]["/".join(location.split(os.sep)[-3:])] = sorted(perform(k) for k, _ in entity.launch_arguments)
        if name in expanded:
            for child in entity.execute(ctx):
                walk(child)
    elif isinstance(entity, launch.LaunchDescription):
        for child in entity.entities:
            walk(child)
    elif isinstance(entity, launch.Action) and not isinstance(entity, (launch.actions.ExecuteProcess, launch.actions.RegisterEventHandler)):
        for child in entity.execute(ctx) or []:
            walk(child)

for action in env_actions:
    walk(action)
print(json.dumps(found))
"""

_BASE_ARGS = {"env.managed": "true", "sim": "dummy", "env.id": "7", "env.ns": "env7/task_generator_node"}
_STRIPPED = ("arena_auditory", "arena_hearing")
_ACOUSTICS = "launch/acoustics/acoustics.launch.py"
_AUDITORY = "arena_auditory/launch/arena_auditory.launch.py"
_HEARING = "launch/hearing/hearing.launch.py"
_ARENA_HEARING = "arena_hearing/launch/hearing.launch.py"


def _env(*, with_auditory: bool) -> dict[str, str]:
    if with_auditory:
        return dict(os.environ)
    prefixes = [p for p in os.environ["AMENT_PREFIX_PATH"].split(os.pathsep) if os.path.basename(p) not in _STRIPPED]
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if not any(package in p for package in _STRIPPED)]
    return {**os.environ, "AMENT_PREFIX_PATH": os.pathsep.join(prefixes), "PYTHONPATH": os.pathsep.join(paths)}


def _launch(args: dict[str, str], *, with_auditory: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", _DRIVER, json.dumps({**_BASE_ARGS, **args})], env=_env(with_auditory=with_auditory), capture_output=True, text=True, timeout=120, check=False)


def _found(proc: subprocess.CompletedProcess[str]) -> dict:
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _auditory_params(found: dict) -> dict:
    return {name: params for name, params in found["params"].items() if name != "task_generator_node"}


@pytest.fixture(scope="module", autouse=True)
def _require_ament() -> None:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")


def test_task_generator_imports_without_arena_auditory(tmp_path) -> None:
    modules = ("task_generator.simulators.human", "task_generator.simulators.acoustics.arena", "task_generator.simulators.hearing.arena", "task_generator.tasks.modules.sounds.impl")
    code = f"import importlib, importlib.util\nfor name in {modules!r}: importlib.import_module(name)\nassert importlib.util.find_spec('arena_auditory') is None\nassert importlib.util.find_spec('arena_hearing') is None"
    proc = subprocess.run([sys.executable, "-c", code], env=_env(with_auditory=False), cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr


def test_acoustics_none_launches_without_arena_auditory() -> None:
    found = _found(_launch({"acoustics": "none"}, with_auditory=False))
    assert _ACOUSTICS not in found["includes"]
    assert _HEARING not in found["includes"]
    assert found["params"]["task_generator_node"]["acoustics"] == "none"
    assert "robot.mobile.params_overlay" not in found["params"]["task_generator_node"]


@pytest.mark.parametrize(
    "args",
    [
        {"acoustics": "none", "auditory.static_sounds": "[{name: radio, asset_id: radio_loop, position: {x: 1.0, y: 2.0}}]"},
        {"acoustics": "none", "task.modules": "sounds"},
    ],
)
def test_sounds_module_runs_without_arena_auditory(args: dict[str, str]) -> None:
    found = _found(_launch(args, with_auditory=False))
    assert "sounds" in found["params"]["task_generator_node"]["task.modules"].split(",")
    assert _ACOUSTICS not in found["includes"]


def test_acoustics_arena_without_arena_auditory_names_the_feature() -> None:
    proc = _launch({"acoustics": "arena"}, with_auditory=False)
    assert proc.returncode != 0
    assert "arena feature auditory install" in proc.stderr


def test_hearing_without_arena_hearing_names_the_feature() -> None:
    proc = _launch({"acoustics": "arena", "robot.hearing": "bus"}, with_auditory=False)
    assert proc.returncode != 0
    assert "arena feature hearing install" in proc.stderr


def test_old_auditory_selector_names_its_replacement() -> None:
    proc = _launch({"auditory": "arena"}, with_auditory=False)
    assert proc.returncode != 0
    assert "auditory:=arena is now acoustics:=arena" in proc.stderr


def test_hearing_needs_arena_acoustics() -> None:
    proc = _launch({"acoustics": "none", "robot.hearing": "bus"}, with_auditory=False)
    assert proc.returncode != 0
    assert "needs acoustics:=arena" in proc.stderr


@pytest.mark.usefixtures("requires_auditory")
def test_acoustics_arena_leaves_empty_paths_to_the_auditory_defaults() -> None:
    found = _found(_launch({"acoustics": "arena"}, with_auditory=True))
    assert _ACOUSTICS in found["includes"]
    assert _AUDITORY in found["includes"]
    assert _auditory_params(found)
    for name, params in _auditory_params(found).items():
        assert "array.spec" not in params, name


@pytest.mark.usefixtures("requires_auditory")
def test_acoustics_arena_forwards_explicit_paths() -> None:
    found = _found(_launch({"acoustics": "arena", "auditory.array.spec": "/tmp/array.yaml"}, with_auditory=True))
    assert _auditory_params(found)
    for name, params in _auditory_params(found).items():
        assert params["array.spec"] == "/tmp/array.yaml", name


@pytest.mark.usefixtures("requires_auditory", "requires_hearing")
def test_hearing_included_only_when_enabled() -> None:
    without = _found(_launch({"acoustics": "arena"}, with_auditory=True))
    assert _HEARING not in without["includes"]
    assert _ARENA_HEARING not in without["includes"]
    found = _found(_launch({"acoustics": "arena", "robot.hearing": "srp", "robot.hearing.srp.hop_s": "0.05"}, with_auditory=True))
    assert _HEARING in found["includes"]
    assert "robot.hearing.srp.hop_s" in found["includes"][_ARENA_HEARING]
    assert "robot.hearing.policy" not in found["includes"][_ARENA_HEARING]
    assert found["params"]["task_generator_node"]["robot.mobile.params_overlay"].endswith("arena_hearing/config/nav2_overlay.yaml")
    assert "robot.hearing.srp.hop_s" not in found["params"]["task_generator_node"]


@pytest.mark.usefixtures("requires_auditory", "requires_hearing")
def test_hearing_emission_level_of_an_unknown_kind_is_rejected() -> None:
    proc = _launch({"acoustics": "arena", "robot.hearing": "srp", "robot.hearing.belief.emission_db.footstp": "55.0"}, with_auditory=True)
    assert proc.returncode != 0
    assert "robot.hearing.belief.emission_db.footstp names no detect kind" in proc.stderr


@pytest.mark.usefixtures("requires_auditory", "requires_hearing")
def test_hearing_emission_level_of_the_onset_kind_is_forwarded() -> None:
    found = _found(_launch({"acoustics": "arena", "robot.hearing": "srp", "robot.hearing.belief.emission_db.onset": "60.0"}, with_auditory=True))
    assert "robot.hearing.belief.emission_db.onset" in found["includes"][_ARENA_HEARING]
