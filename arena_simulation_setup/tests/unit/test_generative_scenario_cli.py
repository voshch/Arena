from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.formats import load_world
from arena_simulation_setup.generative.scenario.cli import _grid_loader, _non_negative_float, _override_robot, _parser, _positive_float, _positive_int, _route_reachable, _world_parts, main

SHIPPED = Path(__file__).resolve().parents[2] / "worlds"


def _worlds(tmp_path: Path) -> Path:
    root = tmp_path / "worlds"
    (root / "sbend").mkdir(parents=True)
    shutil.copy(SHIPPED / "sbend_corridor" / "world.yaml", root / "sbend" / "world.yaml")
    return root


def _populate(tmp_path: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, str, str]:
    rc = main(["populate", "sbend", "--worlds", str(_worlds(tmp_path)), *extra])
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_a_coordinate_robot_start_is_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _populate(tmp_path, capsys, "--robot-start", "1,1.5,0")
    assert rc == 0
    assert "start=[1.0, 1.5, 0.0]" in out


def test_a_robot_goal_applies_without_a_robot_start(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _populate(tmp_path, capsys, "--robot-goal", "8,11.5")
    assert rc == 0
    assert "goal=[8.0, 11.5]" in out


def test_a_zone_robot_goal_is_kept_as_a_zone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _populate(tmp_path, capsys, "--robot-start", "corridor_south", "--robot-goal", "corridor_north")
    assert rc == 0
    assert "start='corridor_south' goal='corridor_north'" in out


@pytest.mark.parametrize("value", ["nowhere", "1,2,x", "1", "1,2,3,4"])
def test_a_bad_robot_endpoint_names_the_valid_forms(tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str) -> None:
    rc, _, err = _populate(tmp_path, capsys, "--robot-start", value)
    assert rc == 1
    assert "x,y[,yaw]" in err
    assert "corridor_north" in err


def test_a_new_start_keeps_the_route_and_a_new_goal_replaces_it() -> None:
    zones = [{"name": "a"}, {"name": "b"}]
    robot = {"start": "a", "phases": [{"goto": "b"}, {"goto": "a"}], "level": "1", "goal_level": "2"}
    moved = _override_robot(robot, zones, start="0,0", goal=None)
    assert moved == {**robot, "start": [0.0, 0.0]}
    retargeted = _override_robot(robot, zones, start=None, goal="b")
    assert retargeted == {"start": "a", "goal": "b", "level": "1"}


def _promptgen(worlds: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, str, str]:
    rc = main(["promptgen", "sbend", "--worlds", str(worlds), "--oracle", *extra])
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_promptgen_without_any_base_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, err = _promptgen(_worlds(tmp_path), capsys)
    assert rc == 1
    assert "nothing generated: none of the bases normal, break, night, empty could be loaded" in err


def test_promptgen_fails_when_a_prompt_crashes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    scenarios = worlds / "sbend" / "scenarios"
    doc = yaml.safe_load((scenarios / "normal" / "scenario.yaml").read_text())
    (scenarios / "broken").mkdir()
    (scenarios / "broken" / "scenario.yaml").write_text(yaml.safe_dump({**doc, "humans": [{"name": "x", "pose": ["a", "b"]}]}))
    capsys.readouterr()
    rc, out, err = _promptgen(worlds, capsys, "--bases", "broken,normal", "--ids", "u_002")
    assert "crashed=1" in out and "ok=1" in out, out
    assert rc == 1
    assert "1 prompt(s) crashed" in err


def test_a_case_the_scenario_model_rejects_fails_only_its_prompt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    scenarios = worlds / "sbend" / "scenarios"
    doc = yaml.safe_load((scenarios / "normal" / "scenario.yaml").read_text())
    (scenarios / "odd").mkdir()
    shutil.copy(scenarios / "normal" / "talker.yaml", scenarios / "odd" / "talker.yaml")
    (scenarios / "odd" / "scenario.yaml").write_text(yaml.safe_dump({**doc, "knobs": {"speed": "high"}}))
    capsys.readouterr()
    rc, out, err = _promptgen(worlds, capsys, "--bases", "odd,normal", "--ids", "u_002", "--write")
    assert "odd/u_002: the scenario model rejects the case" in out and "knobs" in out, out
    assert "rejected=1" in out and "ok=1" in out and "wrote 1" in out, out
    assert rc == 1
    assert "1 case(s) not written" in err
    assert (scenarios / "normal__u_002" / "scenario.yaml").is_file()
    assert not (scenarios / "odd__u_002").exists()


@pytest.mark.parametrize("verb", ["zones", "classify"])
def test_a_malformed_world_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str], verb: str) -> None:
    worlds = _worlds(tmp_path)
    (worlds / "sbend" / "world.yaml").write_text("format: arena-world/2\nlevels: [\n")
    assert main([verb, "sbend", "--worlds", str(worlds)]) == 1
    assert "sbend: a world file is not valid YAML" in capsys.readouterr().err


def test_promptgen_succeeds_when_a_case_is_realized(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    capsys.readouterr()
    rc, out, err = _promptgen(worlds, capsys, "--bases", "normal", "--ids", "u_002")
    assert rc == 0, out + err
    assert "ok=1" in out or "partial=1" in out


def test_every_verb_and_option_has_help() -> None:
    verbs = next(a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction))
    missing = [f"{name} {action.dest}" for name, sub in verbs.choices.items() for action in sub._actions if not action.help]
    assert missing == []


@pytest.mark.parametrize("verb", ["zones", "classify", "populate", "promptgen", "suite"])
def test_a_bare_verb_prints_its_help_and_the_worlds(tmp_path: Path, capsys: pytest.CaptureFixture[str], verb: str) -> None:
    assert main([verb, "--worlds", str(_worlds(tmp_path))]) == 2
    out = capsys.readouterr().out
    assert f"usage: generate_scenario {verb}" in out
    assert "sbend" in out


def test_an_unknown_world_names_the_worlds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["zones", "nope", "--worlds", str(_worlds(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "'nope'" in err and "sbend" in err


def test_prune_without_write_says_to_add_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, err = _populate(tmp_path, capsys, "--prune")
    assert rc == 1
    assert "add --write" in err


def test_unknown_bases_and_ids_name_the_valid_ones(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    capsys.readouterr()
    rc, _, err = _promptgen(worlds, capsys, "--bases", "normal,nope")
    assert rc == 1
    assert "nope" in err and "break, empty, night, normal" in err
    rc, _, err = _promptgen(worlds, capsys, "--ids", "u_002,nope")
    assert rc == 1
    assert "nope" in err and "u_001" in err


def _run(*argv: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "arena_simulation_setup.generative.scenario.cli", *argv], env={**os.environ, "ARENA_WORLD_PATH": "", **env}, capture_output=True, text=True, timeout=300, check=False)


def _checkout(tmp_path: Path) -> Path:
    arena = tmp_path / "arena"
    shutil.copytree(_worlds(tmp_path), arena / "arena_simulation_setup" / "worlds")
    (arena / "arena_evaluation" / "arena_evaluation" / "configs" / "benchmark" / "suites").mkdir(parents=True)
    return arena


def test_listing_honors_arena_world_path(tmp_path: Path) -> None:
    arena = _checkout(tmp_path)
    extra = tmp_path / "extra"
    shutil.copytree(arena / "arena_simulation_setup" / "worlds" / "sbend", extra / "mine")
    done = _run("zones", ARENA_DIR=str(arena), ARENA_WORLD_PATH=str(extra))
    assert done.returncode == 2
    assert "mine, sbend" in done.stdout


def test_writes_default_to_the_source_tree(tmp_path: Path) -> None:
    arena = _checkout(tmp_path)
    done = _run("populate", "sbend", "--write", ARENA_DIR=str(arena))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (arena / "arena_simulation_setup" / "worlds" / "sbend" / "scenarios" / "normal" / "scenario.yaml").is_file()
    done = _run("suite", "sbend", "--write", ARENA_DIR=str(arena))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (arena / "arena_evaluation" / "arena_evaluation" / "configs" / "benchmark" / "suites" / "edge_case.yaml").is_file()


def test_without_a_root_the_verb_asks_for_one(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "ARENA_DIR"}
    done = subprocess.run([sys.executable, "-m", "arena_simulation_setup.generative.scenario.cli", "populate", "sbend"], env={**env, "ARENA_WORLD_PATH": "", "ARENA_BUCKETS": ","}, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 1
    assert "no resolver has it" in done.stderr
    assert "--worlds" in done.stderr and "ARENA_DIR" in done.stderr


def test_a_world_outside_the_source_roots_is_read_through_the_resolvers_but_not_written(tmp_path: Path) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    arena = _checkout(tmp_path)
    cache = tmp_path / "assets"
    cache.mkdir()
    resolved = Path(WorldIdentifier("hospital_1").resolve_sync().path)
    done = _run("zones", "hospital_1", ARENA_DIR=str(arena), ARENA_BUCKETS=",", ARENA_ASSETS_DIR=str(cache))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "transit" in done.stdout
    done = _run("populate", "hospital_1", "--write", ARENA_DIR=str(arena), ARENA_BUCKETS=",", ARENA_ASSETS_DIR=str(cache))
    assert done.returncode == 1
    assert f"resolves to {resolved}, which is no source world root ({arena / 'arena_simulation_setup' / 'worlds'})" in done.stderr
    assert not any(cache.rglob("*"))


def test_writes_into_an_install_prefix_or_the_asset_cache_are_refused(tmp_path: Path) -> None:
    src = _worlds(tmp_path)
    prefix = tmp_path / "install" / "pkg"
    shutil.copytree(src, prefix / "share" / "worlds")
    done = _run("populate", "sbend", "--write", "--worlds", str(prefix / "share" / "worlds"), AMENT_PREFIX_PATH=f"{os.environ.get('AMENT_PREFIX_PATH', '')}:{prefix}")
    assert done.returncode == 1
    assert "install prefix" in done.stderr
    assert not (prefix / "share" / "worlds" / "sbend" / "scenarios").exists()
    assets = tmp_path / "assets"
    shutil.copytree(src, assets / "worlds")
    done = _run("suite", "sbend", "--worlds", str(assets / "worlds"), "--out", str(assets / "s.yaml"), "--write", ARENA_ASSETS_DIR=str(assets))
    assert done.returncode == 1
    assert "asset cache" in done.stderr


@pytest.mark.parametrize("level", [0, "0", "2"])
def test_the_base_route_keeps_the_robots_levels(level: int | str) -> None:
    from arena_simulation_setup.generative.scenario.route import robot_for

    spec = robot_for({"start": [0.0, 0.0, 0.0], "goal": [20.0, 0.0], "level": level, "goal_level": level}, [])
    assert spec["level"] == level and spec["goal_level"] == level


def test_a_world_loads_only_its_own_prompt_set(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.prompts import load_sets

    for name in ("universal", "hospital_1", "hospital_1_bare"):
        (tmp_path / f"{name}.yaml").write_text(f"- {{id: {name}, text: t}}\n")
    assert [p.id for p in load_sets("hospital_1", directory=tmp_path)] == ["universal", "hospital_1"]
    assert [p.id for p in load_sets("hospital_1_bare", directory=tmp_path)] == ["universal", "hospital_1_bare"]


_NO_MODEL = {k: v for k, v in os.environ.items() if k not in ("LLM_API_ENDPOINT", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ARENA_DATA_DIR")}


def _python(code: str, cwd: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env={**_NO_MODEL, **env}, capture_output=True, text=True, timeout=120, check=False)


def test_the_default_llm_cache_is_absolute_without_arena_data_dir(tmp_path: Path) -> None:
    done = _python("from arena_simulation_setup.generative.scenario.promptgen.llm import DEFAULT_CACHE; print(DEFAULT_CACHE)", tmp_path, ARENA_WS_DIR=str(tmp_path / "ws"))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(tmp_path / "ws" / "data" / "promptgen" / "llm_cache")


def test_a_truncated_cache_entry_is_a_miss(tmp_path: Path) -> None:
    code = """
from pathlib import Path
from arena_simulation_setup.generative.scenario.promptgen.llm import Unavailable, call
from arena_simulation_setup.generative.scenario.propose.client import DEFAULT_MODEL, prompt_sha256
cache = Path('cache')
cache.mkdir()
(cache / f"{prompt_sha256(DEFAULT_MODEL, 's', 'u')}.json").write_text('{"model": "m", "answ')
try:
    call('s', 'u', cache_dir=cache)
except Unavailable:
    print('miss')
"""
    done = _python(code, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "miss"


def test_a_cache_entry_is_written_whole_or_not_at_all(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.llm import _read_cached, _write_cached

    path = tmp_path / "k.json"
    _write_cached(path, {"model": "m", "answer": "{}"})
    assert [p.name for p in tmp_path.iterdir()] == ["k.json"]
    answer = _read_cached(path, "m")
    assert answer is not None and answer.text == "{}" and answer.cached


def _sealed_rooms(tmp_path: Path, n: int = 4) -> Path:
    zones = "\n".join(f"  - {{name: room_{i}, polygon: [[{10 * i}.0, 0.0], [{10 * i + 4}.0, 0.0], [{10 * i + 4}.0, 4.0], [{10 * i}.0, 4.0]]}}" for i in range(n))
    walls = "\n".join(f"  - {{path: [[{10 * i}.0, 0.0], [{10 * i + 4}.0, 0.0], [{10 * i + 4}.0, 4.0], [{10 * i}.0, 4.0], [{10 * i}.0, 0.0]]}}" for i in range(n))
    root = tmp_path / "sealed" / "rooms"
    root.mkdir(parents=True)
    (root / "world.yaml").write_text(f"format: arena-world/2\nlevels:\n- id: '0'\n  zones:\n{zones}\n  walls:\n{walls}\n")
    return root


def test_the_route_checks_of_a_world_share_one_grid(tmp_path: Path) -> None:
    root = _sealed_rooms(tmp_path)
    grid = _grid_loader(root, load_world(root))
    zones = [{"name": f"room_{i}", "corners": [{"x": 10 * i + dx, "y": dy} for dx, dy in ((0, 0), (4, 0), (4, 4), (0, 4))]} for i in range(4)]
    assert [_route_reachable(grid, zones, {"start": f"room_{i}", "goal": f"room_{j}"}) for i in range(4) for j in range(i + 1, 4)] == [False] * 6
    assert _route_reachable(grid, zones, {"start": [1.0, 2.0, 0.0], "goal": [3.0, 2.0]})
    assert grid.cache_info().misses == 1


def test_without_a_reachable_pair_the_farthest_zones_are_the_endpoints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _sealed_rooms(tmp_path)
    world = load_world(root)
    _, robot = _world_parts(root, world, _grid_loader(root, world))
    assert robot == {"start": "room_0", "goal": "room_3"}


def test_a_rejected_base_leaves_nothing_behind(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.materialize import write_base

    world = _worlds(tmp_path) / "sbend"
    doc = {"humans": [{"name": "a", "pose": "corridor_south", "agent": {"agent_type": "./talker.yaml"}}], "knobs": {"speed": "high"}}
    with pytest.raises(ValueError, match="knobs"):
        write_base(doc, world_root=world, profile="normal")
    assert not (world / "scenarios").exists()


def test_populate_names_the_assets_stamped_as_the_runtime_box(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    world = worlds / "sbend"
    doc = yaml.safe_load((world / "world.yaml").read_text())
    doc["levels"][0]["objects"] = [{"name": "bare", "pose": [1.0, 1.5, 0.0], "model": "Pytest_BarePlant"}]
    (world / "world.yaml").write_text(yaml.safe_dump(doc))
    (world / "assets" / "Common" / "Object" / "Pytest_BarePlant").mkdir(parents=True)
    assert main(["populate", "sbend", "--worlds", str(worlds)]) == 0
    assert "1 asset(s) with no annotated footprint, stamped as the runtime's 2 m box: Pytest_BarePlant" in capsys.readouterr().out


def test_a_written_base_carries_its_talker_type(tmp_path: Path) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    normal = worlds / "sbend" / "scenarios" / "normal"
    assert "./talker.yaml" in (normal / "scenario.yaml").read_text()
    assert (normal / "talker.yaml").is_file()


@pytest.mark.parametrize(
    ("check", "value", "expected"),
    [
        (_positive_float, "0", "expected a number above 0"),
        (_positive_float, "-0.2", "expected a number above 0"),
        (_positive_float, "nan", "is not a number"),
        (_positive_float, "fast", "is not a number"),
        (_non_negative_float, "-1", "expected a number of at least 0"),
        (_non_negative_float, "inf", "is not a number"),
        (_positive_int, "0", "expected a whole number of at least 1"),
        (_positive_int, "1.5", "is not a number"),
    ],
)
def test_a_value_out_of_range_names_the_valid_range(check, value: str, expected: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=expected):
        check(value)


def test_a_value_in_range_is_parsed() -> None:
    assert (_positive_float("0.35"), _non_negative_float("0"), _positive_int("3")) == (0.35, 0.0, 3)


@pytest.mark.parametrize("argv", [["populate", "sbend", "--robot-radius", "0"], ["suite", "sbend", "--episodes", "0"]])
def test_the_verbs_refuse_a_value_out_of_range(capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(argv)
    assert exited.value.code == 2
    assert "expected a" in capsys.readouterr().err


def _doorway(tmp_path: Path) -> Path:
    root = tmp_path / "doorway"
    root.mkdir()
    walls = [{"path": [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0], [0.0, 0.0]]}, {"path": [[5.0, 0.0], [5.0, 2.8]]}]
    zones = [{"name": "west", "polygon": [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]]}, {"name": "east", "polygon": [[5.0, 0.0], [10.0, 0.0], [10.0, 4.0], [5.0, 4.0]]}]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones, "walls": walls}]}))
    return root


def test_the_robot_radius_decides_whether_the_base_route_passes_a_doorway(tmp_path: Path) -> None:
    root = _doorway(tmp_path)
    grid = _grid_loader(root, load_world(root))
    robot = {"start": [2.0, 2.0, 0.0], "goal": [8.0, 2.0]}
    zones: list = []
    assert _route_reachable(grid, zones, robot, 0.35)
    assert not _route_reachable(grid, zones, robot, 0.8)


_COUNT_WORLD_READS = """
import sys
from pathlib import Path
reads = []
target = str(Path(sys.argv[1]) / "world.yaml")
sys.addaudithook(lambda event, args: reads.append(1) if event == "open" and str(args[0]) == target else None)
from arena_simulation_setup.generative.scenario.cli import main
from arena_simulation_setup.generative.scenario.formats import load_world
load_world(sys.argv[1])
once = len(reads)
for verb in ("zones", "classify", "populate"):
    reads.clear()
    assert main([verb, Path(sys.argv[1]).name, "--worlds", str(Path(sys.argv[1]).parent)]) == 0
    print(verb, len(reads) // once if once else -1, len(reads) % once if once else -1)
"""


def test_each_verb_reads_the_world_once(tmp_path: Path) -> None:
    world = _worlds(tmp_path) / "sbend"
    done = subprocess.run([sys.executable, "-c", _COUNT_WORLD_READS, str(world)], env={**os.environ, "ARENA_WORLD_PATH": ""}, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [line.split()[1:] for line in done.stdout.splitlines() if line.split()[0] in ("zones", "classify", "populate")] == [["1", "0"]] * 3
