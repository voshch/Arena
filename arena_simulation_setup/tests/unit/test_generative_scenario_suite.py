from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.generative.scenario.cli import main
from arena_simulation_setup.generative.core.generated import GENERATED_MARKER
from arena_simulation_setup.generative.scenario.suite import build, scenarios


def _scenario(world: Path, name: str, *, generated: bool) -> None:
    (world / "scenarios" / name).mkdir(parents=True)
    (world / "scenarios" / name / "scenario.yaml").write_text((GENERATED_MARKER + "\n" if generated else "") + "robots: []\n")


def _world(tmp_path: Path) -> Path:
    world = tmp_path / "worlds" / "toy"
    for name in ("normal", "normal__u_001", "break__u_002"):
        _scenario(world, name, generated=True)
    for name in ("default", "edge_headon"):
        _scenario(world, name, generated=False)
    return world


def test_scenarios_default_to_what_the_generator_wrote(tmp_path: Path) -> None:
    world = _world(tmp_path)
    assert scenarios(world) == ["break__u_002", "normal", "normal__u_001"]
    assert scenarios(world, ["edge_*", "*__u_001"]) == ["edge_headon", "normal__u_001"]


def test_every_stage_runs_its_scenario_through_the_stock_scenario_modes() -> None:
    doc = build({"toy": ["normal__u_001", "normal"]}, robot="mpo700", episodes=3, seed=2000, timeout="45s")
    assert doc["launch"] == {"human": "arena"}
    assert [s["name"] for s in doc["stages"]] == ["toy.normal__u_001", "toy.normal"]
    stage = doc["stages"][0]
    assert (stage["map"], stage["robot"], stage["episodes"], stage["seed"], stage["timeout"]) == ("toy", "mpo700", 3, 2000, "45s")
    assert stage["tm_robots"] == stage["tm_obstacles"] == "scenario"
    assert stage["config"] == {"scenario": {"file": "normal__u_001"}}


def test_cli_writes_the_suite(tmp_path: Path) -> None:
    _world(tmp_path)
    out = tmp_path / "s.yaml"
    assert main(["suite", "toy", "--worlds", str(tmp_path / "worlds"), "--out", str(out), "--episodes", "2", "--write"]) == 0
    stages = yaml.safe_load(out.read_text())["stages"]
    assert [s["config"]["scenario"]["file"] for s in stages] == ["break__u_002", "normal", "normal__u_001"]
    assert main(["suite", "toy", "--worlds", str(tmp_path / "worlds"), "--out", str(out), "--match", "nope*", "--write"]) == 1


def test_without_write_the_suite_is_printed_not_written(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _world(tmp_path)
    out = tmp_path / "s.yaml"
    assert main(["suite", "toy", "--worlds", str(tmp_path / "worlds"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert not out.exists()
    assert f"(dry run - pass --write to write these 3 stage(s) to {out})" in printed
    assert [s["name"] for s in yaml.safe_load(printed.split("scenario(s)\n")[-1].split("(dry run")[0])["stages"]] == ["toy.break__u_002", "toy.normal", "toy.normal__u_001"]


def test_a_bare_or_unknown_verb_names_the_verbs(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in ([], ["nope"]):
        with pytest.raises(SystemExit):
            main(argv)
        assert "zones,classify,populate,promptgen,suite" in capsys.readouterr().err
