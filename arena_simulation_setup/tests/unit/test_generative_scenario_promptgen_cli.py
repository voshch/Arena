from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.cli import main

pytest.importorskip("numpy")

_ROOMS = {"reception": (0, 0, 8, 8), "central_hallway": (8, 3, 24, 5), "open_work_area": (24, 0, 32, 8)}
_PROMPT = {"id": "t_001", "text": "The lights go out.", "targets": ["A:blackout"]}


def _world(tmp_path: Path) -> Path:
    worlds = tmp_path / "worlds"
    root = worlds / "toy"
    root.mkdir(parents=True)
    zones = [{"name": n, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "floor": False} for n, (x0, y0, x1, y1) in _ROOMS.items()]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    walker = {"name": "normal_1", "model": "Hospital/doctor_male_caucasian_middleage", "pose": "reception", "agent": {"agent_type": "adult", "desired_velocity": 1.1}, "waypoints": ["reception", "open_work_area"], "waypoint_mode": "reverse"}
    base = root / "scenarios" / "normal"
    base.mkdir(parents=True)
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", "robots": [{"start": [2.0, 4.0, 0.0], "phases": [{"goto": [30.0, 4.0, 0.0]}]}], "humans": [walker]}, sort_keys=False))
    (tmp_path / "prompts.yaml").write_text(yaml.safe_dump([_PROMPT]))
    return worlds


def _promptgen(tmp_path: Path, *extra: str) -> int:
    return main(["promptgen", "toy", "--worlds", str(tmp_path / "worlds"), "--bases", "normal", "--prompts", str(tmp_path / "prompts.yaml"), "--ids", "t_001", "--oracle", "--write", *extra])


def test_promptgen_designs_the_case_for_the_robot_radius_it_is_given(tmp_path: Path) -> None:
    worlds = _world(tmp_path)
    assert _promptgen(tmp_path, "--robot-radius", "0.6", "--robot-speed", "0.7", "--hold", "2") == 0
    robot = yaml.safe_load((worlds / "toy" / "scenarios" / "normal__t_001" / "provenance.yaml").read_text())["robot"]
    assert robot == {"speed_mps": 0.7, "hold_s": 2.0, "traversals": 1, "tolerance_m": None, "radius_m": 0.6}


@pytest.mark.parametrize("flag", [("--robot-radius", "0"), ("--robot-speed", "0"), ("--traversals", "0"), ("--hold", "-1"), ("--tolerance", "-0.5")])
def test_promptgen_names_the_valid_range_of_a_robot_argument_out_of_range(tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: tuple[str, str]) -> None:
    _world(tmp_path)
    with pytest.raises(SystemExit) as exited:
        _promptgen(tmp_path, *flag)
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert flag[0] in err
    assert "expected a" in err
