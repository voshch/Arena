"""`arena cam <name> --lockstep` with no recording: gated live playback against a stand-in sim viewport."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import rclpy
import yaml
from standin import Runtime

FPS = 10
PERIOD_NS = 1_000_000_000 // FPS


@pytest.fixture(autouse=True)
def _rclpy():
    rclpy.init()
    yield
    rclpy.shutdown()


def _play(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "arena_cam.cli", *args, "--sim"], capture_output=True, text=True, timeout=120, check=False)


def test_with_no_run_each_frame_steps_the_sim_once_and_captures_nothing() -> None:
    with Runtime() as runtime:
        done = _play("hold", "duration=1", "--lockstep", "--fps", str(FPS))
        assert runtime.wait(lambda: len(runtime.keyframes) >= 8, timeout=30.0)
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 10
    assert runtime.captures == []
    assert len(runtime.holds) == 2


def test_a_run_gates_each_frame_on_the_cams_beat() -> None:
    with Runtime(run_period_ns=PERIOD_NS) as runtime:
        start = runtime.clock_ns
        done = _play("hold", "duration=1", "--lockstep", "--fps", str(FPS))
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 0
    assert runtime.holds == []
    assert len(runtime.beats) == 11
    assert [b - a for a, b in zip(runtime.beats, runtime.beats[1:], strict=False)] == [PERIOD_NS] * 10
    assert runtime.beats[0] >= start
    assert ("cam", "") in runtime.channels
    assert runtime.channels[-1] == ("", "")


def test_the_fps_flag_sets_the_live_frame_count() -> None:
    with Runtime() as runtime:
        done = _play("hold", "duration=1", "--lockstep", "--fps", "5")
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 5


def test_a_rig_plays_its_live_camera_only(tmp_path: Path) -> None:
    rig = {"live": "b", "cameras": {"a": {"reference": {"entity": "a_link"}, "timeline": [{"hold": {"duration": 3.0}}]}, "b": {"reference": {"entity": "b_link"}, "timeline": [{"hold": {"duration": 1.0}}]}}}
    path = tmp_path / "rig.yaml"
    path.write_text(yaml.safe_dump(rig, sort_keys=False))
    with Runtime() as runtime:
        done = _play(str(path), "--lockstep", "--fps", str(FPS))
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 10
    assert "b_link" in runtime.references
    assert "a_link" not in runtime.references


@pytest.mark.parametrize(
    ("entity", "frame"),
    [
        ("${robot}", "env_0/jackal/base_link"),
        ("${robot.jackal}", "env_0/jackal/base_link"),
        ("${human.edge_0}", "body_env_0_agent_4"),
        ("${env}/camera_link", "env_0/camera_link"),
    ],
)
def test_a_context_token_fills_from_the_live_env(entity: str, frame: str) -> None:
    with Runtime() as runtime:
        done = _play("track", f"entity={entity}")
    assert done.returncode == 0, done.stderr
    assert runtime.references == [frame]


def test_an_unfillable_token_stops_the_take_naming_what_the_env_has() -> None:
    with Runtime():
        done = _play("track", "entity=${robot.ghost}")
    assert done.returncode != 0
    assert "jackal" in done.stderr


def test_a_projection_step_reaches_the_viewport(tmp_path: Path) -> None:
    path = tmp_path / "shot.yaml"
    path.write_text(yaml.safe_dump({"projection": "orthographic", "timeline": [{"hold": {"duration": 0.2}}]}))
    with Runtime() as runtime:
        done = _play(str(path), "--lockstep", "--fps", str(FPS))
    assert done.returncode == 0, done.stderr
    assert runtime.projections == ["orthographic"]


def test_params_reach_a_camera_file(tmp_path: Path) -> None:
    path = tmp_path / "shot.yaml"
    path.write_text(yaml.safe_dump({"params": {"seconds": 5.0}, "timeline": [{"hold": {"duration": "${seconds}"}}]}))
    with Runtime() as runtime:
        done = _play(str(path), "seconds=0.3", "--lockstep", "--fps", str(FPS))
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 3


def test_without_lockstep_playback_stays_on_the_wall_clock() -> None:
    with Runtime() as runtime:
        done = _play("hold", "duration=0.5")
        assert runtime.wait(lambda: len(runtime.keyframes) >= 1, timeout=30.0)
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 0
    assert runtime.holds == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["hold", "--fps", "5"], "--fps only applies"),
        (["hold", "-f"], "-f only applies"),
        (["hold", "--lockstep", "--mount", "base_link"], "--mount films in lockstep"),
        (["serve", "duration=1"], "serve takes only"),
    ],
)
def test_flag_rules_exit_with_the_rule(args: list[str], message: str) -> None:
    done = _play(*args)
    assert done.returncode != 0
    assert message in done.stderr
