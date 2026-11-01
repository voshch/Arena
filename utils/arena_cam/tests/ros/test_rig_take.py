"""`arena cam <rig>.yaml --record --lockstep` and `--mount` against a stand-in sim viewport, every camera at one sim time."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import rclpy
import yaml
from standin import Runtime

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on PATH")

FPS = 10
PERIOD_NS = 1_000_000_000 // FPS

RIG = {
    "projection": "perspective",
    "live": "chase",
    "cameras": {
        "wide": {"timeline": [{"hold": {"duration": 1.0}}]},
        "chase": {"reference": {"entity": "chassis", "mode": "yaw"}, "timeline": [{"hold": {"duration": 0.5}}]},
        "nose": {"mount": {"frame": "nose_link", "at": [0.2, 0.0, 0.6], "fov": 1.2}},
    },
}


@pytest.fixture(autouse=True)
def _rclpy():
    rclpy.init()
    yield
    rclpy.shutdown()


def _frames(path: Path) -> int:
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False)
    return int(probe.stdout.strip() or 0)


def _take(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "arena_cam.cli", *args, "--sim", "--fps", str(FPS), "--lockstep"], capture_output=True, text=True, timeout=120, check=False)


def _rig_file(tmp_path: Path) -> Path:
    path = tmp_path / "rig.yaml"
    path.write_text(yaml.safe_dump(RIG, sort_keys=False))
    return path


def test_a_rig_writes_one_file_per_camera_with_equal_frame_counts(tmp_path: Path) -> None:
    with Runtime():
        done = _take(str(_rig_file(tmp_path)), f"--record={tmp_path / 'take.mp4'}")
    assert done.returncode == 0, done.stderr
    assert {name: _frames(tmp_path / f"take-{name}.mp4") for name in ("wide", "chase", "nose")} == {"wide": 10, "chase": 10, "nose": 10}


def test_every_camera_of_a_frame_shows_one_sim_time_under_its_own_reference(tmp_path: Path) -> None:
    with Runtime() as runtime:
        done = _take(str(_rig_file(tmp_path)), f"--record={tmp_path / 'take.mp4'}")
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 10
    frames = [runtime.captures[i : i + 3] for i in range(0, len(runtime.captures), 3)]
    assert len(frames) == 10
    assert all(len({capture.stamp_ns for capture in frame}) == 1 for frame in frames)
    assert [capture.reference for capture in frames[0]] == ["world", "chassis", "nose_link"]
    assert frames[0][2].fov == pytest.approx(1.2)
    assert frames[0][2].clip_near == -1.0
    assert runtime.projections == ["perspective"]
    assert runtime.references[-1] == "world"


def test_a_rig_rides_a_run_with_each_frame_on_a_frozen_tick(tmp_path: Path) -> None:
    with Runtime(run_period_ns=PERIOD_NS) as runtime:
        done = _take(str(_rig_file(tmp_path)), f"--record={tmp_path / 'take.mp4'}")
    assert done.returncode == 0, done.stderr
    assert runtime.steps == 0
    assert len(runtime.beats) == 11
    frames = [runtime.captures[i : i + 3] for i in range(0, len(runtime.captures), 3)]
    assert all(len({capture.stamp_ns for capture in frame}) == 1 for frame in frames)
    assert frames[0][0].stamp_ns == runtime.beats[0] + PERIOD_NS
    assert "off the first camera's sim time: 0" in done.stderr


def test_a_camera_that_ended_holds_its_reference_until_the_take_ends(tmp_path: Path) -> None:
    with Runtime() as runtime:
        done = _take(str(_rig_file(tmp_path)), f"--record={tmp_path / 'take.mp4'}")
    assert done.returncode == 0, done.stderr
    assert sum(capture.reference == "chassis" for capture in runtime.captures) == 10


def test_a_mount_films_next_to_a_lone_camera_and_hands_it_back(tmp_path: Path) -> None:
    with Runtime() as runtime:
        done = _take("hold", "duration=0.5", f"--record={tmp_path / 'pass.mp4'}", "--mount", "base_link@0.2,0,0.6")
    assert done.returncode == 0, done.stderr
    assert _frames(tmp_path / "pass-sim.mp4") == 5
    assert _frames(tmp_path / "pass-base_link.mp4") == 5
    assert [capture.reference for capture in runtime.captures[:2]] == ["world", "base_link"]
    assert runtime.references[-1] == "world"
    assert "latched" not in runtime.references
    assert "off the first camera's sim time: 0" in done.stderr


def test_a_rig_without_lockstep_is_refused(tmp_path: Path) -> None:
    with Runtime():
        done = subprocess.run([sys.executable, "-m", "arena_cam.cli", str(_rig_file(tmp_path)), "--sim", f"--record={tmp_path / 'take.mp4'}"], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode != 0
    assert "lockstep" in done.stderr


def test_a_camera_whose_source_selects_no_surface_is_refused(tmp_path: Path) -> None:
    rig = {"cameras": {"a": {"timeline": [{"hold": {"duration": 0.1}}]}, "b": {"source": "viz", "timeline": [{"hold": {"duration": 0.1}}]}}}
    path = tmp_path / "rig.yaml"
    path.write_text(yaml.safe_dump(rig, sort_keys=False))
    with Runtime():
        done = _take(str(path), f"--record={tmp_path / 'take.mp4'}")
    assert done.returncode != 0
    assert "select no surface" in done.stderr
