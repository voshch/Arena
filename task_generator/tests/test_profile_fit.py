"""Gait profile fitter: synthesized clips back to their profile, the documented errors, the CLI surfaces."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from task_generator.simulators.human.fit import FitError, fit_clip, load_clip, main
from task_generator.simulators.human.gait import GaitGenerator
from task_generator.simulators.human.profile import DEFAULT_WALK, JOINT_NAMES, LIBRARY_DIR, CadenceLaw, GainLaw, GaitProfile, PoseProfile, Signal, dump_gait_profile, load_library, resolve_pose_profile

_WALKING = 1
FPS = 20.0
WIDE = {"l_r_hip": [-0.5, 0.7], "r_r_hip": [-0.5, 0.7], "l_knee": [-2.5, 0.1], "r_knee": [-2.5, 0.1]}
OPTIONS = ("CLIP.npy", "-o", "--output", "--fps", "--harmonics", "--speed", "--report", "--name")


def _clip(pose: PoseProfile, cadence: float, seconds: float = 6.0, start: float = 0.3) -> tuple[list[dict[str, float]], list[float]]:
    gen = GaitGenerator()
    phases = [start + 2.0 * math.pi * cadence * i / FPS for i in range(round(seconds * FPS))]
    return [gen.compute(0, _WALKING, 1.2, 0.0, phase=phi, profile=pose) for phi in phases], phases


def _walk_clip(cadence: float = 1.0, seconds: float = 6.0) -> tuple[list[dict[str, float]], list[float]]:
    return _clip(resolve_pose_profile({"walk": {"limits": WIDE}}), cadence, seconds)


def _linear(signal: Signal, harmonics: int = 3) -> list[float]:
    padded = signal.harmonics + ((0.0, 0.0),) * (harmonics - len(signal.harmonics))
    return [signal.mean] + [part for amp, ph in padded for part in (amp * math.cos(ph), amp * math.sin(ph))]


def _errors(profile: GaitProfile, frames: list[dict[str, float]], phases: list[float]) -> list[float]:
    return [pose.get(joint, 0.0) - frame[joint] for frame, phi in zip(frames, phases, strict=True) for pose in (profile.evaluate(phi, 1.0),) for joint in JOINT_NAMES]


def _save(path: Path, frames: list[dict[str, float]]) -> Path:
    np.save(path, np.array([{"angles": frame} for frame in frames], dtype=object))
    return path


@pytest.mark.parametrize("name", [DEFAULT_WALK, "seated"])
def test_dump_reproduces_the_shipped_library_files(name: str) -> None:
    assert dump_gait_profile(load_library()[name]) == (LIBRARY_DIR / f"{name}.yaml").read_text()


def test_round_trip_recovers_the_walk_profile_and_reloads_from_yaml(tmp_path: Path) -> None:
    frames, phases = _walk_clip()
    profile, report = fit_clip(frames, FPS, name="walk_fit")
    walk = load_library()[DEFAULT_WALK]

    assert profile.symmetric and report.symmetric
    assert profile.phase_warp.split == 0.5 and report.split == 0.5
    assert report.cycles == 6 and report.frames_used == 120
    assert report.cadence_hz == pytest.approx(1.0, abs=1e-9)
    assert set(profile.joints) == set(walk.joints)
    for joint, expected in walk.joints.items():
        fitted = profile.joints[joint]
        assert (fitted.key, fitted.shift) == (expected.key, expected.shift)
        assert fitted.signal.mean == pytest.approx(expected.signal.mean, abs=1e-6)
        assert len(fitted.signal.harmonics) == len(expected.signal.harmonics)
        for (amp, ph), (amp_expected, ph_expected) in zip(fitted.signal.harmonics, expected.signal.harmonics, strict=True):
            assert amp == pytest.approx(amp_expected, abs=1e-6)
            assert math.remainder(ph - ph_expected, 2.0 * math.pi) == pytest.approx(0.0, abs=1e-6)
    assert profile.cadence == CadenceLaw(base=report.cadence_hz, per_speed=0.0, lo=report.cadence_hz, hi=report.cadence_hz)
    assert profile.gain == GainLaw(lo=1.0, hi=1.0, factor=1.0)
    assert report.cadence_law == "fixed"
    assert set(profile.limits) == set(WIDE) == set(report.limits_widened)
    assert report.worst_rms < 1e-9

    (tmp_path / "walk_fit.yaml").write_text(dump_gait_profile(profile))
    loaded = load_library(tmp_path)["walk_fit"]
    assert loaded == profile
    assert max(abs(e) for e in _errors(loaded, frames, phases)) < 1e-6
    replayed, _ = _clip(PoseProfile(walk=loaded, run=loaded), 1.0)
    assert all(replayed[i][joint] == pytest.approx(frames[i][joint], abs=1e-6) for i in range(len(frames)) for joint in JOINT_NAMES)


def test_asymmetric_clip_recovers_split_and_per_side_signals() -> None:
    pose = resolve_pose_profile({"walk": {"symmetric": False, "phase_warp": {"split": 0.58}, "limits": WIDE, "joints": {"r_knee": {"scale": 0.4}}}})
    frames, phases = _clip(pose, 0.9)
    profile, report = fit_clip(frames, FPS, name="limp")

    assert not profile.symmetric and not report.symmetric
    assert profile.phase_warp.split == pytest.approx(0.58, abs=0.01)
    assert report.split == profile.phase_warp.split
    knee = load_library()[DEFAULT_WALK].joints["r_knee"].signal
    fitted = profile.joints["r_knee"]
    assert (fitted.key, fitted.shift) == ("r_knee", 0.0)
    assert fitted.signal.mean == pytest.approx(0.4 * knee.mean, rel=0.02)
    for (amp, _), (amp_expected, _) in zip(fitted.signal.harmonics, knee.harmonics, strict=True):
        assert amp == pytest.approx(0.4 * amp_expected, rel=0.02)
    assert profile.joints["l_knee"].signal.harmonics[0][0] == pytest.approx(knee.harmonics[0][0], rel=0.02)
    errors = _errors(profile, frames, phases)
    assert math.sqrt(sum(e * e for e in errors) / len(errors)) < 0.01
    assert report.worst_rms < 0.01


def test_noisy_clip_recovers_coefficients_within_the_noise() -> None:
    frames, _ = _walk_clip()
    rng = np.random.default_rng(7)
    noisy = [{joint: angle + float(rng.normal(0.0, 0.01)) for joint, angle in frame.items()} for frame in frames]
    profile, report = fit_clip(noisy, FPS, name="noisy")

    assert profile.symmetric and profile.phase_warp.split == 0.5
    assert report.cadence_hz == pytest.approx(1.0, abs=0.002)
    for joint, expected in load_library()[DEFAULT_WALK].joints.items():
        fitted = profile.joints[joint]
        assert (fitted.key, fitted.shift) == (expected.key, expected.shift)
        assert _linear(fitted.signal) == pytest.approx(_linear(expected.signal), abs=0.01)
    assert 0.005 < report.worst_rms < 0.02


def test_non_integer_period_recovers_the_walk_profile() -> None:
    frames, phases = _walk_clip(cadence=FPS / 23.7)
    profile, report = fit_clip(frames, FPS, name="odd")

    assert profile.symmetric and profile.phase_warp.split == 0.5
    assert report.period_s * FPS == pytest.approx(23.7, abs=1e-3)
    assert report.cycles == 5 and report.frames_used == 119
    for joint, expected in load_library()[DEFAULT_WALK].joints.items():
        assert _linear(profile.joints[joint].signal) == pytest.approx(_linear(expected.signal), abs=1e-3)
    assert max(abs(e) for e in _errors(profile, frames, phases)) < 1e-3


def test_speed_puts_the_cadence_law_through_the_measured_point() -> None:
    frames, _ = _walk_clip()
    profile, report = fit_clip(frames, FPS, name="paced", speed=1.2)
    assert report.cadence_law == "through_speed" and report.speed == 1.2
    assert profile.cadence == CadenceLaw(base=0.4, per_speed=pytest.approx(0.5, abs=1e-9), lo=0.4, hi=2.2)
    assert profile.cadence(1.2) == pytest.approx(report.cadence_hz, abs=1e-9)
    slow, slow_report = fit_clip(_walk_clip(cadence=0.35, seconds=12.0)[0], FPS, name="slow", speed=0.2)
    assert slow.cadence.lo == pytest.approx(0.35, abs=1e-6) and slow.cadence(0.2) == pytest.approx(slow_report.cadence_hz, abs=1e-9)


def test_clip_shorter_than_two_cycles_is_rejected() -> None:
    frames, _ = _walk_clip(seconds=1.5)
    with pytest.raises(FitError, match="at least 2 whole walking cycles"):
        fit_clip(frames, FPS, name="short")
    with pytest.raises(FitError, match="at least 2 whole walking cycles"):
        fit_clip(frames[:3], FPS, name="short")


def test_constant_clip_is_rejected() -> None:
    frames, _ = _walk_clip()
    with pytest.raises(FitError, match="constant over the clip"):
        fit_clip([frames[0]] * 80, FPS, name="held")


def test_unfittable_inputs_name_what_to_change() -> None:
    frames, _ = _walk_clip()
    for harmonics in (0, 9):
        with pytest.raises(FitError, match="harmonics takes 1 to 8"):
            fit_clip(frames, FPS, name="x", harmonics=harmonics)
    with pytest.raises(FitError, match="no frames"):
        fit_clip([], FPS, name="x")
    with pytest.raises(FitError, match="lacks \\['l_r_hip', 'r_r_hip'\\]"):
        fit_clip([{"waist": 0.1}] * 80, FPS, name="x")
    with pytest.raises(FitError, match="every frame must list the same joints"):
        fit_clip(frames[:60] + [{"l_r_hip": 0.0, "r_r_hip": 0.0}] * 60, FPS, name="x")
    with pytest.raises(FitError, match="lower --harmonics or pass the clip's real --fps"):
        fit_clip(_walk_clip(cadence=4.0)[0], FPS, name="x")
    with pytest.raises(FitError, match="speed takes"):
        fit_clip(frames, FPS, name="x", speed=0.0)


def test_cli_help_names_every_option_and_the_next_step(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for option in (*OPTIONS, "example:", "next:", "profiles/", "1 to 8"):
        assert option in text, option


def test_cli_bare_invocation_prints_usage_and_an_example() -> None:
    result = subprocess.run([sys.executable, "-m", "task_generator.simulators.human.fit"], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage: python3 -m task_generator.simulators.human.fit" in result.stderr
    assert "example: python3 -m task_generator.simulators.human.fit" in result.stderr
    assert "--help" in result.stderr


def test_cli_bad_arguments_name_the_accepted_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clip = _save(tmp_path / "walk.npy", _walk_clip()[0])
    out = tmp_path / "walk.yaml"
    cases = (
        ([str(tmp_path / "nope.npy"), "-o", str(out)], "does not exist, pass a .npy object array of frames"),
        ([str(clip), "-o", str(out), "--harmonics", "9"], "--harmonics takes 1 to 8, got 9"),
        ([str(clip), "-o", str(out), "--fps", "0"], "--fps takes a positive frame rate"),
        ([str(clip), "-o", str(out), "--speed", "-1"], "--speed takes the clip's walking speed"),
        ([str(clip), "-o", str(out), "--name", "idle"], "pass --name NAME"),
        ([str(clip), "-o", str(tmp_path / "missing" / "walk.yaml")], "create it or pass another path"),
        ([str(clip)], "-o/--output"),
    )
    for arguments, message in cases:
        with pytest.raises(SystemExit) as exit_info:
            main(arguments)
        assert exit_info.value.code == 2
        error = capsys.readouterr().err
        assert "usage:" in error and message in error, arguments
    assert not out.exists()


def test_cli_unfittable_clip_exits_1_and_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    short = _save(tmp_path / "short.npy", _walk_clip(seconds=1.5)[0])
    garbage = tmp_path / "garbage.npy"
    garbage.write_text("not a clip")
    out = tmp_path / "out.yaml"
    assert main([str(short), "-o", str(out)]) == 1
    assert "fit: " in capsys.readouterr().err
    assert main([str(garbage), "-o", str(out)]) == 1
    assert "is not readable as a clip" in capsys.readouterr().err
    assert not out.exists()


def test_cli_writes_a_loadable_profile_and_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frames, phases = _walk_clip()
    clip = _save(tmp_path / "walk.npy", frames)
    assert load_clip(clip) == frames
    assert main([str(clip), "-o", str(tmp_path / "walk_cli.yaml"), "--speed", "1.2", "--report", str(tmp_path / "report.json"), "--fps", "20"]) == 0
    printed = capsys.readouterr().out
    for line in ("profile walk_cli: 6 cycles over 120 of 120 frames", "cadence: 1.000000 Hz", "symmetric: true", "phase_warp.split: 0.5", "worst joint:", "rms per joint (rad):", "wrote "):
        assert line in printed, line

    loaded = load_library(tmp_path)["walk_cli"]
    assert loaded.name == "walk_cli" and loaded.symmetric
    assert max(abs(e) for e in _errors(loaded, frames, phases)) < 1e-6
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["cadence_law"] == "through_speed" and report["speed"] == 1.2
    assert report["cycles"] == 6 and report["symmetric"] is True and report["split"] == 0.5
    assert set(report["rms"]) == set(JOINT_NAMES)
    assert report["worst_rms"] == max(report["rms"].values()) == report["rms"][report["worst_joint"]]
