"""Two-person retargeting: contacts survive on Arena's bodies, placements match the source, files and CLI round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from arena_simulation_setup.adapters.arena_animation_retarget.cli import main as cli_main
from arena_simulation_setup.adapters.arena_animation_retarget.pair import load_pair, retarget_pair, window
from arena_simulation_setup.adapters.arena_animation_retarget.smplx import REST_OFFSETS, J, forward_kinematics

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget import FPS, load_motion  # noqa: E402

SEP = 0.7  # m between the pelvises
HEIGHT = 0.95  # pelvis height (Y up)


def _handshake(T: int = 90):
    """Person a at the origin facing +Z, b at SEP facing -Z; both right arms swing from the side to meet in front of
    the chest (wrists 8 cm apart, like a grip), hold, and drop again. Returns local rotations and roots per person."""
    Ry = Rotation.from_rotvec([0, np.pi, 0]).as_matrix()
    roots = [np.array([0.0, HEIGHT, 0.0]), np.array([0.0, HEIGHT, SEP])]
    rots = [np.eye(3), Ry]
    _, P = forward_kinematics(np.tile(np.eye(3), (1, 22, 1, 1)), np.zeros((1, 3)))
    sh = P[0, J["r_shoulder"]]  # body frame
    meet = np.array([0.0, sh[1] + HEIGHT - 0.3, SEP / 2])  # world, chest height, between them
    arm = REST_OFFSETS[J["r_elbow"]] + REST_OFFSETS[J["r_wrist"]]
    u = np.clip(np.sin(np.pi * np.arange(T) / (T - 1)) * 1.4 - 0.2, 0, 1)  # rest, out, hold, back, rest
    out = []
    for i, (root, R) in enumerate(zip(roots, rots)):
        grip = meet + np.array([0.0, 0.0, -0.04 if i == 0 else 0.04])
        tgt = R.T @ (grip - root - R @ sh)  # shoulder -> grip in the body frame
        tgt = tgt / np.linalg.norm(tgt) * min(np.linalg.norm(tgt), 0.98 * np.linalg.norm(arm))
        ax = np.cross(arm, tgt)
        ang = np.arctan2(np.linalg.norm(ax), arm @ tgt)
        L = np.tile(np.eye(3), (T, 22, 1, 1))
        L[:, J["r_shoulder"]] = Rotation.from_rotvec(ax / np.linalg.norm(ax) * (ang * u)[:, None]).as_matrix()
        L[:, 0] = R
        out.append((L, np.tile(root, (T, 1))))
    return out


def _motions():
    return tuple(load_motion(L, root) for L, root in _handshake())


def _save_pair(path: Path, persons) -> Path:
    rec = {"fps": FPS, "up": "y"}
    for i, (L, root) in enumerate(persons):
        aa = Rotation.from_matrix(L.reshape(-1, 3, 3)).as_rotvec().reshape(len(L), 22, 3)
        rec[f"p{i}_root_orient"], rec[f"p{i}_pose_body"] = aa[:, 0], aa[:, 1:].reshape(len(L), 63)
        rec[f"p{i}_trans"] = root
    np.savez(path, **rec)
    return path


def test_source_contact_window():
    m0, m1 = _motions()
    f0, f1 = window(m0, m1, "auto")
    assert 0 < f0 < f1 < len(m0), "the window trims the frames before the hands meet and after they part"
    assert window(m0, m1, "all") == (0, len(m0))


def test_contacts_meet_on_arena_bodies():
    """Arena's arms are proportioned differently: without anchors the hands miss; with them they meet as in the source."""
    res = retarget_pair(*_motions(), window_mode="all")
    c = res.contact
    assert c["contact_frames"] > 0.2
    assert c["contact_mean"] < 3.0, c
    assert c["contact_mean_noanchor"] > 2 * c["contact_mean"], c


def test_placements_match_source():
    res = retarget_pair(*_motions(), window_mode="all", passes=1)
    a, b = res.placements
    dx, dy, dyaw = a.to_local(b)
    assert dx == pytest.approx(SEP, abs=0.01) and abs(dy) < 0.01
    assert abs(abs(dyaw) - np.pi) < 0.02  # facing each other


def test_pair_files_and_dirs(tmp_path):
    persons = _handshake()
    m0, m1 = _motions()
    f = _save_pair(tmp_path / "shake.npz", persons)
    for got in (load_pair(f),):
        for g, m in zip(got, (m0, m1)):
            np.testing.assert_allclose(g.joints, m.joints, atol=1e-6)
    d = tmp_path / "G001T000A000R000"  # the Inter-X dataset layout: P1.npz + P2.npz with SMPL-X parameters, Y up
    d.mkdir()
    for i, (L, root) in enumerate(persons):
        aa = Rotation.from_matrix(L.reshape(-1, 3, 3)).as_rotvec().reshape(len(L), 22, 3)
        np.savez(d / f"P{i + 1}.npz", root_orient=aa[:, 0], pose_body=aa[:, 1:].reshape(len(L), 63), trans=root)
    for g, m in zip(load_pair(d), (m0, m1)):
        np.testing.assert_allclose(g.joints, m.joints, atol=1e-6)


def test_cli_pair(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _save_pair(src / "shake.npz", _handshake())
    out = tmp_path / "anim"
    assert cli_main([str(src), "-o", str(out), "--pair", "--prefix", "t_", "--report", str(tmp_path / "r.csv"), "-q"]) == 0
    for role, partner in (("a", "t_shake_b"), ("b", "t_shake_a")):
        d = out / f"t_shake_{role}"
        assert (d / f"t_shake_{role}.anim").exists()
        ann = yaml.safe_load((d / "annotation.yaml").read_text())
        assert ann["pair"]["partner"] == partner and ann["pair"]["role"] == role
        assert ann["pair"]["separation"] == pytest.approx(SEP, abs=0.02)
    assert "contact_mean" in (tmp_path / "r.csv").read_text()


@pytest.mark.parametrize("flag", ["--name", "--desc"])
def test_cli_pair_single_source_flags_refuse_several_pairs(tmp_path, flag):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("one", "two"):
        _save_pair(src / f"{name}.npz", _handshake(30))
    out = tmp_path / "anim"
    with pytest.raises(SystemExit, match="single SRC"):
        cli_main([str(src), "-o", str(out), "--pair", flag, "x", "--method", "none", "-q"])
    assert not out.exists()


@pytest.mark.parametrize("value", ["10", "20:5", "a:b"])
def test_cli_pair_window_refuses_a_malformed_range(tmp_path, capsys, value):
    with pytest.raises(SystemExit):
        cli_main([str(tmp_path), "-o", str(tmp_path / "anim"), "--pair", "--window", value])
    assert "F0:F1" in capsys.readouterr().err


def test_window_outside_the_clip_holds_no_frames():
    m0, m1 = _motions()
    with pytest.raises(ValueError, match="no frames"):
        window(m0, m1, (500, 900))


def test_cli_pair_directory_as_source_takes_its_own_name_from_path(tmp_path):
    d = tmp_path / "G001"
    d.mkdir()
    for i, (L, root) in enumerate(_handshake(30)):
        aa = Rotation.from_matrix(L.reshape(-1, 3, 3)).as_rotvec().reshape(len(L), 22, 3)
        np.savez(d / f"P{i + 1}.npz", root_orient=aa[:, 0], pose_body=aa[:, 1:].reshape(len(L), 63), trans=root)
    out = tmp_path / "anim"
    assert cli_main([str(d), "-o", str(out), "--pair", "--name-from-path", "--method", "none", "-q"]) == 0
    assert (out / "G001_a" / "G001_a.anim").exists()


@pytest.mark.parametrize("flag", ["--no-fold", "--check", "--no-score"])
def test_cli_pair_refuses_single_person_flags(tmp_path, capsys, flag):
    with pytest.raises(SystemExit):
        cli_main([str(tmp_path), "-o", str(tmp_path / "anim"), "--pair", flag])
    assert "--pair takes no" in capsys.readouterr().err


def test_chunked_pair_solve_matches_one_chunk():
    whole = retarget_pair(*_motions(), window_mode="all", passes=1)
    chunked = retarget_pair(*_motions(), window_mode="all", passes=1, chunk_frames=1)
    for a, b in zip(whole.clips, chunked.clips, strict=True):
        assert np.allclose(np.stack(list(a.angles.values())), np.stack(list(b.angles.values())), atol=0.05)


def test_cli_pair_too_short_for_the_window_fails_alone(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _save_pair(src / "long.npz", _handshake(90))
    _save_pair(src / "short.npz", _handshake(30))
    out = tmp_path / "anim"
    assert cli_main([str(src), "-o", str(out), "--pair", "--window", "40:60", "--method", "none", "-q"]) == 1
    assert (out / "long_a" / "long_a.anim").exists()
    assert not (out / "short_a").exists()
