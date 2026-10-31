"""Retarget tests on synthetic SMPL-X motion (built-in rest skeleton), plus a real Kimodo clip when one is available."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.spatial.transform import Rotation

from arena_simulation_setup.adapters.arena_animation_retarget import fidelity, load, load_clip, retarget, retarget_batch, write_clip
from arena_simulation_setup.adapters.arena_animation_retarget.contract import WIRE
from arena_simulation_setup.adapters.arena_animation_retarget.fk_torch import ContractFK
from arena_simulation_setup.adapters.arena_animation_retarget.ik import HARD_LIMITS
from arena_simulation_setup.adapters.arena_animation_retarget.retarget import prepare, resample
from arena_simulation_setup.adapters.arena_animation_retarget.smplx import BODY_TO_ROS, REST_OFFSETS, J, Motion, forward_kinematics, rest_offsets_from_model

sys.path.insert(0, str(Path(__file__).parent))
import reference  # noqa: E402

FPS = 30.0
# Flexion axis (canonical frame: X left, Y up, Z forward) and sign of the natural bend: knees fold the shin back, elbows
# (T-pose, arms along +-X) fold the forearm forward.
HINGES = {"l_knee": ([1.0, 0, 0], 1), "r_knee": ([1.0, 0, 0], 1), "l_elbow": ([0, 1.0, 0], -1), "r_elbow": ([0, 1.0, 0], 1)}


def _locals(T: int, seed: int, scale: float = 0.35) -> np.ndarray:
    """Smooth random local rotations (T, 22, 3, 3): every joint wanders within about +-scale rad; knees and elbows
    bend like the hinges they mostly are (0 to ~1.4 rad) with a little ab-duction and twist."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FPS
    aa = np.zeros((T, 22, 3))
    for k in range(3):
        f, ph, amp = rng.uniform(0.2, 1.0, (22, 3)), rng.uniform(0, 2 * np.pi, (22, 3)), rng.normal(0, scale, (22, 3))
        aa += amp * np.sin(2 * np.pi * f * t[:, None, None] + ph) / (k + 1)
    aa[:, J["pelvis"]] *= 0.3
    for j, (axis, sign) in HINGES.items():
        bend = 0.7 * (1 + np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t + rng.uniform(0, 2 * np.pi)))
        aa[:, J[j]] = 0.15 * aa[:, J[j]] + sign * bend[:, None] * np.asarray(axis)
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)


def _root(T: int) -> np.ndarray:
    return np.stack([np.linspace(0, 1.0, T), np.full(T, 0.95), np.zeros(T)], 1)  # walk along X at pelvis height (Y up)


def load_motion(L, root, fps=FPS):
    """In-memory equivalent of loading a rotation-output file (Y-up world)."""
    G, P = forward_kinematics(L, root)
    W = BODY_TO_ROS
    return Motion(global_rots=np.einsum("ij,tkjl,ml->tkim", W, G, BODY_TO_ROS), joints=P @ W.T, fps=fps, rest_offsets=REST_OFFSETS)


def _save_rot(path: Path, L: np.ndarray, root: np.ndarray) -> Path:
    G, P = forward_kinematics(L, root)
    np.savez(path, global_rot_mats=G.astype(np.float32), posed_joints=P.astype(np.float32), local_rot_mats=L.astype(np.float32))
    return path


def _save_params(path: Path, L: np.ndarray, root: np.ndarray, world: np.ndarray = np.eye(3)) -> Path:
    L = L.copy()
    L[:, 0] = world @ L[:, 0]  # the root orientation carries the world change
    aa = Rotation.from_matrix(L.reshape(-1, 3, 3)).as_rotvec().reshape(len(L), 66)
    np.savez(path, poses=np.concatenate([aa, np.zeros((len(L), 99))], 1), trans=root @ world.T, mocap_frame_rate=FPS)
    return path


def _worst(sc, keys) -> float:
    return max(sc[k][0] for k in keys)


def _clap(T: int = 40) -> np.ndarray:
    """Both arms swing from the sides to the front until the wrists meet in front of the chest, then hold."""
    L = np.tile(np.eye(3), (T, 22, 1, 1))
    _, P = forward_kinematics(L[:1], np.zeros((1, 3)))
    sh = {s: P[0, J[f"{s}_shoulder"]] for s in "lr"}
    arm = np.linalg.norm(REST_OFFSETS[J["l_elbow"]]) + np.linalg.norm(REST_OFFSETS[J["l_wrist"]])
    half = abs(sh["l"][0] - sh["r"][0]) / 2
    meet = (sh["l"] + sh["r"]) / 2 + np.array([0, -0.05, np.sqrt((0.97 * arm) ** 2 - half**2 - 0.05**2)])
    u = np.clip(np.arange(T) / (0.6 * T), 0, 1)
    for s in "lr":
        rest = REST_OFFSETS[J[f"{s}_elbow"]] + REST_OFFSETS[J[f"{s}_wrist"]]
        tgt = meet - sh[s]
        ax = np.cross(rest, tgt)
        ang = np.arctan2(np.linalg.norm(ax), rest @ tgt)
        L[:, J[f"{s}_shoulder"]] = Rotation.from_rotvec(ax / np.linalg.norm(ax) * (ang * u)[:, None]).as_matrix()
    return L


# ---- loading -------------------------------------------------------------------------------------------------------


def test_formats_agree(tmp_path):
    """Rotation output (Y up) and axis-angle parameters (Y up, then Z up) describe the same motion."""
    L, root = _locals(60, 0), _root(60)
    a = load(_save_rot(tmp_path / "rot.npz", L, root))
    b = load(_save_params(tmp_path / "p_y.npz", L, root), up="y")
    c = load(_save_params(tmp_path / "p_z.npz", L, root, world=BODY_TO_ROS), up="z")  # same motion, Z-up world
    for m in (b, c):
        assert np.allclose(m.global_rots, a.global_rots, atol=1e-5)
        assert np.allclose(m.joints, a.joints, atol=1e-5)
    assert np.allclose(a.rest_offsets, REST_OFFSETS, atol=1e-5)  # measured from the clip


def test_load_reads_past_a_pickled_array_it_never_touches(tmp_path):
    L, root = _locals(10, 0), _root(10)
    path = tmp_path / "rot.npz"
    G, P = forward_kinematics(L, root)
    np.savez(path, global_rot_mats=G, posed_joints=P, gender=np.array([{"a": 1}], dtype=object))
    assert len(load(path)) == 10


def test_load_refuses_a_pickled_motion(tmp_path):
    path = tmp_path / "rot.npz"
    np.savez(path, poses=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError, match="allow_pickle"):
        load(path)


def test_model_rest_skeleton(tmp_path):
    """A model file's J_regressor/v_template/shapedirs give the rest offsets of its shape."""
    joints = np.cumsum(np.r_[[np.zeros(3)], REST_OFFSETS[1:]], 0)  # any 22 points will do
    dirs = np.zeros((22, 3, 2))
    dirs[:, 1, 0] = np.linspace(0, 0.1, 22)  # beta 0 stretches along Y
    np.savez(tmp_path / "SMPLX_TEST.npz", v_template=joints, J_regressor=np.eye(22), shapedirs=dirs)
    off, pelvis = rest_offsets_from_model(tmp_path / "SMPLX_TEST.npz", np.array([1.0, 0.0]))
    want = joints + dirs[:, :, 0]
    assert np.allclose(pelvis, want[0])
    assert np.allclose(off[J["l_knee"]], want[J["l_knee"]] - want[J["l_hip"]])


# ---- rotation retarget ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["none", "ik"])
def test_rest_pose(method):
    """T-pose in: straight trunk, level collars, untwisted legs and arms. SMPL-X's rest shape (shrugged clavicles,
    sideways-kinked knees and elbows) must not turn into a pose; only its small in-plane rest bend (<8 deg) remains."""
    m = load_motion(np.tile(np.eye(3), (5, 22, 1, 1)), _root(5))
    f = retarget(m, fps=FPS, method=method).angles
    for k in ("r_waist", "y_waist", "waist", "r_spine", "y_spine", "spine", "r_chest", "y_chest", "chest", "l_y_collar", "l_p_collar", "r_y_collar", "r_p_collar", "l_y_hip", "r_y_hip", "l_y_ankle", "r_y_ankle"):
        assert np.allclose(f[k], 0.0, atol=0.03), (k, f[k][0])
    for k in ("l_elbow", "r_elbow", "l_knee", "r_knee"):
        assert np.all(np.abs(f[k]) < np.radians(8)), (k, f[k][0])


def test_rotation_matches_reference():
    """The batched rotation method and the per-frame scipy reference give the same body (bones within ~1 deg)."""
    m = load_motion(_locals(60, 7), _root(60))
    p = prepare(m)
    ref = reference.retarget_frames(m)
    ref_clip = resample(p, np.stack([ref["angles"][k] for k in WIRE], 1), FPS)
    new = retarget_batch([m], fps=FPS, method="rotation", preps=[p])[0]
    a, b = fidelity.score(m, ref_clip, prep=p), fidelity.score(m, new, prep=p)
    for k in fidelity.BENDS + fidelity.LINES:
        assert abs(a[k][0] - b[k][0]) < 1.5, (k, a[k][0], b[k][0])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rotation_limbs_match(seed):
    m = load_motion(_locals(90, seed), _root(90))
    sc = fidelity.score(m, retarget(m, fps=FPS, method="rotation"))
    assert _worst(sc, fidelity.BENDS) < 2.5  # the twist anchor trades ~1-2 deg of direction for a stable twist


# ---- IK ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2])
def test_ik_reaches_better(seed):
    """Whole-body IK puts hands and feet where the source's are (scaled to Arena's limbs), far better than the
    rotation retarget, without breaking the bones."""
    m = load_motion(_locals(90, seed), _root(90))
    rot, ik = (fidelity.score(m, retarget(m, fps=FPS, method=k)) for k in ("rotation", "ik"))
    assert _worst(ik, fidelity.REACH) < 2.5, ik
    assert _worst(ik, fidelity.REACH) < 0.5 * _worst(rot, fidelity.REACH)
    assert _worst(ik, fidelity.BENDS) < 15.0


def test_clap_closes():
    """Source wrists meet in front of the chest: the Arena hands meet too, despite its wider shoulders."""
    m = load_motion(_clap(), _root(40))
    rot, ik = (fidelity.score(m, retarget(m, fps=FPS, method=k)) for k in ("rotation", "ik"))
    assert ik["hands_contact"][0] < 4.0, ik["hands_contact"]  # % of arm length: < ~2.5 cm
    assert ik["hands_contact"][0] < 0.5 * rot["hands_contact"][0]


def test_ik_respects_hard_limits():
    m = load_motion(_locals(90, 5, scale=0.6), _root(90))
    a = retarget(m, fps=FPS).angles
    for k, (lo, hi) in HARD_LIMITS.items():
        assert a[k].min() >= lo - 1e-6 and a[k].max() <= hi + 1e-6, (k, a[k].min(), a[k].max())


def test_lying_down_keeps_shape():
    """Pelvis pitched back 90 deg (lying on the back): folded tilt keeps the limbs, the pelvis is near the floor."""
    T = 30
    L = _locals(T, 4, scale=0.2)
    L[:, 0] = Rotation.from_euler("x", -np.pi / 2).as_matrix() @ L[:, 0]  # canonical X = left: pitch onto the back
    m = load_motion(L, np.stack([np.zeros(T), np.full(T, 0.12), np.zeros(T)], 1))
    clip = retarget(m, fps=FPS)
    assert _worst(fidelity.score(m, clip), fidelity.REACH) < 3.0
    assert np.all(clip.root_z < 0.7)  # well under the standing ~0.95 (random knee bends reach below the pelvis)
    assert np.all(np.abs(np.abs(clip.angles["waist"]) - np.pi / 2) < 0.6)  # the tilt went into the waist


def test_continuity():
    """A slow full arm raise past horizontal and back: no branch flips (jumps of ~pi) between frames. Near
    arm-straight-down the shoulder azimuth swings fast by design (the pole of its sphere)."""
    T = 120
    L = np.tile(np.eye(3), (T, 22, 1, 1))
    s = np.sin(np.linspace(0, np.pi, T))
    for side, sign in (("l_shoulder", 1), ("r_shoulder", -1)):
        L[:, J[side]] = Rotation.from_euler("z", sign * (-1.3 + 2.6 * s)[:, None]).as_matrix()  # down -> overhead
    m = load_motion(L, _root(T))
    for method in ("none", "ik"):
        clip = retarget(m, fps=FPS, method=method)
        for k, v in clip.angles.items():
            assert np.abs(np.diff(v)).max() < 1.0, (method, k)
    assert _worst(fidelity.score(m, clip), fidelity.REACH) < 2.0


def test_batch_matches_alone():
    """Solving clips together gives the same bodies as solving each alone (the batch never couples clips)."""
    ms = [load_motion(_locals(60, s), _root(60)) for s in (11, 12, 13)]
    together = retarget_batch(ms, fps=FPS)
    fk = ContractFK(dtype=torch.float64)
    for m, c in zip(ms, together):
        alone = retarget(m, fps=FPS)
        pa, _ = fk(torch.as_tensor(np.stack([c.angles[k] for k in WIRE], 1)))
        pb, _ = fk(torch.as_tensor(np.stack([alone.angles[k] for k in WIRE], 1)))
        assert max(float((pa[k] - pb[k]).norm(dim=-1).max()) for k in pa) < 0.02  # m


# ---- files ---------------------------------------------------------------------------------------------------------


def test_write_and_load(tmp_path):
    m = load_motion(_locals(45, 5), _root(45))
    clip = retarget(m)
    d = write_clip(clip, tmp_path, "smplx_test", loop=True, tags=["domain::care"], note="synthetic.")
    z = np.load(d / "smplx_test.anim")
    assert sorted(str(n) for n in z["clip/joint_names"]) == sorted(WIRE) and len(WIRE) == 40
    assert z["clip/angles"].shape == (len(clip.t), 40) and z["clip/root_xy_yaw"].shape == (len(clip.t), 3)
    assert z["root/quat"].shape == (len(clip.t), 4)
    ann = yaml.safe_load((d / "annotation.yaml").read_text())
    assert ann["name"] == "smplx_test" and ann["path"] == "Animation/smplx_test" and ann["loop"] is True
    assert ann["fps"] == pytest.approx(20.0) and ann["tags"] == ["domain::care"]
    back = load_clip(d)
    assert all(np.allclose(back.angles[k], clip.angles[k]) for k in WIRE)


def test_cli_help_does_not_import_torch():
    code = "import sys\nfrom arena_simulation_setup.adapters.arena_animation_retarget.cli import _parser\n_parser().format_help()\nsys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_cli_batch(tmp_path):
    from arena_simulation_setup.adapters.arena_animation_retarget.cli import main

    src = tmp_path / "src" / "sub"
    src.mkdir(parents=True)
    _save_rot(src / "a.npz", _locals(40, 21), _root(40))
    _save_params(src / "b.npz", _locals(50, 22), _root(50), world=BODY_TO_ROS)  # Z-up parameters
    np.savez(src / "shape.npz", betas=np.zeros(10))  # not a motion: reported, the batch goes on
    out, rep = tmp_path / "out", tmp_path / "report.csv"
    rc = main([str(tmp_path / "src"), "-o", str(out), "--jobs", "2", "--report", str(rep), "-q"])
    assert rc == 1  # shape.npz failed
    assert (out / "a" / "a.anim").exists() and (out / "b" / "b.anim").exists()
    lines = rep.read_text().splitlines()
    assert len(lines) == 3 and "l_arm_reach_mean" in lines[0]
    assert main([str(src / "a.npz"), "-o", str(out), "--skip-existing", "-q"]) == 0


SAMPLE = Path(os.environ.get("ARENA_RETARGET_SAMPLE", "/opt/arena_ws/data/recordings/kimodo_compare/fidelity/src/kimodo_collapse_to_ground.npz"))


@pytest.mark.skipif(not SAMPLE.is_file(), reason="no Kimodo sample (set ARENA_RETARGET_SAMPLE)")
def test_real_kimodo_collapse():
    m = load(SAMPLE)
    clip = retarget(m, fps=m.fps)
    sc = fidelity.score(m, clip)
    assert _worst(sc, fidelity.REACH) < 5.0, sc
    assert clip.root_z[-1] < 0.35  # ends on the ground
