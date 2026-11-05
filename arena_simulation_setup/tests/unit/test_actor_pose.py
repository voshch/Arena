"""shared.actor_pose: a bundle's character posed from wire angles must match what Isaac's wire pose provider draws."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from arena_simulation_setup.shared.actor_pose import ActorPose, TemplatePose
from arena_simulation_setup.shared.body import SEMANTIC_DOFS, quat_to_matrix, semantic_to_body

REPO = Path(__file__).resolve().parents[3]
BUNDLES = sorted(p.parent.parent for p in REPO.glob("_assets/*/*/Human/*/rig/*.rig") if p.stem == p.parent.parent.name)


@pytest.fixture(scope="module")
def bundle() -> Path:
    if not BUNDLES:
        pytest.skip("no human bundle with a rig in this checkout's _assets")
    return next((b for b in BUNDLES if b.name == "patient_male_caucasian_old"), BUNDLES[0])


def _random_angles(seed: int, scale: float = 0.6) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {dof: float(rng.uniform(-scale, scale)) for dof in SEMANTIC_DOFS}


def test_the_template_body_stands_on_the_floor() -> None:
    pos = TemplatePose().positions({})
    assert min(pos[j][2] for j in ("left_ankle", "right_ankle", "left_foot", "right_foot")) == pytest.approx(0.03, abs=1e-9)


def test_the_character_stands_on_its_lowest_sole_point_in_any_pose(bundle: Path) -> None:
    body = ActorPose(bundle)
    assert body._soles, "a rig with sole joints gives heel and toe points"
    for seed in range(5):
        R, P, lift = body.bone_worlds(_random_angles(seed))
        lowest = min(float((R[j] @ p + P[j])[2]) for j, p in body._soles) + lift
        assert lowest == pytest.approx(0.0, abs=1e-9)


def test_positions_match_isaacs_wire_pose_provider(bundle: Path) -> None:
    sys.path.insert(0, str(REPO / "arena_isaac" / "arena_isaac"))
    convert = pytest.importorskip("peds.convert")
    external = pytest.importorskip("peds.providers.external")
    ground = pytest.importorskip("peds.ground")
    from arena_assets.rig import load_rig

    name = bundle.name
    skin_doc = convert._Collada(str(bundle / "rig" / "meshes" / f"{name}.dae"))
    joints = convert._parse_skeleton(skin_doc)
    skinned = convert._parse_skinned_meshes(skin_doc)
    clip = bundle / "rig" / "clips" / ("neutral.dae" if (bundle / "rig" / "clips" / "neutral.dae").is_file() else "idle.dae")
    neutral = convert._neutral_pose(joints, convert._parse_animations(convert._Collada(str(clip))))
    rig = load_rig(bundle / "rig" / f"{name}.rig")
    soles = convert._rig_soles(rig, joints, skinned)
    prov = external.ExternalPoseProvider(tuple(j.path for j in joints), np.array(neutral[0]), np.array(neutral[1]), rig)
    sole = ground.Soles(prov._bone_parent, [s["joint"] for s in soles], np.array([s["point"] for s in soles]))
    index = {j.name: i for i, j in enumerate(joints)}
    joint_bone = {j: index[rows[0][0]] for j in rig.joints if (rows := rig.bones_of(j)) and rows[0][0] in index}
    body = ActorPose(bundle)
    clock = 0.0
    for seed in range(10):
        angles = _random_angles(seed)
        rot = semantic_to_body(angles, body.stance)
        for _ in range(3):  # the same frame until the provider's interpolation delay sees only it, at full weight
            clock += 0.05
            prov.push(clock, list(rot), [tuple(q) for q in rot.values()], (0.0, 0.0, 0.0), [], [])
            pose = prov.evaluate(clock, 1.0)
        W, P = [None] * len(joints), [None] * len(joints)
        for i, parent in enumerate(prov._bone_parent):
            R = quat_to_matrix(pose.rotations[i])
            W[i], P[i] = (R, np.array(pose.translations[i])) if parent < 0 else (W[parent] @ R, P[parent] + W[parent] @ np.array(pose.translations[i]))
        lift = -sole.height(pose)
        mine = body.positions(angles)
        for joint, b in joint_bone.items():
            assert mine[joint] == pytest.approx(P[b] + np.array([0.0, 0.0, lift]), abs=1e-6), joint
