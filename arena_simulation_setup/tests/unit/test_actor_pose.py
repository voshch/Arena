"""shared.actor_pose: a bundle's character posed from wire angles must match what Isaac's wire pose provider draws."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from arena_assets.rig import write_rig
from arena_simulation_setup.shared.actor_pose import ActorPose, TemplatePose
from arena_simulation_setup.shared.body import DEFAULT_STANCE, SEMANTIC_DOFS, load_template, matrix_to_quat, quat_to_matrix, semantic_to_body
from arena_simulation_setup.testing import make_rig

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

    body = ActorPose(bundle)
    assert body.neutral_clip is not None
    skin_doc = convert._Collada(str(body.skin))
    joints = convert._parse_skeleton(skin_doc)
    skinned = convert._parse_skinned_meshes(skin_doc)
    neutral = convert._neutral_pose(joints, convert._parse_animations(convert._Collada(str(body.neutral_clip))))
    rig = load_rig(bundle / "rig" / f"{bundle.name}.rig")
    soles = convert._rig_soles(rig, joints, skinned)
    prov = external.ExternalPoseProvider(tuple(j.path for j in joints), np.array(neutral[0]), np.array(neutral[1]), rig)
    sole = ground.Soles(prov._bone_parent, [s["joint"] for s in soles], np.array([s["point"] for s in soles])) if soles else None
    index = {j.name: i for i, j in enumerate(joints)}
    joint_bone = {j: index[rows[0][0]] for j in rig.joints if (rows := rig.bones_of(j)) and rows[0][0] in index}
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
        lift = 0.0 if sole is None else -sole.height(pose)
        mine = body.positions(angles)
        for joint, b in joint_bone.items():
            assert mine[joint] == pytest.approx(P[b] + np.array([0.0, 0.0, lift]), abs=1e-6), joint


# A synthetic bundle in a temp dir, so ActorPose runs without real assets: the skin skeleton is Arena's template body
# (one bone per joint, rest = the template offsets), plus a `prop` bone under the right wrist that only the neutral clip
# keys, a non-joint wrapper node, heel and toe skin vertices 3 cm below the lower of each foot's two joints.

NS = "http://www.collada.org/2005/11/COLLADASchema"
PROP_REST, PROP_KEY = np.array([0.1, 0.0, 0.0]), np.array([0.2, 0.0, 0.05])
SOLES = ("left_ankle", "left_foot", "right_ankle", "right_foot")


def _m(t: np.ndarray) -> str:
    m = np.eye(4)
    m[:3, 3] = t
    return " ".join(f"{v:.9g}" for v in m.ravel())


def _rest_world() -> dict[str, np.ndarray]:
    t = load_template()
    world: dict[str, np.ndarray] = {}
    for name, parent in t.parents.items():
        world[name] = t.offsets[name] + (world[parent] if parent else 0.0)
    world["prop"] = world["right_wrist"] + PROP_REST
    return world


def _skin_dae(feet: tuple[str, ...]) -> str:
    t = load_template()
    world = _rest_world()
    children: dict[str | None, list[str]] = {}
    for name, parent in t.parents.items():
        children.setdefault(parent, []).append(name)
    children.setdefault("right_wrist", []).append("prop")
    offset = {**t.offsets, "prop": PROP_REST}

    def node(name: str) -> str:
        return f'<node id="{name}" sid="{name}" type="JOINT"><matrix>{_m(offset[name])}</matrix>{"".join(node(c) for c in children.get(name, []))}</node>'

    names = list(world)
    verts, weights = [], []
    for side in ("left", "right"):
        if side not in feet:
            continue
        ankle, foot = world[f"{side}_ankle"], world[f"{side}_foot"]
        z = min(ankle[2], foot[2]) - 0.03
        verts += [(ankle[0] - 0.05, ankle[1], z, f"{side}_ankle"), (foot[0] + 0.05, foot[1], z, f"{side}_foot"), (ankle[0], ankle[1], z + 0.2, f"{side}_ankle")]
    verts.append((0.0, 0.0, 1.0, "pelvis"))  # an unweighted vertex (vcount 0) belongs to no bone
    points = " ".join(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z, _ in verts)
    vcount = " ".join("0" if i == len(verts) - 1 else "1" for i in range(len(verts)))
    v = " ".join(f"{names.index(owner)} 0" for x, y, z, owner in verts[:-1])
    inv = " ".join(_m(-world[n]) for n in names)
    return f"""<?xml version="1.0"?><COLLADA xmlns="{NS}" version="1.4.1">
<library_geometries><geometry id="mesh"><mesh><source id="mp"><float_array id="mpa">{points}</float_array><technique_common><accessor source="#mpa" count="{len(verts)}" stride="3"/></technique_common></source>
<vertices id="mv"><input semantic="POSITION" source="#mp"/></vertices></mesh></geometry></library_geometries>
<library_controllers><controller id="c"><skin source="#mesh"><source id="jn"><Name_array>{" ".join(names)}</Name_array></source>
<source id="ib"><float_array>{inv}</float_array></source><source id="w"><float_array>1</float_array></source>
<joints><input semantic="JOINT" source="#jn"/><input semantic="INV_BIND_MATRIX" source="#ib"/></joints>
<vertex_weights><input semantic="JOINT" source="#jn" offset="0"/><input semantic="WEIGHT" source="#w" offset="1"/><vcount>{vcount}</vcount><v>{v}</v></vertex_weights></skin></controller></library_controllers>
<library_visual_scenes><visual_scene id="s"><node id="Armature">{node("pelvis")}</node></visual_scene></library_visual_scenes></COLLADA>"""


def _clip_dae() -> str:
    return f"""<?xml version="1.0"?><COLLADA xmlns="{NS}" version="1.4.1"><library_animations>
<animation><source id="o"><float_array>{_m(PROP_KEY)} {_m(PROP_KEY)}</float_array></source><sampler id="s"><input semantic="OUTPUT" source="#o"/></sampler>
<channel source="#s" target="prop/transform"/></animation>
<animation><source id="o2"><float_array>0</float_array></source><sampler id="s2"><input semantic="OUTPUT" source="#o2"/></sampler>
<channel source="#s2" target="prop/rotateX.ANGLE"/></animation></library_animations></COLLADA>"""


def _synthetic(root: Path, *, clip: str | None = "neutral", feet: tuple[str, ...] = ("left", "right"), spread: bool = False, drivers=None) -> Path:
    t = load_template()
    bundle = root / "synth"
    (bundle / "rig" / "meshes").mkdir(parents=True)
    (bundle / "rig" / "clips").mkdir()
    (bundle / "rig" / "meshes" / "synth.dae").write_text(_skin_dae(feet))
    animation = ""
    if clip is not None:
        (bundle / "rig" / "clips" / f"{clip}.dae").write_text(_clip_dae())
        animation = f'<animation name="{clip}"><filename>clips/{clip}.dae</filename></animation>'
    (bundle / "rig" / "synth.sdf").write_text(f"<sdf><actor><skin><filename>meshes/synth.dae</filename></skin>{animation}</actor></sdf>")
    rows = {n: (p, [(n, 1.0, (0.0, 0.0, 0.0, 1.0))]) for n, p in t.parents.items()}
    if spread:  # the wrist joint turns its own bone halfway and the prop bone all the way
        rows["right_wrist"] = ("right_elbow", [("right_wrist", 0.5, (0.0, 0.0, 0.0, 1.0)), ("prop", 0.5, (0.0, 0.0, 0.0, 1.0))])
    stance = {n: matrix_to_quat(DEFAULT_STANCE[n]) for n in t.parents if n in DEFAULT_STANCE}
    write_rig(bundle / "rig" / "synth.rig", make_rig(rows, stance=stance, soles=SOLES, drivers=drivers))
    return bundle


def test_a_synthetic_character_on_the_template_skeleton_poses_like_the_template(tmp_path: Path) -> None:
    body, template = ActorPose(_synthetic(tmp_path)), TemplatePose()
    assert template.positions({}).keys() <= body.positions({}).keys()
    for seed in range(4):
        angles = _random_angles(seed, scale=0.4) if seed else {}
        mine, ref = body.positions(angles), template.positions(angles)
        for joint in ref:  # the same skeleton and rotations; only how each stands on the floor differs
            assert mine[joint] - mine["pelvis"] == pytest.approx(ref[joint] - ref["pelvis"], abs=1e-9), joint
    assert body.positions({})["pelvis"] == pytest.approx(template.positions({})["pelvis"], abs=1e-9)  # soles 3 cm below the feet in both
    R, P, lift = body.bone_worlds(_random_angles(7))
    assert min(float((R[j] @ p + P[j])[2]) for j, p in body._soles) + lift == pytest.approx(0.0, abs=1e-12)
    assert len(body._soles) == 4


def test_bones_outside_the_rig_hold_the_neutral_clip(tmp_path: Path) -> None:
    for clip, offset in (("neutral", PROP_KEY), ("idle", PROP_KEY), (None, PROP_REST)):
        root = tmp_path / str(clip)
        body = ActorPose(_synthetic(root, clip=clip))
        names = [b.name for b in body._bones]
        R, P, _ = body.bone_worlds(_random_angles(3))
        wrist, prop = names.index("right_wrist"), names.index("prop")
        assert P[prop] == pytest.approx(P[wrist] + R[wrist] @ offset, abs=1e-9), clip


def test_a_joint_spread_over_two_bones_turns_the_first_part_way_and_the_last_fully(tmp_path: Path) -> None:
    plain, split = ActorPose(_synthetic(tmp_path / "plain")), ActorPose(_synthetic(tmp_path / "split", spread=True))
    names = [b.name for b in plain._bones]
    angles = {"r_wrist": 0.6, "r_r_wrist": 0.5}
    (Rp, _, _), (Rs, _, _) = plain.bone_worlds(angles), split.bone_worlds(angles)
    elbow, wrist, prop = names.index("right_elbow"), names.index("right_wrist"), names.index("prop")
    assert Rs[prop] == pytest.approx(Rp[wrist], abs=1e-9)

    def angle(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))

    assert angle(Rs[elbow], Rs[wrist]) == pytest.approx(0.5 * angle(Rp[elbow], Rp[wrist]), abs=1e-9)


def test_a_foot_without_skin_vertices_and_a_distance_driven_rig_get_no_sole_points(tmp_path: Path) -> None:
    assert len(ActorPose(_synthetic(tmp_path / "one", feet=("left",)))._soles) == 2
    wheels = ActorPose(_synthetic(tmp_path / "wheel", drivers={"wheel": {"bone": "pelvis", "axis": (0.0, 1.0, 0.0), "from": "distance"}}))
    assert wheels._soles == []
    assert wheels.positions({})["pelvis"] == pytest.approx(load_template().offsets["pelvis"], abs=1e-9)  # stays on its skeleton root


def test_a_broken_bundle_is_refused(tmp_path: Path) -> None:
    bundle = _synthetic(tmp_path)
    sdf, rig, skin = bundle / "rig" / "synth.sdf", bundle / "rig" / "synth.rig", bundle / "rig" / "meshes" / "synth.dae"
    good_sdf, good_skin = sdf.read_text(), skin.read_text()
    sdf.write_text("<sdf><model/></sdf>")
    with pytest.raises(ValueError, match="no <actor>"):
        ActorPose(bundle)
    sdf.write_text(good_sdf)
    skin.write_text(good_skin.replace("<visual_scene", "<other_scene").replace("</visual_scene>", "</other_scene>"))
    with pytest.raises(ValueError, match="no visual scene"):
        ActorPose(bundle)
    skin.write_text(good_skin.replace(f"<matrix>{_m(PROP_REST)}</matrix>", ""))
    with pytest.raises(ValueError, match="has no matrix"):
        ActorPose(bundle)
    skin.write_text(good_skin)
    rig.unlink()
    with pytest.raises(ValueError, match="no rig"):
        ActorPose(bundle)
