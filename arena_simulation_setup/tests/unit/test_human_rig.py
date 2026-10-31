from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_assets.utils.human import read_neutral, read_skin
from arena_simulation_setup.shared.body import BODY_JOINTS, PARENTS, Template, axis_rotation, forward, load_template, matrix_to_quat, quat_to_matrix, semantic_to_body, stance_of
from arena_assets.rig import Rig, load_rig
from arena_simulation_setup.tree.assets.human_rig import CMU_MB, GAME_ENGINE, build_rig, leg_chains, main, neutral_clip, rig_map

TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "body" / "template.yaml"
CHILD = {"pelvis": "spine1", "spine3": "neck", "left_wrist": "left_middle1", "right_wrist": "right_middle1"}
OLD_RIG = {
    "skeleton": "cmu",
    "neutral": {"clip": "neutral", "frame": 0},
    "rest": {"clip": "idle", "frame": 0},
    "root_offset": [0.0, 0.0, -0.3316],
    "drivers": {"l_wheel": {"bone": "WheelL", "from": "distance", "radius": 0.3, "lateral": 0.288, "axis": [0.0, 1.0, 0.0]}},
    "clips": {"idle": "clips/idle.dae"},
}


_SDF = '<sdf><actor name="alice"><skin><filename>meshes/alice.dae</filename></skin><animation name="idle"><filename>clips/idle.dae</filename></animation></actor></sdf>'


@pytest.fixture(scope="module")
def template() -> Template:
    return load_template(TEMPLATE)


def _matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


def _bind(template: Template, seed: int, bone_map: dict = CMU_MB) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Heads and world rotations of every bone of a map: the template under a random pose, each bone with its own roll, plus a wheel bone."""
    rng = np.random.default_rng(seed)
    pose = {joint: matrix_to_quat(axis_rotation(rng.normal(size=3), float(rng.uniform(-0.5, 0.5)))) for joint in bone_map}
    positions, frames = forward(template, pose)
    heads: dict[str, np.ndarray] = {}
    turns: dict[str, np.ndarray] = {}
    for joint, chain in bone_map.items():
        for k, (bone, _) in enumerate(chain):
            heads[bone] = positions[joint] + 0.5 * k * (positions["head"] - positions[joint])
            turns[bone] = frames[joint] @ axis_rotation(rng.normal(size=3), float(rng.uniform(-math.pi, math.pi)))
    heads["WheelL"] = np.array([0.0, 0.3, 0.3])
    turns["WheelL"] = np.eye(3)
    return heads, turns


def _bone_parents(template: Template, heads: dict[str, np.ndarray], bone_map: dict = CMU_MB) -> dict[str, str | None]:
    """Each bone's parent bone: chains in joint order, a joint's first bone under its parent joint's last, extra bones under the pelvis."""
    parents: dict[str, str | None] = {}
    for joint, chain in bone_map.items():
        parent = template.parents[joint]
        above = bone_map[parent][-1][0] if parent is not None else None
        for bone, _ in chain:
            if bone in heads:
                parents[bone] = above
                above = bone
    return parents | {bone: bone_map["pelvis"][0][0] for bone in heads if bone not in parents}


def _scene(heads: dict[str, np.ndarray], turns: dict[str, np.ndarray], parents: dict[str, str | None]) -> str:
    """Nested joint nodes whose local transforms give the bones' world transforms."""

    def node(bone: str) -> str:
        parent = parents[bone]
        world = _matrix(turns[bone], heads[bone])
        local = world if parent is None else np.linalg.inv(_matrix(turns[parent], heads[parent])) @ world
        children = "".join(node(child) for child, above in parents.items() if above == bone)
        values = " ".join(f"{v:.12g}" for v in local.reshape(-1))
        return f'<node id="{bone}" sid="{bone}" type="JOINT"><matrix sid="transform">{values}</matrix>{children}</node>'

    return f'<library_visual_scenes><visual_scene id="scene">{node(next(bone for bone, above in parents.items() if above is None))}</visual_scene></library_visual_scenes>'


def _write_clip(path: Path, keys: dict[str, np.ndarray]) -> None:
    """A COLLADA clip of one key per joint."""
    channels = ""
    for k, (bone, local) in enumerate(keys.items()):
        values = " ".join(f"{v:.12g}" for v in local.reshape(-1))
        channels += (
            f'<animation id="a{k}"><source id="in{k}"><float_array id="ia{k}" count="1">0</float_array></source>'
            f'<source id="out{k}"><float_array id="oa{k}" count="16">{values}</float_array></source>'
            f'<sampler id="s{k}"><input semantic="INPUT" source="#in{k}"/><input semantic="OUTPUT" source="#out{k}"/></sampler>'
            f'<channel source="#s{k}" target="{bone}/transform"/></animation>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><library_animations>{channels}</library_animations></COLLADA>')


def _write_skin(path: Path, heads: dict[str, np.ndarray], turns: dict[str, np.ndarray], parents: dict[str, str | None]) -> None:
    """A COLLADA skin with one vertex per bone, each bone's inverse bind in centimeters to exercise the scale."""
    names = list(heads)
    scale = np.diag([100.0, 100.0, 100.0, 1.0])
    binds = [np.linalg.inv(_matrix(turns[n], heads[n]) @ scale) for n in names]
    floats = " ".join(f"{v:.12g}" for b in binds for v in b.reshape(-1))
    points = " ".join(f"{v:.9g}" for n in names for v in heads[n])
    pairs = " ".join(f"{i} {i}" for i in range(len(names)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_geometries><geometry id="body-mesh"><mesh>
    <source id="body-positions"><float_array id="pa" count="{3 * len(names)}">{points}</float_array>
      <technique_common><accessor source="#pa" count="{len(names)}" stride="3"/></technique_common></source>
    <vertices id="body-vertices"><input semantic="POSITION" source="#body-positions"/></vertices>
  </mesh></geometry></library_geometries>
  <library_controllers><controller id="body-skin"><skin source="#body-mesh">
    <bind_shape_matrix>1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1</bind_shape_matrix>
    <source id="joints"><Name_array id="ja" count="{len(names)}">{" ".join(names)}</Name_array></source>
    <source id="binds"><float_array id="ba" count="{16 * len(names)}">{floats}</float_array>
      <technique_common><accessor source="#ba" count="{len(names)}" stride="16"/></technique_common></source>
    <source id="weights"><float_array id="wa" count="{len(names)}">{" ".join("1" for _ in names)}</float_array>
      <technique_common><accessor source="#wa" count="{len(names)}" stride="1"/></technique_common></source>
    <joints><input semantic="JOINT" source="#joints"/><input semantic="INV_BIND_MATRIX" source="#binds"/></joints>
    <vertex_weights count="{len(names)}"><input semantic="JOINT" source="#joints" offset="0"/><input semantic="WEIGHT" source="#weights" offset="1"/>
      <vcount>{" ".join("1" for _ in names)}</vcount><v>{pairs}</v></vertex_weights>
  </skin></controller></library_controllers>
  {_scene(heads, turns, parents)}
</COLLADA>
"""
    )


def _bundle(root: Path, template: Template, seed: int = 3, rig: dict | None = None, bone_map: dict = CMU_MB) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    bundle = root / "alice"
    heads, turns = _bind(template, seed, bone_map)
    _write_skin(bundle / "meshes" / "alice.dae", heads, turns, _bone_parents(template, heads, bone_map))
    (bundle / "alice.sdf").write_text(_SDF)
    if rig is not None:
        (bundle / "rig.yaml").write_text(yaml.safe_dump(rig))
    return bundle, heads, turns


def _children(template: Template, bone_map: dict = CMU_MB) -> dict[str, str]:
    out: dict[str, str] = {}
    for joint, parent in template.parents.items():
        if parent is not None and joint in bone_map:
            out.setdefault(parent, CHILD.get(parent, joint))
    return out


def _down(heads: dict[str, np.ndarray], bone: str) -> np.ndarray:
    """The unit direction from a leg bone's head to its child's head."""
    child = {"LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "thigh_l": "calf_l", "calf_l": "foot_l", "thigh_r": "calf_r", "calf_r": "foot_r"}[bone]
    along = heads[child] - heads[bone]
    return along / np.linalg.norm(along)


def _written(bundle: Path, name: str = "alice") -> Rig:
    rig = load_rig(bundle / "rig" / f"{name}.rig")
    assert rig is not None
    return rig


def test_build_rig_carries_the_bundle_settings_and_drivers_over_the_joints(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template, rig=OLD_RIG)
    rig = build_rig(bundle, template)
    assert rig.settings == {"skeleton": "smplx", "source_rig": "cmu_mb", "neutral": OLD_RIG["neutral"], "rest": OLD_RIG["rest"], "root_offset": OLD_RIG["root_offset"]}
    assert rig.joints == BODY_JOINTS
    assert rig.sole_names == ("left_ankle", "left_foot", "right_ankle", "right_foot")
    assert rig.parent_names == {joint: template.parents[joint] for joint in BODY_JOINTS}
    (driver,) = rig.drivers
    assert (driver.name, driver.bone, driver.axis.tolist()) == ("l_wheel", "WheelL", [0.0, 1.0, 0.0])
    assert driver.settings == {"from": "distance", "radius": 0.3, "lateral": 0.288}


def test_a_bundle_without_a_rig_file_gets_no_drivers_and_only_the_derived_settings(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template)
    rig = build_rig(bundle, template)
    assert rig.drivers == ()
    assert rig.settings == {"skeleton": "smplx", "source_rig": "cmu_mb"}


def test_the_neck_spreads_evenly_over_both_neck_bones(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template)
    neck = build_rig(bundle, template).bones_of("neck")
    assert [(bone, weight) for bone, weight, _ in neck] == [("Neck", 0.5), ("Neck1", 0.5)]


@pytest.mark.parametrize("seed", [3, 17])
def test_written_offsets_put_every_bone_on_its_template_direction(tmp_path: Path, template: Template, seed: int) -> None:
    bundle, heads, turns = _bundle(tmp_path, template, seed=seed)
    assert main([str(bundle)]) == 0
    rig = _written(bundle)
    joint_head = {joint: heads[chain[0][0]] for joint, chain in CMU_MB.items()}
    for joint, child in _children(template).items():
        want = template.offsets[child] / np.linalg.norm(template.offsets[child])
        for bone, _, offset in rig.bones_of(joint):
            local = turns[bone].T @ (joint_head[child] - joint_head[joint])
            assert quat_to_matrix(offset) @ (local / np.linalg.norm(local)) == pytest.approx(want, abs=1e-5)


def test_all_semantic_dofs_at_zero_pose_every_bone_in_the_neutral_clip_stance(tmp_path: Path, template: Template) -> None:
    bundle, heads, turns = _bundle(tmp_path, template, seed=7)
    (bundle / "alice.sdf").write_text('<sdf><actor><animation name="idle"><filename>clips/idle.dae</filename></animation></actor></sdf>')
    rng = np.random.default_rng(8)
    parents = _bone_parents(template, heads)
    keys = {}
    for bone in ("LeftArm", "RightForeArm", "Spine", "Head", "LeftShoulder"):
        above = parents[bone]
        bent = turns[bone] @ axis_rotation(rng.normal(size=3), 0.4)
        keys[bone] = np.linalg.inv(_matrix(turns[above], heads[above])) @ _matrix(bent, heads[bone])
    _write_clip(bundle / "clips" / "idle.dae", keys)
    assert neutral_clip(bundle, {}) == bundle / "clips" / "idle.dae"
    rig = build_rig(bundle, template)
    neutral = read_neutral(bundle / "meshes" / "alice.dae", bundle / "clips" / "idle.dae")
    _, frames = forward(template, semantic_to_body({}, stance_of(rig)))
    for joint in rig.joints:
        bone, _, offset = rig.bones_of(joint)[-1]
        assert frames[joint] @ quat_to_matrix(offset) == pytest.approx(neutral[bone], abs=1e-4)
    for bone in ("LeftUpLeg", "LeftLeg"):
        assert neutral[bone] @ (turns[bone].T @ np.array(_down(heads, bone))) == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_a_rig_names_its_neutral_clip_over_the_idle_one(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template)
    for name in ("idle", "stand"):
        _write_clip(bundle / "clips" / f"{name}.dae", {})
    assert neutral_clip(bundle, {"neutral": {"clip": "stand", "frame": 0}, "clips": {"idle": "clips/idle.dae", "stand": "clips/stand.dae"}}) == bundle / "clips" / "stand.dae"
    (bundle / "alice.sdf").unlink()
    assert neutral_clip(bundle, {"clips": {"walk": "clips/idle.bvh"}}) is None


def test_dry_run_prints_the_rig_and_leaves_the_bundle(tmp_path: Path, template: Template, capsys: pytest.CaptureFixture[str]) -> None:
    bundle, _, _ = _bundle(tmp_path, template, rig=OLD_RIG)
    before = _snapshot(bundle)
    assert main([str(bundle), "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["joints"]["left_hip"] == "pelvis" and "LeftUpLeg" in {row["bone"] for row in printed["bones"]}
    assert _snapshot(bundle) == before


def _snapshot(bundle: Path) -> dict[str, bytes]:
    return {str(path.relative_to(bundle)): path.read_bytes() for path in sorted(bundle.rglob("*")) if path.is_file()}


def test_rig_human_adds_only_the_rig_directory_and_leaves_the_old_files_unchanged(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template, rig=OLD_RIG)
    _write_clip(bundle / "clips" / "idle.dae", {})
    before = _snapshot(bundle)
    assert main([str(bundle)]) == 0
    after = _snapshot(bundle)
    assert {key: after[key] for key in before} == before
    assert sorted(set(after) - set(before)) == ["rig/alice.rig", "rig/alice.sdf"]
    assert (bundle / "rig.yaml").read_text() == yaml.safe_dump(OLD_RIG)


def test_the_rig_sdf_points_at_the_bundle_skin_and_clips_one_directory_up(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template)
    assert main([str(bundle)]) == 0
    root = ET.parse(bundle / "rig" / "alice.sdf").getroot()
    assert root.findtext("actor/skin/filename") == "../meshes/alice.dae"
    assert root.findtext("actor/animation/filename") == "../clips/idle.dae"
    assert root.find("actor").get("name") == "alice"


def test_a_bundle_with_its_own_rig_skin_gets_the_rig_but_keeps_its_rig_sdf(tmp_path: Path, template: Template) -> None:
    bundle, heads, turns = _bundle(tmp_path, template)
    _write_skin(bundle / "rig" / "meshes" / "alice.dae", heads, turns, _bone_parents(template, heads))
    (bundle / "rig" / "alice.sdf").write_text("<sdf>mine</sdf>")
    assert main([str(bundle)]) == 0
    assert (bundle / "rig" / "alice.sdf").read_text() == "<sdf>mine</sdf>"
    assert _written(bundle).joints == BODY_JOINTS


def test_a_skin_missing_a_cmu_mb_bone_is_refused_naming_it(tmp_path: Path, template: Template, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = tmp_path / "bob"
    heads, turns = _bind(template, 5)
    del heads["LeftToeBase"], turns["LeftToeBase"]
    _write_skin(bundle / "meshes" / "bob.dae", heads, turns, _bone_parents(template, heads))
    with pytest.raises(SystemExit) as exit_info:
        main([str(bundle)])
    assert exit_info.value.code == 2
    assert "LeftToeBase" in capsys.readouterr().err
    assert not (bundle / "rig").exists()


def test_a_directory_without_a_skin_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "carol").mkdir()
    with pytest.raises(SystemExit) as exit_info:
        main([str(tmp_path / "carol")])
    assert exit_info.value.code == 2
    assert "meshes/carol.dae" in capsys.readouterr().err


def test_the_mcp_tool_writes_the_same_rig_as_the_command(tmp_path: Path, template: Template) -> None:
    assets = pytest.importorskip("arena_simulation_setup.mcp.assets")
    bundle, _, _ = _bundle(tmp_path, template, rig=OLD_RIG)
    before = _snapshot(bundle)
    previewed = assets.assets_rig_human(str(bundle), dry_run=True)
    assert previewed["written"] == []
    assert _snapshot(bundle) == before
    written = assets.assets_rig_human(str(bundle))
    assert written["path"] == str(bundle / "rig" / "alice.rig") and len(written["written"]) == 2
    by_tool = (bundle / "rig" / "alice.rig").read_bytes()
    (bundle / "rig" / "alice.rig").unlink()
    assert main([str(bundle)]) == 0
    assert (bundle / "rig" / "alice.rig").read_bytes() == by_tool
    assert written["rig"] == previewed["rig"]


def test_the_mcp_tool_refuses_a_missing_bundle_and_a_missing_skin(tmp_path: Path) -> None:
    assets = pytest.importorskip("arena_simulation_setup.mcp.assets")
    from arena_mcp.core import ToolError

    with pytest.raises(ToolError, match="no such bundle"):
        assets.assets_rig_human(str(tmp_path / "nobody"))
    (tmp_path / "carol").mkdir()
    with pytest.raises(ToolError, match="meshes/carol.dae"):
        assets.assets_rig_human(str(tmp_path / "carol"))


def test_the_bare_command_prints_its_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "usage: rig_human" in err and "--dry-run" in err


def test_the_map_follows_the_bones_the_skin_has(tmp_path: Path, template: Template) -> None:
    cmu, _, _ = _bundle(tmp_path / "cmu", template)
    assert rig_map(read_skin(cmu / "meshes" / "alice.dae").frames) == "cmu_mb"
    engine, _, _ = _bundle(tmp_path / "engine", template, bone_map=GAME_ENGINE)
    assert rig_map(read_skin(engine / "meshes" / "alice.dae").frames) == "game_engine"


def test_a_game_engine_skin_gets_the_body_and_all_30_digit_joints(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template, bone_map=GAME_ENGINE)
    rig = build_rig(bundle, template)
    assert rig.joints == tuple(joint for joint in template.parents if joint in GAME_ENGINE)
    assert len(rig.joints) == len(rig.stance) == 52
    assert rig.parent_names == {joint: PARENTS[joint] for joint in GAME_ENGINE}
    assert [bone for bone, _, _ in rig.bones_of("neck")] == ["neck_01"]
    assert rig.bones_of("right_pinky3")[0][0] == "pinky_03_r"
    assert rig.bones_of("left_thumb1")[0][0] == "thumb_01_l"


@pytest.mark.parametrize("seed", [3, 17])
def test_game_engine_offsets_put_every_bone_on_its_template_direction(tmp_path: Path, template: Template, seed: int) -> None:
    bundle, heads, turns = _bundle(tmp_path, template, seed=seed, bone_map=GAME_ENGINE)
    assert main([str(bundle)]) == 0
    rig = _written(bundle)
    joint_head = {joint: heads[chain[0][0]] for joint, chain in GAME_ENGINE.items()}
    for joint, child in _children(template, GAME_ENGINE).items():
        want = template.offsets[child] / np.linalg.norm(template.offsets[child])
        for bone, _, offset in rig.bones_of(joint):
            local = turns[bone].T @ (joint_head[child] - joint_head[joint])
            assert quat_to_matrix(offset) @ (local / np.linalg.norm(local)) == pytest.approx(want, abs=1e-5)


def test_a_game_engine_bind_pose_without_a_clip_is_the_zero_pose_stance(tmp_path: Path, template: Template) -> None:
    bundle, _, _ = _bundle(tmp_path, template, seed=7, bone_map=GAME_ENGINE)
    assert neutral_clip(bundle, {}) is None
    rig = build_rig(bundle, template)
    neutral = read_neutral(bundle / "meshes" / "alice.dae", None, *leg_chains(GAME_ENGINE))
    _, frames = forward(template, semantic_to_body({}, stance_of(rig)))
    for joint in rig.joints:
        bone, _, offset = rig.bones_of(joint)[-1]
        assert frames[joint] @ quat_to_matrix(offset) == pytest.approx(neutral[bone], abs=1e-4)


def test_a_game_engine_rig_hangs_its_legs_straight_down_whatever_the_idle_clip_bends(tmp_path: Path, template: Template) -> None:
    bundle, heads, turns = _bundle(tmp_path, template, seed=7, bone_map=GAME_ENGINE)
    parents = _bone_parents(template, heads, GAME_ENGINE)
    keys = {}
    for bone in ("thigh_l", "calf_l", "thigh_r", "calf_r"):
        above = parents[bone]
        assert above is not None
        bent = turns[bone] @ axis_rotation(np.array([0.0, 1.0, 0.0]), 0.4)
        keys[bone] = np.linalg.inv(_matrix(turns[above], heads[above])) @ _matrix(bent, heads[bone])
    _write_clip(bundle / "clips" / "idle.dae", keys)
    rig = build_rig(bundle, template)
    _, frames = forward(template, semantic_to_body({}, stance_of(rig)))
    for joint, bone in (("left_hip", "thigh_l"), ("left_knee", "calf_l"), ("right_hip", "thigh_r"), ("right_knee", "calf_r")):
        world = frames[joint] @ quat_to_matrix(rig.bones_of(joint)[-1][2])
        assert world @ (turns[bone].T @ _down(heads, bone)) == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_a_game_engine_skin_missing_a_finger_bone_is_refused_naming_it(tmp_path: Path, template: Template, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = tmp_path / "bob"
    heads, turns = _bind(template, 5, GAME_ENGINE)
    del heads["ring_02_l"], turns["ring_02_l"]
    _write_skin(bundle / "meshes" / "bob.dae", heads, turns, _bone_parents(template, heads, GAME_ENGINE))
    with pytest.raises(SystemExit) as exit_info:
        main([str(bundle)])
    assert exit_info.value.code == 2
    assert "ring_02_l" in capsys.readouterr().err
