"""rig_human: the .rig (arena_runtime humans/BODY.md, Section 5) and rig/ SDF of a human bundle, from its skin's bind skeleton."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from arena_assets.rig import RIG_SUFFIX, Rig, RigDriver, write_rig
from arena_assets.utils.human import read_neutral, read_skin
from arena_simulation_setup.shared.body import FINGERS, Template, load_template, matrix_to_quat, quat_to_matrix, rest_offsets

RIG_YAML = "rig.yaml"
SUB_BUNDLE = "rig"
SKELETON = "smplx"
SOLES = ("left_ankle", "left_foot", "right_ankle", "right_foot")
CMU_MB: dict[str, tuple[tuple[str, float], ...]] = {
    "pelvis": (("Hips", 1.0),),
    "left_hip": (("LeftUpLeg", 1.0),),
    "right_hip": (("RightUpLeg", 1.0),),
    "spine1": (("LowerBack", 1.0),),
    "left_knee": (("LeftLeg", 1.0),),
    "right_knee": (("RightLeg", 1.0),),
    "spine2": (("Spine", 1.0),),
    "left_ankle": (("LeftFoot", 1.0),),
    "right_ankle": (("RightFoot", 1.0),),
    "spine3": (("Spine1", 1.0),),
    "left_foot": (("LeftToeBase", 1.0),),
    "right_foot": (("RightToeBase", 1.0),),
    "neck": (("Neck", 0.5), ("Neck1", 0.5)),
    "left_collar": (("LeftShoulder", 1.0),),
    "right_collar": (("RightShoulder", 1.0),),
    "head": (("Head", 1.0),),
    "left_shoulder": (("LeftArm", 1.0),),
    "right_shoulder": (("RightArm", 1.0),),
    "left_elbow": (("LeftForeArm", 1.0),),
    "right_elbow": (("RightForeArm", 1.0),),
    "left_wrist": (("LeftHand", 1.0),),
    "right_wrist": (("RightHand", 1.0),),
}
GAME_ENGINE: dict[str, tuple[tuple[str, float], ...]] = {
    "pelvis": (("pelvis", 1.0),),
    "left_hip": (("thigh_l", 1.0),),
    "right_hip": (("thigh_r", 1.0),),
    "spine1": (("spine_01", 1.0),),
    "left_knee": (("calf_l", 1.0),),
    "right_knee": (("calf_r", 1.0),),
    "spine2": (("spine_02", 1.0),),
    "left_ankle": (("foot_l", 1.0),),
    "right_ankle": (("foot_r", 1.0),),
    "spine3": (("spine_03", 1.0),),
    "left_foot": (("ball_l", 1.0),),
    "right_foot": (("ball_r", 1.0),),
    "neck": (("neck_01", 1.0),),
    "left_collar": (("clavicle_l", 1.0),),
    "right_collar": (("clavicle_r", 1.0),),
    "head": (("head", 1.0),),
    "left_shoulder": (("upperarm_l", 1.0),),
    "right_shoulder": (("upperarm_r", 1.0),),
    "left_elbow": (("lowerarm_l", 1.0),),
    "right_elbow": (("lowerarm_r", 1.0),),
    "left_wrist": (("hand_l", 1.0),),
    "right_wrist": (("hand_r", 1.0),),
    **{f"{side}_{digit}{k}": ((f"{digit}_0{k}_{s}", 1.0),) for s, side in (("l", "left"), ("r", "right")) for digit in FINGERS for k in (1, 2, 3)},
}
RIG_MAPS: dict[str, dict[str, tuple[tuple[str, float], ...]]] = {"cmu_mb": CMU_MB, "game_engine": GAME_ENGINE}
_KEPT = ("neutral", "rest", "root_offset")


def skin_path(bundle: Path) -> Path:
    """The bundle's skin, rig/meshes/<bundle name>.dae when the sub-bundle has one, else meshes/<bundle name>.dae."""
    for path in (bundle / SUB_BUNDLE / "meshes" / f"{bundle.name}.dae", bundle / "meshes" / f"{bundle.name}.dae"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"{bundle}: no skin at meshes/{bundle.name}.dae, a human bundle carries its skin as meshes/<bundle name>.dae")


def read_settings(bundle: Path) -> dict:
    """The bundle's rig.yaml, empty without one."""
    path = bundle / RIG_YAML
    doc = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: a rig file is a mapping, got {type(doc).__name__}")
    return doc


def neutral_clip(bundle: Path, rig: Mapping[str, object]) -> Path | None:
    """The COLLADA clip of the neutral stance beside the skin: rig.yaml `neutral`, else the actor's neutral or idle animation, None without one."""
    sub = bundle / SUB_BUNDLE
    root = sub if (sub / "meshes" / f"{bundle.name}.dae").is_file() else bundle
    clips: dict[str, str] = {}
    sdf = root / f"{bundle.name}.sdf"
    if sdf.is_file():
        clips = {animation.get("name", ""): animation.findtext("filename", "") for animation in ET.parse(sdf).getroot().iter("animation")}
    declared = rig.get("clips")
    if root == bundle and isinstance(declared, Mapping):
        clips |= {str(name): str(file) for name, file in declared.items()}
    neutral = rig.get("neutral")
    named = neutral.get("clip") if isinstance(neutral, Mapping) else None
    for name in (named, "neutral", "idle"):
        file = clips.get(str(name), "")
        if file.endswith(".dae") and (root / file).is_file():
            return root / file
    return None


def rig_map(bones: Mapping[str, object]) -> str:
    """The name of the bone map a skin's bones fit: game_engine with any of its bones, else cmu_mb."""
    return "game_engine" if any(bone in bones for chain in GAME_ENGINE.values() for bone, _ in chain) else "cmu_mb"


def leg_chains(bones: Mapping[str, Sequence[tuple[str, float]]]) -> tuple[list[tuple[str, str, str]], set[str]]:
    """A bone map's hip, knee, ankle bones per leg, and every leg bone the neutral stance takes from the skin."""
    chains = [(bones[f"{side}_hip"][-1][0], bones[f"{side}_knee"][-1][0], bones[f"{side}_ankle"][-1][0]) for side in ("left", "right")]
    held = {bone for side in ("left", "right") for joint in ("hip", "knee", "ankle", "foot") for bone, _ in bones.get(f"{side}_{joint}", ())}
    return chains, held


def rig_joints(skin: Path, clip: Path | None, template: Template, bones: Mapping[str, Sequence[tuple[str, float]]] | None = None) -> Rig:
    """The joints, stance, bones and soles of a skin: a spread joint gets one bone row per chain entry, the stance is the neutral pose."""
    bind = read_skin(skin)
    name = rig_map(bind.frames)
    bones = RIG_MAPS[name] if bones is None else bones
    needed = sorted({bone for chain in bones.values() for bone, _ in chain})
    missing = [bone for bone in needed if bone not in bind.frames]
    if missing:
        raise ValueError(f"{skin}: bones {missing} are not in the skin, the {name} rig maps the joints onto {needed}")
    neutral = read_neutral(skin, clip, *leg_chains(bones))
    offsets = rest_offsets(template, {joint: [bone for bone, _ in chain] for joint, chain in bones.items()}, bind.joints, bind.frames)
    names = list(offsets)
    parents = [names.index(parent) if (parent := template.parents[joint]) is not None else -1 for joint in names]
    stance = []
    bone_names: list[str] = []
    bone_joints: list[int] = []
    bone_weights: list[float] = []
    bone_offsets: list[np.ndarray] = []
    for index, (joint, pairs) in enumerate(offsets.items()):
        for (bone, offset), (_, weight) in zip(pairs, bones[joint], strict=True):
            bone_names.append(bone)
            bone_joints.append(index)
            bone_weights.append(weight)
            bone_offsets.append(offset)
        last, offset = pairs[-1]
        stance.append(matrix_to_quat(neutral[last] @ quat_to_matrix(offset).T))
    soles = [names.index(joint) for joint in SOLES if joint in names]
    return Rig(
        tuple(names),
        np.array(parents, dtype=np.int32),
        np.array(stance, dtype=float).reshape(-1, 4),
        tuple(bone_names),
        np.array(bone_joints, dtype=np.int32),
        np.array(bone_weights, dtype=float),
        np.array(bone_offsets, dtype=float).reshape(-1, 4),
        np.array(soles, dtype=np.int32),
        settings={"skeleton": SKELETON, "source_rig": name},
    )


def declared_drivers(settings: Mapping[str, object], source: str) -> tuple[RigDriver, ...]:
    """The `drivers` section of a bundle's rig.yaml as rig drivers, `bone` and `axis` split from the rest."""
    section = settings.get("drivers") or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{source}: drivers takes {{<wire joint>: {{bone: <skin bone>, axis: [x, y, z], from: ...}}}}, got {section!r}")
    drivers = []
    for joint, spec in section.items():
        if not isinstance(spec, Mapping) or "bone" not in spec or "axis" not in spec:
            raise ValueError(f"{source}: driver {joint!r} takes {{bone: <skin bone>, axis: [x, y, z], from: ...}}, got {spec!r}")
        drivers.append(RigDriver(str(joint), str(spec["bone"]), spec["axis"], {key: value for key, value in spec.items() if key not in ("bone", "axis")}))
    return tuple(drivers)


def build_rig(bundle: Path, template: Template) -> Rig:
    """The bundle's rig: joints from its skin, `drivers` and the `neutral`, `rest` and `root_offset` settings from its rig.yaml."""
    declared = read_settings(bundle)
    rig = rig_joints(skin_path(bundle), neutral_clip(bundle, declared), template)
    settings = {**rig.settings, **{key: declared[key] for key in _KEPT if key in declared}}
    return dataclasses.replace(rig, settings=settings, drivers=declared_drivers(declared, str(bundle / RIG_YAML)))


def skin_sdf(bundle: Path) -> str:
    """The bundle's actor SDF with every relative file reference moved up one directory, for rig/<name>.sdf."""
    tree = ET.parse(bundle / f"{bundle.name}.sdf")
    for element in tree.getroot().iter():
        if element.tag in ("filename", "uri") and element.text and "://" not in element.text and not element.text.startswith("/"):
            element.text = f"../{element.text}"
    ET.indent(tree)
    return ET.tostring(tree.getroot(), encoding="unicode") + "\n"


def write_bundle(bundle: Path, rig: Rig) -> list[Path]:
    """Write rig/<name>.rig, and rig/<name>.sdf when the bundle has an SDF and no rig/ skin of its own, rig.yaml and the old files untouched."""
    sub = bundle / SUB_BUNDLE
    written = [write_rig(sub / f"{bundle.name}{RIG_SUFFIX}", rig)]
    sdf = bundle / f"{bundle.name}.sdf"
    if sdf.is_file() and not (sub / "meshes" / f"{bundle.name}.dae").is_file():
        target = sub / f"{bundle.name}.sdf"
        target.write_text(skin_sdf(bundle))
        written.append(target)
    return written


def describe(rig: Rig) -> dict[str, object]:
    """Joints with parents, bone rows, soles, drivers and settings of a rig as plain data."""
    parents = rig.parent_names
    return {
        "joints": {joint: parents[joint] for joint in rig.joints},
        "bones": [{"bone": bone, "joint": rig.joints[joint], "weight": float(weight)} for bone, joint, weight in zip(rig.bones, rig.bone_joints, rig.bone_weights, strict=True)],
        "soles": list(rig.sole_names),
        "drivers": {driver.name: {"bone": driver.bone, "axis": driver.axis.tolist(), **driver.settings} for driver in rig.drivers},
        "settings": dict(rig.settings),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rig_human",
        description=(
            "Write the body rig of a human bundle: map the body joints of the Arena rig onto the skin's cmu_mb or game_engine bones (game_engine adds the 30 digit joints) and store each bone's rest offset, "
            "its world rotation once the bind skeleton is swung into the template zero pose (arms out sideways, legs straight down). "
            "The drivers and the neutral, rest and root_offset settings come from the bundle's rig.yaml, which is only read. Nothing outside rig/ is written."
        ),
        epilog=(
            "output: <bundle>/rig/<name>.rig with joints, stance (each joint in the neutral clip's pose, legs hanging), bones (neck spread over Neck and Neck1), soles and drivers, "
            "plus <bundle>/rig/<name>.sdf pointing at ../meshes/<name>.dae and ../clips unless rig/meshes/<name>.dae exists. "
            "examples: rig_human ~/arena_humans/out/arenian | rig_human ./Human/arenian_wheelchair --dry-run. "
            "exit 0 on success, 2 on a bad argument, a bundle without meshes/<name>.dae or a skin missing a bone of its map."
        ),
    )
    parser.add_argument("bundle", type=Path, help="human bundle directory holding <name>.sdf and meshes/<name>.dae, <name> being the directory name")
    parser.add_argument("--dry-run", action="store_true", help="print the rig as json instead of writing it")
    parser.add_argument("--template", type=Path, metavar="YAML", help="template skeleton (default: configs/body/template.yaml of arena_simulation_setup)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point, writes or prints the bundle's rig."""
    parser = _parser()
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        parser.print_help(sys.stderr)
        return 2
    args = parser.parse_args(args_list)
    bundle: Path = args.bundle.expanduser()
    if not bundle.is_dir():
        parser.error(f"bundle {str(bundle)!r} is not a directory, pass a human bundle directory holding <name>.sdf and meshes/<name>.dae")
    try:
        rig = build_rig(bundle, load_template(args.template))
    except (OSError, ValueError) as e:
        parser.error(str(e))
    if args.dry_run:
        print(json.dumps(describe(rig), indent=2))
        return 0
    written = write_bundle(bundle, rig)
    print(f"wrote {', '.join(str(path) for path in written)}: {len(rig.joints)} joints, soles {', '.join(rig.sole_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
