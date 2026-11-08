"""Joint positions of a human bundle's character posed by wire angles, from the bundle's own files: no simulator needed.

The simulators render a human by turning the wire's semantic angles into body-rig rotations over the character's rig
stance (`semantic_to_body`), composing them down the bundle's `.rig` onto the skin skeleton's bones, holding every other
bone at the neutral clip's first frame, and standing the skeleton on its lowest sole point. `ActorPose` does the same
with numpy, following Isaac's wire pose provider (`arena_isaac/peds/providers/external.py`) and its converter's
skeleton, neutral pose and sole points (`peds/convert.py`), so positions match what Isaac draws (tests hold it to 1 mm
where both are importable). Use it wherever geometry must meet the rendered body, e.g. a hand closing on a prop.

    body = ActorPose(bundle_dir)                      # <bundle>/rig/<name>.sdf, .rig, meshes/<name>.dae, clips/
    pos = body.positions({"l_elbow": 1.2, ...})      # {template joint: (3,) m}, x forward, y left, z up, soles on z = 0
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arena_assets.rig import Rig, load_rig
from arena_assets.utils.geom import nearest_rotation
from arena_assets.utils.human import first_keys, read_skin, scene_nodes
from arena_simulation_setup.utils.geometry import Orientation

from .body import forward, load_template, semantic_to_body, stance_of

SOLE_BAND = 0.01  # m above a foot's lowest bind vertex that still counts as sole (Isaac's converter)
FEET = ("left_ankle", "right_ankle", "left_foot", "right_foot")


@dataclass(frozen=True)
class _Bone:
    name: str
    parent: int
    rest: np.ndarray  # (4, 4) local


def _skeleton(skin: Path) -> list[_Bone]:
    """Joint nodes of the skin's visual scene parents first, each under its nearest joint ancestor (Isaac's skeleton)."""
    root = ET.parse(skin).getroot()
    nodes = scene_nodes(root, root.tag[: root.tag.index("}") + 1], skin)
    bone_of: dict[int, int] = {}
    bones: list[_Bone] = []
    for i, (parent, local, joint, _) in enumerate(nodes):
        while parent >= 0 and parent not in bone_of:
            parent = nodes[parent][0]
        if joint is not None:
            bone_of[i] = len(bones)
            bones.append(_Bone(joint, bone_of[parent] if parent >= 0 else -1, local))
    return bones


class ActorPose:
    """A human bundle's character, posed from wire angles like the simulators render it (Isaac's pose provider)."""

    def __init__(self, bundle: str | Path) -> None:
        bundle = Path(bundle)
        name = bundle.name
        sdf = ET.parse(bundle / "rig" / f"{name}.sdf").getroot()
        actor = sdf.find("actor")
        if actor is None:
            raise ValueError(f"{bundle}: rig/{name}.sdf has no <actor>")
        self.skin = bundle / "rig" / (actor.findtext("skin/filename") or "")
        clips = {a.get("name"): bundle / "rig" / (a.findtext("filename") or "") for a in actor.iterfind("animation")}
        rig = load_rig(bundle / "rig" / f"{name}.rig")
        if rig is None or not rig.joints:
            raise ValueError(f"{bundle}: no rig/{name}.rig with joints")
        self.rig: Rig = rig
        self.stance = stance_of(rig)
        self._bones = _skeleton(self.skin)
        clip = clips.get("neutral") or clips.get("idle")
        self.neutral_clip = clip if clip is not None and clip.is_file() else None
        keys = first_keys(self.neutral_clip) if self.neutral_clip is not None else {}
        locals_ = [keys[b.name] if b.name in keys and b.parent >= 0 else b.rest for b in self._bones]  # the root pinned to rest
        self._clip = [Orientation.from_matrix(nearest_rotation(m)) for m in locals_]
        self._trans = [np.array(m[:3, 3], dtype=float) for m in locals_]
        bone_index: dict[str, int] = {}
        for i, b in enumerate(self._bones):
            bone_index.setdefault(b.name, i)
        # rig joints -> (parent, spread, ((bone, cumulative share, offset), ...)), as the provider builds them
        self._joints = []
        for jname, parent in zip(rig.joints, rig.parents.tolist(), strict=True):
            rows = rig.bones_of(jname)
            shares = np.cumsum([w for _, w, _ in rows]).tolist()
            bones = tuple((bone_index[b], s, Orientation.from_xyzw(off).normalized()) for (b, _, off), s in zip(rows, shares, strict=True) if b in bone_index)
            self._joints.append((parent, len(rows) > 1, bones))
        world = [Orientation.from_xyzw(q).normalized() for q in rig.stance]
        self._stance_locals = [(world[p].inverse() * world[i]).normalized() if p >= 0 else world[i] for i, p in enumerate(rig.parents.tolist())]
        self._joint_index = {n: i for i, n in enumerate(rig.joints)}
        self.joint_bone = {j: bone_index[rows[0][0]] for j in rig.joints if (rows := rig.bones_of(j)) and rows[0][0] in bone_index}
        # sole points: heel and toe of each (foot bone, toe bone) pair of the rig's sole joints, in the bone's frame
        skin = read_skin(self.skin)
        inv, verts = skin.inverse_binds, list(zip(skin.points, skin.dominant, strict=True))
        self._soles: list[tuple[int, np.ndarray]] = []
        if not any(d.settings.get("from") == "distance" for d in rig.drivers):
            sole_bones = [rig.bones_of(n)[-1][0] for n in rig.sole_names]
            for foot, toe in zip(sole_bones[::2], sole_bones[1::2], strict=False):
                cloud = [(p, o) for p, o in verts if o in (foot, toe)]
                if not cloud or foot not in inv or toe not in inv:
                    continue
                pts = np.array([p for p, _ in cloud])
                fwd = np.linalg.inv(inv[toe])[:3, 3] - np.linalg.inv(inv[foot])[:3, 3]
                fwd[2] = 0.0
                sole = np.flatnonzero(pts[:, 2] <= pts[:, 2].min() + SOLE_BAND)
                along = pts[sole] @ fwd
                mid = 0.5 * (along.min() + along.max())
                for half in (sole[along <= mid], sole[along > mid]):
                    v = half[int(np.argmin(pts[half, 2]))]
                    owner = cloud[v][1]
                    self._soles.append((bone_index[owner], (inv[owner] @ np.append(pts[v], 1.0))[:3]))

    def bone_worlds(self, angles: Mapping[str, float]) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        """World rotation matrices and positions of every skin bone for semantic angles, and the lift that stands the
        skeleton on its lowest sole point (added to z by `positions`)."""
        rotations = semantic_to_body(dict(angles), self.stance)
        locals_ = list(self._stance_locals)
        for joint, q in rotations.items():
            i = self._joint_index.get(joint)
            if i is not None:
                locals_[i] = Orientation.from_xyzw(q).normalized()
        mapped: dict[int, Orientation] = {}
        globals_: list[Orientation] = []
        for (parent, spread, bones), local in zip(self._joints, locals_, strict=True):
            pg = globals_[parent] if parent >= 0 else Orientation.identity()
            globals_.append(pg * local)
            for bone, share, offset in bones:
                base = pg * local**share if spread else globals_[-1]
                mapped[bone] = base * offset
        R: list[np.ndarray] = []
        P: list[np.ndarray] = []
        Wq: list[Orientation] = []
        for i, b in enumerate(self._bones):
            pq = Wq[b.parent] if b.parent >= 0 else Orientation.identity()
            local = (pq.inverse() * mapped[i]).normalized() if i in mapped else self._clip[i]
            Wq.append(pq * local)
            R.append(Wq[-1].to_matrix())
            P.append(self._trans[i].copy() if b.parent < 0 else P[b.parent] + R[b.parent] @ self._trans[i])
        lift = -min((float((R[j] @ p + P[j])[2]) for j, p in self._soles), default=0.0)
        return R, P, lift

    def positions(self, angles: Mapping[str, float]) -> dict[str, np.ndarray]:
        """Position of every rig joint (template names, its first skin bone), soles standing on z = 0. A rig with a distance
        driver (a wheelchair) has no sole points and stays on its skeleton root, as Isaac draws it."""
        _, P, lift = self.bone_worlds(angles)
        up = np.array([0.0, 0.0, lift])
        return {j: P[i] + up for j, i in self.joint_bone.items()}


class TemplatePose:
    """Arena's template body with the same interface: feet on the floor (ankle and foot joints sit ~3 cm above the sole)."""

    def __init__(self) -> None:
        self._template = load_template()

    def positions(self, angles: Mapping[str, float]) -> dict[str, np.ndarray]:
        pos, _ = forward(self._template, semantic_to_body(dict(angles)))
        floor = min(pos[j][2] for j in FEET) - 0.03
        return {j: p - np.array([0.0, 0.0, floor]) for j, p in pos.items()}


__all__ = ["ActorPose", "TemplatePose"]
