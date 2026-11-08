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

from .body import forward, load_template, matrix_to_quat, quat_to_matrix, semantic_to_body, stance_of

SOLE_BAND = 0.01  # m above a foot's lowest bind vertex that still counts as sole (Isaac's converter)
FEET = ("left_ankle", "right_ankle", "left_foot", "right_foot")


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qnorm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return np.array([0.0, 0.0, 0.0, 1.0]) if n == 0.0 else q / n


def _qpow(q: np.ndarray, e: float) -> np.ndarray:
    x, y, z, w = q if q[3] >= 0.0 else -q
    s = float(np.sqrt(x * x + y * y + z * z))
    if s < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    half = e * float(np.arctan2(s, w))
    k = np.sin(half) / s
    return np.array([x * k, y * k, z * k, np.cos(half)])


def _quat(m: np.ndarray) -> np.ndarray:
    """xyzw of the proper rotation nearest to a (possibly scaled or mirrored) 3x3 matrix."""
    u, _, vt = np.linalg.svd(m)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        rotation = u @ vt
    return matrix_to_quat(rotation)


@dataclass(frozen=True)
class _Bone:
    name: str
    parent: int
    rest: np.ndarray  # (4, 4) local


def _skeleton(root: ET.Element, ns: str) -> list[_Bone]:
    """Joint nodes of the visual scene depth first, parents first; non-joint nodes are skipped (Isaac's skeleton)."""
    bones: list[_Bone] = []

    def recurse(node: ET.Element, parent: int) -> None:
        for child in node.findall(f"{ns}node"):
            if child.get("type") != "JOINT":
                recurse(child, parent)
                continue
            text = child.findtext(f"{ns}matrix")
            if not text:
                raise ValueError(f"joint {child.get('sid')} has no <matrix>")
            bones.append(_Bone(child.get("sid") or child.get("id") or "", parent, np.array(text.split(), dtype=float).reshape(4, 4)))
            recurse(child, len(bones) - 1)

    scene = root.find(f"{ns}library_visual_scenes/{ns}visual_scene")
    if scene is None:
        raise ValueError("COLLADA has no visual_scene")
    recurse(scene, -1)
    return bones


def _first_keys(clip: Path) -> dict[str, np.ndarray]:
    """Local transform of every joint a COLLADA clip keys, at its first key."""
    root = ET.parse(clip).getroot()
    ns = root.tag[: root.tag.index("}") + 1]
    by_id = {e.get("id"): e for e in root.iter() if e.get("id")}
    keys: dict[str, np.ndarray] = {}
    for channel in root.iter(f"{ns}channel"):
        joint, _, attribute = (channel.get("target") or "").partition("/")
        if attribute != "transform":
            continue
        sampler = by_id[(channel.get("source") or "").lstrip("#")]
        output = next(e.get("source") for e in sampler.iterfind(f"{ns}input") if e.get("semantic") == "OUTPUT")
        keys[joint] = np.array((by_id[(output or "").lstrip("#")].findtext(f"{ns}float_array") or "").split()[:16], dtype=float).reshape(4, 4)
    return keys


def _skin_bind(root: ET.Element, ns: str) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, str]]]:
    """Inverse bind matrix per skin joint, and every bind-space vertex with its most-weighted joint."""
    by_id = {e.get("id"): e for e in root.iter() if e.get("id")}

    def floats(ref: str | None) -> np.ndarray:
        return np.array((by_id[(ref or "").lstrip("#")].findtext(f"{ns}float_array") or "").split(), dtype=float)

    inv: dict[str, np.ndarray] = {}
    verts: list[tuple[np.ndarray, str]] = []
    for skin in root.iter(f"{ns}skin"):
        shape = np.array((skin.findtext(f"{ns}bind_shape_matrix") or "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1").split(), dtype=float).reshape(4, 4)
        inputs = {e.get("semantic"): e.get("source") for e in skin.iterfind(f"{ns}joints/{ns}input")}
        src = by_id[(inputs["JOINT"] or "").lstrip("#")]
        names = (src.findtext(f"{ns}Name_array") or src.findtext(f"{ns}IDREF_array") or "").split()
        mats = floats(inputs["INV_BIND_MATRIX"]).reshape(-1, 4, 4)
        for n, m in zip(names, mats, strict=True):
            inv.setdefault(n, m)
        vw = skin.find(f"{ns}vertex_weights")
        if vw is None:
            continue
        offs = {e.get("semantic"): (int(e.get("offset", "0")), e.get("source")) for e in vw.iterfind(f"{ns}input")}
        jo, wo, wsrc = offs["JOINT"][0], offs["WEIGHT"][0], offs["WEIGHT"][1]
        weights = floats(wsrc)
        counts = np.array((vw.findtext(f"{ns}vcount") or "").split(), dtype=int)
        pairs = np.array((vw.findtext(f"{ns}v") or "").split(), dtype=int).reshape(-1, max(jo, wo) + 1)
        mesh = by_id[(skin.get("source") or "").lstrip("#")]
        pos = mesh.find(f"{ns}mesh/{ns}vertices/{ns}input[@semantic='POSITION']")
        points = floats(pos.get("source") if pos is not None else None).reshape(-1, 3)
        start = 0
        for vertex, count in zip(points, counts, strict=True):
            row = pairs[start : start + count]
            start += count
            if count:
                verts.append(((shape @ np.append(vertex, 1.0))[:3], names[row[int(np.argmax(weights[row[:, wo]])), jo]]))
    return inv, verts


class ActorPose:
    """A human bundle's character, posed from wire angles like the simulators render it (Isaac's pose provider)."""

    def __init__(self, bundle: str | Path) -> None:
        bundle = Path(bundle)
        name = bundle.name
        sdf = ET.parse(bundle / "rig" / f"{name}.sdf").getroot()
        actor = sdf.find("actor")
        if actor is None:
            raise ValueError(f"{bundle}: rig/{name}.sdf has no <actor>")
        skin_file = bundle / "rig" / (actor.findtext("skin/filename") or "")
        clips = {a.get("name"): bundle / "rig" / (a.findtext("filename") or "") for a in actor.iterfind("animation")}
        rig = load_rig(bundle / "rig" / f"{name}.rig")
        if rig is None or not rig.joints:
            raise ValueError(f"{bundle}: no rig/{name}.rig with joints")
        self.rig: Rig = rig
        self.stance = stance_of(rig)
        root = ET.parse(skin_file).getroot()
        ns = root.tag[: root.tag.index("}") + 1]
        self._bones = _skeleton(root, ns)
        neutral_clip = clips.get("neutral") or clips.get("idle")
        keys = _first_keys(neutral_clip) if neutral_clip is not None and neutral_clip.is_file() else {}
        locals_ = [keys[b.name] if b.name in keys and b.parent >= 0 else b.rest for b in self._bones]  # the root pinned to rest
        self._clip = [_quat(m[:3, :3]) for m in locals_]
        self._trans = [np.array(m[:3, 3], dtype=float) for m in locals_]
        bone_index: dict[str, int] = {}
        for i, b in enumerate(self._bones):
            bone_index.setdefault(b.name, i)
        # rig joints -> (parent, spread, ((bone, cumulative share, offset), ...)), as the provider builds them
        self._joints = []
        for jname, parent in zip(rig.joints, rig.parents.tolist(), strict=True):
            rows = rig.bones_of(jname)
            shares = np.cumsum([w for _, w, _ in rows]).tolist()
            bones = tuple((bone_index[b], s, _qnorm(np.asarray(off, dtype=float))) for (b, _, off), s in zip(rows, shares, strict=True) if b in bone_index)
            self._joints.append((parent, len(rows) > 1, bones))
        world = [_qnorm(np.asarray(q, dtype=float)) for q in rig.stance]
        self._stance_locals = [_qnorm(_qmul(_qconj(world[p]), world[i])) if p >= 0 else world[i] for i, p in enumerate(rig.parents.tolist())]
        self._joint_index = {n: i for i, n in enumerate(rig.joints)}
        self.joint_bone = {j: bone_index[rows[0][0]] for j in rig.joints if (rows := rig.bones_of(j)) and rows[0][0] in bone_index}
        # sole points: heel and toe of each (foot bone, toe bone) pair of the rig's sole joints, in the bone's frame
        inv, verts = _skin_bind(root, ns)
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
                locals_[i] = _qnorm(np.asarray(q, dtype=float))
        mapped: dict[int, np.ndarray] = {}
        globals_: list[np.ndarray] = []
        for (parent, spread, bones), local in zip(self._joints, locals_, strict=True):
            pg = globals_[parent] if parent >= 0 else np.array([0.0, 0.0, 0.0, 1.0])
            globals_.append(_qmul(pg, local))
            for bone, share, offset in bones:
                base = _qmul(pg, _qpow(local, share)) if spread else globals_[-1]
                mapped[bone] = _qmul(base, offset)
        R: list[np.ndarray] = []
        P: list[np.ndarray] = []
        Wq: list[np.ndarray] = []
        for i, b in enumerate(self._bones):
            pq = Wq[b.parent] if b.parent >= 0 else np.array([0.0, 0.0, 0.0, 1.0])
            local = _qnorm(_qmul(_qconj(pq), mapped[i])) if i in mapped else self._clip[i]
            Wq.append(_qmul(pq, local))
            R.append(quat_to_matrix(Wq[-1]))
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
