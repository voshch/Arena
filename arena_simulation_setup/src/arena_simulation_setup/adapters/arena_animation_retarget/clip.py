"""Arena Animation clip files: ``<out>/<name>/<name>.anim`` + ``annotation.yaml``.

The ``clip`` group carries what Arena's loader reads (``joint_names``, ``angles`` (N, DOF), ``t``, ``root_xy_yaw``,
``animation_state``), the ``root`` group ``z`` and ``quat`` (pelvis height and orientation). A clip retargeted without
folding the pelvis tilt is annotated ``pelvis_tilt: true``: Arena's player then turns the ped's body by ``root/quat``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from arena_simulation_setup.tree.assets.Animation import ANIM_SUFFIX, write_anim

from .retarget import Clip


def write_clip(clip: Clip, out_dir: str | Path, name: str, *, loop: bool = False, desc: str | None = None, tags: list[str] | None = None, note: str = "", extra: dict | None = None) -> Path:
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    names = sorted(clip.angles)
    write_anim(
        d / f"{name}{ANIM_SUFFIX}",
        {
            "clip/joint_names": np.array(names),
            "clip/angles": np.stack([clip.angles[k] for k in names], 1),
            "clip/t": clip.t,
            "clip/root_xy_yaw": clip.root_xy_yaw,
            "clip/animation_state": np.zeros(len(clip.t), dtype=np.int16),
            "root/z": clip.root_z,
            "root/quat": clip.root_quat,
        },
    )
    tilt = "pelvis tilt folded into waist/hips" if clip.fold_tilt else "pelvis-relative body, apply root/quat"
    ann = {"name": name, "path": f"Animation/{name}", "desc": desc or name.replace("_", " "), "tags": list(tags or ["domain::common"]), "fps": round(clip.fps, 6), "loop": bool(loop), "note": " ".join(x for x in (note, f"Retargeted from SMPL-X by arena_animation_retarget ({tilt}; root/z, root/quat = pelvis).") if x)}
    if not clip.fold_tilt:
        ann["pelvis_tilt"] = True
    ann.update(extra or {})
    (d / "annotation.yaml").write_text(yaml.safe_dump(ann, sort_keys=False))
    return d


def load_clip(path: str | Path) -> Clip:
    """Read a clip .anim (or its directory) back. Clips without root/z or root/quat get zeros / identity."""
    p = Path(path)
    if p.is_dir():
        p = p / f"{p.name}{ANIM_SUFFIX}"
    z = np.load(p, allow_pickle=False)
    names = [str(n) for n in z["clip/joint_names"]]
    n = len(z["clip/t"])
    return Clip(
        angles={k: z["clip/angles"][:, i].astype(float) for i, k in enumerate(names)},
        t=z["clip/t"].astype(float),
        root_xy_yaw=z["clip/root_xy_yaw"].astype(float),
        root_z=z["root/z"].astype(float) if "root/z" in z.files else np.zeros(n),
        root_quat=z["root/quat"].astype(float) if "root/quat" in z.files else np.tile([0.0, 0, 0, 1], (n, 1)),
    )
