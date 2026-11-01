"""Arena's 40-DOF human wire contract: the target skeleton.

The axes, chain order and FK come from ``task_generator``'s ``pointing/skeleton.py`` (URDF-faithful, what RViz draws and
the Isaac bone map consumes), so they are never restated here. That module is plain numpy, but importing it through its
package would pull in rclpy, so it is loaded straight from its file. Lookup order: ``$ARENA_SKELETON``, the installed
``task_generator`` package, then this repository's source tree.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

_REL = Path("simulators/human/pointing/skeleton.py")


def _candidates() -> list[Path]:
    out = []
    if os.environ.get("ARENA_SKELETON"):
        out.append(Path(os.environ["ARENA_SKELETON"]))
    try:
        spec = importlib.util.find_spec("task_generator")  # locates the package without running its __init__
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        out += [Path(p) / _REL for p in (spec.submodule_search_locations or [])]
    out.append(Path(__file__).resolve().parents[5] / "task_generator" / "task_generator" / _REL)  # repo checkout
    return out


def load_skeleton() -> ModuleType:
    for path in _candidates():
        if path.is_file():
            spec = importlib.util.spec_from_file_location("arena_wire_skeleton", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"Arena skeleton.py not found; set ARENA_SKELETON to task_generator/.../pointing/skeleton.py. Tried: {', '.join(map(str, _candidates()))}")


SK = load_skeleton()

# The wire DOFs a full-body clip carries (40), in contract naming.
WIRE: list[str] = ["r_waist", "y_waist", "waist", "r_spine", "y_spine", "spine", "r_chest", "y_chest", "chest", "r_head", "y_head", "p_head"] + [
    f"{s}_{d}" for s in "lr" for d in ("y_collar", "p_collar", "y_shoulder", "p_shoulder", "r_shoulder", "elbow", "r_wrist", "wrist", "y_hip", "p_hip", "r_hip", "knee", "y_ankle", "ankle")
]
