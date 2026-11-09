"""Arena's 42-DOF human wire contract: the target skeleton.

The axes, chain order and FK come from ``arena_runtime``'s ``humans/pointing/skeleton.py`` (URDF-faithful, what RViz draws and
the Isaac bone map consumes), so they are never restated here. That module is plain numpy, but importing it through its
package would pull in rclpy, so it is loaded straight from its file. Lookup order: ``$ARENA_SKELETON``, the installed
``arena_runtime`` package, then this repository's source tree.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

_REL = Path("humans/pointing/skeleton.py")


def _candidates(rel: Path = _REL, env: str = "ARENA_SKELETON") -> list[Path]:
    out = []
    if os.environ.get(env):
        out.append(Path(os.environ[env]))
    try:
        spec = importlib.util.find_spec("arena_runtime")  # locates the package without running its __init__
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        out += [Path(p) / rel for p in (spec.submodule_search_locations or [])]
    out.append(Path(__file__).resolve().parents[5] / "arena_runtime" / "arena_runtime" / "arena_runtime" / rel)  # repo checkout
    return out


def load_module(rel: Path, name: str, env: str) -> ModuleType:
    """A plain-numpy arena_runtime module loaded straight from its file (its package would pull in rclpy)."""
    for path in _candidates(rel, env):
        if path.is_file():
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"Arena {rel.name} not found; set {env} to arena_runtime/arena_runtime/arena_runtime/{rel}. Tried: {', '.join(map(str, _candidates(rel, env)))}")


def load_skeleton() -> ModuleType:
    return load_module(_REL, "arena_wire_skeleton", "ARENA_SKELETON")


SK = load_skeleton()

# The wire DOFs a full-body clip carries (42), in contract naming.
WIRE: list[str] = (
    ["r_waist", "y_waist", "waist", "r_spine", "y_spine", "spine", "r_chest", "y_chest", "chest", "r_head", "y_head", "p_head"]
    + [f"{s}_{d}" for s in "lr" for d in ("y_collar", "p_collar", "y_shoulder", "p_shoulder", "r_shoulder", "elbow", "r_wrist", "wrist", "y_hip", "p_hip", "r_hip", "knee", "y_ankle", "ankle")]
    + [f"{s}_d_wrist" for s in "lr"]
)
