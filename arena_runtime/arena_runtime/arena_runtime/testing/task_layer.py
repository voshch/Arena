"""Generate a task layer that the seam discovers, for tests of a runtime other than the default."""

from __future__ import annotations

from pathlib import Path

from arena_runtime.env.task_layer import TASK_LAYER_GROUP


def register_task_layer(root: Path, name: str, source: str) -> None:
    """Write source as the module `name` and register it as task layer `name`, found once root is on sys.path."""
    (root / f"{name}.py").write_text(source)
    info = root / f"{name}-0.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.0\n")
    (info / "entry_points.txt").write_text(f"[{TASK_LAYER_GROUP}]\n{name} = {name}\n")
