"""Module-level handle to the live SceneStore for free-function service handlers."""

from __future__ import annotations

from arena_mujoco.scene import SceneStore

_scene_store: SceneStore | None = None


def set_scene_store(store: SceneStore) -> None:
    """Install the process-wide SceneStore, called once during node startup."""
    global _scene_store
    _scene_store = store


def get_scene_store() -> SceneStore:
    """Return the installed SceneStore, raising RuntimeError if it was never set."""
    if _scene_store is None:
        raise RuntimeError('scene store not initialized, call set_scene_store first')
    return _scene_store


__all__ = [
    'get_scene_store',
    'set_scene_store',
]
