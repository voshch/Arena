"""Where the AnimationManager's canned clips come from: the Animation asset kind (world/local asset dirs, then the bucket)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from arena_simulation_setup.tree.assets.Animation import AnimationClip

# Masks an annotation can name instead of listing joints. Authored by body group: capture noise and twist wraps make
# per-joint motion a poor guide. The contact clips turn the forearm, so theirs also carry the wrist pair.
_TORSO_HEAD = ("r_waist", "y_waist", "waist", "r_spine", "y_spine", "spine", "r_chest", "y_chest", "chest", "r_head", "y_head", "p_head")
_ARM = ("y_collar", "p_collar", "y_shoulder", "p_shoulder", "r_shoulder", "elbow")
_WRIST = ("r_wrist", "wrist")
JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "upper_body": (*_TORSO_HEAD, *(f"{s}_{j}" for s in "lr" for j in _ARM)),
    "upper_body_wrists": (*_TORSO_HEAD, *(f"{s}_{j}" for s in "lr" for j in (*_ARM, *_WRIST))),
}


def joint_mask(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """An annotation's ``joints``: a JOINT_GROUPS name, a joint list, or omitted (empty = whole body)."""
    if value is None:
        return ()
    if isinstance(value, str):
        return JOINT_GROUPS[value]
    return tuple(value)


class ClipSource(Protocol):
    def names(self) -> list[str]:
        """Every clip this source can load."""
        ...

    def load(self, names: Sequence[str]) -> dict[str, AnimationClip | Exception]:
        """Each requested clip, or why it could not be loaded."""
        ...


class AssetClips:
    """Clips resolved through AnimationIdentifier, fetched from the bucket on first use and cached on disk."""

    def names(self) -> list[str]:
        from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier

        return sorted({identifier.name for identifier in AnimationIdentifier.listall(network=True)})

    def load(self, names: Sequence[str]) -> dict[str, AnimationClip | Exception]:
        from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier

        async def resolve_all() -> list:
            # resolved together, so the bucket fetches batch into one request
            return await asyncio.gather(*(AnimationIdentifier.parse(name).resolve() for name in names), return_exceptions=True)

        out: dict[str, AnimationClip | Exception] = {}
        for name, view in zip(names, AnimationIdentifier._run_sync(resolve_all()), strict=True):
            try:
                out[name] = view if isinstance(view, Exception) else view.clip
            except Exception as exc:  # unreadable payload
                out[name] = exc
        return out


class NoClips:
    """No canned clips: synthesis and transients only."""

    def names(self) -> list[str]:
        return []

    def load(self, names: Sequence[str]) -> dict[str, AnimationClip | Exception]:
        return {name: FileNotFoundError(name) for name in names}
