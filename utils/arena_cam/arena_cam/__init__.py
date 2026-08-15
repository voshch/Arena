"""Cam: control of the Arena viewport cameras over /arena/viewport/*.

The scripting entry points resolve lazily, so the pure motion core (`fly`) and the
key mapping import without rclpy present.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .camera import Camera
    from .shot import load_shot
    from .surfaces import TargetSelection

__all__ = ["Camera", "TargetSelection", "load_shot"]


def __getattr__(name: str) -> object:
    if name == "TargetSelection":
        from .surfaces import TargetSelection

        return TargetSelection
    if name in ("Camera", "load_shot"):
        from .shots import discover

        discover()
        if name == "Camera":
            from .camera import Camera

            return Camera
        from .shot import load_shot

        return load_shot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
