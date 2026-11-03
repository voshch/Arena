"""Internal REGISTRY dict and register decorator. Imported by renderer modules."""

from __future__ import annotations

from typing import Callable

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

RendererFn = Callable[[AdapterDisplay, "RobotDescriptor | None"], "dict[str, object] | None"]

REGISTRY: dict[DisplayKind, RendererFn] = {}


def register(kind: DisplayKind) -> Callable[[RendererFn], RendererFn]:
    """Decorator that registers a renderer for a DisplayKind."""

    def decorator(fn: RendererFn) -> RendererFn:
        REGISTRY[kind] = fn
        return fn

    return decorator


def rviz_extra(style: StyleSpec) -> dict[str, object]:
    """The style's `extra.rviz` mapping, empty when unset."""
    extra = style.extra.get("rviz", {})
    if not isinstance(extra, dict):
        raise TypeError(f"style extra 'rviz' must be a mapping, got {extra!r}")
    return extra
