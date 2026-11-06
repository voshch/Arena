"""Internal REGISTRY and register decorator for rerun display kinds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import rclpy.node
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind


@dataclass
class RendererCtx:
    """Bridge context handed to every renderer."""

    env_id: int
    node: rclpy.node.Node
    env_namespace: str = ""  # where the env's topics live, e.g. /arena/env_0 (its task generator's namespace)


RendererFn = Callable[[AdapterDisplay, "RobotDescriptor | None", RendererCtx], None]

REGISTRY: dict[DisplayKind, RendererFn] = {}


def register(kind: DisplayKind) -> Callable[[RendererFn], RendererFn]:
    """Decorator that registers a renderer for a DisplayKind."""

    def decorator(fn: RendererFn) -> RendererFn:
        REGISTRY[kind] = fn
        return fn

    return decorator
