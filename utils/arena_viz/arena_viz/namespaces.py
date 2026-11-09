"""Topic namespace conventions shared by the visualizers."""

from __future__ import annotations

import os


def env_namespace(taskgen_node: str) -> str:
    """The env's topic namespace from its task generator node (`/arena/env_0/arena_env_node` -> `/arena/env_0`)."""
    return os.path.dirname(taskgen_node.rstrip("/")).rstrip("/")
