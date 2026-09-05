"""Parse gz `scene/info` textproto into model name -> entity id, without gz or ROS imports."""

from __future__ import annotations

import re

_MODEL_OPEN = re.compile(r"^\s*model\s*\{\s*$")
_ID = re.compile(r"^\s*id:\s*(\d+)\s*$")
_NAME = re.compile(r'^\s*name:\s*"((?:[^"\\]|\\.)*)"\s*$')


def parse_scene_models(text: str) -> dict[str, int]:
    """Top-level `model { name id }` pairs of a gz.msgs.Scene textproto, newest id winning on a duplicate name."""
    models: dict[str, int] = {}
    depth = 0
    in_model = False
    name: str | None = None
    entity_id: int | None = None
    for line in text.splitlines():
        if not in_model and depth == 0 and _MODEL_OPEN.match(line):
            in_model, depth, name, entity_id = True, 1, None, None
            continue
        if in_model and depth == 1:
            if (m := _ID.match(line)) is not None:
                entity_id = int(m.group(1))
            elif (m := _NAME.match(line)) is not None:
                name = m.group(1)
        depth += line.count("{") - line.count("}")
        if in_model and depth == 0:
            if name is not None and entity_id is not None:
                models[name] = max(entity_id, models.get(name, -1))
            in_model = False
    return models
