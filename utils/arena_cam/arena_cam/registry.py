"""Single-source registry of camera verbs.

Decorate an action/segment class with `@primitive("verb")` and it becomes usable
from `Camera.add`, the shot YAML, and the CLI with no other edits. Aliases stack:

    @primitive("look")
    @primitive("cut")
    class _Look(_Action): ...
"""

from __future__ import annotations

from collections.abc import Callable

PRIMITIVES: dict[str, type] = {}


def primitive(verb: str) -> Callable[[type], type]:
    def register(cls: type) -> type:
        cls.verb = verb
        PRIMITIVES[verb] = cls
        return cls

    return register
