from __future__ import annotations

from pathlib import Path

import attrs
import yaml

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.tree import FallbackResolver, Identifier, SimplePathResolver
from arena_simulation_setup.utils.cattrs import Parseable


@attrs.define
class GestureKeyframe:
    pose: str
    t: float


@attrs.define
class GestureSpec(Parseable):
    keyframes: list[GestureKeyframe]
    interp: str = "linear"

    def __attrs_post_init__(self) -> None:
        if self.interp != "linear":
            raise ValueError(f"interp must be 'linear'; got {self.interp!r}")

    def required_poses(self) -> frozenset[str]:
        return frozenset(kf.pose for kf in self.keyframes)


class GestureIdentifier(Identifier[GestureSpec]):
    def relpath(self) -> Path:
        return Path(f"{self.name}.yaml")

    def load(self, path: Path, /, **kwargs: object) -> GestureSpec:
        del kwargs
        with open(path) as f:
            data = yaml.safe_load(f)
        return GestureSpec.parse(data)


GestureIdentifier.use(SimplePathResolver(GestureIdentifier, ASS_DIR / "configs" / "gestures"))
GestureIdentifier.use(FallbackResolver(GestureIdentifier, ASS_DIR / "configs" / "gestures"))
