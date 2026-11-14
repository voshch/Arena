import math
import pathlib
import typing

import attrs
import yaml

from arena_fleet.geometry import Point, Wall


@attrs.frozen
class Door:
    """A managed door of the resource table: its polygon in the level frame and centre."""

    id: str
    polygon: tuple[Point, ...]

    @property
    def c(self) -> Point:
        n = len(self.polygon)
        return sum(p[0] for p in self.polygon) / n, sum(p[1] for p in self.polygon) / n


@attrs.frozen
class DoorLeaf:
    """A world door whose leaf position tells how far open it is."""

    name: str
    c: Point
    span: float


@attrs.frozen
class Site:
    """World walls and doors, the managed doors and the parked robot starts of one level."""

    walls: tuple[Wall, ...]
    leaves: tuple[DoorLeaf, ...]
    doors: dict[str, Door]
    starts: tuple[Point, ...]

    @classmethod
    def load(cls, world: pathlib.Path, resources: pathlib.Path, scenario: pathlib.Path, level: str = "0") -> "Site":
        lv = next((x for x in _mapping(world)["levels"] if str(x["id"]) == level), None)
        if lv is None:
            raise ValueError(f"{world}: no level {level}")
        walls: list[Wall] = []
        for w in lv["walls"]:
            path = [_point(p) for p in w["path"]]
            walls += [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        leaves: list[DoorLeaf] = []
        for d in lv.get("doors", []):
            a, b = _point(d["span"][0]), _point(d["span"][1])
            leaves.append(DoorLeaf(str(d["name"]), ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), math.dist(a, b)))
        doors = {str(r["id"]): Door(str(r["id"]), tuple(_point(p) for p in r["polygon"])) for r in _mapping(resources)["resources"] if r["kind"] == "door"}
        if not doors:
            raise ValueError(f"{resources}: no resource of kind door")
        starts = tuple(_point(r["start"]) for r in _mapping(scenario)["robots"])
        return cls(walls=tuple(walls), leaves=tuple(leaves), doors=doors, starts=starts)


def _mapping(path: pathlib.Path) -> dict[str, typing.Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return typing.cast(dict[str, typing.Any], data)


def _point(value: object) -> Point:
    if not isinstance(value, list | tuple) or len(value) < 2:
        raise ValueError(f"expected [x, y, ...], got {value!r}")
    seq = typing.cast(list[float], value)
    return float(seq[0]), float(seq[1])
