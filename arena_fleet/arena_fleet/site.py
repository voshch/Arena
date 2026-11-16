import math
import pathlib
import typing

import attrs
import yaml

from arena_fleet.geometry import Point, Wall, area, inside


@attrs.frozen
class Door:
    """A managed door: the smallest world zone holding the door's span, its centre and its crossing axis."""

    id: str
    polygon: tuple[Point, ...]
    normal: Point

    @property
    def c(self) -> Point:
        n = len(self.polygon)
        return sum(p[0] for p in self.polygon) / n, sum(p[1] for p in self.polygon) / n

    @property
    def along(self) -> Point:
        return -self.normal[1], self.normal[0]

    def u(self, p: Point) -> float:
        """Signed distance of p from the door plane, along the crossing axis."""
        return (p[0] - self.c[0]) * self.normal[0] + (p[1] - self.c[1]) * self.normal[1]

    def v(self, p: Point) -> float:
        """Offset of p from the door axis, along the door span."""
        return (p[0] - self.c[0]) * self.along[0] + (p[1] - self.c[1]) * self.along[1]

    def at(self, u: float, v: float) -> Point:
        return self.c[0] + u * self.normal[0] + v * self.along[0], self.c[1] + u * self.normal[1] + v * self.along[1]


@attrs.frozen
class DoorLeaf:
    """A world door: its name, the centre of its span and the unit normal of the span."""

    name: str
    c: Point
    normal: Point


@attrs.frozen
class Site:
    """World walls, doors and zones, the managed doors and the parked robot starts of one level."""

    walls: tuple[Wall, ...]
    leaves: tuple[DoorLeaf, ...]
    doors: dict[str, Door]
    zones: dict[str, tuple[Point, ...]]
    starts: tuple[Point, ...]
    level: str

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
            length = math.dist(a, b)
            leaves.append(DoorLeaf(str(d["name"]), ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), ((b[1] - a[1]) / length, (a[0] - b[0]) / length)))
        zones = {str(z["name"]): tuple(_point(p) for p in z["polygon"]) for z in lv.get("zones", [])}
        doors: dict[str, Door] = {}
        for r in _mapping(resources)["resources"]:
            if r["kind"] != "door":
                continue
            name = str(r["id"])
            leaf = next((x for x in leaves if x.name == name), None)
            if leaf is None:
                raise ValueError(f"{resources}: {world} level {level} has no door {name}")
            holding = [z for z in zones.values() if inside(leaf.c, z)]
            if not holding:
                raise ValueError(f"{world}: no zone of level {level} holds door {name}")
            doors[name] = Door(name, min(holding, key=area), leaf.normal)
        if not doors:
            raise ValueError(f"{resources}: no resource of kind door")
        starts = tuple(_point(r["start"]) for r in _mapping(scenario)["robots"])
        return cls(walls=tuple(walls), leaves=tuple(leaves), doors=doors, zones=zones, starts=starts, level=level)


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
