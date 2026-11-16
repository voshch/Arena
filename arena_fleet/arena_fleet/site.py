import math
import pathlib
import typing

import attrs
import yaml
from arena_simulation_setup.tree.World.model import World

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
    """The world level that holds the managed doors: its walls, doors and zones, the managed doors and the goal zones."""

    world: str
    level: str
    walls: tuple[Wall, ...]
    leaves: tuple[DoorLeaf, ...]
    doors: dict[str, Door]
    zones: dict[str, tuple[Point, ...]]
    goal_zones: tuple[str, ...]

    @classmethod
    def load(cls, world: World, name: str, resources: pathlib.Path) -> "Site":
        """The site of the parsed world `name` as resources.yaml manages it: its managed doors fix the level."""
        data = _mapping(resources)
        if data["world"] != name:
            raise ValueError(f"{resources} is written for world {data['world']}, not {name}")
        managed = [str(r["id"]) for r in data["resources"] if r["kind"] == "door"]
        if not managed:
            raise ValueError(f"{resources}: no resource of kind door")
        levels = [lid for lid, lv in world.levels.items() if set(managed) <= {d.name for d in lv.doors}]
        if len(levels) != 1:
            raise ValueError(f"{resources}: the managed doors {managed} must all lie on one level of world {name}, they lie on {levels}")
        level = levels[0]
        lv = world.levels[level]
        walls = tuple(((w.start.x, w.start.y), (w.end.x, w.end.y)) for w in lv.all_walls)
        leaves: list[DoorLeaf] = []
        for d in lv.doors:
            a, b = (d.start.x, d.start.y), (d.end.x, d.end.y)
            length = math.dist(a, b)
            leaves.append(DoorLeaf(d.name, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), ((b[1] - a[1]) / length, (a[0] - b[0]) / length)))
        zones = {z.name: tuple((p.x, p.y) for p in z.polygon) for z in lv.zones}
        doors: dict[str, Door] = {}
        for door in managed:
            leaf = next(x for x in leaves if x.name == door)
            holding = [z for z in zones.values() if inside(leaf.c, z)]
            if not holding:
                raise ValueError(f"no zone of level {level} of world {name} holds door {door}")
            doors[door] = Door(door, min(holding, key=area), leaf.normal)
        goal_zones = tuple(str(z) for z in data["goal_zones"])
        unknown = sorted(set(goal_zones) - set(zones))
        if unknown:
            raise ValueError(f"{resources}: goal zones {unknown} are not zones of level {level}, which has {sorted(zones)}")
        return cls(world=name, level=level, walls=walls, leaves=tuple(leaves), doors=doors, zones=zones, goal_zones=goal_zones)


def _mapping(path: pathlib.Path) -> dict[str, typing.Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return typing.cast(dict[str, typing.Any], data)
