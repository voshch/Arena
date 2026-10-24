"""ProcTHOR-10K houses as worlds of typed rooms, walls, doors and furniture, Unity x,z,y mapped to Arena x,y,z with the house origin kept."""

from __future__ import annotations

import gzip
import json
import logging
import math
import re
import shutil
import typing
from collections import Counter
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path

from arena_assets.importers import IMPORTERS, Attribution, Importer, import_root
from arena_assets.importers.providers import Http, Provider
from arena_simulation_setup.generative.world.utils import world_of
from arena_simulation_setup.shared import Door, Obstacle
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.World import Level, WallPath, WorldIdentifier, WorldView, Zone, derive
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position, Vector3

if typing.TYPE_CHECKING:
    from arena_assets.importers.ai2thor import AI2THOR

BASE_URL = "https://media.githubusercontent.com/media/allenai/procthor-10k/main"
SPLITS = {"train": 10_000, "val": 1_000, "test": 1_000}
LICENSE = "Apache-2.0"
AUTHORS = "Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Jordi Salvador, Kiana Ehsani, Winson Han, Eric Kolve, Ali Farhadi, Aniruddha Kembhavi, Roozbeh Mottaghi (Allen Institute for AI)"
CITATION = "Deitke et al., ProcTHOR: Large-Scale Embodied AI Using Procedural Generation, NeurIPS 2022"
DOOR_KIND = "hinged"
OBJECTS = "ai2thor"
OBJECT_DOMAIN = "Common"
EPS_M = 1e-3

logger = logging.getLogger(__name__)

_WORD_START = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

Point = tuple[float, float]
Segment = tuple[Point, Point]


def split_item(item: str) -> tuple[str, int]:
    split, _, index = item.partition("/")
    if split not in SPLITS or not index.isdigit() or int(index) >= SPLITS[split]:
        raise ValueError(f"no ProcTHOR-10K house '{item}', items are <split>/<index> with split in {', '.join(SPLITS)}")
    return split, int(index)


def world_name(item: str) -> str:
    split, index = split_item(item)
    return f"procthor_{split}_{index}"


def zone_name(room: dict) -> str:
    """Room type in snake case, the spelling of the ASA room facet, with the upstream room number."""
    return f"{_WORD_START.sub('_', room['roomType']).lower()}_{room['id'].rpartition('|')[2]}"


class Houses:
    """Forward cursor over one split file, reopened only to step back."""

    def __init__(self, src: Path):
        self.src = src
        self._lines: Generator[str, None, None] | None = None
        self._next = 0

    def house(self, index: int) -> dict:
        if self._lines is None or index < self._next:
            self.close()
            self._lines = _lines(self.src)
            self._next = 0
        for line in self._lines:
            self._next += 1
            if self._next > index:
                return json.loads(line)
        self.close()
        raise IndexError(f"{self.src} has no house {index}")

    def close(self) -> None:
        if self._lines is not None:
            self._lines.close()
            self._lines = None


def _lines(src: Path) -> Generator[str, None, None]:
    if src.suffix == ".gz":
        with gzip.open(src, "rt") as f:
            yield from f
    else:
        with open(src) as f:
            yield from f


def _point(p: dict) -> Point:
    return (round(float(p["x"]), 4), round(float(p["z"]), 4))


def _ground(polygon: list[dict]) -> Segment:
    a, b = sorted(polygon, key=lambda p: p["y"])[:2]
    return _point(a), _point(b)


def _canonical(segment: Segment) -> tuple[Segment, bool]:
    start, end = segment
    if start <= end:
        return (start, end), False
    return (end, start), True


def _lerp(segment: Segment, t: float) -> Position:
    (x0, y0), (x1, y1) = segment
    return Position(round(x0 + t * (x1 - x0), 4), round(y0 + t * (y1 - y0), 4), 0.0)


def _name(upstream_id: str) -> str:
    return upstream_id.replace("|", "_")


def _merged(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _pieces(length: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in _merged(holes):
        if a - cursor > EPS_M:
            pieces.append((cursor, a))
        cursor = max(cursor, b)
    if length - cursor > EPS_M:
        pieces.append((cursor, length))
    return pieces


def level(house: dict) -> tuple[Level, dict[str, str]]:
    """The house as one level, rooms as zones with their room type as description, walls cut around every door hole."""
    zones = [Zone(name=zone_name(room), polygon=[Position(*_point(p), 0.0) for p in room["floorPolygon"]]) for room in house["rooms"]]
    descriptions = {zone.name: room["roomType"] for zone, room in zip(zones, house["rooms"], strict=True)}
    segments = dict.fromkeys(_canonical(_ground(wall["polygon"]))[0] for wall in house["walls"] if not wall.get("empty", False))

    walls_by_id = {wall["id"]: wall for wall in house["walls"]}
    holes: dict[Segment, list[tuple[float, float]]] = {}
    doors: list[Door] = []
    for door in house["doors"]:
        openable = door.get("openable")
        if openable is False:
            continue
        segment, flipped = _canonical(_ground(walls_by_id[door["wall0"]]["polygon"]))
        length = math.dist(*segment)
        a = min(max(float(door["holePolygon"][0]["x"]), 0.0), length)
        b = min(max(float(door["holePolygon"][1]["x"]), 0.0), length)
        if flipped:
            a, b = length - b, length - a
        if b - a < EPS_M:
            continue
        holes.setdefault(segment, []).append((a, b))
        if openable:
            doors.append(Door(name=_name(door["id"]), start=_lerp(segment, a / length), end=_lerp(segment, b / length), kind=DOOR_KIND))

    walls = [WallPath(path=[_lerp(segment, a / math.dist(*segment)), _lerp(segment, b / math.dist(*segment))]) for segment in segments for a, b in _pieces(math.dist(*segment), holes.get(segment, []))]
    return Level(zones=zones, walls=walls, doors=doors), descriptions


def house_objects(house: dict) -> Iterator[dict]:
    """Every placed object, receptacle children included, in upstream order."""
    stack = list(reversed(house["objects"]))
    while stack:
        obj = stack.pop()
        yield obj
        stack.extend(reversed(obj.get("children", [])))


def model(asset_id: str) -> ObjectIdentifier:
    return ObjectIdentifier(name=asset_id, domain=OBJECT_DOMAIN)


def pose(obj: dict, center: tuple[float, float, float]) -> Pose:
    """Arena pose of an object whose built model has its bounding box centre at *center*, from the Unity bounding box centre and ZXY Euler angles."""
    rotation = obj["rotation"]
    orientation = Orientation.from_yaw(-math.radians(rotation["y"])) * Orientation.from_euler((-math.radians(rotation["x"]), 0.0, 0.0)) * Orientation.from_euler((0.0, -math.radians(rotation["z"]), 0.0)) * Orientation.from_yaw(math.pi)
    offset = orientation * Vector3(*center)
    position = obj["position"]
    return Pose(
        position=Position(round(position["x"] - offset.x, 4), round(position["z"] - offset.y, 4), round(position["y"] - offset.z, 4)),
        orientation=Orientation(*(round(value, 6) for value in orientation)),
    )


def furnish(level: Level, house: dict, centers: dict[str, tuple[float, float, float]]) -> None:
    """Place every object whose model has a centre on the level, named <asset>_<n> as native worlds name entities."""
    counts: Counter[str] = Counter()
    for obj in house_objects(house):
        asset = obj["assetId"]
        if asset not in centers:
            continue
        level.objects.append(Obstacle(name=f"{asset}_{counts[asset]}", model=model(asset), pose=pose(obj, centers[asset])))
        counts[asset] += 1


class ProcTHOR(Importer):
    meta = IMPORTERS.meta("world", "procthor")

    def __init__(self, cache: Path | None = None, provider: Provider | None = None, objects: AI2THOR | None = None, objects_out: Path | None = None):
        super().__init__(cache, provider)
        self._houses: Houses | None = None
        self._objects = objects
        self._objects_out = objects_out

    @property
    def objects(self) -> AI2THOR:
        if self._objects is None:
            self._objects = typing.cast("AI2THOR", IMPORTERS.load("object", OBJECTS)())
        return self._objects

    @property
    def objects_out(self) -> Path:
        return self._objects_out if self._objects_out is not None else type(self.objects).default_out()

    def centers(self, assets: set[str]) -> dict[str, tuple[float, float, float]]:
        """Bounding box centre of each asset's native model, importing those no resolver holds yet."""
        if not assets:
            return {}
        available = {identifier.shortname for identifier in ObjectIdentifier.listall(network=True)}
        missing = sorted(asset for asset in assets if model(asset).shortname not in available)
        imported = self.objects.import_items(missing, self.objects_out) if missing else {}
        centers: dict[str, tuple[float, float, float]] = {}
        for asset in sorted(assets):
            if asset in missing and asset not in imported:
                continue
            try:
                bbox = model(asset).resolve_sync().bbox
            except FileNotFoundError:
                if asset in imported:
                    raise RuntimeError(f"imported {asset} to {imported[asset]}, but no object resolver reads {self.objects_out}") from None
                logger.warning("%s is listed but does not resolve, not placed", model(asset).shortname)
                continue
            if bbox is None:
                logger.warning("%s has no bounding box, not placed", model(asset).shortname)
                continue
            centers[asset] = bbox[1]
        return centers

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def items(self) -> Iterator[str]:
        for split, size in SPLITS.items():
            for index in range(size):
                yield f"{split}/{index}"

    def fetch(self, item: str) -> Path:
        split, _ = split_item(item)
        return self.provider.get(f"{split}.jsonl.gz", self.cache / f"{split}.jsonl.gz")

    def house(self, src: Path, index: int) -> dict:
        if self._houses is None or self._houses.src != src:
            self.close()
            self._houses = Houses(src)
        return self._houses.house(index)

    def close(self) -> None:
        if self._houses is not None:
            self._houses.close()

    def import_items(self, items: Iterable[str], out: Path) -> dict[str, Path]:
        try:
            return super().import_items(items, out)
        finally:
            self.close()

    def convert(self, item: str, src: Path, out: Path) -> str:
        _, index = split_item(item)
        identifier = WorldIdentifier(world_name(item))
        target = out / identifier.relpath()
        if target.exists():
            shutil.rmtree(target)
        house = self.house(src, index)
        floor, descriptions = level(house)
        furnish(floor, house, self.centers({obj["assetId"] for obj in house_objects(house)}))
        built = world_of(floor, descriptions)
        built.kg = derive(built)
        WorldView(target).save(built)
        return str(identifier.relpath())

    def attribution(self, item: str) -> Attribution:
        split, _ = split_item(item)
        return Attribution(source=self.meta.name, url=self.provider.url(f"{split}.jsonl.gz"), license=LICENSE, authors=AUTHORS, citation=CITATION)
