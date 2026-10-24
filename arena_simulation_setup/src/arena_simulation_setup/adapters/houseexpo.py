"""HouseExpo floor plans as worlds: the vertex ring is the free space, its outline the walls, the room boxes the zones."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import shapely

from arena_assets.importers import IMPORTERS, Attribution, Importer, import_root
from arena_assets.importers.providers import Http, Provider
from arena_simulation_setup.generative.world.layout import Area, Layout, Region, compile_layout
from arena_simulation_setup.tree.World import World, WorldView, derive

REPO = "TeaganLi/HouseExpo"
REVISION = "45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/HouseExpo"
ARCHIVE = "json.tar.gz"
INDEX = "index.json"
PLANS = "json"
LICENSE = "MIT (repository), floor plans built on the SUNCG dataset"
AUTHORS = "Tingguang Li (The Chinese University of Hong Kong), Danny Ho, Chenming Li, Delong Zhu, Chaoqun Wang, Max Q.-H. Meng"
CITATION = "Li et al., HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms on Mobile Robots, arXiv:1903.09845, 2019"
GRID_M = 0.001
NARROW_M = 0.3

_ID = re.compile(r"[0-9a-f]{32}")
_MEMBER = re.compile(r"json/([0-9a-f]{32})\.json")


def plan_id(item: str) -> str:
    if not _ID.fullmatch(item):
        raise ValueError(f"no HouseExpo plan '{item}', items are plan ids of 32 hex characters")
    return item


def world_name(item: str) -> str:
    return f"houseexpo_{plan_id(item)}"


class Plans:
    """Plans of the archive by id, through an index written beside it and one gzip stream that rewinds only to step back."""

    def __init__(self, archive: Path):
        self.archive = archive
        self._index: dict[str, list[int]] | None = None
        self._stream: gzip.GzipFile | None = None

    @property
    def index(self) -> dict[str, list[int]]:
        """Offset and size of every plan in the unpacked stream, in archive order."""
        if self._index is not None:
            return self._index
        path = self.archive.with_name(INDEX)
        if not path.is_file():
            with tarfile.open(self.archive, "r:gz") as tar:
                built = {match[1]: [member.offset_data, member.size] for member in tar if member.isfile() and (match := _MEMBER.fullmatch(member.name))}
            part = path.with_name(path.name + ".part")
            part.write_text(json.dumps(built))
            part.replace(path)
        index: dict[str, list[int]] = json.loads(path.read_text())
        self._index = index
        return index

    def read(self, item: str) -> bytes:
        if item not in self.index:
            raise ValueError(f"no HouseExpo plan '{item}' among the {len(self.index)} of {self.archive.name}")
        offset, size = self.index[item]
        if self._stream is None:
            self._stream = gzip.open(self.archive)
        self._stream.seek(offset)
        return self._stream.read(size)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _polygons(geometry: shapely.Geometry) -> list[shapely.Polygon]:
    return [part for part in shapely.get_parts(shapely.get_parts(geometry)) if isinstance(part, shapely.Polygon) and part.area > 0]


def _area(geometry: shapely.Geometry) -> shapely.Polygon | shapely.MultiPolygon:
    """The polygons of a geometry as one area."""
    parts = _polygons(geometry)
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


def floor(plan: dict) -> list[shapely.Polygon]:
    """Free space of a plan: its vertex ring without zero-width spikes, split where it touches itself."""
    verts = plan["verts"]
    repaired = shapely.make_valid(shapely.Polygon(verts)) if len(verts) > 2 else shapely.Polygon()
    parts = [clean for part in _polygons(repaired) for clean in _polygons(shapely.set_precision(part, GRID_M))]
    if not parts:
        raise ValueError(f"HouseExpo plan '{plan['id']}' has no floor, its {len(verts)} vertices enclose no area")
    return parts


def boxes(plan: dict) -> list[Region]:
    """Room boxes smallest first, a box under several categories once."""
    labels: dict[tuple[float, ...], list[tuple[str, int]]] = {}
    for category, found in plan["room_category"].items():
        for index, box in enumerate(found):
            labels.setdefault(tuple(box), []).append((category, index))
    regions: list[tuple[float, tuple[str, int], Region]] = []
    for box, found in labels.items():
        category, index = min(found)
        x0, y0, x1, y1 = box
        polygon = shapely.box(x0, y0, x1, y1)
        regions.append((polygon.area, (category, index), Region(name=f"{category.lower()}_{index}", polygon=polygon, description=", ".join(sorted({category for category, _ in found})))))
    return [region for _, _, region in sorted(regions, key=lambda entry: entry[:2])]


def rooms(free: list[shapely.Polygon], claims: list[Region]) -> list[Region]:
    """Floor each box keeps after the boxes before it, pieces narrower than NARROW_M joined to the neighbor sharing the longest edge."""
    whole = _area(shapely.union_all(free))
    wide: list[tuple[int, shapely.Polygon | shapely.MultiPolygon]] = []
    narrow: list[tuple[int, shapely.Polygon]] = []
    taken: list[shapely.Polygon | shapely.MultiPolygon] = []
    for owner, claim in enumerate([*claims, Region(name="", polygon=whole)]):
        piece = claim.polygon.difference(shapely.union_all(taken), grid_size=GRID_M) if taken else claim.polygon
        for part in _polygons(piece.intersection(whole, grid_size=GRID_M)):
            if part.buffer(-NARROW_M / 2).is_empty:
                narrow.append((owner, part))
            else:
                wide.append((owner, part))
        taken.append(claim.polygon)
    for owner, part in narrow:
        shared = shapely.length(shapely.intersection([host for _, host in wide], part, grid_size=GRID_M))
        if len(shared) and shared.max() > 0:
            host = int(shared.argmax())
            wide[host] = (wide[host][0], _area(wide[host][1].union(part, grid_size=GRID_M)))
        else:
            wide.append((owner, part))
    return [Region(name=claim.name, polygon=_area(shapely.union_all(parts)), description=claim.description) for owner, claim in enumerate(claims) if (parts := [part for host, part in wide if host == owner])]


def world(plan: dict) -> World:
    free = floor(plan)
    return compile_layout(Layout(areas=[Area(part) for part in free], regions=rooms(free, boxes(plan))), ceiling=True, grid=GRID_M)[0]


class HouseExpo(Importer):
    meta = IMPORTERS.meta("world", "houseexpo")

    def __init__(self, cache: Path | None = None, provider: Provider | None = None):
        super().__init__(cache, provider)
        self._plans: Plans | None = None

    @property
    def plans(self) -> Plans:
        if self._plans is None:
            self._plans = Plans(self.provider.get(ARCHIVE, self.cache / ARCHIVE))
        return self._plans

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def items(self) -> Iterator[str]:
        return iter(self.plans.index)

    def close(self) -> None:
        if self._plans is not None:
            self._plans.close()

    def import_items(self, items: Iterable[str], out: Path) -> dict[str, Path]:
        try:
            return super().import_items(items, out)
        finally:
            self.close()

    def fetch(self, item: str) -> Path:
        dest = self.cache / PLANS / f"{plan_id(item)}.json"
        if dest.is_file():
            return dest
        data = self.plans.read(item)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(data)
        part.replace(dest)
        return dest

    def convert(self, item: str, src: Path, out: Path) -> str:
        name = world_name(item)
        built = world(json.loads(src.read_text()))
        built.kg = derive(built)
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        WorldView(target).save(built)
        return name

    def attribution(self, item: str) -> Attribution:
        plan_id(item)
        return Attribution(source=self.meta.name, url=self.provider.url(ARCHIVE), license=LICENSE, authors=AUTHORS, citation=CITATION)
