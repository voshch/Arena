"""Deterministic acoustics scenario generation.

The generator intentionally treats the checked-in acoustics world tree as input.
Only scenario directories carrying :data:`GENERATED_PREFIX` and a matching
``metadata.yaml`` marker are ever considered generated content.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import warnings
import wave
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import yaml

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

GENERATED_PREFIX = "hearing__"
GENERATED_BY = "arena_acoustics_generator_v1"
GENERATOR_VERSION = "1.0.0"

HEADING_YAWS = {
    "front": math.pi / 2.0,
    "left": math.pi,
    "back": -math.pi / 2.0,
    "right": 0.0,
}


def _stable_int(*parts: object) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _stable_digest(*parts: object, length: int = 10) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the half-open interval ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def relative_bearing(robot_xy: Sequence[float], robot_yaw: float, pedestrian_xy: Sequence[float]) -> float:
    return wrap_to_pi(math.atan2(pedestrian_xy[1] - robot_xy[1], pedestrian_xy[0] - robot_xy[0]) - robot_yaw)


def bearing_class(angle: float) -> str:
    """Classify relative bearing using exact 45/135 degree boundaries.

    ``front=[-45,45)``, ``left=[45,135)``, ``back=[135,180) U
    [-180,-135)``, and ``right=[-135,-45)``.
    """
    value = wrap_to_pi(angle)
    quarter = math.pi / 4.0
    if -quarter <= value < quarter:
        return "front"
    if quarter <= value < 3.0 * quarter:
        return "left"
    if -3.0 * quarter <= value < -quarter:
        return "right"
    return "back"


def parse_duration(value: str | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)].strip()) * multiplier
    raise ValueError(f"invalid duration: {value!r}")


@dataclass(frozen=True)
class DiscoveryReport:
    root: str
    expected_count: int
    discovered_count: int
    selected_count: int
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    malformed: tuple[dict[str, str], ...]
    count_matches: bool

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class WorldCountError(ValueError):
    def __init__(self, report: DiscoveryReport):
        self.report = report
        super().__init__(f"expected {report.expected_count} acoustics worlds, discovered {report.discovered_count}; included={report.selected_count}, excluded={len(report.excluded)}, malformed={len(report.malformed)}")


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def discover_worlds(
    root: str | Path,
    *,
    expected_count: int = 99,
    include: Sequence[str] = ("*",),
    exclude: Sequence[str] = (),
    strict: bool = False,
) -> tuple[list[Path], DiscoveryReport]:
    """Discover world directories from ``<world>/0/world.yaml``.

    Expected-count validation is applied before include/exclude filtering so a
    targeted smoke run does not hide a damaged or incomplete source dataset.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"acoustics worlds root does not exist: {root_path}")

    valid: list[Path] = []
    malformed: list[dict[str, str]] = []
    for entry in sorted((item for item in root_path.iterdir() if item.is_dir()), key=lambda p: p.name):
        world_yaml = entry / "0" / "world.yaml"
        missing = [rel for rel in ("0/world.yaml", "0/map.yaml", "0/map.png") if not (entry / rel).is_file()]
        if not world_yaml.is_file():
            malformed.append({"world": entry.name, "reason": "missing 0/world.yaml"})
            continue
        if missing:
            malformed.append({"world": entry.name, "reason": "missing " + ", ".join(missing)})
        valid.append(entry)

    selected: list[Path] = []
    excluded_names: list[str] = []
    for path in valid:
        if not _matches(path.name, include) or _matches(path.name, exclude):
            excluded_names.append(path.name)
        else:
            selected.append(path)

    report = DiscoveryReport(
        root=str(root_path),
        expected_count=int(expected_count),
        discovered_count=len(valid),
        selected_count=len(selected),
        included=tuple(path.name for path in selected),
        excluded=tuple(excluded_names),
        malformed=tuple(malformed),
        count_matches=len(valid) == int(expected_count),
    )
    if strict and not report.count_matches:
        raise WorldCountError(report)
    return selected, report


@dataclass
class WorldGeometry:
    world: str
    zone_names: list[str]
    zone_polygons: list[Any]
    centers: list[tuple[float, float]]
    adjacency: dict[int, list[int]]
    navigable: Any
    robot_navigable: Any
    walls: Any
    routes: list[list[tuple[float, float]]]
    cycle_routes: list[list[tuple[float, float]]]
    corners: list[list[tuple[float, float]]]
    topology: str
    width_class: str
    angled: bool
    warnings: list[str] = field(default_factory=list)

    def attributes(self) -> dict[str, Any]:
        return {
            "topology": self.topology,
            "width_class": self.width_class,
            "angled": self.angled,
            "zone_count": len(self.zone_names),
            "route_count": len(self.routes),
            "cycle_route_count": len(self.cycle_routes),
            "corner_count": len(self.corners),
        }


def _repository_validate_world(world_dir: Path) -> None:
    """Validate the world structure without importing ROS-facing packages.

    Scenario generation is a source-tree operation. Importing ``World.py``
    pulls in asset resolvers backed by ament/ROS even though geometry extraction
    only needs the level YAML. Keep the same required structural contract here
    and leave the full runtime-schema round trip to ROS integration tests.
    """
    path = world_dir / "0" / "world.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    zones = raw.get("zones") if isinstance(raw, dict) else raw
    if not isinstance(zones, list) or not zones:
        raise ValueError(f"{path} must contain a non-empty zones list")
    for index, zone in enumerate(zones):
        if not isinstance(zone, dict):
            raise ValueError(f"{path}: zone {index} must be a mapping")
        corners = zone.get("corners")
        if not isinstance(corners, list) or len(corners) < 3:
            raise ValueError(f"{path}: zone {index} must contain at least three corners")
        for corner_index, corner in enumerate(corners):
            if not isinstance(corner, dict) or not {"x", "y"} <= corner.keys():
                raise ValueError(f"{path}: zone {index} corner {corner_index} must contain x and y")
            float(corner["x"])
            float(corner["y"])


def _points_from_geometry(geometry: BaseGeometry) -> list[tuple[float, float]]:
    from shapely.geometry.base import BaseMultipartGeometry

    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [(float(x), float(y)) for x, y in geometry.coords]
    if geometry.geom_type == "Point":
        return [(float(geometry.x), float(geometry.y))]
    if not isinstance(geometry, BaseMultipartGeometry):
        return []
    result: list[tuple[float, float]] = []
    for part in geometry.geoms:
        result.extend(_points_from_geometry(part))
    return result


def _minimum_rotated_rectangle(geometry: BaseGeometry) -> BaseGeometry:
    # GEOS may emit benign floating-point warnings while computing an oriented
    # envelope even when it returns a valid rectangle.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return geometry.minimum_rotated_rectangle


def _centerline(poly: BaseGeometry, clearance: float) -> list[tuple[float, float]] | None:
    from shapely.geometry import LineString

    safe = poly.buffer(-clearance)
    if safe.is_empty:
        safe = poly
    rectangle = _minimum_rotated_rectangle(safe)
    coords = list(rectangle.exterior.coords)[:-1]
    if len(coords) < 4:
        return None
    edges = [(math.dist(coords[index], coords[(index + 1) % len(coords)]), coords[index], coords[(index + 1) % len(coords)]) for index in range(len(coords))]
    _, start, end = max(edges, key=lambda item: item[0])
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    dx, dy = dx / norm, dy / norm
    center = safe.representative_point()
    span = max(safe.bounds[2] - safe.bounds[0], safe.bounds[3] - safe.bounds[1], 1.0) * 3.0
    axis = LineString([(center.x - dx * span, center.y - dy * span), (center.x + dx * span, center.y + dy * span)])
    points = _points_from_geometry(axis.intersection(safe))
    if len(points) < 2:
        return None
    pair = max(itertools.combinations(points, 2), key=lambda value: math.dist(*value))
    length = math.dist(*pair)
    if length <= 0.2:
        return None
    inset = min(0.1, length / 4.0)
    ux = (pair[1][0] - pair[0][0]) / length
    uy = (pair[1][1] - pair[0][1]) / length
    return [
        (pair[0][0] + ux * inset, pair[0][1] + uy * inset),
        (pair[1][0] - ux * inset, pair[1][1] - uy * inset),
    ]


def _bfs(adjacency: dict[int, list[int]], start: int, goal: int) -> list[int] | None:
    queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, [*path, nxt]))
    return None


def _route_length(route: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False))


def _route_inside(route: Sequence[Sequence[float]], polygon: BaseGeometry, radius: float) -> bool:
    from shapely.geometry import LineString

    if len(route) < 2 or _route_length(route) <= 0.1:
        return False
    line = LineString(route)
    return polygon.buffer(1e-6).covers(line.buffer(max(radius, 0.01), cap_style=2, join_style=2))


def _find_graph_cycle(adjacency: dict[int, list[int]]) -> list[int] | None:
    visited: set[int] = set()

    def visit(node: int, parent: int, path: list[int]) -> list[int] | None:
        visited.add(node)
        for nxt in adjacency[node]:
            if nxt == parent:
                continue
            if nxt in path:
                start = path.index(nxt)
                return [*path[start:], nxt]
            if nxt not in visited:
                result = visit(nxt, node, [*path, nxt])
                if result:
                    return result
        return None

    for node in sorted(adjacency):
        if node not in visited:
            result = visit(node, -1, [node])
            if result:
                return result
    return None


def load_world_geometry(world_dir: str | Path, clearances: dict[str, float]) -> WorldGeometry:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    path = Path(world_dir)
    _repository_validate_world(path)
    raw = yaml.safe_load((path / "0" / "world.yaml").read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("zones")
    if not isinstance(raw, list):
        raise ValueError("level world.yaml must contain a top-level zones list")

    names: list[str] = []
    polygons: list[Any] = []
    wall_segments: list[Any] = []
    angled = False
    for index, zone in enumerate(raw):
        corners = [(float(item["x"]), float(item["y"])) for item in zone.get("corners", [])]
        if len(corners) < 3:
            continue
        poly = Polygon(corners)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        names.append(str(zone.get("name", f"zone_{index:02d}")))
        polygons.append(poly)
        for wall in zone.get("walls", []):
            a, b = wall.get("start", {}), wall.get("end", {})
            if {"x", "y"} <= a.keys() and {"x", "y"} <= b.keys():
                wall_segments.append(LineString([(float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"]))]))
        for a, b in zip(corners, [*corners[1:], corners[0]], strict=False):
            angle = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 90.0
            if 2.0 < angle < 88.0:
                angled = True
    if not polygons:
        raise ValueError("world has no interpretable navigable zone polygons")

    floor = unary_union(polygons).buffer(0)
    ped_clearance = float(clearances.get("pedestrian_wall", 0.25))
    robot_clearance = float(clearances.get("jackal_wall", 0.55))
    navigable = floor.buffer(-ped_clearance)
    robot_navigable = floor.buffer(-robot_clearance)
    if navigable.is_empty or robot_navigable.is_empty:
        raise ValueError("world disappears after configured clearance erosion")

    centers: list[tuple[float, float]] = []
    local_lines: list[list[tuple[float, float]] | None] = []
    for polygon in polygons:
        safe = polygon.buffer(-ped_clearance)
        point = (safe if not safe.is_empty else polygon).representative_point()
        centers.append((float(point.x), float(point.y)))
        local_lines.append(_centerline(polygon, ped_clearance))

    connection_tolerance = float(clearances.get("zone_connection", 0.2))
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(polygons))}
    for left, right in itertools.combinations(range(len(polygons)), 2):
        distance = polygons[left].distance(polygons[right])
        shared_length = polygons[left].boundary.intersection(polygons[right].boundary).length
        overlap_area = polygons[left].intersection(polygons[right]).area
        connected_boundary = shared_length > connection_tolerance or overlap_area > connection_tolerance**2
        connected_gap = 1e-9 < distance <= connection_tolerance
        if connected_boundary or connected_gap:
            connector = LineString([centers[left], centers[right]])
            if floor.buffer(connection_tolerance).covers(connector):
                adjacency[left].append(right)
                adjacency[right].append(left)
    for neighbors in adjacency.values():
        neighbors.sort()

    routes: list[list[tuple[float, float]]] = []
    cycle_routes: list[list[tuple[float, float]]] = []
    for line in local_lines:
        if line and _route_inside(line, navigable, 0.05):
            routes.extend([line, list(reversed(line))])
    terminals = [index for index, edges in adjacency.items() if len(edges) <= 1]
    endpoints = terminals if len(terminals) >= 2 else list(adjacency)
    for left, right in itertools.combinations(endpoints, 2):
        indices = _bfs(adjacency, left, right)
        if not indices:
            continue
        route = [centers[index] for index in indices]
        if _route_inside(route, navigable, 0.05):
            routes.extend([route, list(reversed(route))])
    cycle = _find_graph_cycle(adjacency)
    if cycle:
        cycle_route = [centers[index] for index in cycle]
        if _route_inside(cycle_route, navigable, 0.05):
            cycle_routes = [cycle_route, list(reversed(cycle_route))]
            routes.extend(cycle_routes)
    routes.sort(key=lambda route: (-_route_length(route), tuple(route)))
    unique: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for route in routes:
        key = tuple((round(x, 5), round(y, 5)) for x, y in route)
        if key not in seen:
            seen.add(key)
            unique.append(route)
    routes = unique
    if not routes:
        raise ValueError("no clearance-valid centerline route could be derived")

    corners: list[list[tuple[float, float]]] = []
    for route in routes:
        for index in range(1, len(route) - 1):
            a, b, c = route[index - 1 : index + 2]
            first = math.atan2(b[1] - a[1], b[0] - a[0])
            second = math.atan2(c[1] - b[1], c[0] - b[0])
            turn = abs(wrap_to_pi(second - first))
            if math.radians(20) <= turn <= math.radians(160):
                corners.append([a, b, c])

    degrees = [len(edges) for edges in adjacency.values()]
    if any(degree >= 4 for degree in degrees):
        topology = "multi_arm"
    elif any(degree == 3 for degree in degrees):
        topology = "junction"
    elif corners or angled:
        topology = "corner"
    elif len(polygons) == 1:
        topology = "corridor_or_room"
    else:
        topology = "connected"

    widths: list[float] = []
    for polygon in polygons:
        rect = list(_minimum_rotated_rectangle(polygon).exterior.coords)
        lengths = [math.dist(a, b) for a, b in zip(rect, rect[1:], strict=False) if math.dist(a, b) > 1e-6]
        if lengths:
            widths.append(min(lengths))
    low, high = (min(widths), max(widths)) if widths else (0.0, 0.0)
    if high - low > 1.0:
        width_class = "mixed"
    elif high < 2.25:
        width_class = "narrow"
    elif low >= 4.5:
        width_class = "wide"
    else:
        width_class = "standard"

    return WorldGeometry(
        world=path.name,
        zone_names=names,
        zone_polygons=polygons,
        centers=centers,
        adjacency=adjacency,
        navigable=navigable,
        robot_navigable=robot_navigable,
        walls=unary_union(wall_segments) if wall_segments else unary_union([]),
        routes=routes,
        cycle_routes=cycle_routes,
        corners=corners,
        topology=topology,
        width_class=width_class,
        angled=angled,
    )


def has_line_of_sight(geometry: WorldGeometry, a: Sequence[float], b: Sequence[float]) -> bool:
    from shapely.geometry import LineString

    line = LineString([a[:2], b[:2]])
    if not geometry.navigable.buffer(0.3).covers(line):
        return False
    if geometry.walls.is_empty:
        return True
    hit = line.intersection(geometry.walls)
    if hit.is_empty:
        return True
    # Endpoint contact with a wall is harmless; a positive-length overlap or an
    # interior point intersection is an occlusion.
    for point in _points_from_geometry(hit):
        if math.dist(point, a[:2]) > 1e-4 and math.dist(point, b[:2]) > 1e-4:
            return False
    return hit.length <= 1e-6


@dataclass(frozen=True)
class Candidate:
    world: str
    count: int
    composition: str
    human_types: tuple[str, ...]
    models: tuple[str, ...]
    speeds: tuple[float, ...]
    speed_profile: str
    trajectory_pattern: str
    robot_state: str
    robot_heading: str
    robot_motion_relation: str
    seed: int
    role_variant: int
    naturalistic: bool = False

    @property
    def family_id(self) -> str:
        return "|".join(
            [
                self.world,
                str(self.count),
                ",".join(self.human_types),
                ",".join(self.models),
                self.speed_profile,
                self.trajectory_pattern,
                self.robot_state,
                self.robot_heading,
                self.robot_motion_relation,
                f"role={self.role_variant}",
            ]
        )

    @property
    def stable_id(self) -> str:
        return f"{self.family_id}|seed={self.seed}"

    @property
    def scenario_name(self) -> str:
        speed_name = {
            "all_slow": "slow",
            "all_fast": "fast",
            "alternating": "mixed_speed",
            "type_independent_seeded_mix": "seeded_mixed_speed",
            "naturalistic_type_conditioned": "naturalistic_speed",
        }.get(self.speed_profile, self.speed_profile)
        model_hash = _stable_digest(*self.models, length=6)
        return f"{GENERATED_PREFIX}n{self.count:02d}__{self.composition}__{self.trajectory_pattern}__{speed_name}__r{self.robot_state}__h{self.robot_heading}__m{model_hash}__s{self.seed:04d}"


def _composition_types(name: str, count: int, seed: int, role_variant: int) -> tuple[str, ...]:
    del seed, role_variant
    fixed = {
        "single": ["adult"],
        "pair": ["adult", "adult"],
        "group": ["adult"] * count,
    }
    values = list(fixed.get(name, []))
    if len(values) != count:
        raise ValueError(f"composition {name!r} does not produce {count} pedestrians")
    return tuple(values)


def _assign_speeds(
    types: Sequence[str],
    profile: str,
    values: Sequence[float],
    seed: int,
    naturalistic_priors: dict[str, Sequence[float]],
) -> tuple[float, ...]:
    slow, fast = float(values[0]), float(values[1])
    if profile == "all_slow":
        return tuple(slow for _ in types)
    if profile == "all_fast":
        return tuple(fast for _ in types)
    if profile == "alternating":
        return tuple(slow if index % 2 == 0 else fast for index in range(len(types)))
    if profile == "type_independent_seeded_mix":
        return tuple((slow, fast)[_stable_int(seed, index, "speed") % 2] for index in range(len(types)))
    if profile == "naturalistic_type_conditioned":
        return tuple(float(naturalistic_priors.get(human_type, (slow, fast))[_stable_int(seed, index) % 2]) for index, human_type in enumerate(types))
    raise ValueError(f"unknown speed profile: {profile}")


def _model_for(human_type: str, index: int, seed: int, mapping: dict[str, dict[str, Any]]) -> str:
    models = mapping[human_type].get("models", [])
    if not models:
        fallback = mapping[human_type].get("fallback_model")
        if not fallback:
            raise ValueError(f"human type {human_type!r} has no model or fallback_model")
        models = [fallback]
    return str(models[_stable_int(seed, index, human_type) % len(models)])


def _candidate_factors(candidate: Candidate) -> dict[str, str]:
    return {
        "count": str(candidate.count),
        "composition": candidate.composition,
        "types": ",".join(candidate.human_types),
        "models": ",".join(candidate.models),
        "speed_profile": candidate.speed_profile,
        "type_speed": ",".join(f"{t}:{s:g}" for t, s in zip(candidate.human_types, candidate.speeds, strict=False)),
        "trajectory": candidate.trajectory_pattern,
        "robot_state": candidate.robot_state,
        "heading": candidate.robot_heading,
        "motion_relation": candidate.robot_motion_relation,
        "role_variant": str(candidate.role_variant),
    }


def _pair_tokens(candidate: Candidate) -> set[str]:
    factors = _candidate_factors(candidate)
    return {f"{a}={factors[a]}|{b}={factors[b]}" for a, b in itertools.combinations(sorted(factors), 2)}


def stratified_pairwise_select(candidates: Sequence[Candidate], limit: int, seed: int) -> list[Candidate]:
    """Greedy deterministic pair coverage with marginal-level balancing."""
    remaining = []
    for item in sorted(candidates, key=lambda candidate: _stable_int(seed, candidate.stable_id)):
        factors = _candidate_factors(item)
        remaining.append(
            (
                item,
                _pair_tokens(item),
                tuple(f"{key}={value}" for key, value in factors.items()),
                _stable_int(seed, item.stable_id) & 0x7FFFFFFF,
            )
        )
    selected: list[Candidate] = []
    covered_pairs: set[str] = set()
    level_counts: Counter[str] = Counter()
    while remaining and len(selected) < limit:

        def score(entry: tuple[Candidate, set[str], tuple[str, ...], int]) -> tuple[int, int, int]:
            _item, pairs, levels, tie_break = entry
            return (
                len(pairs - covered_pairs),
                -sum(level_counts[level] for level in levels),
                -tie_break,
            )

        best = max(remaining, key=score)
        remaining.remove(best)
        item, pairs, levels, _tie_break = best
        selected.append(item)
        covered_pairs.update(pairs)
        level_counts.update(levels)
    return selected


def _candidate_template_key(candidate: Candidate) -> tuple[Any, ...]:
    """Return the factor identity shared by equivalent candidates in each world."""
    return (
        candidate.count,
        candidate.composition,
        candidate.human_types,
        candidate.models,
        candidate.speeds,
        candidate.speed_profile,
        candidate.trajectory_pattern,
        candidate.robot_state,
        candidate.robot_heading,
        candidate.robot_motion_relation,
        candidate.seed,
        candidate.role_variant,
        candidate.naturalistic,
    )


def counterbalanced_order(preferred: Sequence[Candidate], all_candidates: Sequence[Candidate]) -> list[Candidate]:
    """Place each mixed assignment next to its deterministic reversed variant."""

    def key(item: Candidate) -> tuple[Any, ...]:
        return (
            item.world,
            item.count,
            item.composition,
            item.speed_profile,
            item.trajectory_pattern,
            item.robot_state,
            item.robot_heading,
            item.seed,
        )

    lookup: dict[tuple[Any, ...], dict[int, Candidate]] = {}
    for item in all_candidates:
        lookup.setdefault(key(item), {})[item.role_variant] = item
    result: list[Candidate] = []
    seen: set[str] = set()
    for item in preferred:
        if item.stable_id in seen:
            continue
        result.append(item)
        seen.add(item.stable_id)
        if len(set(item.human_types)) > 1:
            counterpart = lookup.get(key(item), {}).get(1 - item.role_variant)
            if counterpart is not None and counterpart.stable_id not in seen:
                result.append(counterpart)
                seen.add(counterpart.stable_id)
    return result


def build_candidates(world: str, spec: dict[str, Any]) -> list[Candidate]:
    counts = [int(value) for value in spec["pedestrian_counts"]]
    compositions = spec["human_compositions"]
    patterns = spec["trajectory_patterns"]
    speed_profiles = list(spec["speed_profiles"])
    naturalistic_enabled = bool(spec.get("naturalistic", {}).get("enabled", False))
    if not naturalistic_enabled:
        speed_profiles = [name for name in speed_profiles if name != "naturalistic_type_conditioned"]
    seed_values = [int(value) for value in spec.get("seeds", [42])]
    repetitions = int(spec.get("repetitions", 1))
    mapping = spec["human_types"]
    speed_values = spec["speed_values"]
    priors = spec.get("naturalistic", {}).get("type_speed_values", {})

    candidates: list[Candidate] = []
    for count in counts:
        count_patterns = list(patterns[str(count)] if str(count) in patterns else patterns[count])
        for corner_pattern in spec.get("corner_scenarios", []):
            is_group = str(corner_pattern) == "multi_person_corner_burst"
            if (count == 5 and is_group) or (count == 1 and not is_group):
                if corner_pattern not in count_patterns:
                    count_patterns.append(corner_pattern)
        for composition, pattern, speed_profile, state, heading, base_seed, repetition, role_variant in itertools.product(
            compositions[str(count)] if str(count) in compositions else compositions[count],
            count_patterns,
            speed_profiles,
            spec["robot_states"],
            spec["robot_headings"],
            seed_values,
            range(repetitions),
            range(1),
        ):
            seed = base_seed + repetition
            human_types = _composition_types(str(composition), count, seed, role_variant)
            models = tuple(_model_for(kind, index, seed + role_variant, mapping) for index, kind in enumerate(human_types))
            speeds = _assign_speeds(human_types, str(speed_profile), speed_values, seed + role_variant, priors)
            relation = "stationary" if state == "idle" else ("forward" if _stable_int(seed, pattern) % 2 == 0 else "reverse")
            candidates.append(
                Candidate(
                    world=world,
                    count=count,
                    composition=str(composition),
                    human_types=human_types,
                    models=models,
                    speeds=speeds,
                    speed_profile=str(speed_profile),
                    trajectory_pattern=str(pattern),
                    robot_state=str(state),
                    robot_heading=str(heading),
                    robot_motion_relation=relation,
                    seed=seed,
                    role_variant=role_variant,
                    naturalistic=str(speed_profile) == "naturalistic_type_conditioned",
                )
            )
    return candidates


CORNER_PATTERNS = {
    "corner_emerge",
    "corner_disappear",
    "corner_cross_front",
    "visible_plus_occluded",
    "corner_burst",
    "occluded_to_visible",
    "visible_to_occluded",
    "hidden_cross_front",
    "robot_approaches_corner",
    "same_direction_around_corner",
    "opposing_around_corner",
    "multi_person_corner_burst",
}

MULTI_ARM_PATTERNS = {
    "different_sides_converging",
    "different_sides_crossing",
    "different_sides_same_exit",
    "multi_arm_crossing",
    "merge_shuffle",
    "split_shuffle",
}


def _interpolate_route(route: Sequence[Sequence[float]], fraction: float) -> tuple[float, float]:
    from shapely.geometry import LineString

    line = LineString(route)
    point = line.interpolate(max(0.0, min(1.0, fraction)), normalized=True)
    return float(point.x), float(point.y)


def _angle_distance(a: float, b: float) -> float:
    return abs(wrap_to_pi(a - b))


def _robot_segment(geometry: WorldGeometry, candidate: Candidate, tolerance: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
    target = HEADING_YAWS[candidate.robot_heading]
    desired_route_angle = target if candidate.robot_motion_relation == "forward" else wrap_to_pi(target + math.pi)
    choices: list[tuple[float, tuple[float, float], tuple[float, float]]] = []
    for route in geometry.routes:
        for raw_start, raw_end in zip(route, route[1:], strict=False):
            for trim in (0.0, 0.05, 0.1, 0.2, 0.3):
                start = (
                    raw_start[0] + (raw_end[0] - raw_start[0]) * trim,
                    raw_start[1] + (raw_end[1] - raw_start[1]) * trim,
                )
                end = (
                    raw_start[0] + (raw_end[0] - raw_start[0]) * (1.0 - trim),
                    raw_start[1] + (raw_end[1] - raw_start[1]) * (1.0 - trim),
                )
                if math.dist(start, end) <= 0.2 or not _route_inside([start, end], geometry.robot_navigable, 0.01):
                    continue
                angle = math.atan2(end[1] - start[1], end[0] - start[0])
                delta = _angle_distance(angle, desired_route_angle)
                if delta <= tolerance:
                    choices.append((delta, start, end))
                break
    if not choices:
        return None
    _, start, end = min(choices, key=lambda item: (item[0], -math.dist(item[1], item[2]), item[1], item[2]))
    return start, end


def _choose_ped_routes(geometry: WorldGeometry, candidate: Candidate) -> list[list[tuple[float, float]]]:
    routes = geometry.routes
    if candidate.trajectory_pattern == "loop":
        if not geometry.cycle_routes:
            raise ValueError("true loop is not clearance-valid in this world; loop skipped instead of forcing a retrace")
        routes = geometry.cycle_routes
    if candidate.trajectory_pattern in CORNER_PATTERNS:
        if not geometry.corners:
            raise ValueError("pattern requires a verified corner")
        base = geometry.corners[_stable_int(candidate.seed, candidate.trajectory_pattern) % len(geometry.corners)]
        routes = [base, list(reversed(base)), *routes]
    if candidate.trajectory_pattern in MULTI_ARM_PATTERNS and max(map(len, geometry.adjacency.values()), default=0) < 3:
        raise ValueError("pattern requires at least three connected arms")
    chosen: list[list[tuple[float, float]]] = []
    for index in range(candidate.count):
        route = list(routes[index % len(routes)])
        if "opposite" in candidate.trajectory_pattern or "opposing" in candidate.trajectory_pattern:
            if index % 2:
                route.reverse()
        chosen.append(route)
    return chosen


def _waypoint_mode(pattern: str) -> str:
    if pattern in {"one_way", "corner_emerge", "corner_disappear", "corner_cross_front", "occluded_to_visible", "visible_to_occluded", "hidden_cross_front"}:
        return "once"
    if pattern == "out_and_back" or "opposing_around_corner" in pattern:
        return "reverse"
    return "repeat"


def render_candidate(
    geometry: WorldGeometry,
    candidate: Candidate,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from shapely.geometry import Point

    clearance = spec["geometry_clearances"]
    min_robot_ped = float(clearance["robot_to_pedestrian_initial"])
    min_ped_ped = float(clearance["inter_pedestrian"])
    heading_yaw = HEADING_YAWS[candidate.robot_heading]
    duration = parse_duration(spec["collection"]["duration"])

    if candidate.robot_state == "moving":
        segment = _robot_segment(geometry, candidate, math.radians(float(spec.get("robot_route_heading_tolerance_deg", 20))))
        if segment is None:
            raise ValueError("no nonholonomic route segment matches requested heading/relation")
        robot_start, robot_end = segment
        segment_length = math.dist(robot_start, robot_end)
        assumed_speed = float(spec.get("robot_nominal_speed", 0.5))
        traversals = max(1, min(12, int(math.ceil(duration * assumed_speed / max(segment_length, 0.1)))))
        phases = []
        for index in range(traversals):
            target = robot_end if index % 2 == 0 else robot_start
            reverse_yaw = wrap_to_pi(heading_yaw + (math.pi if candidate.robot_motion_relation == "forward" else 0.0))
            phases.append({"goto": [round(target[0], 5), round(target[1], 5), round(reverse_yaw if index % 2 else heading_yaw, 6)]})
        robot_route_direction = bearing_class(wrap_to_pi(math.atan2(robot_end[1] - robot_start[1], robot_end[0] - robot_start[0]) - math.pi / 2))
    else:
        if candidate.trajectory_pattern in {
            "robot_approaches_corner",
            "same_direction_around_corner",
            "opposing_around_corner",
        }:
            raise ValueError("corner pattern requires a moving robot")
        anchor = geometry.robot_navigable.representative_point()
        robot_start = (float(anchor.x), float(anchor.y))
        phases = []
        robot_route_direction = "none"

    if not geometry.robot_navigable.buffer(1e-6).covers(Point(robot_start)):
        raise ValueError("robot anchor is outside Jackal-clear navigable space")

    routes = _choose_ped_routes(geometry, candidate)
    emerge_patterns = {"corner_emerge", "occluded_to_visible", "robot_approaches_corner"}
    disappear_patterns = {"corner_disappear", "visible_to_occluded"}
    if candidate.trajectory_pattern in emerge_patterns | disappear_patterns | {"hidden_cross_front"}:
        oriented = []
        for route in routes:
            first = _interpolate_route(route, 0.08)
            last = _interpolate_route(route, 0.92)
            first_visible = has_line_of_sight(geometry, robot_start, first)
            last_visible = has_line_of_sight(geometry, robot_start, last)
            wants_emerge = candidate.trajectory_pattern in emerge_patterns | {"hidden_cross_front"}
            if wants_emerge and first_visible and not last_visible:
                route = list(reversed(route))
            elif not wants_emerge and not first_visible and last_visible:
                route = list(reversed(route))
            oriented.append(route)
        routes = oriented
    dynamic: list[dict[str, Any]] = []
    pedestrian_meta: list[dict[str, Any]] = []
    spawn_points: list[tuple[float, float]] = []
    route_slots: list[tuple[list[tuple[float, float]], tuple[float, float]]] = []
    for slot_index, route in enumerate(routes):
        fraction = 0.08 + 0.16 * slot_index
        if fraction > 0.82:
            fraction = 0.08 + 0.08 * slot_index
        spawn = _interpolate_route(route, fraction)
        if math.dist(spawn, robot_start) < min_robot_ped:
            spawn = _interpolate_route(route, min(0.92, fraction + 0.35))
        if math.dist(spawn, robot_start) < min_robot_ped:
            raise ValueError("insufficient initial robot-to-pedestrian separation")
        if any(math.dist(spawn, other) < min_ped_ped for other in spawn_points):
            spawn = _interpolate_route(route, min(0.95, fraction + 0.12 * (slot_index + 1)))
        if any(math.dist(spawn, other) < min_ped_ped for other in spawn_points):
            raise ValueError("insufficient inter-pedestrian spawn separation")
        if not geometry.navigable.buffer(1e-6).covers(Point(spawn)):
            raise ValueError("pedestrian spawn is outside navigable space")
        spawn_points.append(spawn)
        route_slots.append((route, spawn))

    # Assign deterministic role indices from nearest to farthest initial range.
    route_slots.sort(key=lambda item: (math.dist(item[1], robot_start), item[1], tuple(item[0])))
    routes = [route for route, _spawn in route_slots]
    spawn_points = [spawn for _route, spawn in route_slots]

    for index, (human_type, model, speed, route, spawn) in enumerate(zip(candidate.human_types, candidate.models, candidate.speeds, routes, spawn_points, strict=False)):
        mode = _waypoint_mode(candidate.trajectory_pattern)
        route_yaw = math.atan2(route[1][1] - route[0][1], route[1][0] - route[0][0])
        route_points = [spawn, *route[1:]]
        if len(route_points) < 2 or _route_length(route_points) <= 0.1:
            route_points = [spawn, route[-1] if math.dist(spawn, route[-1]) > 0.1 else route[0]]
        waypoints = [[round(x, 5), round(y, 5), 0.0] for x, y in route_points]
        requested_type = human_type
        mapping = spec["human_types"][human_type]
        fallback_used = model == mapping.get("fallback_model") and model not in mapping.get("models", [])
        dynamic.append(
            {
                "name": f"ped_{index + 1:02d}",
                "model": model,
                "pose": [round(spawn[0], 5), round(spawn[1], 5), round(route_yaw, 6)],
                "agent": {"agent_type": mapping.get("agent_type", human_type), "desired_velocity": float(speed)},
                "waypoint_mode": mode,
                "waypoints": waypoints,
            }
        )
        pedestrian_meta.append(
            {
                "id": f"ped_{index + 1:02d}",
                "requested_human_type": requested_type,
                "agent_type": mapping.get("agent_type", human_type),
                "model": model,
                "fallback_model_used": fallback_used,
                "configured_speed": float(speed),
                "trajectory_pattern": candidate.trajectory_pattern,
                "waypoint_mode": mode,
                "route": waypoints,
                "initial_visible": has_line_of_sight(geometry, robot_start, spawn),
                "initial_range": math.dist(robot_start, spawn),
                "initial_relative_bearing": relative_bearing(robot_start, heading_yaw, spawn),
                "initial_bearing_class": bearing_class(relative_bearing(robot_start, heading_yaw, spawn)),
                "role_index": index,
            }
        )

    if candidate.trajectory_pattern in CORNER_PATTERNS:
        visibility_sequences = [[item["initial_visible"], *[has_line_of_sight(geometry, robot_start, point) for point in route[1:]]] for item, route in zip(pedestrian_meta, routes, strict=False)]
        flat = [value for sequence in visibility_sequences for value in sequence]
        if not flat or all(flat) or not any(flat):
            raise ValueError("corner route does not produce a verified visibility transition")
        if candidate.trajectory_pattern in emerge_patterns and not any(not sequence[0] and any(sequence[1:]) for sequence in visibility_sequences):
            raise ValueError("occluded-to-visible pattern does not begin occluded and become visible")
        if candidate.trajectory_pattern in disappear_patterns and not any(sequence[0] and not all(sequence[1:]) for sequence in visibility_sequences):
            raise ValueError("visible-to-occluded pattern does not begin visible and become occluded")
        if candidate.trajectory_pattern == "visible_plus_occluded":
            initial = [sequence[0] for sequence in visibility_sequences]
            if not any(initial) or all(initial):
                raise ValueError("visible_plus_occluded does not contain both initial visibility roles")
        if "cross_front" in candidate.trajectory_pattern or candidate.trajectory_pattern == "hidden_cross_front":
            sector = math.radians(float(spec.get("front_sector_half_angle_deg", 45)))
            if not any(abs(relative_bearing(robot_start, heading_yaw, point)) <= sector for route in routes for point in route):
                raise ValueError("corner route never enters the robot forward sector")

    scenario = {
        "robots": [
            {
                "start": [round(robot_start[0], 5), round(robot_start[1], 5), round(heading_yaw, 6)],
                "phases": phases,
            }
        ],
        "static": [],
        "dynamic": dynamic,
    }
    metadata = {
        "generated_by": GENERATED_BY,
        "generator_version": GENERATOR_VERSION,
        "stable_id": candidate.stable_id,
        "scenario_id": candidate.scenario_name,
        "world": candidate.world,
        "seed": candidate.seed,
        "naturalistic_profile": candidate.naturalistic,
        "world_geometry": geometry.attributes(),
        "robot": {
            "state": candidate.robot_state,
            "heading": candidate.robot_heading,
            "yaw": heading_yaw,
            "route_direction": robot_route_direction,
            "motion_relation": candidate.robot_motion_relation,
            "route": phases,
        },
        "pedestrian_count": candidate.count,
        "composition": candidate.composition,
        "speed_profile": candidate.speed_profile,
        "trajectory_pattern": candidate.trajectory_pattern,
        "pedestrians": pedestrian_meta,
        "collection": spec["collection"],
        "audio": spec["audio"],
        "ground_truth_topics": spec["ground_truth_topics"],
    }
    return scenario, metadata


def validate_rendered_scenario(scenario: dict[str, Any]) -> None:
    """Validate the emitted subset of Arena's Scenario schema without ROS.

    Generated mappings use only primitive fields consumed by ``Scenario``,
    ``RobotGoal`` and ``DynamicObstacle``. This mirrors that contract without
    importing asset or ament resolvers during source-file generation.
    """
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a mapping")
    allowed = {"robots", "static", "dynamic", "regions"}
    unsupported = set(scenario) - allowed
    if unsupported:
        raise ValueError(f"scenario contains unsupported fields: {sorted(unsupported)}")
    robots = scenario.get("robots")
    if not isinstance(robots, list) or not robots:
        raise ValueError("scenario must contain at least one robot")
    for index, robot in enumerate(robots):
        if not isinstance(robot, dict):
            raise ValueError(f"robot {index} must be a mapping")
        _validate_pose(robot.get("start"), f"robot {index} start")
        phases = robot.get("phases", [])
        if not isinstance(phases, list):
            raise ValueError(f"robot {index} phases must be a list")
        for phase_index, phase in enumerate(phases):
            if not isinstance(phase, dict) or set(phase) != {"goto"}:
                raise ValueError(f"robot {index} phase {phase_index} must contain only goto")
            _validate_pose(phase["goto"], f"robot {index} phase {phase_index} goto")

    if not isinstance(scenario.get("static"), list):
        raise ValueError("static must be a list")
    dynamic = scenario.get("dynamic")
    if not isinstance(dynamic, list):
        raise ValueError("dynamic must be a list")
    supported_modes = {"once", "reverse", "repeat"}
    for index, obstacle in enumerate(dynamic):
        if not isinstance(obstacle, dict):
            raise ValueError(f"dynamic obstacle {index} must be a mapping")
        for field_name in ("name", "model", "pose", "agent", "waypoint_mode", "waypoints"):
            if field_name not in obstacle:
                raise ValueError(f"dynamic obstacle {index} is missing {field_name}")
        if not isinstance(obstacle["name"], str) or not obstacle["name"]:
            raise ValueError(f"dynamic obstacle {index} has an invalid name")
        if not isinstance(obstacle["model"], str) or not obstacle["model"]:
            raise ValueError(f"dynamic obstacle {index} has an invalid model")
        _validate_pose(obstacle["pose"], f"dynamic obstacle {index} pose")
        agent = obstacle["agent"]
        if not isinstance(agent, dict) or not isinstance(agent.get("agent_type"), str) or not agent["agent_type"]:
            raise ValueError(f"dynamic obstacle {index} has an invalid agent type")
        if float(agent.get("desired_velocity", 0.0)) <= 0.0:
            raise ValueError(f"dynamic obstacle {index} has a non-positive desired velocity")
        if obstacle["waypoint_mode"] not in supported_modes:
            raise ValueError(f"dynamic obstacle {index} has an unsupported waypoint mode")
        waypoints = obstacle["waypoints"]
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ValueError("scenario contains a zero-length pedestrian path")
        for waypoint_index, waypoint in enumerate(waypoints):
            _validate_pose(waypoint, f"dynamic obstacle {index} waypoint {waypoint_index}")
        points = {(round(float(point[0]), 8), round(float(point[1]), 8)) for point in waypoints}
        if len(points) < 2:
            raise ValueError("scenario contains a zero-length pedestrian path")


def _validate_pose(value: object, label: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
        raise ValueError(f"{label} must be an [x, y] or [x, y, yaw] sequence")
    try:
        coordinates = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains a non-numeric coordinate") from exc
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError(f"{label} contains a non-finite coordinate")


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _yaml_bytes(value: object) -> bytes:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")


def _safe_write_scenario(world_dir: Path, name: str, scenario: dict[str, Any], metadata: dict[str, Any]) -> str:
    target = world_dir / "scenarios" / name
    scenario_bytes = _yaml_bytes(scenario)
    metadata_bytes = _yaml_bytes(metadata)
    if target.exists():
        scenario_path = target / "scenario.yaml"
        metadata_path = target / "metadata.yaml"
        if scenario_path.is_file() and metadata_path.is_file() and scenario_path.read_bytes() == scenario_bytes and metadata_path.read_bytes() == metadata_bytes:
            return "unchanged"
        raise FileExistsError(f"refusing to overwrite existing scenario directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    (target / "scenario.yaml").write_bytes(scenario_bytes)
    (target / "metadata.yaml").write_bytes(metadata_bytes)
    return "written"


def generated_cleanup_targets(worlds: Iterable[Path]) -> list[Path]:
    targets: list[Path] = []
    for world in worlds:
        scenarios = world / "scenarios"
        if not scenarios.is_dir():
            continue
        for candidate in sorted(scenarios.glob(f"{GENERATED_PREFIX}*")):
            marker = candidate / "metadata.yaml"
            if not candidate.is_dir() or candidate.is_symlink() or not marker.is_file():
                continue
            try:
                raw = yaml.safe_load(marker.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(raw, dict) and raw.get("generated_by") == GENERATED_BY:
                targets.append(candidate)
    return targets


def clean_generated(worlds: Iterable[Path], *, execute: bool) -> list[str]:
    targets = generated_cleanup_targets(worlds)
    if execute:
        for target in targets:
            shutil.rmtree(target)
    return [str(target) for target in targets]


def validate_models(spec: dict[str, Any], *, resolve: bool) -> list[dict[str, str]]:
    """Validate model identifiers from the source tree, without ROS resolvers."""
    identifier_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    configured: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    for human_type, config in spec["human_types"].items():
        for model in [*config.get("models", []), *([config["fallback_model"]] if config.get("fallback_model") else [])]:
            name = str(model)
            configured[name] = str(human_type)
            if not identifier_pattern.fullmatch(name):
                errors.append({"human_type": str(human_type), "model": name, "error": "invalid identifier syntax"})
    if resolve:
        catalog = _repository_human_model_catalog()
        for name, human_type in configured.items():
            if identifier_pattern.fullmatch(name) and name not in catalog:
                errors.append(
                    {
                        "human_type": human_type,
                        "model": name,
                        "error": "model is not referenced by checked-in human scenarios",
                    }
                )
    return errors


def _repository_human_model_catalog() -> set[str]:
    """Collect model names used by checked-in, hand-authored scenarios."""
    repository_root = Path(__file__).resolve().parents[4]
    roots = [
        repository_root / "arena_simulation_setup" / "worlds",
        repository_root / "arena_simulation_setup" / "acoustics" / "worlds",
    ]
    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            model = value.get("model")
            if isinstance(model, str):
                names.add(model)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("scenario.yaml"):
            if any(part.startswith(GENERATED_PREFIX) for part in path.parts):
                continue
            try:
                visit(yaml.safe_load(path.read_text(encoding="utf-8")))
            except (OSError, yaml.YAMLError):
                continue
    return names


def validate_agent_types(spec: dict[str, Any]) -> list[str]:
    configured = {str(config.get("agent_type", key)) for key, config in spec["human_types"].items()}
    try:
        from arena_humansim.core.agents import BUILTIN_AGENTS

        supported = set(BUILTIN_AGENTS)
    except ImportError:
        repository_root = Path(__file__).resolve().parents[4]
        catalog = repository_root / "humansim" / "arena_humansim" / "config" / "agent_types"
        supported = {path.stem for path in catalog.glob("*.yaml")}
    return sorted(configured - supported)


def _representative_worlds(geometries: Sequence[WorldGeometry], limit: int) -> list[WorldGeometry]:
    selected: list[WorldGeometry] = []
    # First cover topology families, then width families, before filling finer
    # topology/width/angle strata. This prevents lexical ordering from making a
    # smoke suite entirely out of one topology.
    for topology in sorted({item.topology for item in geometries}):
        candidates = [item for item in geometries if item.topology == topology]
        selected.append(sorted(candidates, key=lambda item: (item.width_class, item.world))[0])
        if len(selected) == limit:
            return selected
    for width in sorted({item.width_class for item in geometries}):
        candidates = [item for item in geometries if item.width_class == width and item not in selected]
        if candidates:
            selected.append(sorted(candidates, key=lambda item: (item.topology, item.world))[0])
            if len(selected) == limit:
                return selected
    groups: dict[tuple[str, str, bool], list[WorldGeometry]] = {}
    for geometry in geometries:
        groups.setdefault((geometry.topology, geometry.width_class, geometry.angled), []).append(geometry)
    for key in sorted(groups):
        candidate = sorted(groups[key], key=lambda item: item.world)[0]
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for geometry in sorted(geometries, key=lambda item: item.world):
            if geometry not in selected:
                selected.append(geometry)
                if len(selected) == limit:
                    break
    return selected


def build_split_manifests(
    rows: Sequence[dict[str, Any]],
    protocols: Sequence[str] = (
        "seen_world_unseen_scenario",
        "held_out_world",
        "leave_one_topology_out",
    ),
) -> dict[str, Any]:
    def partition(key: str, buckets: int = 5) -> str:
        value = _stable_int(key) % buckets
        return "test" if value == 0 else ("validation" if value == 1 else "train")

    seen = {row["stable_id"]: partition(row["family_id"]) for row in rows}
    held_world = {row["stable_id"]: partition(row["world"]) for row in rows}
    topologies = sorted({row["world_geometry"]["topology"] for row in rows})
    loto = {topology: {row["stable_id"]: ("test" if row["world_geometry"]["topology"] == topology else "train") for row in rows} for topology in topologies}
    compositions = sorted({row["composition"] for row in rows})
    held_composition = compositions[_stable_int("held-composition") % len(compositions)] if compositions else None
    comp = {row["stable_id"]: ("test" if row["composition"] == held_composition else "train") for row in rows}
    type_speeds = sorted({token for row in rows for token in row["type_speed_combinations"]})
    held_type_speed = type_speeds[_stable_int("held-type-speed") % len(type_speeds)] if type_speeds else None
    bias = {row["stable_id"]: ("test" if held_type_speed in row["type_speed_combinations"] else "train") for row in rows}
    available = {
        "leakage_group": "scenario_family (world + factors excluding seed)",
        "seen_world_unseen_scenario": seen,
        "held_out_world": held_world,
        "leave_one_topology_out": loto,
        "held_out_human_composition": {"held_out": held_composition, "assignments": comp},
        "held_out_type_speed": {"held_out": held_type_speed, "assignments": bias},
    }
    requested = set(protocols)
    unknown = requested - (set(available) - {"leakage_group"})
    if unknown:
        raise ValueError(f"unknown dataset split protocol(s): {sorted(unknown)}")
    return {
        "leakage_group": available["leakage_group"],
        **{name: available[name] for name in protocols},
    }


def validate_audio_payload(
    samples: Sequence[float],
    *,
    sample_rate: int | None,
    channels: int | None,
    silence_rms_threshold: float = 1e-5,
    clipping_fraction_threshold: float = 0.01,
) -> list[str]:
    errors: list[str] = []
    if not sample_rate or sample_rate <= 0:
        errors.append("missing or invalid sample-rate metadata")
    if not channels or channels <= 0:
        errors.append("missing or invalid channel-count metadata")
    if not samples:
        errors.append("audio is empty")
        return errors
    rms = math.sqrt(sum(float(value) ** 2 for value in samples) / len(samples))
    if rms <= silence_rms_threshold:
        errors.append(f"audio is silent: rms={rms:g} <= {silence_rms_threshold:g}")
    clipped = sum(abs(float(value)) >= 0.999 for value in samples) / len(samples)
    if clipped > clipping_fraction_threshold:
        errors.append(f"audio is clipped: fraction={clipped:g} > {clipping_fraction_threshold:g}")
    return errors


SUPPORTED_AUDIO_MESSAGE_TYPES = {
    "audio_common_msgs/msg/AudioData",
    "audio_msgs/msg/Audio",
}


def validate_audio_preflight(
    audio_config: dict[str, Any],
    available_topics: dict[str, Sequence[str]],
    observed_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate configured ROS audio topics before an episode starts."""
    errors: list[str] = []
    for topic in audio_config.get("topics", []):
        name = str(topic["name"])
        advertised = set(available_topics.get(name, ()))
        if not advertised:
            errors.append(f"audio topic missing: {name}")
            continue
        configured = set(topic.get("message_types", ()))
        supported = advertised & configured & SUPPORTED_AUDIO_MESSAGE_TYPES
        if not supported:
            errors.append(f"audio topic {name} has unsupported type(s): {sorted(advertised)}")
        metadata = observed_metadata.get(name, {})
        sample_rate = metadata.get("sample_rate", audio_config.get("sample_rate"))
        channels = metadata.get("channels", audio_config.get("channels"))
        if not sample_rate:
            errors.append(f"audio topic {name} is missing sample-rate metadata")
        if not channels:
            errors.append(f"audio topic {name} is missing channel-count metadata")
    return errors


def frame_label(
    *,
    timestamp: float,
    episode_id: str,
    world: str,
    scenario_id: str,
    world_geometry: dict[str, Any],
    robot_pose: Sequence[float],
    robot_velocity: Sequence[float],
    robot_heading: str,
    robot_state: str,
    pedestrian: dict[str, Any],
    audio: dict[str, Any],
    line_of_sight: bool,
    occlusion_transition: str,
) -> dict[str, Any]:
    """Create one synchronized, flat ground-truth label row."""
    robot_xy = robot_pose[:2]
    ped_xy = pedestrian["pose"][:2]
    bearing = relative_bearing(robot_xy, float(robot_pose[2]), ped_xy)
    return {
        "timestamp": float(timestamp),
        "episode_id": episode_id,
        "world": world,
        "scenario_id": scenario_id,
        "world_topology": world_geometry.get("topology"),
        "world_width_class": world_geometry.get("width_class"),
        "world_angled": world_geometry.get("angled"),
        "robot_x": float(robot_xy[0]),
        "robot_y": float(robot_xy[1]),
        "robot_yaw": float(robot_pose[2]),
        "robot_vx": float(robot_velocity[0]),
        "robot_vy": float(robot_velocity[1]),
        "robot_heading": robot_heading,
        "robot_state": robot_state,
        "pedestrian_id": pedestrian["id"],
        "pedestrian_human_type": pedestrian["human_type"],
        "pedestrian_model": pedestrian["model"],
        "pedestrian_configured_speed": float(pedestrian["configured_speed"]),
        "pedestrian_actual_speed": float(pedestrian["actual_speed"]),
        "pedestrian_x": float(ped_xy[0]),
        "pedestrian_y": float(ped_xy[1]),
        "pedestrian_vx": float(pedestrian["velocity"][0]),
        "pedestrian_vy": float(pedestrian["velocity"][1]),
        "trajectory_pattern": pedestrian["trajectory_pattern"],
        "range_to_robot": math.dist(robot_xy, ped_xy),
        "relative_bearing": bearing,
        "bearing_class": bearing_class(bearing),
        "line_of_sight": bool(line_of_sight),
        "occlusion_transition": occlusion_transition,
        "audio_sample_rate": int(audio["sample_rate"]),
        "audio_channel_count": int(audio["channels"]),
        "microphone_frame": audio["microphone_frame"],
    }


def build_episode_manifest(
    generation_metadata: dict[str, Any],
    *,
    git_revision: str,
    actual_duration: float,
    audio_schema: dict[str, Any],
    validation_result: dict[str, Any],
) -> dict[str, Any]:
    """Build the required per-episode ``episode.yaml`` mapping."""
    expected_duration = parse_duration(generation_metadata["collection"]["duration"])
    return {
        "git_revision": git_revision,
        "generator_version": generation_metadata["generator_version"],
        "seed": generation_metadata["seed"],
        "world": generation_metadata["world"],
        "scenario_id": generation_metadata["scenario_id"],
        "duration": generation_metadata["collection"]["duration"],
        "expected_duration": expected_duration,
        "actual_duration": float(actual_duration),
        "robot": generation_metadata["robot"],
        "pedestrians": generation_metadata["pedestrians"],
        "audio": audio_schema,
        "validation_result": validation_result,
    }


def validate_episode_timing(
    timestamps: Sequence[float],
    *,
    expected_duration: float,
    actual_duration: float,
    tolerance: float,
) -> list[str]:
    errors: list[str] = []
    if any(right < left for left, right in zip(timestamps, timestamps[1:], strict=False)):
        errors.append("timestamps are not monotonic")
    if abs(float(actual_duration) - float(expected_duration)) > float(tolerance):
        errors.append(f"duration outside tolerance: expected={expected_duration:g}, actual={actual_duration:g}, tolerance={tolerance:g}")
    return errors


def write_audio_wav(
    path: str | Path,
    samples: Sequence[float],
    *,
    sample_rate: int,
    channels: int,
) -> None:
    """Write normalized interleaved float samples as signed 16-bit PCM WAV."""
    if len(samples) % channels:
        raise ValueError("interleaved audio sample count is not divisible by channels")
    pcm = bytearray()
    for value in samples:
        clipped = max(-1.0, min(1.0, float(value)))
        pcm.extend(struct.pack("<h", int(round(clipped * 32767.0))))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(int(channels))
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(bytes(pcm))


def write_frame_labels_parquet(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("frame labels are empty")
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("writing frame_labels.parquet requires polars") from exc
    pl.DataFrame(rows).write_parquet(Path(path))


def finalize_episode(
    dataset_root: str | Path,
    *,
    episode_id: str,
    generation_metadata: dict[str, Any],
    scenario: dict[str, Any],
    recording_mcap: str | Path,
    audio_samples: Sequence[float],
    audio_schema: dict[str, Any],
    labels: Sequence[dict[str, Any]],
    actual_duration: float,
    git_revision: str,
) -> Path:
    """Validate and atomically materialize one required dataset episode tree."""
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / episode_id
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing episode: {target}")
    source_mcap = Path(recording_mcap)
    if not source_mcap.is_file() or source_mcap.stat().st_size == 0:
        raise ValueError(f"recording MCAP is missing or empty: {source_mcap}")

    sample_rate = audio_schema.get("sample_rate")
    channels = audio_schema.get("channels")
    audio_errors = validate_audio_payload(
        audio_samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_rms_threshold=float(audio_schema.get("silence_rms_threshold", 1e-5)),
        clipping_fraction_threshold=float(audio_schema.get("clipping_fraction_threshold", 0.01)),
    )
    expected_duration = parse_duration(generation_metadata["collection"]["duration"])
    tolerance = parse_duration(generation_metadata["collection"].get("duration_tolerance", 0.25))
    timing_errors = validate_episode_timing(
        [float(row["timestamp"]) for row in labels],
        expected_duration=expected_duration,
        actual_duration=actual_duration,
        tolerance=tolerance,
    )
    validation_result = {
        "valid": not audio_errors and not timing_errors,
        "audio_errors": audio_errors,
        "timing_errors": timing_errors,
    }
    if not validation_result["valid"]:
        raise ValueError(f"episode validation failed: {validation_result}")

    temp = Path(tempfile.mkdtemp(prefix=f".{episode_id}.", dir=root))
    try:
        episode = build_episode_manifest(
            generation_metadata,
            git_revision=git_revision,
            actual_duration=actual_duration,
            audio_schema=audio_schema,
            validation_result=validation_result,
        )
        (temp / "episode.yaml").write_bytes(_yaml_bytes(episode))
        (temp / "scenario.yaml").write_bytes(_yaml_bytes(scenario))
        shutil.copy2(source_mcap, temp / "recording.mcap")
        write_audio_wav(
            temp / "audio.wav",
            audio_samples,
            sample_rate=int(sample_rate),
            channels=int(channels),
        )
        write_frame_labels_parquet(temp / "frame_labels.parquet", labels)
        (temp / "validation.json").write_text(json.dumps(validation_result, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def _manifest_row(candidate: Candidate, metadata: dict[str, Any]) -> dict[str, Any]:
    closest = min(metadata["pedestrians"], key=lambda item: item["initial_range"])
    return {
        "stable_id": candidate.stable_id,
        "family_id": candidate.family_id,
        "world": candidate.world,
        "scenario": candidate.scenario_name,
        "seed": candidate.seed,
        "pedestrian_count": candidate.count,
        "human_types": list(candidate.human_types),
        "models": list(candidate.models),
        "speeds": list(candidate.speeds),
        "type_speed_combinations": sorted({f"{kind}:{speed:g}" for kind, speed in zip(candidate.human_types, candidate.speeds, strict=False)}),
        "composition": candidate.composition,
        "speed_profile": candidate.speed_profile,
        "trajectory_pattern": candidate.trajectory_pattern,
        "robot_state": candidate.robot_state,
        "robot_heading": candidate.robot_heading,
        "robot_route_direction": metadata["robot"]["route_direction"],
        "robot_motion_relation": candidate.robot_motion_relation,
        "naturalistic_profile": candidate.naturalistic,
        "world_geometry": metadata["world_geometry"],
        "initial_closest_type": closest["requested_human_type"],
        "initial_visible_types": [item["requested_human_type"] for item in metadata["pedestrians"] if item["initial_visible"]],
        "pedestrians": metadata["pedestrians"],
    }


def mixed_role_balance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], Counter[str]] = {}
    for row in rows:
        if len(set(row["human_types"])) <= 1:
            continue
        grouped.setdefault((row["world"], row["composition"]), Counter()).update([row["initial_closest_type"]])
    groups = []
    unbalanced = []
    for (world, composition), counts in sorted(grouped.items()):
        item = {"world": world, "composition": composition, "initial_closest": dict(sorted(counts.items()))}
        groups.append(item)
        if len(counts) > 1 and sum(counts.values()) >= 2 and max(counts.values()) - min(counts.values()) > 1:
            unbalanced.append(item)
    return {"groups": groups, "unbalanced": unbalanced, "valid": not unbalanced}


def _suite(rows: Sequence[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    duration = spec["collection"]["duration"]
    watchdog = spec["collection"]["watchdog_timeout"]
    stages = []
    for row in rows:
        stages.append(
            {
                "name": _stable_digest(row["stable_id"], length=16),
                "map": row["world"],
                "robot": spec.get("robot", "jackal"),
                "tm_robots": "scenario",
                "tm_obstacles": "scenario",
                "episodes": 1,
                "duration": duration,
                "watchdog_timeout": watchdog,
                "config": {"scenario": {"file": row["scenario"]}},
            }
        )
    return {"stages": stages}


@dataclass
class GenerationResult:
    discovery: DiscoveryReport
    validation_report: dict[str, Any]
    manifest: list[dict[str, Any]]
    skipped: list[dict[str, str]]
    cleanup_targets: list[str]
    written: int
    unchanged: int


def load_spec(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dataset spec must be a YAML mapping")
    if len(value.get("speed_values", [])) != 2:
        raise ValueError("speed_values must contain exactly two values")
    if set(value.get("human_types", {})) != {"adult"}:
        raise ValueError("human_types must configure the standard adult population")
    collection = value.get("collection", {})
    duration = parse_duration(collection.get("duration", 0))
    watchdog = parse_duration(collection.get("watchdog_timeout", 0))
    if duration <= 0 or watchdog <= duration:
        raise ValueError("collection duration must be positive and watchdog_timeout must be larger")
    return value


def generate(
    spec_path: str | Path,
    *,
    profile: str,
    write: bool = False,
    validate_only: bool = False,
    strict_world_count: bool | None = None,
    expected_world_count: int | None = None,
    seed: int | None = None,
    world: Sequence[str] = (),
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    full_factorial: bool = False,
    clean: bool = False,
) -> GenerationResult:
    spec_file = Path(spec_path).resolve()
    spec = load_spec(spec_file)
    acoustics_root = spec_file.parent
    worlds_root = (acoustics_root / spec.get("worlds_root", "worlds")).resolve()
    include_patterns = list(include) if include else list(spec.get("worlds", {}).get("include", ["*"]))
    if world:
        include_patterns = list(world)
    exclude_patterns = [*spec.get("worlds", {}).get("exclude", []), *exclude]
    strict = bool(spec.get("worlds", {}).get("strict_count", True)) if strict_world_count is None else strict_world_count
    expected = int(spec.get("expected_world_count", 99) if expected_world_count is None else expected_world_count)
    selected_worlds, discovery = discover_worlds(
        worlds_root,
        expected_count=expected,
        include=include_patterns,
        exclude=exclude_patterns,
        strict=False,
    )
    if strict and not discovery.count_matches:
        raise WorldCountError(discovery)

    cleanup_targets = clean_generated(selected_worlds, execute=write and not validate_only) if clean else []
    invalid_agent_types = validate_agent_types(spec)
    model_errors = validate_models(spec, resolve=bool(spec.get("validation", {}).get("resolve_models", True)))
    validation: dict[str, Any] = {
        "valid": not invalid_agent_types and not model_errors,
        "world_discovery": discovery.asdict(),
        "invalid_agent_types": invalid_agent_types,
        "model_errors": model_errors,
        "geometry": {},
        "scenario_errors": [],
        "warnings": [],
    }
    if not discovery.count_matches:
        validation["warnings"].append(f"expected {expected} worlds but discovered {discovery.discovered_count}; continuing because strict mode is disabled")
    if invalid_agent_types or model_errors:
        raise ValueError(f"human catalog validation failed: agent_types={invalid_agent_types}, models={model_errors}")

    geometries: list[tuple[Path, WorldGeometry]] = []
    skipped: list[dict[str, str]] = []
    for world_path in selected_worlds:
        try:
            geometry = load_world_geometry(world_path, spec["geometry_clearances"])
            geometries.append((world_path, geometry))
            validation["geometry"][world_path.name] = {"valid": True, **geometry.attributes(), "warnings": geometry.warnings}
        except Exception as exc:  # noqa: BLE001 - aggregate all malformed-world diagnostics
            validation["geometry"][world_path.name] = {"valid": False, "error": str(exc)}
            skipped.append({"world": world_path.name, "reason": f"geometry: {exc}"})
    validation["valid"] = validation["valid"] and all(item["valid"] for item in validation["geometry"].values())

    if profile not in spec["profiles"]:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(spec['profiles'])}")
    profile_config = spec["profiles"][profile]
    if profile == "smoke":
        representatives = int(profile_config.get("worlds", 5))
        picked = {item.world for item in _representative_worlds([geometry for _, geometry in geometries], representatives)}
        geometries = [(path, geometry) for path, geometry in geometries if geometry.world in picked]

    generation_seed = int(spec.get("selection_seed", 42) if seed is None else seed)
    episode_limit = int(profile_config.get("episodes_per_world", spec.get("maximum_episodes_per_world", 96)))
    if profile == "smoke" and "episodes_total" in profile_config:
        episode_limit = max(1, math.ceil(int(profile_config["episodes_total"]) / max(len(geometries), 1)))

    pairwise_template_keys: list[tuple[Any, ...]] = []
    if not full_factorial:
        template_candidates = build_candidates("__selection_template__", spec)
        pairwise_template = stratified_pairwise_select(
            template_candidates,
            min(len(template_candidates), episode_limit * 4),
            generation_seed,
        )
        pairwise_template_keys = [_candidate_template_key(item) for item in pairwise_template]

    prepared: list[tuple[Path, Candidate, dict[str, Any], dict[str, Any]]] = []
    for world_path, geometry in geometries:
        candidates = build_candidates(geometry.world, spec)
        stable_order = sorted(candidates, key=lambda item: _stable_int(generation_seed, item.stable_id))
        if full_factorial:
            ordered = stable_order
        else:
            candidates_by_template = {_candidate_template_key(item): item for item in candidates}
            pairwise = [candidates_by_template[key] for key in pairwise_template_keys]
            pairwise_ids = {item.stable_id for item in pairwise}
            preferred = [*pairwise, *(item for item in stable_order if item.stable_id not in pairwise_ids)]
            ordered = counterbalanced_order(preferred, candidates)
        accepted = 0
        for candidate in ordered:
            if not full_factorial and accepted >= episode_limit:
                break
            try:
                scenario, metadata = render_candidate(geometry, candidate, spec)
                validate_rendered_scenario(scenario)
                prepared.append((world_path, candidate, scenario, metadata))
                accepted += 1
            except Exception as exc:  # noqa: BLE001 - every impossible factor must be reported
                skipped.append({"world": geometry.world, "stable_id": candidate.stable_id, "reason": str(exc)})
        if accepted == 0:
            validation["scenario_errors"].append({"world": geometry.world, "error": "no feasible candidates"})

    manifest = [_manifest_row(candidate, metadata) for _, candidate, _, metadata in prepared]
    manifest.sort(key=lambda row: (row["world"], row["scenario"], row["stable_id"]))
    prepared_by_id = {candidate.stable_id: (world_path, candidate, scenario, metadata) for world_path, candidate, scenario, metadata in prepared}
    prepared = [prepared_by_id[row["stable_id"]] for row in manifest]
    validation["valid"] = validation["valid"] and not validation["scenario_errors"]
    validation["mixed_role_balance"] = mixed_role_balance(manifest)
    validation["valid"] = validation["valid"] and validation["mixed_role_balance"]["valid"]
    validation["summary"] = {
        "worlds_selected": len(geometries),
        "scenarios_valid": len(manifest),
        "combinations_skipped": len(skipped),
    }

    written = 0
    unchanged = 0
    if write and not validate_only:
        for world_path, candidate, scenario, metadata in prepared:
            status = _safe_write_scenario(world_path, candidate.scenario_name, scenario, metadata)
            written += status == "written"
            unchanged += status == "unchanged"

        output_root = (acoustics_root / spec.get("output_root", "generated")).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "generation_manifest.yaml").write_bytes(_yaml_bytes({"episodes": manifest}))
        (output_root / "validation_report.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_root / "skipped_combinations.json").write_text(json.dumps(skipped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        suite = _suite(manifest, spec)
        (output_root / f"suite_{profile}.yaml").write_bytes(_yaml_bytes(suite))
        split_root = output_root / "splits"
        split_root.mkdir(exist_ok=True)
        split_protocols = spec.get("dataset_splits", {}).get("protocols", ())
        splits = build_split_manifests(manifest, split_protocols)
        desired_split_files = {f"{name}.yaml" for name in splits if name != "leakage_group"}
        for existing_split in split_root.glob("*.yaml"):
            if existing_split.name not in desired_split_files:
                existing_split.unlink()
        for name, split in splits.items():
            if name == "leakage_group":
                continue
            (split_root / f"{name}.yaml").write_bytes(_yaml_bytes({"leakage_group": splits["leakage_group"], "split": split}))

        repo_root = spec_file.parents[2]
        suite_target = repo_root / "arena_evaluation" / "arena_evaluation" / "configs" / "benchmark" / "suites" / f"acoustics_{profile}.yaml"
        suite_target.parent.mkdir(parents=True, exist_ok=True)
        suite_target.write_bytes(_yaml_bytes(suite))
        episode_template = {
            "git_revision": _git_revision(repo_root),
            "generator_version": GENERATOR_VERSION,
            "dataset_layout": "dataset/<episode_id>/{episode.yaml,scenario.yaml,recording.mcap,audio.wav,frame_labels.parquet,validation.json}",
            "required_episode_fields": [
                "seed",
                "world",
                "scenario_id",
                "expected_duration",
                "actual_duration",
                "robot",
                "pedestrians",
                "audio",
                "validation_result",
            ],
        }
        (output_root / "episode_manifest_template.yaml").write_bytes(_yaml_bytes(episode_template))

    return GenerationResult(
        discovery=discovery,
        validation_report=validation,
        manifest=manifest,
        skipped=skipped,
        cleanup_targets=cleanup_targets,
        written=written,
        unchanged=unchanged,
    )
