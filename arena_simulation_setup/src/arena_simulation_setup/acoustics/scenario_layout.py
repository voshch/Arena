"""Normalize generated acoustics scenarios to a fixed robot/listener layout.

The original generated matrix reversed both the robot and pedestrian routes for
``b-to-a`` cases.  That changes the listener pose between otherwise comparable
recordings.  It also emitted each pedestrian's spawn as its first waypoint and
left the waypoint mode at its repeating default.  In HumanSim that creates an
initial zero-length leg and makes the intended direction ambiguous.

The route/layout migration applies to scenarios whose directory name starts
with ``hearing__``. The robot always occupies end A. The direction factor
applies only to pedestrians: A-to-B starts near the robot and B-to-A starts at
the far end. Multi-person cases use a compact, collision-free
lateral/longitudinal formation so sources are not co-located and arrive from
different microphone sides. Every acoustics scenario, including older numeric
ones, is also migrated to Arena's bundled pedestrian model so preloading and
simulation do not depend on an optional network asset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image, ImageFilter

GENERATED_PREFIX = "hearing__"
NAME_PATTERN = re.compile(
    r"^hearing__world-(?P<world>.+)__robot-(?P<robot>idle|moving)"
    r"__pedestrians-(?P<count>[123])__ends-(?P<direction>a-to-b|b-to-a)$"
)

# This model is shipped with arena_simulation_setup and is also HumanSim's
# fallback. Dataset generation must not depend on an optional network asset.
PEDESTRIAN_MODEL = "arenian"
PEDESTRIAN_AGENT_TYPE = "adult"
PEDESTRIAN_SPEED_MPS = 1.20
PEDESTRIAN_CLEARANCE_M = 0.35
NEAR_ROBOT_OFFSET_M = 1.50
ROBOT_STOP_CLEARANCE_M = 1.00
FORMATION_LONGITUDINAL_M = 0.75
FORMATION_LATERAL_M = 0.45
FIRST_TARGET_LEAD_M = 0.75

Point2 = tuple[float, float]
Cell = tuple[int, int]


@dataclass(frozen=True)
class MapFrame:
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    width: int
    height: int

    def world_to_cell(self, point: Sequence[float]) -> Cell:
        dx = float(point[0]) - self.origin_x
        dy = float(point[1]) - self.origin_y
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        column = int(round(local_x / self.resolution - 0.5))
        bottom_row = int(round(local_y / self.resolution - 0.5))
        return self.height - bottom_row - 1, column


@dataclass(frozen=True)
class LayoutResult:
    checked: int
    changed: int
    written: int


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return value


def _xy(value: Sequence[float]) -> Point2:
    if len(value) < 2:
        raise ValueError(f"pose/waypoint has fewer than two coordinates: {value!r}")
    return float(value[0]), float(value[1])


def _deduplicate(points: Iterable[Point2]) -> list[Point2]:
    result: list[Point2] = []
    for point in points:
        if not result or math.dist(result[-1], point) > 1e-6:
            result.append(point)
    return result


def _length(points: Sequence[Point2]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))


def _point_at(points: Sequence[Point2], distance: float) -> Point2:
    remaining = max(0.0, min(float(distance), _length(points)))
    for start, end in zip(points, points[1:], strict=False):
        segment = math.dist(start, end)
        if remaining <= segment:
            if segment <= 1e-12:
                return start
            fraction = remaining / segment
            return (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
        remaining -= segment
    return points[-1]


def _slice(points: Sequence[Point2], start_distance: float, end_distance: float) -> list[Point2]:
    total = _length(points)
    start_distance = max(0.0, min(start_distance, total))
    end_distance = max(start_distance, min(end_distance, total))
    result = [_point_at(points, start_distance)]
    traversed = 0.0
    for left, right in zip(points, points[1:], strict=False):
        traversed += math.dist(left, right)
        if start_distance + 1e-6 < traversed < end_distance - 1e-6:
            result.append(right)
    result.append(_point_at(points, end_distance))
    return _deduplicate(result)


def _yaw(start: Sequence[float], end: Sequence[float]) -> float:
    return math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))


def _pose(point: Sequence[float], yaw: float) -> list[float]:
    return [round(float(point[0]), 4), round(float(point[1]), 4), round(float(yaw), 6)]


def _load_safe_cells(world_dir: Path) -> tuple[MapFrame, set[Cell]]:
    level_dir = world_dir / "0"
    map_path = level_dir / "map.yaml"
    raw = _load_yaml(map_path)
    resolution = float(raw["resolution"])
    origin = raw["origin"]
    image_path = level_dir / str(raw["image"])
    with Image.open(image_path) as source:
        grayscale = source.convert("L")
    width, height = grayscale.size
    negate = bool(int(raw.get("negate", 0)))
    free_threshold = float(raw.get("free_thresh", 0.196))
    occupancy = grayscale.point(lambda value: value if negate else 255 - value, mode="L")
    limit = int(math.floor(free_threshold * 255.0))
    free = occupancy.point(lambda value: 255 if value <= limit else 0, mode="L")
    clearance_pixels = max(1, int(math.ceil(PEDESTRIAN_CLEARANCE_M / resolution)))
    safe = free.filter(ImageFilter.MinFilter(2 * clearance_pixels + 1))
    cells = {divmod(index, width) for index, value in enumerate(safe.tobytes()) if value == 255}
    frame = MapFrame(
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        width=width,
        height=height,
    )
    return frame, cells


def _is_safe(point: Point2, frame: MapFrame, safe_cells: set[Cell]) -> bool:
    row, column = frame.world_to_cell(point)
    return 0 <= row < frame.height and 0 <= column < frame.width and (row, column) in safe_cells


def _offset_at(
    route: Sequence[Point2],
    distance: float,
    lateral: float,
    frame: MapFrame,
    safe_cells: set[Cell],
) -> Point2:
    point = _point_at(route, distance)
    if abs(lateral) <= 1e-9:
        return point
    before = _point_at(route, max(0.0, distance - 0.10))
    after = _point_at(route, min(_length(route), distance + 0.10))
    heading = _yaw(before, after)
    for scale in (1.0, 0.75, 0.5, 0.25):
        candidate = (
            point[0] - math.sin(heading) * lateral * scale,
            point[1] + math.cos(heading) * lateral * scale,
        )
        if _is_safe(candidate, frame, safe_cells):
            return candidate
    return point


def _canonical_route(matrix: dict[tuple[str, int, str], dict[str, Any]]) -> list[Point2]:
    # A moving A-to-B case preserves the complete route in robot phases.
    for count in (1, 2, 3):
        scenario = matrix.get(("moving", count, "a-to-b"))
        if scenario:
            robot = scenario["robots"][0]
            route = [_xy(robot["start"]), *(_xy(phase["goto"]) for phase in robot.get("phases", []))]
            route = _deduplicate(route)
            if len(route) >= 2:
                return route

    # Fallback for matrices containing idle cases only: legacy A-to-B humans
    # walked B-to-A, so reverse one of their waypoint paths.
    for (robot_state, _count, direction), scenario in matrix.items():
        if robot_state == "idle" and direction == "a-to-b" and scenario.get("dynamic"):
            human = scenario["dynamic"][0]
            reverse_route = [_xy(human["pose"]), *(_xy(point) for point in human.get("waypoints", []))]
            route = list(reversed(_deduplicate(reverse_route)))
            if len(route) >= 2:
                return route
    raise ValueError("cannot recover an A-to-B route from the generated scenario matrix")


def _formation_offsets(count: int) -> list[tuple[float, float]]:
    centre = (count - 1) / 2.0
    return [(index * FORMATION_LONGITUDINAL_M, (index - centre) * FORMATION_LATERAL_M) for index in range(count)]


def _render(
    route_a_to_b: Sequence[Point2],
    robot_state: str,
    count: int,
    direction: str,
    frame: MapFrame,
    safe_cells: set[Cell],
) -> dict[str, Any]:
    total = _length(route_a_to_b)
    minimum = NEAR_ROBOT_OFFSET_M + FIRST_TARGET_LEAD_M + ROBOT_STOP_CLEARANCE_M
    if total <= minimum + FORMATION_LONGITUDINAL_M:
        raise ValueError(f"route length {total:.2f} m is too short for the pedestrian formation")

    robot_yaw = _yaw(route_a_to_b[0], route_a_to_b[1])
    phases: list[dict[str, list[float]]] = []
    if robot_state == "moving":
        for previous, target in zip(route_a_to_b, route_a_to_b[1:], strict=False):
            phases.append({"goto": _pose(target, _yaw(previous, target))})

    pedestrian_route = list(route_a_to_b if direction == "a-to-b" else reversed(route_a_to_b))
    route_total = _length(pedestrian_route)
    base_start = NEAR_ROBOT_OFFSET_M if direction == "a-to-b" else 0.0
    end_distance = route_total if direction == "a-to-b" else route_total - ROBOT_STOP_CLEARANCE_M
    dynamic: list[dict[str, Any]] = []
    spawn_points: list[Point2] = []
    terminal_points: list[Point2] = []
    for index, (longitudinal, lateral) in enumerate(_formation_offsets(count), start=1):
        start_distance = max(0.0, base_start + longitudinal)
        individual_end_distance = end_distance - longitudinal
        spawn = _offset_at(pedestrian_route, start_distance, lateral, frame, safe_cells)
        # Around tight bends, equal arc-length spacing can still overlap in
        # Euclidean space. Advance only as far as needed to clear the previous
        # footprint while preserving the compact formation.
        while spawn_points and math.dist(spawn_points[-1], spawn) < 2 * PEDESTRIAN_CLEARANCE_M:
            start_distance += 0.10
            if start_distance + FIRST_TARGET_LEAD_M >= individual_end_distance:
                raise ValueError(f"pedestrian {index} cannot be placed without overlap")
            spawn = _offset_at(pedestrian_route, start_distance, lateral, frame, safe_cells)
        target_distance = min(individual_end_distance, start_distance + FIRST_TARGET_LEAD_M)
        if target_distance <= start_distance + 1e-6:
            raise ValueError(f"pedestrian {index} has no traversable route")
        first_target = _offset_at(pedestrian_route, target_distance, lateral, frame, safe_cells)
        terminal = _offset_at(pedestrian_route, individual_end_distance, lateral, frame, safe_cells)
        while terminal_points and math.dist(terminal_points[-1], terminal) < 2 * PEDESTRIAN_CLEARANCE_M:
            individual_end_distance -= 0.10
            if individual_end_distance <= target_distance + 0.10:
                raise ValueError(f"pedestrian {index} terminal overlaps another pedestrian")
            terminal = _offset_at(pedestrian_route, individual_end_distance, lateral, frame, safe_cells)
        suffix = _slice(pedestrian_route, target_distance, individual_end_distance)
        waypoints = _deduplicate([first_target, *suffix[1:-1], terminal])
        if not waypoints or math.dist(spawn, waypoints[0]) < 0.10:
            raise ValueError(f"pedestrian {index} starts on its first waypoint")
        dynamic.append(
            {
                "name": f"ped_{index}",
                "model": PEDESTRIAN_MODEL,
                "pose": _pose(spawn, _yaw(spawn, waypoints[0])),
                "velocity": PEDESTRIAN_SPEED_MPS,
                "waypoint_mode": "reverse",
                "agent": {
                    "agent_type": PEDESTRIAN_AGENT_TYPE,
                    "desired_velocity": PEDESTRIAN_SPEED_MPS,
                },
                "waypoints": [_pose(point, 0.0) for point in waypoints],
            }
        )
        spawn_points.append(spawn)
        terminal_points.append(terminal)

    poses = [tuple(human["pose"][:2]) for human in dynamic]
    if len(set(poses)) != count:
        raise ValueError("pedestrian formation contains duplicate spawn positions")
    if any(math.dist(left, right) < 2 * PEDESTRIAN_CLEARANCE_M for left, right in zip(poses, poses[1:], strict=False)):
        raise ValueError("pedestrian formation contains overlapping spawn footprints")
    return {
        "robots": [{"start": _pose(route_a_to_b[0], robot_yaw), "phases": phases}],
        "static": [],
        "dynamic": dynamic,
    }


def normalize_world(world_dir: Path, *, write: bool) -> LayoutResult:
    scenario_root = world_dir / "scenarios"
    files = sorted(scenario_root.glob(f"{GENERATED_PREFIX}*/scenario.yaml"))
    if not files:
        return LayoutResult(checked=0, changed=0, written=0)
    matrix: dict[tuple[str, int, str], dict[str, Any]] = {}
    paths: dict[tuple[str, int, str], Path] = {}
    for path in files:
        match = NAME_PATTERN.fullmatch(path.parent.name)
        if not match or match.group("world") != world_dir.name:
            raise ValueError(f"invalid generated scenario name: {path.parent.name}")
        key = (match.group("robot"), int(match.group("count")), match.group("direction"))
        if key in matrix:
            raise ValueError(f"duplicate generated scenario variant in {world_dir}")
        matrix[key] = _load_yaml(path)
        paths[key] = path
    expected = {(robot, count, direction) for robot in ("idle", "moving") for count in (1, 2, 3) for direction in ("a-to-b", "b-to-a")}
    if set(matrix) != expected:
        missing = sorted(expected - set(matrix))
        raise ValueError(f"{world_dir.name}: generated scenario matrix is incomplete; missing {missing}")

    frame, safe_cells = _load_safe_cells(world_dir)
    route = _canonical_route(matrix)
    replacements = {}
    for key in sorted(matrix):
        try:
            replacements[key] = _render(route, key[0], key[1], key[2], frame, safe_cells)
        except ValueError as exc:
            raise ValueError(f"{world_dir.name} {key}: {exc}") from exc
    changed_keys = [key for key in replacements if yaml.safe_dump(matrix[key], sort_keys=False) != yaml.safe_dump(replacements[key], sort_keys=False)]
    if write:
        for key in changed_keys:
            paths[key].write_text(yaml.safe_dump(replacements[key], sort_keys=False), encoding="utf-8")
    return LayoutResult(checked=len(matrix), changed=len(changed_keys), written=len(changed_keys) if write else 0)


def normalize_world_models(world_dir: Path, *, write: bool) -> LayoutResult:
    files = sorted((world_dir / "scenarios").glob("*/scenario.yaml"))
    changed: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        scenario = _load_yaml(path)
        updated = False
        for pedestrian in scenario.get("dynamic") or []:
            if not isinstance(pedestrian, dict):
                continue
            if pedestrian.get("model") != PEDESTRIAN_MODEL:
                pedestrian["model"] = PEDESTRIAN_MODEL
                updated = True
        if updated:
            changed.append((path, scenario))
    if write:
        for path, scenario in changed:
            path.write_text(
                yaml.safe_dump(scenario, sort_keys=False),
                encoding="utf-8",
            )
    return LayoutResult(
        checked=len(files),
        changed=len(changed),
        written=len(changed) if write else 0,
    )


def normalize(root: Path, patterns: Sequence[str], *, write: bool) -> LayoutResult:
    totals = LayoutResult(0, 0, 0)
    worlds = sorted(path for path in root.iterdir() if path.is_dir())
    if patterns:
        import fnmatch

        worlds = [path for path in worlds if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in patterns)]
    for world in worlds:
        layout_result = normalize_world(world, write=write)
        model_result = normalize_world_models(world, write=write)
        # A generated scenario can need both migrations; report changed files,
        # not the number of transformations applied to those files.
        changed = min(
            model_result.checked,
            layout_result.changed + model_result.changed,
        )
        written = min(
            model_result.checked,
            layout_result.written + model_result.written,
        )
        totals = LayoutResult(
            totals.checked + model_result.checked,
            totals.changed + changed,
            totals.written + written,
        )
    return totals


def _default_root() -> Path:
    source = Path(__file__).resolve().parents[3] / "acoustics" / "worlds"
    if source.is_dir():
        return source
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("arena_simulation_setup")) / "worlds"
    except (ImportError, LookupError):
        return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds-root", type=Path, default=_default_root())
    parser.add_argument("--world", action="append", default=[], metavar="NAME_OR_GLOB")
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite generated layouts and use the bundled model in every scenario",
    )
    args = parser.parse_args(argv)
    try:
        result = normalize(args.worlds_root, args.world, write=args.write)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": "write" if args.write else "check", **result.__dict__}, indent=2))
    return 0 if args.write or result.changed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
