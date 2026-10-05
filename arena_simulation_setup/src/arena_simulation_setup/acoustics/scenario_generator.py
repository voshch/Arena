"""Generate the fixed acoustics scenario matrix from Arena occupancy maps.

The occupancy grid is authoritative for placement.  ``world.yaml`` is used as
an independent geometry check, while ``map.yaml`` supplies the ROS map-to-world
transform used by RViz.  This avoids the common image-row/+Y inversion bug.
"""

from __future__ import annotations

import fnmatch
import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image, ImageFilter

GENERATED_PREFIX = "hearing__"
ROBOT_STATES = ("idle", "moving")
PEDESTRIAN_COUNTS = (1, 2, 3)
END_DIRECTIONS = ("a-to-b", "b-to-a")

# Fixed implementation values, deliberately not scenario factors.
ROBOT_CLEARANCE_M = 0.50
PEDESTRIAN_SPACING_M = 0.80
PEDESTRIAN_MODEL = "female_adult_medical_01"
PEDESTRIAN_AGENT_TYPE = "adult"
PEDESTRIAN_SPEED_MPS = 1.20

Cell = tuple[int, int]
Point2 = tuple[float, float]


@dataclass(frozen=True)
class MapFrame:
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    width: int
    height: int

    def cell_to_world(self, cell: Cell) -> Point2:
        """Return the world-frame centre of an image ``(row, column)`` cell."""
        row, column = cell
        local_x = (column + 0.5) * self.resolution
        # OccupancyGrid starts at the lower-left; PNG rows start at the top.
        local_y = (self.height - row - 0.5) * self.resolution
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        return (
            self.origin_x + cosine * local_x - sine * local_y,
            self.origin_y + sine * local_x + cosine * local_y,
        )

    def world_to_cell(self, point: Sequence[float]) -> Cell:
        """Return the nearest image cell for an RViz/world-frame point."""
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
class WorldRoute:
    world: str
    end_a: Point2
    end_b: Point2
    points_a_to_b: tuple[Point2, ...]
    length: float


@dataclass(frozen=True)
class PlannedScenario:
    world_dir: Path
    name: str
    data: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.world_dir / "scenarios" / self.name / "scenario.yaml"


def discover_worlds(root: str | Path, patterns: Sequence[str] = ()) -> list[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"worlds root does not exist: {root_path}")
    worlds = sorted(
        (path for path in root_path.iterdir() if path.is_dir() and (path / "0" / "world.yaml").is_file() and (path / "0" / "map.yaml").is_file()),
        key=lambda path: path.name,
    )
    if patterns:
        worlds = [path for path in worlds if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in patterns)]
    if not worlds:
        selection = ", ".join(patterns) if patterns else str(root_path)
        raise ValueError(f"no acoustics worlds matched: {selection}")
    return worlds


def scenario_name(world: str, robot_state: str, pedestrian_count: int, direction: str) -> str:
    if robot_state not in ROBOT_STATES:
        raise ValueError(f"unsupported robot state: {robot_state}")
    if pedestrian_count not in PEDESTRIAN_COUNTS:
        raise ValueError(f"unsupported pedestrian count: {pedestrian_count}")
    if direction not in END_DIRECTIONS:
        raise ValueError(f"unsupported end direction: {direction}")
    return f"{GENERATED_PREFIX}world-{world}__robot-{robot_state}__pedestrians-{pedestrian_count}__ends-{direction}"


def _load_map(level_dir: Path) -> tuple[MapFrame, set[Cell]]:
    map_path = level_dir / "map.yaml"
    raw = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{map_path} must contain a mapping")
    resolution = float(raw["resolution"])
    origin = raw.get("origin")
    if resolution <= 0 or not isinstance(origin, list) or len(origin) < 3:
        raise ValueError(f"{map_path} has an invalid resolution or origin")

    image_path = level_dir / str(raw["image"])
    with Image.open(image_path) as source:
        grayscale = source.convert("L")
    width, height = grayscale.size
    negate = bool(int(raw.get("negate", 0)))
    free_threshold = float(raw.get("free_thresh", 0.196))

    # ROS map_server: negate=0 maps white to zero occupancy (free).
    limit = int(math.floor(free_threshold * 255.0))
    occupancy = grayscale.point(
        (lambda value: value if negate else 255 - value),
        mode="L",
    )
    free = occupancy.point(lambda value: 255 if value <= limit else 0, mode="L")

    # MinFilter erodes white free space, inflating walls by the Jackal radius.
    clearance_pixels = max(1, int(math.ceil(ROBOT_CLEARANCE_M / resolution)))
    kernel_size = 2 * clearance_pixels + 1
    safe_image = free.filter(ImageFilter.MinFilter(kernel_size))
    safe_cells = {divmod(index, width) for index, value in enumerate(safe_image.tobytes()) if value == 255}
    if not safe_cells:
        raise ValueError(f"{image_path} contains no free cells after robot-clearance inflation")

    frame = MapFrame(
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        width=width,
        height=height,
    )
    return frame, _largest_component(safe_cells)


def _largest_component(cells: set[Cell]) -> set[Cell]:
    unseen = set(cells)
    largest: set[Cell] = set()
    while unseen:
        seed = min(unseen)
        component = {seed}
        stack = [seed]
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            for neighbor, _cost in _neighbors(current, cells):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return largest


def _load_world_polygons(world_path: Path) -> list[list[Point2]]:
    raw = yaml.safe_load(world_path.read_text(encoding="utf-8"))
    zones = raw.get("zones") if isinstance(raw, dict) else raw
    if not isinstance(zones, list) or not zones:
        raise ValueError(f"{world_path} must contain a non-empty zones list")
    polygons: list[list[Point2]] = []
    for index, zone in enumerate(zones):
        corners = zone.get("corners", []) if isinstance(zone, dict) else []
        if len(corners) < 3:
            raise ValueError(f"{world_path}: zone {index} has fewer than three corners")
        try:
            polygon = [(float(point["x"]), float(point["y"])) for point in corners]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{world_path}: zone {index} has invalid corner coordinates") from exc
        polygons.append(polygon)
    return polygons


def _point_segment_distance(point: Point2, start: Point2, end: Point2) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-18:
        return math.dist(point, start)
    fraction = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared),
    )
    projection = start[0] + fraction * dx, start[1] + fraction * dy
    return math.dist(point, projection)


def _polygon_covers(point: Point2, polygon: Sequence[Point2], tolerance: float) -> bool:
    inside = False
    for start, end in zip(polygon, [*polygon[1:], polygon[0]], strict=False):
        if _point_segment_distance(point, start, end) <= tolerance:
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = start[0] + ((point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1]))
            if crossing_x > point[0]:
                inside = not inside
    return inside


def _route_agrees_with_world(route: Sequence[Point2], polygons: Sequence[Sequence[Point2]], resolution: float) -> bool:
    step = resolution / 2.0
    for start, end in zip(route, route[1:], strict=False):
        samples = max(1, int(math.ceil(math.dist(start, end) / step)))
        for index in range(samples + 1):
            fraction = index / samples
            point = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            if not any(_polygon_covers(point, polygon, resolution) for polygon in polygons):
                return False
    return True


def _neighbors(cell: Cell, safe_cells: set[Cell]) -> Iterable[tuple[Cell, float]]:
    row, column = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        neighbor = row + dr, column + dc
        if neighbor not in safe_cells:
            continue
        if dr and dc and ((row + dr, column) not in safe_cells or (row, column + dc) not in safe_cells):
            continue
        yield neighbor, math.sqrt(2.0) if dr and dc else 1.0


def _dijkstra(start: Cell, safe_cells: set[Cell]) -> dict[Cell, float]:
    distances = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, cell = heapq.heappop(queue)
        if distance != distances.get(cell):
            continue
        for neighbor, cost in _neighbors(cell, safe_cells):
            candidate = distance + cost
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _astar(start: Cell, goal: Cell, safe_cells: set[Cell]) -> list[Cell]:
    distances = {start: 0.0}
    previous: dict[Cell, Cell] = {}
    queue = [(math.dist(start, goal), 0.0, start)]
    while queue:
        _estimate, distance, cell = heapq.heappop(queue)
        if distance != distances.get(cell):
            continue
        if cell == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            return path
        for neighbor, cost in _neighbors(cell, safe_cells):
            candidate = distance + cost
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = cell
                estimate = candidate + math.dist(neighbor, goal)
                heapq.heappush(queue, (estimate, candidate, neighbor))
    raise ValueError("selected corridor ends are disconnected")


def _boundary_distances(safe_cells: set[Cell]) -> dict[Cell, int]:
    """Return four-neighbor distance from each safe cell to the safe boundary."""
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    boundary = [cell for cell in safe_cells if any((cell[0] + dr, cell[1] + dc) not in safe_cells for dr, dc in cardinal)]
    distances = {cell: 0 for cell in boundary}
    queue = deque(boundary)
    while queue:
        cell = queue.popleft()
        for dr, dc in cardinal:
            neighbor = cell[0] + dr, cell[1] + dc
            if neighbor in safe_cells and neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances


def _refine_endpoint(
    endpoint: Cell,
    safe_cells: set[Cell],
    boundary_distances: dict[Cell, int],
    resolution: float,
) -> Cell:
    """Move a graph-diameter corner onto a safely inset corridor centre."""
    target_clearance = max(1, int(math.ceil(1.0 / resolution)))
    search_radius = max(2, int(math.ceil(2.0 / resolution)))
    row0, column0 = endpoint
    choices = [cell for row in range(row0 - search_radius, row0 + search_radius + 1) for column in range(column0 - search_radius, column0 + search_radius + 1) if (cell := (row, column)) in safe_cells and math.dist(cell, endpoint) <= search_radius]
    return max(
        choices,
        key=lambda cell: (
            min(boundary_distances[cell], target_clearance),
            -math.dist(cell, endpoint),
            cell,
        ),
    )


def _grid_line_is_safe(start: Cell, end: Cell, safe_cells: set[Cell]) -> bool:
    row0, column0 = start
    row1, column1 = end
    samples = max(abs(row1 - row0), abs(column1 - column0)) * 4 + 1
    for index in range(samples + 1):
        fraction = index / max(samples, 1)
        cell = (
            round(row0 + (row1 - row0) * fraction),
            round(column0 + (column1 - column0) * fraction),
        )
        if cell not in safe_cells:
            return False
    return True


def _simplify_cells(path: Sequence[Cell], safe_cells: set[Cell]) -> list[Cell]:
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        target = len(path) - 1
        while target > anchor + 1 and not _grid_line_is_safe(path[anchor], path[target], safe_cells):
            target -= 1
        simplified.append(path[target])
        anchor = target
    return simplified


def derive_world_route(world_dir: str | Path) -> WorldRoute:
    path = Path(world_dir)
    level = path / "0"
    frame, safe_cells = _load_map(level)
    polygons = _load_world_polygons(level / "world.yaml")

    # Two occupancy-graph sweeps give the terminal-to-terminal diameter for
    # tree-shaped corridors and deterministic antipodal points for rings.
    seed = min(safe_cells, key=lambda cell: frame.cell_to_world(cell))
    first_distances = _dijkstra(seed, safe_cells)
    first = max(first_distances, key=lambda cell: (first_distances[cell], cell))
    second_distances = _dijkstra(first, safe_cells)
    second = max(second_distances, key=lambda cell: (second_distances[cell], cell))
    clearance = _boundary_distances(safe_cells)
    first = _refine_endpoint(first, safe_cells, clearance, frame.resolution)
    second = _refine_endpoint(second, safe_cells, clearance, frame.resolution)
    end_a_cell, end_b_cell = sorted((first, second), key=lambda cell: frame.cell_to_world(cell))
    if end_a_cell == end_b_cell:
        raise ValueError(f"{path.name}: fewer than two distinct map-aligned corridor ends")

    try:
        grid_path = _astar(end_a_cell, end_b_cell, safe_cells)
    except ValueError as exc:
        raise ValueError(f"{path.name}: selected corridor ends are disconnected") from exc
    grid_path = _simplify_cells(grid_path, safe_cells)
    route = tuple(frame.cell_to_world(cell) for cell in grid_path)

    if not _route_agrees_with_world(route, polygons, frame.resolution):
        raise ValueError(f"{path.name}: RViz-aligned route does not agree with world.yaml geometry")
    length = sum(math.dist(left, right) for left, right in zip(route, route[1:], strict=False))
    if length <= 2 * PEDESTRIAN_SPACING_M + 1.0:
        raise ValueError(f"{path.name}: selected route is too short for three pedestrians")
    return WorldRoute(path.name, route[0], route[-1], route, length)


def _yaw(start: Sequence[float], end: Sequence[float]) -> float:
    return math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))


def _pose(point: Sequence[float], yaw: float) -> list[float]:
    return [round(float(point[0]), 4), round(float(point[1]), 4), round(float(yaw), 6)]


def _route_from_distance(points: Sequence[Point2], distance: float) -> list[Point2]:
    remaining = max(0.0, distance)
    for index, (start, end) in enumerate(zip(points, points[1:], strict=False)):
        segment = math.dist(start, end)
        if remaining <= segment:
            fraction = remaining / segment if segment else 0.0
            spawn = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            suffix = [spawn, *points[index + 1 :]]
            return [suffix[0], *[point for point in suffix[1:] if math.dist(point, suffix[0]) > 1e-6]]
        remaining -= segment
    return [points[-1]]


def render_scenario(route: WorldRoute, robot_state: str, pedestrian_count: int, direction: str) -> dict[str, Any]:
    name = scenario_name(route.world, robot_state, pedestrian_count, direction)
    del name  # validation is intentional even when this function is used directly.
    robot_route = list(route.points_a_to_b)
    if direction == "b-to-a":
        robot_route.reverse()
    pedestrian_route = list(reversed(robot_route))

    robot_yaw = _yaw(robot_route[0], robot_route[1])
    phases = []
    if robot_state == "moving":
        for index, point in enumerate(robot_route[1:], start=1):
            arrival_yaw = _yaw(robot_route[index - 1], point)
            phases.append({"goto": _pose(point, arrival_yaw)})

    dynamic = []
    for index in range(pedestrian_count):
        ped_route = _route_from_distance(pedestrian_route, index * PEDESTRIAN_SPACING_M)
        if len(ped_route) < 2:
            raise ValueError(f"{route.world}: pedestrian {index + 1} has no traversable route")
        ped_yaw = _yaw(ped_route[0], ped_route[1])
        dynamic.append(
            {
                "name": f"ped_{index + 1}",
                "model": PEDESTRIAN_MODEL,
                "pose": _pose(ped_route[0], ped_yaw),
                "agent": {
                    "agent_type": PEDESTRIAN_AGENT_TYPE,
                    "desired_velocity": PEDESTRIAN_SPEED_MPS,
                },
                "waypoints": [_pose(point, 0.0) for point in ped_route],
            }
        )

    return {
        "robots": [{"start": _pose(robot_route[0], robot_yaw), "phases": phases}],
        "static": [],
        "dynamic": dynamic,
    }


def plan_world(world_dir: str | Path) -> list[PlannedScenario]:
    path = Path(world_dir)
    route = derive_world_route(path)
    scenarios = []
    for robot_state, count, direction in itertools.product(ROBOT_STATES, PEDESTRIAN_COUNTS, END_DIRECTIONS):
        scenarios.append(
            PlannedScenario(
                world_dir=path,
                name=scenario_name(path.name, robot_state, count, direction),
                data=render_scenario(route, robot_state, count, direction),
            )
        )
    if len(scenarios) != 12:
        raise AssertionError("the fixed scenario matrix must contain exactly 12 cases")
    return scenarios


def plan(root: str | Path, patterns: Sequence[str] = ()) -> list[PlannedScenario]:
    scenarios = []
    for world in discover_worlds(root, patterns):
        scenarios.extend(plan_world(world))
    return scenarios


def _yaml_bytes(data: dict[str, Any]) -> bytes:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")


def write_scenarios(scenarios: Sequence[PlannedScenario]) -> tuple[int, int]:
    pending: list[tuple[Path, bytes]] = []
    unchanged = 0
    for scenario in scenarios:
        content = _yaml_bytes(scenario.data)
        target = scenario.path
        if target.exists():
            if target.read_bytes() == content:
                unchanged += 1
                continue
            raise FileExistsError(f"refusing to overwrite existing scenario: {target}")
        pending.append((target, content))

    # Do not partially generate a batch when any existing file conflicts.
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=False)
        target.write_bytes(content)
    return len(pending), unchanged
