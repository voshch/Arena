"""SidewalkBench scanned sidewalk scenes as worlds: sidewalk and crosswalk zones, mesh slice outlines, the splat bundle with the collision mesh, and a scenario per route."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import attrs
import numpy as np
import shapely
import yaml

from arena_assets.impl.splat import ObjectCrop, SplatError, frame_splat, gaussian_ply_stage, write_annotation
from arena_assets.importers import IMPORTERS, Attribution, Importer, import_root
from arena_assets.importers.providers import HFHub, Provider
from arena_assets.utils.mesh_io import read_mesh
from arena_simulation_setup.generative.world.utils import to_walls, world_of
from arena_simulation_setup.shared import Obstacle, Position
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.World import Level, WorldView, Zone, derive
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT
from arena_simulation_setup.tree.World.write import POSITION_DECIMALS
from arena_simulation_setup.utils.geometry import Pose

REPO = "UCLA-VAIL/SidewalkBench-GS-Assets"
URL = f"https://huggingface.co/datasets/{REPO}"
SCENES = ("9am", "beverly_hills_crosswalk", "beverly_hills_intersection", "cars", "dark", "hill", "loop", "midvale", "triangle", "ucla_crosswalk", "ucla_sidewalk_backward")
SCENE_FILES = ("annotation.json", "centerlines.json", "semantic_grid_0p1m.npz", "mesh.ply", "gs.ply")
CONE_ASSET = "Warehouse/Traffic_Cone"
DOMAIN = "Common"
ASSET_TYPE = "Object"
LICENSE = "MIT"
AUTHORS = "Zhizheng Liu, Honglin He, Vivek Alumootil, Akshat Pandya, Brad Squicciarini, Wayne Wu, Bolei Zhou (UCLA VAIL and Coco Robotics)"
CITATION = (
    "@inproceedings{liu2026sidewalkbench, title={SidewalkBench: Benchmarking Visual Navigation on Urban Sidewalks}, "
    "author={Zhizheng Liu and Honglin He and Vivek Alumootil and Akshat Pandya and Brad Squicciarini and Wayne Wu and Bolei Zhou}, "
    "booktitle={Conference on Robot Learning}, year={2026}, eprint={2606.16953}, archivePrefix={arXiv}}"
)
HUMAN_MODEL = "arenian"
HUMAN_AGENT = "adult"
WALK_SPEED_MPS = (1.1, 1.65)
WAYPOINT_STEP_M = 3.0
CONE_RANGE_M = 3.0
END_CLEARANCE_M = 0.6
END_STEP_M = 0.05
CURVE_TURN_DEG = 30.0
TURN_WINDOW = 5
LONG_MODE = "long_horizon"

CROP_MARGIN_M = 10.0
CROP_BELOW_M = 5.0
CROP_ABOVE_M = 40.0
GROUND_FILL_CELLS = 30
SLICE_M = (0.3, 1.5)
SAMPLE_M = 0.05
MAX_STEPS = 200
SAMPLE_BUDGET = 1 << 21
MIN_HITS = 6
SIMPLIFY_M = 0.1
MIN_ZONE_AREA_M2 = 2.0
MIN_OBSTACLE_AREA_M2 = 0.04
BUNDLE_BOX = ObjectCrop(width=0.02, depth=0.02, height=0.04, floor=0.02)

Cell = tuple[int, int]


def scene_name(item: str) -> str:
    if item not in SCENES:
        raise ValueError(f"no SidewalkBench scene '{item}', items are {', '.join(SCENES)}")
    return item


def world_name(item: str) -> str:
    return f"sidewalkbench_{scene_name(item)}"


@attrs.frozen
class Grid:
    """The 0.1 m semantic grid: sidewalk and crosswalk masks plus the ground height under every cell, indexed [row = y, column = x]."""

    origin: tuple[float, float]
    resolution: float
    sidewalk: np.ndarray
    crosswalk: np.ndarray
    labeled: np.ndarray
    ground: np.ndarray

    @classmethod
    def read(cls, path: Path) -> Grid:
        with np.load(path) as data:
            height = np.asarray(data["semantic_z"], np.float64)
            valid = data["semantic_ground_like"].astype(bool) & np.isfinite(height)
            return cls(
                origin=(float(data["origin_xy"][0]), float(data["origin_xy"][1])),
                resolution=float(data["resolution_m"][0]),
                sidewalk=data["sidewalk"].astype(bool),
                crosswalk=data["crosswalk"].astype(bool),
                labeled=data["semantic"] != -1,
                ground=filled(np.where(valid, height, np.nan)),
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.sidewalk.shape

    def cell(self, x: float, y: float) -> Cell | None:
        column = math.floor((x - self.origin[0]) / self.resolution)
        row = math.floor((y - self.origin[1]) / self.resolution)
        if 0 <= row < self.shape[0] and 0 <= column < self.shape[1]:
            return row, column
        return None

    def height_at(self, x: float, y: float) -> float:
        """Ground height under a point, NaN outside the grid or beyond the filled ground."""
        cell = self.cell(x, y)
        return float(self.ground[cell]) if cell is not None else math.nan

    def extent(self) -> tuple[float, float, float, float]:
        """x and y bounds of the labeled cells."""
        rows, columns = np.nonzero(self.labeled)
        if not len(rows):
            raise ValueError("the semantic grid labels no cell")
        x0, y0 = self.origin
        return (
            x0 + columns.min() * self.resolution,
            y0 + rows.min() * self.resolution,
            x0 + (columns.max() + 1) * self.resolution,
            y0 + (rows.max() + 1) * self.resolution,
        )


def filled(ground: np.ndarray, cells: int = GROUND_FILL_CELLS) -> np.ndarray:
    """Ground heights with each missing cell taking the mean of its known neighbors, repeated for up to cells rings."""
    ground = ground.copy()
    height, width = ground.shape
    for _ in range(cells):
        missing = np.isnan(ground)
        if not missing.any():
            break
        padded = np.pad(ground, 1, constant_values=np.nan)
        neighbors = np.stack([padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        known = ~np.isnan(neighbors)
        count = known.sum(axis=0)
        total = np.where(known, neighbors, 0.0).sum(axis=0)
        ground = np.where(missing & (count > 0), total / np.maximum(count, 1), ground)
    return ground


def polygons(mask: np.ndarray, grid: Grid, min_area: float, fill_holes: bool) -> list[shapely.Polygon]:
    """Connected pieces of a cell mask as simplified polygons, largest first, pieces under min_area dropped."""
    padded = np.pad(mask, ((0, 0), (1, 1))).astype(np.int8)
    edges = np.diff(padded, axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, stops = np.nonzero(edges == -1)
    if not len(rows):
        return []
    x0, y0 = grid.origin
    res = grid.resolution
    union = shapely.union_all(shapely.box(x0 + starts * res, y0 + rows * res, x0 + stops * res, y0 + (rows + 1) * res))
    pieces = [piece for piece in shapely.get_parts(union) if isinstance(piece, shapely.Polygon)]
    shapes = [shapely.Polygon(piece.exterior) if fill_holes else piece for piece in pieces]
    kept = [shape for shape in (piece.simplify(SIMPLIFY_M) for piece in shapes) if isinstance(shape, shapely.Polygon) and shape.area >= min_area]
    return sorted(kept, key=lambda shape: -shape.area)


def outer_pieces(geometry: shapely.geometry.base.BaseGeometry) -> list[shapely.Polygon]:
    """Polygon parts of a geometry snapped to the precision the world file stores, holes filled, largest first, parts under MIN_ZONE_AREA_M2 dropped."""
    snapped = shapely.set_precision(geometry, 10.0**-POSITION_DECIMALS)
    outers = [shapely.Polygon(part.exterior) for part in shapely.get_parts(snapped) if isinstance(part, shapely.Polygon)]
    return sorted((outer for outer in outers if outer.area >= MIN_ZONE_AREA_M2), key=lambda outer: -outer.area)


def zones(grid: Grid) -> list[Zone]:
    """Zones sidewalk and crosswalk tiling the walkable cells without gaps, the crosswalk keeping the cells both layers label, a piece named <zone>_<i> when a layer has several."""
    walkable = shapely.set_precision(shapely.union_all(polygons(grid.sidewalk | grid.crosswalk, grid, MIN_ZONE_AREA_M2, fill_holes=True)), 10.0**-POSITION_DECIMALS)
    crosswalks = outer_pieces(shapely.intersection(shapely.union_all(polygons(grid.crosswalk, grid, MIN_ZONE_AREA_M2, fill_holes=True)), walkable))
    sidewalks = outer_pieces(shapely.difference(walkable, shapely.union_all(crosswalks)))
    found: list[Zone] = []
    for name, pieces in (("sidewalk", sidewalks), ("crosswalk", crosswalks)):
        for index, piece in enumerate(pieces):
            ring = [Position(x, y, 0.0) for x, y in list(piece.exterior.coords)[:-1]]
            found.append(Zone(name=name if len(pieces) == 1 else f"{name}_{index}", polygon=ring, floor=None, ceiling=None))
    return found


def lattice(steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric weights of the sample points on a triangle cut into steps rows."""
    u, v = np.meshgrid(np.arange(steps + 1), np.arange(steps + 1), indexing="ij")
    inside = u + v <= steps
    return u[inside] / steps, v[inside] / steps


def slice_hits(points: np.ndarray, triangles: np.ndarray, grid: Grid) -> np.ndarray:
    """Per cell, how many mesh surface samples lie within SLICE_M above the local ground."""
    corners = points[triangles]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    longest = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)])
    steps = np.clip(np.ceil(longest / SAMPLE_M), 1, MAX_STEPS).astype(np.int64)
    hits = np.zeros(grid.shape, np.int64)
    for count in np.unique(steps):
        u, v = lattice(int(count))
        members = np.nonzero(steps == count)[0]
        per_batch = max(1, SAMPLE_BUDGET // len(u))
        for lo in range(0, len(members), per_batch):
            pick = members[lo : lo + per_batch]
            samples = (1 - u - v)[None, :, None] * a[pick][:, None] + u[None, :, None] * b[pick][:, None] + v[None, :, None] * c[pick][:, None]
            hits += cell_hits(samples.reshape(-1, 3), grid)
    return hits


def cell_hits(samples: np.ndarray, grid: Grid) -> np.ndarray:
    columns = np.floor((samples[:, 0] - grid.origin[0]) / grid.resolution).astype(np.int64)
    rows = np.floor((samples[:, 1] - grid.origin[1]) / grid.resolution).astype(np.int64)
    inside = (rows >= 0) & (rows < grid.shape[0]) & (columns >= 0) & (columns < grid.shape[1])
    rows, columns, height = rows[inside], columns[inside], samples[inside, 2]
    above = height - grid.ground[rows, columns]
    band = (above >= SLICE_M[0]) & (above <= SLICE_M[1])
    flat = rows[band] * grid.shape[1] + columns[band]
    return np.bincount(flat, minlength=grid.sidewalk.size).reshape(grid.shape)


def outlines(mesh: Path, grid: Grid) -> list[shapely.Polygon]:
    """Polygons around the mesh within SLICE_M above the local ground, in the scene frame."""
    points, triangles = read_mesh(mesh)
    return polygons(slice_hits(points, triangles, grid) >= MIN_HITS, grid, MIN_OBSTACLE_AREA_M2, fill_holes=False)


@attrs.frozen
class Frame:
    """Bundle frame of a scene: the scene shifted by -center in x and y and by -ground in z, so the world pose of the bundle is (center, 0)."""

    center: tuple[float, float]
    ground: float
    crop: ObjectCrop

    @property
    def to_bundle(self) -> np.ndarray:
        shift = np.eye(4)
        shift[:3, 3] = (-self.center[0], -self.center[1], -self.ground)
        return shift


def frame_of(grid: Grid, anchor: tuple[float, float]) -> Frame:
    """Frame centered on the labeled extent, its zero height the ground under anchor."""
    x0, y0, x1, y1 = grid.extent()
    center = (round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2))
    ground = grid.height_at(*anchor)
    known = grid.ground[np.isfinite(grid.ground)]
    if not len(known):
        raise ValueError("the semantic grid holds no ground height")
    if math.isnan(ground):
        ground = float(np.median(known))
    ground = round(ground, 3)
    crop = ObjectCrop(
        width=x1 - x0 + 2 * CROP_MARGIN_M,
        depth=y1 - y0 + 2 * CROP_MARGIN_M,
        height=float(known.max()) - ground + CROP_ABOVE_M,
        floor=float(known.min()) - ground - CROP_BELOW_M,
    )
    return Frame(center=center, ground=ground, crop=crop)


def write_bundle(scene: Path, bundle: Path, name: str, frame: Frame) -> None:
    """Write the object bundle of a scene: its Gaussians in the crop and its mesh as the collider."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "scene.usdc"
        try:
            gaussian_ply_stage(scene / "gs.ply", stage, frame.to_bundle, frame.crop)
            splat = frame_splat(stage, bundle, name, frame.to_bundle, frame.crop, [], collider_mesh=scene / "mesh.ply")
        except SplatError as error:
            raise RuntimeError(str(error)) from error
    write_annotation(bundle, name, BUNDLE_BOX, f"SidewalkBench scan {name}, Gaussians with the collision mesh.", f"{URL}/{scene.name}", splat.collider)


def route_kind(route: dict, crosswalks: dict[str, bool]) -> str:
    """long for a long-horizon route, crosswalk when it follows a crosswalk centerline, curve past CURVE_TURN_DEG of heading change, else straight."""
    if route["mode"] == LONG_MODE:
        return "long"
    if any(crosswalks[identifier] for identifier in route["centerline_ids"]):
        return "crosswalk"
    steps = np.diff(np.asarray(route["centerline_xy"], np.float64), axis=0)
    heading = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    smooth = np.convolve(heading, np.ones(TURN_WINDOW) / TURN_WINDOW, mode="valid")
    return "curve" if math.degrees(float(np.abs(np.diff(smooth)).sum())) > CURVE_TURN_DEG else "straight"


def pose_entry(x: float, y: float, z: float, yaw: float) -> dict[str, float]:
    return {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4), "yaw": round(yaw, 4)}


def walking_speed(identity: str, scene: str, index: int) -> float:
    low, high = WALK_SPEED_MPS
    return round(low + (high - low) * zlib.crc32(f"{scene}/{identity}/{index}".encode()) / 0xFFFFFFFF, 3)


def thinned(path: np.ndarray) -> np.ndarray:
    """Points of a polyline about WAYPOINT_STEP_M apart, the last one kept."""
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    keep = [0]
    for index in range(1, len(path)):
        if along[index] - along[keep[-1]] >= WAYPOINT_STEP_M:
            keep.append(index)
    if keep[-1] != len(path) - 1:
        keep.append(len(path) - 1)
    return path[keep]


def humans(route: dict, scene: str) -> list[dict]:
    """Static pedestrians standing where they are, dynamic ones walking the centerline toward the route start."""
    entries: list[dict] = []
    for index, ped in enumerate(route["static_pedestrians"]):
        x, y = ped["xy"]
        entries.append(
            {
                "name": f"ped_static_{index}",
                "model": HUMAN_MODEL,
                "pose": [round(x, 4), round(y, 4), round(math.radians(ped["yaw_deg"]), 4)],
                "agent": {"agent_type": HUMAN_AGENT},
                "waypoint_mode": "once",
                "waypoints": [[round(x, 4), round(y, 4)]],
            }
        )
    centerline = np.asarray(route["centerline_xy"], np.float64)
    for index, ped in enumerate(route["dynamic_pedestrians"]):
        x, y = ped["xy"]
        nearest = int(np.argmin(np.linalg.norm(centerline - (x, y), axis=1)))
        waypoints = thinned(centerline[nearest::-1])
        speed = walking_speed(ped["identity"], scene, index)
        ahead = waypoints[1] if len(waypoints) > 1 else waypoints[0]
        heading = math.atan2(ahead[1] - y, ahead[0] - x)
        entries.append(
            {
                "name": f"ped_walker_{index}",
                "model": HUMAN_MODEL,
                "pose": [round(x, 4), round(y, 4), round(heading, 4)],
                "velocity": speed,
                "agent": {"agent_type": HUMAN_AGENT, "desired_velocity": speed},
                "waypoint_mode": "once",
                "waypoints": [[round(float(px), 4), round(float(py), 4)] for px, py in waypoints],
            }
        )
    return entries


def cones(route: dict, annotation: dict, frame: Frame) -> list[dict]:
    """Cones within CONE_RANGE_M of the route centerline."""
    line = shapely.LineString(route["centerline_xy"])
    return [
        {
            "name": f"cone_{cone['id']}",
            "model": CONE_ASSET,
            "pose": pose_entry(cone["xy"][0], cone["xy"][1], cone["settled"]["xyz"][2] - frame.ground, math.radians(cone["yaw_deg"])),
        }
        for cone in annotation["cones"]
        if line.distance(shapely.Point(cone["xy"])) <= CONE_RANGE_M
    ]


def clear_end(point: tuple[float, float], centerline: shapely.LineString, obstacles: shapely.geometry.base.BaseGeometry, from_end: bool) -> tuple[float, float]:
    """The point when END_CLEARANCE_M from every obstacle, else the first such point walking the centerline inward from that end, else the point."""
    if obstacles.is_empty or obstacles.distance(shapely.Point(point)) >= END_CLEARANCE_M:
        return point
    length = centerline.length
    for step in np.arange(0.0, length + END_STEP_M, END_STEP_M):
        candidate = centerline.interpolate(length - step if from_end else step)
        if obstacles.distance(candidate) >= END_CLEARANCE_M:
            return float(candidate.x), float(candidate.y)
    return point


def scenarios(annotation: dict, centerlines: dict, frame: Frame, grid: Grid, obstacles: shapely.geometry.base.BaseGeometry) -> dict[str, dict]:
    """One scenario per route named <kind>_<i>, i counting the routes of a kind in route order, its start and goal kept clear of the obstacles."""
    crosswalks = {line["id"]: bool(line["crosswalk"]) for line in centerlines["centerlines"]}
    routes = sorted(annotation["routes"], key=lambda route: int(route["id"].rpartition("_")[2]))
    counts: dict[str, int] = {}
    built: dict[str, dict] = {}
    for route in routes:
        kind = route_kind(route, crosswalks)
        name = f"{kind}_{counts.get(kind, 0)}"
        counts[kind] = counts.get(kind, 0) + 1
        centerline = shapely.LineString(route["centerline_xy"])
        sx, sy = clear_end(tuple(route["start"]["xy"]), centerline, obstacles, from_end=False)
        gx, gy = clear_end(tuple(route["goal"]["xy"]), centerline, obstacles, from_end=True)
        ground = grid.height_at(sx, sy)
        end = np.asarray(route["centerline_xy"][-2:], np.float64)
        scenario: dict = {
            "format": SCENARIO_FORMAT,
            "objects": cones(route, annotation, frame),
            "humans": humans(route, annotation["scene"]),
            "robots": [
                {
                    "start": pose_entry(sx, sy, 0.0 if math.isnan(ground) else ground - frame.ground, math.radians(route["start"]["yaw_deg"])),
                    "phases": [{"goto": [round(gx, 4), round(gy, 4), round(math.atan2(end[1, 1] - end[0, 1], end[1, 0] - end[0, 0]), 4)]}],
                }
            ],
        }
        built[name] = {key: value for key, value in scenario.items() if value}
    return built


class SidewalkBenchWorld(Importer):
    meta = IMPORTERS.meta("world", "sidewalkbench")

    @classmethod
    def default_provider(cls) -> Provider:
        return HFHub(REPO)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def items(self) -> Iterator[str]:
        return iter(SCENES)

    def fetch(self, item: str) -> Path:
        scene = scene_name(item)
        for name in SCENE_FILES:
            self.provider.get(f"{scene}/{name}", self.cache / scene / name)
        return self.cache / scene

    def convert(self, item: str, src: Path, out: Path) -> str:
        scene = scene_name(item)
        name = world_name(scene)
        annotation = json.loads((src / "annotation.json").read_text())
        centerlines = json.loads((src / "centerlines.json").read_text())
        grid = Grid.read(src / "semantic_grid_0p1m.npz")
        frame = frame_of(grid, tuple(annotation["routes"][0]["start"]["xy"]))
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        obstacles = outlines(src / "mesh.ply", grid)
        level = Level(zones=zones(grid), walls=[wall for outline in obstacles for wall in to_walls(outline)])
        level.objects.append(Obstacle(name=name, model=ObjectIdentifier(name=name, domain=DOMAIN), pose=Pose.parse([*frame.center, 0.0])))
        built = world_of(level, {zone.name: zone.name.partition("_")[0] for zone in level.zones})
        built.kg = derive(built)
        WorldView(target).save(built)
        write_bundle(src, target / "assets" / DOMAIN / ASSET_TYPE / name, name, frame)
        for key, scenario in scenarios(annotation, centerlines, frame, grid, shapely.union_all(obstacles)).items():
            path = target / "scenarios" / key / "scenario.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(scenario, sort_keys=False))
        return name

    def attribution(self, item: str) -> Attribution:
        scene_name(item)
        return Attribution(source=self.meta.name, url=URL, license=LICENSE, authors=AUTHORS, citation=CITATION)
